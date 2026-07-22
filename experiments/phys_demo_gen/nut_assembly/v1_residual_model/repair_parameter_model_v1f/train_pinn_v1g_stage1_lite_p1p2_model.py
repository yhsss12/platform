#!/usr/bin/env python3
"""V1-G-stage1-lite-p1p2：aligned-original init + 轻量 P1/P2 physics aux（排除 demo_3）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
for path in (_V1_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pinn_v1f_repair_model import (  # noqa: E402
    PINNV1FRepairModel,
    V1GStage1LiteP1P2Config,
    compute_v1g_stage1_lite_losses,
)
from train_pinn_v1f_balanced_model import DemoPairBatchSampler, set_seed  # noqa: E402
from v1f_100base_utils import DEFAULT_ALIGNED_MODEL, DEFAULT_DATASET_NPZ, OLD_DEMO_KEYS  # noqa: E402
from v1f_repair_dataset import V1FRepairDataset, load_v1f_npz  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "v1g_stage1_lite_p1p2" / "trained_model"
DEMO_3_TRAIN_IDX = OLD_DEMO_KEYS.index("demo_3")
EXCLUDED_TRAINING_DEMOS = ("demo_3",)


def split_trainable_indices(indices: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def trainable_indices(bundle: dict, *, exclude_demo_idx: int) -> np.ndarray:
    demo_idx = bundle["demo_idx"]
    return np.where(demo_idx != exclude_demo_idx)[0]


def train_one_epoch(model, loader, optimizer, device, physics, *, grad_clip: float = 1.0) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["features"])
        losses = compute_v1g_stage1_lite_losses(out, batch, physics=physics)
        losses["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_losses(model, loader, device, physics) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["features"])
        losses = compute_v1g_stage1_lite_losses(out, batch, physics=physics)
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-G-stage1-lite-p1p2 model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset missing: {args.dataset}")
    if not args.init_checkpoint.exists():
        raise SystemExit(f"Init checkpoint missing: {args.init_checkpoint}")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.output_dir.parent

    bundle = load_v1f_npz(args.dataset)
    eligible = trainable_indices(bundle, exclude_demo_idx=DEMO_3_TRAIN_IDX)
    train_idx, val_idx = split_trainable_indices(eligible, args.val_frac, args.seed)

    physics = V1GStage1LiteP1P2Config.lite_p1p2()
    config = {
        "model_version": "V1-G-stage1-lite-p1p2",
        "physics_loss": {
            "lambda_transport": physics.lambda_transport,
            "lambda_xy": physics.lambda_xy,
            "lambda_lift": physics.lambda_lift,
            "lambda_retention": physics.lambda_retention,
            "excluded_from_training": [
                "E_contact",
                "E_bilateral",
                "E_dynamics",
                "E_slip",
                "E_coupling",
                "E_insert_depth",
                "E_axis_alignment",
                "E_vertical_approach",
                "E_final_pose",
                "E_jamming",
            ],
            "insertion_residuals_opt_in_only": True,
        },
        "init_checkpoint": str(args.init_checkpoint),
        "dataset": str(args.dataset),
        "aligned_original_preserved": True,
        "excluded_training_demos": {
            "demo_3": {
                "repairability": "non_repairable_under_current_pipeline",
                "include_in_v1g_lite_training": False,
                "include_in_v1g_lite_validation": False,
                "keep_for_diagnostic": True,
                "failure_stage": "insertion_contact",
                "failure_reason": "axis_misalignment / jamming",
            }
        },
        "training_samples": {
            "eligible": int(len(eligible)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "excluded_demo_3": int(np.sum(bundle["demo_idx"] == DEMO_3_TRAIN_IDX)),
        },
    }
    (args.output_dir / "v1g_stage1_lite_p1p2_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": config}, indent=2))
        return 0

    train_ds = V1FRepairDataset(args.dataset, train_idx)
    val_ds = V1FRepairDataset(args.dataset, val_idx)
    batch_sampler = DemoPairBatchSampler(train_ds.demo_group_id, batch_size=args.batch_size, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINNV1FRepairModel(input_dim=int(bundle["features"].shape[1])).to(device)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, device, physics)
        val_m = evaluate_losses(model, val_loader, device, physics)
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}": v for k, v in val_m.items()},
            }
        )
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(
                f"epoch {epoch} val={val_m['loss']:.4f} physics_aux={val_m.get('physics_aux_total', 0):.4f} "
                f"retention={val_m.get('old_demo_retention', 0):.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_checkpoint or (args.output_dir / "model_v1g_stage1_lite_p1p2.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(bundle["features"].shape[1]),
            "model_version": "V1-G-stage1-lite-p1p2",
            "init_checkpoint": str(args.init_checkpoint),
            "physics_loss_config": config["physics_loss"],
            "excluded_training_demos": list(EXCLUDED_TRAINING_DEMOS),
        },
        ckpt_path,
    )
    log = {
        "model_version": "V1-G-stage1-lite-p1p2",
        "checkpoint": str(ckpt_path),
        "init_checkpoint": str(args.init_checkpoint),
        "best_val_loss": best_val,
        "epochs": args.epochs,
        "excluded_training_demos": list(EXCLUDED_TRAINING_DEMOS),
        "history_tail": history[-5:],
        "config": config,
    }
    (output_root / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
