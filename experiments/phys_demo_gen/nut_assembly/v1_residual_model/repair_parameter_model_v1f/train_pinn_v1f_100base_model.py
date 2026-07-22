#!/usr/bin/env python3
"""Train V1-F-100Base PINN from aligned-original init."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
for path in (_V1_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pinn_v1f_repair_model import (  # noqa: E402
    PINNV1FRepairModel,
    V1FPhysicsLossConfig,
    compute_v1f_100base_losses,
)
from train_pinn_v1f_balanced_model import DemoPairBatchSampler, set_seed  # noqa: E402
from train_residual_model import split_indices  # noqa: E402
from v1f_100base_utils import (  # noqa: E402
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_DATASET_NPZ,
    DEFAULT_100BASE_OUTPUT,
    DEFAULT_SANITY_REPORT,
)  # noqa: E402
from v1f_repair_dataset import V1FRepairDataset, load_v1f_npz  # noqa: E402

DEFAULT_OUTPUT = DEFAULT_100BASE_OUTPUT / "trained_model"


def train_one_epoch(model, loader, optimizer, device, physics, *, grad_clip: float = 1.0) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["features"])
        losses = compute_v1f_100base_losses(out, batch, physics=physics)
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
        losses = compute_v1f_100base_losses(out, batch, physics=physics)
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-F-100Base model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="跳过 pre-train sanity gate 校验（不推荐；仅调试）",
    )
    args = parser.parse_args()

    if "balanced_v2" in str(args.init_checkpoint) or "v2" in args.init_checkpoint.name:
        raise SystemExit(f"Refusing v2 init checkpoint: {args.init_checkpoint}")

    sanity_path = DEFAULT_SANITY_REPORT
    if not args.skip_sanity_check:
        if not sanity_path.exists():
            raise SystemExit(
                f"Pre-train sanity report missing: {sanity_path}\n"
                "Run run_v1f_100base_pretrain_sanity_gate.py after dataset build."
            )
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        gate = sanity.get("training_gate", {})
        if not gate.get("training_allowed"):
            reasons = gate.get("block_reasons") or ["unknown"]
            raise SystemExit(
                "Pre-train sanity gate blocked training:\n- " + "\n- ".join(str(r) for r in reasons)
            )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_v1f_npz(args.dataset)
    train_idx, val_idx = split_indices(len(bundle["features"]), args.val_frac, args.seed)

    train_ds = V1FRepairDataset(args.dataset, train_idx)
    val_ds = V1FRepairDataset(args.dataset, val_idx)
    batch_sampler = DemoPairBatchSampler(train_ds.demo_group_id, batch_size=args.batch_size, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = V1FPhysicsLossConfig.hundredbase()
    model = PINNV1FRepairModel(input_dim=int(bundle["features"].shape[1])).to(device)
    if not args.init_checkpoint.exists():
        raise SystemExit(f"Init checkpoint missing: {args.init_checkpoint}")
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: list[dict] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, physics, grad_clip=args.grad_clip)
        val_metrics = evaluate_losses(model, val_loader, device, physics)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        val_loss = val_metrics.get("loss", float("inf"))
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(
                f"epoch {epoch} val_loss={val_metrics['loss']:.4f} "
                f"retention={train_metrics.get('old_demo_retention', 0):.4f} "
                f"ranking={train_metrics.get('pairwise_ranking', 0):.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_dir / "model_v1f_100base.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(bundle["features"].shape[1]),
            "model_version": "V1-F-100Base",
            "init_checkpoint": str(args.init_checkpoint),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        },
        ckpt_path,
    )
    log = {
        "model_version": "V1-F-100Base",
        "dataset": str(args.dataset),
        "checkpoint": str(ckpt_path),
        "init_checkpoint": str(args.init_checkpoint),
        "best_val_loss": best_val,
        "loss_components": [
            "old_demo_retention",
            "success_focal",
            "pairwise_ranking",
            "failure_weighted_component",
            "total_consistency",
            "uncertainty_nll",
        ],
        "history_tail": history[-5:],
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(ckpt_path), "best_val_loss": best_val}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
