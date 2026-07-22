#!/usr/bin/env python3
"""Train V1-F-100Base-R1 PINN（demo_uid 隔离 + old-demo ranking gate early stop）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_V1F_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1F_DIR.parent.parent
for path in (_EXPERIMENT_DIR, _V1F_DIR.parent, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pinn_v1f_repair_model import (  # noqa: E402
    PINNV1FRepairModel,
    V1FPhysicsLossConfig,
    compute_v1f_100base_r1_losses,
)
from train_pinn_v1f_balanced_model import DemoPairBatchSampler, set_seed  # noqa: E402
from v1f_100base_r1_utils import (  # noqa: E402
    ALIGNED_ORIGINAL_JSONL,
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_DATASET_NPZ,
    DEFAULT_LOSS_GATE_REPORT,
    DEFAULT_TRAINED_MODEL,
    split_indices_by_demo_uid,
)
from v1f_100base_r1_validation_pool import evaluate_old_demo_ranking_gate  # noqa: E402
from v1f_repair_dataset import V1FRepairDataset, load_v1f_npz  # noqa: E402

DEFAULT_OUTPUT = DEFAULT_TRAINED_MODEL.parent


def train_one_epoch(model, loader, optimizer, device, physics, *, grad_clip: float = 1.0) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["features"])
        losses = compute_v1f_100base_r1_losses(out, batch, physics=physics)
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
        losses = compute_v1f_100base_r1_losses(out, batch, physics=physics)
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-F-100Base-R1 model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--loss-gate-report", type=Path, default=DEFAULT_LOSS_GATE_REPORT)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ranking-gate-every", type=int, default=20, help="每 N epoch 运行 old-demo ranking gate")
    parser.add_argument("--skip-loss-gate-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只校验配置，不训练")
    args = parser.parse_args()

    if "balanced_v2" in str(args.init_checkpoint) or "v2" in args.init_checkpoint.name:
        raise SystemExit(f"Refusing v2 init checkpoint: {args.init_checkpoint}")

    if not args.skip_loss_gate_check:
        if not args.loss_gate_report.exists():
            raise SystemExit(
                f"Pre-train loss gate report missing: {args.loss_gate_report}\n"
                "Run run_v1f_100base_r1_pretrain_loss_gate.py after dataset build."
            )
        gate = json.loads(args.loss_gate_report.read_text(encoding="utf-8")).get("training_gate", {})
        if not gate.get("training_allowed"):
            raise SystemExit(
                "Pre-train loss gate blocked training:\n- "
                + "\n- ".join(str(r) for r in gate.get("block_reasons", ["unknown"]))
            )

    if args.dry_run:
        print(json.dumps({"dry_run": True, "training_skipped": True}, indent=2))
        return 0

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_v1f_npz(args.dataset)
    train_idx, val_idx = split_indices_by_demo_uid(bundle["demo_group_id"], val_frac=args.val_frac, seed=args.seed)

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
    ranking_gate_history: list[dict] = []
    best_val = float("inf")
    best_state = None
    best_ranking_gate: dict | None = None
    best_epoch: int | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, physics, grad_clip=args.grad_clip)
        val_metrics = evaluate_losses(model, val_loader, device, physics)
        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        ranking_gate_passed = False
        ranking_gate_result: dict | None = None
        val_loss = val_metrics.get("loss", float("inf"))
        should_check_ranking = (
            np.isfinite(val_loss)
            and (val_loss < best_val or epoch % args.ranking_gate_every == 0 or epoch == args.epochs)
        )
        if should_check_ranking:
            tmp_ckpt = args.output_dir / "_ranking_gate_tmp.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": int(bundle["features"].shape[1]),
                    "model_version": "V1-F-100Base-R1-ranking-gate-tmp",
                },
                tmp_ckpt,
            )
            ranking_gate_result = evaluate_old_demo_ranking_gate(
                candidate_model=tmp_ckpt,
                baseline_model=args.baseline_checkpoint,
                aligned_jsonl=ALIGNED_ORIGINAL_JSONL,
                seed=args.seed,
            )
            tmp_ckpt.unlink(missing_ok=True)
            ranking_gate_passed = bool(ranking_gate_result.get("passed"))
            ranking_gate_history.append({"epoch": epoch, "val_loss": val_loss, **ranking_gate_result})

        record["ranking_gate_passed"] = ranking_gate_passed
        record["ranking_gate_checked"] = should_check_ranking
        history.append(record)

        if np.isfinite(val_loss) and val_loss < best_val and ranking_gate_passed:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ranking_gate = ranking_gate_result

        if epoch % 20 == 0:
            print(
                f"epoch {epoch} val_loss={val_metrics['loss']:.4f} "
                f"retention={train_metrics.get('old_demo_retention', 0):.4f} "
                f"ranking={train_metrics.get('pairwise_ranking', 0):.4f} "
                f"ranking_gate={'PASS' if ranking_gate_passed else 'FAIL'}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_dir / "model_v1f_100base_r1.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(bundle["features"].shape[1]),
            "model_version": "V1-F-100Base-R1",
            "init_checkpoint": str(args.init_checkpoint),
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "split_policy": "demo_uid",
        },
        ckpt_path,
    )
    log = {
        "model_version": "V1-F-100Base-R1",
        "dataset": str(args.dataset),
        "checkpoint": str(ckpt_path),
        "init_checkpoint": str(args.init_checkpoint),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "best_ranking_gate": best_ranking_gate,
        "early_stopping_policy": "val_loss + old_demo_ranking_gate",
        "loss_fn": "compute_v1f_100base_r1_losses",
        "ranking_gate_history": ranking_gate_history[-5:],
        "history_tail": history[-5:],
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(ckpt_path), "best_val_loss": best_val}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
