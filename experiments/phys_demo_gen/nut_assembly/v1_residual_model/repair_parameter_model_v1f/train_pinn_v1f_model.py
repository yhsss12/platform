#!/usr/bin/env python3
"""V1-F：训练 Uncertainty-aware PINN Repair Parameter Model。"""
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
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))
if str(_V1F_DIR) not in sys.path:
    sys.path.insert(0, str(_V1F_DIR))

from pinn_v1f_repair_model import (  # noqa: E402
    INPUT_DIM_V1F,
    PINNV1FRepairModel,
    V1FPhysicsLossConfig,
    compute_v1f_pinn_losses,
)
from train_residual_model import split_indices  # noqa: E402
from v1f_repair_dataset import V1FRepairDataset, load_v1f_npz  # noqa: E402

_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "repair_parameter_dataset_v1f.npz"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, device, physics, *, grad_clip: float = 0.0) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["features"])
        losses = compute_v1f_pinn_losses(out, batch, physics=physics)
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
        losses = compute_v1f_pinn_losses(out, batch, physics=physics)
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-F uncertainty-aware PINN repair model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-filename",
        default="model_v1f.pt",
        help="Output checkpoint filename under --output-dir",
    )
    parser.add_argument(
        "--model-version",
        default="V1-F_uncertainty_aware_repair_field",
        help="Model version string stored in checkpoint",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Fine-tune from existing checkpoint")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--physics-profile",
        choices=("full", "aligned_finetune"),
        default="full",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_v1f_npz(args.dataset)
    train_idx, val_idx = split_indices(len(bundle["features"]), args.val_frac, args.seed)

    train_loader = DataLoader(
        V1FRepairDataset(args.dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        V1FRepairDataset(args.dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = (
        V1FPhysicsLossConfig.aligned_finetune()
        if args.physics_profile == "aligned_finetune"
        else V1FPhysicsLossConfig.full()
    )
    model = PINNV1FRepairModel(
        input_dim=int(bundle["features"].shape[1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history: list[dict] = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, physics, grad_clip=args.grad_clip)
        val_metrics = evaluate_losses(model, val_loader, device, physics)
        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        val_loss = val_metrics.get("loss", float("inf"))
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            print(f"epoch {epoch} val_loss={val_metrics['loss']:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = args.output_dir / args.checkpoint_filename
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(bundle["features"].shape[1]),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "model_version": args.model_version,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        },
        checkpoint_path,
    )

    log = {
        "model_version": args.model_version,
        "dataset": str(args.dataset),
        "checkpoint": str(checkpoint_path),
        "input_dim": int(bundle["features"].shape[1]),
        "expected_input_dim": INPUT_DIM_V1F,
        "num_samples": int(len(bundle["features"])),
        "best_val_loss": best_val,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "history_tail": history[-5:],
    }
    (args.output_dir / "train_v1f_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "checkpoint": str(checkpoint_path), "best_val_loss": best_val}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
