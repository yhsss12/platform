#!/usr/bin/env python3
"""V1-E：训练 PINN Repair Parameter Residual Field Model。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

_V1E_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1E_DIR.parent
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))
if str(_V1E_DIR) not in sys.path:
    sys.path.insert(0, str(_V1E_DIR))

from pinn_repair_parameter_model import (  # noqa: E402
    INPUT_DIM,
    PINNRepairParameterModel,
    RepairPhysicsLossConfig,
    compute_repair_pinn_losses,
)
from repair_dataset import RepairParameterDataset, load_repair_npz  # noqa: E402
from train_residual_model import split_indices  # noqa: E402

_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "repair_parameter_dataset.npz"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, device, physics) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["features"])
        losses = compute_repair_pinn_losses(out, batch, physics=physics)
        losses["loss"].backward()
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
        losses = compute_repair_pinn_losses(out, batch, physics=physics)
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-E PINN repair parameter model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_repair_npz(args.dataset)
    train_idx, val_idx = split_indices(len(bundle["features"]), args.val_frac, args.seed)

    train_loader = DataLoader(
        RepairParameterDataset(args.dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        RepairParameterDataset(args.dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = RepairPhysicsLossConfig.full()
    model = PINNRepairParameterModel(
        input_dim=int(bundle["features"].shape[1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, physics)
        val_metrics = evaluate_losses(model, val_loader, device, physics)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(bundle["features"].shape[1]),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "model_version": "V1-E_repair_parameter_field",
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        },
        args.output_dir / "model.pt",
    )

    log = {
        "model_version": "V1-E_PINNRepairParameterModel",
        "dataset": str(args.dataset),
        "input_dim": int(bundle["features"].shape[1]),
        "expected_input_dim": INPUT_DIM,
        "best_val_loss": best_val,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "physics_config": {
            "component_supervision": True,
            "total_consistency": True,
            "success_margin": True,
            "monotonic_repair": True,
        },
        "history_tail": history[-5:],
        "notes": [
            "Primary role: repair-parameter candidate pruning, not trajectory-only scoring.",
            "Explicit energy used as physics supervision/baseline only.",
        ],
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "best_val_loss": best_val}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
