"""V1-D：训练 PINN-style Physics-Informed Residual Energy Model。"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pinn_residual_energy_model import PINNResidualEnergyModel, PhysicsLossConfig, compute_pinn_losses
from residual_dataset import ResidualEnergyDataset, load_npz_dataset
from train_residual_model import split_indices

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET_V1C = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c" / "training_dataset.npz"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_pinn"

ABLATION_PRESETS = {
    "full": PhysicsLossConfig.full(),
    "no_phys_components": PhysicsLossConfig.no_phys_components(),
    "no_total_consistency": PhysicsLossConfig.no_total_consistency(),
    "no_margin": PhysicsLossConfig.no_margin(),
    "supervised_only": PhysicsLossConfig.supervised_only(),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, device, physics: PhysicsLossConfig) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["features"])
        losses = compute_pinn_losses(out, batch, batch["features"], physics=physics)
        losses["loss"].backward()
        optimizer.step()
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_losses(model, loader, device, physics: PhysicsLossConfig) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["features"])
        losses = compute_pinn_losses(out, batch, batch["features"], physics=physics)
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def train_pinn_model(
    *,
    dataset_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: Path | None = None,
    physics: PhysicsLossConfig | None = None,
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    seed: int = 42,
    save_model: bool = True,
    model_tag: str = "v1d_pinn",
) -> tuple[PINNResidualEnergyModel, dict]:
    physics = physics or PhysicsLossConfig.full()
    set_seed(seed)
    bundle = load_npz_dataset(dataset_path)

    train_ds = ResidualEnergyDataset(dataset_path, train_idx)
    val_ds = ResidualEnergyDataset(dataset_path, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINNResidualEnergyModel(
        input_dim=bundle["features"].shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, physics)
        val_metrics = evaluate_losses(model, val_loader, device, physics)
        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "model_tag": model_tag,
        "physics_config": {
            "use_phys_components": physics.use_phys_components,
            "use_total_consistency": physics.use_total_consistency,
            "use_margin": physics.use_margin,
        },
        "best_val_loss": best_val,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "feature_dim": int(bundle["features"].shape[1]),
        "device": str(device),
        "history_tail": history[-5:],
    }

    if save_model and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": bundle["features"].shape[1],
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "model_version": "v1d_pinn",
                "physics_config": info["physics_config"],
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
            },
            output_dir / "model.pt",
        )
    return model, info


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-D PINN residual energy model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_V1C)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--physics-ablation",
        choices=list(ABLATION_PRESETS.keys()),
        default="full",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_src = args.dataset.parent / "training_dataset_summary.json"
    if summary_src.exists():
        shutil.copy2(summary_src, args.output_dir / "training_dataset_summary.json")

    bundle = load_npz_dataset(args.dataset)
    train_idx, val_idx = split_indices(len(bundle["features"]), args.val_frac, args.seed)
    physics = ABLATION_PRESETS[args.physics_ablation]

    model, info = train_pinn_model(
        dataset_path=args.dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        output_dir=args.output_dir,
        physics=physics,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        model_tag=f"v1d_pinn_{args.physics_ablation}",
    )

    log = {
        "model_version": "V1-D_PINN_style_residual_energy_mlp",
        "dataset": str(args.dataset),
        "physics_ablation": args.physics_ablation,
        "epochs": args.epochs,
        **info,
        "notes": [
            "PINN-style: explicit Nut Assembly geometry/grasp physics residuals in loss.",
            "Not a PDE PINN; not PINA; not claimed as final generalization model.",
            "Use group split evaluation as primary generalization check.",
        ],
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "best_val_loss": info["best_val_loss"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
