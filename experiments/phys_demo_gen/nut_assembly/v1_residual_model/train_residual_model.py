"""V1-A / V1-B / V1-C：训练 PyTorch Residual Energy Model。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from residual_dataset import ResidualEnergyDataset, load_npz_dataset
from residual_energy_model import ENERGY_WEIGHTS, ResidualEnergyModel

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_OUTPUT_V1A = _EXPERIMENT_DIR / "outputs" / "v1_residual_model"
DEFAULT_OUTPUT_V1B = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1b"
DEFAULT_OUTPUT_V1C = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c"
DEFAULT_DATASET_V1A = DEFAULT_OUTPUT_V1A / "training_dataset.npz"
DEFAULT_DATASET_V1B = DEFAULT_OUTPUT_V1B / "training_dataset.npz"
DEFAULT_DATASET_V1C = DEFAULT_OUTPUT_V1C / "training_dataset.npz"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    use_outcome_loss: bool = True,
    use_grasp_lift_loss: bool = False,
) -> dict[str, torch.Tensor]:
    pred_total = out["E_total"]
    pred_components = out["E_components"]
    target_total = batch["target_E_total"]
    target_components = batch["targets_components"]

    l_energy = nn.functional.mse_loss(pred_total, target_total)
    l_components = nn.functional.mse_loss(pred_components, target_components)
    l_success = nn.functional.binary_cross_entropy_with_logits(
        out["success_logit"], batch["success_flag"]
    )
    l_failure = nn.functional.cross_entropy(out["failure_type_logits"], batch["failure_type_idx"])

    weights = ENERGY_WEIGHTS.to(pred_components.device)
    total_consistent = torch.sum(pred_components * weights, dim=-1)
    l_consistency = nn.functional.mse_loss(pred_total, total_consistent)

    total = l_energy + 0.5 * l_components + 0.5 * l_success + 0.2 * l_failure + 0.5 * l_consistency
    losses = {
        "loss": total,
        "L_energy": l_energy,
        "L_components": l_components,
        "L_success": l_success,
        "L_failure": l_failure,
        "L_consistency": l_consistency,
    }

    if use_outcome_loss and "outcome_logits" in out:
        l_outcome = nn.functional.cross_entropy(out["outcome_logits"], batch["outcome_idx"])
        losses["L_outcome"] = l_outcome
        losses["loss"] = total + 0.2 * l_outcome

    if use_grasp_lift_loss and "grasp_success_logit" in out:
        l_grasp = nn.functional.binary_cross_entropy_with_logits(
            out["grasp_success_logit"], batch["grasp_success_flag"]
        )
        losses["L_grasp_success"] = l_grasp
        losses["loss"] = losses["loss"] + 0.3 * l_grasp

    if use_grasp_lift_loss and "lift_success_logit" in out:
        l_lift = nn.functional.binary_cross_entropy_with_logits(
            out["lift_success_logit"], batch["lift_success_flag"]
        )
        losses["L_lift_success"] = l_lift
        losses["loss"] = losses["loss"] + 0.3 * l_lift

    return losses


def train_one_epoch(
    model: ResidualEnergyModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_outcome_loss: bool,
    use_grasp_lift_loss: bool,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["features"])
        losses = compute_losses(
            out,
            batch,
            use_outcome_loss=use_outcome_loss,
            use_grasp_lift_loss=use_grasp_lift_loss,
        )
        losses["loss"].backward()
        optimizer.step()
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_losses(
    model: ResidualEnergyModel,
    loader: DataLoader,
    device: torch.device,
    *,
    use_outcome_loss: bool,
    use_grasp_lift_loss: bool,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["features"])
        losses = compute_losses(
            out,
            batch,
            use_outcome_loss=use_outcome_loss,
            use_grasp_lift_loss=use_grasp_lift_loss,
        )
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = max(1, int(round(n * val_frac))) if n >= 3 else 1
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


def train_model(
    *,
    dataset_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_version: str = "v1c",
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    seed: int = 42,
) -> tuple[ResidualEnergyModel, dict[str, Any]]:
    set_seed(seed)
    bundle = load_npz_dataset(dataset_path)
    train_ds = ResidualEnergyDataset(dataset_path, train_idx)
    val_ds = ResidualEnergyDataset(dataset_path, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_outcome_loss = model_version in ("v1b", "v1c")
    use_grasp_lift_loss = model_version == "v1c"
    model = ResidualEnergyModel(
        input_dim=bundle["features"].shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        predict_outcome=use_outcome_loss,
        predict_grasp_lift=use_grasp_lift_loss,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_outcome_loss=use_outcome_loss,
            use_grasp_lift_loss=use_grasp_lift_loss,
        )
        val_metrics = evaluate_losses(
            model,
            val_loader,
            device,
            use_outcome_loss=use_outcome_loss,
            use_grasp_lift_loss=use_grasp_lift_loss,
        )
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
        "best_val_loss": best_val,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "feature_dim": int(bundle["features"].shape[1]),
        "device": str(device),
        "history_tail": history[-5:],
    }
    return model, info


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-A / V1-B / V1-C residual energy model")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-version", choices=["v1a", "v1b", "v1c"], default="v1b")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["random", "group"], default="random")
    args = parser.parse_args()

    default_datasets = {
        "v1a": DEFAULT_DATASET_V1A,
        "v1b": DEFAULT_DATASET_V1B,
        "v1c": DEFAULT_DATASET_V1C,
    }
    default_outputs = {
        "v1a": DEFAULT_OUTPUT_V1A,
        "v1b": DEFAULT_OUTPUT_V1B,
        "v1c": DEFAULT_OUTPUT_V1C,
    }
    if args.dataset is None:
        args.dataset = default_datasets[args.model_version]
    if args.output_dir is None:
        args.output_dir = default_outputs[args.model_version]

    if args.split_mode == "group":
        print(
            json.dumps(
                {
                    "error": "Use train_group_split_model.py or evaluate_group_split.py for group splits.",
                    "hint": "python evaluate_group_split.py",
                },
                indent=2,
            )
        )
        return 1

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_npz_dataset(args.dataset)
    n = len(bundle["features"])
    train_idx, val_idx = split_indices(n, args.val_frac, args.seed)

    model, train_info = train_model(
        dataset_path=args.dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        model_version=args.model_version,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
    )
    best_val = train_info["best_val_loss"]

    use_outcome_loss = args.model_version in ("v1b", "v1c")
    use_grasp_lift_loss = args.model_version == "v1c"
    model_path = args.output_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": bundle["features"].shape[1],
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "predict_outcome": use_outcome_loss,
            "predict_grasp_lift": use_grasp_lift_loss,
            "model_version": args.model_version,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        },
        model_path,
    )

    version_label = f"{args.model_version.upper()}_grasp_aware_residual_energy_mlp" if args.model_version == "v1c" else f"{args.model_version.upper()}_residual_energy_mlp"
    notes = [
        "Multi-failure-mode learnable residual model prototype; not a final PINN.",
    ]
    if args.model_version == "v1c":
        notes.append("V1-C supervision from V0/V0.5, V2-A, V2-B2.6, V2-B3, V2-B4 grasp_refinement labels.")
    elif args.model_version == "v1b":
        notes.append("V1-B supervision from V0/V0.5, V2-A, V2-B2.6, V2-B3 sim rollout labels.")

    log = {
        "model_version": version_label,
        "dataset": str(args.dataset),
        "num_samples": n,
        "feature_dim": int(bundle["features"].shape[1]),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "epochs": args.epochs,
        "best_val_loss": best_val,
        "device": train_info["device"],
        "history": train_info.get("history_tail", []),
        "notes": notes,
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(model_path), "best_val_loss": best_val}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
