#!/usr/bin/env python3
"""P9-B: Train NutAssembly-PINN v1 trajectory delta repair model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[3]
_INTEGRATION = _REPO / "integrations" / "NutAssemblyMimicGen"
_DATA_ROOT = Path(os.environ.get("EAI_DATA_ROOT") or (_REPO / "eai-data")).expanduser()
if str(_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION))

from utils.pinn_repair_v1 import DELTA_DIM, FEATURE_DIM, TrajectoryDeltaMLP  # noqa: E402

DEFAULT_DATASET = _DATA_ROOT / "runs/nut_assembly/pinn_training/repair_training_dataset.npz"
DEFAULT_OUTPUT = _DATA_ROOT / "assets/models/pinn/nut_assembly_pinn_v1"


@dataclass
class LossWeights:
    reconstruction: float = 1.0
    align: float = 0.35
    insert_axis: float = 0.25
    pose_tilt: float = 0.15
    smoothness: float = 0.15
    workspace: float = 0.10


class RepairDataset(Dataset):
    def __init__(self, npz_path: Path, indices: np.ndarray) -> None:
        bundle = np.load(npz_path)
        self.features = bundle["features"].astype(np.float32)
        self.deltas = bundle["trajectory_delta"].astype(np.float32)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        return {
            "features": torch.from_numpy(self.features[i]),
            "delta": torch.from_numpy(self.deltas[i]),
        }


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = max(1, int(n * val_frac))
    return idx[val_n:], idx[:val_n]


def compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor,
    *,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    l_recon = nn.functional.mse_loss(pred, target)
    segment = 48
    pred_seg = pred.view(-1, segment, 3)
    target_seg = target.view(-1, segment, 3)
    pred_xy = pred_seg[:, :, :2]
    target_xy = target_seg[:, :, :2]
    l_align = nn.functional.mse_loss(pred_xy, target_xy)
    pred_z = pred_seg[:, :, 2]
    target_z = target_seg[:, :, 2]
    l_insert = nn.functional.mse_loss(pred_z, target_z)
    tilt_proxy = torch.mean(torch.abs(pred_seg[:, -1, :2]), dim=1)
    l_tilt = torch.mean(tilt_proxy)
    if pred_seg.shape[1] > 1:
        l_smooth = torch.mean(torch.abs(pred_seg[:, 1:] - pred_seg[:, :-1]))
    else:
        l_smooth = torch.zeros((), device=pred.device)
    workspace_penalty = torch.relu(torch.abs(pred_seg) - 0.08).mean()
    loss = (
        weights.reconstruction * l_recon
        + weights.align * l_align
        + weights.insert_axis * l_insert
        + weights.pose_tilt * l_tilt
        + weights.smoothness * l_smooth
        + weights.workspace * workspace_penalty
    )
    return {
        "loss": loss,
        "L_reconstruction": l_recon,
        "L_align": l_align,
        "L_insert_axis": l_insert,
        "L_pose_tilt": l_tilt,
        "L_smoothness": l_smooth,
        "L_workspace": workspace_penalty,
    }


def train_one_epoch(model, loader, optimizer, device, weights) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        features = batch["features"].to(device)
        target = batch["delta"].to(device)
        optimizer.zero_grad()
        pred = model(features)
        losses = compute_losses(pred, target, features, weights=weights)
        losses["loss"].backward()
        optimizer.step()
        bs = len(features)
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device, weights) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        features = batch["features"].to(device)
        target = batch["delta"].to(device)
        pred = model(features)
        losses = compute_losses(pred, target, features, weights=weights)
        bs = len(features)
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train NutAssembly-PINN v1 repair model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bundle = np.load(args.dataset)
    features = bundle["features"]
    deltas = bundle["trajectory_delta"]
    input_dim = int(features.shape[1])
    output_dim = int(deltas.shape[1])
    train_idx, val_idx = split_indices(len(features), args.val_frac, args.seed)

    train_loader = DataLoader(RepairDataset(args.dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RepairDataset(args.dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryDeltaMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = LossWeights()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, device, weights)
        val_losses = evaluate(model, val_loader, device, weights)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses["loss"],
                "val_loss": val_losses["loss"],
            }
        )
        if val_losses["loss"] < best_val:
            best_val = val_losses["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == args.epochs:
            print(f"[p9-b] epoch={epoch} train={train_losses['loss']:.6f} val={val_losses['loss']:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = args.output_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "model_id": "nut_assembly_pinn_v1",
            "pipeline_version": "model_v1",
        },
        model_path,
    )

    train_log = {
        "epochs": args.epochs,
        "best_val_loss": best_val,
        "history": history[-10:],
        "device": str(device),
        "sampleCount": int(len(features)),
        "inputDim": input_dim,
        "outputDim": output_dim,
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")

    final_val = evaluate(model, val_loader, device, weights)
    eval_report = {
        "valLoss": final_val["loss"],
        "lossComponents": {k: float(v) for k, v in final_val.items()},
        "sampleCount": int(len(features)),
        "valSampleCount": int(len(val_idx)),
    }
    (args.output_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")

    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    trained_at = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    dataset_rel = str(args.dataset.resolve().relative_to(_REPO)) if args.dataset.is_relative_to(_REPO) else str(args.dataset)
    script_rel = "integrations/NutAssemblyMimicGen/scripts/train_pinn_repair_model.py"

    metadata = {
        "modelId": "nut_assembly_pinn_v1",
        "taskTemplateId": "nut_assembly_single_arm",
        "modelType": "pinn_repair",
        "displayName": "NutAssembly-PINN v1",
        "repairStages": ["align_over_peg", "descend_insert"],
        "inputSchema": "nut_assembly_repair_v1",
        "outputSchema": "trajectory_delta_v1",
        "status": "available",
        "backend": "torch_model",
        "modelFile": "model.pt",
        "pipelineVersionHeuristic": "v1_heuristic",
        "pipelineVersionModel": "model_v1",
        "modelSource": "trained_in_p9",
        "trainingScript": script_rel,
        "trainingDataset": dataset_rel,
        "trainingSamples": int(len(features)),
        "inputDim": input_dim,
        "outputDim": output_dim,
        "hiddenDim": args.hidden_dim,
        "numLayers": args.num_layers,
        "createdAt": trained_at,
        "trainedAt": trained_at,
        "modelSha256": model_sha256,
        "modelSizeBytes": len(model_bytes),
        "parentModel": None,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[p9-b] model saved: {model_path}")
    print(f"[p9-b] best_val_loss={best_val:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
