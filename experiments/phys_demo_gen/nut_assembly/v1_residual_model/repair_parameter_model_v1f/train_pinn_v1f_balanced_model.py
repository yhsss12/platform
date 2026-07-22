#!/usr/bin/env python3
"""Task 3：训练 V1-F-aligned-plus-balanced（focal + pairwise ranking + failure-weighted）。"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

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
    compute_v1f_balanced_losses,
)
from train_residual_model import split_indices  # noqa: E402
from v1f_repair_dataset import V1FRepairDataset, load_v1f_npz  # noqa: E402

_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "repair_parameter_dataset_v1f_plus_balanced.npz"
DEFAULT_INIT = (
    _EXPERIMENT_DIR
    / "outputs"
    / "v1f_aligned_repair_parameter_model"
    / "original_failed"
    / "trained_model"
    / "model_v1f_aligned_original.pt"
)
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "trained_model"


class DemoPairBatchSampler(Sampler[list[int]]):
    """尽量让每个 batch 含同一 demo 的多条样本，便于 pairwise ranking。"""

    def __init__(
        self,
        demo_group_ids: np.ndarray,
        *,
        batch_size: int,
        samples_per_demo: int = 2,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.samples_per_demo = samples_per_demo
        self.rng = random.Random(seed)
        self.groups: dict[int, list[int]] = defaultdict(list)
        for i, gid in enumerate(demo_group_ids):
            self.groups[int(gid)].append(i)
        self.group_ids = [g for g, idxs in self.groups.items() if len(idxs) >= 2 and g >= 0]

    def __iter__(self):
        order = list(self.group_ids)
        self.rng.shuffle(order)
        batch: list[int] = []
        for gid in order:
            idxs = self.groups[gid][:]
            self.rng.shuffle(idxs)
            take = idxs[: self.samples_per_demo]
            if len(batch) + len(take) > self.batch_size:
                if batch:
                    yield batch
                batch = take[:]
            else:
                batch.extend(take)
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        all_indices = [i for idxs in self.groups.values() for i in idxs]
        while batch:
            if len(batch) < self.batch_size:
                batch.extend(self.rng.sample(all_indices, k=min(self.batch_size - len(batch), len(all_indices))))
            yield batch[: self.batch_size]
            batch = batch[self.batch_size :]

    def __len__(self) -> int:
        total = sum(len(v) for v in self.groups.values())
        return max(1, total // self.batch_size)


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
        losses = compute_v1f_balanced_losses(out, batch, physics=physics)
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
        losses = compute_v1f_balanced_losses(out, batch, physics=physics)
        bs = len(batch["features"])
        count += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-F-aligned-plus-balanced model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_v1f_npz(args.dataset)
    train_idx, val_idx = split_indices(len(bundle["features"]), args.val_frac, args.seed)

    train_ds = V1FRepairDataset(args.dataset, train_idx)
    val_ds = V1FRepairDataset(args.dataset, val_idx)
    batch_sampler = DemoPairBatchSampler(
        train_ds.demo_group_id, batch_size=args.batch_size, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = V1FPhysicsLossConfig.balanced()
    model = PINNV1FRepairModel(input_dim=int(bundle["features"].shape[1])).to(device)
    if args.init_checkpoint.exists():
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
        if epoch % 30 == 0:
            print(f"epoch {epoch} val_loss={val_metrics['loss']:.4f} ranking={train_metrics.get('pairwise_ranking', 0):.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_dir / "model_v1f_aligned_plus_balanced.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(bundle["features"].shape[1]),
            "model_version": "V1-F-aligned-plus-balanced",
            "init_checkpoint": str(args.init_checkpoint),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        },
        ckpt_path,
    )
    log = {
        "model_version": "V1-F-aligned-plus-balanced",
        "dataset": str(args.dataset),
        "checkpoint": str(ckpt_path),
        "init_checkpoint": str(args.init_checkpoint),
        "best_val_loss": best_val,
        "history_tail": history[-5:],
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(ckpt_path), "best_val_loss": best_val}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
