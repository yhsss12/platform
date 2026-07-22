"""V1-C.5：在指定 group split 上训练 residual model。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from group_split_utils import (
    build_failure_mode_holdout_splits,
    build_leave_one_demo_out_splits,
    load_enriched_meta,
    split_train_val,
)
from train_residual_model import train_model

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET_V1C = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c" / "training_dataset.npz"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c_group_split"


def resolve_split(meta_records, split_id: str) -> dict:
    splits = build_leave_one_demo_out_splits(meta_records) + build_failure_mode_holdout_splits(meta_records)
    for split in splits:
        if split["split_id"] == split_id:
            return split
    available = [s["split_id"] for s in splits]
    raise ValueError(f"Unknown split_id={split_id}. Available: {available}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1-C model on one group split")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_V1C)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-id", type=str, required=True)
    parser.add_argument("--model-version", choices=["v1c"], default="v1c")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta_records = load_enriched_meta(args.dataset)
    split = resolve_split(meta_records, args.split_id)
    train_idx, val_idx = split_train_val(split["train_idx"], args.val_frac, args.seed)

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

    split_dir = args.output_dir / "models" / args.split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    model_path = split_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": train_info["feature_dim"],
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "predict_outcome": True,
            "predict_grasp_lift": True,
            "model_version": args.model_version,
            "split_id": args.split_id,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": split["test_idx"].tolist(),
        },
        model_path,
    )

    log = {
        "split_id": args.split_id,
        "split_type": split.get("split_type"),
        "test_demo": split.get("test_demo"),
        "holdout_failure_mode": split.get("holdout_failure_mode"),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "held_out_test_size": int(len(split["test_idx"])),
        **train_info,
        "model_path": str(model_path),
    }
    (split_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
