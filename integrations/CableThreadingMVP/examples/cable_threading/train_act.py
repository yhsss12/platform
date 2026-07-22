#!/usr/bin/env python3
"""Platform entry: ACT training for image + proprio HDF5 datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.cable_threading.act_lab.config import ActLabConfig
from examples.cable_threading.act_lab.trainer import train_act_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ACT policy")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = ActLabConfig.from_yaml(args.config)
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.seed is not None:
        cfg.seed = args.seed
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.kl_weight is not None:
        cfg.kl_weight = args.kl_weight

    if args.debug:
        cfg.image_size = min(cfg.image_size, 64)
        cfg.batch_size = args.batch_size or 4
        cfg.max_train_samples = 64
        cfg.max_batches_per_epoch = 8
        cfg.enc_layers = min(cfg.enc_layers, 2)
        cfg.hidden_dim = min(cfg.hidden_dim, 256)

    metrics_path = Path(args.metrics_path).expanduser().resolve() if args.metrics_path else None
    ckpt = train_act_policy(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        cfg=cfg,
        device=args.device,
        debug=args.debug,
        metrics_path=metrics_path,
    )
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
