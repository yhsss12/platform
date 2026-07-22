#!/usr/bin/env python3
"""Platform entry: Diffusion Policy training for single-arm cable threading."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.cable_threading.dp_lab.config import DpLabConfig
from examples.cable_threading.dp_lab.trainer import train_diffusion_policy

DEFAULT_CONFIG = Path(__file__).resolve().parent / "dp_configs" / "cable_threading.yaml"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Diffusion Policy (cable threading)")
    parser.add_argument("--dataset", type=str, default=None, help="single HDF5 path (legacy)")
    parser.add_argument("--datasets", type=str, default=None, help="comma-separated HDF5 paths")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--vision-encoder", type=str, default=None, choices=["resnet18", "tiny_cnn"])
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n-obs-steps", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--num-diffusion-steps", type=int, default=None)
    parser.add_argument("--use-ema", type=str, default=None, help="true/false")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = DpLabConfig.from_yaml(args.config) if Path(args.config).is_file() else DpLabConfig()
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.seed is not None:
        cfg.seed = args.seed
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.vision_encoder is not None:
        cfg.vision_encoder = args.vision_encoder
    if args.horizon is not None:
        cfg.horizon = args.horizon
    if args.n_obs_steps is not None:
        cfg.n_obs_steps = args.n_obs_steps
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    if args.num_inference_steps is not None:
        cfg.num_inference_steps = args.num_inference_steps
    if args.num_diffusion_steps is not None:
        cfg.num_diffusion_steps = args.num_diffusion_steps
    if args.use_ema is not None:
        cfg.use_ema = _parse_bool(args.use_ema)
    if args.ema_decay is not None:
        cfg.ema_decay = args.ema_decay
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay

    if args.debug:
        cfg.vision_encoder = "tiny_cnn"
        cfg.image_size = args.image_size or 64
        cfg.batch_size = args.batch_size or 4
        cfg.num_diffusion_steps = 5
        cfg.num_inference_steps = 5
        cfg.use_ema = False
        cfg.max_train_windows = 64
        cfg.max_batches_per_epoch = 8

    dataset_paths: list[str] = []
    if args.datasets:
        dataset_paths = [part.strip() for part in args.datasets.split(",") if part.strip()]
    elif args.dataset:
        dataset_paths = [args.dataset.strip()]
    if not dataset_paths:
        parser.error("--dataset or --datasets is required")

    ckpt = train_diffusion_policy(
        dataset_path=dataset_paths if len(dataset_paths) > 1 else dataset_paths[0],
        out_dir=args.out_dir,
        cfg=cfg,
        device=args.device,
        debug=args.debug,
        init_checkpoint_path=args.init_checkpoint,
    )
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
