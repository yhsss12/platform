#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from dp_lab.config import DpLabConfig
from dp_lab.trainer import train_diffusion_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Diffusion Policy for cable threading (lab)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/run")
    parser.add_argument("--config", type=str, default=str(LAB_ROOT / "configs" / "cable_threading.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n-obs-steps", type=int, default=None)
    parser.add_argument("--vision-encoder", type=str, default=None, choices=["resnet18", "tiny_cnn"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = DpLabConfig.from_yaml(args.config)
    if args.debug:
        cfg.vision_encoder = "tiny_cnn"
        cfg.image_size = args.image_size or 64
        cfg.batch_size = args.batch_size or 4
        cfg.num_diffusion_steps = 5
        cfg.num_inference_steps = 5
        cfg.use_ema = False
        cfg.max_train_windows = 64
        cfg.max_batches_per_epoch = 8
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.horizon is not None:
        cfg.horizon = args.horizon
    if args.n_obs_steps is not None:
        cfg.n_obs_steps = args.n_obs_steps
    if args.vision_encoder is not None:
        cfg.vision_encoder = args.vision_encoder

    ckpt = train_diffusion_policy(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        cfg=cfg,
        device=args.device,
        debug=args.debug,
    )
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
