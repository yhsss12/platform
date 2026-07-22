#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from dp_lab.config import DpLabConfig
from dp_lab.dataset import CableThreadingDpDataset
from dp_lab.model import ConditionalDiffusionPolicy
from dp_lab.normalizer import DatasetStats


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke inference for DP lab checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    payload = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    train_config = payload.get("train_config") or {}
    cfg = DpLabConfig(**{k: train_config[k] for k in DpLabConfig.__dataclass_fields__ if k in train_config})
    stats = DatasetStats.from_dict(payload["normalizer"])
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    ds = CableThreadingDpDataset(args.dataset, cfg, stats, split="train")
    batch = ds[0]
    batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}

    model = ConditionalDiffusionPolicy(
        action_dim=cfg.action_dim,
        horizon=cfg.horizon,
        low_dim_dim=cfg.low_dim_dim,
        n_obs_steps=cfg.n_obs_steps,
        num_cameras=cfg.num_cameras,
        image_size=cfg.image_size,
        num_diffusion_steps=cfg.num_diffusion_steps,
        vision_encoder=cfg.vision_encoder,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    from dp_lab.policy_runtime import _set_inference_mode

    _set_inference_mode(model)

    actions = model.predict_actions(batch, num_inference_steps=cfg.num_inference_steps)
    actions_np = actions[0].cpu().numpy()
    actions_real = stats.action.unnormalize(actions_np)
    print(f"predicted action chunk shape: {actions_real.shape}")
    print(f"first action: {actions_real[0]}")
    print("infer smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
