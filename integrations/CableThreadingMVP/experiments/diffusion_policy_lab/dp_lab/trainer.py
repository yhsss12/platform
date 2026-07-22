from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dp_lab.config import DpLabConfig
from dp_lab.dataset import CableThreadingDpDataset, compute_dataset_stats
from dp_lab.model import ConditionalDiffusionPolicy
from dp_lab.normalizer import DatasetStats

logger = logging.getLogger(__name__)


class EmaModel:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return self.shadow.state_dict()


def train_diffusion_policy(
    *,
    dataset_path: str | Path,
    out_dir: str | Path,
    cfg: DpLabConfig,
    device: str = "cuda",
    debug: bool = False,
) -> Path:
    out = Path(out_dir).expanduser().resolve()
    ckpt_dir = out / "checkpoints"
    log_dir = out / "logs"
    config_dir = out / "config"
    for d in (ckpt_dir, log_dir, config_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    torch.manual_seed(cfg.seed)
    device_obj = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    dataset_path = Path(dataset_path).expanduser().resolve()

    stats = compute_dataset_stats(dataset_path, cfg)
    max_windows = cfg.max_train_windows
    if debug and max_windows is None:
        max_windows = 64
    train_ds = CableThreadingDpDataset(
        dataset_path,
        cfg,
        stats,
        split="train",
        max_windows=max_windows,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = ConditionalDiffusionPolicy(
        action_dim=cfg.action_dim,
        horizon=cfg.horizon,
        low_dim_dim=cfg.low_dim_dim,
        n_obs_steps=cfg.n_obs_steps,
        num_cameras=cfg.num_cameras,
        image_size=cfg.image_size,
        num_diffusion_steps=cfg.num_diffusion_steps,
        vision_encoder=cfg.vision_encoder,
    ).to(device_obj)

    ema = EmaModel(model, decay=cfg.ema_decay) if cfg.use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    epochs = 2 if debug else cfg.num_epochs
    max_batches = cfg.max_batches_per_epoch
    if debug and max_batches is None:
        max_batches = 8
    train_config = {
        **cfg.to_dict(),
        "dataset": str(dataset_path),
        "backend": "diffusion_policy",
        "debug": debug,
        "normalizer": stats.to_dict(),
    }
    (config_dir / "train_config.json").write_text(
        json.dumps(train_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "device=%s demos=%s windows=%s vision=%s",
        device_obj,
        len(train_ds.demo_names),
        len(train_ds),
        cfg.vision_encoder,
    )
    best_loss = float("inf")
    final_ckpt = ckpt_dir / "model_final.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = {k: v.to(device_obj) for k, v in batch.items()}
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            epoch_loss += float(loss.item())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

        avg_loss = epoch_loss / max(batches, 1)
        logger.info("Epoch %s Loss: %.6f", epoch, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            state_dict = ema.shadow.state_dict() if ema is not None else model.state_dict()
            payload = {
                "state_dict": state_dict,
                "backend": "diffusion_policy",
                "shape_meta": {
                    "action_dim": cfg.action_dim,
                    "horizon": cfg.horizon,
                    "n_obs_steps": cfg.n_obs_steps,
                    "n_action_steps": cfg.n_action_steps,
                    "image_keys": cfg.image_keys,
                    "low_dim_keys": cfg.low_dim_keys,
                    "image_size": cfg.image_size,
                    "vision_encoder": cfg.vision_encoder,
                },
                "normalizer": stats.to_dict(),
                "train_config": train_config,
            }
            torch.save(payload, final_ckpt)

    if not final_ckpt.is_file():
        raise RuntimeError("training finished without checkpoint")

    logger.info("saved checkpoint: %s", final_ckpt)
    return final_ckpt
