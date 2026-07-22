from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import DpLabConfig
from .dataset import CableThreadingDpDataset, compute_dataset_stats
from .model import ConditionalDiffusionPolicy
from .normalizer import DatasetStats

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
    dataset_path: str | Path | list[str | Path],
    out_dir: str | Path,
    cfg: DpLabConfig,
    device: str = "cuda",
    debug: bool = False,
    init_checkpoint_path: str | Path | None = None,
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
    dataset_paths = [
        Path(p).expanduser().resolve()
        for p in (dataset_path if isinstance(dataset_path, list) else [dataset_path])
    ]
    if not dataset_paths:
        raise ValueError("at least one dataset path is required")

    init_ckpt: dict[str, Any] | None = None
    if init_checkpoint_path:
        ckpt_path = Path(init_checkpoint_path).expanduser().resolve()
        payload = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload or "normalizer" not in payload:
            raise ValueError("init checkpoint must contain state_dict and normalizer")
        init_ckpt = payload
        stats = DatasetStats.from_dict(init_ckpt["normalizer"])
    else:
        stats = compute_dataset_stats(dataset_paths, cfg)
    max_windows = cfg.max_train_windows
    if debug and max_windows is None:
        max_windows = 64
    train_ds = CableThreadingDpDataset(
        dataset_paths if len(dataset_paths) > 1 else dataset_paths[0],
        cfg,
        stats,
        split="train",
        max_windows=max_windows,
    )
    if len(train_ds) == 0:
        raise ValueError(
            f"no training windows in {dataset_paths} "
            f"(need length >= horizon={cfg.horizon}, n_obs_steps={cfg.n_obs_steps})"
        )
    drop_last = True
    if len(train_ds) < cfg.batch_size:
        drop_last = False
        logger.warning(
            "train windows (%s) < batch_size (%s); using drop_last=False",
            len(train_ds),
            cfg.batch_size,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last,
    )

    model = ConditionalDiffusionPolicy(
        action_dim=cfg.action_dim,
        horizon=cfg.horizon,
        low_dim_dim=cfg.resolved_low_dim_dim,
        n_obs_steps=cfg.n_obs_steps,
        num_cameras=max(cfg.num_cameras, 0),
        image_size=cfg.image_size,
        num_diffusion_steps=cfg.num_diffusion_steps,
        vision_encoder=cfg.vision_encoder,
    ).to(device_obj)

    if init_ckpt is not None:
        model.load_state_dict(init_ckpt["state_dict"], strict=True)

    ema = EmaModel(model, decay=cfg.ema_decay) if cfg.use_ema else None
    if ema is not None and init_ckpt is not None:
        ema.shadow.load_state_dict(init_ckpt["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    epochs = 2 if debug else cfg.num_epochs
    max_batches = cfg.max_batches_per_epoch
    if debug and max_batches is None:
        max_batches = 8
    train_config = {
        **cfg.to_checkpoint_dict(),
        "dataset": str(dataset_paths[0]),
        "datasets": [str(path) for path in dataset_paths],
        "backend": "diffusion_policy",
        "debug": debug,
        "normalizer": stats.to_dict(),
    }
    if init_checkpoint_path:
        train_config["init_checkpoint"] = str(Path(init_checkpoint_path).expanduser().resolve())
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
    total_batches = 0
    epoch_losses: list[float] = []

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
            total_batches += 1
            if max_batches is not None and batches >= max_batches:
                break

        if batches == 0:
            logger.warning("Epoch %s skipped: no training batches (drop_last=%s)", epoch, drop_last)
            continue

        avg_loss = epoch_loss / batches
        epoch_losses.append(avg_loss)
        logger.info("Epoch %s Loss: %.6f", epoch, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            state_dict = ema.shadow.state_dict() if ema is not None else model.state_dict()
            payload = {
                "state_dict": state_dict,
                "backend": "diffusion_policy",
                "action_key": cfg.action_key,
                "shape_meta": {
                    "action_dim": cfg.action_dim,
                    "action_key": cfg.action_key,
                    "action_mode": cfg.action_mode,
                    "controller_type": cfg.controller_type,
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

    if total_batches == 0:
        raise RuntimeError(
            f"training produced no batches (windows={len(train_ds)}, batch_size={cfg.batch_size}, "
            f"drop_last={drop_last}); refusing to save checkpoint"
        )
    if not final_ckpt.is_file():
        raise RuntimeError("training finished without checkpoint")

    action_samples: list[np.ndarray] = []
    import numpy as np

    with torch.no_grad():
        for batch in train_loader:
            action_samples.append(batch["actions"].cpu().numpy())
            if len(action_samples) >= 8:
                break
    if action_samples:
        action_arr = np.concatenate(action_samples, axis=0).reshape(-1, cfg.action_dim)
        sat_mask = np.abs(action_arr) >= 0.95
        action_saturation_ratio = float(np.mean(sat_mask))
    else:
        action_saturation_ratio = 0.0

    train_diagnostics = {
        "final_loss": float(epoch_losses[-1]) if epoch_losses else None,
        "min_loss": float(min(epoch_losses)) if epoch_losses else None,
        "num_epochs_ran": len(epoch_losses),
        "action_saturation_ratio": action_saturation_ratio,
        "joint_action_clip_ratio": None,
    }
    train_config["train_diagnostics"] = train_diagnostics
    (config_dir / "train_diagnostics.json").write_text(
        json.dumps(train_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config_dir / "train_config.json").write_text(
        json.dumps(train_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("saved checkpoint: %s", final_ckpt)
    logger.info(
        "train_diagnostics final_loss=%.6f min_loss=%.6f action_saturation_ratio=%.4f",
        train_diagnostics["final_loss"] or 0.0,
        train_diagnostics["min_loss"] or 0.0,
        action_saturation_ratio,
    )
    return final_ckpt
