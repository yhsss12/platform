from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import ActLabConfig
from .dataset import ActDataset
from .model import ActPolicy

logger = logging.getLogger(__name__)


def _append_metrics(metrics_path: Path, payload: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def train_act_policy(
    *,
    dataset_path: str | Path,
    out_dir: str | Path,
    cfg: ActLabConfig,
    device: str = "cuda",
    debug: bool = False,
    metrics_path: Path | None = None,
) -> Path:
    out = Path(out_dir).expanduser().resolve()
    ckpt_dir = out / "checkpoints"
    log_dir = out / "logs"
    config_dir = out / "config"
    for directory in (ckpt_dir, log_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

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

    max_samples = cfg.max_train_samples
    if debug and max_samples is None:
        max_samples = 64
    train_ds = ActDataset(dataset_path, cfg, split="train", max_samples=max_samples)
    val_ds = ActDataset(dataset_path, cfg, split="val", max_samples=max(8, max_samples // 5) if max_samples else None)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

    state_dim = int(cfg.state_dim or 0)
    if train_ds:
        sample = train_ds[0]
        proprio_dim = int(sample["proprio"].numel())
        if state_dim <= 0 or state_dim != proprio_dim:
            if state_dim > 0 and state_dim != proprio_dim:
                logger.warning(
                    "cfg.state_dim=%s mismatches proprio dim=%s; using proprio dim",
                    state_dim,
                    proprio_dim,
                )
            state_dim = proprio_dim

    model = ActPolicy(
        action_dim=cfg.action_dim,
        chunk_size=cfg.chunk_size,
        state_dim=state_dim,
        num_cameras=cfg.num_cameras,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        kl_weight=cfg.kl_weight,
        enc_layers=cfg.enc_layers,
        nheads=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device_obj)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    epochs = 2 if debug else cfg.num_epochs
    max_batches = cfg.max_batches_per_epoch
    if debug and max_batches is None:
        max_batches = 8

    train_config = {**cfg.to_dict(), "dataset": str(dataset_path), "backend": "act", "debug": debug, "state_dim": state_dim}
    (config_dir / "train_config.json").write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "device=%s demos=%s samples=%s cameras=%s chunk=%s",
        device_obj,
        len(train_ds.demo_names),
        len(train_ds),
        cfg.num_cameras,
        cfg.chunk_size,
    )

    best_loss = float("inf")
    best_ckpt = ckpt_dir / "model_best.pt"
    final_ckpt = ckpt_dir / "model_final.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = {k: v.to(device_obj) for k, v in batch.items()}
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
        avg_train = train_loss / max(batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device_obj) for k, v in batch.items()}
                val_loss += float(model(batch).item())
                val_batches += 1
                if max_batches is not None and val_batches >= max(2, max_batches // 2):
                    break
        avg_val = val_loss / max(val_batches, 1) if val_batches else None

        logger.info("Epoch %s Loss: %.6f", epoch, avg_train)
        if avg_val is not None:
            logger.info("Validation Epoch %s Loss: %.6f", epoch, avg_val)

        progress = epoch / epochs
        metric_row: dict[str, Any] = {
            "epoch": epoch,
            "totalEpochs": epochs,
            "trainLoss": avg_train,
            "currentLoss": avg_train,
            "progress": progress,
        }
        if avg_val is not None:
            metric_row["validLoss"] = avg_val
        if metrics_path is not None:
            _append_metrics(metrics_path, metric_row)

        payload = {
            "state_dict": model.state_dict(),
            "backend": "act",
            "shape_meta": {
                "action_dim": cfg.action_dim,
                "action_key": cfg.action_key,
                "action_mode": cfg.action_mode,
                "controller_type": cfg.controller_type,
                "eval_executor": cfg.eval_executor,
                "trained_action_mode": cfg.trained_action_mode or cfg.action_mode,
                "gripper_action_key": cfg.gripper_action_key,
                "chunk_size": cfg.chunk_size,
                "state_dim": state_dim,
                "low_dim_dim": cfg.low_dim_dim,
                "image_keys": cfg.image_keys,
                "low_dim_keys": cfg.low_dim_keys,
                "preferred_policy_schema_id": cfg.preferred_policy_schema_id,
                "observation_schema": cfg.observation_schema,
                "action_schema": cfg.action_schema,
                "controller_schema": cfg.controller_schema,
                "side_channel_schema": cfg.side_channel_schema,
            },
            "config": train_config,
            "train_config": train_config,
        }
        torch.save(payload, final_ckpt)
        if avg_val is not None and avg_val < best_loss:
            best_loss = avg_val
            torch.save(payload, best_ckpt)
        elif avg_val is None and avg_train < best_loss:
            best_loss = avg_train
            torch.save(payload, best_ckpt)

    logger.info("saved checkpoint: %s", final_ckpt)
    return final_ckpt
