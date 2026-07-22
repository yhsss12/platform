from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import ActLabConfig
from .model import ActPolicy


def load_act_policy_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> tuple[ActPolicy, ActLabConfig, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid ACT checkpoint format: {checkpoint_path}")

    shape_meta = dict(payload.get("shape_meta") or {})
    config_raw = dict(payload.get("config") or {})
    fields = ActLabConfig.__dataclass_fields__  # type: ignore[attr-defined]
    cfg = ActLabConfig(**{k: v for k, v in config_raw.items() if k in fields})

    if shape_meta.get("image_keys"):
        cfg.image_keys = list(shape_meta["image_keys"])
    if shape_meta.get("low_dim_keys"):
        cfg.low_dim_keys = list(shape_meta["low_dim_keys"])
    if shape_meta.get("chunk_size"):
        cfg.chunk_size = int(shape_meta["chunk_size"])
    if shape_meta.get("action_dim"):
        cfg.action_dim = int(shape_meta["action_dim"])
    if shape_meta.get("action_key"):
        cfg.action_key = str(shape_meta["action_key"])
    if shape_meta.get("action_mode"):
        cfg.action_mode = str(shape_meta["action_mode"])
    if shape_meta.get("controller_type"):
        cfg.controller_type = str(shape_meta["controller_type"])
    if shape_meta.get("eval_executor"):
        cfg.eval_executor = str(shape_meta["eval_executor"])
    if shape_meta.get("trained_action_mode"):
        cfg.trained_action_mode = str(shape_meta["trained_action_mode"])
    if shape_meta.get("gripper_action_key"):
        cfg.gripper_action_key = str(shape_meta["gripper_action_key"])
    if shape_meta.get("low_dim_dim") is not None:
        cfg.low_dim_dim = int(shape_meta["low_dim_dim"])
    if shape_meta.get("preferred_policy_schema_id"):
        cfg.preferred_policy_schema_id = str(shape_meta["preferred_policy_schema_id"])
    state_dim = int(shape_meta.get("state_dim") or config_raw.get("state_dim") or cfg.state_dim or 0)

    device_obj = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
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
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg, payload


def _resize_image_hwc(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return (tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def _predict_action_chunk(model: ActPolicy, images: torch.Tensor, proprio: torch.Tensor) -> np.ndarray:
    mu, _logvar = model.encode(images, proprio)
    z = mu
    fused = model.fuse(torch.cat([model.vision(images), model.proprio(proprio)], dim=-1))
    encoded = model.encoder(fused.unsqueeze(1)).squeeze(1)
    pred = model.action_head(torch.cat([encoded, z], dim=-1))
    chunk = pred.view(-1, model.chunk_size, model.action_dim)[0]
    return chunk.cpu().numpy().astype(np.float32)


class ACTPolicyAdapter:
    """仿真 rollout 用：与 RobomimicPolicyAdapter / DiffusionPolicyAdapter 相同 reset/act 接口。"""

    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        self.model, self.cfg, self.payload = load_act_policy_from_checkpoint(checkpoint, device=device)
        self.device = next(self.model.parameters()).device
        self._action_queue: deque[np.ndarray] = deque()
        self.shape_meta = dict(self.payload.get("shape_meta") or {})

    def reset(self) -> None:
        self._action_queue.clear()

    def _extract_frame(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        images = []
        for key in self.cfg.image_keys:
            if key not in obs:
                raise KeyError(f"obs missing image key {key!r}, available={sorted(obs.keys())}")
            img = np.asarray(obs[key])
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            if img.ndim == 3:
                img = _resize_image_hwc(img, self.cfg.image_size)
            images.append(img)

        low_parts = []
        for key in self.cfg.low_dim_keys:
            if key not in obs:
                raise KeyError(f"obs missing low_dim key {key!r}, available={sorted(obs.keys())}")
            low_parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        low = np.concatenate(low_parts, axis=0) if low_parts else np.zeros((0,), dtype=np.float32)
        images_arr = np.stack(images, axis=0) if images else np.zeros((0, self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8)
        return images_arr, low

    def _build_batch(self, images: np.ndarray, low: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size == 0:
            raise ValueError("ACT 模型需要图像观测，但当前帧未提供 camera obs")
        images_f = images.astype(np.float32) / 255.0
        images_t = torch.from_numpy(images_f).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        proprio = torch.from_numpy(low.astype(np.float32)).unsqueeze(0).to(self.device)
        return images_t, proprio

    @torch.no_grad()
    def _replenish_actions(self, obs: dict[str, Any]) -> None:
        images, low = self._extract_frame(obs)
        images_t, proprio = self._build_batch(images, low)
        chunk = _predict_action_chunk(self.model, images_t, proprio)
        for idx in range(min(self.cfg.chunk_size, len(chunk))):
            self._action_queue.append(chunk[idx].astype(np.float32))

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        if not self._action_queue:
            self._replenish_actions(obs)
        if not self._action_queue:
            raise RuntimeError("ACT policy produced empty action queue")
        return self._action_queue.popleft()
