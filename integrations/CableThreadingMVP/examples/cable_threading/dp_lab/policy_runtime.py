from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import DpLabConfig
from .model import ConditionalDiffusionPolicy
from .normalizer import DatasetStats


def load_policy_from_checkpoint(checkpoint_path: str | Path, device: str = "cuda") -> tuple[ConditionalDiffusionPolicy, DpLabConfig, DatasetStats, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    train_config = payload.get("train_config") or {}
    fields = DpLabConfig.__dataclass_fields__  # type: ignore[attr-defined]
    cfg = DpLabConfig(**{k: train_config[k] for k in fields if k in train_config})
    stats = DatasetStats.from_dict(payload["normalizer"])

    device_obj = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusionPolicy(
        action_dim=cfg.action_dim,
        horizon=cfg.horizon,
        low_dim_dim=cfg.resolved_low_dim_dim,
        n_obs_steps=cfg.n_obs_steps,
        num_cameras=cfg.num_cameras,
        image_size=cfg.image_size,
        num_diffusion_steps=cfg.num_diffusion_steps,
        vision_encoder=cfg.vision_encoder,
    ).to(device_obj)
    model.load_state_dict(payload["state_dict"])
    _set_inference_mode(model)
    return model, cfg, stats, payload


def _set_inference_mode(model: ConditionalDiffusionPolicy) -> None:
    """Put diffusion head in eval; keep vision BatchNorm in train.

    ResNet18 uses ImageNet running stats in eval(), which does not match robot
    images seen during DP training (batch stats). eval() caused ~1e4 action
    outputs and one-direction arm drift in simulation.
    """
    model.eval()
    if model.vision is not None:
        model.vision.train()


def _resize_image_hwc(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)


class DiffusionPolicyAdapter:
    """仿真 rollout 用：与 RobomimicPolicyAdapter 相同 reset/act 接口。"""

    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        self.model, self.cfg, self.stats, self.payload = load_policy_from_checkpoint(checkpoint, device=device)
        self.device = next(self.model.parameters()).device
        self._image_history: deque[np.ndarray] = deque(maxlen=self.cfg.n_obs_steps)
        self._low_history: deque[np.ndarray] = deque(maxlen=self.cfg.n_obs_steps)
        self._action_queue: deque[np.ndarray] = deque()

    def reset(self) -> None:
        self._image_history.clear()
        self._low_history.clear()
        self._action_queue.clear()

    def _extract_frame(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        images = []
        for key in self.cfg.image_keys:
            if key not in obs:
                raise KeyError(f"obs missing image key {key!r}, available={sorted(obs.keys())}")
            img = np.asarray(obs[key], dtype=np.uint8)
            if img.ndim == 3:
                img = _resize_image_hwc(img, self.cfg.image_size)
            images.append(img)

        low_parts = []
        for key in self.cfg.low_dim_keys:
            if key not in obs:
                raise KeyError(f"obs missing low_dim key {key!r}, available={sorted(obs.keys())}")
            low_parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        low = np.concatenate(low_parts, axis=0)
        images_arr = np.stack(images, axis=0)  # (N_cam, H, W, C)
        return images_arr, low

    def _push_obs(self, obs: dict[str, Any]) -> None:
        images, low = self._extract_frame(obs)
        self._image_history.append(images)
        self._low_history.append(low)

    def _build_batch(self) -> dict[str, torch.Tensor]:
        images_list = list(self._image_history)
        low_list = list(self._low_history)
        while len(images_list) < self.cfg.n_obs_steps:
            images_list.insert(0, images_list[0])
            low_list.insert(0, low_list[0])

        images = np.stack(images_list[-self.cfg.n_obs_steps :], axis=0).astype(np.float32) / 255.0
        images = np.transpose(images, (0, 1, 4, 2, 3))  # T, N_cam, C, H, W
        low = np.stack(low_list[-self.cfg.n_obs_steps :], axis=0).astype(np.float32)
        low_norm = self.stats.low_dim.normalize(low)

        return {
            "images": torch.from_numpy(images).unsqueeze(0).to(self.device),
            "low_dim": torch.from_numpy(low_norm).unsqueeze(0).to(self.device),
        }

    @torch.no_grad()
    def _replenish_actions(self) -> None:
        batch = self._build_batch()
        actions_norm = self.model.predict_actions(batch, num_inference_steps=self.cfg.num_inference_steps)[0]
        actions = self.stats.action.unnormalize(actions_norm.cpu().numpy())
        for idx in range(min(self.cfg.n_action_steps, len(actions))):
            self._action_queue.append(actions[idx].astype(np.float32))

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        self._push_obs(obs)
        if not self._action_queue:
            self._replenish_actions()
        if not self._action_queue:
            raise RuntimeError("diffusion policy produced empty action queue")
        return self._action_queue.popleft()
