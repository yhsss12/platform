from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .openpi_inference import PI0_SHIM_MARKER, infer_pi0_action_chunk
from .pi0_smoke_inference import (
    DEFAULT_IMAGE_KEYS,
    DEFAULT_LOW_DIM_KEYS,
    predict_smoke_pi0_action,
)


def _load_pi0_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"pi0 checkpoint not found: {checkpoint_path}")

    raw = path.read_bytes()
    if raw == PI0_SHIM_MARKER or raw.startswith(b"PI0_PLATFORM_SHIM"):
        raise RuntimeError("pi0 checkpoint 来自平台 shim，请使用真实 openpi 训练产物进行评测")

    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload

    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        raise ValueError(f"无法加载 pi0 checkpoint: {checkpoint_path}") from exc

    raise ValueError(f"无法加载 pi0 checkpoint: {checkpoint_path}")


def _load_train_config(train_config_path: str | Path | None) -> dict[str, Any]:
    if not train_config_path:
        return {}
    path = Path(train_config_path).expanduser()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


class Pi0PolicyAdapter:
    """仿真 rollout：支持 pi0 LeRobot smoke checkpoint 与 openpi 推理。"""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        *,
        train_config_path: str | Path | None = None,
        task_instruction: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint).expanduser()
        self.payload = _load_pi0_checkpoint_payload(self.checkpoint_path)
        self.train_config = _load_train_config(train_config_path)
        self.device = device
        self.format = str(self.payload.get("format") or "")
        self.is_smoke_checkpoint = self.format == "pi0_lerobot_smoke_v1"
        self.state_dim = int(
            self.payload.get("state_dim")
            or self.payload.get("stateDim")
            or self.train_config.get("stateDim")
            or self.train_config.get("state_dim")
            or 9
        )
        self.action_dim = int(
            self.payload.get("action_dim")
            or self.payload.get("actionDim")
            or self.train_config.get("actionDim")
            or self.train_config.get("action_dim")
            or 8
        )
        self.action_horizon = int(
            self.payload.get("action_horizon") or self.payload.get("chunk_size") or 8
        )
        self.camera_keys = list(
            self.payload.get("image_keys")
            or self.payload.get("camera_keys")
            or self.train_config.get("imageKeys")
            or self.train_config.get("image_keys")
            or DEFAULT_IMAGE_KEYS
        )
        self.low_dim_keys = list(
            self.payload.get("low_dim_keys")
            or self.payload.get("lowDimKeys")
            or self.train_config.get("lowDimKeys")
            or self.train_config.get("low_dim_keys")
            or DEFAULT_LOW_DIM_KEYS
        )
        self.field_mapping = dict(
            self.payload.get("field_mapping") or self.train_config.get("field_mapping") or {}
        )
        self.task_instruction = str(
            task_instruction
            or self.payload.get("task_instruction")
            or self.payload.get("taskInstruction")
            or self.train_config.get("taskInstruction")
            or self.train_config.get("task_instruction")
            or ""
        ).strip()
        self._action_queue: list[np.ndarray] = []
        self._step = 0

        if self.is_smoke_checkpoint and self.action_dim != 8:
            raise ValueError(f"pi0 smoke checkpoint action_dim must be 8, got {self.action_dim}")
        if self.is_smoke_checkpoint and self.state_dim != 9:
            raise ValueError(f"pi0 smoke checkpoint state_dim must be 9, got {self.state_dim}")

    def reset(self) -> None:
        self._action_queue.clear()
        self._step = 0

    def predict(self, obs: dict[str, Any], task_instruction: str | None = None) -> np.ndarray:
        instruction = str(task_instruction or self.task_instruction or "").strip()
        if not instruction:
            raise ValueError("pi0 policy runtime requires taskInstruction")
        action = self._predict_once(obs, task_instruction=instruction)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"pi0 policy output action_dim expected {self.action_dim}, got {action.shape[0]}")
        return action

    def _predict_once(self, obs: dict[str, Any], *, task_instruction: str) -> np.ndarray:
        if self.is_smoke_checkpoint:
            return predict_smoke_pi0_action(
                obs,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                task_instruction=task_instruction,
                step=self._step,
                image_keys=self.camera_keys,
                low_dim_keys=self.low_dim_keys,
                field_mapping=self.field_mapping,
            )
        self._require_camera_obs(obs)
        chunk = infer_pi0_action_chunk(
            checkpoint_path=self.checkpoint_path,
            obs=obs,
            camera_keys=self.camera_keys,
            low_dim_keys=self.low_dim_keys,
            action_dim=self.action_dim,
            device=self.device,
            action_horizon=self.action_horizon,
        )
        if not chunk:
            raise RuntimeError("pi0 policy produced empty action chunk")
        return np.asarray(chunk[0], dtype=np.float32)

    def _require_camera_obs(self, obs: dict[str, Any]) -> None:
        missing = [key for key in self.camera_keys if key not in obs]
        if missing:
            raise KeyError(
                "pi0 模型需要图像观测，但当前评测环境未提供 camera obs："
                + ", ".join(missing)
            )
        if self.low_dim_keys:
            missing_low = [key for key in self.low_dim_keys if key not in obs]
            if missing_low:
                raise KeyError(
                    "pi0 模型需要 low-dim 观测，但当前评测环境未提供："
                    + ", ".join(missing_low)
                )

    def _replenish_actions(self, obs: dict[str, Any]) -> None:
        if self.is_smoke_checkpoint:
            instruction = str(self.task_instruction or "").strip()
            if not instruction:
                raise ValueError("pi0 smoke policy requires taskInstruction")
            for _ in range(max(1, self.action_horizon)):
                self._action_queue.append(
                    predict_smoke_pi0_action(
                        obs,
                        state_dim=self.state_dim,
                        action_dim=self.action_dim,
                        task_instruction=instruction,
                        step=self._step + len(self._action_queue),
                        image_keys=self.camera_keys,
                        low_dim_keys=self.low_dim_keys,
                        field_mapping=self.field_mapping,
                    )
                )
            return
        self._require_camera_obs(obs)
        chunk = infer_pi0_action_chunk(
            checkpoint_path=self.checkpoint_path,
            obs=obs,
            camera_keys=self.camera_keys,
            low_dim_keys=self.low_dim_keys,
            action_dim=self.action_dim,
            device=self.device,
            action_horizon=self.action_horizon,
        )
        for row in chunk:
            self._action_queue.append(np.asarray(row, dtype=np.float32))

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        if not self._action_queue:
            self._replenish_actions(obs)
        if not self._action_queue:
            raise RuntimeError("pi0 policy produced empty action queue")
        self._step += 1
        action = self._action_queue.pop(0)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"pi0 policy output action_dim expected {self.action_dim}, got {action.shape[0]}")
        return action
