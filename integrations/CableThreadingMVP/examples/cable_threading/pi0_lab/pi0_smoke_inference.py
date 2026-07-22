"""Smoke-compatible pi0 joint-space inference (no openpi dependency)."""
from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_IMAGE_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
DEFAULT_LOW_DIM_KEYS = ["robot0_joint_pos", "robot0_gripper_qpos"]


def build_pi0_state_vector(
    obs: dict[str, Any],
    *,
    low_dim_keys: list[str] | None = None,
    state_dim: int = 9,
) -> np.ndarray:
    keys = list(low_dim_keys or DEFAULT_LOW_DIM_KEYS)
    parts: list[np.ndarray] = []
    for key in keys:
        if key not in obs:
            raise KeyError(f"pi0 state construction missing observation key: {key}")
        parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    state = np.concatenate(parts, axis=0)
    if int(state.shape[0]) != int(state_dim):
        raise ValueError(f"pi0 state_dim expected {state_dim}, got {state.shape[0]}")
    return state.astype(np.float32)


def build_pi0_openpi_observation(
    obs: dict[str, Any],
    *,
    image_keys: list[str] | None = None,
    low_dim_keys: list[str] | None = None,
    state_dim: int = 9,
    task_instruction: str,
    field_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_keys = list(image_keys or DEFAULT_IMAGE_KEYS)
    for key in image_keys:
        if key not in obs:
            raise KeyError(f"pi0 observation missing image key: {key}")
    state = build_pi0_state_vector(obs, low_dim_keys=low_dim_keys, state_dim=state_dim)
    mapping = dict(field_mapping or {})
    openpi_images = dict(mapping.get("openpi_image_keys") or {})
    payload: dict[str, Any] = {
        "task_instruction": task_instruction,
        "observation.state": state,
    }
    if openpi_images:
        for platform_key, openpi_key in openpi_images.items():
            payload[str(openpi_key)] = np.asarray(obs[platform_key])
    else:
        payload["image.base_0_rgb"] = np.asarray(obs[image_keys[0]])
        payload["image.left_wrist_0_rgb"] = np.asarray(obs[image_keys[1] if len(image_keys) > 1 else image_keys[0]])
        if bool(mapping.get("third_wrist_padding", True)):
            payload["image.right_wrist_0_rgb"] = np.zeros_like(payload["image.left_wrist_0_rgb"])
            payload["image.right_wrist_0_mask"] = np.zeros(payload["image.left_wrist_0_rgb"].shape[:2], dtype=bool)
    return payload


def predict_smoke_pi0_action(
    obs: dict[str, Any],
    *,
    state_dim: int,
    action_dim: int,
    task_instruction: str,
    step: int,
    image_keys: list[str] | None = None,
    low_dim_keys: list[str] | None = None,
    field_mapping: dict[str, Any] | None = None,
) -> np.ndarray:
    if not str(task_instruction or "").strip():
        raise ValueError("pi0 smoke inference requires taskInstruction")
    _ = build_pi0_openpi_observation(
        obs,
        image_keys=image_keys,
        low_dim_keys=low_dim_keys,
        state_dim=state_dim,
        task_instruction=task_instruction,
        field_mapping=field_mapping,
    )
    rng = np.random.default_rng(abs(hash(task_instruction)) % (2**32) + step)
    action = rng.normal(loc=0.0, scale=0.02, size=int(action_dim)).astype(np.float32)
    action[-1] = np.clip(action[-1], -0.05, 0.05)
    if action.shape[0] != int(action_dim):
        raise ValueError(f"pi0 action_dim expected {action_dim}, got {action.shape[0]}")
    return action
