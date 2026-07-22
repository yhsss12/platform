"""Runtime ACT eval executor resolution for cable_threading run.py (no backend deps)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

JOINT_ACTION_MODES = frozenset({"joint_delta", "joint_delta_derived"})


def _load_train_config(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
    except OSError:
        return {}
    if not isinstance(payload, dict):
        return {}
    train_config = payload.get("train_config")
    if isinstance(train_config, dict):
        return train_config
    config = payload.get("config")
    return config if isinstance(config, dict) else {}


def _load_shape_meta(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
    except OSError:
        return {}
    if not isinstance(payload, dict):
        return {}
    shape_meta = payload.get("shape_meta")
    return dict(shape_meta) if isinstance(shape_meta, dict) else {}


def resolve_act_eval_runtime(
    *,
    policy: str,
    checkpoint_path: str | Path | None = None,
    eval_executor: str | None = None,
    controller_type: str | None = None,
    action_mode: str | None = None,
) -> dict[str, str]:
    if policy != "act":
        return {
            "policyType": policy,
            "evalExecutor": "osc_pose",
            "controllerType": "OSC_POSE",
            "actionMode": "legacy",
            "sideChannelMode": "policy",
        }

    shape_meta = _load_shape_meta(checkpoint_path) if checkpoint_path else {}
    train_config = _load_train_config(checkpoint_path) if checkpoint_path else {}

    resolved_executor = str(
        eval_executor
        or shape_meta.get("eval_executor")
        or train_config.get("eval_executor")
        or ""
    ).strip()
    resolved_controller = str(
        controller_type
        or shape_meta.get("controller_type")
        or train_config.get("controller_type")
        or ""
    ).strip()
    resolved_action_mode = str(
        action_mode
        or shape_meta.get("trained_action_mode")
        or shape_meta.get("action_mode")
        or train_config.get("trained_action_mode")
        or train_config.get("action_mode")
        or ""
    ).strip()

    action_dim = shape_meta.get("action_dim") or train_config.get("action_dim")
    low_dim_keys = shape_meta.get("low_dim_keys") or train_config.get("low_dim_keys") or []

    if resolved_controller == "JOINT_POSITION" and resolved_executor == "osc_pose":
        raise ValueError("evalExecutor osc_pose inconsistent with controller_type JOINT_POSITION")

    if int(action_dim or 0) == 7 and (
        resolved_executor == "joint_position"
        or resolved_controller == "JOINT_POSITION"
        or resolved_action_mode in JOINT_ACTION_MODES
    ):
        raise ValueError("7D OSC actions cannot be treated as 8D joint actions for joint_position executor")

    if (
        resolved_executor == "joint_position"
        or resolved_controller == "JOINT_POSITION"
        or resolved_action_mode in JOINT_ACTION_MODES
        or (
            int(action_dim or 0) == 8
            and list(low_dim_keys) == ["robot0_joint_pos", "robot0_gripper_qpos"]
        )
    ):
        return {
            "policyType": "act",
            "evalExecutor": "joint_position",
            "controllerType": "JOINT_POSITION",
            "actionMode": resolved_action_mode or "joint_delta_derived",
            "sideChannelMode": "policy",
        }

    return {
        "policyType": "act",
        "evalExecutor": "osc_pose",
        "controllerType": resolved_controller or "OSC_POSE",
        "actionMode": resolved_action_mode or "osc_pose_delta_eef",
        "sideChannelMode": "policy",
    }
