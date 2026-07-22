"""Runtime pi0 eval executor resolution for cable_threading run.py (no backend deps)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JOINT_ACTION_MODES = frozenset({"joint_delta", "joint_delta_derived"})
PI0_SMOKE_FORMAT = "pi0_lerobot_smoke_v1"
DEFAULT_IMAGE_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
DEFAULT_LOW_DIM_KEYS = ["robot0_joint_pos", "robot0_gripper_qpos"]


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        payload = _load_json_file(path)
        if payload:
            return payload
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_pi0_joint_space_checkpoint(
    checkpoint_payload: dict[str, Any],
    train_config: dict[str, Any] | None = None,
) -> bool:
    cfg = dict(train_config or {})
    payload = dict(checkpoint_payload or {})
    model_type = str(payload.get("modelType") or payload.get("backend") or cfg.get("modelType") or "").lower()
    if model_type not in {"pi0", "openpi"} and payload.get("format") != PI0_SMOKE_FORMAT:
        return False
    state_dim = _coerce_int(payload.get("state_dim") or payload.get("stateDim") or cfg.get("stateDim"))
    action_dim = _coerce_int(payload.get("action_dim") or payload.get("actionDim") or cfg.get("actionDim"))
    robot = str(payload.get("robot") or cfg.get("robot") or "").strip()
    controller_type = str(
        payload.get("controller_type")
        or payload.get("controllerType")
        or cfg.get("controllerType")
        or ""
    ).strip()
    action_mode = str(
        payload.get("action_mode")
        or payload.get("actionMode")
        or cfg.get("actionMode")
        or cfg.get("action_mode")
        or ""
    ).strip()
    if robot and robot != "Panda":
        return False
    if state_dim != 9 or action_dim != 8:
        return False
    if controller_type != "JOINT_POSITION":
        return False
    if action_mode and action_mode not in JOINT_ACTION_MODES:
        return False
    return True


def resolve_pi0_eval_runtime(
    *,
    policy: str,
    checkpoint_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    eval_executor: str | None = None,
    controller_type: str | None = None,
    action_mode: str | None = None,
    robot: str | None = None,
    task_instruction: str | None = None,
) -> dict[str, Any]:
    if policy != "pi0":
        return {
            "policyType": policy,
            "evalExecutor": "osc_pose",
            "controllerType": "OSC_POSE",
            "actionMode": "legacy",
            "sideChannelMode": "policy",
        }

    checkpoint_payload = _load_checkpoint_payload(checkpoint_path) if checkpoint_path else {}
    train_config = _load_json_file(Path(train_config_path).expanduser()) if train_config_path else {}

    resolved_executor = str(
        eval_executor
        or checkpoint_payload.get("eval_executor")
        or train_config.get("evalExecutor")
        or train_config.get("eval_executor")
        or ""
    ).strip()
    resolved_controller = str(
        controller_type
        or checkpoint_payload.get("controller_type")
        or checkpoint_payload.get("controllerType")
        or train_config.get("controllerType")
        or train_config.get("controller_type")
        or ""
    ).strip()
    resolved_action_mode = str(
        action_mode
        or checkpoint_payload.get("action_mode")
        or checkpoint_payload.get("actionMode")
        or train_config.get("actionMode")
        or train_config.get("action_mode")
        or ""
    ).strip()
    action_dim = _coerce_int(
        checkpoint_payload.get("action_dim")
        or checkpoint_payload.get("actionDim")
        or train_config.get("actionDim")
        or train_config.get("action_dim")
    )
    state_dim = _coerce_int(
        checkpoint_payload.get("state_dim")
        or checkpoint_payload.get("stateDim")
        or train_config.get("stateDim")
        or train_config.get("state_dim")
    )
    resolved_robot = str(
        robot
        or checkpoint_payload.get("robot")
        or train_config.get("robot")
        or ""
    ).strip()
    resolved_task_instruction = str(
        task_instruction
        or checkpoint_payload.get("task_instruction")
        or checkpoint_payload.get("taskInstruction")
        or train_config.get("taskInstruction")
        or train_config.get("task_instruction")
        or ""
    ).strip()

    joint_ready = is_pi0_joint_space_checkpoint(checkpoint_payload, train_config)
    if resolved_controller == "JOINT_POSITION" and resolved_executor == "osc_pose":
        raise ValueError("evalExecutor osc_pose inconsistent with controller_type JOINT_POSITION")

    if int(action_dim or 0) == 7 and (
        resolved_executor == "joint_position"
        or resolved_controller == "JOINT_POSITION"
        or resolved_action_mode in JOINT_ACTION_MODES
    ):
        raise ValueError("7D OSC actions cannot be treated as 8D joint actions for joint_position executor")

    if (
        joint_ready
        or resolved_executor == "joint_position"
        or resolved_controller == "JOINT_POSITION"
        or resolved_action_mode in JOINT_ACTION_MODES
        or (action_dim == 8 and state_dim == 9)
    ):
        if resolved_robot and resolved_robot != "Panda":
            raise ValueError(f"pi0 joint-space eval requires Panda robot, got {resolved_robot}")
        if not resolved_task_instruction:
            raise ValueError("pi0 joint-space eval requires taskInstruction")
        return {
            "policyType": "pi0",
            "modelType": "pi0",
            "policyRuntime": "pi0",
            "evalExecutor": "joint_position",
            "controllerType": "JOINT_POSITION",
            "actionMode": resolved_action_mode or "joint_delta_derived",
            "sideChannelMode": "policy",
            "robot": resolved_robot or "Panda",
            "stateDim": state_dim or 9,
            "actionDim": action_dim or 8,
            "taskInstruction": resolved_task_instruction,
        }

    return {
        "policyType": "pi0",
        "modelType": "pi0",
        "policyRuntime": "pi0",
        "evalExecutor": "osc_pose",
        "controllerType": resolved_controller or "OSC_POSE",
        "actionMode": resolved_action_mode or "legacy",
        "sideChannelMode": "policy",
        "robot": resolved_robot or "UR5e",
        "stateDim": state_dim,
        "actionDim": action_dim or 7,
        "taskInstruction": resolved_task_instruction,
    }
