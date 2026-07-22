"""Runtime DP eval executor resolution for cable_threading run.py (no backend deps)."""
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
    cfg = payload.get("train_config")
    return cfg if isinstance(cfg, dict) else {}


def resolve_dp_eval_runtime(
    *,
    policy: str,
    checkpoint_path: str | Path | None = None,
    eval_executor: str | None = None,
    controller_type: str | None = None,
    action_mode: str | None = None,
) -> dict[str, str]:
    if policy in {"scripted", "random"}:
        return {
            "policyType": "expert",
            "evalExecutor": "osc_pose",
            "controllerType": "OSC_POSE",
            "actionMode": "expert",
            "sideChannelMode": "policy",
        }
    if policy != "diffusion_policy":
        return {
            "policyType": policy,
            "evalExecutor": "osc_pose",
            "controllerType": "OSC_POSE",
            "actionMode": "legacy",
            "sideChannelMode": "policy",
        }

    if eval_executor:
        resolved_executor = eval_executor
    else:
        train_config = _load_train_config(checkpoint_path) if checkpoint_path else {}
        resolved_executor = str(train_config.get("eval_executor") or "").strip()
        action_mode = action_mode or str(
            train_config.get("trained_action_mode") or train_config.get("action_mode") or ""
        ).strip()
        controller_type = controller_type or str(train_config.get("controller_type") or "").strip()
        if not resolved_executor:
            if (
                controller_type == "JOINT_POSITION"
                or action_mode in JOINT_ACTION_MODES
            ):
                resolved_executor = "joint_position"

    if resolved_executor == "joint_position" or action_mode in JOINT_ACTION_MODES:
        return {
            "policyType": "diffusion_policy",
            "evalExecutor": "joint_position",
            "controllerType": "JOINT_POSITION",
            "actionMode": action_mode or "joint_delta",
            "sideChannelMode": "policy",
        }

    return {
        "policyType": "diffusion_policy",
        "evalExecutor": "osc_pose",
        "controllerType": controller_type or "OSC_POSE",
        "actionMode": action_mode or "osc_pose_delta_eef",
        "sideChannelMode": "policy",
    }
