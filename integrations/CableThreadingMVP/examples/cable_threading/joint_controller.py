"""JOINT_POSITION controller helpers for cable_threading DP eval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CABLE_MVP_ROOT = Path(__file__).resolve().parents[2]
JOINT_PART_CONFIG = (
    _CABLE_MVP_ROOT / "robosuite" / "controllers" / "config" / "default" / "parts" / "joint_position.json"
)


def build_joint_position_controller_config(*, robot: str = "Panda") -> dict[str, Any]:
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(robot=robot)
    joint_part = json.loads(JOINT_PART_CONFIG.read_text(encoding="utf-8"))
    joint_part["input_type"] = "delta"
    body_parts = cfg.get("body_parts") or {}
    arm_key = "right" if "right" in body_parts else "arms"
    if arm_key == "arms":
        cfg["body_parts"]["arms"]["right"] = {**joint_part, "gripper": {"type": "GRIP"}}
    else:
        cfg["body_parts"]["right"] = {**joint_part, "gripper": {"type": "GRIP"}}
    return cfg


def make_joint_position_env(
    *,
    robot: str = "Panda",
    cable_model: str = "composite_cable",
    grasp_mode: str = "attachment",
    difficulty: str = "easy",
    horizon: int = 600,
    seed: int | None = None,
    use_camera_obs: bool = True,
    has_offscreen_renderer: bool | None = None,
    camera_names: list[str] | None = None,
    **kwargs,
):
    from examples.cable_threading.utils import make_env

    if has_offscreen_renderer is None:
        has_offscreen_renderer = use_camera_obs
    if camera_names is None:
        camera_names = ["agentview", "robot0_eye_in_hand"] if use_camera_obs else None
    controller_configs = build_joint_position_controller_config(robot=robot)
    env_kwargs = dict(kwargs)
    if camera_names is not None:
        env_kwargs["camera_names"] = camera_names
    return make_env(
        robot=robot,
        cable_model=cable_model,
        grasp_mode=grasp_mode,
        difficulty=difficulty,
        horizon=horizon,
        seed=seed,
        use_camera_obs=use_camera_obs,
        has_offscreen_renderer=has_offscreen_renderer,
        controller_configs=controller_configs,
        **env_kwargs,
    )
