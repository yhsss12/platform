"""V2-B5.2：asymmetric grasp correction + B5.1 lift templates。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from grasp_waypoint_builder import GraspSearchParams, apply_grasp_params_to_eef_waypoints
from lift_v2b51_refiner import LIFT_V2B51_TEMPLATES, LiftV2B51Params
from refined_waypoint_builder import load_eef_pose_sequence
from trajectory_parameterization import TrajectoryProxy, load_trajectory_proxy


@dataclass
class LiftV2B52Params(LiftV2B51Params):
    """B5.1 参数 + asymmetric grasp correction。"""

    grasp_lateral_bias_x: float = 0.0
    grasp_lateral_bias_y: float = 0.0
    right_finger_bias: float = 0.0
    approach_yaw_bias: float = 0.0
    gripper_asym_close_offset: float = 0.0
    squeeze_close_strength: float = 0.0
    extra_close_steps: int = 0
    post_close_hold_steps: int = 0
    reclose_strength: float = 0.0
    second_micro_lift_height: float = 0.0
    second_micro_lift_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LIFT_V2B52_EXTRA_SPACE: dict[str, list[float | int]] = {
    "grasp_lateral_bias_x": [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04],
    "grasp_lateral_bias_y": [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04],
    "right_finger_bias": [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
    "approach_yaw_bias": [-0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.08],
    "gripper_asym_close_offset": [-0.25, -0.15, -0.08, 0.0, 0.08, 0.15],
    "squeeze_close_strength": [0.0, 0.2, 0.35, 0.5, 0.65, 0.8],
    "extra_close_steps": [0, 3, 5, 8, 12, 15],
    "post_close_hold_steps": [0, 5, 10, 15, 20, 30],
    "reclose_strength": [0.0, 0.15, 0.3, 0.45, 0.6],
    "second_micro_lift_height": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
    "second_micro_lift_steps": [0, 8, 12, 16, 20, 30],
}


def _merge_search_space() -> dict[str, list[float | int | str]]:
    from lift_v2b51_refiner import LIFT_V2B51_SEARCH_SPACE

    space = {k: list(v) for k, v in LIFT_V2B51_SEARCH_SPACE.items()}
    for k, v in LIFT_V2B52_EXTRA_SPACE.items():
        space[k] = list(v)
    return space


LIFT_V2B52_SEARCH_SPACE = _merge_search_space()


def lift_v2b52_from_b51(params: dict[str, Any] | LiftV2B51Params) -> LiftV2B52Params:
    raw = params.to_dict() if hasattr(params, "to_dict") else dict(params)
    valid = {f.name for f in fields(LiftV2B52Params)}
    merged = {k: v for k, v in raw.items() if k in valid}
    if "squeeze_close_strength" not in merged and "squeeze_close_gain" in raw:
        merged["squeeze_close_strength"] = float(raw["squeeze_close_gain"])
    return LiftV2B52Params(**merged)


def lift_v2b52_params_from_dict(raw: dict[str, Any]) -> LiftV2B52Params:
    valid = {f.name for f in fields(LiftV2B52Params)}
    return LiftV2B52Params(**{k: v for k, v in raw.items() if k in valid})


def apply_lift_v2b52_params_to_eef_waypoints(
    proxy: TrajectoryProxy,
    eef_pose: np.ndarray,
    params: LiftV2B52Params,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    from lift_v2b51_refiner import apply_lift_v2b51_params_to_eef_waypoints

    b51 = LiftV2B51Params(**{k: getattr(params, k) for k in LiftV2B51Params.__dataclass_fields__})
    b51.grasp_xy_offset_x += float(params.grasp_lateral_bias_x)
    b51.grasp_xy_offset_y += float(params.grasp_lateral_bias_y) + float(params.right_finger_bias)
    b51.lateral_correction_x += float(params.grasp_lateral_bias_x) * 0.5
    b51.lateral_correction_y += float(params.grasp_lateral_bias_y) * 0.5
    b51.squeeze_close_gain = max(float(b51.squeeze_close_gain), float(params.squeeze_close_strength))
    b51.contact_settle_steps = max(int(b51.contact_settle_steps), int(params.extra_close_steps) + 10)
    b51.post_grasp_settle_steps = max(int(b51.post_grasp_settle_steps), int(params.post_close_hold_steps))
    if float(params.second_micro_lift_height) > 0:
        b51.micro_lift_height_stage2 = float(params.second_micro_lift_height)
    if int(params.second_micro_lift_steps) > 0:
        b51.micro_lift_steps_stage2 = int(params.second_micro_lift_steps)
    b51.reclose_after_micro_lift_steps = max(
        int(b51.reclose_after_micro_lift_steps), int(round(float(params.reclose_strength) * 12))
    )

    refined, gripper, phases = apply_lift_v2b51_params_to_eef_waypoints(proxy, eef_pose, b51)
    grasp_idx = phases["grasp_index"]
    length = proxy.length

    yaw = float(params.approach_yaw_bias)
    if abs(yaw) > 1e-6:
        approach_start = max(0, grasp_idx - 15)
        for step in range(approach_start, grasp_idx + 1):
            w = (step - approach_start) / max(1, grasp_idx - approach_start)
            rot = refined[step, :3, :3].copy()
            c, s = math.cos(yaw * w), math.sin(yaw * w)
            yaw_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
            refined[step, :3, :3] = yaw_mat @ rot

    asym = float(params.gripper_asym_close_offset)
    if asym != 0.0:
        close_start = max(0, grasp_idx - 2)
        close_end = min(length - 1, grasp_idx + int(params.contact_settle_steps) + int(params.extra_close_steps))
        for step in range(close_start, close_end + 1):
            gripper[step] = min(gripper[step], -0.85 - abs(asym))

    extra_end = min(length - 1, grasp_idx + int(params.extra_close_steps))
    if int(params.extra_close_steps) > 0:
        for step in range(grasp_idx, extra_end + 1):
            gripper[step] = min(gripper[step], -0.92 - float(params.squeeze_close_strength) * 0.1)

    hold_end = min(length - 1, grasp_idx + int(params.post_close_hold_steps))
    if hold_end > grasp_idx:
        hold = refined[grasp_idx, :3, 3].copy()
        rot = refined[grasp_idx, :3, :3].copy()
        for step in range(grasp_idx + 1, hold_end + 1):
            refined[step, :3, 3] = hold
            refined[step, :3, :3] = rot

    phases["contact_window_end"] = min(length - 1, grasp_idx + int(params.contact_settle_steps) + int(params.extra_close_steps))
    return refined, gripper, phases


def build_lift_v2b52_waypoints_from_hdf5(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: LiftV2B52Params,
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
    refined, gripper, phases = apply_lift_v2b52_params_to_eef_waypoints(proxy, original_eef, params)
    return proxy, original_eef, refined, gripper, phases
