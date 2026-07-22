"""V2-B5.4：lift-preserving transport refiner（B5.2 + lift-hold-transport 模板）。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from lift_v2b52_refiner import (
    LIFT_V2B52_SEARCH_SPACE,
    LiftV2B52Params,
    apply_lift_v2b52_params_to_eef_waypoints,
    build_lift_v2b52_waypoints_from_hdf5,
    lift_v2b52_from_b51,
    lift_v2b52_params_from_dict,
)
from trajectory_parameterization import TrajectoryProxy, load_trajectory_proxy
from refined_waypoint_builder import load_eef_pose_sequence

LIFT_V2B54_TEMPLATES = (
    "pre_close_hold",
    "pre_lift_reclose",
    "post_micro_lift_squeeze",
    "lift_hold_before_transport",
    "lift_before_transport_gate",
)


@dataclass
class LiftV2B54Params(LiftV2B52Params):
    """B5.4 lift-preserving transport 参数。"""

    pre_close_hold_steps: int = 0
    pre_lift_reclose_steps: int = 0
    pre_lift_reclose_strength: float = 0.0
    post_micro_lift_squeeze_steps: int = 0
    post_micro_lift_squeeze_strength: float = 0.0
    lift_hold_before_transport_steps: int = 0
    gripper_close_depth_bonus: float = 0.0
    lateral_preload_scale: float = 1.0
    weak_lift_before_transport_m: float = 0.002
    transport_phase_delay_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def enabled_b54_templates(self) -> set[str]:
        return set(LIFT_V2B54_TEMPLATES)


LIFT_V2B54_EXTRA_SPACE: dict[str, list[float | int]] = {
    "pre_close_hold_steps": [0, 5, 10, 15, 20, 30, 40],
    "pre_lift_reclose_steps": [0, 3, 5, 8, 12, 16],
    "pre_lift_reclose_strength": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
    "post_micro_lift_squeeze_steps": [0, 3, 5, 8, 12, 16],
    "post_micro_lift_squeeze_strength": [0.0, 0.2, 0.35, 0.5, 0.65],
    "lift_hold_before_transport_steps": [0, 5, 10, 15, 20, 30],
    "gripper_close_depth_bonus": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    "lateral_preload_scale": [0.0, 0.25, 0.5, 0.75, 1.0],
    "weak_lift_before_transport_m": [0.0015, 0.002, 0.0025, 0.003],
    "transport_phase_delay_steps": [0, 5, 10, 15, 20, 30],
}


def _merge_b54_search_space() -> dict[str, list[float | int | str]]:
    space = {k: list(v) for k, v in LIFT_V2B52_SEARCH_SPACE.items()}
    for k, v in LIFT_V2B54_EXTRA_SPACE.items():
        space[k] = list(v)
    space["template_mask"] = list(
        {
            "all",
            "lower_approach,squeeze_close,contact_settle,micro_lift,reclose_after_micro_lift,slow_lift",
            "contact_settle,micro_lift,reclose_after_micro_lift,slow_lift,lift_hold_before_transport",
            "squeeze_close,contact_settle,micro_lift,pre_lift_reclose,slow_lift,lift_hold_before_transport",
            "lower_approach,squeeze_close,contact_settle,two_stage_lift,pre_close_hold,pre_lift_reclose,slow_lift",
        }
    )
    return space


LIFT_V2B54_SEARCH_SPACE = _merge_b54_search_space()


def lift_v2b54_from_prior(params: dict[str, Any] | LiftV2B52Params) -> LiftV2B54Params:
    base = lift_v2b52_from_b51(params) if not isinstance(params, LiftV2B52Params) else params
    raw = base.to_dict() if hasattr(base, "to_dict") else dict(params)
    valid = {f.name for f in fields(LiftV2B54Params)}
    merged = {k: v for k, v in raw.items() if k in valid}
    return LiftV2B54Params(**merged)


def lift_v2b54_params_from_dict(raw: dict[str, Any]) -> LiftV2B54Params:
    valid = {f.name for f in fields(LiftV2B54Params)}
    return LiftV2B54Params(**{k: v for k, v in raw.items() if k in valid})


def apply_lift_v2b54_params_to_eef_waypoints(
    proxy: TrajectoryProxy,
    eef_pose: np.ndarray,
    params: LiftV2B54Params,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    b52 = LiftV2B52Params(**{k: getattr(params, k) for k in LiftV2B52Params.__dataclass_fields__})
    b52.post_close_hold_steps = max(int(b52.post_close_hold_steps), int(params.pre_close_hold_steps))
    b52.squeeze_close_strength = max(float(b52.squeeze_close_strength), float(params.gripper_close_depth_bonus))
    b52.reclose_strength = max(float(b52.reclose_strength), float(params.pre_lift_reclose_strength))
    b52.reclose_after_micro_lift_steps = max(
        int(b52.reclose_after_micro_lift_steps), int(params.post_micro_lift_squeeze_steps)
    )
    b52.lift_speed_scale = min(float(b52.lift_speed_scale), 0.25)
    b52.lateral_correction_x *= float(params.lateral_preload_scale)
    b52.lateral_correction_y *= float(params.lateral_preload_scale)

    refined, gripper, phases = apply_lift_v2b52_params_to_eef_waypoints(proxy, eef_pose, b52)
    grasp_idx = phases["grasp_index"]
    lift_begin = phases["lift_begin"]
    stage1_end = phases["stage1_end"]
    stage2_end = phases["stage2_end"]
    length = proxy.length

    hold_end = min(length - 1, grasp_idx + int(params.pre_close_hold_steps))
    if hold_end > grasp_idx:
        hold = refined[grasp_idx, :3, 3].copy()
        rot = refined[grasp_idx, :3, :3].copy()
        for step in range(grasp_idx + 1, hold_end + 1):
            refined[step, :3, 3] = hold
            refined[step, :3, :3] = rot
            gripper[step] = min(gripper[step], -0.94 - float(params.gripper_close_depth_bonus))

    reclose_start = max(grasp_idx, lift_begin - int(params.pre_lift_reclose_steps))
    reclose_end = max(reclose_start, lift_begin - 1)
    if int(params.pre_lift_reclose_steps) > 0 and reclose_end >= reclose_start:
        for step in range(reclose_start, reclose_end + 1):
            gripper[step] = min(gripper[step], -0.95 - float(params.pre_lift_reclose_strength) * 0.12)

    squeeze_end = min(length - 1, stage1_end + int(params.post_micro_lift_squeeze_steps))
    if int(params.post_micro_lift_squeeze_steps) > 0:
        for step in range(stage1_end, squeeze_end + 1):
            gripper[step] = min(gripper[step], -0.96 - float(params.post_micro_lift_squeeze_strength) * 0.1)

    transport_start = min(length - 1, stage2_end + int(params.transport_phase_delay_steps))
    lift_hold_end = min(length - 1, transport_start + int(params.lift_hold_before_transport_steps))
    if lift_hold_end > transport_start:
        hold_pos = refined[transport_start, :3, 3].copy()
        hold_rot = refined[transport_start, :3, :3].copy()
        for step in range(transport_start, lift_hold_end + 1):
            refined[step, :3, 3] = hold_pos
            refined[step, :3, :3] = hold_rot
            gripper[step] = min(gripper[step], -0.99)

    phases["pre_lift_reclose_start"] = reclose_start
    phases["pre_lift_reclose_end"] = reclose_end
    phases["post_micro_squeeze_end"] = squeeze_end
    phases["transport_start"] = transport_start
    phases["lift_hold_transport_end"] = lift_hold_end
    phases["weak_lift_gate_m"] = float(params.weak_lift_before_transport_m)
    return refined, gripper, phases


def build_lift_v2b54_waypoints_from_hdf5(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: LiftV2B54Params,
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
    refined, gripper, phases = apply_lift_v2b54_params_to_eef_waypoints(proxy, original_eef, params)
    return proxy, original_eef, refined, gripper, phases
