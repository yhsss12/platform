"""V2-B5.5：pre-lift reclose + slow vertical lift refiner。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from lift_v2b54_refiner import (
    LIFT_V2B54_SEARCH_SPACE,
    LiftV2B54Params,
    apply_lift_v2b54_params_to_eef_waypoints,
    build_lift_v2b54_waypoints_from_hdf5,
    lift_v2b54_from_prior,
    lift_v2b54_params_from_dict,
)
from refined_waypoint_builder import load_eef_pose_sequence
from trajectory_parameterization import TrajectoryProxy, load_trajectory_proxy


@dataclass
class LiftV2B55Params(LiftV2B54Params):
    """B5.5 pre-lift reclose + slow vertical lift 参数。"""

    second_reclose_steps: int = 0
    second_reclose_strength: float = 0.0
    slow_vertical_lift_steps: int = 0
    vertical_only_lift_steps: int = 0
    max_slip_gate: float = 0.05
    pre_lift_hold_extra_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LIFT_V2B55_EXTRA_SPACE: dict[str, list[float | int]] = {
    "second_reclose_steps": [0, 4, 8, 12, 16, 20],
    "second_reclose_strength": [0.0, 0.2, 0.35, 0.5, 0.65, 0.8],
    "slow_vertical_lift_steps": [0, 8, 12, 16, 24, 32],
    "vertical_only_lift_steps": [0, 5, 10, 15, 20, 30],
    "max_slip_gate": [0.02, 0.03, 0.04, 0.05, 0.06],
    "pre_lift_hold_extra_steps": [0, 5, 10, 15, 20, 30],
    "pre_close_hold_steps": [10, 15, 20, 30, 40, 50],
    "pre_lift_reclose_steps": [5, 8, 12, 16, 20, 24],
    "pre_lift_reclose_strength": [0.3, 0.45, 0.6, 0.75, 0.85],
    "post_micro_lift_squeeze_steps": [3, 5, 8, 12, 16],
    "post_micro_lift_squeeze_strength": [0.25, 0.4, 0.55, 0.7],
    "lift_hold_before_transport_steps": [5, 10, 15, 20, 30, 40],
    "transport_phase_delay_steps": [10, 15, 20, 30, 40, 50],
    "gripper_close_depth_bonus": [0.1, 0.15, 0.2, 0.25, 0.3],
    "lateral_preload_scale": [0.0, 0.1, 0.2, 0.35, 0.5],
    "lift_speed_scale": [0.05, 0.08, 0.1, 0.12, 0.15],
}


def _merge_b55_search_space() -> dict[str, list[float | int | str]]:
    space = {k: list(v) for k, v in LIFT_V2B54_SEARCH_SPACE.items()}
    for k, v in LIFT_V2B55_EXTRA_SPACE.items():
        space[k] = list(v)
    space["template_mask"] = [
        "squeeze_close,contact_settle,micro_lift,pre_lift_reclose,reclose_after_micro_lift,slow_lift",
        "contact_settle,micro_lift,pre_close_hold,pre_lift_reclose,reclose_after_micro_lift,slow_lift",
        "lower_approach,squeeze_close,contact_settle,two_stage_lift,pre_lift_reclose,slow_lift,lift_hold_before_transport",
    ]
    return space


LIFT_V2B55_SEARCH_SPACE = _merge_b55_search_space()


def lift_v2b55_from_prior(params: dict[str, Any]) -> LiftV2B55Params:
    base = lift_v2b54_from_prior(params)
    raw = base.to_dict()
    valid = {f.name for f in fields(LiftV2B55Params)}
    return LiftV2B55Params(**{k: v for k, v in raw.items() if k in valid})


def lift_v2b55_params_from_dict(raw: dict[str, Any]) -> LiftV2B55Params:
    valid = {f.name for f in fields(LiftV2B55Params)}
    return LiftV2B55Params(**{k: v for k, v in raw.items() if k in valid})


def apply_lift_v2b55_params_to_eef_waypoints(
    proxy: TrajectoryProxy,
    eef_pose: np.ndarray,
    params: LiftV2B55Params,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    b54 = LiftV2B54Params(**{k: getattr(params, k) for k in LiftV2B54Params.__dataclass_fields__})
    b54.pre_close_hold_steps = max(int(b54.pre_close_hold_steps), int(params.pre_lift_hold_extra_steps))
    b54.pre_lift_reclose_steps = max(int(b54.pre_lift_reclose_steps), 8)
    b54.pre_lift_reclose_strength = max(float(b54.pre_lift_reclose_strength), 0.4)
    b54.lift_speed_scale = min(float(b54.lift_speed_scale), 0.12)
    b54.lateral_preload_scale = min(float(b54.lateral_preload_scale), 0.35)
    b54.transport_phase_delay_steps = max(int(b54.transport_phase_delay_steps), 15)
    b54.lift_hold_before_transport_steps = max(int(b54.lift_hold_before_transport_steps), 10)
    b54.reclose_strength = max(float(b54.reclose_strength), float(params.second_reclose_strength))

    refined, gripper, phases = apply_lift_v2b54_params_to_eef_waypoints(proxy, eef_pose, b54)
    grasp_idx = phases["grasp_index"]
    lift_begin = phases["lift_begin"]
    stage1_end = phases["stage1_end"]
    stage2_end = phases["stage2_end"]
    length = proxy.length

    second_start = max(stage1_end, lift_begin)
    second_end = min(length - 1, second_start + int(params.second_reclose_steps))
    if int(params.second_reclose_steps) > 0:
        for step in range(second_start, second_end + 1):
            gripper[step] = min(gripper[step], -0.97 - float(params.second_reclose_strength) * 0.12)

    vert_end = min(length - 1, lift_begin + int(params.vertical_only_lift_steps))
    if vert_end > lift_begin:
        anchor = refined[lift_begin, :3, 3].copy()
        for step in range(lift_begin, vert_end + 1):
            pos = refined[step, :3, 3].copy()
            pos[0], pos[1] = anchor[0], anchor[1]
            refined[step, :3, 3] = pos

    slow_end = min(length - 1, vert_end + int(params.slow_vertical_lift_steps))
    phases["second_reclose_end"] = second_end
    phases["vertical_only_lift_end"] = vert_end
    phases["slow_vertical_lift_end"] = slow_end
    phases["max_slip_gate"] = float(params.max_slip_gate)
    return refined, gripper, phases


def build_lift_v2b55_waypoints_from_hdf5(
    hdf5_path: str, demo_key: str, label: str, params: LiftV2B55Params
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    original = load_eef_pose_sequence(hdf5_path, demo_key)
    refined, gripper, phases = apply_lift_v2b55_params_to_eef_waypoints(proxy, original, params)
    return proxy, original, refined, gripper, phases
