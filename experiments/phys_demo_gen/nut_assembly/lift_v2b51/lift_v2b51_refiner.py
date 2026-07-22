"""V2-B5.1：contact-aware lift refiner（8 种显式模板）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from grasp_waypoint_builder import GraspSearchParams, apply_grasp_params_to_eef_waypoints
from refined_waypoint_builder import load_eef_pose_sequence
from trajectory_parameterization import TrajectoryProxy, load_trajectory_proxy

LIFT_V2B51_TEMPLATES = (
    "lower_approach",
    "squeeze_close",
    "contact_settle",
    "micro_lift",
    "reclose_after_micro_lift",
    "slow_lift",
    "two_stage_lift",
    "lateral_correction",
)


@dataclass
class LiftV2B51Params:
    """V2-B5.1 contact-aware lift refiner 参数。"""

    grasp_xy_offset_x: float = 0.0
    grasp_xy_offset_y: float = 0.0
    lateral_correction_x: float = 0.0
    lateral_correction_y: float = 0.0
    pre_grasp_height: float = 0.05
    approach_height: float = 0.02
    lower_approach_delta: float = 0.0
    gripper_close_shift: float = 0.0
    regrasp_shift: float = 0.0
    gripper_extra_close: float = 0.0
    squeeze_close_gain: float = 0.0
    contact_settle_steps: int = 25
    post_grasp_settle_steps: int = 10
    micro_lift_height_stage1: float = 0.03
    micro_lift_height_stage2: float = 0.06
    micro_lift_steps_stage1: int = 15
    micro_lift_steps_stage2: int = 25
    reclose_after_micro_lift_steps: int = 0
    lift_pause_between_stages: int = 5
    lift_speed_scale: float = 0.35
    lift_direction_bias_z: float = 0.0
    enable_two_stage_lift: float = 1.0
    micro_lift_check_threshold: float = 0.005
    nut_follow_threshold: float = 0.05
    gripper_hold_steps: int = 30
    post_extension_steps: int = 0
    extension_lift_height: float = 0.05
    template_mask: str = "all"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def enabled_templates(self) -> set[str]:
        if self.template_mask in ("all", "", "full"):
            return set(LIFT_V2B51_TEMPLATES)
        return {t.strip() for t in self.template_mask.split(",") if t.strip()}


LIFT_V2B51_SEARCH_SPACE: dict[str, list[float | int | str]] = {
    "grasp_xy_offset_x": [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08],
    "grasp_xy_offset_y": [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08],
    "lateral_correction_x": [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04],
    "lateral_correction_y": [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04],
    "pre_grasp_height": [0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
    "approach_height": [0.005, 0.01, 0.02, 0.03, 0.04],
    "lower_approach_delta": [0.0, 0.01, 0.02, 0.03, 0.04],
    "gripper_close_shift": [-25, -20, -15, -10, -5, 0],
    "regrasp_shift": [-15, -10, -5, 0, 5, 10],
    "gripper_extra_close": [-0.45, -0.35, -0.25, -0.15, -0.05, 0.0],
    "squeeze_close_gain": [0.0, 0.15, 0.30, 0.45, 0.60],
    "contact_settle_steps": [10, 20, 30, 40, 50, 60],
    "post_grasp_settle_steps": [5, 10, 15, 20, 30],
    "micro_lift_height_stage1": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    "micro_lift_steps_stage1": [8, 12, 16, 20, 30, 40],
    "micro_lift_height_stage2": [0.03, 0.05, 0.07, 0.09, 0.12],
    "micro_lift_steps_stage2": [12, 20, 30, 40, 50],
    "reclose_after_micro_lift_steps": [0, 3, 5, 8, 12],
    "lift_pause_between_stages": [0, 3, 5, 8, 12, 20],
    "lift_speed_scale": [0.10, 0.15, 0.20, 0.25, 0.35, 0.50],
    "lift_direction_bias_z": [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
    "enable_two_stage_lift": [0.0, 1.0],
    "micro_lift_check_threshold": [0.003, 0.005, 0.008, 0.010, 0.015],
    "nut_follow_threshold": [0.025, 0.03, 0.04, 0.05, 0.06],
    "gripper_hold_steps": [20, 30, 40, 50, 60],
    "post_extension_steps": [0, 20, 40, 60, 80],
    "extension_lift_height": [0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
    "template_mask": [
        "all",
        "lower_approach,squeeze_close,contact_settle,micro_lift,slow_lift",
        "contact_settle,micro_lift,reclose_after_micro_lift,slow_lift",
        "lower_approach,squeeze_close,contact_settle,two_stage_lift,slow_lift,lateral_correction",
        "squeeze_close,contact_settle,micro_lift,reclose_after_micro_lift,two_stage_lift,slow_lift",
    ],
}

CONTACT_AWARE_SEED_PARAMS = LiftV2B51Params(
    grasp_xy_offset_x=0.02,
    grasp_xy_offset_y=-0.04,
    lateral_correction_x=0.02,
    lateral_correction_y=-0.02,
    pre_grasp_height=0.02,
    approach_height=0.005,
    lower_approach_delta=0.03,
    gripper_close_shift=-20,
    regrasp_shift=-5,
    gripper_extra_close=-0.35,
    squeeze_close_gain=0.45,
    contact_settle_steps=40,
    post_grasp_settle_steps=20,
    micro_lift_height_stage1=0.015,
    micro_lift_height_stage2=0.05,
    micro_lift_steps_stage1=20,
    micro_lift_steps_stage2=35,
    reclose_after_micro_lift_steps=8,
    lift_pause_between_stages=10,
    lift_speed_scale=0.15,
    lift_direction_bias_z=0.01,
    enable_two_stage_lift=1.0,
    micro_lift_check_threshold=0.005,
    nut_follow_threshold=0.03,
    gripper_hold_steps=40,
    post_extension_steps=60,
    extension_lift_height=0.10,
    template_mask="all",
)


def apply_lift_v2b51_params_to_eef_waypoints(
    proxy: TrajectoryProxy,
    eef_pose: np.ndarray,
    params: LiftV2B51Params,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    enabled = params.enabled_templates()
    grasp_idx = proxy.phases.grasp_index
    length = proxy.length

    pre_height = float(params.pre_grasp_height)
    approach_height = float(params.approach_height)
    if "lower_approach" in enabled:
        pre_height = max(0.005, pre_height - float(params.lower_approach_delta))
        approach_height = max(0.002, approach_height - float(params.lower_approach_delta) * 0.5)

    close_shift = float(params.gripper_close_shift + params.regrasp_shift)
    extra_close = float(params.gripper_extra_close)
    if "squeeze_close" in enabled:
        extra_close -= float(params.squeeze_close_gain) * 0.35
        close_shift -= float(params.squeeze_close_gain) * 8.0

    contact_steps = int(params.contact_settle_steps)
    if "contact_settle" not in enabled:
        contact_steps = max(5, contact_steps // 2)

    micro_h1 = float(params.micro_lift_height_stage1)
    micro_s1 = int(params.micro_lift_steps_stage1)
    if "micro_lift" not in enabled:
        micro_h1 *= 0.5
        micro_s1 = max(5, micro_s1 // 2)

    micro_h2 = float(params.micro_lift_height_stage2)
    micro_s2 = int(params.micro_lift_steps_stage2)
    two_stage = float(params.enable_two_stage_lift)
    if "two_stage_lift" not in enabled:
        two_stage = 0.0
        micro_h2 = micro_h1
        micro_s2 = micro_s1

    speed = float(params.lift_speed_scale)
    if "slow_lift" in enabled:
        speed = min(speed, 0.35)

    grasp_params = GraspSearchParams(
        grasp_xy_offset_x=params.grasp_xy_offset_x,
        grasp_xy_offset_y=params.grasp_xy_offset_y,
        pre_grasp_height=pre_height,
        approach_height=approach_height,
        gripper_close_shift=close_shift,
        gripper_hold_steps=contact_steps,
        lift_height=float(micro_h1 + micro_h2),
        lift_steps=int(micro_s1 + micro_s2),
        speed_scale=speed,
    )
    refined, shifted_gripper = apply_grasp_params_to_eef_waypoints(proxy, eef_pose, grasp_params)

    if "lateral_correction" in enabled:
        approach_start = max(0, grasp_idx - 15)
        for step in range(approach_start, grasp_idx + 1):
            w = (step - approach_start) / max(1, grasp_idx - approach_start)
            refined[step, 0, 3] += float(params.lateral_correction_x) * w
            refined[step, 1, 3] += float(params.lateral_correction_y) * w

    if extra_close != 0.0:
        shifted_gripper = np.clip(shifted_gripper - extra_close, -1.0, 0.0)

    if "squeeze_close" in enabled and float(params.squeeze_close_gain) > 0:
        squeeze_start = max(0, grasp_idx - 2)
        squeeze_end = min(length - 1, grasp_idx + contact_steps)
        for step in range(squeeze_start, squeeze_end + 1):
            w = min(1.0, float(params.squeeze_close_gain))
            shifted_gripper[step] = min(shifted_gripper[step], -0.85 - 0.12 * w)

    settle_end = min(length - 1, grasp_idx + int(params.post_grasp_settle_steps))
    if settle_end > grasp_idx:
        hold = refined[grasp_idx, :3, 3].copy()
        rot = refined[grasp_idx, :3, :3].copy()
        for step in range(grasp_idx + 1, settle_end + 1):
            refined[step, :3, 3] = hold
            refined[step, :3, :3] = rot

    lift_begin = settle_end + 1
    stage1_end = min(length - 1, lift_begin + micro_s1)
    pause_end = min(length - 1, stage1_end + int(params.lift_pause_between_stages))
    stage2_end = min(length - 1, pause_end + micro_s2)

    if stage1_end > lift_begin:
        denom = max(1, stage1_end - lift_begin)
        for step in range(lift_begin, stage1_end + 1):
            g = (step - lift_begin) / denom
            refined[step, 2, 3] += micro_h1 * g
            refined[step, 2, 3] += float(params.lift_direction_bias_z) * g

    if "reclose_after_micro_lift" in enabled and int(params.reclose_after_micro_lift_steps) > 0:
        reclose_start = stage1_end + 1
        reclose_open_end = min(length - 1, reclose_start + 2)
        reclose_close_end = min(length - 1, reclose_open_end + int(params.reclose_after_micro_lift_steps))
        for step in range(reclose_start, reclose_open_end + 1):
            shifted_gripper[step] = min(0.15, shifted_gripper[step] + 0.25)
        for step in range(reclose_open_end + 1, reclose_close_end + 1):
            shifted_gripper[step] = -1.0
        pause_end = max(pause_end, reclose_close_end)

    if two_stage > 0.5 and stage2_end > pause_end:
        if pause_end > stage1_end:
            ppos = refined[stage1_end, :3, 3].copy()
            prot = refined[stage1_end, :3, :3].copy()
            for step in range(stage1_end + 1, pause_end + 1):
                refined[step, :3, 3] = ppos
                refined[step, :3, :3] = prot
        denom2 = max(1, stage2_end - pause_end)
        for step in range(pause_end + 1, stage2_end + 1):
            g = (step - pause_end) / denom2
            refined[step, 2, 3] += micro_h2 * g
            refined[step, 2, 3] += float(params.lift_direction_bias_z) * g * 0.5

    phases = {
        "grasp_index": grasp_idx,
        "lift_begin": lift_begin,
        "stage1_end": stage1_end,
        "pause_end": pause_end,
        "stage2_end": stage2_end,
        "contact_window_end": min(length - 1, grasp_idx + contact_steps),
    }
    refined, shifted_gripper = _extend_trajectory_for_post_lift(
        refined, shifted_gripper.reshape(-1), phases, params
    )
    return refined, shifted_gripper, phases


def _extend_trajectory_for_post_lift(
    refined: np.ndarray,
    gripper: np.ndarray,
    phases: dict[str, int],
    params: LiftV2B51Params,
) -> tuple[np.ndarray, np.ndarray]:
    ext_steps = int(params.post_extension_steps)
    if ext_steps <= 0:
        return refined, gripper
    last_pose = refined[-1].copy()
    z_step = float(params.extension_lift_height) / max(1, ext_steps)
    extra_poses: list[np.ndarray] = []
    extra_grip: list[float] = []
    for i in range(ext_steps):
        pose = last_pose.copy()
        pose[2, 3] += z_step * (i + 1)
        extra_poses.append(pose)
        extra_grip.append(-1.0)
    refined_ext = np.concatenate([refined, np.stack(extra_poses, axis=0)], axis=0)
    grip_ext = np.concatenate([gripper, np.asarray(extra_grip, dtype=float)])
    phases["stage2_end"] = len(refined_ext) - 1
    phases["extension_end"] = len(refined_ext) - 1
    return refined_ext, grip_ext


def build_lift_v2b51_waypoints_from_hdf5(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: LiftV2B51Params,
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
    refined, gripper, phases = apply_lift_v2b51_params_to_eef_waypoints(proxy, original_eef, params)
    return proxy, original_eef, refined, gripper, phases


def lift_v2b51_params_from_dict(raw: dict[str, Any]) -> LiftV2B51Params:
    valid = {f.name for f in fields(LiftV2B51Params)}
    return LiftV2B51Params(**{k: v for k, v in raw.items() if k in valid})
