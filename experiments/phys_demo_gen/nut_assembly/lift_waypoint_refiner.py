"""V1-F：lift_failed demo 的 lift-aware waypoint / gripper refiner（不修改 object_poses）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from grasp_waypoint_builder import GraspSearchParams, apply_grasp_params_to_eef_waypoints
from refined_waypoint_builder import load_eef_pose_sequence
from trajectory_parameterization import TrajectoryProxy, _shift_1d_signal, load_trajectory_proxy


@dataclass
class LiftRepairParams(GraspSearchParams):
    """V1-F lift-aware repair 参数（在 grasp 参数基础上扩展）。"""

    micro_lift_height: float = 0.06
    micro_lift_steps: int = 20
    regrasp_shift: float = 0.0
    gripper_extra_close: float = 0.0
    lift_speed_scale: float = 1.0
    lift_pause_steps: int = 0
    contact_hold_steps: int = 20
    post_grasp_settle_steps: int = 5
    lift_direction_bias: float = 0.0
    nut_follow_threshold: float = 0.05

    def __post_init__(self) -> None:
        # 与 grasp 字段对齐
        if self.lift_height == 0.06 and self.micro_lift_height != 0.06:
            self.lift_height = float(self.micro_lift_height)
        if self.lift_steps == 20 and self.micro_lift_steps != 20:
            self.lift_steps = int(self.micro_lift_steps)
        if self.speed_scale == 1.0 and self.lift_speed_scale != 1.0:
            self.speed_scale = float(self.lift_speed_scale)

    def to_dict(self) -> dict[str, Any]:
        base = asdict(self)
        return base


LIFT_REPAIR_SEARCH_SPACE: dict[str, list[float | int]] = {
    "grasp_xy_offset_x": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "grasp_xy_offset_y": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "pre_grasp_height": [0.03, 0.05, 0.07, 0.09],
    "approach_height": [0.01, 0.02, 0.03],
    "gripper_close_shift": [-15, -10, -5, 0, 5],
    "gripper_hold_steps": [10, 20, 30, 40],
    "micro_lift_height": [0.04, 0.06, 0.08, 0.10, 0.12],
    "micro_lift_steps": [10, 15, 20, 30, 40],
    "regrasp_shift": [-10, -5, 0, 5, 10],
    "gripper_extra_close": [-0.2, -0.1, 0.0, 0.1],
    "lift_speed_scale": [0.3, 0.4, 0.6, 0.8, 1.0],
    "lift_pause_steps": [0, 3, 5, 10],
    "contact_hold_steps": [10, 20, 30, 40],
    "post_grasp_settle_steps": [0, 3, 5, 10, 15],
    "lift_direction_bias": [-0.02, -0.01, 0.0, 0.01, 0.02],
    "nut_follow_threshold": [0.03, 0.04, 0.05, 0.06, 0.08],
}


def apply_lift_params_to_eef_waypoints(
    proxy: TrajectoryProxy,
    eef_pose: np.ndarray,
    params: LiftRepairParams,
) -> tuple[np.ndarray, np.ndarray]:
    grasp_params = GraspSearchParams(
        grasp_xy_offset_x=params.grasp_xy_offset_x,
        grasp_xy_offset_y=params.grasp_xy_offset_y,
        pre_grasp_height=params.pre_grasp_height,
        approach_height=params.approach_height,
        gripper_close_shift=params.gripper_close_shift + params.regrasp_shift,
        gripper_hold_steps=int(params.contact_hold_steps or params.gripper_hold_steps),
        lift_height=float(params.micro_lift_height),
        lift_steps=int(params.micro_lift_steps),
        speed_scale=float(params.lift_speed_scale),
    )
    refined, shifted_gripper = apply_grasp_params_to_eef_waypoints(proxy, eef_pose, grasp_params)

    grasp_idx = proxy.phases.grasp_index
    length = proxy.length

    if params.gripper_extra_close != 0.0:
        shifted_gripper = np.clip(shifted_gripper - float(params.gripper_extra_close), -1.0, 0.0)

    settle_end = min(length - 1, grasp_idx + int(params.post_grasp_settle_steps))
    if settle_end > grasp_idx:
        hold_pos = refined[grasp_idx, :3, 3].copy()
        hold_rot = refined[grasp_idx, :3, :3].copy()
        for step in range(grasp_idx + 1, settle_end + 1):
            refined[step, :3, 3] = hold_pos
            refined[step, :3, :3] = hold_rot

    lift_start = min(length - 1, settle_end + 1)
    pause_end = min(length - 1, lift_start + int(params.lift_pause_steps))
    if pause_end > lift_start:
        pause_pos = refined[lift_start, :3, 3].copy()
        pause_rot = refined[lift_start, :3, :3].copy()
        for step in range(lift_start, pause_end + 1):
            refined[step, :3, 3] = pause_pos
            refined[step, :3, :3] = pause_rot

    lift_begin = min(length - 1, pause_end + 1)
    lift_end = min(length - 1, lift_begin + int(params.micro_lift_steps))
    if lift_end > lift_begin:
        denom = max(1, lift_end - lift_begin)
        for step in range(lift_begin, lift_end + 1):
            gamma = (step - lift_begin) / denom
            refined[step, 2, 3] += float(params.micro_lift_height) * gamma
            refined[step, 2, 3] += float(params.lift_direction_bias) * gamma
            refined[step, 0, 3] += 0.5 * float(params.lift_direction_bias) * gamma

    return refined, shifted_gripper.reshape(-1)


def build_lift_waypoints_from_hdf5(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: LiftRepairParams,
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
    refined_eef, shifted_gripper = apply_lift_params_to_eef_waypoints(proxy, original_eef, params)
    return proxy, original_eef, refined_eef, shifted_gripper


def lift_params_from_dict(raw: dict[str, Any]) -> LiftRepairParams:
    valid = {f.name for f in fields(LiftRepairParams)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    return LiftRepairParams(**kwargs)
