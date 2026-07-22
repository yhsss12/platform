"""V2-B4：grasp_failed demo 的 eef waypoint 构建（不修改 object_poses）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from refined_waypoint_builder import load_eef_pose_sequence
from trajectory_parameterization import TrajectoryProxy, _shift_1d_signal, load_trajectory_proxy


@dataclass
class GraspSearchParams:
    """V2-B4 grasp-stage sim-in-loop 搜索参数（仅作用于 eef waypoint / gripper 时序）。"""

    grasp_xy_offset_x: float = 0.0
    grasp_xy_offset_y: float = 0.0
    pre_grasp_height: float = 0.05
    approach_height: float = 0.02
    gripper_close_shift: float = 0.0
    gripper_hold_steps: int = 20
    lift_height: float = 0.06
    lift_steps: int = 20
    speed_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_grasp_params_to_eef_waypoints(
    proxy: TrajectoryProxy,
    eef_pose: np.ndarray,
    params: GraspSearchParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在 grasp 窗口修正 eef waypoint 与 gripper 时序。

    仅修改 eef 4x4 waypoint，不触碰 nut/peg object state。
    """
    refined = eef_pose.copy()
    phases = proxy.phases
    grasp_idx = phases.grasp_index
    length = proxy.length

    shifted_gripper = _shift_1d_signal(proxy.gripper_action, float(params.gripper_close_shift))

    pre_start = max(0, grasp_idx - 15)
    denom_pre = max(1, grasp_idx - pre_start)
    for step in range(pre_start, grasp_idx + 1):
        ramp = (step - pre_start) / denom_pre
        refined[step, 2, 3] += float(params.pre_grasp_height) * ramp

    approach_start = max(0, grasp_idx - 8)
    denom_approach = max(1, grasp_idx - approach_start)
    for step in range(approach_start, grasp_idx + 1):
        beta = (step - approach_start) / denom_approach
        refined[step, 2, 3] += float(params.approach_height) * beta

    grasp_xy = np.array([float(params.grasp_xy_offset_x), float(params.grasp_xy_offset_y)], dtype=float)
    grasp_end = min(length - 1, grasp_idx + int(params.gripper_hold_steps))
    denom_grasp = max(1, grasp_end - max(0, grasp_idx - 5))
    for step in range(max(0, grasp_idx - 5), grasp_end + 1):
        weight = (step - max(0, grasp_idx - 5)) / denom_grasp
        refined[step, :2, 3] += grasp_xy * weight

    lift_start = min(length - 1, grasp_idx + 1)
    lift_end = min(length - 1, lift_start + int(params.lift_steps))
    if lift_end > lift_start:
        denom_lift = max(1, lift_end - lift_start)
        for step in range(lift_start, lift_end + 1):
            gamma = (step - lift_start) / denom_lift
            refined[step, 2, 3] += float(params.lift_height) * gamma

    hold_end = min(length, grasp_idx + int(params.gripper_hold_steps) + 1)
    if hold_end > grasp_idx + 1:
        hold_pos = refined[grasp_idx, :3, 3].copy()
        hold_rot = refined[grasp_idx, :3, :3].copy()
        for step in range(grasp_idx + 1, hold_end):
            refined[step, :2, 3] = hold_pos[:2]
            refined[step, 2, 3] = max(refined[step, 2, 3], hold_pos[2])
            refined[step, :3, :3] = hold_rot

    return refined, shifted_gripper.reshape(-1)


def build_grasp_waypoints_from_hdf5(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: GraspSearchParams,
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
    refined_eef, shifted_gripper = apply_grasp_params_to_eef_waypoints(proxy, original_eef, params)
    return proxy, original_eef, refined_eef, shifted_gripper
