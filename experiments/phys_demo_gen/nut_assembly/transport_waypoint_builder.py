"""V2-B3：transport_failed demo 的 eef waypoint 构建（不修改 object_poses）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from refined_waypoint_builder import (
    _rotate_z_about_point,
    _rotate_z_mat,
    load_eef_pose_sequence,
)
from trajectory_parameterization import (
    TrajectoryProxy,
    _gripper_closed_mask,
    _shift_1d_signal,
    load_trajectory_proxy,
)

NEAR_PEG_XY_THRESH = 0.08


@dataclass
class TransportSearchParams:
    """V2-B3 transport sim-in-loop 搜索参数（仅作用于 eef waypoint / 闭环 action）。"""

    transport_xy_gain: float = 1.0
    transport_xy_offset_scale: float = 1.0
    pre_align_height: float = 0.06
    lift_height: float = 0.06
    approach_steps: int = 20
    transport_steps: int = 40
    hold_steps: int = 10
    gripper_close_shift: float = 0.0
    speed_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_theta_with_transport_params(
    base_theta: dict[str, Any],
    params: TransportSearchParams,
) -> dict[str, Any]:
    """将 CEM best_theta 与 transport 搜索参数合并（不 attenuate）。"""
    theta = {key: (np.array(val).copy() if isinstance(val, (list, np.ndarray)) else val) for key, val in base_theta.items()}
    transport = np.asarray(theta["transport_xy_offset"], dtype=float).reshape(2).copy()
    transport *= float(params.transport_xy_offset_scale)
    theta["transport_xy_offset"] = transport
    theta["speed_scale"] = float(params.speed_scale)
    theta["gripper_close_shift"] = float(base_theta.get("gripper_close_shift", 0.0)) + float(
        params.gripper_close_shift
    )
    return theta


def apply_transport_params_to_eef_waypoints(
    proxy: TrajectoryProxy,
    eef_pose: np.ndarray,
    theta: dict[str, Any],
    params: TransportSearchParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 CEM theta + transport 搜索参数映射为 refined eef waypoint。

    transport_failed demo 使用完整 transport theta（不做 rollout attenuation）。
    """
    transport_xy = np.asarray(theta["transport_xy_offset"], dtype=float).reshape(2)
    pre_align_xy = np.asarray(theta["pre_align_xy_offset"], dtype=float).reshape(2)
    align_yaw = float(theta["align_yaw_offset"])
    insert_z = float(theta["insert_z_offset"])
    grip_close_shift = float(theta["gripper_close_shift"])
    release_shift = float(theta.get("release_shift", 0.0))

    refined = eef_pose.copy()
    phases = proxy.phases
    grasp_idx = phases.grasp_index
    t_min_xy = phases.t_min_xy
    final_idx = phases.final_index
    length = proxy.length

    shifted_gripper = _shift_1d_signal(proxy.gripper_action, grip_close_shift)
    shifted_gripper = _shift_1d_signal(shifted_gripper, release_shift * 0.5)

    # approach：抓取前抬升
    approach_start = max(0, grasp_idx - int(params.approach_steps))
    denom_approach = max(1, grasp_idx - approach_start)
    for step in range(approach_start, grasp_idx + 1):
        ramp = (step - approach_start) / denom_approach
        refined[step, 2, 3] += float(params.lift_height) * ramp

    # transport window：xy 偏移 + gain
    t0 = grasp_idx
    t1 = min(length - 1, max(t_min_xy, t0 + int(params.transport_steps)))
    denom_transport = max(1, t1 - t0)
    gain = float(params.transport_xy_gain)
    for step in range(t0, t1 + 1):
        ramp = (step - t0) / denom_transport
        offset = transport_xy * ramp * gain
        refined[step, :3, 3] += np.array([offset[0], offset[1], 0.0])

    # pre-align：xy 微调 + 高度
    align_start = max(t0, t_min_xy - 12)
    denom_align = max(1, t_min_xy - align_start)
    peg_center = proxy.peg_pos[t_min_xy, :2]
    for step in range(align_start, t_min_xy + 1):
        beta = (step - align_start) / denom_align
        refined[step, :2, 3] += pre_align_xy * beta
        refined[step, 2, 3] += float(params.pre_align_height) * beta

    # hold：到达 peg 附近后短暂保持 eef
    hold_end = min(length, t_min_xy + int(params.hold_steps))
    if hold_end > t_min_xy:
        hold_pos = refined[t_min_xy, :3, 3].copy()
        hold_rot = refined[t_min_xy, :3, :3].copy()
        for step in range(t_min_xy, hold_end):
            refined[step, :3, 3] = hold_pos
            refined[step, :3, :3] = hold_rot

    # insertion window：保留 CEM insert_z（transport demo 通常 z 已接近 target）
    denom_insert = max(1, final_idx - t_min_xy)
    for step in range(t_min_xy, length):
        gamma = (step - t_min_xy) / denom_insert
        refined[step, 2, 3] += insert_z * gamma

    # align_yaw
    yaw_start = max(0, t_min_xy - 5)
    yaw_end = min(length, t_min_xy + 15)
    for step in range(yaw_start, yaw_end):
        weight = 1.0 - abs(step - t_min_xy) / max(1, yaw_end - yaw_start)
        angle = align_yaw * weight
        refined[step, :3, :3] = _rotate_z_mat(refined[step, :3, :3], angle)
        refined[step, :2, 3] = _rotate_z_about_point(refined[step, :2, 3], angle, peg_center)

    return refined, shifted_gripper.reshape(-1)


def build_transport_waypoints_from_hdf5(
    hdf5_path: str,
    demo_key: str,
    label: str,
    base_theta: dict[str, Any],
    params: TransportSearchParams,
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    effective_theta = merge_theta_with_transport_params(base_theta, params)
    original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
    refined_eef, shifted_gripper = apply_transport_params_to_eef_waypoints(
        proxy, original_eef, effective_theta, params
    )
    return proxy, original_eef, refined_eef, shifted_gripper, effective_theta
