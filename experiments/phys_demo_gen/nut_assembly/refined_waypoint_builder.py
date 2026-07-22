"""V2-B2：将 CEM best_theta 映射为 refined eef waypoint（不修改 nut_pose / object_poses）。"""
from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from trajectory_parameterization import (
    TrajectoryProxy,
    _gripper_closed_mask,
    _shift_1d_signal,
    load_trajectory_proxy,
)

NEAR_PEG_XY_THRESH = 0.08
YAW_ALIGN_THRESH = 0.05


def _attenuate_theta_for_rollout(proxy: TrajectoryProxy, theta: dict[str, Any]) -> dict[str, Any]:
    """
    根据原始轨迹残差 attenuate theta：insertion_failed 时不应再大幅 transport。
    proxy CEM 会同时移动 proxy nut；真实 rollout 只动 eef，需更保守。
    """
    adjusted = {key: (np.array(val).copy() if isinstance(val, (list, np.ndarray)) else val) for key, val in theta.items()}
    min_xy = float(proxy.xy_distance_baseline.min())
    min_yaw = float(proxy.yaw_error_baseline.min())
    final_z = float(proxy.z_difference_baseline[-1])

    transport = np.asarray(adjusted["transport_xy_offset"], dtype=float).reshape(2).copy()
    pre_align = np.asarray(adjusted["pre_align_xy_offset"], dtype=float).reshape(2).copy()

    if min_xy <= NEAR_PEG_XY_THRESH:
        scale = max(0.05, min_xy / NEAR_PEG_XY_THRESH)
        transport *= scale
        pre_align *= scale

    if min_yaw <= YAW_ALIGN_THRESH:
        adjusted["align_yaw_offset"] = float(adjusted["align_yaw_offset"]) * max(0.1, min_yaw / YAW_ALIGN_THRESH)

    if final_z > 0.02:
        # insertion 失败：保留完整 insert_z，略增 z 修正力度
        adjusted["insert_z_offset"] = float(adjusted["insert_z_offset"]) * 1.05

    adjusted["transport_xy_offset"] = transport
    adjusted["pre_align_xy_offset"] = pre_align
    return adjusted


def load_eef_pose_sequence(hdf5_path: str, demo_key: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as handle:
        return handle[f"data/{demo_key}/datagen_info/eef_pose"][:].astype(float)


def _rotate_z_about_point(pos_xy: np.ndarray, angle: float, center_xy: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rel = pos_xy - center_xy
    out = np.empty_like(rel)
    out[0] = c * rel[0] - s * rel[1]
    out[1] = s * rel[0] + c * rel[1]
    return out + center_xy


def _rotate_z_mat(rot: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot


def apply_theta_to_eef_waypoints(
    proxy: TrajectoryProxy,
    eef_pose: np.ndarray,
    theta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 theta 仅作用于 eef 4x4 waypoint 与 gripper 时序。

    不修改 nut_pose / sim object state；返回 refined eef poses 与 shifted gripper actions。
    """
    transport_xy = np.asarray(theta["transport_xy_offset"], dtype=float).reshape(2)
    pre_align_xy = np.asarray(theta["pre_align_xy_offset"], dtype=float).reshape(2)
    align_yaw = float(theta["align_yaw_offset"])
    insert_z = float(theta["insert_z_offset"])
    grip_close_shift = float(theta["gripper_close_shift"])
    release_shift = float(theta["release_shift"])

    refined = eef_pose.copy()
    phases = proxy.phases
    grasp_idx = phases.grasp_index
    t_min_xy = phases.t_min_xy
    final_idx = phases.final_index
    length = proxy.length

    shifted_gripper = _shift_1d_signal(proxy.gripper_action, grip_close_shift)
    shifted_gripper = _shift_1d_signal(shifted_gripper, release_shift * 0.5)
    gripper_closed = _gripper_closed_mask(shifted_gripper, proxy.grasp_signal)

    # 1) transport window：eef xy waypoint
    t0, t1 = phases.transport_window
    denom_transport = max(1, t1 - t0)
    for step in range(t0, t1 + 1):
        ramp = (step - t0) / denom_transport
        offset = transport_xy * ramp
        refined[step, :3, 3] += np.array([offset[0], offset[1], 0.0])

    # 2) pre-align window：eef xy 微调
    align_start = max(t0, t_min_xy - 12)
    denom_align = max(1, t_min_xy - align_start)
    peg_center = proxy.peg_pos[t_min_xy, :2]
    for step in range(align_start, t_min_xy + 1):
        beta = (step - align_start) / denom_align
        delta = pre_align_xy * beta
        refined[step, :2, 3] += delta

    # 3) insertion window：eef z waypoint
    denom_insert = max(1, final_idx - t_min_xy)
    for step in range(t_min_xy, length):
        gamma = (step - t_min_xy) / denom_insert
        z_delta = insert_z * gamma
        refined[step, 2, 3] += z_delta

    # 4) align_yaw：作用于 eef 姿态（绕 z），不直接改 nut yaw
    yaw_start = max(0, t_min_xy - 5)
    yaw_end = min(length, t_min_xy + 15)
    for step in range(yaw_start, yaw_end):
        weight = 1.0 - abs(step - t_min_xy) / max(1, yaw_end - yaw_start)
        angle = align_yaw * weight
        refined[step, :3, :3] = _rotate_z_mat(refined[step, :3, :3], angle)
        refined[step, :2, 3] = _rotate_z_about_point(refined[step, :2, 3], angle, peg_center)

    return refined, shifted_gripper.reshape(-1)


def build_refined_waypoints_from_hdf5(
    hdf5_path: str,
    demo_key: str,
    label: str,
    theta: dict[str, Any],
    *,
    rollout_safe: bool = True,
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    effective_theta = _attenuate_theta_for_rollout(proxy, theta) if rollout_safe else theta
    original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
    refined_eef, shifted_gripper = apply_theta_to_eef_waypoints(proxy, original_eef, effective_theta)
    return proxy, original_eef, refined_eef, shifted_gripper
