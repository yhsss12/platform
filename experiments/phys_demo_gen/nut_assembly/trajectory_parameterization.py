"""Nut Assembly 轨迹阶段划分与可优化参数 theta（V2-A proxy parameterization）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from extract_features import (
    NutAssemblyFeatures,
    action_acceleration_stats,
    grasp_signal_index,
    list_demo_keys,
    square_yaw_error,
)

# ---------------------------------------------------------------------------
# Theta bounds — CEM 搜索空间
# ---------------------------------------------------------------------------

THETA_KEYS = [
    "transport_xy_offset",
    "pre_align_xy_offset",
    "align_yaw_offset",
    "insert_z_offset",
    "speed_scale",
    "gripper_close_shift",
    "release_shift",
]

THETA_BOUNDS: dict[str, tuple[float, float]] = {
    "transport_xy_offset_x": (-0.35, 0.35),
    "transport_xy_offset_y": (-0.35, 0.35),
    "pre_align_xy_offset_x": (-0.08, 0.08),
    "pre_align_xy_offset_y": (-0.08, 0.08),
    "align_yaw_offset": (-0.785, 0.785),
    "insert_z_offset": (-0.12, 0.02),
    "speed_scale": (0.5, 1.2),
    "gripper_close_shift": (-10.0, 10.0),
    "release_shift": (-10.0, 10.0),
}

THETA_DIM = len(THETA_BOUNDS)
THETA_BOUND_KEYS = list(THETA_BOUNDS.keys())


@dataclass
class TrajectoryPhases:
    grasp_index: int
    t_min_xy: int
    final_index: int
    transport_window: tuple[int, int]
    insertion_window: tuple[int, int]


@dataclass
class TrajectoryProxy:
    """内存中的 proxy 轨迹状态（只读加载，不修改 HDF5）。"""

    demo_key: str
    label: str
    source_file: str
    actions: np.ndarray
    nut_pos: np.ndarray
    nut_rot: np.ndarray
    peg_pos: np.ndarray
    peg_rot: np.ndarray
    eef_pos: np.ndarray
    target_pos: np.ndarray
    gripper_action: np.ndarray
    grasp_signal: np.ndarray
    phases: TrajectoryPhases
    xy_distance_baseline: np.ndarray = field(repr=False)
    z_difference_baseline: np.ndarray = field(repr=False)
    yaw_error_baseline: np.ndarray = field(repr=False)

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def empty_theta() -> dict[str, Any]:
    return {
        "transport_xy_offset": np.zeros(2, dtype=float),
        "pre_align_xy_offset": np.zeros(2, dtype=float),
        "align_yaw_offset": 0.0,
        "insert_z_offset": 0.0,
        "speed_scale": 1.0,
        "gripper_close_shift": 0.0,
        "release_shift": 0.0,
    }


def theta_to_vector(theta: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(theta["transport_xy_offset"][0]),
            float(theta["transport_xy_offset"][1]),
            float(theta["pre_align_xy_offset"][0]),
            float(theta["pre_align_xy_offset"][1]),
            float(theta["align_yaw_offset"]),
            float(theta["insert_z_offset"]),
            float(theta["speed_scale"]),
            float(theta["gripper_close_shift"]),
            float(theta["release_shift"]),
        ],
        dtype=float,
    )


def vector_to_theta(vector: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.shape[0] != THETA_DIM:
        raise ValueError(f"expected theta dim {THETA_DIM}, got {vector.shape[0]}")
    return {
        "transport_xy_offset": np.array([vector[0], vector[1]], dtype=float),
        "pre_align_xy_offset": np.array([vector[2], vector[3]], dtype=float),
        "align_yaw_offset": float(vector[4]),
        "insert_z_offset": float(vector[5]),
        "speed_scale": float(vector[6]),
        "gripper_close_shift": float(vector[7]),
        "release_shift": float(vector[8]),
    }


def clip_theta_vector(vector: np.ndarray) -> np.ndarray:
    clipped = np.array(vector, dtype=float).copy()
    for index, key in enumerate(THETA_BOUND_KEYS):
        low, high = THETA_BOUNDS[key]
        clipped[index] = np.clip(clipped[index], low, high)
    return clipped


def theta_bounds_as_arrays() -> tuple[np.ndarray, np.ndarray]:
    lows = np.array([THETA_BOUNDS[key][0] for key in THETA_BOUND_KEYS], dtype=float)
    highs = np.array([THETA_BOUNDS[key][1] for key in THETA_BOUND_KEYS], dtype=float)
    return lows, highs


def extract_phases(
    xy_distance: np.ndarray,
    grasp_index: int | None,
    length: int,
) -> TrajectoryPhases:
    grasp = 0 if grasp_index is None else int(grasp_index)
    grasp = max(0, min(grasp, length - 1))
    t_min_xy = int(np.argmin(xy_distance))
    t_min_xy = max(grasp, min(t_min_xy, length - 1))
    final_index = length - 1
    return TrajectoryPhases(
        grasp_index=grasp,
        t_min_xy=t_min_xy,
        final_index=final_index,
        transport_window=(grasp, t_min_xy),
        insertion_window=(t_min_xy, final_index),
    )


def load_trajectory_proxy(hdf5_path: str, demo_key: str, label: str) -> TrajectoryProxy:
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle[f"data/{demo_key}"]
        actions = demo["actions"][:].astype(float)
        nut = demo["datagen_info/object_poses/square_nut"][:].astype(float)
        peg = demo["datagen_info/object_poses/square_peg"][:].astype(float)
        eef = demo["datagen_info/eef_pose"][:].astype(float)
        target = demo["datagen_info/target_pose"][:].astype(float)
        gripper = demo["datagen_info/gripper_action"][:].astype(float)
        grasp = demo["datagen_info/subtask_term_signals/grasp"][:].astype(float)

    nut_pos = nut[:, :3, 3].copy()
    peg_pos = peg[:, :3, 3].copy()
    xy_distance = np.linalg.norm(nut_pos[:, :2] - peg_pos[:, :2], axis=1)
    z_difference = nut_pos[:, 2] - peg_pos[:, 2]
    yaw_error = square_yaw_error(nut[:, :3, :3], peg[:, :3, :3])

    grasp_idx = grasp_signal_index(grasp)
    phases = extract_phases(xy_distance, grasp_idx, len(actions))

    return TrajectoryProxy(
        demo_key=demo_key,
        label=label,
        source_file=hdf5_path,
        actions=actions,
        nut_pos=nut_pos,
        nut_rot=nut[:, :3, :3].copy(),
        peg_pos=peg_pos,
        peg_rot=peg[:, :3, :3].copy(),
        eef_pos=eef[:, :3, 3].copy(),
        target_pos=target[:, :3, 3].copy(),
        gripper_action=gripper.reshape(-1).copy(),
        grasp_signal=grasp.reshape(-1).copy(),
        phases=phases,
        xy_distance_baseline=xy_distance,
        z_difference_baseline=z_difference,
        yaw_error_baseline=yaw_error,
    )


def load_all_proxies(hdf5_path: str, label: str) -> list[TrajectoryProxy]:
    with h5py.File(hdf5_path, "r") as handle:
        keys = list_demo_keys(handle["data"])
    return [load_trajectory_proxy(hdf5_path, key, label) for key in keys]


def _rotate_z(points_xy: np.ndarray, angle: float, center_xy: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rotated = points_xy - center_xy
    out = np.empty_like(rotated)
    out[:, 0] = c * rotated[:, 0] - s * rotated[:, 1]
    out[:, 1] = s * rotated[:, 0] + c * rotated[:, 1]
    return out + center_xy


def _shift_1d_signal(signal: np.ndarray, shift_steps: float) -> np.ndarray:
    """离散信号平移 proxy（边界填充）。"""
    length = len(signal)
    shift = int(round(shift_steps))
    if shift == 0:
        return signal.copy()
    out = np.empty_like(signal)
    if shift > 0:
        out[:shift] = signal[0]
        out[shift:] = signal[:-shift]
    else:
        shift = abs(shift)
        out[-shift:] = signal[-1]
        out[:-shift] = signal[shift:]
    return out


def _gripper_closed_mask(gripper_action: np.ndarray, grasp_signal: np.ndarray) -> np.ndarray:
    closed_gripper = gripper_action < 0.0
    grasp_active = grasp_signal > 0.5
    return closed_gripper | grasp_active


def features_from_proxy_arrays(
    proxy: TrajectoryProxy,
    nut_pos: np.ndarray,
    nut_rot: np.ndarray,
    actions: np.ndarray,
) -> NutAssemblyFeatures:
    xy_distance = np.linalg.norm(nut_pos[:, :2] - proxy.peg_pos[:, :2], axis=1)
    z_difference = nut_pos[:, 2] - proxy.peg_pos[:, 2]
    yaw_error = square_yaw_error(nut_rot, proxy.peg_rot)
    acc_mean, acc_max = action_acceleration_stats(actions)

    return NutAssemblyFeatures(
        demo_key=proxy.demo_key,
        label=proxy.label,
        source_file=proxy.source_file,
        length=proxy.length,
        final_nut_peg_xy_distance=float(xy_distance[-1]),
        min_nut_peg_xy_distance=float(xy_distance.min()),
        final_nut_peg_z_difference=float(z_difference[-1]),
        min_nut_peg_yaw_error=float(yaw_error.min()),
        final_nut_peg_yaw_error=float(yaw_error[-1]),
        action_acceleration_mean=acc_mean,
        action_acceleration_max=acc_max,
        grasp_signal_index=proxy.phases.grasp_index,
    )


def apply_theta_to_proxy_features(
    proxy: TrajectoryProxy,
    theta: dict[str, Any] | np.ndarray,
) -> NutAssemblyFeatures:
    """
    Offline proxy refinement：将 theta 作用于内存 proxy 轨迹并重算特征。

    注意：这不是可执行 rollout，也不修改 HDF5 中的 object_poses。
    仅在 grasp 后 transport / insertion 窗口内，按规则生成 proxy nut/eef 轨迹。
    """
    if isinstance(theta, np.ndarray):
        theta = vector_to_theta(theta)

    transport_xy = np.asarray(theta["transport_xy_offset"], dtype=float).reshape(2)
    pre_align_xy = np.asarray(theta["pre_align_xy_offset"], dtype=float).reshape(2)
    align_yaw = float(theta["align_yaw_offset"])
    insert_z = float(theta["insert_z_offset"])
    speed_scale = float(np.clip(theta["speed_scale"], 0.5, 1.2))
    grip_close_shift = float(theta["gripper_close_shift"])
    release_shift = float(theta["release_shift"])

    phases = proxy.phases
    grasp_idx = phases.grasp_index
    t_min_xy = phases.t_min_xy
    final_idx = phases.final_index
    length = proxy.length

    proxy_nut_pos = proxy.nut_pos.copy()
    proxy_nut_rot = proxy.nut_rot.copy()
    proxy_eef_pos = proxy.eef_pos.copy()
    proxy_target_pos = proxy.target_pos.copy()
    proxy_actions = proxy.actions.copy()
    proxy_gripper = _shift_1d_signal(proxy.gripper_action, grip_close_shift)
    proxy_gripper = _shift_1d_signal(proxy_gripper, release_shift * 0.5)
    gripper_closed = _gripper_closed_mask(proxy_gripper, proxy.grasp_signal)

    attach_offset = proxy.nut_pos[grasp_idx, :2] - proxy.eef_pos[grasp_idx, :2]

    # 1) transport window：eef / target xy 偏移；夹持时 nut 跟随 eef
    t0, t1 = phases.transport_window
    denom_transport = max(1, t1 - t0)
    for step in range(t0, t1 + 1):
        ramp = (step - t0) / denom_transport
        offset = transport_xy * ramp
        proxy_eef_pos[step, :2] += offset
        proxy_target_pos[step, :2] += offset * 0.8
        if gripper_closed[step]:
            proxy_nut_pos[step, :2] = proxy_eef_pos[step, :2] + attach_offset
        else:
            proxy_nut_pos[step, :2] += offset * 0.35

    # 2) pre-align：t_min_xy 附近额外 xy 微调
    align_start = max(t0, t_min_xy - 12)
    denom_align = max(1, t_min_xy - align_start)
    for step in range(align_start, t_min_xy + 1):
        beta = (step - align_start) / denom_align
        delta = pre_align_xy * beta
        proxy_eef_pos[step, :2] += delta
        if gripper_closed[step]:
            proxy_nut_pos[step, :2] = proxy_eef_pos[step, :2] + attach_offset
        else:
            proxy_nut_pos[step, :2] += delta * 0.5

    # 3) insertion window：z 方向 proxy 偏移
    denom_insert = max(1, final_idx - t_min_xy)
    for step in range(t_min_xy, length):
        gamma = (step - t_min_xy) / denom_insert
        z_delta = insert_z * gamma
        proxy_eef_pos[step, 2] += z_delta
        proxy_target_pos[step, 2] += z_delta * 0.9
        if gripper_closed[step] or step >= t_min_xy:
            proxy_nut_pos[step, 2] += z_delta

    # 4) align yaw：t_min_xy 附近 nut yaw proxy
    yaw_start = max(0, t_min_xy - 5)
    yaw_end = min(length, t_min_xy + 15)
    peg_center = proxy.peg_pos[t_min_xy, :2]
    for step in range(yaw_start, yaw_end):
        weight = 1.0 - abs(step - t_min_xy) / max(1, yaw_end - yaw_start)
        angle = align_yaw * weight
        proxy_nut_pos[step, :2] = _rotate_z(proxy_nut_pos[step : step + 1, :2], angle, peg_center)[0]
        rot_z = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        proxy_nut_rot[step] = rot_z @ proxy_nut_rot[step]

    # 5) speed_scale：缩放 action 差分影响平滑性 proxy
    if abs(speed_scale - 1.0) > 1e-6 and length > 1:
        deltas = np.diff(proxy_actions, axis=0) / speed_scale
        proxy_actions[0] = proxy.actions[0]
        for step in range(1, length):
            proxy_actions[step] = proxy_actions[step - 1] + deltas[step - 1]

    # 6) 插入后保持 nut xy 接近 t_min_xy 时刻（避免末端漂移）
    if t_min_xy < final_idx:
        hold_xy = proxy_nut_pos[t_min_xy, :2].copy()
        tail_blend = np.linspace(0.0, 1.0, final_idx - t_min_xy + 1)
        for offset, step in enumerate(range(t_min_xy, length)):
            blend = tail_blend[offset]
            proxy_nut_pos[step, :2] = (1.0 - 0.35 * blend) * proxy_nut_pos[step, :2] + 0.35 * blend * hold_xy

    return features_from_proxy_arrays(proxy, proxy_nut_pos, proxy_nut_rot, proxy_actions)


def suggest_initial_theta(proxy: TrajectoryProxy) -> np.ndarray:
    """根据失败残差给 CEM 初始均值一个物理方向 hint。"""
    theta = theta_to_vector(empty_theta())
    grasp = proxy.phases.grasp_index
    nut_xy = proxy.nut_pos[grasp, :2]
    peg_xy = proxy.peg_pos[grasp, :2]
    toward_peg = peg_xy - nut_xy
    dist = float(np.linalg.norm(toward_peg))
    if dist > 1e-6:
        unit = toward_peg / dist
        move = min(0.32, dist * 0.95)
        theta[0] = float(np.clip(unit[0] * move, *THETA_BOUNDS["transport_xy_offset_x"]))
        theta[1] = float(np.clip(unit[1] * move, *THETA_BOUNDS["transport_xy_offset_y"]))

    final_z = float(proxy.z_difference_baseline[-1])
    if final_z > 0.02:
        theta[5] = float(np.clip(-0.10, *THETA_BOUNDS["insert_z_offset"]))

    min_yaw = float(proxy.yaw_error_baseline.min())
    if min_yaw > 0.05:
        theta[4] = float(np.clip(-0.15, *THETA_BOUNDS["align_yaw_offset"]))

    return theta
