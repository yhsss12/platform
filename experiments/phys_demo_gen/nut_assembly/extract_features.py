"""从 MimicGen / RoboSuite Nut Assembly HDF5 demo 提取物理特征。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import h5py
import numpy as np


@dataclass
class NutAssemblyFeatures:
    demo_key: str
    label: str
    source_file: str
    length: int
    final_nut_peg_xy_distance: float
    min_nut_peg_xy_distance: float
    final_nut_peg_z_difference: float
    min_nut_peg_yaw_error: float
    final_nut_peg_yaw_error: float
    action_acceleration_mean: float
    action_acceleration_max: float
    grasp_signal_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_demo_keys(data_grp: h5py.Group) -> list[str]:
    return sorted(data_grp.keys(), key=lambda k: int(k.split("_")[-1]))


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0])


def square_yaw_error(nut_rot: np.ndarray, peg_rot: np.ndarray) -> np.ndarray:
    """方螺母四重旋转对称：yaw 误差折叠到 [-pi/4, pi/4]。"""
    delta = yaw_from_rot(nut_rot) - yaw_from_rot(peg_rot)
    delta = (delta + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return np.abs(delta)


def action_acceleration_stats(actions: np.ndarray) -> tuple[float, float]:
    if len(actions) < 3:
        return 0.0, 0.0
    velocity = np.diff(actions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    norms = np.linalg.norm(acceleration, axis=1)
    return float(np.mean(norms)), float(np.max(norms))


def grasp_signal_index(grasp_signal: np.ndarray) -> int | None:
    signal = grasp_signal.squeeze()
    indices = np.where(signal > 0.5)[0]
    return int(indices[0]) if len(indices) else None


def extract_demo_features(
    demo_grp: h5py.Group,
    demo_key: str,
    label: str,
    source_file: str,
) -> NutAssemblyFeatures:
    actions = demo_grp["actions"][:]
    nut = demo_grp["datagen_info/object_poses/square_nut"][:]
    peg = demo_grp["datagen_info/object_poses/square_peg"][:]
    grasp = demo_grp["datagen_info/subtask_term_signals/grasp"][:]

    nut_pos = nut[:, :3, 3]
    peg_pos = peg[:, :3, 3]
    xy_distance = np.linalg.norm(nut_pos[:, :2] - peg_pos[:, :2], axis=1)
    z_difference = nut_pos[:, 2] - peg_pos[:, 2]
    yaw_error = square_yaw_error(nut[:, :3, :3], peg[:, :3, :3])

    acc_mean, acc_max = action_acceleration_stats(actions)

    return NutAssemblyFeatures(
        demo_key=demo_key,
        label=label,
        source_file=source_file,
        length=int(actions.shape[0]),
        final_nut_peg_xy_distance=float(xy_distance[-1]),
        min_nut_peg_xy_distance=float(xy_distance.min()),
        final_nut_peg_z_difference=float(z_difference[-1]),
        min_nut_peg_yaw_error=float(yaw_error.min()),
        final_nut_peg_yaw_error=float(yaw_error[-1]),
        action_acceleration_mean=acc_mean,
        action_acceleration_max=acc_max,
        grasp_signal_index=grasp_signal_index(grasp),
    )


def load_features_from_hdf5(path: str, label: str) -> list[NutAssemblyFeatures]:
    features: list[NutAssemblyFeatures] = []
    with h5py.File(path, "r") as handle:
        for demo_key in list_demo_keys(handle["data"]):
            demo_grp = handle[f"data/{demo_key}"]
            features.append(extract_demo_features(demo_grp, demo_key, label, path))
    return features
