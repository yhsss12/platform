"""Square_D0 / Nut Assembly V0.5 显式物理约束轨迹能量模型（含归一化能量，供 CEM 优化）。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from extract_features import NutAssemblyFeatures

# 审计标定参考值（raw energy）
SUCCESS_FINAL_XY_REF = 0.004
SUCCESS_FINAL_Z_REF = -0.021
SUCCESS_MIN_YAW_REF = 0.005
ON_PEG_XY_THRESH = 0.03
NEAR_PEG_XY_THRESH = 0.08
YAW_ALIGN_THRESH = 0.05
Z_INSERT_RISK_THRESH = 0.02

DEFAULT_WEIGHTS = {
    "w_xy": 3.0,
    "w_transport": 3.0,
    "w_yaw": 2.0,
    "w_z": 2.0,
    "w_smooth": 0.2,
}

# V0.5 归一化尺度（CEM 优化目标）
XY_THRESHOLD = 0.03
TRANSPORT_THRESHOLD = 0.03
YAW_THRESHOLD = 0.05
Z_SUCCESS_TARGET = -0.021
Z_TOLERANCE = 0.02
SMOOTH_THRESHOLD = 2.5

SMOOTHNESS_REF_MEAN = 0.45
SMOOTHNESS_REF_MAX = 2.2
SMOOTHNESS_ABNORMAL_FACTOR = 1.35

OPTIMIZATION_TARGETS: dict[str, list[str]] = {
    "transport_failed": ["transport_xy_offset", "pre_align_pose", "gripper_timing"],
    "alignment_failed": ["align_yaw", "pre_insert_pose"],
    "insertion_failed": ["insert_z", "insertion_speed", "release_timing"],
    "smoothness_issue": ["speed_scale", "waypoint_smoothing"],
    "success": [],
    "unknown_failed": ["transport_xy_offset", "insert_z", "align_yaw"],
}


@dataclass
class EnergyBreakdown:
    demo_key: str
    label: str
    length: int
    E_xy: float
    E_transport: float
    E_yaw: float
    E_z: float
    E_smooth: float
    E_total: float
    E_xy_norm: float
    E_transport_norm: float
    E_yaw_norm: float
    E_z_norm: float
    E_smooth_norm: float
    E_total_norm: float
    contribution_xy: float
    contribution_transport: float
    contribution_yaw: float
    contribution_z: float
    contribution_smooth: float
    failure_type: str
    source_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "demo_name": self.demo_key,
            "label": self.label,
            "length": self.length,
            "E_xy": self.E_xy,
            "E_transport": self.E_transport,
            "E_yaw": self.E_yaw,
            "E_z": self.E_z,
            "E_smooth": self.E_smooth,
            "E_total": self.E_total,
            "E_xy_norm": self.E_xy_norm,
            "E_transport_norm": self.E_transport_norm,
            "E_yaw_norm": self.E_yaw_norm,
            "E_z_norm": self.E_z_norm,
            "E_smooth_norm": self.E_smooth_norm,
            "E_total_norm": self.E_total_norm,
            "contribution_xy": self.contribution_xy,
            "contribution_transport": self.contribution_transport,
            "contribution_yaw": self.contribution_yaw,
            "contribution_z": self.contribution_z,
            "contribution_smooth": self.contribution_smooth,
            "failure_type": self.failure_type,
            "source_file": self.source_file,
        }


# ---------------------------------------------------------------------------
# Raw energy (V0)
# ---------------------------------------------------------------------------


def compute_xy_energy(final_xy_distance: float) -> float:
    baseline = max(SUCCESS_FINAL_XY_REF, 1e-6)
    excess = max(0.0, final_xy_distance - baseline)
    normalized = final_xy_distance / baseline
    return float(normalized**2 + (excess / 0.01) ** 2)


def compute_transport_energy(min_xy_distance: float) -> float:
    if min_xy_distance <= ON_PEG_XY_THRESH:
        ratio = min_xy_distance / ON_PEG_XY_THRESH
        return float(ratio**2)
    excess = min_xy_distance - ON_PEG_XY_THRESH
    return float(1.0 + (excess / 0.01) ** 2)


def compute_yaw_energy(min_yaw_error: float, final_yaw_error: float) -> float:
    min_term = (min_yaw_error / max(YAW_ALIGN_THRESH, 1e-6)) ** 2
    final_term = 0.25 * (final_yaw_error / max(YAW_ALIGN_THRESH, 1e-6)) ** 2
    return float(min_term + final_term)


def compute_z_insert_energy(final_z_difference: float) -> float:
    deviation = final_z_difference - SUCCESS_FINAL_Z_REF
    base = (deviation / 0.01) ** 2
    positive_penalty = 0.0
    if final_z_difference > Z_INSERT_RISK_THRESH:
        positive_penalty = 5.0 * ((final_z_difference - Z_INSERT_RISK_THRESH) / 0.01) ** 2
    return float(base + positive_penalty)


def compute_smoothness_energy(
    action_acceleration_mean: float,
    action_acceleration_max: float,
) -> float:
    mean_term = (action_acceleration_mean / SMOOTHNESS_REF_MEAN) ** 2
    max_term = 0.25 * (action_acceleration_max / SMOOTHNESS_REF_MAX) ** 2
    return float(mean_term + max_term)


# ---------------------------------------------------------------------------
# Normalized energy (V0.5 — CEM-ready)
# ---------------------------------------------------------------------------


def compute_normalized_xy_energy(final_xy_distance: float) -> float:
    return float(final_xy_distance / XY_THRESHOLD)


def compute_normalized_transport_energy(min_xy_distance: float) -> float:
    return float(min_xy_distance / TRANSPORT_THRESHOLD)


def compute_normalized_yaw_energy(min_yaw_error: float) -> float:
    return float(min_yaw_error / YAW_THRESHOLD)


def compute_normalized_z_energy(final_z_difference: float) -> float:
    return float(max(0.0, final_z_difference - Z_TOLERANCE) / Z_TOLERANCE)


def compute_normalized_smoothness_energy(action_acceleration_max: float) -> float:
    return float(action_acceleration_max / SMOOTH_THRESHOLD)


def compute_normalized_total_energy(
    e_xy_norm: float,
    e_transport_norm: float,
    e_yaw_norm: float,
    e_z_norm: float,
    e_smooth_norm: float,
    weights: dict[str, float] | None = None,
) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return float(
        w["w_xy"] * e_xy_norm
        + w["w_transport"] * e_transport_norm
        + w["w_yaw"] * e_yaw_norm
        + w["w_z"] * e_z_norm
        + w["w_smooth"] * e_smooth_norm
    )


def compute_contribution_ratios(
    e_xy_norm: float,
    e_transport_norm: float,
    e_yaw_norm: float,
    e_z_norm: float,
    e_smooth_norm: float,
    e_total_norm: float,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    weighted = {
        "contribution_xy": w["w_xy"] * e_xy_norm,
        "contribution_transport": w["w_transport"] * e_transport_norm,
        "contribution_yaw": w["w_yaw"] * e_yaw_norm,
        "contribution_z": w["w_z"] * e_z_norm,
        "contribution_smooth": w["w_smooth"] * e_smooth_norm,
    }
    denom = e_total_norm if e_total_norm > 1e-12 else sum(weighted.values())
    if denom <= 1e-12:
        return {key: 0.0 for key in weighted}
    return {key: float(value / denom) for key, value in weighted.items()}


# ---------------------------------------------------------------------------
# Classification & scoring
# ---------------------------------------------------------------------------


def classify_failure_type(
    features: NutAssemblyFeatures,
    smoothness_energy: float,
    smoothness_threshold: float | None = None,
) -> str:
    if features.label == "success":
        return "success"

    min_xy = features.min_nut_peg_xy_distance
    final_z = features.final_nut_peg_z_difference
    min_yaw = features.min_nut_peg_yaw_error

    smooth_thresh = smoothness_threshold
    if smooth_thresh is None:
        smooth_thresh = SMOOTHNESS_REF_MEAN**2 * SMOOTHNESS_ABNORMAL_FACTOR

    if min_xy <= NEAR_PEG_XY_THRESH and final_z > Z_INSERT_RISK_THRESH:
        return "insertion_failed"
    if min_xy > ON_PEG_XY_THRESH:
        return "transport_failed"
    if min_xy <= NEAR_PEG_XY_THRESH and min_yaw > YAW_ALIGN_THRESH:
        return "alignment_failed"
    if smoothness_energy > smooth_thresh:
        return "smoothness_issue"
    return "unknown_failed"


def get_optimization_targets(failure_type: str) -> list[str]:
    return list(OPTIMIZATION_TARGETS.get(failure_type, OPTIMIZATION_TARGETS["unknown_failed"]))


def _coerce_features(traj_features: NutAssemblyFeatures | dict[str, Any]) -> NutAssemblyFeatures:
    if isinstance(traj_features, NutAssemblyFeatures):
        return traj_features
    required = [
        "demo_key",
        "label",
        "source_file",
        "length",
        "final_nut_peg_xy_distance",
        "min_nut_peg_xy_distance",
        "final_nut_peg_z_difference",
        "min_nut_peg_yaw_error",
        "final_nut_peg_yaw_error",
        "action_acceleration_mean",
        "action_acceleration_max",
    ]
    payload = dict(traj_features)
    for key in required:
        if key not in payload:
            raise ValueError(f"score_candidate_trajectory missing field: {key}")
    if "grasp_signal_index" not in payload:
        payload["grasp_signal_index"] = None
    return NutAssemblyFeatures(**payload)


def score_candidate_trajectory(
    traj_features: NutAssemblyFeatures | dict[str, Any],
    smoothness_threshold: float | None = None,
) -> dict[str, Any]:
    """CEM 候选轨迹评分 API：返回归一化总能量、分项与优化目标。"""
    features = _coerce_features(traj_features)
    breakdown = compute_total_energy(features, smoothness_threshold=smoothness_threshold)
    return {
        "E_total_norm": breakdown.E_total_norm,
        "E_total_raw": breakdown.E_total,
        "components": {
            "xy": breakdown.E_xy_norm,
            "transport": breakdown.E_transport_norm,
            "yaw": breakdown.E_yaw_norm,
            "z": breakdown.E_z_norm,
            "smooth": breakdown.E_smooth_norm,
        },
        "contributions": {
            "xy": breakdown.contribution_xy,
            "transport": breakdown.contribution_transport,
            "yaw": breakdown.contribution_yaw,
            "z": breakdown.contribution_z,
            "smooth": breakdown.contribution_smooth,
        },
        "failure_type": breakdown.failure_type,
        "optimization_targets": get_optimization_targets(breakdown.failure_type),
    }


def compute_total_energy(
    features: NutAssemblyFeatures,
    weights: dict[str, float] | None = None,
    smoothness_threshold: float | None = None,
) -> EnergyBreakdown:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    e_xy = compute_xy_energy(features.final_nut_peg_xy_distance)
    e_transport = compute_transport_energy(features.min_nut_peg_xy_distance)
    e_yaw = compute_yaw_energy(features.min_nut_peg_yaw_error, features.final_nut_peg_yaw_error)
    e_z = compute_z_insert_energy(features.final_nut_peg_z_difference)
    e_smooth = compute_smoothness_energy(
        features.action_acceleration_mean,
        features.action_acceleration_max,
    )
    e_total = (
        w["w_xy"] * e_xy
        + w["w_transport"] * e_transport
        + w["w_yaw"] * e_yaw
        + w["w_z"] * e_z
        + w["w_smooth"] * e_smooth
    )

    e_xy_norm = compute_normalized_xy_energy(features.final_nut_peg_xy_distance)
    e_transport_norm = compute_normalized_transport_energy(features.min_nut_peg_xy_distance)
    e_yaw_norm = compute_normalized_yaw_energy(features.min_nut_peg_yaw_error)
    e_z_norm = compute_normalized_z_energy(features.final_nut_peg_z_difference)
    e_smooth_norm = compute_normalized_smoothness_energy(features.action_acceleration_max)
    e_total_norm = compute_normalized_total_energy(
        e_xy_norm, e_transport_norm, e_yaw_norm, e_z_norm, e_smooth_norm, weights=w
    )

    contributions = compute_contribution_ratios(
        e_xy_norm, e_transport_norm, e_yaw_norm, e_z_norm, e_smooth_norm, e_total_norm, weights=w
    )

    failure_type = classify_failure_type(features, e_smooth, smoothness_threshold)

    return EnergyBreakdown(
        demo_key=features.demo_key,
        label=features.label,
        length=features.length,
        E_xy=e_xy,
        E_transport=e_transport,
        E_yaw=e_yaw,
        E_z=e_z,
        E_smooth=e_smooth,
        E_total=float(e_total),
        E_xy_norm=e_xy_norm,
        E_transport_norm=e_transport_norm,
        E_yaw_norm=e_yaw_norm,
        E_z_norm=e_z_norm,
        E_smooth_norm=e_smooth_norm,
        E_total_norm=e_total_norm,
        contribution_xy=contributions["contribution_xy"],
        contribution_transport=contributions["contribution_transport"],
        contribution_yaw=contributions["contribution_yaw"],
        contribution_z=contributions["contribution_z"],
        contribution_smooth=contributions["contribution_smooth"],
        failure_type=failure_type,
        source_file=features.source_file,
    )


def clone_features_with_overrides(
    features: NutAssemblyFeatures,
    **overrides: Any,
) -> NutAssemblyFeatures:
    """用于 sensitivity check：在不跑仿真的情况下虚拟修正残差。"""
    return replace(features, **overrides)
