"""V1-F：lift_failed 专项显式残差能量（来自 MuJoCo rollout 指标）。"""
from __future__ import annotations

from typing import Any

import numpy as np


def _safe_ratio(value: float, ref: float, *, cap: float = 10.0) -> float:
    ref = max(ref, 1e-6)
    return float(min(cap, value / ref))


def compute_lift_residual_energies(result: dict[str, Any]) -> dict[str, float]:
    """
    从 rollout 结果计算 lift-aware 残差分量（越小越好）。

    E_lift_follow: nut 在 lift 窗口是否跟随 eef 上升
    E_grasp_contact: grasp 时 eef-nut 距离
    E_object_displacement: grasp 后 nut 位移不足
    E_eef_nut_coupling: lift 阶段 eef-nut 耦合距离
    E_lift_stability: lift 高度不足 / z 抖动
    """
    nut_lift_delta = float(result.get("nut_lift_delta", 0.0))
    target_lift = float(result.get("target_micro_lift_height", result.get("micro_lift_height", 0.06)))
    eef_nut_at_grasp = float(result.get("eef_nut_distance_at_grasp", 0.1))
    min_eef_nut = float(result.get("min_eef_nut_distance", eef_nut_at_grasp))
    nut_disp = float(result.get("nut_displacement_after_grasp", 0.0))
    lift_follow_score = float(result.get("lift_follow_score", 0.0))
    nut_z_std = float(result.get("nut_z_std_during_lift", 0.0))
    follow_thresh = float(result.get("nut_follow_threshold", 0.05))

    lift_shortfall = max(0.0, target_lift - nut_lift_delta) / max(target_lift, 1e-6)
    e_lift_follow = float(lift_shortfall**2 + (1.0 - np.clip(lift_follow_score, 0.0, 1.0)) ** 2)

    e_grasp_contact = _safe_ratio(eef_nut_at_grasp, follow_thresh)

    disp_shortfall = max(0.0, 0.03 - nut_disp) / 0.03
    e_object_displacement = float(disp_shortfall**2)

    coupling_excess = max(0.0, min_eef_nut - follow_thresh) / max(follow_thresh, 1e-6)
    e_eef_nut_coupling = float(coupling_excess**2)

    stability_penalty = nut_z_std / 0.01 + lift_shortfall
    e_lift_stability = float(stability_penalty**2)

    e_grasp = float(0.6 * e_grasp_contact + 0.4 * e_object_displacement)
    e_lift = float(0.5 * e_lift_follow + 0.3 * e_lift_stability + 0.2 * e_eef_nut_coupling)

    return {
        "E_lift_follow": e_lift_follow,
        "E_grasp_contact": e_grasp_contact,
        "E_object_displacement": e_object_displacement,
        "E_eef_nut_coupling": e_eef_nut_coupling,
        "E_lift_stability": e_lift_stability,
        "E_grasp_norm": e_grasp,
        "E_lift_norm": e_lift,
    }


def merge_v1f_energy_targets(result: dict[str, Any]) -> dict[str, float]:
    """合并标准 energy + lift residual，供 V1-F 数据集写入。"""
    lift = compute_lift_residual_energies(result)
    return {
        "rollout_E_xy_norm": float(result.get("E_xy_norm", 0.0)),
        "rollout_E_transport_norm": float(result.get("E_transport_norm", 0.0)),
        "rollout_E_yaw_norm": float(result.get("E_yaw_norm", 0.0)),
        "rollout_E_z_norm": float(result.get("E_z_norm", 0.0)),
        "rollout_E_smooth_norm": float(result.get("E_smooth_norm", 0.0)),
        "rollout_E_grasp_norm": lift["E_grasp_norm"],
        "rollout_E_lift_norm": lift["E_lift_norm"],
        "rollout_E_total_norm": float(result.get("E_total_norm", 0.0)),
        **{f"rollout_{k}": v for k, v in lift.items() if k.startswith("E_")},
    }
