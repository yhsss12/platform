"""V2-B5.1：contact-aware lift 显式残差能量。"""
from __future__ import annotations

from typing import Any

import numpy as np


def _safe_ratio(value: float, ref: float, *, cap: float = 10.0) -> float:
    ref = max(ref, 1e-6)
    return float(min(cap, value / ref))


def compute_contact_aware_lift_energies(result: dict[str, Any]) -> dict[str, float]:
    """
    从 rollout + contact diagnostics 计算 contact-aware 残差（越小越好）。

    E_contact_presence: 无 finger-nut contact
    E_bilateral_contact: 无双侧 contact / bilateral steps 不足
    E_contact_duration: contact 持续时间过短
    E_lift_follow: nut 未跟随 eef 上升
    E_nut_eef_coupling: nut/eef lift 耦合比过低
    E_lift_stability: lift 高度不足 / xy slip 过大
    E_slow_lift_smoothness: lift 速度过快导致 slip
    """
    left_c = float(result.get("left_finger_contact_count", 0))
    right_c = float(result.get("right_finger_contact_count", 0))
    bilateral_steps = float(result.get("bilateral_contact_steps", 0))
    contact_duration = float(result.get("contact_duration", 0))
    nut_z_lift = float(result.get("nut_z_lift_delta", result.get("nut_lift_phase_delta", 0.0)))
    eef_z_lift = float(result.get("eef_z_lift_delta", 0.0))
    coupling = float(result.get("nut_eef_coupling_ratio", 0.0))
    nut_xy_slip = float(result.get("nut_xy_slip", 0.0))
    lift_follow = float(result.get("lift_follow_score", 0.0))
    target_lift = float(result.get("target_micro_lift_height", 0.06))
    lift_speed = float(result.get("lift_speed_scale", result.get("lift_v2b51_params", {}).get("lift_speed_scale", 1.0)))

    total_contact = left_c + right_c
    e_contact_presence = float(1.0 / (1.0 + total_contact))

    bilateral_target = max(3.0, contact_duration * 0.3)
    bilateral_shortfall = max(0.0, bilateral_target - bilateral_steps) / max(bilateral_target, 1e-6)
    e_bilateral_contact = float(bilateral_shortfall**2 + (0.0 if left_c > 0 and right_c > 0 else 1.0))

    duration_target = 8.0
    duration_shortfall = max(0.0, duration_target - contact_duration) / duration_target
    e_contact_duration = float(duration_shortfall**2)

    lift_shortfall = max(0.0, target_lift - nut_z_lift) / max(target_lift, 1e-6)
    e_lift_follow = float(lift_shortfall**2 + (1.0 - np.clip(lift_follow, 0.0, 1.0)) ** 2)

    coupling_target = 0.35
    coupling_shortfall = max(0.0, coupling_target - max(0.0, coupling)) / coupling_target
    e_nut_eef_coupling = float(coupling_shortfall**2)

    slip_penalty = _safe_ratio(nut_xy_slip, 0.02)
    stability_penalty = lift_shortfall + slip_penalty * 0.5
    e_lift_stability = float(stability_penalty**2)

    speed_excess = max(0.0, lift_speed - 0.45) / 0.55
    smoothness_penalty = speed_excess + _safe_ratio(nut_xy_slip, 0.015) * 0.5
    e_slow_lift_smoothness = float(smoothness_penalty**2)

    e_contact_total = float(
        0.20 * e_contact_presence
        + 0.20 * e_bilateral_contact
        + 0.10 * e_contact_duration
        + 0.15 * e_lift_follow
        + 0.15 * e_nut_eef_coupling
        + 0.10 * e_lift_stability
        + 0.10 * e_slow_lift_smoothness
    )

    return {
        "E_contact_presence": e_contact_presence,
        "E_bilateral_contact": e_bilateral_contact,
        "E_contact_duration": e_contact_duration,
        "E_lift_follow": e_lift_follow,
        "E_nut_eef_coupling": e_nut_eef_coupling,
        "E_lift_stability": e_lift_stability,
        "E_slow_lift_smoothness": e_slow_lift_smoothness,
        "E_contact_aware_total": e_contact_total,
    }
