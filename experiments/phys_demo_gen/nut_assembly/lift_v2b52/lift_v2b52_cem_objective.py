"""V2-B5.2 CEM multi-objective score。"""
from __future__ import annotations

from typing import Any

PARTIAL_THRESH = 0.005
WEAK_THRESH = 0.002

W_LIFT = 3.0
W_RIGHT = 1.5
W_BILATERAL = 2.0
W_COUPLING = 2.0
W_SLIP = 1.0
W_INSTABILITY = 1.0


def nut_z_delta(rec: dict[str, Any]) -> float:
    return float(rec.get("nut_z_lift_delta", rec.get("nut_lift_phase_delta", 0.0)))


def partial_lift_success(rec: dict[str, Any]) -> bool:
    return bool(rec.get("partial_lift_success")) or nut_z_delta(rec) >= PARTIAL_THRESH


def weak_lift_positive(rec: dict[str, Any]) -> bool:
    return nut_z_delta(rec) >= WEAK_THRESH


def compute_cem_score(rec: dict[str, Any]) -> dict[str, float]:
    nz = nut_z_delta(rec)
    norm_lift = max(0.0, min(1.0, nz / PARTIAL_THRESH))
    right_count = float(rec.get("right_finger_contact_count", 0))
    right_score = min(1.0, right_count / 40.0)
    bilateral = float(rec.get("bilateral_contact_steps", 0))
    bilateral_score = min(1.0, bilateral / 25.0)
    coupling = max(0.0, float(rec.get("nut_eef_coupling_ratio", 0.0)))
    coupling_score = min(1.0, coupling / 0.5)
    slip = float(rec.get("nut_xy_slip", 0.0))
    norm_slip = min(1.0, slip / 0.05)
    z_std = float(rec.get("nut_z_std_during_lift", 0.0))
    instability = min(1.0, z_std / 0.02 + max(0.0, -nz) / 0.01)

    total = (
        W_LIFT * norm_lift
        + W_RIGHT * right_score
        + W_BILATERAL * bilateral_score
        + W_COUPLING * coupling_score
        - W_SLIP * norm_slip
        - W_INSTABILITY * instability
    )
    return {
        "cem_score": float(total),
        "norm_lift": norm_lift,
        "right_score": right_score,
        "bilateral_score": bilateral_score,
        "coupling_score": coupling_score,
        "norm_slip": norm_slip,
        "instability_penalty": instability,
    }
