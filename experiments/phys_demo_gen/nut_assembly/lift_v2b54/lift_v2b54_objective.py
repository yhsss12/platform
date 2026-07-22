"""V2-B5.4 lift-preserving transport objective (P0 gate + P1-P3)."""
from __future__ import annotations

from typing import Any

PARTIAL_THRESH = 0.005
WEAK_THRESH = 0.002
SUCCESS_LIFT_P50 = 0.039
DEMO3_BASELINE_PEG_XY = 0.33
B53_WEAK_COUNT = 2
TRANSPORT_THRESH = 0.03

W_P1_LIFT = 6.0
W_P2_TRANSPORT = 3.0
W_P2_XY = 3.0
W_P3_COUPLING = 1.5
W_P3_SLIP = 1.0
W_P3_STABILITY = 1.0

P0_FAIL_PENALTY = 2000.0
UNILATERAL_PENALTY = 1000.0

CSV_REQUIRED_COLUMNS = [
    "E_lift",
    "E_transport",
    "E_xy",
    "E_contact",
    "E_bilateral",
    "E_slip",
    "E_coupling",
    "final_nut_peg_xy",
    "min_nut_peg_xy",
    "nut_z_lift_delta",
    "right_finger_contact_count",
    "bilateral_contact_steps",
    "nut_eef_coupling_ratio",
    "lift_preserving_score",
    "p0_gate_pass",
    "elite_eligible",
    "transport_improved",
    "weak_lift_positive",
]


def nut_z_delta(rec: dict[str, Any]) -> float:
    return float(rec.get("nut_z_lift_delta", rec.get("nut_lift_phase_delta", 0.0)))


def peg_xy(rec: dict[str, Any]) -> float:
    return float(rec.get("final_nut_peg_xy", DEMO3_BASELINE_PEG_XY))


def min_peg_xy(rec: dict[str, Any]) -> float:
    return float(rec.get("min_nut_peg_xy", peg_xy(rec)))


def partial_lift_success(rec: dict[str, Any]) -> bool:
    return bool(rec.get("partial_lift_success")) or nut_z_delta(rec) >= PARTIAL_THRESH


def weak_lift_positive(rec: dict[str, Any]) -> bool:
    return nut_z_delta(rec) >= WEAK_THRESH


def transport_improved(rec: dict[str, Any], *, baseline: float = DEMO3_BASELINE_PEG_XY) -> bool:
    return min_peg_xy(rec) < baseline * 0.97 or peg_xy(rec) < baseline * 0.97


def has_bilateral_contact(rec: dict[str, Any]) -> bool:
    return int(rec.get("right_finger_contact_count", 0)) > 0 and int(rec.get("bilateral_contact_steps", 0)) > 0


def is_unilateral_lever_path(rec: dict[str, Any]) -> bool:
    return int(rec.get("right_finger_contact_count", 0)) <= 0 or int(rec.get("bilateral_contact_steps", 0)) <= 0


def p0_gate_pass(rec: dict[str, Any]) -> bool:
    if rec.get("rollout_timeout") or rec.get("rollout_error"):
        return False
    if is_unilateral_lever_path(rec):
        return False
    if not transport_improved(rec):
        return False
    return True


def is_elite_eligible(rec: dict[str, Any]) -> bool:
    """P0 + weak lift preferred for elite."""
    if not p0_gate_pass(rec):
        return False
    return weak_lift_positive(rec)


def _f(rec: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    if key in rec and rec[key] is not None:
        return float(rec[key])
    return fallback


def compute_lift_preserving_score(rec: dict[str, Any]) -> dict[str, Any]:
    nz = nut_z_delta(rec)
    e_transport = _f(rec, "E_transport_norm", min_peg_xy(rec) / TRANSPORT_THRESH)
    e_xy = _f(rec, "E_xy_norm", peg_xy(rec) / TRANSPORT_THRESH)
    lift_shortfall = max(0.0, PARTIAL_THRESH - nz) / PARTIAL_THRESH
    lift_norm = min(1.0, max(0.0, nz / SUCCESS_LIFT_P50))
    e_lift = float(0.55 * lift_shortfall + 0.45 * (1.0 - lift_norm))
    e_contact = _f(rec, "E_contact_presence", 1.0)
    e_bilateral = _f(rec, "E_bilateral_contact", 1.0)
    e_slip = min(1.0, float(rec.get("nut_xy_slip", 0.0)) / 0.05)
    e_coupling = _f(rec, "E_nut_eef_coupling", 1.0)
    e_stability = min(1.0, _f(rec, "E_lift_stability", 0.0) + float(rec.get("nut_z_std_during_lift", 0.0)) / 0.02)

    lift_gate = 1.0 if nz >= PARTIAL_THRESH else max(0.0, nz / PARTIAL_THRESH)
    weak_bonus = 0.5 if weak_lift_positive(rec) else 0.0

    p1 = W_P1_LIFT * lift_norm * lift_gate + weak_bonus
    p2 = W_P2_TRANSPORT * max(0.0, 1.0 - min(1.0, e_transport)) + W_P2_XY * max(0.0, 1.0 - min(1.0, e_xy))
    p3 = W_P3_COUPLING * max(0.0, 1.0 - min(1.0, e_coupling)) - W_P3_SLIP * e_slip - W_P3_STABILITY * e_stability

    total = p1 + p2 + p3
    hard = 0.0
    if not p0_gate_pass(rec):
        hard -= P0_FAIL_PENALTY
    if is_unilateral_lever_path(rec):
        hard -= UNILATERAL_PENALTY
    if p0_gate_pass(rec) and not weak_lift_positive(rec):
        hard -= 200.0

    score = float(total + hard)
    gate = p0_gate_pass(rec)

    breakdown = {
        "E_lift": e_lift,
        "E_transport": e_transport,
        "E_xy": e_xy,
        "E_contact": e_contact,
        "E_bilateral": e_bilateral,
        "E_slip": e_slip,
        "E_coupling": e_coupling,
        "E_stability": e_stability,
        "tier_P1_lift_reward": float(p1),
        "tier_P2_transport_reward": float(p2),
        "tier_P3_dynamics_reward": float(p3),
        "hard_penalty": float(hard),
        "lift_gate": float(lift_gate),
        "lift_norm_vs_p50": float(lift_norm),
    }

    return {
        **breakdown,
        "lift_preserving_score": score,
        "p0_gate_pass": gate,
        "elite_eligible": is_elite_eligible(rec),
        "transport_improved": transport_improved(rec),
        "weak_lift_positive": weak_lift_positive(rec),
        "partial_lift_success": partial_lift_success(rec),
        "is_unilateral_lever_path": is_unilateral_lever_path(rec),
        "residual_breakdown": breakdown,
    }


def flatten_for_csv(rec: dict[str, Any]) -> dict[str, Any]:
    scores = compute_lift_preserving_score(rec) if "lift_preserving_score" not in rec else rec
    row = {col: scores.get(col, rec.get(col)) for col in CSV_REQUIRED_COLUMNS}
    for key in ("cem_round", "cem_index", "outcome_label"):
        if key in rec:
            row[key] = rec[key]
    if isinstance(rec.get("lift_v2b54_params"), dict):
        row["lift_v2b54_params_json"] = __import__("json").dumps(rec["lift_v2b54_params"])
    return row
