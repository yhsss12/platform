"""V2-B5.3 contact-gated transport-and-lift objective (4-tier residual priority)."""
from __future__ import annotations

from typing import Any

PARTIAL_THRESH = 0.005
WEAK_THRESH = 0.002
SUCCESS_LIFT_P50 = 0.039
DEMO3_BASELINE_PEG_XY = 0.33
TRANSPORT_THRESH = 0.03

# Tier weights — P1 > P2 > P3 > P4
W_P1_TRANSPORT = 5.0
W_P1_XY = 5.0
W_P2_LIFT = 3.0
W_P3_CONTACT = 2.0
W_P3_BILATERAL = 2.0
W_P4_SLIP = 1.0
W_P4_COUPLING = 1.0
W_P4_DYNAMICS = 1.0

UNILATERAL_PENALTY = 1000.0
CONTACT_NO_TRANSPORT_PENALTY = 500.0
NO_TRANSPORT_ELITE_PENALTY = 300.0

CSV_REQUIRED_COLUMNS = [
    "E_transport",
    "E_xy",
    "E_lift",
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
    "transport_lift_score",
    "elite_eligible",
    "transport_improved",
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


def is_contact_rich_no_transport(rec: dict[str, Any]) -> bool:
    if not has_bilateral_contact(rec):
        return False
    rich = (
        int(rec.get("right_finger_contact_count", 0)) >= 30
        and int(rec.get("bilateral_contact_steps", 0)) >= 10
    )
    stagnant = min_peg_xy(rec) >= DEMO3_BASELINE_PEG_XY * 0.97 and nut_z_delta(rec) < WEAK_THRESH
    return rich and stagnant


def is_elite_eligible(rec: dict[str, Any]) -> bool:
    """Elite must pass P1 transport + P3 contact gates."""
    if rec.get("rollout_timeout") or rec.get("rollout_error"):
        return False
    if is_unilateral_lever_path(rec):
        return False
    if is_contact_rich_no_transport(rec):
        return False
    if not transport_improved(rec):
        return False
    return True


def _f(rec: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    if key in rec and rec[key] is not None:
        return float(rec[key])
    return fallback


def compute_residual_breakdown(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Four-tier residual breakdown (lower E_* is better).

    P1: E_transport, E_xy
    P2: E_lift (partial gate 0.005m, weak 0.002m, ref p50 0.039m)
    P3: E_contact, E_bilateral
    P4: E_slip, E_coupling, E_dynamics
    """
    nz = nut_z_delta(rec)
    final_xy = peg_xy(rec)
    min_xy = min_peg_xy(rec)

    e_transport = _f(rec, "E_transport_norm", min_xy / TRANSPORT_THRESH)
    e_xy = _f(rec, "E_xy_norm", final_xy / TRANSPORT_THRESH)

    lift_shortfall = max(0.0, PARTIAL_THRESH - nz) / PARTIAL_THRESH
    lift_norm_ref = max(0.0, 1.0 - min(1.0, nz / SUCCESS_LIFT_P50))
    e_lift_phys = _f(rec, "E_lift_norm", _f(rec, "E_lift_follow", lift_shortfall))
    e_lift = float(0.6 * lift_shortfall + 0.25 * lift_norm_ref + 0.15 * e_lift_phys)
    if is_unilateral_lever_path(rec) and nz >= WEAK_THRESH:
        e_lift += 0.5

    e_contact = _f(rec, "E_contact_presence", 1.0)
    e_bilateral = _f(rec, "E_bilateral_contact", 1.0)

    slip_raw = float(rec.get("nut_xy_slip", 0.0))
    e_slip = min(1.0, slip_raw / 0.05)
    e_coupling = _f(rec, "E_nut_eef_coupling", 1.0)
    z_std = float(rec.get("nut_z_std_during_lift", 0.0))
    e_dynamics = min(
        1.0,
        _f(rec, "E_lift_stability", 0.0) + _f(rec, "E_slow_lift_smoothness", 0.0) * 0.5 + z_std / 0.02 * 0.2,
    )

    p1_transport = W_P1_TRANSPORT * max(0.0, 1.0 - min(1.0, e_transport))
    p1_xy = W_P1_XY * max(0.0, 1.0 - min(1.0, e_xy))
    lift_gate = 1.0 if nz >= PARTIAL_THRESH else max(0.0, nz / PARTIAL_THRESH)
    weak_bonus = 0.15 if nz >= WEAK_THRESH else 0.0
    p2_lift = W_P2_LIFT * max(0.0, 1.0 - min(1.0, e_lift)) * lift_gate + weak_bonus

    right_count = int(rec.get("right_finger_contact_count", 0))
    bilateral_steps = int(rec.get("bilateral_contact_steps", 0))
    p3_contact = W_P3_CONTACT * max(0.0, 1.0 - min(1.0, e_contact)) * min(1.0, right_count / 40.0)
    p3_bilateral = W_P3_BILATERAL * max(0.0, 1.0 - min(1.0, e_bilateral)) * min(1.0, bilateral_steps / 25.0)

    p4_slip = -W_P4_SLIP * e_slip
    p4_coupling = W_P4_COUPLING * max(0.0, 1.0 - min(1.0, e_coupling))
    p4_dynamics = -W_P4_DYNAMICS * e_dynamics

    tier_total = p1_transport + p1_xy + p2_lift + p3_contact + p3_bilateral + p4_slip + p4_coupling + p4_dynamics

    hard_penalty = 0.0
    if is_unilateral_lever_path(rec):
        hard_penalty -= UNILATERAL_PENALTY
    elif is_contact_rich_no_transport(rec):
        hard_penalty -= CONTACT_NO_TRANSPORT_PENALTY
    if not transport_improved(rec):
        hard_penalty -= NO_TRANSPORT_ELITE_PENALTY

    transport_lift_score = float(tier_total + hard_penalty)
    elite_ok = is_elite_eligible(rec)

    breakdown = {
        "E_transport": float(e_transport),
        "E_xy": float(e_xy),
        "E_lift": float(e_lift),
        "E_contact": float(e_contact),
        "E_bilateral": float(e_bilateral),
        "E_slip": float(e_slip),
        "E_coupling": float(e_coupling),
        "E_dynamics": float(e_dynamics),
        "tier_P1_transport_reward": float(p1_transport),
        "tier_P1_xy_reward": float(p1_xy),
        "tier_P2_lift_reward": float(p2_lift),
        "tier_P3_contact_reward": float(p3_contact),
        "tier_P3_bilateral_reward": float(p3_bilateral),
        "tier_P4_slip_penalty": float(p4_slip),
        "tier_P4_coupling_reward": float(p4_coupling),
        "tier_P4_dynamics_penalty": float(p4_dynamics),
        "hard_penalty": float(hard_penalty),
        "lift_gate": float(lift_gate),
        "weak_lift_milestone": bool(nz >= WEAK_THRESH),
        "partial_lift_gate": bool(nz >= PARTIAL_THRESH),
    }

    return {
        **breakdown,
        "transport_lift_score": transport_lift_score,
        "residual_priority_total": transport_lift_score,
        "elite_eligible": elite_ok,
        "transport_improved": transport_improved(rec),
        "is_unilateral_lever_path": is_unilateral_lever_path(rec),
        "is_contact_rich_no_transport": is_contact_rich_no_transport(rec),
        "residual_breakdown": breakdown,
    }


def compute_transport_lift_score(rec: dict[str, Any]) -> dict[str, Any]:
    """Alias used by CEM loop / reports."""
    return compute_residual_breakdown(rec)


def flatten_for_csv(rec: dict[str, Any]) -> dict[str, Any]:
    """Ensure CSV row contains required residual + metric columns."""
    breakdown = compute_residual_breakdown(rec) if "E_transport" not in rec else rec
    row = {col: breakdown.get(col, rec.get(col)) for col in CSV_REQUIRED_COLUMNS}
    for key in ("cem_round", "cem_index", "outcome_label", "transport_lift_score"):
        if key in rec:
            row[key] = rec[key]
    if isinstance(rec.get("lift_v2b53_params"), dict):
        row["lift_v2b53_params_json"] = __import__("json").dumps(rec["lift_v2b53_params"])
    return row


def score_cem_candidate_with_optional_physics(
    rec: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CEM 评分：默认 v2b53 breakdown；enable_physics_residual_repair=true 时叠加 physics 层。"""
    import os

    base = compute_residual_breakdown(rec)
    if os.environ.get("enable_physics_residual_repair", "").strip().lower() != "true":
        return base
    try:
        import sys
        from pathlib import Path

        exp = Path(__file__).resolve().parents[1]
        if str(exp) not in sys.path:
            sys.path.insert(0, str(exp))
        from physics_residual_repair import cem_physics_residual_objective

        physics = cem_physics_residual_objective(rec, context or {})
        combined = 0.45 * base["transport_lift_score"] + 0.55 * physics["physics_cem_score"]
        return {
            **base,
            **physics,
            "transport_lift_score": float(combined),
            "residual_priority_total": float(combined),
            "physics_layer_enabled": True,
        }
    except ImportError:
        return base
