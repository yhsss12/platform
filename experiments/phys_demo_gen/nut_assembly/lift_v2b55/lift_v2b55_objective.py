"""V2-B5.5 pre-lift reclose + slow vertical lift objective."""
from __future__ import annotations

from typing import Any

PARTIAL_THRESH = 0.005
WEAK_THRESH = 0.002
LIFT_CEILING_BREAK = 0.0035
SUCCESS_LIFT_P50 = 0.039
DEMO3_BASELINE_PEG_XY = 0.33
B54_MAX_LIFT = 0.0022625020316447975
B54_MIN_FINAL_PEG = 0.26675912166458693
MAX_PEG_REGRESSION = 0.35
MAX_SLIP_DEFAULT = 0.05

W_P1_LIFT = 8.0
W_P2_PEG = 2.5
W_P3_COUPLING = 2.0
W_P3_STABILITY = 1.0

P0_FAIL = 2500.0
PEG_REGRESSION_PENALTY = 800.0
SLIP_PENALTY = 400.0

CSV_COLUMNS = [
    "E_lift", "nut_z_lift_delta", "final_nut_peg_xy", "min_nut_peg_xy",
    "nut_xy_slip", "nut_eef_coupling_ratio",
    "right_finger_contact_count", "bilateral_contact_steps",
    "prelift_slow_lift_score", "p0_gate_pass", "elite_eligible", "weak_lift_positive",
]


def nut_z_delta(rec: dict[str, Any]) -> float:
    return float(rec.get("nut_z_lift_delta", rec.get("nut_lift_phase_delta", 0.0)))


def peg_xy(rec: dict[str, Any]) -> float:
    return float(rec.get("final_nut_peg_xy", DEMO3_BASELINE_PEG_XY))


def min_peg_xy(rec: dict[str, Any]) -> float:
    return float(rec.get("min_nut_peg_xy", peg_xy(rec)))


def nut_xy_slip(rec: dict[str, Any]) -> float:
    return float(rec.get("nut_xy_slip", 0.0))


def partial_lift_success(rec: dict[str, Any]) -> bool:
    return bool(rec.get("partial_lift_success")) or nut_z_delta(rec) >= PARTIAL_THRESH


def weak_lift_positive(rec: dict[str, Any]) -> bool:
    return nut_z_delta(rec) >= WEAK_THRESH


def has_bilateral_contact(rec: dict[str, Any]) -> bool:
    return int(rec.get("right_finger_contact_count", 0)) > 0 and int(rec.get("bilateral_contact_steps", 0)) > 0


def slip_ok(rec: dict[str, Any], *, max_slip: float = MAX_SLIP_DEFAULT) -> bool:
    return nut_xy_slip(rec) <= max_slip


def p0_gate_pass(rec: dict[str, Any]) -> bool:
    if rec.get("rollout_timeout") or rec.get("rollout_error"):
        return False
    if not has_bilateral_contact(rec):
        return False
    if not slip_ok(rec):
        return False
    return True


def is_elite_eligible(rec: dict[str, Any]) -> bool:
    return p0_gate_pass(rec) and weak_lift_positive(rec)


def compute_prelift_slow_lift_score(rec: dict[str, Any]) -> dict[str, Any]:
    nz = nut_z_delta(rec)
    final_peg = peg_xy(rec)
    slip = nut_xy_slip(rec)
    coupling = max(0.0, float(rec.get("nut_eef_coupling_ratio", 0.0)))
    lift_norm = min(1.0, nz / SUCCESS_LIFT_P50)
    lift_gate = 1.0 if nz >= PARTIAL_THRESH else max(0.0, nz / PARTIAL_THRESH)
    e_lift = max(0.0, 1.0 - lift_norm * lift_gate)

    p1 = W_P1_LIFT * lift_norm * lift_gate + (0.8 if weak_lift_positive(rec) else 0.0)
    peg_score = max(0.0, 1.0 - final_peg / DEMO3_BASELINE_PEG_XY)
    p2 = W_P2_PEG * peg_score
    p3 = W_P3_COUPLING * min(1.0, coupling / 0.5)
    z_std = float(rec.get("nut_z_std_during_lift", 0.0))
    p3 -= W_P3_STABILITY * min(1.0, z_std / 0.02)

    hard = 0.0
    if not p0_gate_pass(rec):
        hard -= P0_FAIL
    if final_peg > MAX_PEG_REGRESSION and nz >= WEAK_THRESH:
        hard -= PEG_REGRESSION_PENALTY
    if slip > MAX_SLIP_DEFAULT:
        hard -= SLIP_PENALTY

    score = float(p1 + p2 + p3 + hard)
    return {
        "E_lift": float(e_lift),
        "prelift_slow_lift_score": score,
        "p0_gate_pass": p0_gate_pass(rec),
        "elite_eligible": is_elite_eligible(rec),
        "weak_lift_positive": weak_lift_positive(rec),
        "partial_lift_success": partial_lift_success(rec),
        "lift_norm": lift_norm,
        "lift_gate": lift_gate,
        "tier_P1_lift": p1,
        "tier_P2_peg": p2,
        "tier_P3_dynamics": p3,
        "hard_penalty": hard,
        "residual_breakdown": {
            "P1_lift": p1, "P2_peg": p2, "P3_dynamics": p3, "hard": hard,
        },
    }


def flatten_for_csv(rec: dict[str, Any]) -> dict[str, Any]:
    s = compute_prelift_slow_lift_score(rec) if "prelift_slow_lift_score" not in rec else rec
    row = {c: s.get(c, rec.get(c)) for c in CSV_COLUMNS}
    for k in ("cem_round", "cem_index", "outcome_label"):
        if k in rec:
            row[k] = rec[k]
    if isinstance(rec.get("lift_v2b55_params"), dict):
        row["lift_v2b55_params_json"] = __import__("json").dumps(rec["lift_v2b55_params"])
    return row
