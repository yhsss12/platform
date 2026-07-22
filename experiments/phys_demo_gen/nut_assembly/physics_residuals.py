"""Nut Assembly 独立 physics residual 层（8 类物理残差，不覆盖 aligned-original 模型）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from energy_model import TRANSPORT_THRESHOLD, XY_THRESHOLD

RESIDUAL_KEYS = (
    "E_transport",
    "E_xy",
    "E_lift",
    "E_contact",
    "E_bilateral",
    "E_dynamics",
    "E_slip",
    "E_coupling",
)

# 优先级加权：P1 > P2 > P3 > P4（total_score 为加权和，越小越好）
DEFAULT_RESIDUAL_WEIGHTS: dict[str, float] = {
    "E_transport": 5.0,
    "E_xy": 5.0,
    "E_lift": 3.0,
    "E_contact": 2.0,
    "E_bilateral": 2.0,
    "E_dynamics": 1.0,
    "E_slip": 1.0,
    "E_coupling": 1.0,
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    "success_lift_min": 0.005,
    "success_lift_p50": 0.039,
    "success_xy_threshold": float(XY_THRESHOLD),
    "success_transport_threshold": float(TRANSPORT_THRESHOLD),
    "success_contact_threshold": 0.012,
    "success_bilateral_steps_min": 3.0,
    "success_coupling_min": 0.35,
    "success_slip_max": 0.02,
    "dynamics_delta_ref": 0.03,
}

LIFT_REGRESSION_TOLERANCE = 0.15
CONTACT_DIST_FALLBACK = 0.012
RANKING_NORM_CAP = 2.0

P1P2_GATE_KEYS = ("E_transport", "E_xy", "E_lift")
P1P2_WEIGHT_KEYS = ("E_transport", "E_xy", "E_lift")


def _get_float(traj: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        if key in traj and traj[key] is not None:
            return float(traj[key])
    return default


def _min_peg_xy(traj: dict[str, Any]) -> float:
    val = _get_float(
        traj,
        "min_nut_peg_xy",
        "min_nut_peg_xy_distance",
        default=None,
    )
    if val is not None:
        return val
    return float(_get_float(traj, "final_nut_peg_xy", "final_nut_peg_xy_distance", default=0.33) or 0.33)


def _final_peg_xy(traj: dict[str, Any]) -> float:
    val = _get_float(
        traj,
        "final_nut_peg_xy",
        "final_nut_peg_xy_distance",
        default=None,
    )
    if val is not None:
        return val
    return _min_peg_xy(traj)


def _nut_lift_delta(traj: dict[str, Any]) -> float:
    return float(
        _get_float(
            traj,
            "nut_z_lift_delta",
            "nut_lift_delta",
            "nut_lift_phase_delta",
            default=0.0,
        )
        or 0.0
    )


def _residual_entry(raw: float, normalized: float, *, source: str) -> dict[str, Any]:
    return {"raw": float(raw), "normalized": float(normalized), "source": source}


def calibrate_thresholds_from_success_demos(success_trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """从 success demo rollout 统计基准阈值。"""
    if not success_trajectories:
        return dict(DEFAULT_THRESHOLDS)

    lifts = [_nut_lift_delta(t) for t in success_trajectories]
    final_xys = [_final_peg_xy(t) for t in success_trajectories]
    min_xys = [_min_peg_xy(t) for t in success_trajectories]
    left_contacts = [
        float(_get_float(t, "left_finger_contact_count", default=0.0) or 0.0) for t in success_trajectories
    ]
    right_contacts = [
        float(_get_float(t, "right_finger_contact_count", default=0.0) or 0.0) for t in success_trajectories
    ]
    bilateral = [
        float(_get_float(t, "bilateral_contact_steps", default=0.0) or 0.0) for t in success_trajectories
    ]
    couplings = [
        float(_get_float(t, "nut_eef_coupling_ratio", default=0.0) or 0.0) for t in success_trajectories
    ]
    slips = [float(_get_float(t, "nut_xy_slip", default=0.0) or 0.0) for t in success_trajectories]
    eef_nut = [
        float(_get_float(t, "eef_nut_distance_at_grasp", "min_eef_nut_distance", default=0.05) or 0.05)
        for t in success_trajectories
    ]

    lift_arr = np.asarray(lifts, dtype=float)
    return {
        "success_lift_min": float(max(0.002, np.percentile(lift_arr, 10))),
        "success_lift_p50": float(np.percentile(lift_arr, 50)),
        "success_xy_threshold": float(max(0.01, np.percentile(final_xys, 95))),
        "success_transport_threshold": float(max(0.01, np.percentile(min_xys, 95))),
        "success_contact_threshold": float(max(0.008, np.percentile(eef_nut, 90))),
        "success_bilateral_steps_min": float(max(1.0, np.percentile(bilateral, 10))),
        "success_coupling_min": float(max(0.15, np.percentile(couplings, 25))),
        "success_slip_max": float(max(0.01, np.percentile(slips, 90))),
        "dynamics_delta_ref": float(DEFAULT_THRESHOLDS["dynamics_delta_ref"]),
        "num_success_demos": len(success_trajectories),
    }


def load_success_trajectories_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            rollout = rec.get("rollout", rec)
            if rollout:
                rows.append(rollout)
    return rows


def resolve_thresholds(context: dict[str, Any] | None) -> dict[str, float]:
    ctx = context or {}
    merged = dict(DEFAULT_THRESHOLDS)
    if isinstance(ctx.get("thresholds"), dict):
        merged.update({k: float(v) for k, v in ctx["thresholds"].items() if k in merged or k.startswith("success_")})
    if isinstance(ctx.get("physics_thresholds"), dict):
        merged.update({k: float(v) for k, v in ctx["physics_thresholds"].items()})
    return merged


def _contact_metrics(traj: dict[str, Any], thresholds: dict[str, float]) -> tuple[float, float, str, str]:
    """返回 (contact_score, bilateral_score, contact_source, bilateral_source)。"""
    left = int(_get_float(traj, "left_finger_contact_count", default=0.0) or 0)
    right = int(_get_float(traj, "right_finger_contact_count", default=0.0) or 0)
    bilateral_steps = int(_get_float(traj, "bilateral_contact_steps", default=0.0) or 0)
    contact_duration = int(_get_float(traj, "contact_duration", default=0.0) or 0)

    has_contact_flags = left > 0 or right > 0 or bilateral_steps > 0 or contact_duration > 0
    if has_contact_flags:
        contact_score = float(min(1.0, (left + right) / max(20.0, contact_duration + 1.0)))
        bilateral_target = max(thresholds["success_bilateral_steps_min"], 1.0)
        bilateral_score = float(min(1.0, bilateral_steps / bilateral_target))
        return contact_score, bilateral_score, "measured", "measured"

    # fallback: nut 到左右 gripper / eef 的距离近似
    eef_dist = _get_float(
        traj,
        "eef_nut_distance_at_grasp",
        "min_eef_nut_distance",
        default=None,
    )
    left_dist = _get_float(traj, "left_finger_nut_distance", default=None)
    right_dist = _get_float(traj, "right_finger_nut_distance", default=None)
    if left_dist is None or right_dist is None:
        if eef_dist is not None:
            left_dist = right_dist = float(eef_dist)
        else:
            left_dist = right_dist = float(_final_peg_xy(traj))

    contact_thresh = thresholds["success_contact_threshold"]
    left_ok = float(left_dist < contact_thresh)
    right_ok = float(right_dist < contact_thresh)
    contact_score = float(0.5 * (left_ok + right_ok))
    bilateral_score = float(left_ok * right_ok)
    return contact_score, bilateral_score, "fallback", "fallback"


def _dynamics_metrics(traj: dict[str, Any], thresholds: dict[str, float]) -> tuple[float, str]:
    nut_delta = np.array(
        [
            float(_get_float(traj, "nut_dx", default=0.0) or 0.0),
            float(_get_float(traj, "nut_dy", default=0.0) or 0.0),
            _nut_lift_delta(traj),
        ],
        dtype=float,
    )
    grip_delta = np.array(
        [
            float(_get_float(traj, "eef_dx", "gripper_dx", default=0.0) or 0.0),
            float(_get_float(traj, "eef_dy", "gripper_dy", default=0.0) or 0.0),
            float(_get_float(traj, "eef_z_lift_delta", default=0.0) or 0.0),
        ],
        dtype=float,
    )
    if np.linalg.norm(grip_delta) > 1e-6 or np.linalg.norm(nut_delta) > 1e-6:
        ref = max(thresholds["dynamics_delta_ref"], 1e-6)
        mismatch = float(np.linalg.norm(nut_delta - grip_delta) / ref)
        return min(1.0, mismatch), "measured"

    z_std = float(_get_float(traj, "nut_z_std_during_lift", default=0.0) or 0.0)
    stability = float(_get_float(traj, "E_lift_stability", default=0.0) or 0.0)
    fallback = min(1.0, z_std / 0.02 + stability * 0.5)
    return fallback, "fallback"


def compute_physics_residuals(
    trajectory: dict[str, Any],
    context: dict[str, Any] | None = None,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    计算 8 类 physics residual。

    返回:
      - residuals: 每项含 raw / normalized / source
      - total_score: 加权归一化残差（越小越好）
      - thresholds_used
    """
    traj = trajectory or {}
    ctx = context or {}
    thresholds = resolve_thresholds(ctx)
    w = {**DEFAULT_RESIDUAL_WEIGHTS, **(weights or {})}

    min_xy = _min_peg_xy(traj)
    final_xy = _final_peg_xy(traj)
    lift_delta = _nut_lift_delta(traj)

    transport_norm = min_xy / max(thresholds["success_transport_threshold"], 1e-6)
    transport_source = "measured" if _get_float(traj, "min_nut_peg_xy", "min_nut_peg_xy_distance") is not None else "fallback"
    if transport_source == "fallback" and _get_float(traj, "E_transport_norm") is not None:
        transport_norm = float(traj["E_transport_norm"])
        transport_source = "fallback"

    xy_norm = final_xy / max(thresholds["success_xy_threshold"], 1e-6)
    xy_source = "measured" if _get_float(traj, "final_nut_peg_xy", "final_nut_peg_xy_distance") is not None else "fallback"
    if xy_source == "fallback" and _get_float(traj, "E_xy_norm") is not None:
        xy_norm = float(traj["E_xy_norm"])
        xy_source = "fallback"

    lift_min = thresholds["success_lift_min"]
    lift_p50 = thresholds["success_lift_p50"]
    lift_shortfall = max(0.0, lift_min - lift_delta) / max(lift_min, 1e-6)
    lift_norm_ref = max(0.0, 1.0 - min(1.0, lift_delta / max(lift_p50, 1e-6)))
    lift_norm = float(0.55 * min(1.0, lift_shortfall) + 0.45 * lift_norm_ref)
    lift_source = "measured" if lift_delta > 0.0 or _get_float(traj, "nut_z_lift_delta", "nut_lift_delta") is not None else "fallback"

    contact_score, bilateral_score, contact_source, bilateral_source = _contact_metrics(traj, thresholds)
    contact_norm = float(1.0 - contact_score)
    bilateral_norm = float(1.0 - bilateral_score)

    slip_raw = float(_get_float(traj, "nut_xy_slip", default=0.0) or 0.0)
    slip_norm = slip_raw / max(thresholds["success_slip_max"], 1e-6)
    slip_source = "measured" if "nut_xy_slip" in traj else "fallback"

    coupling_raw = float(_get_float(traj, "nut_eef_coupling_ratio", default=0.0) or 0.0)
    if "nut_eef_coupling_ratio" in traj:
        coupling_norm = float(max(0.0, thresholds["success_coupling_min"] - max(0.0, coupling_raw)) / max(thresholds["success_coupling_min"], 1e-6))
        coupling_source = "measured"
    else:
        follow = float(_get_float(traj, "lift_follow_score", default=0.0) or 0.0)
        coupling_norm = float(1.0 - np.clip(follow, 0.0, 1.0))
        coupling_source = "fallback"

    dynamics_norm, dynamics_source = _dynamics_metrics(traj, thresholds)

    residuals = {
        "E_transport": _residual_entry(min_xy, transport_norm, source=transport_source),
        "E_xy": _residual_entry(final_xy, xy_norm, source=xy_source),
        "E_lift": _residual_entry(lift_delta, lift_norm, source=lift_source),
        "E_contact": _residual_entry(1.0 - contact_score, contact_norm, source=contact_source),
        "E_bilateral": _residual_entry(1.0 - bilateral_score, bilateral_norm, source=bilateral_source),
        "E_dynamics": _residual_entry(dynamics_norm, dynamics_norm, source=dynamics_source),
        "E_slip": _residual_entry(slip_raw, slip_norm, source=slip_source),
        "E_coupling": _residual_entry(coupling_raw, coupling_norm, source=coupling_source),
    }

    raw_total_score = float(
        sum(w[key] * residuals[key]["normalized"] for key in RESIDUAL_KEYS)
        / max(sum(w.values()), 1e-6)
    )
    capped_norms = {
        key: float(min(RANKING_NORM_CAP, residuals[key]["normalized"])) for key in RESIDUAL_KEYS
    }
    ranking_score = float(
        sum(w[key] * capped_norms[key] for key in RESIDUAL_KEYS) / max(sum(w.values()), 1e-6)
    )

    return {
        "residuals": residuals,
        "total_score": raw_total_score,
        "raw_total_score": raw_total_score,
        "ranking_score": ranking_score,
        "ranking_norm_cap": RANKING_NORM_CAP,
        "weights": w,
        "thresholds_used": thresholds,
        "priority_tiers": {
            "P1": ["E_transport", "E_xy"],
            "P2": ["E_lift"],
            "P3": ["E_contact", "E_bilateral"],
            "P4": ["E_dynamics", "E_slip", "E_coupling"],
        },
    }


def source_consistent_keys(
    candidate_breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
) -> list[str]:
    """original 与 candidate source 一致的 residual 键。"""
    src = check_source_consistency(candidate_breakdown, original_breakdown)
    return [k for k in RESIDUAL_KEYS if src["per_residual"].get(k, False)]


def compute_effective_ranking_score(
    candidate_breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[float, list[str], dict[str, Any]]:
    """
    仅 source 一致的 residual 参与 ranking_score。
    不一致项保留在 raw breakdown，但不计入 effective ranking。
    """
    w = {**DEFAULT_RESIDUAL_WEIGHTS, **(weights or {})}
    keys = source_consistent_keys(candidate_breakdown, original_breakdown)
    if not keys:
        return float(candidate_breakdown["ranking_score"]), [], {
            "effective_keys": [],
            "excluded_keys": list(RESIDUAL_KEYS),
            "fallback_to_full_ranking": True,
        }
    capped = {
        k: float(min(RANKING_NORM_CAP, candidate_breakdown["residuals"][k]["normalized"]))
        for k in keys
    }
    denom = sum(w[k] for k in keys)
    score = float(sum(w[k] * capped[k] for k in keys) / max(denom, 1e-6))
    excluded = [k for k in RESIDUAL_KEYS if k not in keys]
    return score, keys, {
        "effective_keys": keys,
        "excluded_keys": excluded,
        "fallback_to_full_ranking": False,
    }


def compute_effective_p1p2_score(
    candidate_breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
) -> tuple[float, list[str]]:
    keys = [k for k in P1P2_WEIGHT_KEYS if k in source_consistent_keys(candidate_breakdown, original_breakdown)]
    if not keys:
        return compute_p1p2_total_score(candidate_breakdown), []
    w = DEFAULT_RESIDUAL_WEIGHTS
    denom = sum(w[k] for k in keys)
    total = sum(
        w[k] * min(RANKING_NORM_CAP, candidate_breakdown["residuals"][k]["normalized"]) for k in keys
    )
    return float(total / max(denom, 1e-6)), keys


def candidate_beats_original(
    candidate: dict[str, Any],
    original: dict[str, Any],
    *,
    lift_tolerance: float = LIFT_REGRESSION_TOLERANCE,
    respect_source_consistency: bool = True,
) -> tuple[bool, dict[str, bool]]:
    """候选接受条件：transport/xy/total 改善，lift 不明显变差。"""
    src = check_source_consistency(candidate, original)
    checks: dict[str, bool] = {}

    if respect_source_consistency and not src["per_residual"].get("E_transport", True):
        checks["E_transport_skipped_source_mismatch"] = True
    else:
        checks["E_transport_improved"] = candidate["residuals"]["E_transport"]["normalized"] < original["residuals"]["E_transport"]["normalized"]

    if respect_source_consistency and not src["per_residual"].get("E_xy", True):
        checks["E_xy_skipped_source_mismatch"] = True
    else:
        checks["E_xy_improved"] = candidate["residuals"]["E_xy"]["normalized"] < original["residuals"]["E_xy"]["normalized"]

    if respect_source_consistency and not src["per_residual"].get("E_lift", True):
        checks["E_lift_skipped_source_mismatch"] = True
    else:
        checks["E_lift_not_worse"] = (
            candidate["residuals"]["E_lift"]["normalized"]
            <= original["residuals"]["E_lift"]["normalized"] + lift_tolerance
        )

    eff_score, eff_keys, _ = compute_effective_ranking_score(candidate, original)
    orig_eff, _, _ = compute_effective_ranking_score(original, original)
    checks["total_score_improved"] = eff_score < orig_eff if eff_keys else candidate["raw_total_score"] < original["raw_total_score"]

    required = [k for k in checks if k.endswith("_improved") or k.endswith("_not_worse")]
    passed = all(checks[k] for k in required)
    return passed, checks


def compute_p1p2_total_score(
    breakdown: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
    use_ranking_cap: bool = True,
) -> float:
    """仅 P1/P2 residual 的加权分（越小越好）。"""
    w = {**DEFAULT_RESIDUAL_WEIGHTS, **(weights or {})}
    denom = sum(w[k] for k in P1P2_WEIGHT_KEYS)
    total = 0.0
    for key in P1P2_WEIGHT_KEYS:
        norm = breakdown["residuals"][key]["normalized"]
        if use_ranking_cap:
            norm = min(RANKING_NORM_CAP, norm)
        total += w[key] * norm
    return float(total / max(denom, 1e-6))


def candidate_passes_p1p2_gate(
    candidate: dict[str, Any],
    original: dict[str, Any],
    *,
    source_consistency: dict[str, Any] | None = None,
    lift_tolerance: float = LIFT_REGRESSION_TOLERANCE,
) -> tuple[bool, dict[str, bool]]:
    """
    demo_3 诊断 gate：仅 E_transport / E_xy / E_lift + P1/P2 total_score。
    source 不一致的 residual 不参与 gate。
    """
    src = source_consistency or check_source_consistency(candidate, original)
    per_src = src["per_residual"]
    checks: dict[str, bool] = {}

    for key in ("E_transport", "E_xy"):
        if not per_src.get(key, True):
            checks[f"{key}_skipped_source_mismatch"] = True
            continue
        checks[f"{key}_improved"] = (
            candidate["residuals"][key]["normalized"] < original["residuals"][key]["normalized"]
        )

    if per_src.get("E_lift", True):
        checks["E_lift_not_worse"] = (
            candidate["residuals"]["E_lift"]["normalized"]
            <= original["residuals"]["E_lift"]["normalized"] + lift_tolerance
        )
    else:
        checks["E_lift_skipped_source_mismatch"] = True

    _, eff_p1p2_keys = compute_effective_p1p2_score(candidate, original)
    if eff_p1p2_keys:
        checks["p1p2_total_score_improved"] = compute_effective_p1p2_score(candidate, original)[0] < compute_effective_p1p2_score(original, original)[0]
    else:
        checks["p1p2_total_score_improved"] = compute_p1p2_total_score(candidate) < compute_p1p2_total_score(original)

    required = [k for k in checks if k.endswith("_improved") or k.endswith("_not_worse")]
    passed = all(checks[k] for k in required)
    return passed, checks


def check_source_consistency(
    candidate_breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
) -> dict[str, Any]:
    """比较 candidate 与 original 各 residual 的 source 是否一致。"""
    per_key: dict[str, bool] = {}
    mismatches: list[str] = []
    for key in RESIDUAL_KEYS:
        c_src = candidate_breakdown["residuals"][key]["source"]
        o_src = original_breakdown["residuals"][key]["source"]
        ok = c_src == o_src
        per_key[key] = ok
        if not ok:
            mismatches.append(key)
    return {
        "source_consistent": len(mismatches) == 0,
        "per_residual": per_key,
        "mismatched_keys": mismatches,
    }


def residual_delta_metrics(
    candidate_breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
) -> dict[str, float]:
    """各 residual normalized 相对 original 的 delta（负值=改善）。"""
    return {
        key: float(
            candidate_breakdown["residuals"][key]["normalized"]
            - original_breakdown["residuals"][key]["normalized"]
        )
        for key in RESIDUAL_KEYS
    }


def fallback_rate(breakdown: dict[str, Any]) -> float:
    residuals = breakdown.get("residuals", {})
    if not residuals:
        return 0.0
    n_fallback = sum(1 for key in RESIDUAL_KEYS if residuals[key].get("source") == "fallback")
    return float(n_fallback / len(RESIDUAL_KEYS))


def format_breakdown_record(
    *,
    label: str,
    demo_key: str,
    trajectory: dict[str, Any],
    breakdown: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成 residual_breakdown.json 单条记录。"""
    row: dict[str, Any] = {
        "label": label,
        "demo_key": demo_key,
        "total_score": breakdown["total_score"],
        "raw_total_score": breakdown.get("raw_total_score", breakdown["total_score"]),
        "ranking_score": breakdown.get("ranking_score", breakdown["total_score"]),
        "residuals": breakdown["residuals"],
        "fallback_rate": fallback_rate(breakdown),
        "success_flag": bool(trajectory.get("success_flag")),
        "nut_z_lift_delta": _nut_lift_delta(trajectory),
        "min_nut_peg_xy": _min_peg_xy(trajectory),
        "final_nut_peg_xy": _final_peg_xy(trajectory),
    }
    if extra:
        row.update(extra)
    return row
