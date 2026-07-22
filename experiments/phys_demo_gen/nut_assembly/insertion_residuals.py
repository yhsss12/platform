"""demo_4 insertion 阶段专用 residual（仅诊断/排序，不进 V1-G 训练 loss）。"""
from __future__ import annotations

from typing import Any

import numpy as np

from energy_model import SUCCESS_FINAL_Z_REF, YAW_ALIGN_THRESH, Z_INSERT_RISK_THRESH
from physics_residuals import _final_peg_xy, _get_float, _min_peg_xy, _residual_entry, resolve_thresholds

INSERTION_RESIDUAL_KEYS = (
    "E_insert_depth",
    "E_axis_alignment",
    "E_vertical_approach",
    "E_final_pose",
    "E_jamming",
)

INSERTION_RANKING_WEIGHTS = {
    "E_insert_depth": 3.0,
    "E_axis_alignment": 2.5,
    "E_vertical_approach": 2.0,
    "E_final_pose": 3.0,
    "E_jamming": 1.5,
}

INSERTION_GATE_TOLERANCE = {
    "E_insert_depth": 0.08,
    "E_axis_alignment": 0.10,
    "E_vertical_approach": 0.12,
    "E_final_pose": 0.10,
    "E_jamming": 0.15,
}

DEFAULT_INSERTION_THRESHOLDS: dict[str, float] = {
    "insert_depth_max_norm": 0.35,
    "axis_alignment_max_norm": 0.40,
    "vertical_approach_max_norm": 0.30,
    "final_pose_max_norm": 0.45,
    "jamming_max_norm": 0.55,
    "jamming_regression_tol": 0.15,
}


def calibrate_insertion_thresholds_from_success_demos(
    success_trajectories: list[dict[str, Any]],
) -> dict[str, float]:
    """从 success demo 统计 insertion gate 阈值。"""
    if not success_trajectories:
        return dict(DEFAULT_INSERTION_THRESHOLDS)

    depth_norms: list[float] = []
    axis_norms: list[float] = []
    vertical_norms: list[float] = []
    pose_norms: list[float] = []
    jam_norms: list[float] = []

    for traj in success_trajectories:
        br = compute_insertion_residuals(traj, context=None)
        depth_norms.append(br["residuals"]["E_insert_depth"]["normalized"])
        axis_norms.append(br["residuals"]["E_axis_alignment"]["normalized"])
        vertical_norms.append(br["residuals"]["E_vertical_approach"]["normalized"])
        pose_norms.append(br["residuals"]["E_final_pose"]["normalized"])
        jam_norms.append(br["residuals"]["E_jamming"]["normalized"])

    depth_arr = np.asarray(depth_norms, dtype=float)
    axis_arr = np.asarray(axis_norms, dtype=float)
    vertical_arr = np.asarray(vertical_norms, dtype=float)
    pose_arr = np.asarray(pose_norms, dtype=float)
    jam_arr = np.asarray(jam_norms, dtype=float)

    return {
        "insert_depth_max_norm": float(max(0.15, np.percentile(depth_arr, 90))),
        "axis_alignment_max_norm": float(max(0.20, np.percentile(axis_arr, 95))),
        "vertical_approach_max_norm": float(max(0.15, np.percentile(vertical_arr, 95))),
        "final_pose_max_norm": float(max(0.20, np.percentile(pose_arr, 90))),
        "jamming_max_norm": float(max(0.30, np.percentile(jam_arr, 90))),
        "jamming_regression_tol": float(DEFAULT_INSERTION_THRESHOLDS["jamming_regression_tol"]),
        "num_success_demos": len(success_trajectories),
    }


def resolve_insertion_thresholds(context: dict[str, Any] | None) -> dict[str, float]:
    ctx = context or {}
    merged = dict(DEFAULT_INSERTION_THRESHOLDS)
    if isinstance(ctx.get("insertion_thresholds"), dict):
        merged.update({k: float(v) for k, v in ctx["insertion_thresholds"].items()})
    return merged


def check_insertion_source_consistency(
    candidate_breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
) -> dict[str, Any]:
    per_key: dict[str, bool] = {}
    mismatches: list[str] = []
    for key in INSERTION_RESIDUAL_KEYS:
        c_src = candidate_breakdown["residuals"][key]["source"]
        o_src = original_breakdown["residuals"][key]["source"]
        ok = c_src == o_src and c_src != "fallback"
        per_key[key] = ok
        if not ok:
            mismatches.append(key)
    return {
        "source_consistent": len(mismatches) == 0,
        "per_residual": per_key,
        "mismatched_keys": mismatches,
        "diagnostic_only_keys": [
            k
            for k in INSERTION_RESIDUAL_KEYS
            if candidate_breakdown["residuals"][k]["source"] == "fallback"
            or original_breakdown["residuals"][k]["source"] == "fallback"
            or not per_key.get(k, False)
        ],
    }


def candidate_passes_insertion_gate(
    candidate_breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
    *,
    insertion_thresholds: dict[str, float] | None = None,
    insertion_source_consistency: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    insertion 硬 gate：仅 source 一致且非 fallback 的 residual 参与。
    不一致或 fallback 项写入 diagnostic_only，不参与 pass/fail。
    """
    thresholds = insertion_thresholds or DEFAULT_INSERTION_THRESHOLDS
    src = insertion_source_consistency or check_insertion_source_consistency(
        candidate_breakdown, original_breakdown
    )
    per_src = src["per_residual"]
    checks: dict[str, Any] = {"diagnostic_only": list(src.get("diagnostic_only_keys", []))}

    gate_specs = (
        ("E_insert_depth", "insert_depth_max_norm", "at_or_below_success_trend"),
        ("E_axis_alignment", "axis_alignment_max_norm", "below_threshold"),
        ("E_vertical_approach", "vertical_approach_max_norm", "reasonable"),
        ("E_final_pose", "final_pose_max_norm", "near_success_reference"),
        ("E_jamming", "jamming_max_norm", "not_elevated"),
    )

    hard_checks: list[str] = []
    for key, thresh_key, check_kind in gate_specs:
        if not per_src.get(key, False):
            checks[f"{key}_skipped_unreliable"] = True
            continue

        c_norm = candidate_breakdown["residuals"][key]["normalized"]
        o_norm = original_breakdown["residuals"][key]["normalized"]
        thresh = thresholds[thresh_key]

        if key == "E_jamming":
            ok = c_norm <= max(thresh, o_norm + thresholds["jamming_regression_tol"])
            checks[f"{key}_not_elevated"] = ok
            checks[f"{key}_candidate_norm"] = c_norm
            checks[f"{key}_original_norm"] = o_norm
            hard_checks.append(f"{key}_not_elevated")
        elif key == "E_insert_depth":
            improved = c_norm < o_norm - INSERTION_GATE_TOLERANCE[key]
            at_trend = c_norm <= thresh
            ok = improved or at_trend
            checks[f"{key}_improved_or_at_trend"] = ok
            checks[f"{key}_candidate_norm"] = c_norm
            checks[f"{key}_threshold"] = thresh
            hard_checks.append(f"{key}_improved_or_at_trend")
        else:
            ok = c_norm <= thresh
            checks[f"{key}_below_threshold"] = ok
            checks[f"{key}_candidate_norm"] = c_norm
            checks[f"{key}_threshold"] = thresh
            hard_checks.append(f"{key}_below_threshold")

    passed = bool(hard_checks) and all(checks[k] for k in hard_checks)
    checks["hard_gate_keys"] = hard_checks
    checks["passed"] = passed
    return passed, checks


def compute_insertion_residuals(
    trajectory: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    traj = trajectory or {}
    thresholds = resolve_thresholds(context)

    final_z = float(_get_float(traj, "final_z_diff", default=0.0) or 0.0)
    min_yaw = float(_get_float(traj, "min_yaw_error", default=0.0) or 0.0)
    final_xy = _final_peg_xy(traj)
    min_xy = _min_peg_xy(traj)
    acc_max = float(_get_float(traj, "action_acceleration_max", default=0.0) or 0.0)
    e_z = float(_get_float(traj, "E_z_norm", default=0.0) or 0.0)

    depth_dev = abs(final_z - SUCCESS_FINAL_Z_REF)
    depth_norm = float(depth_dev / max(Z_INSERT_RISK_THRESH, 1e-6) + e_z * 0.5)
    depth_source = "measured" if "final_z_diff" in traj else "fallback"

    axis_norm = float(min_yaw / max(YAW_ALIGN_THRESH, 1e-6))
    axis_source = "measured" if "min_yaw_error" in traj else "fallback"

    vertical_penalty = float(max(0.0, final_z - Z_INSERT_RISK_THRESH) / max(Z_INSERT_RISK_THRESH, 1e-6))
    vertical_source = depth_source

    pose_norm = float(
        final_xy / max(thresholds["success_xy_threshold"], 1e-6)
        + min_yaw / max(YAW_ALIGN_THRESH, 1e-6)
        + depth_norm * 0.4
    )
    pose_source = "measured" if final_xy < 1.0 else "fallback"

    jam_norm = float(min(2.0, acc_max / 2.5 + max(0.0, final_z - 0.0) * 5.0))
    jam_source = "measured" if acc_max > 0 else "fallback"

    residuals = {
        "E_insert_depth": _residual_entry(depth_dev, depth_norm, source=depth_source),
        "E_axis_alignment": _residual_entry(min_yaw, axis_norm, source=axis_source),
        "E_vertical_approach": _residual_entry(vertical_penalty, vertical_penalty, source=vertical_source),
        "E_final_pose": _residual_entry(pose_norm, pose_norm, source=pose_source),
        "E_jamming": _residual_entry(acc_max, jam_norm, source=jam_source),
    }

    total = float(
        sum(INSERTION_RANKING_WEIGHTS[k] * residuals[k]["normalized"] for k in INSERTION_RESIDUAL_KEYS)
        / sum(INSERTION_RANKING_WEIGHTS.values())
    )

    return {
        "residuals": residuals,
        "insertion_total_score": total,
        "residual_keys": list(INSERTION_RESIDUAL_KEYS),
    }


def combined_demo4_ranking_score(
    physics_breakdown: dict[str, Any],
    insertion_breakdown: dict[str, Any],
    *,
    physics_weight: float = 0.55,
    insertion_weight: float = 0.45,
) -> float:
    return float(
        physics_weight * physics_breakdown.get("ranking_score", physics_breakdown["raw_total_score"])
        + insertion_weight * insertion_breakdown["insertion_total_score"]
    )
