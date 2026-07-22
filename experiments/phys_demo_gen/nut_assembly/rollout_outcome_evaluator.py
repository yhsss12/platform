"""Rollout 结果评估：transport / xy / lift / partial / final success。"""
from __future__ import annotations

from typing import Any

from physics_residuals import _final_peg_xy, _min_peg_xy, _nut_lift_delta, resolve_thresholds

PARTIAL_LIFT_THRESH = 0.005


def evaluate_rollout_outcome(
    rollout: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = resolve_thresholds(context)
    min_xy = _min_peg_xy(rollout)
    final_xy = _final_peg_xy(rollout)
    lift_delta = _nut_lift_delta(rollout)

    transport_success = min_xy <= thresholds["success_transport_threshold"]
    xy_alignment_success = final_xy <= thresholds["success_xy_threshold"]
    lift_success = lift_delta >= thresholds["success_lift_min"]
    partial_lift = lift_delta >= PARTIAL_LIFT_THRESH

    grasp_proxy = bool(rollout.get("grasp_success_proxy", False))
    lift_proxy = bool(rollout.get("lift_success_proxy", False))
    partial_success = bool(
        rollout.get("partial_lift_success")
        or partial_lift
        or (transport_success and xy_alignment_success)
        or grasp_proxy
        or lift_proxy
    )
    final_success = bool(rollout.get("success_flag"))

    failure_reason = _infer_failure_reason(
        rollout=rollout,
        transport_success=transport_success,
        xy_alignment_success=xy_alignment_success,
        lift_success=lift_success,
        partial_success=partial_success,
        final_success=final_success,
    )

    return {
        "transport_success": transport_success,
        "xy_alignment_success": xy_alignment_success,
        "lift_success": lift_success,
        "partial_lift": partial_lift,
        "partial_success": partial_success,
        "final_success": final_success,
        "failure_reason": failure_reason,
        "min_nut_peg_xy": min_xy,
        "final_nut_peg_xy": final_xy,
        "nut_z_lift_delta": lift_delta,
    }


def _infer_failure_reason(
    *,
    rollout: dict[str, Any],
    transport_success: bool,
    xy_alignment_success: bool,
    lift_success: bool,
    partial_success: bool,
    final_success: bool,
) -> str:
    if final_success:
        return "success"
    if partial_success and not final_success:
        return "partial_success_not_final"
    if not transport_success:
        return str(rollout.get("failure_guess") or "transport_failed")
    if not xy_alignment_success:
        return str(rollout.get("failure_guess") or "alignment_failed")
    if not lift_success:
        return str(rollout.get("failure_guess") or "lift_underdeveloped")
    return str(rollout.get("failure_guess") or rollout.get("failure_type") or "unknown_failed")
