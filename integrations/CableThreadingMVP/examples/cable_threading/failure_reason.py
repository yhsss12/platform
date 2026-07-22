"""Unified cable threading episode failure explanation (no robosuite imports)."""


def _summary_flag_is_false(metrics, *keys):
    for key in keys:
        if key in metrics and metrics.get(key) is False:
            return True
    return False


def build_cable_threading_failure_reason(metrics):
    """Explain episode failure from final-step threading_task_metrics fields."""
    if not isinstance(metrics, dict):
        return "未满足最终成功条件"
    if metrics.get("final_success") is True or metrics.get("success") is True:
        return ""

    if _summary_flag_is_false(
        metrics,
        "threaded_final",
        "threaded",
    ) or _summary_flag_is_false(metrics, "cable_low_intersects_pole_segment"):
        thread_completion = metrics.get("thread_completion_final", metrics.get("thread_completion"))
        if thread_completion is not None:
            return f"线缆未完成穿杆（thread_completion={thread_completion}）"
        return "线缆未完成穿杆"

    if _summary_flag_is_false(
        metrics,
        "endpoint_region_final",
        "endpoint_region",
        "endpoint_past_gap_final",
        "endpoint_past_gap",
    ):
        endpoint_err = metrics.get("endpoint_goal_error_final", metrics.get("endpoint_goal_error"))
        if endpoint_err is not None:
            return f"线缆端点未到达目标区域（endpoint_goal_error={endpoint_err}）"
        return "线缆端点未到达目标区域"

    if _summary_flag_is_false(metrics, "straightened_final", "straightened"):
        err = metrics.get("straightness_error_final", metrics.get("straightness_error"))
        if err is not None:
            return f"线缆未拉直（straightness_error={err}）"
        return "线缆未拉直"

    if _summary_flag_is_false(metrics, "settled_on_table_final", "settled_on_table"):
        return "线缆未稳定落桌"

    if _summary_flag_is_false(metrics, "anchor_stable_final"):
        err = metrics.get("anchor_error_final", metrics.get("anchor_error"))
        if err is not None:
            return f"锚点偏移超出容差（anchor_error={err}）"
        return "锚点偏移超出容差"

    peak = metrics.get("peak_height_excess_final", metrics.get("peak_height_excess"))
    try:
        if peak is not None and float(peak) > 1e-6:
            return f"线缆高度超过限制（peak_height_excess={peak}）"
    except (TypeError, ValueError):
        pass

    return "未满足最终成功条件"
