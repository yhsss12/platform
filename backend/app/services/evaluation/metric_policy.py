"""Shared policy for evaluation metrics display and export."""

from __future__ import annotations

from typing import Any

DEPRECATED_METRIC_IDS: dict[str, str] = {
    "metric_runtime_max_runtime_sec_v1": "该指标已废弃：wall time 含义不稳定，不再作为评测指标展示",
}

# 已从产品指标体系中下线：runMetrics 可保留原始字段，但不在主体展示与新建任务选择中出现
DEPRECATED_HIDDEN_METRIC_IDS: dict[str, str] = {
    "metric_runtime_ee_path_length_v1": "该指标已下线：末端轨迹长度不再作为评测指标展示",
    "metric_runtime_smoothness_v1": "该指标已下线：动作平稳性不再作为评测指标展示",
    "metric_runtime_max_action_norm_v1": "该指标已下线：最大动作范数不再作为评测指标展示",
}

# 不在标准报告主体展示（旧遗留 / 非 runtime 指标 / 未实现 resolver / 已下线指标）
REPORT_BODY_EXCLUDED_METRIC_IDS = frozenset(
    {
        *DEPRECATED_METRIC_IDS.keys(),
        *DEPRECATED_HIDDEN_METRIC_IDS.keys(),
        "metric_episode_stability_v1",
    }
)

CABLE_THREADING_COMPUTABLE_METRIC_IDS: tuple[str, ...] = (
    "metric_cable_success_rate_v1",
    "metric_runtime_mean_steps_v1",
    "metric_runtime_max_steps_v1",
    "metric_runtime_video_fps_v1",
    "metric_runtime_control_frequency_v1",
    "metric_runtime_mean_sim_time_sec_v1",
)

DUAL_ARM_COMPUTABLE_METRIC_IDS: tuple[str, ...] = (
    "metric_success_rate_v1",
    "metric_runtime_mean_steps_v1",
    "metric_runtime_max_steps_v1",
    "metric_runtime_video_fps_v1",
    "metric_runtime_control_frequency_v1",
    "metric_runtime_mean_sim_time_sec_v1",
    "metric_runtime_mean_joint_speed_v1",
    "metric_runtime_max_joint_speed_v1",
    "metric_runtime_mean_joint_acceleration_v1",
    "metric_runtime_max_joint_acceleration_v1",
)

ISAAC_STACK_COMPUTABLE_METRIC_IDS: tuple[str, ...] = (
    "isaac_stack_success_rate_v1",
    "isaac_stack_mean_reward_v1",
    "isaac_stack_mean_episode_length_v1",
    "isaac_stack_failure_count_v1",
    "isaac_stack_timeout_rate_v1",
)

TASK_COMPUTABLE_METRIC_IDS: dict[str, tuple[str, ...]] = {
    "cable_threading": CABLE_THREADING_COMPUTABLE_METRIC_IDS,
    "dual_arm_cable_manipulation": DUAL_ARM_COMPUTABLE_METRIC_IDS,
    "block_stacking": ISAAC_STACK_COMPUTABLE_METRIC_IDS,
    "isaaclab_franka_stack_cube": ISAAC_STACK_COMPUTABLE_METRIC_IDS,
    "stacking": ISAAC_STACK_COMPUTABLE_METRIC_IDS,
}

SMOOTHNESS_METRIC_DESCRIPTION = (
    "基于相邻 step 的 action 向量变化量计算：actionDelta_t = ||action_t - action_{t-1}||₂，"
    "smoothnessScore = 1/(1+meanActionDelta)。数值越接近 1 表示 action 输出越平稳；"
    "不是末端轨迹平稳性，不是关节速度/加速度平稳性，不等于真实物理运动平稳性。"
)

KNOWN_METRIC_IDS = frozenset(
    {
        "metric_cable_success_rate_v1",
        "metric_success_rate_v1",
        "success_rate",
        "isaac_stack_success_rate_v1",
        "isaac_stack_mean_reward_v1",
        "isaac_stack_mean_episode_length_v1",
        "isaac_stack_failure_count_v1",
        "isaac_stack_timeout_rate_v1",
        "metric_runtime_mean_steps_v1",
        "metric_runtime_max_steps_v1",
        "metric_runtime_video_fps_v1",
        "metric_runtime_control_frequency_v1",
        "metric_runtime_mean_sim_time_sec_v1",
        "metric_runtime_mean_runtime_sec_v1",
        "metric_runtime_max_action_norm_v1",
        "metric_runtime_smoothness_v1",
        "metric_runtime_ee_path_length_v1",
        "metric_runtime_path_efficiency_v1",
        "metric_runtime_mean_joint_speed_v1",
        "metric_runtime_max_joint_speed_v1",
        "metric_runtime_mean_joint_acceleration_v1",
        "metric_runtime_max_joint_acceleration_v1",
    }
)


def is_deprecated_metric(metric_id: str) -> bool:
    return metric_id in DEPRECATED_METRIC_IDS


def is_hidden_metric(metric_id: str) -> bool:
    return metric_id in DEPRECATED_HIDDEN_METRIC_IDS


def is_report_body_metric(metric_id: str) -> bool:
    return metric_id not in REPORT_BODY_EXCLUDED_METRIC_IDS


def filter_available_metric_ids(metric_ids: list[str], task_type: str) -> list[str]:
    """Keep only metrics that are selectable for the given task type."""
    allowed = set(TASK_COMPUTABLE_METRIC_IDS.get(task_type, ()))
    if not allowed:
        return [
            mid
            for mid in metric_ids
            if is_report_body_metric(mid) and mid in KNOWN_METRIC_IDS
        ]
    seen: set[str] = set()
    result: list[str] = []
    for metric_id in metric_ids:
        if metric_id in seen:
            continue
        if metric_id not in allowed:
            continue
        if not is_report_body_metric(metric_id):
            continue
        seen.add(metric_id)
        result.append(metric_id)
    return result


def apply_smoothness_description(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return entry
    if entry.get("metricId") == "metric_runtime_smoothness_v1" or entry.get("displayName") == "动作平稳性":
        merged = dict(entry)
        merged["description"] = SMOOTHNESS_METRIC_DESCRIPTION
        return merged
    return entry


def partition_metric_results_for_report(
    selected_metric_ids: list[str],
    metric_results: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    body: dict[str, Any] = {}
    deprecated: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []

    ordered_ids = list(selected_metric_ids) + [
        mid for mid in metric_results.keys() if mid not in selected_metric_ids
    ]
    seen: set[str] = set()

    for metric_id in ordered_ids:
        if metric_id in seen:
            continue
        seen.add(metric_id)
        entry = metric_results.get(metric_id)
        entry_dict = entry if isinstance(entry, dict) else {}

        if is_deprecated_metric(metric_id):
            deprecated.append(
                {
                    **entry_dict,
                    "metricId": metric_id,
                    "displayName": entry_dict.get("displayName") or "最大耗时",
                    "deprecated": True,
                    "reason": DEPRECATED_METRIC_IDS[metric_id],
                }
            )
            continue

        if is_hidden_metric(metric_id):
            deprecated.append(
                {
                    **entry_dict,
                    "metricId": metric_id,
                    "displayName": entry_dict.get("displayName") or metric_id,
                    "deprecated": True,
                    "hidden": True,
                    "reason": DEPRECATED_HIDDEN_METRIC_IDS[metric_id],
                }
            )
            continue

        if metric_id not in selected_metric_ids:
            if not entry_dict:
                continue
            if metric_id in REPORT_BODY_EXCLUDED_METRIC_IDS:
                unknown.append({**entry_dict, "metricId": metric_id})
            continue

        if metric_id in REPORT_BODY_EXCLUDED_METRIC_IDS or (
            metric_id not in KNOWN_METRIC_IDS
            and (entry_dict.get("reason") == "未知 metricId" or not entry_dict)
        ):
            unknown.append(
                {
                    **entry_dict,
                    "metricId": metric_id,
                    "displayName": entry_dict.get("displayName") or metric_id,
                    "available": entry_dict.get("available", False),
                    "reason": entry_dict.get("reason") or "未知 metricId",
                }
            )
            continue

        if not entry_dict:
            continue

        if metric_id not in KNOWN_METRIC_IDS and not entry_dict.get("available"):
            unknown.append({**entry_dict, "metricId": metric_id})
            continue

        body[metric_id] = apply_smoothness_description(entry_dict)

    return body, deprecated, unknown


def filter_selected_metric_ids_for_display(selected_metric_ids: list[str]) -> list[str]:
    return [mid for mid in selected_metric_ids if is_report_body_metric(mid)]
