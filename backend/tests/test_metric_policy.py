from __future__ import annotations

from app.services.evaluation.metric_policy import partition_metric_results_for_report
from app.services.evaluation.sim_time_metrics import compute_episode_sim_time_sec, enrich_run_metrics_sim_time


def test_enrich_run_metrics_sim_time_from_steps_and_hz() -> None:
    enriched = enrich_run_metrics_sim_time({"meanSteps": 17100, "controlFrequencyHz": 500})
    assert enriched["meanSimTimeSec"] == 34.2


def test_compute_episode_sim_time_sec() -> None:
    assert compute_episode_sim_time_sec(step_count=137, control_frequency_hz=20) == 6.85


def test_partition_metric_results_filters_deprecated_and_unknown() -> None:
    body, deprecated, unknown = partition_metric_results_for_report(
        [
            "metric_success_rate_v1",
            "metric_runtime_max_runtime_sec_v1",
            "metric_episode_stability_v1",
        ],
        {
            "metric_success_rate_v1": {
                "metricId": "metric_success_rate_v1",
                "displayName": "成功率",
                "available": True,
                "formattedValue": "100%",
            },
            "metric_runtime_max_runtime_sec_v1": {
                "metricId": "metric_runtime_max_runtime_sec_v1",
                "displayName": "最大耗时",
                "available": True,
                "formattedValue": "581.43s",
            },
            "metric_episode_stability_v1": {
                "metricId": "metric_episode_stability_v1",
                "displayName": "metric_episode_stability_v1",
                "available": False,
                "reason": "未知 metricId",
            },
        },
    )
    assert "metric_success_rate_v1" in body
    assert "metric_runtime_max_runtime_sec_v1" not in body
    assert len(deprecated) == 1
    assert deprecated[0]["deprecated"] is True
    assert len(unknown) == 1


def test_partition_hidden_metrics_move_to_deprecated() -> None:
    body, deprecated, unknown = partition_metric_results_for_report(
        [
            "metric_runtime_mean_steps_v1",
            "metric_runtime_smoothness_v1",
            "metric_runtime_max_action_norm_v1",
            "metric_runtime_ee_path_length_v1",
        ],
        {
            "metric_runtime_mean_steps_v1": {
                "metricId": "metric_runtime_mean_steps_v1",
                "displayName": "平均步数",
                "available": True,
                "formattedValue": "120 steps",
            },
            "metric_runtime_smoothness_v1": {
                "metricId": "metric_runtime_smoothness_v1",
                "displayName": "动作平稳性",
                "available": True,
                "formattedValue": "0.83",
            },
        },
    )
    assert "metric_runtime_mean_steps_v1" in body
    assert "metric_runtime_smoothness_v1" not in body
    hidden_ids = {item["metricId"] for item in deprecated}
    assert "metric_runtime_smoothness_v1" in hidden_ids
    assert "metric_runtime_max_action_norm_v1" in hidden_ids
    assert "metric_runtime_ee_path_length_v1" in hidden_ids
    assert unknown == []


def test_filter_available_metric_ids_for_cable_threading() -> None:
    from app.services.evaluation.metric_policy import filter_available_metric_ids

    filtered = filter_available_metric_ids(
        [
            "metric_cable_success_rate_v1",
            "metric_runtime_mean_steps_v1",
            "metric_runtime_max_action_norm_v1",
            "metric_runtime_mean_joint_speed_v1",
        ],
        "cable_threading",
    )
    assert filtered == [
        "metric_cable_success_rate_v1",
        "metric_runtime_mean_steps_v1",
    ]


def test_partition_deprecated_selected_without_metric_entry() -> None:
    body, deprecated, unknown = partition_metric_results_for_report(
        ["metric_runtime_max_runtime_sec_v1", "metric_episode_stability_v1"],
        {},
    )
    assert body == {}
    assert len(deprecated) == 1
    assert deprecated[0]["metricId"] == "metric_runtime_max_runtime_sec_v1"
    assert len(unknown) == 1
    assert unknown[0]["metricId"] == "metric_episode_stability_v1"
