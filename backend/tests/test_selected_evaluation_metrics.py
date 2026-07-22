"""Tests for selected evaluation metric resolution."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.evaluation.selected_evaluation_metrics import (
    finalize_selected_evaluation_metrics,
    normalize_selected_metric_ids,
    resolve_selected_evaluation_metric_results,
)


def test_normalize_selected_metric_ids_maps_legacy_success_rate() -> None:
    normalized = normalize_selected_metric_ids(
        ["success_rate", "metric_runtime_mean_steps_v1"],
        "cable_threading",
    )
    assert normalized == ["metric_cable_success_rate_v1", "metric_runtime_mean_steps_v1"]


def test_resolve_selected_metric_results_only_selected(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    step_dir = job_root / "results" / "step_metrics" / "episode_001"
    step_dir.mkdir(parents=True)
    (step_dir / "summary.json").write_text(
        json.dumps(
            {
                "stepCount": 120,
                "wallTimeSec": 6.0,
                "maxActionNorm": 0.42,
                "smoothnessScore": 0.97,
                "videoFps": 20,
                "controlFrequencyHz": 20,
            }
        ),
        encoding="utf-8",
    )

    aggregate = {"success_rate": 0.8, "total_episodes": 2, "successful_episodes": 1}
    resolved = resolve_selected_evaluation_metric_results(
        [
            "metric_cable_success_rate_v1",
            "metric_runtime_mean_steps_v1",
        ],
        aggregate,
        job_root,
        "cable_threading",
        legacy_fallback=False,
    )

    assert resolved["selectedMetricIds"] == [
        "metric_cable_success_rate_v1",
        "metric_runtime_mean_steps_v1",
    ]
    assert set(resolved["metricResults"].keys()) == set(resolved["selectedMetricIds"])

    success = resolved["metricResults"]["metric_cable_success_rate_v1"]
    assert success["available"] is True
    assert success["formattedValue"] == "80%"

    mean_steps = resolved["metricResults"]["metric_runtime_mean_steps_v1"]
    assert mean_steps["available"] is True
    assert mean_steps["value"] == 120.0


def test_legacy_mean_runtime_metric_alias_uses_mean_sim_time(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    step_dir = job_root / "results" / "step_metrics" / "episode_001"
    step_dir.mkdir(parents=True)
    (step_dir / "summary.json").write_text(
        json.dumps(
            {
                "stepCount": 100,
                "dt": 0.05,
                "controlFrequencyHz": 20.0,
                "simTimeSec": 5.0,
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_selected_evaluation_metric_results(
        ["metric_runtime_mean_runtime_sec_v1"],
        {},
        job_root,
        "cable_threading",
        legacy_fallback=False,
    )
    metric = resolved["metricResults"]["metric_runtime_mean_sim_time_sec_v1"]
    assert metric["displayName"] == "平均仿真时长"
    assert metric["available"] is True
    assert metric["value"] == 5.0
    assert metric["source"] == "runMetrics.meanSimTimeSec"


def test_new_mean_sim_time_metric_id(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    step_dir = job_root / "results" / "step_metrics" / "episode_001"
    step_dir.mkdir(parents=True)
    (step_dir / "summary.json").write_text(
        json.dumps({"stepCount": 80, "dt": 0.05, "controlFrequencyHz": 20.0}),
        encoding="utf-8",
    )
    resolved = resolve_selected_evaluation_metric_results(
        ["metric_runtime_mean_sim_time_sec_v1"],
        {},
        job_root,
        "cable_threading",
        legacy_fallback=False,
    )
    metric = resolved["metricResults"]["metric_runtime_mean_sim_time_sec_v1"]
    assert metric["displayName"] == "平均仿真时长"
    assert metric["value"] == 4.0


def test_finalize_persists_metric_results(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    meta_dir = job_root / "metadata"
    meta_dir.mkdir(parents=True)
    (meta_dir / "evaluation_context.json").write_text(
        json.dumps({"selectedMetricIds": ["metric_cable_success_rate_v1"]}),
        encoding="utf-8",
    )
    aggregate = {"final_success_rate": 1.0}

    result = finalize_selected_evaluation_metrics(
        aggregate,
        job_root,
        None,
        task_type="cable_threading",
        persist=True,
        legacy_fallback=False,
    )

    aggregate_path = job_root / "results" / "aggregate_result.json"
    assert aggregate_path.is_file()
    saved = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert saved["selectedMetricIds"] == ["metric_cable_success_rate_v1"]
    assert "metricResults" in saved
    assert result["metricResults"]["metric_cable_success_rate_v1"]["available"] is True


def test_dual_arm_runtime_metrics_from_trajectory(tmp_path: Path) -> None:
    job_root = tmp_path / "eval_job"
    traj = job_root / "episodes" / "episode_00" / "episode" / "step_00" / "trajectory"
    traj.mkdir(parents=True)
    import numpy as np

    actions = np.random.randn(50, 14).astype(np.float64)
    np.save(traj / "actions.npy", actions)
    qvel = np.random.randn(50, 14).astype(np.float64)
    np.savez(
        traj / "obs.npz",
        left_arm_joint_vel=qvel[:, :7],
        right_arm_joint_vel=qvel[:, 7:],
    )
    (traj / "trajectory_manifest.json").write_text(
        json.dumps(
            {
                "numTransitions": 50,
                "controlFrequency": 100,
                "actionSemantics": "recorded_joint_position_targets",
            }
        ),
        encoding="utf-8",
    )
    (job_root / "results").mkdir(parents=True)
    (job_root / "results" / "aggregate_result.json").write_text(
        json.dumps({"successRate": 1.0, "totalEpisodes": 1, "successEpisodes": 1}),
        encoding="utf-8",
    )
    (job_root / "metadata").mkdir(parents=True)
    (job_root / "metadata" / "evaluation_context.json").write_text(
        json.dumps(
            {
                "selectedMetricIds": [
                    "metric_success_rate_v1",
                    "metric_runtime_mean_steps_v1",
                    "metric_runtime_mean_sim_time_sec_v1",
                ]
            }
        ),
        encoding="utf-8",
    )

    result = finalize_selected_evaluation_metrics(
        json.loads((job_root / "results" / "aggregate_result.json").read_text(encoding="utf-8")),
        job_root,
        None,
        task_type="dual_arm_cable_manipulation",
        persist=True,
        legacy_fallback=False,
    )

    metrics = result["metricResults"]
    assert metrics["metric_success_rate_v1"]["available"] is True
    assert metrics["metric_runtime_mean_steps_v1"]["available"] is True
    assert metrics["metric_runtime_mean_steps_v1"]["value"] == 50.0
    assert metrics["metric_runtime_mean_sim_time_sec_v1"]["available"] is True
    assert metrics["metric_runtime_mean_sim_time_sec_v1"]["value"] == 0.5
