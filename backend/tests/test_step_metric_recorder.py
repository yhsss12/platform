"""Tests for step metric recorder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_COMMON = Path(__file__).resolve().parents[2] / "integrations" / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from step_metrics.step_metric_recorder import (  # noqa: E402
    StepMetricRecorder,
    aggregate_run_metrics_from_summaries,
    attach_run_metrics_to_aggregate,
)


def test_recorder_summary_only(tmp_path: Path) -> None:
    out = tmp_path / "episode_001"
    recorder = StepMetricRecorder(
        job_id="ct_eval_test",
        episode_index=1,
        output_dir=out,
        dt=0.05,
        control_frequency_hz=20.0,
        video_fps=20.0,
    )
    recorder.start_episode()
    a0 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    a1 = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    recorder.record_step(0, action=a0, reward=0.0, done=False, step_wall_sec=0.05)
    recorder.record_step(1, action=a1, reward=0.0, done=False, step_wall_sec=0.05)
    summary = recorder.finish_episode(success=True, timeout=False)

    assert summary["stepCount"] == 2
    assert summary["stepWallTimeSec"] == 0.1
    assert summary["simTimeSec"] == 0.1
    assert summary["wallTimeSec"] == 0.1
    assert summary["maxActionNorm"] is not None
    assert summary["maxActionDelta"] is not None
    assert summary["smoothnessScore"] is not None
    assert (out / "summary.json").is_file()
    assert not (out / "step_arrays.npz").exists()


def test_aggregate_run_metrics(tmp_path: Path) -> None:
    summaries = [
        {
            "stepCount": 100,
            "stepWallTimeSec": 5.0,
            "simTimeSec": 5.0,
            "meanActionNorm": 0.1,
            "maxActionNorm": 0.3,
            "meanActionDelta": 0.02,
            "deltaCount": 99,
            "maxActionDelta": 0.1,
            "smoothnessScore": 0.98,
            "videoFps": 20,
            "controlFrequencyHz": 20,
            "dt": 0.05,
        },
        {
            "stepCount": 120,
            "stepWallTimeSec": 6.0,
            "simTimeSec": 6.0,
            "meanActionNorm": 0.12,
            "maxActionNorm": 0.4,
            "meanActionDelta": 0.03,
            "deltaCount": 119,
            "maxActionDelta": 0.15,
            "smoothnessScore": 0.96,
            "videoFps": 20,
            "controlFrequencyHz": 20,
            "dt": 0.05,
        },
    ]
    run = aggregate_run_metrics_from_summaries(summaries)
    assert run["meanSteps"] == 110.0
    assert run["maxSteps"] == 120
    assert run["meanRuntimeSec"] == 5.5
    assert run["meanSimTimeSec"] == 5.5
    assert run["videoFps"] == 20
    assert run["smoothnessScore"] == round(1.0 / (1.0 + ((0.02 * 99 + 0.03 * 119) / 218)), 6)

    root = tmp_path / "results"
    for idx, item in enumerate(summaries, start=1):
        ep_dir = root / "step_metrics" / f"episode_{idx:03d}"
        ep_dir.mkdir(parents=True)
        (ep_dir / "summary.json").write_text(json.dumps(item), encoding="utf-8")

    aggregate = attach_run_metrics_to_aggregate({"videoFps": 20}, root)
    assert "runMetrics" in aggregate
    assert aggregate["runMetrics"]["meanSteps"] == 110.0


def test_legacy_wall_time_fallback_uses_sim_time() -> None:
    summaries = [
        {"stepCount": 118, "wallTimeSec": 33.0, "dt": 0.05, "controlFrequencyHz": 20.0, "meanActionDelta": 0.2, "deltaCount": 117},
    ]
    run = aggregate_run_metrics_from_summaries(summaries)
    assert run["meanRuntimeSec"] == 5.9
    assert run["meanSimTimeSec"] == 5.9
