"""Tests for runtime metrics service."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.evaluation.runtime_metrics import compute_runtime_metric_values


def test_compute_runtime_metric_values_with_run_metrics(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    step_dir = job_root / "results" / "step_metrics" / "episode_001"
    step_dir.mkdir(parents=True)
    (step_dir / "summary.json").write_text(
        json.dumps(
            {
                "stepCount": 118,
                "wallTimeSec": 6.2,
                "meanActionNorm": 0.12,
                "maxActionNorm": 0.43,
                "meanActionDelta": 0.03,
                "maxActionDelta": 0.21,
                "smoothnessScore": 0.97,
                "videoFps": 20,
                "controlFrequencyHz": 20,
            }
        ),
        encoding="utf-8",
    )

    values = compute_runtime_metric_values("ct_eval_test", job_root)
    assert values["meanSteps"]["available"] is True
    assert values["meanSteps"]["value"] == 118.0
    assert values["smoothnessScore"]["available"] is True
    assert values["eePathLength"]["available"] is False
    assert "eePosition" in (values["eePathLength"]["reason"] or "")


def test_compute_runtime_metric_values_missing(tmp_path: Path) -> None:
    values = compute_runtime_metric_values("eval_test", tmp_path)
    assert values["meanSteps"]["available"] is False
    assert values["maxJointSpeed"]["available"] is False
