from __future__ import annotations

import json
from pathlib import Path

from app.services.cable_threading_service import _append_cable_generation_failure_metrics


def test_append_cable_generation_failure_metrics(tmp_path: Path):
    failures = [
        {
            "episode": 1,
            "seed": 2,
            "summary": {
                "episode": 1,
                "seed": 2,
                "success": False,
                "threaded_final": False,
                "cable_low_intersects_pole_segment": False,
                "straightened_final": False,
                "straightness_error_final": 0.041,
                "failure_reason": "线缆未完成穿杆（thread_completion=0.42）",
            },
        }
    ]
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "failures.json").write_text(json.dumps(failures), encoding="utf-8")

    metrics: dict = {}
    _append_cable_generation_failure_metrics(metrics, tmp_path)

    assert metrics["failedEpisodes"] == 1
    assert len(metrics["failureSummary"]) == 1
    assert metrics["failureSummary"][0]["episodeIndex"] == 2
    assert metrics["failureSummary"][0]["seed"] == 2
    assert "穿杆" in metrics["failureSummary"][0]["failureReason"]
    assert "未拉直" not in metrics["failureSummary"][0]["failureReason"]


def test_append_cable_generation_failure_metrics_infers_reason(tmp_path: Path):
    failures = [
        {
            "episode": 1,
            "seed": 2,
            "summary": {
                "episode": 1,
                "seed": 2,
                "success": False,
                "threaded_final": False,
                "straightened_final": False,
                "straightness_error_final": 0.041,
            },
        }
    ]
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "failures.json").write_text(json.dumps(failures), encoding="utf-8")

    metrics: dict = {}
    _append_cable_generation_failure_metrics(metrics, tmp_path)

    assert "穿杆" in metrics["failureSummary"][0]["failureReason"]
    assert "未拉直" not in metrics["failureSummary"][0]["failureReason"]
