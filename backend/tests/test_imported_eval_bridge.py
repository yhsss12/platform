from __future__ import annotations

from pathlib import Path

import pytest


def test_imported_eval_job_id_valid_in_frontend_pattern():
    from app.services.evaluation.job_paths import is_valid_eval_job_id_format

    assert is_valid_eval_job_id_format("eval_joint_dp_20260624_full_pipeline")


def test_get_imported_eval_cable_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.services import imported_eval_bridge as bridge
    from app.services.evaluation import evaluation_service as eval_svc

    eval_root = tmp_path / "eval"
    videos = eval_root / "videos"
    videos.mkdir(parents=True)
    browser = videos / "eval.browser.mp4"
    browser.write_bytes(b"fake-mp4")

    job_id = "eval_joint_dp_test_pipeline"

    monkeypatch.setattr(
        "app.services.workspace_runtime_paths.resolve_eval_job_root",
        lambda _job_id: eval_root,
    )
    monkeypatch.setattr(
        "app.services.workspace_runtime_paths.is_imported_workspace_eval_job_id",
        lambda candidate: candidate == job_id,
    )
    monkeypatch.setattr(
        eval_svc,
        "get_evaluation_status",
        lambda _job_id: {
            "status": "completed",
            "metrics": {"aggregate": {"total_episodes": 10}},
            "replayUri": f"/api/workspace/evaluation/jobs/{job_id}/video",
            "replayUris": [{"uri": f"/api/workspace/evaluation/jobs/{job_id}/video", "round": 1}],
            "videoAvailable": True,
        },
    )
    monkeypatch.setattr(
        "app.services.evaluation_workbench_basic_info.attach_workbench_basic_info",
        lambda payload, **_: payload,
    )

    status = bridge.get_imported_eval_cable_status(job_id)
    assert status["evalBrowserVideoExists"] is True
    assert status["videoAvailable"] is True
    assert status["replayUri"].endswith("/video")
