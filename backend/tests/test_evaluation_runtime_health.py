"""evaluation_runtime_health 单元测试。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.services.evaluation import evaluation_runtime_health as health


@pytest.fixture
def eval_job_root(tmp_path: Path) -> Path:
    job_id = "eval_20260624_999999_abcd"
    root = tmp_path / "runs" / "evaluations" / "jobs" / job_id
    (root / "logs").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    status = {
        "evalJobId": job_id,
        "status": "running",
        "updatedAt": "2026-06-24T01:00:00+00:00",
    }
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (root / "logs" / "eval.log").write_text("[eval_worker] started\n", encoding="utf-8")
    return root


def test_stale_running_without_process_marked_failed(eval_job_root: Path, monkeypatch):
    import os

    job_id = eval_job_root.name
    stale = time.time() - 7200
    for path in eval_job_root.rglob("*"):
        if path.is_file():
            os.utime(path, (stale, stale))

    monkeypatch.setattr(health, "find_processes_for_job", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(health, "PROJECT_ROOT", eval_job_root.parents[3])

    result = health.inspect_evaluation_runtime_health(job_id, str(eval_job_root), declared_status="running")
    assert result["actualStatus"] == "failed"
    assert result["isProcessAlive"] is False
    assert "失联" in result["reason"]


def test_alive_process_keeps_running(eval_job_root: Path, monkeypatch):
    job_id = eval_job_root.name
    monkeypatch.setattr(
        health,
        "find_processes_for_job",
        lambda *_args, **_kwargs: [{"pid": 12345, "cmdline": f"python run.py {job_id}", "etime": "00:05:00"}],
    )
    monkeypatch.setattr(health, "PROJECT_ROOT", eval_job_root.parents[3])

    result = health.inspect_evaluation_runtime_health(job_id, str(eval_job_root), declared_status="running")
    assert result["actualStatus"] == "running"
    assert result["isProcessAlive"] is True


def test_apply_reconciliation_writes_failed_status(eval_job_root: Path, monkeypatch):
    job_id = eval_job_root.name
    monkeypatch.setattr(health, "find_processes_for_job", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(health, "PROJECT_ROOT", eval_job_root.parents[3])

    stale = time.time() - 7200
    eval_job_root.joinpath("status.json").touch()
    eval_job_root.joinpath("logs", "eval.log").touch()
    for path in (eval_job_root / "status.json", eval_job_root / "logs" / "eval.log"):
        import os

        os.utime(path, (stale, stale))

    health_result = health.reconcile_evaluation_runtime_health(
        job_id,
        str(eval_job_root),
        declared_status="running",
        apply=True,
    )
    assert health_result["actualStatus"] == "failed"
    assert health_result["applied"] is True
    payload = json.loads((eval_job_root / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"


def test_completed_result_detected(eval_job_root: Path, monkeypatch):
    job_id = eval_job_root.name
    aggregate = {"status": "completed", "successRate": 0.8}
    (eval_job_root / "results" / "aggregate_result.json").write_text(
        json.dumps(aggregate),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "find_processes_for_job", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(health, "PROJECT_ROOT", eval_job_root.parents[3])

    result = health.inspect_evaluation_runtime_health(job_id, str(eval_job_root), declared_status="running")
    assert result["actualStatus"] == "completed"


def test_completed_episode_set_survives_parent_worker_restart(eval_job_root: Path, monkeypatch):
    job_id = eval_job_root.name
    status_path = eval_job_root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"evaluationMode": "episode_stability", "totalEpisodes": 1})
    status_path.write_text(json.dumps(status), encoding="utf-8")
    episode_dir = eval_job_root / "episodes" / "episode_00"
    (episode_dir / "results").mkdir(parents=True)
    (episode_dir / "status.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (episode_dir / "results" / "episode_result.json").write_text(
        json.dumps({"episode_success": True}), encoding="utf-8"
    )
    monkeypatch.setattr(health, "find_processes_for_job", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(health, "PROJECT_ROOT", eval_job_root.parents[3])

    result = health.inspect_evaluation_runtime_health(
        job_id, str(eval_job_root), declared_status="running"
    )

    assert result["actualStatus"] == "completed"
    assert result["hasCompleted"] is True


def test_max_age_running_without_process_marked_failed(monkeypatch):
    monkeypatch.setattr(health, "find_processes_for_job", lambda *_args, **_kwargs: [])
    created_at = "2026-06-01T00:00:00+00:00"
    result = health.inspect_evaluation_runtime_health(
        "ct_eval_20260613_204905_626d",
        "/tmp/missing/runtime/path",
        declared_status="running",
        created_at=created_at,
    )
    assert result["actualStatus"] == "failed"
    assert health.MAX_AGE_FAILURE_REASON in result["reason"]


def test_missing_runtime_path_marked_failed(monkeypatch):
    monkeypatch.setattr(health, "find_processes_for_job", lambda *_args, **_kwargs: [])
    result = health.inspect_evaluation_runtime_health(
        "ct_eval_20260617_132553_c0bb",
        "/tmp/pytest-of-ubuntu/missing/job",
        declared_status="running",
        created_at="2026-06-17T05:25:53+00:00",
    )
    assert result["actualStatus"] == "failed"
    assert result["reason"] in {
        health.RUNTIME_MISSING_FAILURE_REASON,
        health.MAX_AGE_FAILURE_REASON,
    }
