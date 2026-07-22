from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from app.services import training_service as train_svc
from app.services.evaluation import evaluation_service as eval_svc


@pytest.fixture()
def training_runtime(tmp_path, monkeypatch):
    training_root = tmp_path / "runs" / "training"
    jobs_root = training_root / "jobs"
    jobs_root.mkdir(parents=True)
    monkeypatch.setattr(train_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train_svc, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(
        "app.services.training_job_sync_service.list_training_jobs_from_db",
        lambda **_: [],
    )
    return jobs_root


@pytest.fixture()
def evaluation_runtime(tmp_path, monkeypatch):
    from app.services.evaluation import job_paths as eval_job_paths

    eval_root = tmp_path / "runs" / "evaluations"
    jobs_root = eval_root / "jobs"
    jobs_root.mkdir(parents=True)
    monkeypatch.setattr(eval_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(eval_svc, "EVAL_OUTPUT_ROOT", eval_root)
    monkeypatch.setattr(eval_job_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(eval_job_paths, "EVAL_OUTPUT_ROOT", eval_root)
    monkeypatch.setattr(
        "app.services.eval_job_db_service.list_evaluation_jobs_from_db",
        lambda **_: [],
    )
    return jobs_root


def _write_train_job(jobs_root, job_id: str, *, deleted: bool = False) -> None:
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "checkpoints").mkdir()
    (job_dir / "checkpoints" / "model.pth").write_text("ckpt", encoding="utf-8")
    payload = {
        "trainJobId": job_id,
        "status": "completed",
        "datasetId": "ds_test",
        "checkpointExists": True,
    }
    if deleted:
        payload["deleted"] = True
        payload["deletedAt"] = "2026-06-15T00:00:00+00:00"
    (job_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_eval_job(jobs_root, job_id: str, *, deleted: bool = False, metadata: bool = False) -> None:
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "videos").mkdir()
    (job_dir / "videos" / "episode_0.mp4").write_text("video", encoding="utf-8")
    payload = {
        "evalJobId": job_id,
        "taskType": "dual_arm_cable_manipulation",
        "evaluationMode": "episode_stability",
        "status": "completed",
        "metrics": {"numEpisodes": 3},
    }
    if deleted:
        payload["deleted"] = True
        payload["deletedAt"] = "2026-06-15T00:00:00+00:00"
    status_path = job_dir / ("metadata/status.json" if metadata else "status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload), encoding="utf-8")


def test_delete_training_job_hard_deletes_runtime(training_runtime, monkeypatch):
    job_id = "train_20260615_120000_abcd"
    _write_train_job(training_runtime, job_id)

    monkeypatch.setattr(
        "app.core.database.SessionLocal",
        lambda: _RaisingSession(),
    )

    result = train_svc.delete_training_job(job_id)
    assert result["deleted"] is True
    assert result["deletedAt"]
    assert result.get("runtimeDeleted") is True
    assert not (training_runtime / job_id).exists()


def test_delete_training_job_allows_legacy_id_format(training_runtime, monkeypatch):
    job_id = "train_20260623_smoke200"
    _write_train_job(training_runtime, job_id)
    monkeypatch.setattr(
        "app.core.database.SessionLocal",
        lambda: _RaisingSession(),
    )
    result = train_svc.delete_training_job(job_id)
    assert result["trainJobId"] == job_id
    assert result["deleted"] is True
    assert not (training_runtime / job_id).exists()


def test_delete_training_job_db_only_without_runtime_dir(training_runtime, monkeypatch):
    job_id = "train_20260623_smoke200"

    class _FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def one_or_none(self):
            return None

        def delete(self, *args, **kwargs):
            return 0

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def query(self, *args, **kwargs):
            return _FakeQuery()

        def delete(self, *args, **kwargs):
            return None

        def commit(self):
            return None

    monkeypatch.setattr("app.core.database.SessionLocal", lambda: _FakeSession())

    with pytest.raises(HTTPException) as exc:
        train_svc.delete_training_job(job_id)
    assert exc.value.status_code == 404


class _RaisingSession:
    """Force sync delete_training_job to take the disk-only hard-delete path."""

    def __enter__(self):
        raise RuntimeError("db unavailable in unit test")

    def __exit__(self, *args):
        return False


def test_list_training_jobs_filters_deleted(training_runtime):
    visible_id = "train_20260615_120001_abcd"
    hidden_id = "train_20260615_120002_abcd"
    _write_train_job(training_runtime, visible_id)
    _write_train_job(training_runtime, hidden_id, deleted=True)
    rows = train_svc.list_training_jobs()
    ids = {row["trainJobId"] for row in rows}
    assert visible_id in ids
    assert hidden_id not in ids


def test_delete_training_job_invalid_id_rejected(training_runtime):
    with pytest.raises(HTTPException) as exc:
        train_svc.delete_training_job("../evil")
    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid training job ID"


def test_delete_training_job_missing_returns_404(training_runtime):
    with pytest.raises(HTTPException) as exc:
        train_svc.delete_training_job("train_20260615_120003_abcd")
    assert exc.value.status_code == 404


def test_delete_evaluation_job_marks_deleted(evaluation_runtime):
    job_id = "eval_20260615_120000_abcd"
    _write_eval_job(evaluation_runtime, job_id)
    result = eval_svc.delete_evaluation_job(job_id)
    assert result["deleted"] is True
    status = json.loads((evaluation_runtime / job_id / "status.json").read_text(encoding="utf-8"))
    assert status["deleted"] is True
    assert (evaluation_runtime / job_id / "videos" / "episode_0.mp4").is_file()


def test_delete_isaac_evaluation_job_marks_deleted(evaluation_runtime):
    job_id = "isaac_eval_20260615_120000_abcd"
    _write_eval_job(evaluation_runtime, job_id)
    result = eval_svc.delete_evaluation_job(job_id)
    assert result["evalJobId"] == job_id
    status = json.loads((evaluation_runtime / job_id / "status.json").read_text(encoding="utf-8"))
    assert status["deleted"] is True


def test_list_evaluation_jobs_filters_deleted(evaluation_runtime):
    visible_id = "eval_20260615_120001_abcd"
    hidden_id = "eval_20260615_120002_abcd"
    _write_eval_job(evaluation_runtime, visible_id)
    _write_eval_job(evaluation_runtime, hidden_id, deleted=True)
    rows = eval_svc.list_evaluation_jobs()
    ids = {row["evalJobId"] for row in rows}
    assert visible_id in ids
    assert hidden_id not in ids


def test_list_evaluation_jobs_reads_metadata_status(evaluation_runtime):
    job_id = "eval_20260615_120003_abcd"
    _write_eval_job(evaluation_runtime, job_id, metadata=True)
    rows = eval_svc.list_evaluation_jobs()
    assert any(row["evalJobId"] == job_id for row in rows)


def test_delete_evaluation_job_invalid_id_rejected(evaluation_runtime):
    with pytest.raises(HTTPException) as exc:
        eval_svc.delete_evaluation_job("bad_id")
    assert exc.value.status_code == 400
