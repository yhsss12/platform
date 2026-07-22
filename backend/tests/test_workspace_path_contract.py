from pathlib import Path

import pytest

from app.services import workspace_job_service as job_service


def test_delete_boundary_accepts_configured_runtime_job(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "data" / "runs"
    job_root = runtime_root / "training" / "jobs" / "train_1"
    monkeypatch.setattr(job_service, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(job_service, "FORBIDDEN_RUNTIME_DELETE_TARGETS", (runtime_root,))

    assert job_service._validate_runtime_delete_path(str(job_root)) == job_root.resolve()


def test_delete_boundary_rejects_job_outside_configured_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    other_root = tmp_path / "code" / "runs"
    old_job_root = other_root / "training" / "jobs" / "train_old"
    monkeypatch.setattr(job_service, "RUNTIME_ROOT", tmp_path / "data" / "runs")
    monkeypatch.setattr(job_service, "FORBIDDEN_RUNTIME_DELETE_TARGETS", ())

    with pytest.raises(job_service.WorkspaceJobDeleteError, match="unsafe path"):
        job_service._validate_runtime_delete_path(str(old_job_root))


def test_delete_boundary_rejects_runtime_prefix_collision(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "data" / "runs"
    monkeypatch.setattr(job_service, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(job_service, "FORBIDDEN_RUNTIME_DELETE_TARGETS", ())

    with pytest.raises(job_service.WorkspaceJobDeleteError, match="unsafe path"):
        job_service._validate_runtime_delete_path(
            str(tmp_path / "data" / "runs-escape" / "training" / "train_1")
        )


def test_delete_boundary_rejects_runtime_root_itself(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "data" / "runs"
    monkeypatch.setattr(job_service, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(job_service, "FORBIDDEN_RUNTIME_DELETE_TARGETS", ())

    with pytest.raises(job_service.WorkspaceJobDeleteError, match="forbidden path"):
        job_service._validate_runtime_delete_path(str(runtime_root))
