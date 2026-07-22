from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.workspace_job import WorkspaceArtifact, WorkspaceJob
from app.services import workspace_job_service as svc
from app.services.workspace_job_service import RuntimeDeleteFailedError, WorkspaceJobDeleteError


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def runtime_env(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runs"
    runtime_root.mkdir()
    monkeypatch.setattr(svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(svc, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(
        svc,
        "FORBIDDEN_RUNTIME_DELETE_TARGETS",
        (
            tmp_path,
            tmp_path / "backend",
            tmp_path / "frontend",
            runtime_root,
        ),
    )
    return runtime_root


def test_delete_workspace_job_async_full_delete(db_session, runtime_env):
    job_dir = runtime_env / "training" / "jobs" / "train_test_delete"
    job_dir.mkdir(parents=True)
    (job_dir / "logs").mkdir()
    (job_dir / "logs" / "train.log").write_text("epoch 1", encoding="utf-8")

    job = WorkspaceJob(
        job_id="train_test_delete",
        job_type="training",
        task_type="unknown",
        task_name="test",
        status="completed",
        source="real",
        runner="train_bc.py",
        runtime_path=str(job_dir),
    )
    db_session.add(job)
    db_session.add(
        WorkspaceArtifact(
            job_id="train_test_delete",
            artifact_type="log",
            name="train.log",
            file_path=str(job_dir / "logs" / "train.log"),
        )
    )
    db_session.commit()

    result = asyncio.run(svc.delete_workspace_job_async(db_session, "train_test_delete"))
    db_session.commit()

    assert result is not None
    assert result["deletedJob"] is True
    assert result["deletedArtifacts"] == 1
    assert result.get("deletedModelAssets", 0) == 0
    assert result["runtimeDeleted"] is True
    assert result["canReindexRecover"] is False
    assert not job_dir.exists()

    remaining_job = db_session.scalar(
        select(WorkspaceJob).where(WorkspaceJob.job_id == "train_test_delete")
    )
    remaining_artifacts = db_session.scalars(
        select(WorkspaceArtifact).where(WorkspaceArtifact.job_id == "train_test_delete")
    ).all()
    assert remaining_job is None
    assert remaining_artifacts == []


def test_delete_workspace_job_async_missing_runtime_path_still_deletes_db(
    db_session, runtime_env
):
    missing_dir = runtime_env / "cable_threading" / "jobs" / "ct_gen_missing"
    job = WorkspaceJob(
        job_id="ct_gen_missing",
        job_type="generate",
        task_type="cable_threading",
        task_name="test",
        status="completed",
        source="real",
        runner="generate.py",
        runtime_path=str(missing_dir),
    )
    db_session.add(job)
    db_session.commit()

    result = asyncio.run(svc.delete_workspace_job_async(db_session, "ct_gen_missing"))
    db_session.commit()

    assert result is not None
    assert result["runtimeDeleted"] is False
    assert result["reason"] == "runtime_path_not_found"
    assert db_session.scalar(select(WorkspaceJob).where(WorkspaceJob.job_id == "ct_gen_missing")) is None


def test_delete_workspace_job_async_returns_none_when_missing(db_session):
    result = asyncio.run(svc.delete_workspace_job_async(db_session, "missing_job"))
    assert result is None


def test_delete_workspace_job_async_skips_unsafe_runtime_path_but_deletes_db(
    db_session, runtime_env
):
    job = WorkspaceJob(
        job_id="train_unsafe",
        job_type="training",
        task_type="unknown",
        task_name="test",
        status="completed",
        source="real",
        runner="train_bc.py",
        runtime_path=str(runtime_env.parent),
    )
    db_session.add(job)
    db_session.commit()

    result = asyncio.run(svc.delete_workspace_job_async(db_session, "train_unsafe"))
    db_session.commit()

    assert result is not None
    assert result["deletedJob"] is True
    assert result["runtimeDeleted"] is False
    assert result["reason"] == "unsafe_runtime_path"
    assert db_session.scalar(select(WorkspaceJob).where(WorkspaceJob.job_id == "train_unsafe")) is None


def test_delete_workspace_job_async_skips_runtime_root_but_deletes_db(db_session, runtime_env):
    job = WorkspaceJob(
        job_id="train_root",
        job_type="training",
        task_type="unknown",
        task_name="test",
        status="completed",
        source="real",
        runner="train_bc.py",
        runtime_path=str(runtime_env),
    )
    db_session.add(job)
    db_session.commit()

    result = asyncio.run(svc.delete_workspace_job_async(db_session, "train_root"))
    db_session.commit()

    assert result is not None
    assert result["deletedJob"] is True
    assert result["runtimeDeleted"] is False
    assert result["reason"] == "unsafe_runtime_path"
    assert db_session.scalar(select(WorkspaceJob).where(WorkspaceJob.job_id == "train_root")) is None


def test_validate_runtime_delete_path_rejects_path_traversal(runtime_env, monkeypatch):
    outside = runtime_env.parent / "outside"
    outside.mkdir()
    monkeypatch.setattr(
        svc,
        "FORBIDDEN_RUNTIME_DELETE_TARGETS",
        (
            svc.PROJECT_ROOT,
            svc.PROJECT_ROOT / "backend",
            svc.PROJECT_ROOT / "frontend",
            runtime_env,
        ),
    )
    with pytest.raises(WorkspaceJobDeleteError, match="unsafe path"):
        svc._validate_runtime_delete_path(str(outside))


def test_delete_workspace_job_async_runtime_delete_failure_keeps_db(
    db_session, runtime_env, monkeypatch
):
    job_dir = runtime_env / "evaluation" / "jobs" / "eval_test_fail"
    job_dir.mkdir(parents=True)
    job = WorkspaceJob(
        job_id="eval_test_fail",
        job_type="evaluation",
        task_type="dual_arm_cable_manipulation",
        task_name="test",
        status="completed",
        source="real",
        runner="eval.py",
        runtime_path=str(job_dir),
    )
    db_session.add(job)
    db_session.commit()

    def _boom(_target):
        raise OSError("permission denied")

    monkeypatch.setattr(svc.shutil, "rmtree", _boom)

    with pytest.raises(RuntimeDeleteFailedError):
        asyncio.run(svc.delete_workspace_job_async(db_session, "eval_test_fail"))

    assert db_session.scalar(select(WorkspaceJob).where(WorkspaceJob.job_id == "eval_test_fail")) is not None
    assert job_dir.exists()
