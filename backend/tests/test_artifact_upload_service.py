"""artifact 上传服务与 registry 幂等测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.artifact_storage_object import ArtifactStorageObject
from app.models.workspace_index import EvalMetricSummary, ModelAsset
from app.models.workspace_job import WorkspaceArtifact, WorkspaceJob
from app.services import artifact_storage_registry as registry
from app.services import artifact_upload_service as upload_svc


@pytest.fixture()
def upload_env(tmp_path, monkeypatch):
    from sqlalchemy import BigInteger
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, _compiler, **_kw):
        return "JSON"

    @compiles(BigInteger, "sqlite")
    def _compile_bigint_sqlite(_type, _compiler, **_kw):
        return "INTEGER"

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    monkeypatch.setattr(registry, "SessionLocal", TestSession)
    monkeypatch.setattr(upload_svc, "SessionLocal", TestSession)
    monkeypatch.setattr(upload_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(upload_svc, "RUNTIME_ROOT", tmp_path / "runs")
    monkeypatch.setattr(upload_svc, "artifact_upload_enabled", lambda: True)

    uploaded: list[tuple[str, str]] = []

    def _fake_upload(local_path, object_key, *, bucket=None, content_type=None):
        uploaded.append((str(local_path), object_key))
        return f"minio://{bucket or 'test-bucket'}/{object_key.lstrip('/')}"

    monkeypatch.setattr(
        "app.services.storage.storage_service.StorageService.upload_file",
        staticmethod(_fake_upload),
    )
    return tmp_path, TestSession, uploaded


def test_registry_skip_upload_when_already_on_minio(upload_env):
    tmp_path, TestSession, _ = upload_env
    local = tmp_path / "demo.ckpt"
    local.write_bytes(b"x")
    registry.register_artifact_pending(
        owner_type="train",
        owner_id="train_1",
        artifact_type="checkpoint_final",
        content_key="final",
        local_path=local,
    )
    registry.mark_artifact_uploaded(
        owner_type="train",
        owner_id="train_1",
        artifact_type="checkpoint_final",
        content_key="final",
        storage_uri="minio://eai-checkpoints/checkpoints/train_1/demo.ckpt",
        local_path=local,
    )
    row = registry.get_artifact_record(
        owner_type="train",
        owner_id="train_1",
        artifact_type="checkpoint_final",
        content_key="final",
    )
    assert registry.should_skip_upload(row) is True


def test_upload_training_checkpoints_updates_model_asset(upload_env):
    tmp_path, TestSession, uploaded = upload_env
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"checkpoint-bytes")
    train_job_id = "train_demo_001"

    with TestSession() as db:
        db.add(
            WorkspaceJob(
                id=1,
                job_id=train_job_id,
                job_type="training",
                task_type="cable_threading",
                status="completed",
                runtime_path=str(tmp_path),
            )
        )
        db.add(
            ModelAsset(
                model_asset_id=f"{train_job_id}_final",
                train_job_id=train_job_id,
                model_name="demo",
                asset_type="final",
                storage_uri=f"file://{ckpt}",
                status="ready",
            )
        )
        db.commit()

    result = upload_svc.upload_training_checkpoints(train_job_id)
    assert result["uploaded"] == 1
    assert uploaded
    with TestSession() as db:
        row = db.query(ModelAsset).filter(ModelAsset.model_asset_id == f"{train_job_id}_final").one()
        assert str(row.storage_uri).startswith("minio://")
    with TestSession() as db:
        obj = db.query(ArtifactStorageObject).one()
        assert obj.status == "uploaded"
        assert obj.owner_type == "train"


def test_upload_evaluation_artifacts(upload_env):
    tmp_path, TestSession, uploaded = upload_env
    eval_job_id = "eval_demo_001"
    job_root = tmp_path / "runs" / "evaluations" / "jobs" / eval_job_id
    (job_root / "results").mkdir(parents=True)
    aggregate = {"successRate": 0.8, "jobId": eval_job_id}
    (job_root / "results" / "aggregate_result.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    (job_root / "videos").mkdir()
    (job_root / "videos" / "eval.mp4").write_bytes(b"fake-video")

    with TestSession() as db:
        db.add(
            WorkspaceJob(
                id=1,
                job_id=eval_job_id,
                job_type="evaluation",
                task_type="dual_arm_cable_manipulation",
                status="completed",
                runtime_path=str(job_root),
            )
        )
        db.commit()

    result = upload_svc.upload_evaluation_artifacts(eval_job_id, job_root=job_root)
    assert result["uploaded"] >= 2
    with TestSession() as db:
        summary = db.query(EvalMetricSummary).filter(EvalMetricSummary.job_id == eval_job_id).one()
        assert summary.report_uri.startswith("minio://")
        assert summary.replay_uri.startswith("minio://")
        artifacts = db.query(WorkspaceArtifact).filter(WorkspaceArtifact.job_id == eval_job_id).all()
        assert any(str(a.url_path).startswith("minio://") for a in artifacts)


def test_upload_idempotent_skips_second_run(upload_env):
    tmp_path, TestSession, uploaded = upload_env
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"checkpoint-bytes")
    train_job_id = "train_demo_002"

    with TestSession() as db:
        db.add(
            WorkspaceJob(
                id=1,
                job_id=train_job_id,
                job_type="training",
                task_type="cable_threading",
                status="completed",
                runtime_path=str(tmp_path),
            )
        )
        db.add(
            ModelAsset(
                model_asset_id=f"{train_job_id}_final",
                train_job_id=train_job_id,
                model_name="demo",
                asset_type="final",
                storage_uri=f"file://{ckpt}",
                status="ready",
            )
        )
        db.commit()

    upload_svc.upload_training_checkpoints(train_job_id)
    first_count = len(uploaded)
    upload_svc.upload_training_checkpoints(train_job_id)
    assert len(uploaded) == first_count
