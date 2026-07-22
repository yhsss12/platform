from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.workspace_index import ModelAsset, TrainingMetricSummary
from app.models.workspace_job import WorkspaceJob
from app.services import training_job_sync_service as sync_svc
from app.services import workspace_model_asset_service as asset_svc
from app.services.checkpoint_registry import register_checkpoint_assets
from app.services.model_asset_db_service import list_model_assets_for_job_from_db


@pytest.fixture()
def sync_env(tmp_path, monkeypatch):
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, _compiler, **_kw):
        return "JSON"

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    jobs_root = tmp_path / "runs" / "training" / "jobs"
    jobs_root.mkdir(parents=True)

    monkeypatch.setattr(sync_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sync_svc, "TRAINING_JOBS_ROOT", jobs_root)
    monkeypatch.setattr(asset_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(asset_svc, "TRAINING_JOBS_ROOT", jobs_root)
    monkeypatch.setattr(sync_svc, "SessionLocal", TestSession)
    monkeypatch.setattr("app.core.database.SessionLocal", TestSession)
    monkeypatch.setattr("app.core.db_session.SessionLocal", TestSession)
    monkeypatch.setattr("app.services.model_asset_db_service.SessionLocal", TestSession)
    monkeypatch.setattr("app.services.workspace_job_service.SessionLocal", TestSession)
    monkeypatch.setattr("app.services.workspace_job_service.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.services.workspace_job_service.RUNTIME_ROOT", tmp_path / "runs")
    monkeypatch.setattr(
        "app.services.workspace_job_service._upsert_artifacts",
        lambda *args, **kwargs: 0,
    )

    return jobs_root, TestSession


def _write_completed_job(jobs_root: Path, train_job_id: str) -> Path:
    train_job_dir = jobs_root / train_job_id
    ckpt_dir = train_job_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (train_job_dir / "config").mkdir(parents=True)
    (train_job_dir / "artifacts").mkdir(parents=True)
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"epochs": 2, "saveFinal": True, "taskName": "demo", "datasetId": "ds1"}),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps(
            {
                "trainJobId": train_job_id,
                "status": "completed",
                "totalEpochs": 2,
                "epoch": 2,
                "datasetId": "ds1",
                "datasetName": "demo-ds",
            }
        ),
        encoding="utf-8",
    )
    (train_job_dir / "artifacts" / "dataset_manifest.json").write_text(
        json.dumps({"datasetId": "ds1", "taskType": "cable_threading"}),
        encoding="utf-8",
    )
    metrics_path = train_job_dir / "artifacts" / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"epoch": 1, "trainLoss": 0.5}),
                json.dumps({"epoch": 2, "trainLoss": 0.2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ckpt_dir / "model_final.pth").write_bytes(b"final-ckpt")
    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest={"datasetId": "ds1", "taskType": "cable_threading"},
        train_config={"epochs": 2, "saveFinal": True, "taskName": "demo"},
        status={"status": "completed", "totalEpochs": 2, "epoch": 2},
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=True,
    )
    return train_job_dir


def _seed_workspace_job(session_factory, train_job_id: str, train_job_dir: Path) -> None:
    """SQLite 测试库 BigInteger 自增兼容：预置 workspace_jobs 行供 sync 更新。"""
    with session_factory() as db:
        existing = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == train_job_id).one_or_none()
        if existing is not None:
            return
        db.add(
            WorkspaceJob(
                id=1,
                job_id=train_job_id,
                job_type="training",
                task_type="cable_threading",
                task_name="demo",
                status="running",
                source="real",
                runner="train_bc.py",
                runtime_path=str(train_job_dir),
            )
        )
        db.commit()


def test_sync_training_job_populates_db(sync_env):
    jobs_root, Session = sync_env
    train_job_id = "train_20260620_150000_abcd"
    train_job_dir = _write_completed_job(jobs_root, train_job_id)
    _seed_workspace_job(Session, train_job_id, train_job_dir)

    sync_svc.sync_training_job_from_runtime(train_job_id)

    summary = sync_svc.get_training_job_summary_from_db(train_job_id)
    assert summary is not None
    assert summary["status"] == "completed"
    assert summary["epoch"] == 2
    assert summary["totalEpochs"] == 2
    assert summary["loss"] == pytest.approx(0.2)
    assert len(summary.get("lossHistory") or []) >= 2

    assets = list_model_assets_for_job_from_db(train_job_id)
    assert any(item.get("checkpointKind") == "final" for item in assets)

    with sync_svc.SessionLocal() as db:
        metric_row = db.query(TrainingMetricSummary).filter_by(job_id=train_job_id).one()
        assert metric_row.best_loss == pytest.approx(0.2)
        assert metric_row.final_loss == pytest.approx(0.2)
        job_row = db.query(WorkspaceJob).filter_by(job_id=train_job_id).one()
        assert job_row.job_type == "training"
        model_rows = db.query(ModelAsset).filter_by(train_job_id=train_job_id).all()
        assert any(row.asset_type == "final" and row.status == "ready" for row in model_rows)


def test_list_training_jobs_from_db_excludes_deleted(sync_env):
    jobs_root, Session = sync_env
    visible_id = "train_20260620_150001_abcd"
    hidden_id = "train_20260620_150002_abcd"
    visible_dir = _write_completed_job(jobs_root, visible_id)
    hidden_dir = _write_completed_job(jobs_root, hidden_id)
    _seed_workspace_job(Session, visible_id, visible_dir)
    with Session() as db:
        db.add(
            WorkspaceJob(
                id=2,
                job_id=hidden_id,
                job_type="training",
                task_type="cable_threading",
                task_name="demo",
                status="running",
                source="real",
                runner="train_bc.py",
                runtime_path=str(hidden_dir),
            )
        )
        db.commit()
    (hidden_dir / "status.json").write_text(
        json.dumps({"trainJobId": hidden_id, "status": "completed", "deleted": True}),
        encoding="utf-8",
    )

    sync_svc.sync_training_job_from_runtime(visible_id)
    sync_svc.sync_training_job_from_runtime(hidden_id)

    rows = sync_svc.list_training_jobs_from_db(sync_stale=False)
    ids = {row["trainJobId"] for row in rows}
    assert visible_id in ids
    assert hidden_id not in ids


def test_get_training_job_summary_maps_pending_status(sync_env):
    jobs_root, Session = sync_env
    job_id = "train_20260620_150004_abcd"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(
        json.dumps({"trainJobId": job_id, "status": "pending", "totalEpochs": 1, "epoch": 0}),
        encoding="utf-8",
    )
    with Session() as db:
        db.add(
            WorkspaceJob(
                id=3,
                job_id=job_id,
                job_type="training",
                task_type="cable_threading",
                task_name="demo",
                status="pending",
                source="real",
                runner="train_act.py",
                runtime_path=str(job_dir),
            )
        )
        db.commit()

    summary = sync_svc.get_training_job_summary_from_db(job_id)
    assert summary is not None
    assert summary["status"] == "queued"


def test_model_assets_detail_from_db_after_sync(sync_env):
    jobs_root, Session = sync_env
    train_job_id = "train_20260620_150003_abcd"
    train_job_dir = _write_completed_job(jobs_root, train_job_id)
    _seed_workspace_job(Session, train_job_id, train_job_dir)
    sync_svc.sync_training_job_from_runtime(train_job_id)

    payload = asset_svc.list_training_job_model_assets_detail(train_job_id)
    assets = payload.get("modelAssets") or []
    assert assets
    final_row = next(item for item in assets if item.get("checkpointKind") == "final")
    assert final_row.get("canEvaluate") is True
    assert final_row.get("displayStatus") == "ready"


def test_summary_uses_metadata_dataset_name_when_metrics_empty(sync_env):
    _, Session = sync_env
    train_job_id = "train_20260620_150004_abcd"
    with Session() as db:
        db.add(
            WorkspaceJob(
                id=3,
                job_id=train_job_id,
                job_type="training",
                task_type="cable_threading",
                task_name="线缆穿杆训练",
                status="completed",
                source="real",
                runner="train_bc.py",
                runtime_path="/missing/runtime/path",
                metadata_json={
                    "datasetName": "线缆穿杆数据_20260620_1200",
                    "datasetId": "ds_meta",
                    "trainConfig": {"downstreamModelType": "bc", "trainingBackend": "robomimic_bc"},
                },
                metrics_json={},
            )
        )
        db.commit()

    summary = sync_svc.get_training_job_summary_from_db(train_job_id)
    assert summary is not None
    assert summary["datasetName"] == "线缆穿杆数据_20260620_1200"
    assert summary["datasetId"] == "ds_meta"
    assert summary["runtimeAvailable"] is False
    assert summary["status"] == "completed"
    assert "运行时工作目录不可用" in str(summary.get("message") or "")


def test_summary_marks_stale_running_as_failed_without_runtime(sync_env):
    _, Session = sync_env
    train_job_id = "train_20260620_150005_abcd"
    with Session() as db:
        db.add(
            WorkspaceJob(
                id=4,
                job_id=train_job_id,
                job_type="training",
                task_type="cable_threading",
                task_name="demo",
                status="running",
                source="real",
                runner="train_bc.py",
                runtime_path="/missing/runtime/path",
                metadata_json={"datasetName": "demo-ds"},
                metrics_json={"epoch": 1, "totalEpochs": 5},
            )
        )
        db.commit()

    summary = sync_svc.get_training_job_summary_from_db(train_job_id)
    assert summary is not None
    assert summary["status"] == "failed"
    assert summary["runtimeAvailable"] is False
