"""训练同步全链路：Final / Best / Epoch + 列表 API 不扫目录。"""

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
from app.services import training_service as train_svc
from app.services import workspace_model_asset_service as asset_svc
from app.services.checkpoint_registry import register_checkpoint_assets
from app.services.model_asset_db_service import list_model_assets_for_job_from_db


@pytest.fixture()
def full_sync_env(tmp_path, monkeypatch):
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, _compiler, **_kw):
        return "JSON"

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    training_root = tmp_path / "runs" / "training"
    jobs_root = training_root / "jobs"
    jobs_root.mkdir(parents=True)

    monkeypatch.setattr(sync_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sync_svc, "TRAINING_JOBS_ROOT", jobs_root)
    monkeypatch.setattr(train_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train_svc, "TRAINING_ROOT", training_root)
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
    monkeypatch.setattr(
        "app.services.checkpoint_archive_service.schedule_training_job_checkpoint_archive",
        lambda *_args, **_kwargs: None,
    )

    return jobs_root, TestSession


def _seed_job(session_factory, train_job_id: str, train_job_dir: Path, *, job_id_num: int = 1) -> None:
    with session_factory() as db:
        db.add(
            WorkspaceJob(
                id=job_id_num,
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


def _write_job_with_all_checkpoints(jobs_root: Path, train_job_id: str) -> Path:
    train_job_dir = jobs_root / train_job_id
    ckpt_dir = train_job_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for sub in ("config", "artifacts", "logs"):
        (train_job_dir / sub).mkdir(parents=True)

    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps(
            {
                "epochs": 3,
                "saveFinal": True,
                "saveBest": True,
                "checkpointIntervalEpochs": 1,
                "taskName": "demo",
                "datasetId": "ds1",
            }
        ),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps(
            {
                "trainJobId": train_job_id,
                "status": "completed",
                "totalEpochs": 3,
                "epoch": 3,
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
    (train_job_dir / "artifacts" / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"epoch": 1, "trainLoss": 0.9}),
                json.dumps({"epoch": 2, "trainLoss": 0.4}),
                json.dumps({"epoch": 3, "trainLoss": 0.15}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (ckpt_dir / "model_epoch_1.pth").write_bytes(b"epoch1")
    (ckpt_dir / "model_epoch_2_best_validation_0.40.pth").write_bytes(b"best")
    (ckpt_dir / "model_final.pth").write_bytes(b"final")

    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest={"datasetId": "ds1", "taskType": "cable_threading"},
        train_config={
            "epochs": 3,
            "saveFinal": True,
            "saveBest": True,
            "checkpointIntervalEpochs": 1,
            "taskName": "demo",
        },
        status={"status": "completed", "totalEpochs": 3, "epoch": 3},
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=True,
    )
    return train_job_dir


def test_full_training_sync_final_best_epoch_and_list_api(full_sync_env, monkeypatch):
    jobs_root, Session = full_sync_env
    train_job_id = "train_20260620_160000_abcd"
    train_job_dir = _write_job_with_all_checkpoints(jobs_root, train_job_id)
    _seed_job(Session, train_job_id, train_job_dir)

    sync_svc.finalize_training_job_sync(train_job_id)

    with Session() as db:
        metric = db.query(TrainingMetricSummary).filter_by(job_id=train_job_id).one()
        assert metric.current_epoch == 3
        assert metric.total_epochs == 3
        assert metric.best_loss == pytest.approx(0.15)
        assert len(metric.loss_series or []) >= 3

        assets = (
            db.query(ModelAsset)
            .filter(ModelAsset.train_job_id == train_job_id, ModelAsset.status != "deleted")
            .all()
        )
        kinds = {row.asset_type for row in assets}
        assert "final" in kinds
        assert "best" in kinds
        assert "epoch" in kinds

        final_rows = [row for row in assets if row.asset_type == "final"]
        assert any(row.status in {"ready", "available"} for row in final_rows)

    db_assets = list_model_assets_for_job_from_db(train_job_id)
    assert any(item.get("checkpointKind") == "final" for item in db_assets)

    detail = asset_svc.list_training_job_model_assets_detail(train_job_id)
    detail_assets = detail.get("modelAssets") or []
    final_detail = next(item for item in detail_assets if item.get("checkpointKind") == "final")
    assert final_detail.get("canEvaluate") is True
    assert final_detail.get("displayStatus") == "ready"

    summary = sync_svc.get_training_job_summary_from_db(train_job_id)
    monkeypatch.setattr(sync_svc, "list_training_jobs_from_db", lambda **_: [summary] if summary else [])

    listed = train_svc.list_training_jobs()
    assert len(listed) == 1
    assert listed[0]["trainJobId"] == train_job_id
    assert listed[0]["status"] == "completed"
    assert listed[0]["epoch"] == 3


def test_reindex_backfills_missing_final_asset(full_sync_env):
    jobs_root, Session = full_sync_env
    train_job_id = "train_20260620_160001_abcd"
    train_job_dir = _write_job_with_all_checkpoints(jobs_root, train_job_id)
    _seed_job(Session, train_job_id, train_job_dir, job_id_num=2)

    result = sync_svc.reindex_runtime_jobs(job_type="training", dry_run=False)
    assert int(result.get("syncedTrainingJobs") or 0) >= 1

    assets = list_model_assets_for_job_from_db(train_job_id)
    assert any(item.get("checkpointKind") == "final" for item in assets)
