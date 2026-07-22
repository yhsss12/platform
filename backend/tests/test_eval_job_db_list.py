"""评测任务 DB-first 列表与结果读取。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.workspace_index import EvalMetricSummary
from app.models.workspace_job import WorkspaceJob
from app.services.evaluation import evaluation_service as eval_svc
from app.services.training_job_sync_service import sync_eval_job_from_runtime


@pytest.fixture()
def eval_db_env(tmp_path, monkeypatch):
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, _compiler, **_kw):
        return "JSON"

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    eval_root = tmp_path / "runs" / "evaluations"
    jobs_root = eval_root / "jobs"
    jobs_root.mkdir(parents=True)

    from app.services.evaluation import job_paths as eval_job_paths

    monkeypatch.setattr(eval_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(eval_svc, "EVAL_OUTPUT_ROOT", eval_root)
    monkeypatch.setattr(eval_job_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(eval_job_paths, "EVAL_OUTPUT_ROOT", eval_root)
    monkeypatch.setattr("app.services.eval_job_db_service.SessionLocal", TestSession)
    monkeypatch.setattr("app.services.training_job_sync_service.SessionLocal", TestSession)
    monkeypatch.setattr("app.services.workspace_job_service.SessionLocal", TestSession)
    monkeypatch.setattr("app.services.workspace_job_service.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.services.workspace_job_service.RUNTIME_ROOT", tmp_path / "runs")
    monkeypatch.setattr(
        "app.services.workspace_job_service._upsert_artifacts",
        lambda *args, **kwargs: 0,
    )

    return jobs_root, TestSession


def _write_eval_job(jobs_root: Path, job_id: str) -> Path:
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "results").mkdir()
    (job_dir / "videos").mkdir()
    (job_dir / "metadata").mkdir()
    aggregate = {
        "evalJobId": job_id,
        "summary": {"successRate": 0.8, "totalEpisodes": 5},
        "taskMetrics": {"successRate": 0.8},
    }
    (job_dir / "results" / "aggregate_result.json").write_text(
        json.dumps(aggregate),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "evalJobId": job_id,
                "taskType": "dual_arm_cable_manipulation",
                "evaluationMode": "episode_stability",
                "status": "completed",
                "metrics": {"successRate": 0.8},
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "videos" / "episode_0.mp4").write_bytes(b"video")
    return job_dir


def test_evaluation_list_db_first(eval_db_env):
    jobs_root, Session = eval_db_env
    job_id = "eval_20260620_170000_abcd"
    job_dir = _write_eval_job(jobs_root, job_id)

    with Session() as db:
        db.add(
            WorkspaceJob(
                id=1,
                job_id=job_id,
                job_type="evaluation",
                task_type="dual_arm_cable_manipulation",
                task_name="评测任务",
                status="completed",
                source="real",
                runner="dual_arm_cable_eval_worker.py",
                runtime_path=str(job_dir),
            )
        )
        db.commit()

    sync_eval_job_from_runtime(job_id)

    rows = eval_svc.list_evaluation_jobs()
    assert any(row["evalJobId"] == job_id for row in rows)
    match = next(row for row in rows if row["evalJobId"] == job_id)
    assert match["status"] == "completed"
    assert match.get("reportUri")

    result = eval_svc.get_evaluation_result(job_id)
    assert result.get("summary", {}).get("successRate") == 0.8

    with Session() as db:
        summary_row = db.query(EvalMetricSummary).filter_by(job_id=job_id).one()
        assert summary_row.report_uri
        assert summary_row.replay_uri
