from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.database import SessionLocal
from app.models.workspace_job import WorkspaceJob
from app.services.eval_job_db_service import (
    delete_pending_evaluation_record,
    list_evaluation_jobs_from_db,
)


def test_delete_pending_evaluation_record_soft_deletes():
    with SessionLocal() as db:
        row = (
            db.query(WorkspaceJob)
            .filter(
                WorkspaceJob.job_type == "evaluation",
                WorkspaceJob.status.in_(["pending", "queued"]),
            )
            .order_by(WorkspaceJob.id.desc())
            .first()
        )
        assert row is not None
        record_id = int(row.id)
        job_id = row.job_id

    result = delete_pending_evaluation_record(record_id)
    assert result["deleted"] is True
    assert result["workspaceJobId"] == record_id
    assert result["jobId"] == job_id

    with SessionLocal() as db:
        updated = db.query(WorkspaceJob).filter(WorkspaceJob.id == record_id).one()
        assert updated.status == "deleted"

    rows = list_evaluation_jobs_from_db(sync_stale=False)
    assert all(item.get("evalJobId") != job_id for item in rows)


def test_delete_pending_rejects_completed_job():
    from app.services.evaluation.job_paths import is_valid_eval_job_id_format

    with SessionLocal() as db:
        rows = (
            db.query(WorkspaceJob)
            .filter(
                WorkspaceJob.job_type == "evaluation",
                WorkspaceJob.status == "completed",
            )
            .all()
        )
        row = next((r for r in rows if is_valid_eval_job_id_format(str(r.job_id or ""))), None)
        assert row is not None
        record_id = int(row.id)

    with pytest.raises(HTTPException) as exc:
        delete_pending_evaluation_record(record_id)
    assert exc.value.status_code == 400


def test_list_includes_workspace_job_id():
    rows = list_evaluation_jobs_from_db(sync_stale=False)
    if not rows:
        pytest.skip("no evaluation jobs in db")
    assert rows[0].get("workspaceJobId") is not None
