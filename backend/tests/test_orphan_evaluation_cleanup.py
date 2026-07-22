from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.database import SessionLocal
from app.models.workspace_job import WorkspaceJob
from app.services.eval_job_db_service import (
    delete_pending_evaluation_record,
    is_valid_evaluation_list_item,
    list_evaluation_jobs_from_db,
)
from app.services.evaluation.job_paths import is_valid_eval_job_id_format


def test_is_valid_eval_job_id_format():
    assert is_valid_eval_job_id_format("ct_eval_20260624_091529_e0b0") is True
    assert is_valid_eval_job_id_format("eval_20260624_091522_5819") is True
    assert is_valid_eval_job_id_format("isaac_eval_20260617_095040_b265") is True
    assert is_valid_eval_job_id_format("ct_eval_smoke200_20260623") is True
    assert is_valid_eval_job_id_format("线缆穿杆评测任务_20260624_0903") is False


def test_is_valid_evaluation_list_item():
    assert is_valid_evaluation_list_item({"evalJobId": "ct_eval_20260624_091529_e0b0"}) is True
    assert is_valid_evaluation_list_item({"workspaceJobId": 123, "evalJobId": "ct_eval_smoke200_20260623"}) is True
    assert is_valid_evaluation_list_item({"evalJobId": "ct_eval_smoke200_20260623"}) is False
    assert is_valid_evaluation_list_item({"taskName": "线缆穿杆评测任务_20260624_0903"}) is False


def test_list_purges_orphan_unknown_smoke_job():
    with SessionLocal() as db:
        row = (
            db.query(WorkspaceJob)
            .filter(
                WorkspaceJob.job_type == "evaluation",
                WorkspaceJob.job_id == "ct_eval_smoke200_20260623",
            )
            .one_or_none()
        )
        if row is None or row.status == "deleted":
            pytest.skip("orphan smoke job already removed")

    rows = list_evaluation_jobs_from_db(sync_stale=False)
    assert all(item.get("evalJobId") != "ct_eval_smoke200_20260623" for item in rows)

    with SessionLocal() as db:
        updated = (
            db.query(WorkspaceJob)
            .filter(WorkspaceJob.job_id == "ct_eval_smoke200_20260623")
            .one_or_none()
        )
        assert updated is not None
        assert updated.status == "deleted"


def test_delete_unknown_orphan_by_workspace_job_id():
    with SessionLocal() as db:
        row = WorkspaceJob(
            job_id="ct_eval_orphan_test_20260624",
            job_type="evaluation",
            task_type="cable_threading",
            task_name="orphan test",
            status="unknown",
            source="test",
            runner="test",
            runtime_path="runs/test/orphan",
            metadata_json={},
            metrics_json={},
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        record_id = int(row.id)

    try:
        result = delete_pending_evaluation_record(record_id)
        assert result["deleted"] is True
        assert result["workspaceJobId"] == record_id
    finally:
        with SessionLocal() as db:
            db.query(WorkspaceJob).filter(WorkspaceJob.id == record_id).delete()
            db.commit()
