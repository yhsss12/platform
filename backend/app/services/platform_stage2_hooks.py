"""Data Platform Stage II sidecar hooks：event + lineage（非阻塞，不改主流程）。"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from app.core.database import SessionLocal
from app.core.events.event_emitter import emit_event
from app.core.events.event_models import EventType
from app.models.workspace_index import ModelAsset
from app.models.workspace_job import WorkspaceJob
from app.services.lineage_service import (
    sync_lineage_for_dataset_job,
    sync_lineage_for_eval_job,
    sync_lineage_for_training_job,
)

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled"})


def after_workspace_job_sync(job_id: str) -> None:
    """sync 完成后异步发射事件 + 写 lineage。"""
    jid = (job_id or "").strip()
    if not jid:
        return
    threading.Thread(target=_stage2_worker, args=(jid,), name=f"stage2-{jid[-10:]}", daemon=True).start()


def emit_artifact_uploaded(
    *,
    job_id: str,
    artifact_type: str,
    storage_uri: str,
    content_key: str = "",
) -> None:
    emit_event(
        EventType.ARTIFACT_UPLOADED,
        job_id,
        payload={
            "artifactType": artifact_type,
            "storageUri": storage_uri,
            "contentKey": content_key,
        },
        source="artifact_upload_worker",
    )


def _stage2_worker(job_id: str) -> None:
    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
            if row is None:
                return
            job_type = row.job_type
            status = row.status
            payload: dict[str, Any] = {
                "jobType": job_type,
                "taskType": row.task_type,
                "status": status,
                "projectId": row.project_id,
            }

        if job_type == "training":
            _emit_training_events(job_id, status, payload)
            sync_lineage_for_training_job(job_id)
            _emit_checkpoint_events(job_id)
        elif job_type == "evaluation":
            _emit_eval_events(job_id, status, payload)
            sync_lineage_for_eval_job(job_id)
        elif job_type == "generate":
            if status in TERMINAL_STATUSES and status == "completed":
                emit_event(EventType.DATASET_INGESTED, job_id, payload=payload, source="sync_hook")
            sync_lineage_for_dataset_job(job_id)
    except Exception as exc:
        logger.warning("stage2 hook failed job_id=%s: %s", job_id, exc)


def _emit_training_events(job_id: str, status: str, payload: dict[str, Any]) -> None:
    if status == "running":
        emit_event(EventType.TRAINING_STARTED, job_id, payload=payload, source="sync_hook")
    if status in TERMINAL_STATUSES:
        if status == "completed":
            emit_event(EventType.TRAINING_COMPLETED, job_id, payload=payload, source="sync_hook")


def _emit_eval_events(job_id: str, status: str, payload: dict[str, Any]) -> None:
    if status == "running":
        emit_event(EventType.EVAL_STARTED, job_id, payload=payload, source="sync_hook")
    if status in TERMINAL_STATUSES:
        if status == "completed":
            emit_event(EventType.EVAL_COMPLETED, job_id, payload=payload, source="sync_hook")


def _emit_checkpoint_events(train_job_id: str) -> None:
    try:
        with SessionLocal() as db:
            assets = (
                db.query(ModelAsset)
                .filter(ModelAsset.train_job_id == train_job_id, ModelAsset.status != "deleted")
                .all()
            )
            for asset in assets:
                emit_event(
                    EventType.CHECKPOINT_CREATED,
                    train_job_id,
                    payload={
                        "modelAssetId": asset.model_asset_id,
                        "checkpointKind": asset.checkpoint_kind or asset.asset_type,
                        "storageUri": asset.storage_uri,
                    },
                    source="sync_hook",
                )
    except Exception as exc:
        logger.warning("checkpoint event emit failed job_id=%s: %s", train_job_id, exc)
