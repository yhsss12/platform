from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.api_timing import log_api_duration, paginate_rows
from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.training import (
    CreateTrainingJobRequest,
    CreateTrainingJobResponse,
    TrainingCapabilitiesResponse,
    TrainingJobDeleteResponse,
    TrainingJobListItem,
    TrainingJobListResponse,
    TrainingJobLogResponse,
    TrainingJobModelResponse,
    TrainingJobStatusResponse,
)
from app.schemas.training_nodes import TrainingNodeListItem, TrainingNodeProbeResponse, TrainingNodesListResponse
from app.services import training_node_service as node_svc
from app.services import training_service as svc
from app.services import workspace_job_service as job_svc
from app.services.audit_service import log_audit_safe
from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache
from app.services.workspace_job_service import WorkspaceJobDeleteError
from app.services.workspace_model_asset_list_cache import invalidate_model_asset_list_cache

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/capabilities", response_model=TrainingCapabilitiesResponse)
async def get_training_capabilities(_: User = Depends(get_current_user)) -> TrainingCapabilitiesResponse:
    result = await asyncio.to_thread(svc.probe_training_capabilities)
    return TrainingCapabilitiesResponse(**result)


@router.get("/nodes", response_model=TrainingNodesListResponse)
async def list_training_nodes(
    refresh: bool = False,
    _: User = Depends(get_current_user),
) -> TrainingNodesListResponse:
    rows = await asyncio.to_thread(node_svc.list_training_nodes, refresh=refresh)
    nodes = [TrainingNodeListItem(**row) for row in rows]
    return TrainingNodesListResponse(nodes=nodes)


@router.get("/nodes/{node_id}", response_model=TrainingNodeProbeResponse)
async def get_training_node_status(
    node_id: str,
    refresh: bool = False,
    _: User = Depends(get_current_user),
) -> TrainingNodeProbeResponse:
    row = await asyncio.to_thread(node_svc.probe_training_node, node_id, refresh=refresh)
    return TrainingNodeProbeResponse(node=TrainingNodeListItem(**row))


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    limit: int = Query(10, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    _: User = Depends(get_current_user),
) -> TrainingJobListResponse:
    with log_api_duration("GET /workspace/training/jobs", limit=limit, offset=offset):
        rows = await asyncio.to_thread(svc.list_training_jobs)
        q = (search or "").strip().lower()
        status_val = (status or "").strip()
        model_val = (model or "").strip()
        filtered = rows
        if status_val:
            filtered = [row for row in filtered if str(row.get("status") or "") == status_val]
        if model_val:
            filtered = [
                row
                for row in filtered
                if model_val
                in " ".join(
                    str(row.get(key) or "")
                    for key in ("downstreamModelType", "trainingBackend", "taskName")
                )
            ]
        if q:
            filtered = [
                row
                for row in filtered
                if q
                in " ".join(
                    str(row.get(key) or "")
                    for key in (
                        "trainJobId",
                        "taskName",
                        "datasetName",
                        "datasetId",
                        "message",
                        "downstreamModelType",
                        "trainingBackend",
                    )
                ).lower()
            ]
        total = len(filtered)
        page_rows = paginate_rows(filtered, limit=limit, offset=offset)
        items: list[TrainingJobListItem] = []
        for row in page_rows:
            try:
                items.append(TrainingJobListItem(**row))
            except ValidationError as exc:
                logger.warning(
                    "skip invalid training job list row trainJobId=%s: %s",
                    row.get("trainJobId"),
                    exc,
                )
    return TrainingJobListResponse(jobs=items, total=total)


@router.post("/jobs", response_model=CreateTrainingJobResponse)
async def create_training_job(
    payload: CreateTrainingJobRequest,
    _: User = Depends(get_current_user),
) -> CreateTrainingJobResponse:
    result = await asyncio.to_thread(svc.create_training_job, payload.model_dump())
    logger.info("training job created id=%s status=%s", result.get("trainJobId"), result.get("status"))
    return CreateTrainingJobResponse(**result)


@router.get("/jobs/{train_job_id}/status", response_model=TrainingJobStatusResponse)
async def get_training_job_status(
    train_job_id: str,
    _: User = Depends(get_current_user),
) -> TrainingJobStatusResponse:
    result = await asyncio.to_thread(svc.get_training_job_status, train_job_id)
    return TrainingJobStatusResponse(**result)


@router.get("/jobs/{train_job_id}/log", response_model=TrainingJobLogResponse)
async def get_training_job_log(
    train_job_id: str,
    _: User = Depends(get_current_user),
) -> TrainingJobLogResponse:
    log = await asyncio.to_thread(svc.read_training_job_log, train_job_id)
    return TrainingJobLogResponse(trainJobId=train_job_id, log=log)


@router.get("/jobs/{train_job_id}/model", response_model=TrainingJobModelResponse)
async def get_training_job_model(
    train_job_id: str,
    _: User = Depends(get_current_user),
) -> TrainingJobModelResponse:
    result = await asyncio.to_thread(svc.get_training_job_model, train_job_id)
    return TrainingJobModelResponse(**result)


@router.delete("/jobs/{train_job_id}", response_model=TrainingJobDeleteResponse)
async def delete_training_job(
    train_job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TrainingJobDeleteResponse:
    """Hard-delete training job: DB rows, artifacts, model assets, and runtime directory.

    Reuses delete_workspace_job_async (same path as DELETE /workspace/jobs/{jobId}).
    Falls back to disk-only cleanup when the job exists under runs but not in DB.
    """
    sanitized = svc._sanitize_train_job_id_for_delete(train_job_id)
    deleted_at = datetime.now(timezone.utc).isoformat()
    try:
        result = await job_svc.delete_workspace_job_async(db, sanitized)
        if result is None:
            disk_deleted = await asyncio.to_thread(svc.hard_delete_training_job_disk_only, sanitized)
            if not disk_deleted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="training job not found",
                )
            invalidate_model_asset_list_cache()
            logger.info("training job disk-only hard-deleted id=%s", sanitized)
            return TrainingJobDeleteResponse(
                trainJobId=sanitized,
                deleted=True,
                deletedAt=deleted_at,
            )

        await db.commit()
        invalidate_workspace_dataset_list_cache()
        invalidate_model_asset_list_cache()
        log_audit_safe(
            user=user,
            action_type="workspace_job_delete",
            action_label="删除训练任务",
            resource_type="workspace_job",
            resource_id=sanitized,
            detail_json={
                "action": "training_job_hard_delete",
                "jobId": sanitized,
                "jobType": result.get("jobType"),
                "taskType": result.get("taskType"),
                "runtimePath": result.get("runtimePath"),
                "deletedArtifacts": result.get("deletedArtifacts"),
                "deletedModelAssets": result.get("deletedModelAssets"),
                "runtimeDeleted": result.get("runtimeDeleted"),
                "operator": user.username,
            },
        )
        logger.info(
            "training job hard-deleted id=%s runtime_deleted=%s artifacts=%s model_assets=%s",
            sanitized,
            result.get("runtimeDeleted"),
            result.get("deletedArtifacts"),
            result.get("deletedModelAssets"),
        )
        return TrainingJobDeleteResponse(
            trainJobId=sanitized,
            deleted=True,
            deletedAt=deleted_at,
        )
    except WorkspaceJobDeleteError as exc:
        await db.rollback()
        logger.warning("delete_training_job rejected id=%s reason=%s", sanitized, exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception("delete_training_job failed id=%s", sanitized)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"training job delete failed: {exc}",
        ) from exc
