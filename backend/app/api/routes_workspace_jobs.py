from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.workspace_job import (
    WorkspaceJobArtifactsResponse,
    WorkspaceJobDeleteResponse,
    WorkspaceJobDetail,
    WorkspaceJobListResponse,
    WorkspaceReindexRequest,
    WorkspaceReindexResponse,
)
from app.services import workspace_job_service as svc
from app.services.audit_service import log_audit_safe
from app.services.workspace_job_service import WorkspaceJobDeleteError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs", response_model=WorkspaceJobListResponse)
async def list_workspace_jobs(
    jobType: str | None = Query(default=None),
    taskType: str | None = Query(default=None),
    status: str | None = Query(default=None),
    source: str | None = Query(default="real"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceJobListResponse:
    try:
        jobs, total = await svc.list_workspace_jobs_async(
            db,
            job_type=jobType,
            task_type=taskType,
            status=status,
            source=source,
            limit=limit,
            offset=offset,
        )
        return WorkspaceJobListResponse(jobs=jobs, total=total)
    except Exception as exc:
        logger.exception("list_workspace_jobs failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"workspace jobs database unavailable: {exc}",
        ) from exc


@router.get("/jobs/{job_id}", response_model=WorkspaceJobDetail)
async def get_workspace_job(
    job_id: str,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceJobDetail:
    try:
        svc.sync_workspace_job_from_runtime(job_id)
        row = await svc.get_workspace_job_async(db, job_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="workspace job not found")
        return WorkspaceJobDetail(**row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("get_workspace_job failed job_id=%s", job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"workspace jobs database unavailable: {exc}",
        ) from exc


@router.get("/jobs/{job_id}/artifacts", response_model=WorkspaceJobArtifactsResponse)
async def list_workspace_job_artifacts(
    job_id: str,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceJobArtifactsResponse:
    try:
        svc.sync_workspace_job_from_runtime(job_id)
        artifacts = await svc.list_workspace_job_artifacts_async(db, job_id)
        if artifacts is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="workspace job not found")
        return WorkspaceJobArtifactsResponse(jobId=job_id, artifacts=artifacts)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("list_workspace_job_artifacts failed job_id=%s", job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"workspace jobs database unavailable: {exc}",
        ) from exc


@router.delete("/jobs/{job_id}", response_model=WorkspaceJobDeleteResponse)
async def delete_workspace_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceJobDeleteResponse:
    try:
        result = await svc.delete_workspace_job_async(db, job_id)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="workspace job not found")
        await db.commit()
        from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

        invalidate_workspace_dataset_list_cache()
        log_audit_safe(
            user=user,
            action_type="workspace_job_delete",
            action_label="删除 workspace job",
            resource_type="workspace_job",
            resource_id=job_id,
            detail_json={
                "action": "workspace_job_delete",
                "jobId": job_id,
                "jobType": result.get("jobType"),
                "taskType": result.get("taskType"),
                "runtimePath": result.get("runtimePath"),
                "deletedArtifacts": result.get("deletedArtifacts"),
                "runtimeDeleted": result.get("runtimeDeleted"),
                "operator": user.username,
            },
        )
        return WorkspaceJobDeleteResponse(**result)
    except WorkspaceJobDeleteError as exc:
        await db.rollback()
        logger.warning("delete_workspace_job rejected job_id=%s reason=%s", job_id, exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception("delete_workspace_job failed job_id=%s", job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"workspace jobs database unavailable: {exc}",
        ) from exc


@router.post("/jobs/reindex", response_model=WorkspaceReindexResponse)
async def reindex_workspace_jobs(
    payload: WorkspaceReindexRequest,
    _: User = Depends(get_current_user),
) -> WorkspaceReindexResponse:
    try:
        from app.services.training_job_sync_service import reindex_runtime_jobs

        result = reindex_runtime_jobs(
            task_type=payload.taskType,
            job_type=payload.jobType,
            dry_run=payload.dryRun,
            overwrite=payload.overwrite,
            restore_deleted=payload.restoreDeleted,
        )
        return WorkspaceReindexResponse(**result)
    except Exception as exc:
        logger.exception("reindex_workspace_jobs failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
