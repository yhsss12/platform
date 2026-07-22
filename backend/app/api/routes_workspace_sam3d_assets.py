from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.responses import FileResponse

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.sam3d_asset import (
    AssetJobCreateResponse,
    AssetJobDeleteResponse,
    AssetJobListResponse,
    AssetJobStatusResponse,
    AssetReconstructRequest,
    AssetRenderMujocoRequest,
    AssetSegmentRequest,
)
from app.services import sam3d_asset_service as svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/jobs", response_model=AssetJobCreateResponse)
async def create_asset_pipeline_job(
    name: str = Form(...),
    image: UploadFile = File(...),
    _: User = Depends(get_current_user),
) -> AssetJobCreateResponse:
    if not image.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="image file required")
    result = await asyncio.to_thread(svc.create_asset_job, name=name, uploaded_file=image)
    return AssetJobCreateResponse(**result)


@router.post("/jobs/{job_id}/segment", response_model=AssetJobStatusResponse)
async def segment_asset_pipeline_job(
    job_id: str,
    payload: AssetSegmentRequest,
    _: User = Depends(get_current_user),
) -> AssetJobStatusResponse:
    result = await asyncio.to_thread(
        svc.start_segment_job,
        job_id,
        payload.model_dump(),
    )
    return AssetJobStatusResponse(**result)


@router.post("/jobs/{job_id}/reconstruct", response_model=AssetJobStatusResponse)
async def reconstruct_asset_pipeline_job(
    job_id: str,
    payload: AssetReconstructRequest,
    _: User = Depends(get_current_user),
) -> AssetJobStatusResponse:
    result = await asyncio.to_thread(
        svc.start_reconstruct_job,
        job_id,
        payload.model_dump(),
    )
    return AssetJobStatusResponse(**result)


@router.post("/jobs/{job_id}/render-mujoco", response_model=AssetJobStatusResponse)
async def render_mujoco_asset_pipeline_job(
    job_id: str,
    payload: AssetRenderMujocoRequest,
    _: User = Depends(get_current_user),
) -> AssetJobStatusResponse:
    result = await asyncio.to_thread(
        svc.render_mujoco_job,
        job_id,
        payload.model_dump(),
    )
    return AssetJobStatusResponse(**result)


@router.get("/jobs/{job_id}", response_model=AssetJobStatusResponse)
async def get_asset_pipeline_job(
    job_id: str,
    _: User = Depends(get_current_user),
) -> AssetJobStatusResponse:
    result = await asyncio.to_thread(svc.get_asset_job_status, job_id)
    return AssetJobStatusResponse(**result)


@router.delete("/jobs/{job_id}", response_model=AssetJobDeleteResponse)
async def delete_asset_pipeline_job(
    job_id: str,
    _: User = Depends(get_current_user),
) -> AssetJobDeleteResponse:
    result = await asyncio.to_thread(svc.delete_asset_job, job_id)
    return AssetJobDeleteResponse(**result)


@router.get("/jobs", response_model=AssetJobListResponse)
async def list_asset_pipeline_jobs(
    limit: int = 50,
    _: User = Depends(get_current_user),
) -> AssetJobListResponse:
    jobs = await asyncio.to_thread(svc.list_asset_jobs, limit=limit)
    items = [AssetJobStatusResponse(**job) for job in jobs]
    return AssetJobListResponse(jobs=items, total=len(items))


@router.get("/jobs/{job_id}/files/{rel_path:path}")
async def get_asset_pipeline_job_file(
    job_id: str,
    rel_path: str,
    _: User = Depends(get_current_user),
) -> FileResponse:
    path = await asyncio.to_thread(svc.get_asset_job_file, job_id, rel_path)
    media_type = svc.guess_media_type(rel_path)
    filename = path.name
    headers = svc.file_download_headers(rel_path, filename)
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=filename,
        headers=headers,
    )
