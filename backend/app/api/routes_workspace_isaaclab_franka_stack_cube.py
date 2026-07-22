from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.responses import FileResponse

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.isaaclab_franka_stack_cube import (
    IsaacLabFrankaStackCubeJobStatusResponse,
)
from app.services import isaaclab_franka_stack_cube_service as svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}/status", response_model=IsaacLabFrankaStackCubeJobStatusResponse)
async def get_isaaclab_franka_stack_cube_job_status(
    job_id: str,
    _: User = Depends(get_current_user),
) -> IsaacLabFrankaStackCubeJobStatusResponse:
    result = svc.get_job_status(job_id)
    return IsaacLabFrankaStackCubeJobStatusResponse(**result)


@router.get("/jobs/{job_id}/video")
async def get_isaaclab_franka_stack_cube_job_video(
    job_id: str,
    episode: str = "ep_000001",
    _: User = Depends(get_current_user),
):
    video_path = svc.resolve_job_video_path(job_id, episode_id=episode)
    if video_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video not found",
        )
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{episode}.mp4",
    )


@router.get("/jobs/{job_id}/live-frame")
async def get_isaaclab_franka_stack_cube_job_live_frame(
    job_id: str,
    _: User = Depends(get_current_user),
):
    frame_path = svc.resolve_job_live_frame_path(job_id)
    if frame_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="live frame not available",
        )
    return FileResponse(
        path=str(frame_path),
        media_type="image/jpeg",
        filename="latest.jpg",
    )


@router.get("/jobs/{job_id}/log")
async def get_isaaclab_franka_stack_cube_job_log(
    job_id: str,
    _: User = Depends(get_current_user),
) -> dict[str, str]:
    tail = svc.read_job_log_tail(job_id)
    return {"jobId": job_id, "tail": tail}


@router.delete("/datasets/{job_id}")
async def delete_isaaclab_franka_stack_cube_dataset(
    job_id: str,
    _: User = Depends(get_current_user),
) -> dict[str, Any]:
    from app.services import workspace_dataset_service as dataset_svc

    validated = svc.validate_job_id(job_id)
    try:
        result = await asyncio.to_thread(dataset_svc.delete_data_generation_dataset, validated)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    logger.info("isaaclab_franka_stack_cube dataset deleted job=%s", validated)
    return result
