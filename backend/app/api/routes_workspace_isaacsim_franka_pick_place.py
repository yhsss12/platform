from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.responses import FileResponse

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.isaacsim_franka_pick_place import (
    IsaacSimFrankaPickPlaceGenerateAsyncResponse,
    IsaacSimFrankaPickPlaceGenerateRequest,
    IsaacSimFrankaPickPlaceJobStatusResponse,
)
from app.services import isaacsim_franka_pick_place_service as svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate-async", response_model=IsaacSimFrankaPickPlaceGenerateAsyncResponse)
async def generate_isaacsim_franka_pick_place_async(
    payload: IsaacSimFrankaPickPlaceGenerateRequest,
    _: User = Depends(get_current_user),
) -> IsaacSimFrankaPickPlaceGenerateAsyncResponse:
    if payload.taskId and payload.taskId != "isaacsim_franka_pick_place":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="taskId must be isaacsim_franka_pick_place",
        )
    result = svc.start_generate_async(
        episodes=payload.episodes,
        seed=payload.seed,
        save_video=payload.saveVideo,
        save_trajectory=payload.saveTrajectory,
        headless=payload.headless,
        task_config_id=payload.taskConfigId,
    )
    logger.info("isaacsim_franka_pick_place generate-async job=%s started", result.get("jobId"))
    from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

    invalidate_workspace_dataset_list_cache()
    return IsaacSimFrankaPickPlaceGenerateAsyncResponse(**result)


@router.get("/jobs/{job_id}/status", response_model=IsaacSimFrankaPickPlaceJobStatusResponse)
async def get_isaacsim_franka_pick_place_job_status(
    job_id: str,
    _: User = Depends(get_current_user),
) -> IsaacSimFrankaPickPlaceJobStatusResponse:
    result = svc.get_job_status(job_id)
    return IsaacSimFrankaPickPlaceJobStatusResponse(**result)


@router.get("/jobs/{job_id}/video")
async def get_isaacsim_franka_pick_place_job_video(
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


@router.get("/jobs/{job_id}/log")
async def get_isaacsim_franka_pick_place_job_log(
    job_id: str,
    _: User = Depends(get_current_user),
) -> dict[str, str]:
    tail = svc.read_job_log_tail(job_id)
    return {"jobId": job_id, "tail": tail}
