from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.responses import FileResponse, Response

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.cable_threading import (
    CableThreadingEvaluateAsyncResponse,
    CableThreadingEvaluateRequest,
    CableThreadingEvaluateResponse,
    CableThreadingGenerateAsyncResponse,
    CableThreadingGenerateRequest,
    CableThreadingGenerateResponse,
    CableThreadingJobStatusResponse,
    CableThreadingVideoRequest,
    CableThreadingVideoResponse,
)
from app.services import cable_threading_service as svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=CableThreadingGenerateResponse)
async def generate_cable_threading_dataset(
    payload: CableThreadingGenerateRequest,
    _: User = Depends(get_current_user),
) -> CableThreadingGenerateResponse:
    result = await asyncio.to_thread(
        svc.run_generate,
        episodes=payload.episodes,
        robot=payload.robot,
        cable_model=payload.cableModel,
        difficulty=payload.difficulty,
        horizon=payload.horizon,
        seed=payload.seed,
        save_hdf5=payload.saveHdf5,
        output_format=payload.outputFormat,
        lerobot_task_instruction=payload.lerobotTaskInstruction or "thread the cable through the pole",
        lerobot_robot=payload.lerobotRobot or payload.robot,
        lerobot_fps=payload.lerobotFps or 20,
    )
    logger.info(
        "cable_threading generate job=%s status=%s success_rate=%s",
        result.get("jobId"),
        result.get("status"),
        (result.get("metrics") or {}).get("finalSuccessRate"),
    )
    return CableThreadingGenerateResponse(**result)


@router.post("/generate-async", response_model=CableThreadingGenerateAsyncResponse)
async def generate_cable_threading_dataset_async(
    payload: CableThreadingGenerateRequest,
    _: User = Depends(get_current_user),
) -> CableThreadingGenerateAsyncResponse:
    result = svc.start_generate_async(
        episodes=payload.episodes,
        robot=payload.robot,
        cable_model=payload.cableModel,
        difficulty=payload.difficulty,
        horizon=payload.horizon,
        seed=payload.seed,
        save_hdf5=payload.saveHdf5,
        output_format=payload.outputFormat,
        save_process_video=payload.saveProcessVideo,
        task_config_id=payload.taskConfigId,
        lerobot_task_instruction=payload.lerobotTaskInstruction or "thread the cable through the pole",
        lerobot_robot=payload.lerobotRobot or payload.robot,
        lerobot_fps=payload.lerobotFps or 20,
    )
    logger.info("cable_threading generate-async job=%s started", result.get("jobId"))
    from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

    invalidate_workspace_dataset_list_cache()
    return CableThreadingGenerateAsyncResponse(**result)


@router.get("/jobs/{job_id}/status", response_model=CableThreadingJobStatusResponse)
async def get_cable_threading_job_status(
    job_id: str,
    _: User = Depends(get_current_user),
) -> CableThreadingJobStatusResponse:
    result = svc.get_job_status(job_id)
    return CableThreadingJobStatusResponse(**result)


@router.get("/jobs/{job_id}/frame")
async def get_cable_threading_job_frame(
    job_id: str,
    _: User = Depends(get_current_user),
):
    frame_path = svc.resolve_job_frame_path(job_id)
    if frame_path is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return FileResponse(
        path=str(frame_path),
        media_type="image/jpeg",
        filename="latest.jpg",
    )


@router.get("/jobs/{job_id}/log")
async def get_cable_threading_job_log(
    job_id: str,
    _: User = Depends(get_current_user),
) -> dict[str, str]:
    tail = svc.read_job_log_tail(job_id)
    return {"jobId": job_id, "tail": tail}


@router.get("/jobs/{job_id}/result")
async def get_cable_threading_eval_result(
    job_id: str,
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    from app.services.imported_eval_bridge import get_imported_eval_result
    from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id

    if is_imported_workspace_eval_job_id(job_id):
        return get_imported_eval_result(job_id)
    if not job_id.startswith("ct_eval_"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="result endpoint only supports ct_eval_* jobs",
        )
    return svc.get_eval_job_result(job_id)


@router.get("/jobs/{job_id}/timeline")
async def get_cable_threading_job_timeline(
    job_id: str,
    _: User = Depends(get_current_user),
):
    timeline_path = svc.resolve_job_timeline_path(job_id)
    if timeline_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="timeline not found",
        )
    return FileResponse(
        path=str(timeline_path),
        media_type="application/json",
        filename="generate_timeline.json",
    )


@router.post("/evaluate-async", response_model=CableThreadingEvaluateAsyncResponse)
async def evaluate_cable_threading_policy_async(
    payload: CableThreadingEvaluateRequest,
    _: User = Depends(get_current_user),
) -> CableThreadingEvaluateAsyncResponse:
    result = svc.start_evaluate_async(
        episodes=payload.episodes,
        robot=payload.robot,
        cable_model=payload.cableModel,
        difficulty=payload.difficulty,
        horizon=payload.horizon,
        seed=payload.seed,
        policy=payload.policy,
        checkpoint=payload.checkpoint,
        device=payload.device,
        task_config_id=payload.taskConfigId,
    )
    logger.info("cable_threading evaluate-async job=%s started", result.get("evalJobId"))
    return CableThreadingEvaluateAsyncResponse(**result)


@router.post("/evaluate", response_model=CableThreadingEvaluateResponse)
async def evaluate_cable_threading_policy(
    payload: CableThreadingEvaluateRequest,
    _: User = Depends(get_current_user),
) -> CableThreadingEvaluateResponse:
    result = await asyncio.to_thread(
        svc.run_evaluate,
        episodes=payload.episodes,
        robot=payload.robot,
        cable_model=payload.cableModel,
        difficulty=payload.difficulty,
        horizon=payload.horizon,
        seed=payload.seed,
        policy=payload.policy,
        checkpoint=payload.checkpoint,
        device=payload.device,
    )
    logger.info(
        "cable_threading evaluate job=%s status=%s success_rate=%s",
        result.get("jobId"),
        result.get("status"),
        (result.get("metrics") or {}).get("successRate"),
    )
    return CableThreadingEvaluateResponse(**result)


@router.post("/video", response_model=CableThreadingVideoResponse)
async def render_cable_threading_video(
    payload: CableThreadingVideoRequest,
    _: User = Depends(get_current_user),
) -> CableThreadingVideoResponse:
    result = await asyncio.to_thread(
        svc.run_video,
        episodes=payload.episodes,
        robot=payload.robot,
        cable_model=payload.cableModel,
        difficulty=payload.difficulty,
        horizon=payload.horizon,
        seed=payload.seed,
    )
    logger.info(
        "cable_threading video job=%s status=%s exists=%s",
        result.get("jobId"),
        result.get("status"),
        result.get("videoExists"),
    )
    return CableThreadingVideoResponse(**result)


@router.get("/jobs/{job_id}/video")
async def get_cable_threading_job_video(
    job_id: str,
    episode: int | None = Query(default=None, ge=0, le=999),
    _: User = Depends(get_current_user),
) -> FileResponse:
    video_path = svc.resolve_job_video_path(job_id, episode=episode)
    if video_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video not found",
        )
    filename = video_path.name
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=filename,
    )


@router.get("/jobs/{job_id}/hdf5-trajectory/{demo_name}")
async def get_cable_threading_hdf5_trajectory_meta(
    job_id: str,
    demo_name: str,
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    return svc.get_hdf5_trajectory_meta(job_id, demo_name)


@router.get("/jobs/{job_id}/hdf5-trajectory/{demo_name}/frame")
async def get_cable_threading_hdf5_trajectory_frame(
    job_id: str,
    demo_name: str,
    camera: str = Query(..., description="RGB observation key, e.g. agentview_image"),
    index: int = Query(0, ge=0, le=100000, description="Frame/step index"),
    quality: int = Query(85, ge=50, le=100),
    _: User = Depends(get_current_user),
):
    from starlette.responses import Response

    frame_bytes = svc.get_hdf5_trajectory_frame(
        job_id,
        demo_name,
        camera=camera,
        frame_index=index,
        quality=quality,
    )
    return Response(content=frame_bytes, media_type="image/jpeg")


@router.get("/jobs/{job_id}/hdf5-trajectory/{demo_name}/step")
async def get_cable_threading_hdf5_trajectory_step(
    job_id: str,
    demo_name: str,
    index: int = Query(0, ge=0, le=100000),
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    return svc.get_hdf5_trajectory_step(job_id, demo_name, step_index=index)
