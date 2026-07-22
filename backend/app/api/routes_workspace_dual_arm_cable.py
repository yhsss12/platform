from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.responses import FileResponse, Response

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.dual_arm_cable import (
    DualArmCableGenerateAsyncResponse,
    DualArmCableGenerateRequest,
    DualArmCableJobStatusResponse,
    DualArmIlExportBuildResponse,
    DualArmIlExportProbeResponse,
)
from app.services import dual_arm_cable_service as svc
from app.services import dual_arm_cable_dataset_service as dataset_svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate-async", response_model=DualArmCableGenerateAsyncResponse)
async def generate_dual_arm_cable_async(
    payload: DualArmCableGenerateRequest,
    _: User = Depends(get_current_user),
) -> DualArmCableGenerateAsyncResponse:
    max_cables = payload.numEpisodes if payload.numEpisodes is not None else payload.maxCables
    result = svc.start_generate_async(
        max_cables=max_cables,
        seed=payload.seed,
        record=payload.record,
        headless=payload.headless,
        stretch_mode=payload.stretchMode,
        release_mode=payload.releaseMode,
        task_config_id=payload.taskConfigId,
    )
    logger.info("dual_arm_cable generate-async job=%s started", result.get("jobId"))
    from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

    invalidate_workspace_dataset_list_cache()
    return DualArmCableGenerateAsyncResponse(**result)


@router.get("/jobs/{job_id}/status", response_model=DualArmCableJobStatusResponse)
async def get_dual_arm_cable_job_status(
    job_id: str,
    _: User = Depends(get_current_user),
) -> DualArmCableJobStatusResponse:
    result = svc.get_job_status(job_id)
    return DualArmCableJobStatusResponse(**result)


@router.get("/jobs/{job_id}/frame")
async def get_dual_arm_cable_job_frame(
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
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@router.get("/jobs/{job_id}/video")
async def get_dual_arm_cable_job_video(
    job_id: str,
    _: User = Depends(get_current_user),
):
    video_path = svc.resolve_job_video_path(job_id)
    if video_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video not found",
        )
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename="generate.mp4",
    )


@router.get("/jobs/{job_id}/log")
async def get_dual_arm_cable_job_log(
    job_id: str,
    _: User = Depends(get_current_user),
) -> dict[str, str]:
    tail = svc.read_job_log_tail(job_id)
    return {"jobId": job_id, "tail": tail}


@router.get("/jobs/{job_id}/result")
async def get_dual_arm_cable_job_result(
    job_id: str,
    _: User = Depends(get_current_user),
):
    result_path = svc.resolve_job_result_path(job_id)
    if result_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="episode_result.json not found",
        )
    return FileResponse(
        path=str(result_path),
        media_type="application/json",
        filename="episode_result.json",
    )


@router.get("/jobs/{job_id}/il-export/probe", response_model=DualArmIlExportProbeResponse)
async def probe_dual_arm_il_export(
    job_id: str,
    _: User = Depends(get_current_user),
) -> DualArmIlExportProbeResponse:
    result = dataset_svc.probe_il_export(job_id)
    return DualArmIlExportProbeResponse(**result)


@router.post("/jobs/{job_id}/il-export/build", response_model=DualArmIlExportBuildResponse)
async def build_dual_arm_il_export(
    job_id: str,
    _: User = Depends(get_current_user),
) -> DualArmIlExportBuildResponse:
    result = dataset_svc.build_il_dataset(job_id)
    return DualArmIlExportBuildResponse(**result)
