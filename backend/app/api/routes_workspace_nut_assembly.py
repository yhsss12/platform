from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response
from starlette.responses import FileResponse

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.nut_assembly import (
    NutAssemblyGenerateAsyncResponse,
    NutAssemblyGenerateRequest,
    NutAssemblyJobStatusResponse,
)
from app.services import nut_assembly_service as svc
from app.services import nut_assembly_dataset_service as dataset_svc

router = APIRouter()
logger = logging.getLogger(__name__)

_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


def _apply_no_store_headers(response: Response) -> None:
    for key, value in _NO_STORE_HEADERS.items():
        response.headers[key] = value


def _resolve_nut_assembly_source_demo(payload: NutAssemblyGenerateRequest) -> tuple[str | None, str | None]:
    """Map workspace sourceDemoDatasetId to backend source demo selection/path."""
    dataset_id = (payload.sourceDemoDatasetId or "").strip()
    if dataset_id == svc.NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID:
        return None, "official"
    source_path = payload.sourceDemoPath
    source_selection = payload.sourceDemoSelection
    if dataset_id and source_path:
        return source_path, source_selection or "custom"
    return source_path, source_selection


@router.post("/generate-async", response_model=NutAssemblyGenerateAsyncResponse)
async def generate_nut_assembly_dataset_async(
    payload: NutAssemblyGenerateRequest,
    _: User = Depends(get_current_user),
) -> NutAssemblyGenerateAsyncResponse:
    source_demo_path, source_demo_selection = _resolve_nut_assembly_source_demo(payload)
    result = svc.start_generate_async(
        task_template_id=payload.taskTemplateId,
        episodes=payload.episodes,
        seed=payload.seed,
        render_video=payload.renderVideo,
        source_demo_path=source_demo_path,
        source_demo_selection=source_demo_selection,
        env_name=payload.envName,
        output_name=payload.outputName,
        horizon=payload.horizon,
        task_config_id=payload.taskConfigId,
        generation_mode=payload.generationMode,
        generation_path=payload.generationPath,
        generation_metadata={
            key: value
            for key, value in {
                "generationPath": payload.generationPath,
                "sourceDemoDatasetId": payload.sourceDemoDatasetId,
                "augmentationAlgorithm": payload.augmentationAlgorithm,
                "seedGenerationCount": payload.seedGenerationCount,
                "seedKeepCount": payload.seedKeepCount,
                "targetCount": payload.targetCount,
                "autoSelectBestSeeds": payload.autoSelectBestSeeds,
                "replayValidation": payload.replayValidation,
                "expertPolicy": payload.expertPolicy,
                "successFilter": payload.successFilter,
                "keepFailedTrajectories": payload.keepFailedTrajectories,
                "enablePinnRepair": payload.enablePinnRepair,
            }.items()
            if value is not None
        },
        physics_enhancement=(
            payload.physicsEnhancement.model_dump(exclude_none=True)
            if payload.physicsEnhancement is not None
            else None
        ),
    )
    logger.info("nut_assembly generate-async job=%s started", result.get("jobId"))
    from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

    invalidate_workspace_dataset_list_cache()
    return NutAssemblyGenerateAsyncResponse(**result)


@router.get("/mimicgen-env-status")
async def get_nut_assembly_mimicgen_env_status(
    refresh: bool = False,
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    return svc.get_mimicgen_env_status(refresh=refresh)


@router.get("/source-demo-status")
async def get_nut_assembly_source_demo_status(
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    return svc.get_source_demo_status()


@router.get("/pinn-model-status")
async def get_nut_assembly_pinn_model_status(
    modelId: str = "nut_assembly_pinn_v1",
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    return svc.get_pinn_model_status(modelId)


@router.get("/jobs/{job_id}", response_model=NutAssemblyJobStatusResponse)
async def get_nut_assembly_job(
    job_id: str,
    response: Response,
    tail: int = 20,
    _: User = Depends(get_current_user),
) -> NutAssemblyJobStatusResponse:
    _apply_no_store_headers(response)
    max_lines = max(1, min(int(tail), 200))
    result = svc.get_generate_job_detail(job_id, log_tail_lines=max_lines)
    return NutAssemblyJobStatusResponse(**result)


@router.get("/jobs/{job_id}/status", response_model=NutAssemblyJobStatusResponse)
async def get_nut_assembly_job_status(
    job_id: str,
    response: Response,
    _: User = Depends(get_current_user),
) -> NutAssemblyJobStatusResponse:
    _apply_no_store_headers(response)
    result = svc.get_generate_job_status(job_id)
    return NutAssemblyJobStatusResponse(**result)


@router.get("/jobs/{job_id}/result")
async def get_nut_assembly_job_result(
    job_id: str,
    response: Response,
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    _apply_no_store_headers(response)
    return svc.get_generate_job_result(job_id)


@router.get("/jobs/{job_id}/logs")
async def get_nut_assembly_job_logs(
    job_id: str,
    response: Response,
    tail: int = 20,
    _: User = Depends(get_current_user),
) -> dict[str, str]:
    _apply_no_store_headers(response)
    max_lines = max(1, min(int(tail), 200))
    log_tail = svc.read_job_log_tail(job_id, max_lines=max_lines)
    return {"jobId": job_id, "tail": log_tail}


@router.get("/jobs/{job_id}/log")
async def get_nut_assembly_job_log(
    job_id: str,
    response: Response,
    tail: int = 20,
    _: User = Depends(get_current_user),
) -> dict[str, str]:
    _apply_no_store_headers(response)
    max_lines = max(1, min(int(tail), 200))
    log_tail = svc.read_job_log_tail(job_id, max_lines=max_lines)
    return {"jobId": job_id, "tail": log_tail}


@router.get("/jobs/{job_id}/video")
async def get_nut_assembly_job_video(
    job_id: str,
    _: User = Depends(get_current_user),
):
    video_path = svc.resolve_job_video_path(job_id)
    if video_path is None:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return FileResponse(path=str(video_path), media_type="video/mp4", filename="generate.mp4")


@router.get("/jobs/{job_id}/training-build/probe")
async def probe_nut_assembly_training_build(
    job_id: str,
    filterMode: str = "valid_for_training_only",
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    mode = filterMode if filterMode in {"all", "all_generated_demos", "success_only", "valid_for_training_only"} else "valid_for_training_only"
    return dataset_svc.probe_training_build(job_id, filter_mode=mode)  # type: ignore[arg-type]


@router.post("/jobs/{job_id}/training-build")
async def build_nut_assembly_training_dataset(
    job_id: str,
    filterMode: str = "valid_for_training_only",
    _: User = Depends(get_current_user),
) -> dict[str, object]:
    mode = filterMode if filterMode in {"all", "all_generated_demos", "success_only", "valid_for_training_only"} else "valid_for_training_only"
    return dataset_svc.build_training_dataset(job_id, filter_mode=mode)  # type: ignore[arg-type]
