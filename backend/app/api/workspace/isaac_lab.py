from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.responses import FileResponse

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.workspace_benchmark import DatasetResponse
from app.schemas.isaac_lab import (
    IsaacLabGenerateDatasetRequest,
    IsaacLabGenerateDatasetResponse,
    IsaacLabImportDemoRequest,
    IsaacLabImportDemoResponse,
    IsaacLabReplayDemoRequest,
    IsaacLabReplayDemoResponse,
    IsaacLabDatasetReplayContextResponse,
    IsaacLabDatasetPlaybackInfo,
    IsaacLabReplayFromDatasetResponse,
    IsaacLabRunJobLogResponse,
    IsaacLabRunJobStatusResponse,
    IsaacLabRuntimeStatusResponse,
    IsaacLabSmokeTestRequest,
    IsaacLabSmokeTestResponse,
)
from app.services.isaac_lab import generate_service as generate_svc
from app.services.isaac_lab import isaac_dataset_service as dataset_svc
from app.services.isaac_lab import isaac_job_service as job_svc
from app.services.isaac_lab import isaac_replay_context_service as replay_ctx_svc
from app.services.isaac_lab import replay_service as replay_svc
from app.services.isaac_lab import smoke_test_service as smoke_svc
from app.services.isaac_lab.job_paths import isaac_job_root
from app.services.isaac_lab.isaac_runtime_service import get_runtime_status

router = APIRouter()


@router.get("/isaac-lab/runtime/status", response_model=IsaacLabRuntimeStatusResponse)
async def get_isaac_lab_runtime_status(
    _: User = Depends(get_current_user),
) -> IsaacLabRuntimeStatusResponse:
    status_payload = await asyncio.to_thread(get_runtime_status)
    return IsaacLabRuntimeStatusResponse(**status_payload)


@router.post("/isaac-lab/smoke-test", response_model=IsaacLabSmokeTestResponse)
async def start_isaac_lab_smoke_test(
    payload: IsaacLabSmokeTestRequest | None = None,
    _: User = Depends(get_current_user),
) -> IsaacLabSmokeTestResponse:
    keyword = (payload.keyword if payload else "Stack") or "Stack"
    result = await asyncio.to_thread(smoke_svc.start_smoke_test, keyword)
    return IsaacLabSmokeTestResponse(**result)


@router.post("/isaac-lab/replay-demo", response_model=IsaacLabReplayDemoResponse)
async def start_isaac_lab_replay_demo(
    payload: IsaacLabReplayDemoRequest,
    _: User = Depends(get_current_user),
) -> IsaacLabReplayDemoResponse:
    result = await asyncio.to_thread(
        replay_svc.start_replay_demo,
        task_id=payload.taskId,
        dataset_file=payload.datasetFile,
        headless=payload.headless,
        enable_cameras=payload.enableCameras,
        video=payload.video,
    )
    return IsaacLabReplayDemoResponse(**result)


@router.post("/isaac-lab/generate-dataset", response_model=IsaacLabGenerateDatasetResponse)
async def start_isaac_lab_generate_dataset(
    payload: IsaacLabGenerateDatasetRequest,
    _: User = Depends(get_current_user),
) -> IsaacLabGenerateDatasetResponse:
    result = await asyncio.to_thread(
        generate_svc.start_generate_dataset,
        task_id=payload.taskId,
        dataset_name=payload.datasetName,
        num_demos=payload.numDemos,
        seed=payload.seed,
        headless=payload.headless,
        enable_cameras=payload.enableCameras,
        generation_mode=payload.generationMode,
        seed_dataset_file=payload.seedDatasetFile,
        seed_dataset_id=payload.seedDatasetId,
        video=payload.video,
        num_envs=payload.numEnvs,
    )
    return IsaacLabGenerateDatasetResponse(**result)


@router.post(
    "/isaac-lab/datasets/import-demo",
    response_model=IsaacLabImportDemoResponse,
)
async def import_isaac_lab_demo_dataset(
    payload: IsaacLabImportDemoRequest,
    _: User = Depends(get_current_user),
) -> IsaacLabImportDemoResponse:
    dataset = await asyncio.to_thread(
        dataset_svc.import_demo_hdf5,
        dataset_file=payload.datasetFile,
        display_name=payload.displayName,
        task_id=payload.taskId,
    )
    return IsaacLabImportDemoResponse(dataset=DatasetResponse(**dataset))


@router.get(
    "/isaac-lab/datasets/{dataset_id}/replay-context",
    response_model=IsaacLabDatasetReplayContextResponse,
)
async def get_isaac_lab_dataset_replay_context(
    dataset_id: str,
    _: User = Depends(get_current_user),
) -> IsaacLabDatasetReplayContextResponse:
    payload = await asyncio.to_thread(replay_ctx_svc.resolve_dataset_playback, dataset_id)
    playback = payload.get("playback")
    playback_model = IsaacLabDatasetPlaybackInfo(**playback) if isinstance(playback, dict) else None
    return IsaacLabDatasetReplayContextResponse(
        dataset=DatasetResponse(**payload["dataset"]),
        sourceJobId=payload.get("sourceJobId"),
        sourceJobStatus=payload.get("sourceJobStatus"),
        replayJobs=payload.get("replayJobs") or [],
        replayJobId=payload.get("replayJobId"),
        replayJobStatus=payload.get("replayJobStatus"),
        replayInProgress=bool(payload.get("replayInProgress")),
        replayFailed=bool(payload.get("replayFailed")),
        playback=playback_model,
        usingPreviewFallback=bool(payload.get("usingPreviewFallback")),
        hasDatasetFile=bool(payload.get("hasDatasetFile")),
        videoSourceLabel=replay_ctx_svc.video_source_label(
            str(playback_model.videoSource) if playback_model else None,
            transcoded=bool(playback_model.transcoded) if playback_model else False,
        ),
    )


@router.post(
    "/isaac-lab/datasets/{dataset_id}/replay",
    response_model=IsaacLabReplayFromDatasetResponse,
)
async def start_isaac_lab_replay_from_dataset(
    dataset_id: str,
    _: User = Depends(get_current_user),
) -> IsaacLabReplayFromDatasetResponse:
    dataset = await asyncio.to_thread(dataset_svc.get_isaac_dataset, dataset_id)
    dataset_file = Path(str(dataset["datasetFile"]))

    existing = await asyncio.to_thread(
        replay_ctx_svc.find_reusable_replay_job,
        dataset_id=dataset_id,
        dataset_file=dataset_file,
    )
    if existing:
        status_value = str(existing.get("status") or "unknown")
        if status_value in {"queued", "running"} or (
            status_value == "completed" and existing.get("videoAvailable")
        ):
            return IsaacLabReplayFromDatasetResponse(
                datasetId=dataset_id,
                jobId=str(existing["jobId"]),
                kind="replay_demo",
                status=status_value,
                runtimePath=str(isaac_job_root(str(existing["jobId"]))),
                statusUrl=f"/api/workspace/isaac-lab/jobs/{existing['jobId']}/status",
                reused=True,
            )

    result = await asyncio.to_thread(
        replay_svc.start_replay_demo,
        task_id=str(dataset.get("taskId") or "Isaac-Stack-Cube-Franka-IK-Rel-v0"),
        dataset_file=str(dataset["datasetFile"]),
        dataset_id=dataset_id,
    )
    return IsaacLabReplayFromDatasetResponse(
        datasetId=dataset_id,
        jobId=result["jobId"],
        kind=result.get("kind", "replay_demo"),
        status=result["status"],
        runtimePath=result.get("runtimePath"),
        statusUrl=result.get("statusUrl"),
        reused=False,
    )


@router.delete("/isaac-lab/datasets/{dataset_id}")
async def delete_isaac_lab_dataset(
    dataset_id: str,
    _: User = Depends(get_current_user),
) -> dict[str, bool]:
    await asyncio.to_thread(dataset_svc.delete_isaac_dataset, dataset_id)
    return {"ok": True}


@router.get("/isaac-lab/jobs/{job_id}/status", response_model=IsaacLabRunJobStatusResponse)
async def get_isaac_lab_run_job_status(
    job_id: str,
    _: User = Depends(get_current_user),
) -> IsaacLabRunJobStatusResponse:
    result = await asyncio.to_thread(job_svc.get_job_status, job_id)
    return IsaacLabRunJobStatusResponse(**result)


@router.get("/isaac-lab/jobs/{job_id}/log", response_model=IsaacLabRunJobLogResponse)
async def get_isaac_lab_run_job_log(
    job_id: str,
    stream: str = Query(default="stdout", pattern="^(stdout|stderr)$"),
    lines: int = Query(default=80, ge=1, le=500),
    _: User = Depends(get_current_user),
) -> IsaacLabRunJobLogResponse:
    tail = await asyncio.to_thread(job_svc.read_job_log_tail, job_id, stream=stream, lines=lines)
    return IsaacLabRunJobLogResponse(jobId=job_id, stream=stream, tail=tail)


@router.get("/isaac-lab/jobs/{job_id}/video")
async def get_isaac_lab_run_job_video(
    job_id: str,
    _: User = Depends(get_current_user),
):
    video_path, note, meta = await asyncio.to_thread(job_svc.resolve_job_browser_video_path, job_id)
    if video_path is None or not video_path.is_file():
        detail = note or "Replay video not available for this job"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )
    headers: dict[str, str] = {}
    if note in {"transcoded", "transcoded_cache"}:
        headers["X-Isaac-Video-Transcoded"] = "1"
    video_source = str(meta.get("videoSource") or "")
    if video_source:
        headers["X-Isaac-Video-Source"] = video_source
    codec = meta.get("codec")
    if codec:
        headers["X-Isaac-Video-Codec"] = str(codec)
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
        headers=headers,
    )


@router.get("/isaac-lab/jobs/{job_id}/live/latest")
async def get_isaac_lab_run_job_live_latest(
    job_id: str,
    _: User = Depends(get_current_user),
):
    frame_path: Path | None = await asyncio.to_thread(job_svc.resolve_job_live_frame_path, job_id)
    if frame_path is None or not frame_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Live frame not available for this job",
        )
    return FileResponse(
        path=str(frame_path),
        media_type="image/jpeg",
        filename="latest.jpg",
    )
