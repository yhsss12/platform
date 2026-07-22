"""Isaac Lab Franka stack cube data generation service."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.core.platform_paths import is_path_within, platform_paths
from app.services.isaaclab_franka_stack_cube_assets import TASK_ID, resolve_job_episode_video_path
from app.services.workspace_job_service import sync_workspace_job_from_runtime

PROJECT_ROOT = platform_paths.project_root
OUTPUT_ROOT = platform_paths.runs_root / "data_generation"

JOB_ID_PATTERN = re.compile(r"^data_gen_\d{8}_\d{6}_[a-f0-9]{4}$")
JOB_STATUS = Path("status.json")
JOB_LOG = Path("logs") / "run.log"
JOB_CONFIG = Path("metadata") / "job_config.json"
JOB_MANIFEST = Path("dataset_manifest.json")

LOG_TAIL_LINES = 40


def _job_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / job_id


def validate_job_id(job_id: str) -> str:
    candidate = (job_id or "").strip()
    if not JOB_ID_PATTERN.match(candidate):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    return candidate


def _assert_job_root(job_root: Path) -> Path:
    jobs_root = (OUTPUT_ROOT / "jobs").resolve()
    resolved = job_root.resolve()
    if not is_path_within(resolved, jobs_root):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid job path",
        )
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_job_status(job_id: str, episode_id: str = "ep_000001") -> dict[str, Any]:
    validated = validate_job_id(job_id)
    sync_workspace_job_from_runtime(validated)
    job_root = _assert_job_root(_job_dir(validated))
    payload = _read_json(job_root / JOB_STATUS)

    video_path, video_meta = resolve_job_episode_video_path(job_root, episode_id)
    episode_manifest = _read_json(job_root / "episodes" / episode_id / "episode_manifest.json")
    dataset_manifest = _read_json(job_root / JOB_MANIFEST)

    aggregate = _read_json(job_root / "results" / "aggregate_metrics.json")
    per_episode = _read_json(job_root / "results" / "per_episode_results.json")
    first_ep = None
    episodes = per_episode.get("episodes")
    if isinstance(episodes, list) and episodes:
        first_ep = episodes[0] if isinstance(episodes[0], dict) else None
    ep_metrics = (first_ep or {}).get("metrics") if isinstance(first_ep, dict) else {}
    if not ep_metrics:
        metrics_path = job_root / "episodes" / episode_id / "metrics.json"
        ep_metrics = _read_json(metrics_path)

    job_config = _read_json(job_root / JOB_CONFIG)

    live_dir = job_root / "live"
    live_status = _read_json(live_dir / "status.json")
    latest_frame = live_dir / "latest.jpg"
    live_frame_exists = latest_frame.is_file()
    live_frame_available = bool(
        live_status.get("liveFrameAvailable")
        if "liveFrameAvailable" in live_status
        else live_frame_exists
    )
    live_frame_black = bool(live_status.get("liveFrameBlack")) if live_status else False

    api_prefix = f"/api/workspace/isaaclab-franka-stack-cube/jobs/{validated}"
    video_exists = video_path is not None and video_path.is_file()
    episode_video_status = episode_manifest.get("video_status") or episode_manifest.get("videoStatus")
    video_status = str(
        video_meta.get("video_status")
        or video_meta.get("videoStatus")
        or episode_video_status
        or ("available" if video_exists else "pending")
    )

    return {
        "jobId": validated,
        "taskId": TASK_ID,
        "status": str(payload.get("status") or "running"),
        "progress": payload.get("progress"),
        "totalEpisodes": payload.get("totalEpisodes"),
        "completedEpisodes": payload.get("completedEpisodes"),
        "successEpisodes": payload.get("successEpisodes"),
        "failedEpisodes": payload.get("failedEpisodes"),
        "outputDir": str(job_root),
        "datasetId": payload.get("datasetId"),
        "runtimeMode": payload.get("runtimeMode") or "isaac_lab",
        "message": str(payload.get("message") or ""),
        "generationMode": payload.get("generationMode") or job_config.get("generationMode"),
        "phase": payload.get("phase"),
        "phaseLabel": payload.get("phaseLabel"),
        "phaseStartedAt": payload.get("phaseStartedAt"),
        "phaseUpdatedAt": payload.get("phaseUpdatedAt"),
        "phaseTimings": payload.get("phaseTimings"),
        "progressMessage": payload.get("progressMessage") or payload.get("message"),
        "errorSummary": payload.get("errorSummary"),
        "requestedDevice": payload.get("requestedDevice") or job_config.get("requestedDevice"),
        "resolvedDevice": payload.get("resolvedDevice") or job_config.get("resolvedDevice"),
        "cudaVisibleDevices": payload.get("cudaVisibleDevices") or job_config.get("cudaVisibleDevices"),
        "isGpuRequested": payload.get("isGpuRequested", job_config.get("isGpuRequested")),
        "torchCudaAvailable": payload.get("torchCudaAvailable", job_config.get("torchCudaAvailable")),
        "videoExists": video_exists,
        "video_status": video_status,
        "videoStatus": video_status,
        "taskIdValidated": bool(video_meta.get("taskIdValidated")),
        "validationError": video_meta.get("validationError"),
        "videoPath": str(video_path) if video_exists else None,
        "episodeId": episode_id,
        "episodeManifest": episode_manifest,
        "datasetManifest": dataset_manifest,
        "metrics": {
            **aggregate,
            **(ep_metrics if isinstance(ep_metrics, dict) else {}),
            **{
                k: v
                for k, v in {
                    "generation_mode": payload.get("generationMode") or job_config.get("generationMode"),
                    "phase": payload.get("phase"),
                }.items()
                if v is not None
            },
        },
        "statusUrl": f"{api_prefix}/status",
        "videoUrl": f"{api_prefix}/video?episode={episode_id}" if video_exists else None,
        "logUrl": f"{api_prefix}/log",
        "manifestPath": str(job_root / JOB_MANIFEST) if (job_root / JOB_MANIFEST).is_file() else None,
        "liveFrameAvailable": live_frame_available,
        "liveFrameBlack": live_frame_black,
        "liveFrameExists": live_frame_exists,
        "enableCameras": True,
        "liveFrameUrl": f"{api_prefix}/live-frame" if live_frame_exists else None,
    }


def resolve_job_live_frame_path(job_id: str) -> Optional[Path]:
    validated = validate_job_id(job_id)
    job_root = _assert_job_root(_job_dir(validated))
    frame_path = job_root / "live" / "latest.jpg"
    if frame_path.is_file() and frame_path.stat().st_size > 0:
        return frame_path
    return None


def resolve_job_video_path(job_id: str, episode_id: str = "ep_000001") -> Optional[Path]:
    validated = validate_job_id(job_id)
    job_root = _assert_job_root(_job_dir(validated))
    video_path, video_meta = resolve_job_episode_video_path(job_root, episode_id)
    if not video_meta.get("taskIdValidated"):
        return None
    video_status = video_meta.get("video_status") or video_meta.get("videoStatus")
    if video_status != "available":
        return None
    return video_path


def resolve_job_log_path(job_id: str) -> Path:
    validated = validate_job_id(job_id)
    job_root = _assert_job_root(_job_dir(validated))
    return job_root / JOB_LOG


def read_job_log_tail(job_id: str, lines: int = LOG_TAIL_LINES) -> str:
    log_path = resolve_job_log_path(job_id)
    chunks: list[str] = []
    validated = validate_job_id(job_id)
    job_root = _assert_job_root(_job_dir(validated))
    platform_log = job_root / "logs" / "platform_run.log"
    for path in (log_path, platform_log):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
            label = "run.log" if path.name == "run.log" else path.name
            tail = "\n".join(content[-lines:])
            if tail.strip():
                chunks.append(f"--- {label} ---\n{tail}")
        except OSError:
            continue
    return "\n\n".join(chunks)
