"""Isaac Sim Franka Pick Place 数据生成服务。"""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.platform_paths import is_path_within, platform_paths
from app.services.isaacsim_franka_pick_place_assets import (
    TASK_ID,
    resolve_job_episode_video_path,
)
from app.services.isaacsim_franka_pick_place_data_worker import execute_job
from app.services.task_config_metadata import build_job_resource_metadata
from app.services.workspace_job_service import (
    record_workspace_job_start,
    sync_workspace_job_from_runtime,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
OUTPUT_ROOT = platform_paths.runs_root / "data_generation"

JOB_ID_PATTERN = re.compile(r"^data_gen_\d{8}_\d{6}_[a-f0-9]{4}$")
JOB_STATUS = Path("status.json")
JOB_LOG = Path("logs") / "run.log"
JOB_CONFIG = Path("metadata") / "job_config.json"
JOB_MANIFEST = Path("dataset_manifest.json")
JOB_VIDEO = Path("videos") / "ep_000001.mp4"

LOG_TAIL_LINES = 40


def make_job_id(prefix: str = "data_gen") -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{suffix}"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _spawn_worker(job_id: str, job_dir: Path, config: dict[str, Any]) -> None:
    def _run() -> None:
        try:
            execute_job(job_dir, job_id, config)
        except Exception as exc:
            logger.exception("isaacsim_franka_pick_place worker failed job=%s", job_id)
            status_path = job_dir / JOB_STATUS
            payload = _read_json(status_path)
            payload.update(
                {
                    "jobId": job_id,
                    "taskId": TASK_ID,
                    "status": "failed",
                    "message": str(exc),
                }
            )
            _write_json(status_path, payload)
            log_path = job_dir / JOB_LOG
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[service] worker_error={exc}\n")
        finally:
            sync_workspace_job_from_runtime(job_id)

    thread = threading.Thread(
        target=_run,
        daemon=True,
        name=f"isfp-gen-{job_id}",
    )
    thread.start()


def start_generate_async(
    *,
    episodes: int,
    seed: int,
    save_video: bool,
    save_trajectory: bool,
    headless: bool,
    task_config_id: Optional[str] = None,
) -> dict[str, Any]:
    if episodes < 1 or episodes > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="episodes must be between 1 and 5",
        )

    job_id = make_job_id("data_gen")
    job_root = _job_dir(job_id)
    for sub in ("metadata", "logs", "results", "episodes", "videos"):
        (job_root / sub).mkdir(parents=True, exist_ok=True)

    config = {
        "episodes": episodes,
        "seed": seed,
        "save_video": save_video,
        "save_trajectory": save_trajectory,
        "headless": headless,
        "taskConfigId": task_config_id,
    }
    _write_json(job_root / JOB_CONFIG, config)

    initial_status = {
        "jobId": job_id,
        "taskId": TASK_ID,
        "status": "running",
        "progress": 0,
        "totalEpisodes": episodes,
        "completedEpisodes": 0,
        "successEpisodes": 0,
        "failedEpisodes": 0,
        "outputDir": str(job_root),
        "message": "数据生成任务已启动",
    }
    _write_json(job_root / JOB_STATUS, initial_status)

    log_path = job_root / JOB_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"[service] started_at={datetime.now().isoformat()}\n")
        handle.write(f"[service] config={json.dumps(config, ensure_ascii=False)}\n\n")

    record_workspace_job_start(
        job_id=job_id,
        job_type="generate",
        task_type=TASK_ID,
        runtime_path=str(job_root),
        runner="isaacsim_franka_pick_place_data_worker",
        status="running",
        metadata=build_job_resource_metadata(
            task_type=TASK_ID,
            task_config_id=task_config_id,
            extra=config,
        ),
    )

    _spawn_worker(job_id, job_root, config)

    api_prefix = f"/api/workspace/isaacsim-franka-pick-place/jobs/{job_id}"
    return {
        "jobId": job_id,
        "taskId": TASK_ID,
        "status": "running",
        "message": "数据生成任务已启动",
        "statusUrl": f"{api_prefix}/status",
        "videoUrl": f"{api_prefix}/video",
    }


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

    api_prefix = f"/api/workspace/isaacsim-franka-pick-place/jobs/{validated}"
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
        "runtimeMode": payload.get("runtimeMode"),
        "message": str(payload.get("message") or ""),
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
        },
        "statusUrl": f"{api_prefix}/status",
        "videoUrl": f"{api_prefix}/video?episode={episode_id}" if video_exists else None,
        "logUrl": f"{api_prefix}/log",
        "manifestPath": str(job_root / JOB_MANIFEST) if (job_root / JOB_MANIFEST).is_file() else None,
    }


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
    if not log_path.is_file():
        return ""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError:
        return ""
