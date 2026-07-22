"""Isaac Lab demo 回放 job（replay_demos.py）。"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.config import settings
from app.services.isaac_lab.cli_runner import IsaacLabCliRunner
from app.services.isaac_lab.isaac_job_utils import (
    finalize_status,
    make_isaac_job_id,
    read_json,
    utc_now_iso,
    write_json,
)
from app.services.isaac_lab.isaac_runtime_service import (
    RUNTIME_NOT_CONFIGURED_MSG,
    assert_runtime_configured_for_commands,
)
from app.services.isaac_lab.job_paths import (
    isaac_job_artifacts_dir,
    isaac_job_metadata_dir,
    isaac_job_replay_manifest_path,
    isaac_job_replay_video_path,
    isaac_job_root,
    isaac_job_status_path,
    isaac_job_stderr_path,
    isaac_job_stdout_path,
    is_isaac_replay_job_id,
)
from app.services.isaac_lab.paths import resolve_isaaclab_root
from app.services.isaac_lab.replay_cli import (
    HDF5_TO_MP4_SCRIPT,
    REPLAY_DEMO_SCRIPT,
    ReplayDemoCliParams,
    build_hdf5_to_mp4_cli_args,
    build_replay_demos_cli_args,
    replay_demos_supports_video_flag,
)

logger = logging.getLogger(__name__)

DEFAULT_REPLAY_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_JOBS: set[str] = set()


def resolve_dataset_file(dataset_file: str) -> Path:
    raw = (dataset_file or "").strip()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="datasetFile is required (provide an Isaac Lab 物块堆叠 HDF5 demo path)",
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        root = resolve_isaaclab_root()
        candidates = [Path.cwd() / path]
        if root is not None:
            candidates.append(root / path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"dataset_file not found: {raw}. Provide a valid 物块堆叠 HDF5 demo path.",
        )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"dataset_file not found: {path}",
        )
    return path.resolve()


def make_isaac_replay_job_id() -> str:
    return make_isaac_job_id("isaac_replay")


def _job_paths_payload(job_id: str) -> dict[str, str]:
    root = isaac_job_root(job_id)
    artifacts = isaac_job_artifacts_dir(job_id)
    return {
        "jobRoot": str(root),
        "stdoutLog": str(isaac_job_stdout_path(job_id)),
        "stderrLog": str(isaac_job_stderr_path(job_id)),
        "statusJson": str(isaac_job_status_path(job_id)),
        "artifactsDir": str(artifacts),
        "replayManifest": str(isaac_job_replay_manifest_path(job_id)),
        "replayVideo": str(isaac_job_replay_video_path(job_id)),
    }


def _find_generated_mp4(artifacts_dir: Path) -> Optional[Path]:
    preferred = artifacts_dir / "replay.mp4"
    if preferred.is_file():
        return preferred
    if not artifacts_dir.is_dir():
        return None
    mp4_files = sorted(artifacts_dir.glob("*.mp4"))
    return mp4_files[0] if mp4_files else None


def _try_generate_video_from_hdf5(
    runner: IsaacLabCliRunner,
    *,
    dataset_file: Path,
    artifacts_dir: Path,
    timeout: int,
) -> tuple[bool, Optional[Path], Optional[str]]:
    """replay_demos 无 --video；若 HDF5 含相机观测，尝试 hdf5_to_mp4 后处理。"""
    if replay_demos_supports_video_flag():
        return False, None, "replay_demos native --video not implemented in wrapper"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifacts_dir / "hdf5_to_mp4.stdout.log"
    stderr_path = artifacts_dir / "hdf5_to_mp4.stderr.log"
    try:
        result = runner.run_to_files(
            HDF5_TO_MP4_SCRIPT,
            *build_hdf5_to_mp4_cli_args(input_file=dataset_file, output_dir=artifacts_dir),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=min(timeout, 600),
        )
    except Exception as exc:
        return False, None, f"hdf5_to_mp4 failed: {exc}"

    if result.returncode != 0 or result.timed_out:
        return False, None, f"hdf5_to_mp4 exit={result.returncode} timed_out={result.timed_out}"

    mp4 = _find_generated_mp4(artifacts_dir)
    if mp4 is None:
        return False, None, "hdf5_to_mp4 completed but no mp4 was produced (dataset may lack camera keys)"

    target = isaac_job_replay_video_path(artifacts_dir.parent.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mp4.resolve() != target.resolve():
        shutil.copy2(mp4, target)
    return True, target, None


def _execute_replay_job(
    job_id: str,
    params: ReplayDemoCliParams,
    *,
    request_video: bool,
) -> None:
    runner = IsaacLabCliRunner.from_settings()
    stdout_path = isaac_job_stdout_path(job_id)
    stderr_path = isaac_job_stderr_path(job_id)
    artifacts_dir = isaac_job_artifacts_dir(job_id)
    timeout = int(getattr(settings, "ISAACLAB_REPLAY_TIMEOUT", 1800) or 1800)

    cli_args = build_replay_demos_cli_args(params)
    started_at = read_json(isaac_job_status_path(job_id)).get("startedAt")

    finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "replay_demo",
            "status": "running",
            "phase": "replay",
            "taskId": params.task_id,
            "datasetFile": str(params.dataset_file),
            "command": runner.build_command(REPLAY_DEMO_SCRIPT, *cli_args),
            "headless": params.headless,
            "enableCameras": params.enable_cameras,
            "videoRequested": request_video,
            "videoAvailable": False,
            "startedAt": started_at,
            "message": "Running Isaac Lab replay_demos.py…",
            "paths": _job_paths_payload(job_id),
        },
    )

    video_available = False
    video_path: Optional[Path] = None
    video_note: Optional[str] = None

    try:
        result = runner.run_to_files(
            REPLAY_DEMO_SCRIPT,
            *cli_args,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
        )
        success = result.returncode == 0 and not result.timed_out

        manifest = {
            "jobId": job_id,
            "kind": "replay_demo",
            "taskId": params.task_id,
            "datasetFile": str(params.dataset_file),
            "replayScript": REPLAY_DEMO_SCRIPT,
            "command": result.command,
            "exitCode": result.returncode,
            "timedOut": result.timed_out,
            "finishedAt": utc_now_iso(),
        }
        write_json(isaac_job_replay_manifest_path(job_id), manifest)

        if success and request_video:
            video_available, video_path, video_note = _try_generate_video_from_hdf5(
                runner,
                dataset_file=params.dataset_file,
                artifacts_dir=artifacts_dir,
                timeout=timeout,
            )
        elif request_video and not replay_demos_supports_video_flag():
            video_note = (
                "replay_demos.py has no --video flag; attempted hdf5_to_mp4 only after successful replay"
            )
        elif not request_video:
            video_note = "video=false"

        finalize_status(
            job_id,
            {
                "jobId": job_id,
                "kind": "replay_demo",
                "status": "completed" if success else "failed",
                "phase": "done",
                "taskId": params.task_id,
                "datasetFile": str(params.dataset_file),
                "command": result.command,
                "headless": params.headless,
                "enableCameras": params.enable_cameras,
                "videoRequested": request_video,
                "videoAvailable": video_available,
                "videoPath": str(video_path) if video_path else None,
                "videoNote": video_note,
                "startedAt": started_at,
                "finishedAt": utc_now_iso(),
                "exitCode": result.returncode,
                "timedOut": result.timed_out,
                "message": (
                    "Replay completed"
                    if success
                    else (
                        f"Replay timed out after {timeout}s"
                        if result.timed_out
                        else f"Replay failed with exit code {result.returncode}"
                    )
                ),
                "paths": _job_paths_payload(job_id),
            },
        )
    except Exception as exc:
        logger.exception("Isaac Lab replay job %s failed", job_id)
        finalize_status(
            job_id,
            {
                "jobId": job_id,
                "kind": "replay_demo",
                "status": "failed",
                "phase": "error",
                "taskId": params.task_id,
                "datasetFile": str(params.dataset_file),
                "videoRequested": request_video,
                "videoAvailable": False,
                "startedAt": started_at,
                "finishedAt": utc_now_iso(),
                "message": str(exc),
                "paths": _job_paths_payload(job_id),
            },
        )
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_JOBS.discard(job_id)


def start_replay_demo(
    *,
    task_id: str = DEFAULT_REPLAY_TASK_ID,
    dataset_file: str,
    dataset_id: str | None = None,
    headless: bool = True,
    enable_cameras: bool = True,
    video: bool = True,
) -> dict[str, Any]:
    assert_runtime_configured_for_commands()

    runner = IsaacLabCliRunner.from_settings()
    if not runner.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=RUNTIME_NOT_CONFIGURED_MSG,
        )

    dataset_path = resolve_dataset_file(dataset_file)
    task = (task_id or DEFAULT_REPLAY_TASK_ID).strip() or DEFAULT_REPLAY_TASK_ID
    params = ReplayDemoCliParams(
        task_id=task,
        dataset_file=dataset_path,
        headless=bool(headless),
        enable_cameras=bool(enable_cameras),
        video=bool(video),
    )

    job_id = make_isaac_replay_job_id()
    job_root = isaac_job_root(job_id)
    job_root.mkdir(parents=True, exist_ok=True)
    isaac_job_artifacts_dir(job_id).mkdir(parents=True, exist_ok=True)
    meta_dir = isaac_job_metadata_dir(job_id)
    meta_dir.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "kind": "replay_demo",
        "taskId": task,
        "datasetFile": str(dataset_path),
        "datasetId": (dataset_id or "").strip() or None,
        "headless": headless,
        "enableCameras": enable_cameras,
        "video": video,
        "script": REPLAY_DEMO_SCRIPT,
        "submittedAt": utc_now_iso(),
    }
    write_json(meta_dir / "request.json", request_payload)

    started_at = utc_now_iso()
    status_payload = finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "replay_demo",
            "status": "queued",
            "phase": "queued",
            "taskId": task,
            "datasetFile": str(dataset_path),
            "command": runner.build_command(REPLAY_DEMO_SCRIPT, *build_replay_demos_cli_args(params)),
            "headless": headless,
            "enableCameras": enable_cameras,
            "videoRequested": video,
            "videoAvailable": False,
            "startedAt": started_at,
            "message": "Replay demo queued",
            "paths": _job_paths_payload(job_id),
        },
    )

    with _ACTIVE_LOCK:
        _ACTIVE_JOBS.add(job_id)

    thread = threading.Thread(
        target=_execute_replay_job,
        args=(job_id, params),
        kwargs={"request_video": bool(video)},
        name=f"isaac-replay-{job_id}",
        daemon=True,
    )
    thread.start()

    return {
        "jobId": job_id,
        "kind": "replay_demo",
        "status": status_payload.get("status", "queued"),
        "runtimePath": str(job_root),
        "statusUrl": f"/api/workspace/isaac-lab/jobs/{job_id}/status",
        "logPaths": {
            "stdout": str(isaac_job_stdout_path(job_id)),
            "stderr": str(isaac_job_stderr_path(job_id)),
        },
    }


def get_replay_job_status(job_id: str) -> dict[str, Any]:
    if not is_isaac_replay_job_id(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Isaac Lab replay job ID format",
        )
    job_root = isaac_job_root(job_id)
    if not job_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Isaac Lab replay job not found",
        )
    payload = read_json(isaac_job_status_path(job_id))
    if not payload:
        payload = {"jobId": job_id, "status": "unknown", "message": "status.json missing"}
    payload.setdefault("jobId", job_id)
    payload.setdefault("paths", _job_paths_payload(job_id))
    replay_video = isaac_job_replay_video_path(job_id)
    payload.setdefault("videoAvailable", replay_video.is_file() and replay_video.stat().st_size > 0)
    return payload


def resolve_replay_video_path(job_id: str) -> Optional[Path]:
    if not is_isaac_replay_job_id(job_id):
        return None
    status_payload = read_json(isaac_job_status_path(job_id))
    if status_payload.get("videoPath"):
        path = Path(str(status_payload["videoPath"]))
        if path.is_file():
            return path
    preferred = isaac_job_replay_video_path(job_id)
    if preferred.is_file():
        return preferred
    return _find_generated_mp4(isaac_job_artifacts_dir(job_id))
