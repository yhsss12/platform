"""Isaac Lab job 目录规范（Phase 1：路径约定，不启动仿真）。"""

from __future__ import annotations

import re
from pathlib import Path

from app.services.isaac_lab.paths import PROJECT_ROOT
from app.core.config import settings
from app.core.platform_paths import platform_paths

ISAAC_EVAL_JOB_ID_PATTERN = re.compile(r"^isaac_eval_\d{8}_\d{6}_[a-f0-9]{4}$")
ISAAC_GEN_JOB_ID_PATTERN = re.compile(r"^isaac_gen_\d{8}_\d{6}_[a-f0-9]{4}$")
ISAAC_RUN_JOB_ID_PATTERN = re.compile(r"^isaac_run_\d{8}_\d{6}_[a-f0-9]{4}$")
ISAAC_REPLAY_JOB_ID_PATTERN = re.compile(r"^isaac_replay_\d{8}_\d{6}_[a-f0-9]{4}$")


def _output_root() -> Path:
    raw = (settings.ISAACLAB_OUTPUT_ROOT or "runs/isaac_lab/jobs").strip()
    if raw == "runs/isaac_lab/jobs":
        return platform_paths.runs_root / "isaac_lab" / "jobs"
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def isaac_job_root(eval_job_id: str) -> Path:
    return _output_root() / eval_job_id


def isaac_job_metadata_dir(eval_job_id: str) -> Path:
    return isaac_job_root(eval_job_id) / "metadata"


def isaac_job_logs_dir(eval_job_id: str) -> Path:
    return isaac_job_root(eval_job_id) / "logs"


def isaac_job_results_dir(eval_job_id: str) -> Path:
    return isaac_job_root(eval_job_id) / "results"


def isaac_job_artifacts_dir(eval_job_id: str) -> Path:
    return isaac_job_root(eval_job_id) / "artifacts"


def isaac_job_stdout_path(job_id: str) -> Path:
    return isaac_job_root(job_id) / "stdout.log"


def isaac_job_stderr_path(job_id: str) -> Path:
    return isaac_job_root(job_id) / "stderr.log"


def isaac_job_status_path(job_id: str) -> Path:
    return isaac_job_root(job_id) / "status.json"


def isaac_job_replay_manifest_path(job_id: str) -> Path:
    return isaac_job_root(job_id) / "replay_manifest.json"


def isaac_job_replay_video_path(job_id: str) -> Path:
    return isaac_job_artifacts_dir(job_id) / "replay.mp4"


def isaac_job_preview_video_path(job_id: str) -> Path:
    return isaac_job_artifacts_dir(job_id) / "preview.mp4"


def isaac_job_videos_dir(job_id: str) -> Path:
    return isaac_job_artifacts_dir(job_id) / "videos"


def isaac_job_browser_preview_video_path(job_id: str) -> Path:
    return isaac_job_artifacts_dir(job_id) / "preview.browser.mp4"


def isaac_job_browser_replay_video_path(job_id: str) -> Path:
    return isaac_job_artifacts_dir(job_id) / "replay.browser.mp4"


def isaac_job_live_dir(job_id: str) -> Path:
    return isaac_job_root(job_id) / "live"


def isaac_job_live_latest_frame_path(job_id: str) -> Path:
    return isaac_job_live_dir(job_id) / "latest.jpg"


def isaac_job_live_frames_dir(job_id: str) -> Path:
    return isaac_job_live_dir(job_id) / "frames"


def isaac_job_live_status_path(job_id: str) -> Path:
    return isaac_job_live_dir(job_id) / "live_status.json"


def isaac_job_dataset_path(job_id: str) -> Path:
    return isaac_job_artifacts_dir(job_id) / "dataset.hdf5"


def isaac_job_generation_manifest_path(job_id: str) -> Path:
    return isaac_job_root(job_id) / "generation_manifest.json"


def isaac_job_metrics_path(job_id: str) -> Path:
    return isaac_job_root(job_id) / "metrics.json"


def is_isaac_replay_job_id(job_id: str) -> bool:
    return ISAAC_REPLAY_JOB_ID_PATTERN.match((job_id or "").strip()) is not None


def is_isaac_lab_cli_job_id(job_id: str) -> bool:
    candidate = (job_id or "").strip()
    return (
        is_isaac_run_job_id(candidate)
        or is_isaac_replay_job_id(candidate)
        or is_isaac_eval_job_id(candidate)
        or is_isaac_gen_job_id(candidate)
    )


def is_isaac_run_job_id(job_id: str) -> bool:
    return ISAAC_RUN_JOB_ID_PATTERN.match((job_id or "").strip()) is not None


def is_isaac_eval_job_id(eval_job_id: str) -> bool:
    from app.services.evaluation.job_paths import is_isaac_eval_job_id as _is_isaac_eval_job_id

    return _is_isaac_eval_job_id(eval_job_id)


def is_isaac_gen_job_id(job_id: str) -> bool:
    return ISAAC_GEN_JOB_ID_PATTERN.match((job_id or "").strip()) is not None
