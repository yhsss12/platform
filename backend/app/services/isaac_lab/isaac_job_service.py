"""Isaac Lab CLI job 统一查询（smoke / replay）。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import HTTPException, status

from app.services.isaac_lab.live_frame_utils import frame_image_is_valid
from app.services.isaac_lab.job_paths import (
    isaac_job_artifacts_dir,
    isaac_job_browser_replay_video_path,
    isaac_job_live_latest_frame_path,
    isaac_job_preview_video_path,
    isaac_job_replay_video_path,
    isaac_job_stderr_path,
    isaac_job_stdout_path,
    isaac_job_videos_dir,
    is_isaac_gen_job_id,
    is_isaac_lab_cli_job_id,
    is_isaac_replay_job_id,
    is_isaac_run_job_id,
)
from app.services.isaac_lab import generate_service as generate_svc
from app.services.isaac_lab import replay_service as replay_svc
from app.services.isaac_lab import smoke_test_service as smoke_svc
from app.services.isaac_lab.video_compat import ensure_browser_playable_mp4, is_browser_compatible_mp4, probe_mp4_codec


def get_job_status(job_id: str) -> dict:
    if is_isaac_run_job_id(job_id):
        return smoke_svc.get_smoke_test_job_status(job_id)
    if is_isaac_replay_job_id(job_id):
        return replay_svc.get_replay_job_status(job_id)
    if is_isaac_gen_job_id(job_id):
        return generate_svc.get_generate_job_status(job_id)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported Isaac Lab job ID format",
    )


def read_job_log_tail(job_id: str, *, stream: str = "stdout", lines: int = 80) -> str:
    if not is_isaac_lab_cli_job_id(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Isaac Lab job ID format",
        )
    path = isaac_job_stdout_path(job_id) if stream == "stdout" else isaac_job_stderr_path(job_id)
    if not path.is_file():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _first_existing_mp4(directory: Path) -> Optional[Path]:
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.mp4"))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _resolve_replay_raw_video(job_id: str) -> Optional[Path]:
    if not is_isaac_replay_job_id(job_id):
        return None
    browser = isaac_job_browser_replay_video_path(job_id)
    if browser.is_file() and browser.stat().st_size > 0 and is_browser_compatible_mp4(browser):
        return browser
    preferred = isaac_job_replay_video_path(job_id)
    if preferred.is_file() and preferred.stat().st_size > 0:
        return preferred
    resolved = replay_svc.resolve_replay_video_path(job_id)
    if resolved is not None and resolved.is_file():
        return resolved
    return _first_existing_mp4(isaac_job_artifacts_dir(job_id))


def resolve_job_video_path(job_id: str) -> Optional[Path]:
    if is_isaac_replay_job_id(job_id):
        return _resolve_replay_raw_video(job_id)

    if is_isaac_gen_job_id(job_id):
        path = generate_svc.resolve_generate_video_path(job_id)
        if path is not None:
            return path
        preview = isaac_job_preview_video_path(job_id)
        if preview.is_file():
            return preview
        replay = isaac_job_replay_video_path(job_id)
        if replay.is_file():
            return replay
        return _first_existing_mp4(isaac_job_videos_dir(job_id))

    return None


def resolve_job_browser_video_path(job_id: str) -> tuple[Optional[Path], Optional[str], dict[str, object]]:
    raw = resolve_job_video_path(job_id)
    if raw is None:
        return None, "Replay video not available for this job", {}
    playable, note = ensure_browser_playable_mp4(raw)
    if playable is None:
        return None, note or "video not browser-compatible", {}
    probe = probe_mp4_codec(raw)
    meta = {
        "codec": probe.get("codec"),
        "transcoded": note in {"transcoded", "transcoded_cache"},
        "rawPath": str(raw),
    }
    if is_isaac_replay_job_id(job_id):
        meta["videoSource"] = "replay" if note is None else "converted"
    elif raw.name.startswith("preview"):
        meta["videoSource"] = "preview" if note is None else "converted"
    else:
        meta["videoSource"] = "videos" if note is None else "converted"
    return playable, note, meta


def resolve_job_live_frame_path(job_id: str) -> Optional[Path]:
    if not is_isaac_gen_job_id(job_id):
        return None
    latest = isaac_job_live_latest_frame_path(job_id)
    if frame_image_is_valid(latest):
        return latest
    return None
