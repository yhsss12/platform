"""Isaac Lab 数据集回放页上下文（视频优先级、replay job 发现）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from app.services.isaac_lab import generate_service as generate_svc
from app.services.isaac_lab import isaac_dataset_service as dataset_svc
from app.services.isaac_lab import job_paths
from app.services.isaac_lab import replay_service as replay_svc
from app.services.isaac_lab.isaac_job_utils import read_json
from app.services.isaac_lab.job_paths import (
    isaac_job_artifacts_dir,
    isaac_job_browser_preview_video_path,
    isaac_job_metadata_dir,
    isaac_job_preview_video_path,
    isaac_job_replay_video_path,
    isaac_job_status_path,
    isaac_job_videos_dir,
    is_isaac_gen_job_id,
    is_isaac_replay_job_id,
)
from app.services.isaac_lab.video_compat import (
    ensure_browser_playable_mp4,
    is_browser_compatible_mp4,
    probe_mp4_codec,
)

VideoSourceKind = Literal["replay", "preview", "videos", "converted", "none"]


def _first_existing_mp4(directory: Path) -> Optional[Path]:
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.glob("*.mp4")):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _resolve_replay_raw_video(job_id: str) -> Optional[Path]:
    if not is_isaac_replay_job_id(job_id):
        return None
    browser = isaac_job_artifacts_dir(job_id) / "replay.browser.mp4"
    if browser.is_file() and browser.stat().st_size > 0 and is_browser_compatible_mp4(browser):
        return browser
    preferred = isaac_job_replay_video_path(job_id)
    if preferred.is_file() and preferred.stat().st_size > 0:
        return preferred
    resolved = replay_svc.resolve_replay_video_path(job_id)
    if resolved is not None and resolved.is_file():
        return resolved
    return _first_existing_mp4(isaac_job_artifacts_dir(job_id))


def _resolve_source_raw_video(job_id: str) -> Optional[Path]:
    if not is_isaac_gen_job_id(job_id):
        return None
    browser = isaac_job_browser_preview_video_path(job_id)
    if browser.is_file() and browser.stat().st_size > 0 and is_browser_compatible_mp4(browser):
        return browser
    generated = generate_svc.resolve_generate_video_path(job_id)
    if generated is not None and generated.is_file():
        return generated
    preview = isaac_job_preview_video_path(job_id)
    if preview.is_file() and preview.stat().st_size > 0:
        return preview
    replay = isaac_job_replay_video_path(job_id)
    if replay.is_file() and replay.stat().st_size > 0:
        return replay
    return _first_existing_mp4(isaac_job_videos_dir(job_id))


def _describe_video_candidate(
    *,
    job_id: str,
    source_kind: Literal["replay", "preview", "videos"],
    raw_path: Path,
) -> dict[str, Any]:
    probe = probe_mp4_codec(raw_path)
    playable_path, transcode_note = ensure_browser_playable_mp4(raw_path)
    transcoded = transcode_note in {"transcoded", "transcoded_cache"}
    video_source: VideoSourceKind = source_kind
    if transcoded:
        video_source = "converted"
    return {
        "videoJobId": job_id,
        "videoSource": video_source,
        "videoSourceKind": source_kind,
        "videoPath": str(playable_path or raw_path),
        "rawVideoPath": str(raw_path),
        "browserVideoPath": str(playable_path) if playable_path else None,
        "codec": str(probe.get("codec") or "unknown"),
        "browserCompatible": bool(probe.get("browserCompatible")),
        "transcoded": transcoded,
        "transcodeNote": transcode_note,
        "playable": playable_path is not None,
    }


def _read_replay_job_row(job_id: str) -> Optional[dict[str, Any]]:
    if not is_isaac_replay_job_id(job_id):
        return None
    root = job_paths._output_root() / job_id
    if not root.is_dir():
        return None
    request = read_json(isaac_job_metadata_dir(job_id) / "request.json")
    status_payload = read_json(isaac_job_status_path(job_id)) or {"jobId": job_id, "status": "unknown"}
    raw_video = _resolve_replay_raw_video(job_id)
    return {
        "jobId": job_id,
        "status": str(status_payload.get("status") or "unknown"),
        "phase": status_payload.get("phase"),
        "message": status_payload.get("message"),
        "exitCode": status_payload.get("exitCode"),
        "datasetFile": request.get("datasetFile") or status_payload.get("datasetFile"),
        "datasetId": request.get("datasetId"),
        "videoAvailable": raw_video is not None,
        "finishedAt": status_payload.get("finishedAt"),
        "updatedAt": status_payload.get("updatedAt"),
    }


def list_replay_jobs_for_dataset(
    *,
    dataset_id: str,
    dataset_file: Path,
) -> list[dict[str, Any]]:
    resolved_file = dataset_file.resolve()
    rows: list[dict[str, Any]] = []
    root = job_paths._output_root()
    if not root.is_dir():
        return rows

    for job_dir in sorted(root.glob("isaac_replay_*"), reverse=True):
        if not job_dir.is_dir():
            continue
        row = _read_replay_job_row(job_dir.name)
        if row is None:
            continue
        row_dataset_id = str(row.get("datasetId") or "").strip()
        row_dataset_file = str(row.get("datasetFile") or "").strip()
        matched = False
        if row_dataset_id and row_dataset_id == dataset_id:
            matched = True
        elif row_dataset_file:
            try:
                matched = Path(row_dataset_file).expanduser().resolve() == resolved_file
            except OSError:
                matched = False
        if matched:
            rows.append(row)
    return rows


def find_reusable_replay_job(
    *,
    dataset_id: str,
    dataset_file: Path,
) -> Optional[dict[str, Any]]:
    jobs = list_replay_jobs_for_dataset(dataset_id=dataset_id, dataset_file=dataset_file)
    if not jobs:
        return None

    for row in jobs:
        if row.get("status") == "completed" and row.get("videoAvailable"):
            return row

    for row in jobs:
        if row.get("status") in {"queued", "running"}:
            return row

    for row in jobs:
        if row.get("status") == "completed":
            return row

    return jobs[0]


def resolve_dataset_playback(dataset_id: str) -> dict[str, Any]:
    dataset = dataset_svc.get_isaac_dataset(dataset_id)
    dataset_file = Path(str(dataset["datasetFile"])).expanduser()
    source_job_id = str(dataset.get("sourceJobId") or "")
    replay_jobs = list_replay_jobs_for_dataset(dataset_id=dataset_id, dataset_file=dataset_file)
    active_replay_job = find_reusable_replay_job(dataset_id=dataset_id, dataset_file=dataset_file)

    source_job_status: Optional[dict[str, Any]] = None
    if source_job_id.startswith("isaac_gen_"):
        try:
            source_job_status = generate_svc.get_generate_job_status(source_job_id)
        except Exception:
            source_job_status = None

    playback: Optional[dict[str, Any]] = None
    preview_fallback: Optional[dict[str, Any]] = None

    if active_replay_job:
        replay_job_id = str(active_replay_job["jobId"])
        raw_replay = _resolve_replay_raw_video(replay_job_id)
        if raw_replay is not None:
            playback = _describe_video_candidate(
                job_id=replay_job_id,
                source_kind="replay",
                raw_path=raw_replay,
            )

    if playback is None and source_job_id.startswith("isaac_gen_"):
        raw_source = _resolve_source_raw_video(source_job_id)
        if raw_source is not None:
            if raw_source.parent.name == "videos":
                source_kind = "videos"
            elif raw_source.name.startswith("preview"):
                source_kind = "preview"
            else:
                source_kind = "preview"
            preview_fallback = _describe_video_candidate(
                job_id=source_job_id,
                source_kind=source_kind,
                raw_path=raw_source,
            )
            playback = preview_fallback

    replay_job_row = active_replay_job
    replay_in_progress = bool(
        replay_job_row
        and replay_job_row.get("status") in {"queued", "running"}
        and (playback is None or playback.get("videoSourceKind") != "replay")
    )
    replay_failed = bool(replay_job_row and replay_job_row.get("status") == "failed")

    return {
        "dataset": dataset,
        "sourceJobId": source_job_id or None,
        "sourceJobStatus": source_job_status,
        "replayJobs": replay_jobs,
        "replayJobId": str(replay_job_row["jobId"]) if replay_job_row else None,
        "replayJobStatus": replay_job_row.get("status") if replay_job_row else None,
        "replayInProgress": replay_in_progress,
        "replayFailed": replay_failed,
        "playback": playback,
        "usingPreviewFallback": bool(
            playback is not None
            and playback.get("videoSourceKind") in {"preview", "videos"}
            and not (replay_job_row and replay_job_row.get("videoAvailable"))
        ),
        "hasDatasetFile": dataset_file.is_file(),
    }


def video_source_label(video_source: str | None, *, transcoded: bool = False) -> str:
    if transcoded or video_source == "converted":
        return "视频来源：浏览器兼容转码视频"
    if video_source == "replay":
        return "视频来源：数据集回放 replay.mp4"
    if video_source == "preview":
        return "视频来源：数据生成 preview.mp4"
    if video_source == "videos":
        return "视频来源：生成过程 videos/*.mp4"
    return "视频来源：暂无"
