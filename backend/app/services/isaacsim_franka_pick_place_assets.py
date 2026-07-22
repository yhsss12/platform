"""Isaac Sim Franka Pick Place 任务资产校验与视频路径解析。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

TASK_ID = "isaacsim_franka_pick_place"

VIDEO_STATUS_PENDING = "pending"
VIDEO_STATUS_AVAILABLE = "available"
VIDEO_STATUS_FAILED = "failed"
VIDEO_STATUS_PARTIAL = "partial"
VIDEO_STATUS_INVALID = "invalid"

VALID_VIDEO_STATUSES = frozenset(
    {
        VIDEO_STATUS_PENDING,
        VIDEO_STATUS_AVAILABLE,
        VIDEO_STATUS_FAILED,
        VIDEO_STATUS_PARTIAL,
        VIDEO_STATUS_INVALID,
    }
)

FORBIDDEN_VIDEO_PATH_KEYWORDS = (
    "cable",
    "thread",
    "threading",
    "dual_arm",
    "dac_gen",
    "ct_gen",
    "cable_threading",
    "dual_arm_cable",
    "panda_composite_cable",
    "isaac_block_stacking",
    "isaac_gen_",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACK_ROOT = (
    PROJECT_ROOT
    / "integrations"
    / "IsaacSimFrankaPickPlace"
)
PACK_DEMO_VIDEO_ROOT = PACK_ROOT / "demo_data" / "videos"
PACK_DEMO_VIDEO_MANIFEST = PACK_DEMO_VIDEO_ROOT / "video_asset_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def contains_forbidden_video_path_hint(path: str | Path) -> bool:
    lowered = str(path).replace("\\", "/").lower()
    return any(keyword in lowered for keyword in FORBIDDEN_VIDEO_PATH_KEYWORDS)


def normalize_episode_video_status(
    *,
    video_available: bool,
    video_status: str | None = None,
    runtime_mode: str = "packaged_assets",
    save_video: bool = True,
    recording_attempted: bool = False,
) -> str:
    if video_status in VALID_VIDEO_STATUSES:
        return str(video_status)
    if video_available:
        return VIDEO_STATUS_AVAILABLE
    if runtime_mode == "isaacsim" and save_video and recording_attempted:
        return VIDEO_STATUS_FAILED
    return VIDEO_STATUS_PENDING


def aggregate_dataset_video_status(episode_results: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return (video_available, video_status) for dataset manifest aggregation."""
    if not episode_results:
        return False, VIDEO_STATUS_PENDING

    statuses: list[str] = []
    for episode in episode_results:
        status = episode.get("video_status")
        if status not in VALID_VIDEO_STATUSES:
            status = VIDEO_STATUS_AVAILABLE if episode.get("video_available") else VIDEO_STATUS_PENDING
        statuses.append(str(status))

    available_count = sum(1 for status in statuses if status == VIDEO_STATUS_AVAILABLE)
    failed_count = sum(1 for status in statuses if status == VIDEO_STATUS_FAILED)
    pending_count = sum(1 for status in statuses if status == VIDEO_STATUS_PENDING)
    total = len(statuses)

    if available_count == total:
        return True, VIDEO_STATUS_AVAILABLE
    if available_count > 0:
        return True, VIDEO_STATUS_PARTIAL
    if failed_count > 0 and pending_count == 0:
        return False, VIDEO_STATUS_FAILED
    return False, VIDEO_STATUS_PENDING


def sync_video_status_fields(payload: dict[str, Any], video_status: str) -> None:
    payload["video_status"] = video_status
    payload["videoStatus"] = video_status


def validate_manifest_task_ids(
    *,
    dataset_manifest: dict[str, Any],
    episode_manifest: dict[str, Any],
    expected_task_id: str = TASK_ID,
) -> tuple[bool, Optional[str]]:
    dataset_task_id = str(
        dataset_manifest.get("source_task_id")
        or dataset_manifest.get("task_id")
        or dataset_manifest.get("taskType")
        or ""
    ).strip()
    episode_task_id = str(episode_manifest.get("task_id") or "").strip()
    expected = expected_task_id.strip()

    if dataset_task_id != expected:
        return False, f"dataset_manifest.task_id mismatch: {dataset_task_id!r}"
    if episode_task_id != expected:
        return False, f"episode_manifest.task_id mismatch: {episode_task_id!r}"
    if dataset_task_id != episode_task_id:
        return False, "dataset_manifest and episode_manifest task_id differ"
    return True, None


def resolve_pack_demo_video_path(episode_id: str) -> Optional[Path]:
    """仅允许复制任务包 manifest 中登记、且属于本任务的 demo 视频。"""
    manifest = _read_json(PACK_DEMO_VIDEO_MANIFEST)
    pack_task_id = str(manifest.get("task_id") or "").strip()
    if pack_task_id != TASK_ID:
        return None

    videos = manifest.get("videos")
    if not isinstance(videos, dict):
        return None

    entry = videos.get(episode_id)
    if not isinstance(entry, dict):
        return None

    entry_task_id = str(entry.get("task_id") or pack_task_id).strip()
    if entry_task_id != TASK_ID:
        return None

    filename = str(entry.get("file") or f"{episode_id}.mp4").strip()
    if not filename or contains_forbidden_video_path_hint(filename):
        return None

    candidate = (PACK_DEMO_VIDEO_ROOT / filename).resolve()
    demo_root = PACK_DEMO_VIDEO_ROOT.resolve()
    if not str(candidate).startswith(str(demo_root)):
        return None
    if contains_forbidden_video_path_hint(candidate):
        return None
    return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None


def resolve_job_episode_video_path(
    job_root: Path,
    episode_id: str = "ep_000001",
    *,
    expected_task_id: str = TASK_ID,
) -> tuple[Optional[Path], dict[str, Any]]:
    job_root = job_root.resolve()
    dataset_manifest = _read_json(job_root / "dataset_manifest.json")
    episode_manifest = _read_json(job_root / "episodes" / episode_id / "episode_manifest.json")

    meta: dict[str, Any] = {
        "episodeId": episode_id,
        "datasetTaskId": dataset_manifest.get("task_id") or dataset_manifest.get("source_task_id"),
        "episodeTaskId": episode_manifest.get("task_id"),
        "taskIdValidated": False,
        "video_status": "pending",
        "videoStatus": "pending",
        "videoPath": None,
        "manifestVideoPath": episode_manifest.get("video_path"),
    }

    ok, reason = validate_manifest_task_ids(
        dataset_manifest=dataset_manifest,
        episode_manifest=episode_manifest,
        expected_task_id=expected_task_id,
    )
    meta["taskIdValidated"] = ok
    if not ok:
        meta["video_status"] = "invalid"
        meta["videoStatus"] = "invalid"
        meta["validationError"] = reason
        return None, meta

    rel_video = episode_manifest.get("video_path")
    manifest_video_status = str(
        episode_manifest.get("video_status") or episode_manifest.get("videoStatus") or ""
    ).strip()

    if manifest_video_status == VIDEO_STATUS_FAILED:
        meta["video_status"] = VIDEO_STATUS_FAILED
        meta["videoStatus"] = VIDEO_STATUS_FAILED
        return None, meta

    if not rel_video:
        status = manifest_video_status or VIDEO_STATUS_PENDING
        meta["video_status"] = status
        meta["videoStatus"] = status
        return None, meta

    rel = str(rel_video).strip().lstrip("/")
    if contains_forbidden_video_path_hint(rel):
        meta["video_status"] = "invalid"
        meta["videoStatus"] = "invalid"
        meta["validationError"] = "forbidden cross-task video path"
        return None, meta

    candidate = (job_root / rel).resolve()
    if not str(candidate).startswith(str(job_root)):
        meta["video_status"] = "invalid"
        meta["videoStatus"] = "invalid"
        meta["validationError"] = "video path escapes job directory"
        return None, meta
    if contains_forbidden_video_path_hint(candidate):
        meta["video_status"] = "invalid"
        meta["videoStatus"] = "invalid"
        meta["validationError"] = "forbidden cross-task video asset"
        return None, meta
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        status = manifest_video_status or VIDEO_STATUS_PENDING
        meta["video_status"] = status
        meta["videoStatus"] = status
        return None, meta

    meta["video_status"] = VIDEO_STATUS_AVAILABLE
    meta["videoStatus"] = VIDEO_STATUS_AVAILABLE
    meta["videoPath"] = str(candidate)
    return candidate, meta
