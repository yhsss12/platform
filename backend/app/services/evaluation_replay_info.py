"""评测回放元数据：区分计划轮数、完成轮数与录制视频数量。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

_EPISODE_VIDEO_PATTERN = re.compile(
    r"^episode[_-](\d+)\.mp4$",
    re.IGNORECASE,
)
_LEGACY_EPISODE_VIDEO_PATTERN = re.compile(
    r"^eval[_-]?episode[_-]?(\d+)\.mp4$",
    re.IGNORECASE,
)
_EPISODE_DIR_PATTERN = re.compile(
    r"^episode[_-](\d+)$",
    re.IGNORECASE,
)
_EPISODE_SUBDIR_VIDEO_NAMES = (
    "video.mp4",
    "replay.mp4",
    "trajectory.mp4",
    "episode_video.mp4",
)


def write_evaluation_video_metadata(
    job_root: Path,
    *,
    evaluation_mode: str,
    source_file_name: Optional[str] = None,
) -> None:
    """Persist evaluation video provenance; do not infer from generate.mp4 filename alone."""
    meta_dir = job_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "sourceKind": "evaluation",
        "evaluationMode": evaluation_mode,
    }
    if source_file_name:
        payload["fileName"] = source_file_name
    (meta_dir / "video_source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_video_source_metadata(job_root: Path) -> dict[str, Any]:
    return _read_json(job_root / "metadata" / "video_source.json")


def resolve_replay_api_prefix(job_id: str) -> str:
    candidate = (job_id or "").strip()
    if candidate.startswith("ct_eval_"):
        return "/api/workspace/cable-threading/jobs"
    return "/api/workspace/evaluation/jobs"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _pick_int(*values: Any) -> Optional[int]:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _compute_success_rate(
    successful: Optional[int], completed: Optional[int], aggregate: dict[str, Any]
) -> Optional[float]:
    if successful is not None and completed and completed > 0:
        return round(successful / completed, 6)
    rate = aggregate.get("success_rate")
    if rate is None:
        rate = aggregate.get("final_success_rate")
    if rate is None:
        rate = aggregate.get("successRate")
    if isinstance(rate, (int, float)):
        return float(rate)
    return None


def _build_warning(
    requested: Optional[int],
    completed: Optional[int],
    recorded: int,
    *,
    is_representative: bool,
) -> Optional[str]:
    if is_representative and recorded == 1:
        if requested and requested > 1:
            return "当前任务仅生成 1 段代表性回放，未找到每轮独立轨迹视频。"
    if requested and completed is not None and completed < requested:
        return "实际完成轮数少于计划轮数"
    if completed is not None and recorded > 0 and completed > recorded and not is_representative:
        return "视频段数少于实际完成轮数，请检查录制配置或 runner 输出"
    return None


def _count_successful_episodes(per_episode: list[dict[str, Any]]) -> Optional[int]:
    if not per_episode:
        return None
    count = 0
    for row in per_episode:
        if not isinstance(row, dict):
            continue
        success = row.get("success")
        if success is None:
            success = row.get("final_success")
        if success is True:
            count += 1
    return count


def _load_per_episode_rows(job_root: Path, results_data: dict[str, Any]) -> list[dict[str, Any]]:
    episodes_raw = results_data.get("episodes")
    if isinstance(episodes_raw, list) and episodes_raw:
        return [row for row in episodes_raw if isinstance(row, dict)]

    per_path = job_root / "results" / "per_episode_results.json"
    if per_path.is_file():
        try:
            loaded = json.loads(per_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                return [row for row in loaded if isinstance(row, dict)]
        except (OSError, json.JSONDecodeError):
            pass
    return []


def _episode_video_in_subdir(episode_dir: Path) -> Optional[Path]:
    for name in _EPISODE_SUBDIR_VIDEO_NAMES:
        candidate = episode_dir / name
        if candidate.is_file():
            return candidate
    nested = episode_dir / "episode" / "episode_video.mp4"
    if nested.is_file():
        return nested
    nested_videos = episode_dir / "videos" / "generate.mp4"
    if nested_videos.is_file():
        return nested_videos
    return None


def _scan_episode_subdir_videos(job_root: Path) -> list[tuple[int, Path, str]]:
    episodes_root = job_root / "episodes"
    if not episodes_root.is_dir():
        return []

    entries: list[tuple[int, Path, str]] = []
    for subdir in sorted(episodes_root.iterdir()):
        if not subdir.is_dir():
            continue
        match = _EPISODE_DIR_PATTERN.match(subdir.name)
        if not match:
            continue
        file_num = int(match.group(1))
        video_path = _episode_video_in_subdir(subdir)
        if video_path is None:
            continue
        entries.append((file_num, video_path, video_path.name))
    return entries


def _normalize_episode_indices(
    entries: list[tuple[int, Path, str]],
) -> list[tuple[int, int, Path, str]]:
    """返回 (display_index_1based, api_episode_0based, path, file_name)。"""
    if not entries:
        return []
    file_nums = [item[0] for item in entries]
    zero_based_files = any(num == 0 for num in file_nums)
    normalized: list[tuple[int, int, Path, str]] = []
    for file_num, path, name in sorted(entries, key=lambda item: item[0]):
        if zero_based_files:
            display_index = file_num + 1
            api_episode = file_num
        else:
            display_index = file_num
            api_episode = max(file_num - 1, 0)
        normalized.append((display_index, api_episode, path, name))
    return normalized


def _scan_flat_videos_dir(videos_dir: Path) -> list[tuple[int, Path, str]]:
    episode_entries: list[tuple[int, Path, str]] = []
    for path in sorted(videos_dir.glob("*.mp4")):
        name = path.name
        lower = name.lower()
        if lower.endswith(".browser.mp4"):
            continue
        match = _EPISODE_VIDEO_PATTERN.match(lower) or _LEGACY_EPISODE_VIDEO_PATTERN.match(lower)
        if match:
            file_num = int(match.group(1))
            episode_entries.append((file_num, path, name))
    return episode_entries


def _scan_replay_videos(
    job_root: Path,
    *,
    job_id: str,
    api_prefix: Optional[str] = None,
) -> tuple[list[dict[str, Any]], int, Optional[str], bool]:
    """返回 (replayUris, recordedVideoCount, defaultReplayUri, has_episode_videos)。"""
    prefix = api_prefix or resolve_replay_api_prefix(job_id)
    videos_dir = job_root / "videos"

    episode_entries = _scan_flat_videos_dir(videos_dir) if videos_dir.is_dir() else []
    if not episode_entries:
        episode_entries = _scan_episode_subdir_videos(job_root)

    representative: Optional[Path] = None
    if videos_dir.is_dir():
        for path in sorted(videos_dir.glob("*.mp4")):
            lower = path.name.lower()
            if lower.endswith(".browser.mp4"):
                continue
            if _EPISODE_VIDEO_PATTERN.match(lower) or _LEGACY_EPISODE_VIDEO_PATTERN.match(lower):
                continue
            if lower in {"eval.mp4", "generate.mp4", "demo.mp4", "replay.mp4"}:
                if representative is None or lower == "eval.mp4":
                    representative = path

    replay_uris: list[dict[str, Any]] = []

    if episode_entries:
        normalized = _normalize_episode_indices(episode_entries)
        for display_index, api_episode, _path, name in normalized:
            replay_uris.append(
                {
                    "episodeIndex": display_index,
                    "uri": f"{prefix}/{job_id}/video?episode={api_episode}",
                    "label": f"第 {display_index} 轮轨迹",
                    "fileName": name,
                    "recordCamera": None,
                }
            )
        default_uri = replay_uris[0]["uri"] if replay_uris else None
        return replay_uris, len(replay_uris), default_uri, True

    if representative and representative.is_file():
        uri = f"{prefix}/{job_id}/video"
        replay_uris.append(
            {
                "episodeIndex": None,
                "uri": uri,
                "label": "代表性回放",
                "fileName": representative.name,
            }
        )
        return replay_uris, 1, uri, False

    return [], 0, None, False


def resolve_episode_video_path(job_root: Path, episode: Optional[int] = None) -> Optional[Path]:
    """解析评测任务单轮视频路径（episode 为 0-based API 参数）。"""
    videos_dir = job_root / "videos"
    if episode is not None:
        for pattern in (f"episode_{episode:02d}.mp4", f"episode_{episode:03d}.mp4", f"episode_{episode + 1:03d}.mp4"):
            flat_candidate = videos_dir / pattern
            if flat_candidate.is_file():
                return flat_candidate
        episode_dir = job_root / "episodes" / f"episode_{episode:02d}"
        if episode_dir.is_dir():
            nested = _episode_video_in_subdir(episode_dir)
            if nested is not None:
                return nested
        return None

    if videos_dir.is_dir():
        for idx in range(100):
            candidate = videos_dir / f"episode_{idx:02d}.mp4"
            if candidate.is_file():
                return candidate
    episodes_root = job_root / "episodes"
    if episodes_root.is_dir():
        for episode_dir in sorted(episodes_root.glob("episode_*")):
            nested = _episode_video_in_subdir(episode_dir)
            if nested is not None:
                return nested
    if videos_dir.is_dir():
        for name in ("eval.mp4", "replay.mp4", "generate.mp4"):
            candidate = videos_dir / name
            if candidate.is_file():
                return candidate
    return None


def build_evaluation_replay_info(
    job_id: str,
    job_root: Path,
    *,
    live: Optional[dict[str, Any]] = None,
    results_data: Optional[dict[str, Any]] = None,
    aggregate_file: Optional[dict[str, Any]] = None,
    status_value: Optional[str] = None,
    api_prefix: Optional[str] = None,
) -> dict[str, Any]:
    live_data = dict(live or {})
    results = dict(results_data or {})
    aggregate = dict(aggregate_file or {})

    eval_context = _read_json(job_root / "metadata" / "evaluation_context.json")
    video_source_meta = _load_video_source_metadata(job_root)
    context_config = eval_context.get("config") if isinstance(eval_context.get("config"), dict) else {}
    video_source_kind = video_source_meta.get("sourceKind") or "evaluation"
    evaluation_mode = (
        video_source_meta.get("evaluationMode")
        or eval_context.get("evaluationMode")
        or context_config.get("evaluationMode")
    )

    record_camera = (
        aggregate.get("recordCamera")
        or live_data.get("recordCamera")
        or eval_context.get("recordCamera")
        or eval_context.get("evalDisplayCamera")
        or context_config.get("evalDisplayCamera")
        or context_config.get("eval_display_camera")
    )
    camera_fallback_used = bool(
        aggregate.get("cameraFallbackUsed")
        if aggregate.get("cameraFallbackUsed") is not None
        else live_data.get("cameraFallbackUsed")
        if live_data.get("cameraFallbackUsed") is not None
        else eval_context.get("cameraFallbackUsed")
    )

    per_episode = _load_per_episode_rows(job_root, results)

    requested_episodes = _pick_int(
        live_data.get("requestedEpisodes"),
        live_data.get("episodes"),
        live_data.get("totalEpisodes"),
        context_config.get("episodes"),
        eval_context.get("episodes"),
        aggregate.get("requested_episodes"),
        aggregate.get("requestedEpisodes"),
        results.get("num_episodes"),
        results.get("numEpisodes"),
        aggregate.get("total_episodes"),
        aggregate.get("totalEpisodes"),
        aggregate.get("episodes") if isinstance(aggregate.get("episodes"), int) else None,
    )

    completed_episodes = _pick_int(
        live_data.get("completedEpisodes"),
        aggregate.get("completed_episodes"),
        aggregate.get("completedEpisodes"),
        aggregate.get("total_episodes"),
        aggregate.get("totalEpisodes"),
        results.get("num_episodes"),
        results.get("numEpisodes"),
        len(per_episode) if per_episode else None,
    )

    if (
        status_value in {"completed", "failed"}
        and requested_episodes
        and (completed_episodes is None or completed_episodes < requested_episodes)
        and per_episode
    ):
        completed_episodes = len(per_episode)

    if status_value == "completed" and requested_episodes and completed_episodes is None:
        completed_episodes = requested_episodes

    successful_episodes = _pick_int(
        aggregate.get("success_episodes"),
        aggregate.get("successEpisodes"),
        aggregate.get("successfulEpisodes"),
        _count_successful_episodes(per_episode),
    )

    failure_count = _pick_int(aggregate.get("failure_count"), aggregate.get("failureCount"))
    failed_episodes: Optional[int] = None
    if failure_count is not None:
        failed_episodes = failure_count
    elif successful_episodes is not None and completed_episodes is not None:
        failed_episodes = max(completed_episodes - successful_episodes, 0)
    else:
        failed_episodes = _pick_int(aggregate.get("failedEpisodes"), aggregate.get("failed_episodes"))

    replay_uris, recorded_video_count, replay_uri, has_episode_videos = _scan_replay_videos(
        job_root,
        job_id=job_id,
        api_prefix=api_prefix,
    )
    if record_camera:
        for item in replay_uris:
            item["recordCamera"] = record_camera

    for item in replay_uris:
        item["sourceKind"] = video_source_kind
        if evaluation_mode:
            item["evaluationMode"] = evaluation_mode
        episode_idx = item.get("episodeIndex")
        if not isinstance(episode_idx, int) or episode_idx <= 0:
            continue
        per_row = next(
            (
                row
                for row in per_episode
                if _pick_int(row.get("episode"), row.get("episodeIndex")) in {episode_idx - 1, episode_idx}
            ),
            None,
        )
        if isinstance(per_row, dict):
            success = per_row.get("success")
            if success is None:
                success = per_row.get("final_success")
            if isinstance(success, bool):
                item["success"] = success

    video_available = recorded_video_count > 0 or bool(live_data.get("evalVideoExists"))
    success_rate = _compute_success_rate(successful_episodes, completed_episodes, aggregate)
    is_representative = recorded_video_count == 1 and bool(replay_uris) and not has_episode_videos
    warning = _build_warning(
        requested_episodes,
        completed_episodes,
        recorded_video_count,
        is_representative=is_representative,
    )

    return {
        "requestedEpisodes": requested_episodes,
        "completedEpisodes": completed_episodes,
        "successfulEpisodes": successful_episodes,
        "failedEpisodes": failed_episodes,
        "successRate": success_rate,
        "recordedVideoCount": recorded_video_count,
        "replayUri": replay_uri,
        "replayUris": replay_uris,
        "videoAvailable": video_available,
        "videoSourceKind": video_source_kind,
        "evaluationMode": evaluation_mode,
        "isRepresentativeVideo": is_representative,
        "currentEpisodeIndex": _pick_int(live_data.get("episode"), live_data.get("currentEpisode")),
        "warning": warning,
        "recordCamera": record_camera,
        "cameraFallbackUsed": camera_fallback_used,
    }


def build_cable_threading_replay_info(
    job_id: str,
    job_root: Path,
    *,
    live: Optional[dict[str, Any]] = None,
    results_data: Optional[dict[str, Any]] = None,
    aggregate_file: Optional[dict[str, Any]] = None,
    status_value: Optional[str] = None,
) -> dict[str, Any]:
    return build_evaluation_replay_info(
        job_id,
        job_root,
        live=live,
        results_data=results_data,
        aggregate_file=aggregate_file,
        status_value=status_value,
        api_prefix="/api/workspace/cable-threading/jobs",
    )
