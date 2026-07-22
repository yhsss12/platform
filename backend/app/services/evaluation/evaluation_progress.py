"""评测任务进度：统一计算 requested / completed / progressPercent / progressLabel。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

_EPISODE_DIR_PATTERN = re.compile(r"^episode[_-](\d+)$", re.IGNORECASE)

_RUNNING_STATUSES = frozenset({"running", "evaluating", "queued", "pending"})
_TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled", "cancelled", "stale"})


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


def _pick_float(*values: Any) -> Optional[float]:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip())
            except ValueError:
                continue
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _count_episode_artifacts(job_root: Path) -> Optional[int]:
    episodes_root = job_root / "episodes"
    if not episodes_root.is_dir():
        return None
    count = 0
    for subdir in episodes_root.iterdir():
        if not subdir.is_dir():
            continue
        if not _EPISODE_DIR_PATTERN.match(subdir.name):
            continue
        if (subdir / "episode" / "episode_result.json").is_file():
            count += 1
            continue
        if (subdir / "run_status.json").is_file():
            count += 1
            continue
        if any(subdir.glob("**/*.mp4")):
            count += 1
    return count if count > 0 else None


def _count_flat_episode_videos(job_root: Path) -> Optional[int]:
    videos_dir = job_root / "videos"
    if not videos_dir.is_dir():
        return None
    count = sum(
        1
        for path in videos_dir.glob("episode_*.mp4")
        if path.is_file() and not path.name.lower().endswith(".browser.mp4")
    )
    return count if count > 0 else None


def _count_per_episode_results(job_root: Path) -> Optional[int]:
    for relative in (
        "results/per_episode_results.json",
        "results/per_episode_result.json",
    ):
        payload = _read_json(job_root / relative)
        episodes = payload.get("episodes")
        if isinstance(episodes, list) and episodes:
            return len(episodes)
        if isinstance(payload, list) and payload:
            return len(payload)
    return None


def _load_runtime_status(job_id: str, runtime_path: Optional[str]) -> dict[str, Any]:
    from app.services.evaluation.evaluation_runtime_health import resolve_evaluation_job_root

    candidate = (job_id or "").strip()
    if not candidate:
        return {}

    job_root = resolve_evaluation_job_root(candidate, runtime_path)
    if job_root is None:
        return {}

    if candidate.startswith("ct_eval_"):
        for rel in ("live/status.json", "status.json"):
            data = _read_json(job_root / rel)
            if data:
                return data
        return {}

    for rel in ("status.json", "metadata/status.json", "live/status.json"):
        data = _read_json(job_root / rel)
        if data:
            return data
    return {}


def resolve_evaluation_progress(
    *,
    status: str,
    metrics: Optional[dict[str, Any]] = None,
    summary_json: Optional[dict[str, Any]] = None,
    runtime_status: Optional[dict[str, Any]] = None,
    job_id: Optional[str] = None,
    runtime_path: Optional[str] = None,
) -> dict[str, Any]:
    """返回 requestedEpisodes / completedEpisodes / progress / progressPercent / progressLabel。"""
    metrics_data = dict(metrics or {})
    summary = dict(summary_json or {})
    runtime = dict(runtime_status or {})

    if not runtime and job_id:
        runtime = _load_runtime_status(job_id, runtime_path)

    status_norm = str(status or runtime.get("status") or metrics_data.get("status") or "").strip().lower()

    requested_episodes = _pick_int(
        metrics_data.get("requestedEpisodes"),
        summary.get("requestedEpisodes"),
        runtime.get("requestedEpisodes"),
        runtime.get("totalEpisodes"),
        metrics_data.get("totalEpisodes"),
        summary.get("totalEpisodes"),
        metrics_data.get("episodes"),
        runtime.get("episodes"),
        metrics_data.get("numEpisodes"),
    )

    completed_episodes = _pick_int(
        metrics_data.get("completedEpisodes"),
        summary.get("completedEpisodes"),
        runtime.get("completedEpisodes"),
        summary.get("total_episodes"),
        metrics_data.get("total_episodes"),
    )

    current_episode = _pick_int(
        metrics_data.get("currentEpisode"),
        summary.get("currentEpisode"),
        runtime.get("currentEpisode"),
    )

    if completed_episodes is None and status_norm in _TERMINAL_STATUSES:
        completed_episodes = _pick_int(
            summary.get("total_episodes"),
            summary.get("totalEpisodes"),
            metrics_data.get("successfulEpisodes"),
            metrics_data.get("success_episodes"),
        )

    if completed_episodes is None and current_episode is not None and requested_episodes:
        phase = str(runtime.get("phase") or "").strip().lower()
        if status_norm in _RUNNING_STATUSES or phase in {"episode_running", "evaluating", "running", "queued"}:
            if current_episode > 0:
                completed_episodes = min(max(current_episode - 1, 0), requested_episodes)
        elif status_norm == "completed":
            completed_episodes = min(current_episode, requested_episodes)

    if completed_episodes is None and job_id:
        from app.services.evaluation.evaluation_runtime_health import resolve_evaluation_job_root

        job_root = resolve_evaluation_job_root(job_id, runtime_path)
        if job_root is not None:
            completed_episodes = _count_per_episode_results(job_root)
            if completed_episodes is None:
                completed_episodes = _count_flat_episode_videos(job_root)
            if completed_episodes is None:
                completed_episodes = _count_episode_artifacts(job_root)

    progress_raw = _pick_float(
        metrics_data.get("progress"),
        summary.get("progress"),
        runtime.get("progress"),
    )
    if progress_raw is not None and progress_raw > 1.0:
        progress_raw = progress_raw / 100.0

    total_episodes = requested_episodes or _pick_int(runtime.get("totalEpisodes"), metrics_data.get("totalEpisodes"))

    if (
        completed_episodes is None
        and progress_raw is not None
        and total_episodes
        and total_episodes > 0
    ):
        completed_episodes = min(total_episodes, max(0, round(progress_raw * total_episodes)))

    if status_norm == "completed" and total_episodes:
        completed_episodes = total_episodes
        progress_raw = 1.0
    elif status_norm in {"failed", "canceled", "cancelled", "stale"} and completed_episodes is None:
        completed_episodes = 0
    elif status_norm in _RUNNING_STATUSES and completed_episodes is None and total_episodes:
        completed_episodes = 0

    progress_float: Optional[float] = None
    progress_percent: Optional[int] = None
    progress_label: Optional[str] = None

    if total_episodes and total_episodes > 0 and completed_episodes is not None:
        progress_float = max(0.0, min(1.0, completed_episodes / total_episodes))
        progress_percent = int(round(progress_float * 100))
        progress_label = f"{completed_episodes}/{total_episodes}"
    elif progress_raw is not None:
        progress_float = max(0.0, min(1.0, progress_raw))
        progress_percent = int(round(progress_float * 100))
        if current_episode and total_episodes:
            progress_label = f"第 {current_episode}/{total_episodes} 轮"
        elif total_episodes:
            progress_label = f"{progress_percent}%"

    return {
        "requestedEpisodes": total_episodes,
        "completedEpisodes": completed_episodes,
        "currentEpisode": current_episode,
        "totalEpisodes": total_episodes,
        "progress": progress_float,
        "progressPercent": progress_percent,
        "progressLabel": progress_label,
    }
