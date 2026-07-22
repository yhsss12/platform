"""Bridge imported standalone eval jobs (eval_joint_dp_*) to cable-threading replay APIs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id, resolve_eval_job_root


def _path_info(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size: Optional[int] = None
    if exists:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
    return {"path": str(path), "exists": exists, "sizeBytes": size}


def _resolve_video_paths(job_root: Path) -> tuple[Path, Path, bool, bool]:
    eval_video = job_root / "videos" / "eval.mp4"
    eval_browser = job_root / "videos" / "eval.browser.mp4"
    browser_exists = eval_browser.is_file() and eval_browser.stat().st_size > 0
    raw_exists = eval_video.is_file() and eval_video.stat().st_size > 0
    return eval_video, eval_browser, raw_exists, browser_exists


def get_imported_eval_cable_status(job_id: str) -> dict[str, Any]:
    from app.services.evaluation.evaluation_service import get_evaluation_status
    from app.services.evaluation_workbench_basic_info import attach_workbench_basic_info

    candidate = (job_id or "").strip()
    if not is_imported_workspace_eval_job_id(candidate):
        raise ValueError(f"not an imported eval job id: {candidate}")

    eval_status = get_evaluation_status(candidate)
    job_root = resolve_eval_job_root(candidate)
    if job_root is None:
        raise FileNotFoundError(f"eval job root not found: {candidate}")

    status_path = job_root / "status.json"
    live: dict[str, Any] = {}
    if status_path.is_file():
        try:
            import json

            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                live = loaded
        except (OSError, json.JSONDecodeError):
            pass

    eval_video, eval_browser, raw_exists, browser_exists = _resolve_video_paths(job_root)
    metrics = eval_status.get("metrics") if isinstance(eval_status.get("metrics"), dict) else {}
    aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else {}

    replay_uri = eval_status.get("replayUri")
    replay_uris = eval_status.get("replayUris") if isinstance(eval_status.get("replayUris"), list) else []
    video_available = bool(
        eval_status.get("videoAvailable") or browser_exists or raw_exists or replay_uri or replay_uris
    )
    video_url = replay_uri or f"/api/workspace/evaluation/jobs/{candidate}/video"

    paths = {
        "evalVideo": _path_info(eval_video),
        "evalBrowserVideo": _path_info(eval_browser),
        "resultsJson": _path_info(job_root / "results" / "eval.results.json"),
        "log": _path_info(job_root / "logs" / "run.log"),
    }

    status_value = str(eval_status.get("status") or live.get("status") or "completed")
    payload: dict[str, Any] = {
        "jobId": candidate,
        "evalJobId": candidate,
        "taskType": "cable_threading",
        "status": status_value,
        "live": live,
        "paths": paths,
        "metrics": metrics,
        "command": "",
        "startedAt": eval_status.get("startedAt"),
        "evalVideoExists": raw_exists or video_available,
        "evalVideoSizeBytes": eval_video.stat().st_size if raw_exists else 0,
        "evalVideoPath": str(eval_video) if raw_exists else None,
        "evalBrowserVideoExists": browser_exists,
        "evalBrowserVideoSizeBytes": eval_browser.stat().st_size if browser_exists else 0,
        "evalBrowserVideoPath": str(eval_browser) if browser_exists else None,
        "browserVideoPath": str(eval_browser) if browser_exists else None,
        "videoUrl": video_url if video_available else None,
        "replayUri": replay_uri or video_url,
        "replayUris": replay_uris,
        "videoAvailable": video_available,
        "successRate": metrics.get("successRate") or aggregate.get("final_success_rate"),
        "requestedEpisodes": aggregate.get("total_episodes") or aggregate.get("num_episodes"),
        "completedEpisodes": aggregate.get("total_episodes") or aggregate.get("num_episodes"),
    }
    if metrics.get("selectedMetricIds"):
        payload["selectedMetricIds"] = metrics["selectedMetricIds"]
    if metrics.get("metricResults"):
        payload["metricResults"] = metrics["metricResults"]
    return attach_workbench_basic_info(payload, eval_job_id=candidate, job_root=job_root)


def resolve_imported_eval_video_path(job_id: str, episode: Optional[int] = None) -> Optional[Path]:
    from app.services.evaluation.evaluation_service import resolve_evaluation_video_path

    candidate = (job_id or "").strip()
    if not is_imported_workspace_eval_job_id(candidate):
        return None
    return resolve_evaluation_video_path(candidate, episode=episode)


def read_imported_eval_log_tail(job_id: str, lines: int = 40) -> str:
    from app.services.benchmark_adapters.registry import resolve_benchmark_adapter_for_eval_job

    candidate = (job_id or "").strip()
    if not is_imported_workspace_eval_job_id(candidate):
        return ""
    adapter = resolve_benchmark_adapter_for_eval_job(candidate)
    if adapter is None:
        return ""
    tail = adapter.get_log(candidate)
    if not tail.strip():
        return ""
    content = tail.splitlines()
    return "\n".join(content[-lines:])


def get_imported_eval_result(job_id: str) -> dict[str, Any]:
    from app.services.evaluation.evaluation_service import get_evaluation_result

    candidate = (job_id or "").strip()
    if not is_imported_workspace_eval_job_id(candidate):
        raise ValueError(f"not an imported eval job id: {candidate}")
    return get_evaluation_result(candidate)
