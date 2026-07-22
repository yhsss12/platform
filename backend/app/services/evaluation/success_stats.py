"""评测任务成功率统计：成功 episode 数 / 总 episode 数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_UNAVAILABLE = {
    "successEpisodes": None,
    "totalEpisodes": None,
    "display": "-/-",
    "available": False,
    "reason": "缺少 successEpisodes 或 totalEpisodes",
}


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


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return None
        value = data.get(key)
        if value is not None:
            return value
    return None


def _extract_episode_items(per_episode: Any) -> list[dict[str, Any]]:
    if isinstance(per_episode, list):
        return [item for item in per_episode if isinstance(item, dict)]
    if not isinstance(per_episode, dict):
        return []
    for key in ("episodes", "results", "items"):
        episodes = per_episode.get(key)
        if isinstance(episodes, list) and episodes:
            return [item for item in episodes if isinstance(item, dict)]
    return []


def _episode_is_success(item: dict[str, Any]) -> bool:
    for key in ("success", "finalSuccess", "final_success", "episodeSuccess"):
        value = item.get(key)
        if isinstance(value, bool):
            return value
    return False


def _count_success_from_episodes(items: list[dict[str, Any]]) -> int:
    return sum(1 for item in items if _episode_is_success(item))


def _aggregate_success_episodes(aggregate: dict[str, Any]) -> Optional[int]:
    summary = aggregate.get("summary") if isinstance(aggregate.get("summary"), dict) else {}
    return _pick_int(
        _nested_get(aggregate, "successEpisodes"),
        _nested_get(aggregate, "successfulEpisodes"),
        _nested_get(aggregate, "successCount"),
        _nested_get(aggregate, "success_episodes"),
        _nested_get(summary, "successEpisodes"),
        _nested_get(summary, "successfulEpisodes"),
        _nested_get(summary, "successCount"),
    )


def _aggregate_total_episodes(aggregate: dict[str, Any]) -> Optional[int]:
    summary = aggregate.get("summary") if isinstance(aggregate.get("summary"), dict) else {}
    return _pick_int(
        _nested_get(aggregate, "totalEpisodes"),
        _nested_get(aggregate, "requestedEpisodes"),
        _nested_get(aggregate, "total_episodes"),
        _nested_get(aggregate, "requested_episodes"),
        _nested_get(aggregate, "episodeCount"),
        _nested_get(summary, "totalEpisodes"),
        _nested_get(summary, "requestedEpisodes"),
    )


def _summary_success_episodes(summary: dict[str, Any]) -> Optional[int]:
    nested = summary.get("successStats") if isinstance(summary.get("successStats"), dict) else {}
    return _pick_int(
        _nested_get(nested, "successEpisodes"),
        _nested_get(summary, "successEpisodes"),
        _nested_get(summary, "successfulEpisodes"),
        _nested_get(summary, "successCount"),
        _nested_get(summary, "success_episodes"),
    )


def _summary_total_episodes(summary: dict[str, Any]) -> Optional[int]:
    nested = summary.get("successStats") if isinstance(summary.get("successStats"), dict) else {}
    return _pick_int(
        _nested_get(nested, "totalEpisodes"),
        _nested_get(summary, "totalEpisodes"),
        _nested_get(summary, "requestedEpisodes"),
        _nested_get(summary, "total_episodes"),
        _nested_get(summary, "requested_episodes"),
    )


def _context_total_episodes(context: dict[str, Any]) -> Optional[int]:
    return _pick_int(
        context.get("numEpisodes"),
        context.get("episodes"),
        context.get("episodeCount"),
        context.get("totalEpisodes"),
        context.get("requestedEpisodes"),
    )


def _status_total_episodes(status: dict[str, Any]) -> Optional[int]:
    return _pick_int(
        status.get("totalEpisodes"),
        status.get("requestedEpisodes"),
        status.get("episodes"),
    )


def _load_runtime_payloads(
    eval_job_id: str,
    *,
    runtime_path: Optional[str] = None,
) -> tuple[Optional[Path], dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    from app.services.evaluation.evaluation_runtime_health import resolve_evaluation_job_root

    job_root = resolve_evaluation_job_root(eval_job_id, runtime_path)
    if job_root is None:
        return None, {}, {}, None, {}

    aggregate = _read_json(job_root / "results" / "aggregate_result.json")
    aggregate_dict = aggregate if isinstance(aggregate, dict) else {}

    per_episode = _read_json(job_root / "results" / "per_episode_results.json")

    status: dict[str, Any] = {}
    if eval_job_id.startswith("ct_eval_"):
        for rel in ("live/status.json", "status.json"):
            data = _read_json(job_root / rel)
            if isinstance(data, dict) and data:
                status = data
                break
    else:
        for rel in ("status.json", "metadata/status.json", "live/status.json"):
            data = _read_json(job_root / rel)
            if isinstance(data, dict) and data:
                status = data
                break

    context: dict[str, Any] = {}
    for rel in (
        "metadata/evaluation_request.json",
        "metadata/evaluation_context.json",
    ):
        data = _read_json(job_root / rel)
        if not isinstance(data, dict):
            continue
        nested = data.get("evaluationRequest")
        context = nested if isinstance(nested, dict) else data
        if context:
            break

    return job_root, aggregate_dict, status, per_episode, context


def _build_result(
    *,
    success_episodes: Optional[int],
    total_episodes: Optional[int],
    source: str,
) -> dict[str, Any]:
    if success_episodes is None or total_episodes is None or total_episodes <= 0:
        result = dict(_UNAVAILABLE)
        if success_episodes is None and total_episodes is None:
            result["reason"] = "缺少 successEpisodes 或 totalEpisodes"
        elif success_episodes is None:
            result["reason"] = "缺少 successEpisodes"
        else:
            result["reason"] = "缺少 totalEpisodes"
        return result

    success_episodes = max(0, int(success_episodes))
    total_episodes = max(1, int(total_episodes))
    if success_episodes > total_episodes:
        success_episodes = total_episodes

    return {
        "successEpisodes": success_episodes,
        "totalEpisodes": total_episodes,
        "display": f"{success_episodes}/{total_episodes}",
        "available": True,
        "source": source,
    }


def resolve_success_stats(
    eval_job_id: str,
    *,
    summary_json: Optional[dict[str, Any]] = None,
    aggregate_result: Optional[dict[str, Any]] = None,
    status_json: Optional[dict[str, Any]] = None,
    context_json: Optional[dict[str, Any]] = None,
    per_episode_results: Any = None,
    runtime_path: Optional[str] = None,
    metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """解析评测任务成功率统计。"""
    candidate = (eval_job_id or "").strip()
    summary = dict(summary_json or {})
    aggregate = dict(aggregate_result or {})
    status = dict(status_json or {})
    context = dict(context_json or {})
    metrics_data = dict(metrics or {})

    if not aggregate or per_episode_results is None or not status or not context:
        loaded_root, loaded_aggregate, loaded_status, loaded_per_episode, loaded_context = _load_runtime_payloads(
            candidate,
            runtime_path=runtime_path,
        )
        if not aggregate and loaded_aggregate:
            aggregate = loaded_aggregate
        if per_episode_results is None and loaded_per_episode is not None:
            per_episode_results = loaded_per_episode
        if not status and loaded_status:
            status = loaded_status
        if not context and loaded_context:
            context = loaded_context
        _ = loaded_root

    cached = summary.get("successStats")
    if isinstance(cached, dict) and cached.get("available") and cached.get("display"):
        return {
            "successEpisodes": cached.get("successEpisodes"),
            "totalEpisodes": cached.get("totalEpisodes"),
            "display": str(cached.get("display")),
            "available": True,
            "source": str(cached.get("source") or "summary_json.successStats"),
        }

    episode_items = _extract_episode_items(per_episode_results)
    success_episodes: Optional[int] = None
    total_episodes: Optional[int] = None
    source = ""

    if episode_items:
        success_episodes = _count_success_from_episodes(episode_items)
        source = "per_episode_results.json"

    if not episode_items:
        per_episode_inline = aggregate.get("perEpisode")
        if isinstance(per_episode_inline, list) and per_episode_inline:
            episode_items = [item for item in per_episode_inline if isinstance(item, dict)]
            if episode_items:
                success_episodes = _count_success_from_episodes(episode_items)
                source = "aggregate_result.perEpisode"

    if success_episodes is None:
        value = _aggregate_success_episodes(aggregate)
        if value is not None:
            success_episodes = value
            source = "aggregate_result.json"

    if success_episodes is None:
        value = _summary_success_episodes(summary)
        if value is not None:
            success_episodes = value
            source = "eval_metric_summary.summary_json"

    if success_episodes is None:
        value = _pick_int(
            metrics_data.get("successEpisodes"),
            metrics_data.get("successfulEpisodes"),
            metrics_data.get("successCount"),
        )
        if value is not None:
            success_episodes = value
            source = "metrics.successEpisodes"

    if success_episodes is None:
        nested = metrics_data.get("aggregateResult")
        if isinstance(nested, dict):
            value = _aggregate_success_episodes(nested)
            if value is not None:
                success_episodes = value
                source = "aggregateResult.successEpisodes"

    if success_episodes is None:
        status_success = _pick_int(status.get("successfulEpisodes"), status.get("successEpisodes"))
        status_norm = str(status.get("status") or metrics_data.get("status") or "").strip().lower()
        has_progress_signal = _pick_int(
            status.get("completedEpisodes"),
            metrics_data.get("completedEpisodes"),
            status.get("episode"),
            status.get("currentEpisode"),
        )
        if status_success is not None and (
            status_success > 0
            or has_progress_signal
            or status_norm in {"completed", "failed", "canceled", "cancelled", "stale"}
        ):
            success_episodes = status_success
            source = "status.json.successfulEpisodes"

    if total_episodes is None:
        total_episodes = _aggregate_total_episodes(aggregate)
        if total_episodes is not None:
            source = source or "aggregate_result.json.totalEpisodes"

    if total_episodes is None:
        total_episodes = _summary_total_episodes(summary)
        if total_episodes is not None:
            source = source or "eval_metric_summary.summary_json.totalEpisodes"

    if total_episodes is None:
        total_episodes = _context_total_episodes(context)
        if total_episodes is not None:
            source = source or "evaluation_context.episodes"

    if total_episodes is None:
        total_episodes = _status_total_episodes(status)
        if total_episodes is not None:
            source = source or "status.json.totalEpisodes"

    if total_episodes is None:
        metrics_total = _pick_int(
            metrics_data.get("totalEpisodes"),
            metrics_data.get("requestedEpisodes"),
            metrics_data.get("numEpisodes"),
        )
        if metrics_total is not None:
            total_episodes = metrics_total
            source = source or "metrics.totalEpisodes"

    if total_episodes is None and episode_items:
        total_episodes = len(episode_items)
        source = source or "per_episode_results.json.length"

    return _build_result(
        success_episodes=success_episodes,
        total_episodes=total_episodes,
        source=source or "unknown",
    )
