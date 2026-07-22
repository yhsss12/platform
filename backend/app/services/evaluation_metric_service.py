from __future__ import annotations

from typing import Any, Optional

ISAAC_STACK_DEFAULT_METRIC_IDS: tuple[str, ...] = (
    "isaac_stack_success_rate_v1",
    "isaac_stack_mean_reward_v1",
    "isaac_stack_mean_episode_length_v1",
    "isaac_stack_failure_count_v1",
    "isaac_stack_timeout_rate_v1",
)

UNIVERSAL_SUCCESS_RATE_METRIC_ID = "metric_success_rate_v1"


def _read_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def compute_timeout_rate(per_episode: dict[str, Any] | None) -> Optional[float]:
    episodes = (per_episode or {}).get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return None
    timeout_count = sum(
        1 for row in episodes if isinstance(row, dict) and row.get("failureReason") == "horizon_reached"
    )
    return timeout_count / len(episodes)


def resolve_metric_value(
    metric_meta: dict[str, Any],
    *,
    aggregate: dict[str, Any],
    per_episode: dict[str, Any] | None = None,
) -> Optional[Any]:
    if metric_meta.get("implemented") is False:
        return None

    mode = str(metric_meta.get("calculationMode") or "aggregate_field")
    metrics_block = aggregate.get("metrics") if isinstance(aggregate.get("metrics"), dict) else {}

    if mode == "aggregate_field":
        field = str(metric_meta.get("sourceField") or "")
        if not field:
            return None
        if "." in field:
            current: Any = aggregate
            for part in field.split("."):
                if not isinstance(current, dict):
                    return None
                current = current.get(part)
            if current is not None:
                return current
        if field in metrics_block and metrics_block[field] is not None:
            return metrics_block[field]
        return aggregate.get(field)

    if mode == "per_episode_failure_reason":
        field = str(metric_meta.get("sourceField") or "failureReason")
        target = str(metric_meta.get("failureReasonValue") or "horizon_reached")
        episodes = (per_episode or {}).get("episodes")
        if not isinstance(episodes, list) or not episodes:
            cached = metrics_block.get("timeoutRate")
            if cached is not None:
                return cached
            return compute_timeout_rate(per_episode)
        matched = sum(
            1 for row in episodes if isinstance(row, dict) and str(row.get(field) or "") == target
        )
        return matched / len(episodes)

    if mode == "aggregate_fields":
        source_fields = metric_meta.get("sourceFields") or []
        if not isinstance(source_fields, list):
            return None
        resolved: dict[str, Any] = {}
        for raw_path in source_fields:
            path = str(raw_path)
            value = _read_nested(aggregate, path)
            if value is not None:
                resolved[path] = value
        return resolved or None

    return None


def attach_isaac_eval_metric_metadata(aggregate: dict[str, Any], per_episode: dict[str, Any]) -> dict[str, Any]:
    timeout_rate = compute_timeout_rate(per_episode)
    metrics_block = aggregate.get("metrics")
    if not isinstance(metrics_block, dict):
        metrics_block = {}
        aggregate["metrics"] = metrics_block
    if timeout_rate is not None:
        metrics_block["timeoutRate"] = timeout_rate

    aggregate["computedMetricIds"] = list(ISAAC_STACK_DEFAULT_METRIC_IDS)
    aggregate["metricsSource"] = "task_evaluation_script"
    return aggregate
