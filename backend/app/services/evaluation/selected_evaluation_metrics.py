"""Resolve user-selected evaluation metrics into structured metricResults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.evaluation.metric_policy import (
    is_deprecated_metric,
    is_report_body_metric,
)
from app.services.evaluation.runtime_metrics import (
    RUNTIME_METRIC_ID_ALIASES,
    RUNTIME_METRIC_ID_TO_SPEC_KEY,
    RUNTIME_METRIC_SPECS,
    attach_runtime_metrics_to_aggregate,
    compute_runtime_metric_values,
)

CABLE_THREADING_TASK_TYPE = "cable_threading"
DUAL_ARM_TASK_TYPE = "dual_arm_cable_manipulation"

SUCCESS_RATE_METRIC_BY_TASK: dict[str, str] = {
    CABLE_THREADING_TASK_TYPE: "metric_cable_success_rate_v1",
    DUAL_ARM_TASK_TYPE: "metric_success_rate_v1",
}

LEGACY_SUCCESS_RATE_ALIASES = frozenset(
    {
        "success_rate",
        "metric_success_rate_v1",
        "metric_cable_success_rate_v1",
        "isaac_stack_success_rate_v1",
    }
)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def normalize_selected_metric_ids(raw_ids: list[str] | None, task_type: str) -> list[str]:
    if not raw_ids:
        return []

    default_success = SUCCESS_RATE_METRIC_BY_TASK.get(task_type, "metric_success_rate_v1")
    normalized: list[str] = []
    for raw in raw_ids:
        value = str(raw).strip()
        if not value:
            continue
        if value in RUNTIME_METRIC_ID_ALIASES:
            value = RUNTIME_METRIC_ID_ALIASES[value]
        if is_deprecated_metric(value):
            continue
        if value in LEGACY_SUCCESS_RATE_ALIASES:
            normalized.append(default_success if task_type == CABLE_THREADING_TASK_TYPE else value)
            continue
        if value == "success_rate":
            normalized.append(default_success)
            continue
        normalized.append(value)
    return _dedupe_selected_metric_ids(normalized)


def _dedupe_selected_metric_ids(items: list[str]) -> list[str]:
    """Dedupe while keeping legacy metric IDs; drop alias if canonical already selected."""
    canonical_present = {RUNTIME_METRIC_ID_ALIASES.get(item, item) for item in items}
    result: list[str] = []
    seen_canonical: set[str] = set()
    for item in items:
        canonical = RUNTIME_METRIC_ID_ALIASES.get(item, item)
        if canonical in seen_canonical:
            continue
        if item in RUNTIME_METRIC_ID_ALIASES and canonical in canonical_present:
            # Prefer canonical ID when both alias and canonical are present.
            if item != canonical and canonical in items:
                continue
        seen_canonical.add(canonical)
        result.append(item)
    return result


def read_selected_metric_ids_from_job(job_root: Path, task_type: str = CABLE_THREADING_TASK_TYPE) -> list[str]:
    context_path = job_root / "metadata" / "evaluation_context.json"
    if not context_path.is_file():
        return []
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(context, dict):
        return []

    eval_request = context.get("evaluationRequest")
    eval_request = eval_request if isinstance(eval_request, dict) else {}
    raw_sources = [
        context.get("selectedMetricIds"),
        context.get("metrics"),
        eval_request.get("selectedMetricIds"),
        eval_request.get("metrics"),
        eval_request.get("selectedMetricKeys"),
    ]
    raw: list[str] = []
    for source in raw_sources:
        if isinstance(source, list):
            raw.extend(str(item) for item in source if str(item).strip())
    config_block = context.get("config")
    if isinstance(config_block, dict):
        config_metrics = config_block.get("metrics")
        if isinstance(config_metrics, list):
            raw.extend(str(item) for item in config_metrics if str(item).strip())
    return normalize_selected_metric_ids(raw, task_type)


def _pick_success_rate(aggregate: dict[str, Any]) -> float | None:
    for key in ("success_rate", "final_success_rate", "successRate", "finalSuccessRate"):
        value = aggregate.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rate = float(value)
            if rate > 1:
                rate = rate / 100.0
            return rate

    successful = aggregate.get("successful_episodes")
    if successful is None:
        successful = aggregate.get("success_episodes")
    completed = aggregate.get("completed_episodes")
    if completed is None:
        completed = aggregate.get("total_episodes")
    if completed is None:
        completed = aggregate.get("num_episodes")

    if isinstance(successful, (int, float)) and isinstance(completed, (int, float)) and completed > 0:
        return float(successful) / float(completed)
    return None


def _format_ratio(value: float) -> str:
    percent = value * 100 if value <= 1 else value
    rounded = round(percent * 10) / 10
    if rounded == int(rounded):
        return f"{int(rounded)}%"
    return f"{rounded}%"


def _format_number(value: float, unit: str = "") -> str:
    if float(value).is_integer():
        text = str(int(value))
    else:
        text = str(round(float(value) * 1000) / 1000)
    if not unit:
        return text
    if unit == "%":
        return _format_ratio(value)
    if unit == "s":
        return f"{text}s"
    if unit in {"steps", "fps", "Hz", "m", "rad/s", "rad/s²"}:
        return f"{text} {unit}"
    return f"{text}{unit}"


def _metric_result_entry(
    *,
    metric_id: str,
    display_name: str,
    value: Any = None,
    formatted_value: str = "-",
    unit: str = "",
    available: bool = False,
    reason: str | None = None,
    source: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    entry = {
        "metricId": metric_id,
        "displayName": display_name,
        "value": value if available else None,
        "formattedValue": formatted_value if available else "-",
        "unit": unit,
        "available": available,
        "reason": reason,
        "source": source,
    }
    if description:
        entry["description"] = description
    return entry


def _resolve_success_rate_metric(
    metric_id: str,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    display_name = "成功率"
    rate = _pick_success_rate(aggregate)
    if rate is None:
        return _metric_result_entry(
            metric_id=metric_id,
            display_name=display_name,
            unit="%",
            available=False,
            reason="aggregate_result 缺少 success_rate 或 episode 计数",
            source="aggregate_result.success_rate",
        )
    return _metric_result_entry(
        metric_id=metric_id,
        display_name=display_name,
        value=rate,
        formatted_value=_format_ratio(rate),
        unit="%",
        available=True,
        source="aggregate_result.success_rate",
    )


def _resolve_runtime_metric(
    metric_id: str,
    aggregate: dict[str, Any],
    runtime_values: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    spec_key = RUNTIME_METRIC_ID_TO_SPEC_KEY.get(metric_id)
    if not spec_key:
        return _metric_result_entry(
            metric_id=metric_id,
            display_name=metric_id,
            available=False,
            reason="未知 metricId",
        )

    spec = RUNTIME_METRIC_SPECS[spec_key]
    display_name = str(spec.get("label") or metric_id)
    if metric_id in RUNTIME_METRIC_ID_ALIASES:
        display_name = str(spec.get("label") or "平均仿真时长")
    description = str(spec.get("description") or "") or None
    unit = str(spec.get("unit") or "")
    runtime_entry = runtime_values.get(spec_key, {})
    available = bool(runtime_entry.get("available"))
    reason = runtime_entry.get("reason")
    source_key = str(spec.get("sourceKey") or spec_key)

    if not available:
        return _metric_result_entry(
            metric_id=metric_id,
            display_name=display_name,
            unit=unit,
            available=False,
            reason=str(reason) if reason else f"缺少 runMetrics.{source_key}",
            source=f"runMetrics.{source_key}",
            description=description,
        )

    value = runtime_entry.get("value")
    if value is None:
        return _metric_result_entry(
            metric_id=metric_id,
            display_name=display_name,
            unit=unit,
            available=False,
            reason=f"runMetrics.{source_key} 未生成",
            source=f"runMetrics.{source_key}",
            description=description,
        )

    numeric = float(value)
    return _metric_result_entry(
        metric_id=metric_id,
        display_name=display_name,
        value=numeric,
        formatted_value=_format_number(numeric, unit),
        unit=unit,
        available=True,
        source=f"runMetrics.{source_key}",
        description=description,
    )


def resolve_selected_evaluation_metric_results(
    selected_metric_ids: list[str],
    aggregate: dict[str, Any],
    runtime_path: Path | str,
    task_type: str,
    *,
    legacy_fallback: bool = True,
) -> dict[str, Any]:
    normalized = normalize_selected_metric_ids(selected_metric_ids, task_type)
    if not normalized and legacy_fallback:
        if _pick_success_rate(aggregate) is not None:
            default_success = SUCCESS_RATE_METRIC_BY_TASK.get(task_type, "metric_success_rate_v1")
            normalized = [default_success]

    runtime_values = compute_runtime_metric_values(
        "",
        Path(runtime_path),
        aggregate=aggregate,
        task_type=task_type,
    )
    default_success = SUCCESS_RATE_METRIC_BY_TASK.get(task_type, "metric_success_rate_v1")

    metric_results: dict[str, dict[str, Any]] = {}
    for metric_id in normalized:
        if is_deprecated_metric(metric_id):
            continue
        if metric_id in LEGACY_SUCCESS_RATE_ALIASES or metric_id == default_success:
            metric_results[metric_id] = _resolve_success_rate_metric(metric_id, aggregate)
            continue
        if metric_id in RUNTIME_METRIC_ID_TO_SPEC_KEY:
            metric_results[metric_id] = _resolve_runtime_metric(metric_id, aggregate, runtime_values)
            continue
        if not is_report_body_metric(metric_id):
            continue
        metric_results[metric_id] = _metric_result_entry(
            metric_id=metric_id,
            display_name=metric_id,
            available=False,
            reason="未知 metricId",
        )

    return {
        "selectedMetricIds": [mid for mid in normalized if not is_deprecated_metric(mid)],
        "metricResults": metric_results,
        "deprecatedMetricIds": [mid for mid in normalized if is_deprecated_metric(mid)],
    }


def finalize_selected_evaluation_metrics(
    aggregate: dict[str, Any],
    job_root: Path,
    selected_metric_ids: list[str] | None,
    *,
    task_type: str = CABLE_THREADING_TASK_TYPE,
    persist: bool = False,
    legacy_fallback: bool = True,
) -> dict[str, Any]:
    merged = attach_runtime_metrics_to_aggregate(dict(aggregate), job_root, task_type=task_type)
    if not selected_metric_ids:
        selected_metric_ids = read_selected_metric_ids_from_job(job_root, task_type)

    resolved = resolve_selected_evaluation_metric_results(
        selected_metric_ids or [],
        merged,
        job_root,
        task_type,
        legacy_fallback=legacy_fallback,
    )
    merged["selectedMetricIds"] = resolved["selectedMetricIds"]
    merged["metricResults"] = resolved["metricResults"]

    if persist:
        aggregate_path = job_root / "results" / "aggregate_result.json"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "aggregate": merged,
        "selectedMetricIds": resolved["selectedMetricIds"],
        "metricResults": resolved["metricResults"],
        "runMetrics": merged.get("runMetrics") if isinstance(merged.get("runMetrics"), dict) else {},
    }
