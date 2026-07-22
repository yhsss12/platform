"""Runtime evaluation metrics from step_metrics summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_INTEGRATIONS_COMMON = _PROJECT_ROOT / "integrations" / "common"
if str(_INTEGRATIONS_COMMON) not in sys.path:
    sys.path.insert(0, str(_INTEGRATIONS_COMMON))

from step_metrics.step_metric_recorder import (  # noqa: E402
    aggregate_run_metrics_from_summaries,
    load_episode_summaries,
)

from app.services.evaluation.sim_time_metrics import enrich_run_metrics_sim_time

RUNTIME_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "meanSteps": {
        "metricId": "metric_runtime_mean_steps_v1",
        "label": "平均步数",
        "unit": "steps",
        "sourceKey": "meanSteps",
    },
    "maxSteps": {
        "metricId": "metric_runtime_max_steps_v1",
        "label": "最大步数",
        "unit": "steps",
        "sourceKey": "maxSteps",
    },
    "videoFps": {
        "metricId": "metric_runtime_video_fps_v1",
        "label": "视频帧率",
        "unit": "fps",
        "sourceKey": "videoFps",
    },
    "controlFrequencyHz": {
        "metricId": "metric_runtime_control_frequency_v1",
        "label": "控制频率",
        "unit": "Hz",
        "sourceKey": "controlFrequencyHz",
    },
    "meanSimTimeSec": {
        "metricId": "metric_runtime_mean_sim_time_sec_v1",
        "label": "平均仿真时长",
        "unit": "s",
        "sourceKey": "meanSimTimeSec",
        "description": "每轮 episode 的仿真时间，按 stepCount / controlFrequencyHz 计算，不等同于 wall time 或视频播放时长。",
    },
    "maxActionNorm": {
        "metricId": "metric_runtime_max_action_norm_v1",
        "label": "最大动作范数",
        "unit": "",
        "sourceKey": "maxActionNorm",
    },
    "smoothnessScore": {
        "metricId": "metric_runtime_smoothness_v1",
        "label": "动作平稳性",
        "unit": "",
        "sourceKey": "smoothnessScore",
        "description": (
            "基于相邻 step 的 action 向量变化量（L2 范数）计算：smoothnessScore = 1/(1+meanActionDelta)。"
            "数值越接近 1 表示 action 输出越平稳；不是末端轨迹或关节动力学平稳性。"
        ),
    },
    "eePathLength": {
        "metricId": "metric_runtime_ee_path_length_v1",
        "label": "末端轨迹长度",
        "unit": "m",
        "requiresFields": ["eePosition"],
    },
    "pathEfficiency": {
        "metricId": "metric_runtime_path_efficiency_v1",
        "label": "轨迹效率",
        "unit": "",
        "requiresFields": ["eePosition"],
    },
    "meanJointSpeed": {
        "metricId": "metric_runtime_mean_joint_speed_v1",
        "label": "平均关节速度",
        "unit": "rad/s",
        "requiresFields": ["qvel"],
    },
    "maxJointSpeed": {
        "metricId": "metric_runtime_max_joint_speed_v1",
        "label": "最大关节速度",
        "unit": "rad/s",
        "requiresFields": ["qvel"],
    },
    "meanJointAcceleration": {
        "metricId": "metric_runtime_mean_joint_acceleration_v1",
        "label": "平均关节加速度",
        "unit": "rad/s²",
        "requiresFields": ["qvel"],
    },
    "maxJointAcceleration": {
        "metricId": "metric_runtime_max_joint_acceleration_v1",
        "label": "最大关节加速度",
        "unit": "rad/s²",
        "requiresFields": ["qvel"],
    },
}

RUNTIME_METRIC_ID_ALIASES: dict[str, str] = {
    "metric_runtime_mean_runtime_sec_v1": "metric_runtime_mean_sim_time_sec_v1",
}

RUNTIME_METRIC_ID_TO_SPEC_KEY: dict[str, str] = {
    str(spec["metricId"]): key for key, spec in RUNTIME_METRIC_SPECS.items()
}
for _alias_id, _canonical_id in RUNTIME_METRIC_ID_ALIASES.items():
    if _canonical_id in RUNTIME_METRIC_ID_TO_SPEC_KEY:
        RUNTIME_METRIC_ID_TO_SPEC_KEY[_alias_id] = RUNTIME_METRIC_ID_TO_SPEC_KEY[_canonical_id]


def _metric_entry(
    *,
    value: Any = None,
    unit: str = "",
    available: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "available": available,
        "reason": reason,
    }


def _load_run_metrics(runtime_path: Path, aggregate: dict[str, Any] | None = None) -> dict[str, Any]:
    run_metrics = {}
    if isinstance(aggregate, dict):
        nested = aggregate.get("runMetrics")
        if isinstance(nested, dict):
            run_metrics = dict(nested)
    if run_metrics:
        return enrich_run_metrics_sim_time(run_metrics)

    summaries = load_episode_summaries(runtime_path / "results" / "step_metrics")
    if summaries:
        return enrich_run_metrics_sim_time(aggregate_run_metrics_from_summaries(summaries))
    return {}


def compute_runtime_metric_values(
    job_id: str,
    runtime_path: Path | str,
    *,
    aggregate: dict[str, Any] | None = None,
    task_type: str | None = None,
) -> dict[str, dict[str, Any]]:
    del job_id
    root = Path(runtime_path)
    run_metrics = _load_run_metrics(root, aggregate)
    if not run_metrics and task_type == "dual_arm_cable_manipulation":
        from app.services.evaluation.dual_arm_runtime_metrics import compute_dual_arm_run_metrics

        run_metrics = compute_dual_arm_run_metrics(root, aggregate)
    has_step_metrics = bool(run_metrics)

    result: dict[str, dict[str, Any]] = {}
    for key, spec in RUNTIME_METRIC_SPECS.items():
        unit = str(spec.get("unit") or "")
        required = spec.get("requiresFields")
        if required:
            result[key] = _metric_entry(
                unit=unit,
                available=False,
                reason=f"缺少 step_metrics.{','.join(required)}",
            )
            continue

        source_key = str(spec.get("sourceKey") or key)
        value = run_metrics.get(source_key)
        if key == "meanSimTimeSec" and value is None:
            mean_steps = run_metrics.get("meanSteps")
            control_hz = run_metrics.get("controlFrequencyHz")
            if isinstance(mean_steps, (int, float)) and isinstance(control_hz, (int, float)) and control_hz > 0:
                value = round(float(mean_steps) / float(control_hz), 4)
        if value is None and not has_step_metrics:
            result[key] = _metric_entry(
                unit=unit,
                available=False,
                reason="缺少 step_metrics summary",
            )
        elif value is None:
            result[key] = _metric_entry(
                unit=unit,
                available=False,
                reason=f"runMetrics.{source_key} 未生成",
            )
        else:
            result[key] = _metric_entry(value=value, unit=unit, available=True)

    return result


def attach_runtime_metrics_to_aggregate(
    aggregate: dict[str, Any],
    runtime_path: Path | str,
    *,
    task_type: str | None = None,
) -> dict[str, Any]:
    merged = dict(aggregate)
    values = compute_runtime_metric_values("", Path(runtime_path), aggregate=merged, task_type=task_type)
    runtime_block = {
        key: {
            "metricId": RUNTIME_METRIC_SPECS[key]["metricId"],
            "displayName": RUNTIME_METRIC_SPECS[key]["label"],
            **entry,
        }
        for key, entry in values.items()
    }
    merged["runtimeMetrics"] = runtime_block
    run_metrics = _load_run_metrics(Path(runtime_path), merged)
    if run_metrics:
        merged["runMetrics"] = run_metrics
        metrics_block = merged.get("metrics")
        if not isinstance(metrics_block, dict):
            metrics_block = {}
            merged["metrics"] = metrics_block
        metrics_block.update(run_metrics)
    return merged


def read_aggregate_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
