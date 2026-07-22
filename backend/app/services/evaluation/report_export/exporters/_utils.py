from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.services.evaluation.report_export.export_options import ExportOptions

_LATEX_SPECIAL = re.compile(r"([\\&%$#_{}~^])")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dash(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def format_file_size(size_bytes: int | None) -> str:
    if size_bytes is None or size_bytes < 0:
        return "-"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def latex_escape(text: Any) -> str:
    raw = dash(text)
    return _LATEX_SPECIAL.sub(r"\\\1", raw.replace("\n", " "))


def ordered_metric_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[str] = list(data.get("selectedMetricIds") or [])
    metric_results: dict[str, Any] = data.get("metricResults") or {}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metric_id in selected:
        entry = metric_results.get(metric_id)
        if not isinstance(entry, dict):
            continue
        rows.append(entry)
        seen.add(metric_id)
    for metric_id, entry in metric_results.items():
        if metric_id in seen or not isinstance(entry, dict):
            continue
        rows.append(entry)
    return rows


def metric_status_label(entry: dict[str, Any], *, include_reason: bool) -> str:
    if entry.get("available"):
        return "可计算"
    reason = entry.get("reason")
    if include_reason and reason:
        return f"不可计算（{reason}）"
    return "不可计算"


def metric_display_value(entry: dict[str, Any]) -> str:
    if entry.get("available"):
        return dash(entry.get("formattedValue") or entry.get("value"))
    return "-"


def section_enabled(options: ExportOptions, section: str) -> bool:
    mapping = {
        "basicInfo": options.include_basic_info,
        "evaluationConfig": options.include_config,
        "metricResults": options.include_metrics,
        "episodeResults": options.include_episodes,
        "videoInfo": options.include_video_info,
        "diagnostics": options.include_diagnostics,
        "runtimeFiles": options.include_runtime_index,
    }
    return bool(mapping.get(section, True))


EPISODE_TABLE_HEADERS = [
    "Episode",
    "成功/失败",
    "步数",
    "仿真时长(s)",
    "最大动作范数",
    "动作平稳性",
    "失败原因",
    "视频文件",
]


def episode_table_row(row: dict[str, Any]) -> list[str]:
    return [
        dash(row.get("episodeIndex")),
        dash(row.get("successLabel")),
        dash(row.get("stepCount")),
        dash(row.get("simTimeSec")),
        dash(row.get("maxActionNorm")),
        dash(row.get("smoothnessScore")),
        dash(row.get("failureReason")),
        dash(row.get("videoFile")),
    ]


VIDEO_TABLE_HEADERS = ["Episode", "文件名", "路径", "FPS", "分辨率", "代表视频"]


def video_table_row(row: dict[str, Any]) -> list[str]:
    return [
        dash(row.get("episodeIndex")),
        dash(row.get("videoFilename")),
        dash(row.get("videoPath")),
        dash(row.get("fps")),
        dash(row.get("resolution")),
        "是" if row.get("isRepresentative") else "否",
    ]
