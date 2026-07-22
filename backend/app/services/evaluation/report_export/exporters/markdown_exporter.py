from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.evaluation.report_export.export_options import ExportOptions
from app.services.evaluation.report_export.exporters._utils import (
    EPISODE_TABLE_HEADERS,
    VIDEO_TABLE_HEADERS,
    dash,
    episode_table_row,
    metric_display_value,
    metric_status_label,
    ordered_metric_rows,
    section_enabled,
    video_table_row,
)


def _kv_lines(pairs: list[tuple[str, str]]) -> list[str]:
    return [f"- **{key}**：{value}" for key, value in pairs]


def export_markdown(data: dict[str, Any], out_dir: Path, options: ExportOptions) -> Path:
    lines: list[str] = ["# 评测报告", ""]

    meta = data.get("reportMeta") or {}
    lines.extend(
        [
            f"- 任务 ID：`{dash(meta.get('jobId'))}`",
            f"- 生成时间：{dash(meta.get('generatedAt'))}",
            "",
        ]
    )
    if data.get("legacyNotice"):
        lines.extend([f"> {data['legacyNotice']}", ""])

    if section_enabled(options, "basicInfo"):
        basic = data.get("basicInfo") or {}
        lines.extend(
            [
                "## 1. 基础信息",
                "",
                *_kv_lines(
                    [
                        ("任务名称", dash(basic.get("taskName"))),
                        ("任务 ID", dash(basic.get("jobId"))),
                        ("评测类型", dash(basic.get("evaluationTypeLabel"))),
                        ("关联任务", dash(basic.get("associatedTaskName"))),
                        ("仿真平台", dash(basic.get("simulationPlatform"))),
                        ("状态", dash(basic.get("statusLabel"))),
                        ("评测对象", dash(basic.get("evaluationObjectLabel"))),
                        ("模型资产", dash(basic.get("modelAssetName"))),
                        ("机器人", dash(basic.get("robotType"))),
                        ("创建时间", dash(basic.get("createdAt"))),
                        ("完成时间", dash(basic.get("finishedAt"))),
                    ]
                ),
                "",
            ]
        )

    if section_enabled(options, "evaluationConfig"):
        cfg = data.get("evaluationConfig") or {}
        lines.extend(
            [
                "## 2. 评测配置",
                "",
                *_kv_lines(
                    [
                        ("Episodes", dash(cfg.get("episodes"))),
                        ("Horizon", dash(cfg.get("horizon"))),
                        ("Seed", dash(cfg.get("seed"))),
                        ("录制视频", dash(cfg.get("recordVideo"))),
                        ("任务类型", dash(cfg.get("taskType"))),
                        ("仿真平台", dash(cfg.get("simulationPlatform"))),
                        ("机器人类型", dash(cfg.get("robotType"))),
                        ("模型资产 ID", dash(cfg.get("modelAssetId"))),
                        ("模型资产名称", dash(cfg.get("modelAssetName"))),
                        ("策略名称", dash(cfg.get("policyName"))),
                        ("数据集", dash(cfg.get("datasetName"))),
                    ]
                ),
                "",
            ]
        )

    if section_enabled(options, "metricResults"):
        lines.extend(["## 3. 指标结果", "", "| 指标 | 值 | 单位 | 状态 | 来源 | 说明 |", "| --- | --- | --- | --- | --- | --- |"])
        for entry in ordered_metric_rows(data):
            reason = dash(entry.get("reason")) if not entry.get("available") else dash(entry.get("description"))
            lines.append(
                "| {name} | {value} | {unit} | {status} | {source} | {reason} |".format(
                    name=dash(entry.get("displayName") or entry.get("metricId")),
                    value=metric_display_value(entry),
                    unit=dash(entry.get("unit")),
                    status=metric_status_label(entry, include_reason=options.include_unavailable_metric_reasons),
                    source=dash(entry.get("source")),
                    reason=reason,
                )
            )
        lines.append("")

    if section_enabled(options, "episodeResults"):
        lines.extend(
            [
                "## 4. Episode 明细",
                "",
                "| " + " | ".join(EPISODE_TABLE_HEADERS) + " |",
                "| " + " | ".join(["---"] * len(EPISODE_TABLE_HEADERS)) + " |",
            ]
        )
        for row in data.get("episodeResults") or []:
            lines.append("| " + " | ".join(episode_table_row(row)) + " |")
        lines.append("")

    if section_enabled(options, "videoInfo"):
        lines.extend(
            [
                "## 5. 视频回放信息",
                "",
                "| " + " | ".join(VIDEO_TABLE_HEADERS) + " |",
                "| " + " | ".join(["---"] * len(VIDEO_TABLE_HEADERS)) + " |",
            ]
        )
        for row in data.get("videoInfo") or []:
            lines.append("| " + " | ".join(video_table_row(row)) + " |")
        lines.append("")

    if section_enabled(options, "diagnostics"):
        diag = data.get("diagnostics") or {}
        lines.extend(
            [
                "## 6. 失败诊断",
                "",
                *_kv_lines(
                    [
                        ("摘要", dash(diag.get("summary"))),
                        ("失败原因", dash(diag.get("failureReason"))),
                        ("异常信息", dash(diag.get("errorMessage"))),
                        ("是否超时", "是" if diag.get("timedOut") else "否"),
                        ("进程异常退出", "是" if diag.get("abnormalExit") else "否"),
                        ("缺失文件", ", ".join(diag.get("missingFiles") or []) or "-"),
                    ]
                ),
                "",
                "### 最后日志摘要",
                "",
                "```",
                dash(diag.get("logTail")),
                "```",
                "",
            ]
        )

    if section_enabled(options, "runtimeFiles"):
        lines.extend(
            [
                "## 7. 原始文件索引",
                "",
                "| 文件名 | 相对路径 | 是否存在 | 大小 | 说明 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in data.get("runtimeFiles") or []:
            lines.append(
                "| {name} | {path} | {exists} | {size} | {desc} |".format(
                    name=dash(row.get("filename")),
                    path=dash(row.get("relativePath")),
                    exists="是" if row.get("exists") else "否",
                    size=dash(row.get("sizeLabel")),
                    desc=dash(row.get("description")),
                )
            )
        lines.append("")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
