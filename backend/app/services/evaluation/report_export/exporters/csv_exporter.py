from __future__ import annotations

import csv
import io
import zipfile
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


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    path.write_text(buffer.getvalue(), encoding="utf-8-sig")


def export_csv_zip(data: dict[str, Any], out_dir: Path, options: ExportOptions) -> Path:
    zip_path = out_dir / "report_csv.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if section_enabled(options, "basicInfo"):
            basic = data.get("basicInfo") or {}
            summary_path = out_dir / "summary.csv"
            _write_csv(
                summary_path,
                ["字段", "值"],
                [
                    ["任务名称", dash(basic.get("taskName"))],
                    ["任务 ID", dash(basic.get("jobId"))],
                    ["评测类型", dash(basic.get("evaluationTypeLabel"))],
                    ["关联任务", dash(basic.get("associatedTaskName"))],
                    ["仿真平台", dash(basic.get("simulationPlatform"))],
                    ["状态", dash(basic.get("statusLabel"))],
                    ["评测对象", dash(basic.get("evaluationObjectLabel"))],
                    ["模型资产", dash(basic.get("modelAssetName"))],
                    ["机器人", dash(basic.get("robotType"))],
                    ["创建时间", dash(basic.get("createdAt"))],
                    ["完成时间", dash(basic.get("finishedAt"))],
                ],
            )
            archive.write(summary_path, arcname="csv/summary.csv")

        if section_enabled(options, "metricResults"):
            metrics_path = out_dir / "metrics.csv"
            _write_csv(
                metrics_path,
                ["指标", "值", "单位", "状态", "来源", "说明"],
                [
                    [
                        dash(entry.get("displayName") or entry.get("metricId")),
                        metric_display_value(entry),
                        dash(entry.get("unit")),
                        metric_status_label(entry, include_reason=options.include_unavailable_metric_reasons),
                        dash(entry.get("source")),
                        dash(entry.get("reason") if not entry.get("available") else entry.get("description")),
                    ]
                    for entry in ordered_metric_rows(data)
                ],
            )
            archive.write(metrics_path, arcname="csv/metrics.csv")

        if section_enabled(options, "episodeResults"):
            episodes_path = out_dir / "episodes.csv"
            _write_csv(
                episodes_path,
                EPISODE_TABLE_HEADERS,
                [episode_table_row(row) for row in data.get("episodeResults") or []],
            )
            archive.write(episodes_path, arcname="csv/episodes.csv")

        run_metrics_path = out_dir / "run_metrics.csv"
        run_metrics = data.get("runMetrics") or {}
        _write_csv(
            run_metrics_path,
            ["键", "值"],
            [[key, dash(value)] for key, value in sorted(run_metrics.items())],
        )
        archive.write(run_metrics_path, arcname="csv/run_metrics.csv")

        if section_enabled(options, "videoInfo"):
            videos_path = out_dir / "videos.csv"
            _write_csv(
                videos_path,
                VIDEO_TABLE_HEADERS,
                [video_table_row(row) for row in data.get("videoInfo") or []],
            )
            archive.write(videos_path, arcname="csv/videos.csv")

        if section_enabled(options, "runtimeFiles"):
            runtime_path = out_dir / "runtime_files.csv"
            _write_csv(
                runtime_path,
                ["文件名", "相对路径", "是否存在", "大小", "说明"],
                [
                    [
                        dash(row.get("filename")),
                        dash(row.get("relativePath")),
                        "是" if row.get("exists") else "否",
                        dash(row.get("sizeLabel")),
                        dash(row.get("description")),
                    ]
                    for row in data.get("runtimeFiles") or []
                ],
            )
            archive.write(runtime_path, arcname="csv/runtime_files.csv")

        if section_enabled(options, "diagnostics"):
            diag_path = out_dir / "diagnostics.csv"
            diag = data.get("diagnostics") or {}
            _write_csv(
                diag_path,
                ["字段", "值"],
                [
                    ["摘要", dash(diag.get("summary"))],
                    ["失败原因", dash(diag.get("failureReason"))],
                    ["异常信息", dash(diag.get("errorMessage"))],
                    ["是否超时", "是" if diag.get("timedOut") else "否"],
                    ["进程异常退出", "是" if diag.get("abnormalExit") else "否"],
                    ["缺失文件", ", ".join(diag.get("missingFiles") or []) or "-"],
                    ["日志摘要", dash(diag.get("logTail"))],
                ],
            )
            archive.write(diag_path, arcname="csv/diagnostics.csv")

    return zip_path
