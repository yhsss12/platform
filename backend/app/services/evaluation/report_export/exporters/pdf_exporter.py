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

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def _register_cjk_font() -> str | None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for candidate in _CJK_FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            try:
                pdfmetrics.registerFont(TTFont("ReportCJK", str(path)))
                return "ReportCJK"
            except Exception:
                continue
    return None


def export_pdf(data: dict[str, Any], out_dir: Path, options: ExportOptions) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("导出 PDF 需要安装 reportlab：pip install reportlab") from exc

    path = out_dir / "report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()
    font_name = _register_cjk_font() or "Helvetica"
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=18,
        leading=22,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=16,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10,
        leading=14,
    )

    story: list[Any] = [Paragraph("评测报告", title_style), Spacer(1, 8)]
    meta = data.get("reportMeta") or {}
    story.append(Paragraph(f"任务 ID：{dash(meta.get('jobId'))}", body_style))
    story.append(Paragraph(f"生成时间：{dash(meta.get('generatedAt'))}", body_style))
    if data.get("legacyNotice"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(str(data["legacyNotice"]), body_style))

    def add_section(title: str, rows: list[list[str]]) -> None:
        story.append(Spacer(1, 10))
        story.append(Paragraph(title, heading_style))
        table = Table(rows, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)

    if section_enabled(options, "basicInfo"):
        basic = data.get("basicInfo") or {}
        add_section(
            "1. 基础信息",
            [
                ["字段", "值"],
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

    if section_enabled(options, "evaluationConfig"):
        cfg = data.get("evaluationConfig") or {}
        add_section(
            "2. 评测配置",
            [
                ["字段", "值"],
                ["Episodes", dash(cfg.get("episodes"))],
                ["Horizon", dash(cfg.get("horizon"))],
                ["Seed", dash(cfg.get("seed"))],
                ["录制视频", dash(cfg.get("recordVideo"))],
                ["任务类型", dash(cfg.get("taskType"))],
                ["模型资产", dash(cfg.get("modelAssetName"))],
            ],
        )

    if section_enabled(options, "metricResults"):
        metric_rows = [["指标", "值", "单位", "状态", "来源", "说明"]]
        for entry in ordered_metric_rows(data):
            metric_rows.append(
                [
                    dash(entry.get("displayName") or entry.get("metricId")),
                    metric_display_value(entry),
                    dash(entry.get("unit")),
                    metric_status_label(entry, include_reason=options.include_unavailable_metric_reasons),
                    dash(entry.get("source")),
                    dash(entry.get("reason") if not entry.get("available") else entry.get("description")),
                ]
            )
        add_section("3. 指标结果", metric_rows)

    if section_enabled(options, "episodeResults"):
        episode_rows = [EPISODE_TABLE_HEADERS]
        for row in data.get("episodeResults") or []:
            episode_rows.append(episode_table_row(row))
        add_section("4. Episode 明细", episode_rows)

    if section_enabled(options, "videoInfo"):
        video_rows = [VIDEO_TABLE_HEADERS]
        for row in data.get("videoInfo") or []:
            video_rows.append(video_table_row(row))
        add_section("5. 视频回放信息", video_rows)

    if section_enabled(options, "diagnostics"):
        diag = data.get("diagnostics") or {}
        add_section(
            "6. 失败诊断",
            [
                ["字段", "值"],
                ["摘要", dash(diag.get("summary"))],
                ["失败原因", dash(diag.get("failureReason"))],
                ["异常信息", dash(diag.get("errorMessage"))],
                ["是否超时", "是" if diag.get("timedOut") else "否"],
                ["进程异常退出", "是" if diag.get("abnormalExit") else "否"],
            ],
        )

    if section_enabled(options, "runtimeFiles"):
        file_rows = [["文件名", "相对路径", "是否存在", "大小"]]
        for row in data.get("runtimeFiles") or []:
            file_rows.append(
                [
                    dash(row.get("filename")),
                    dash(row.get("relativePath")),
                    "是" if row.get("exists") else "否",
                    dash(row.get("sizeLabel")),
                ]
            )
        add_section("7. 原始文件索引", file_rows)

    doc.build(story)
    return path
