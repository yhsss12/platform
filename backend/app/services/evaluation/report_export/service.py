from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException, status

from app.services.evaluation.report_export.export_options import ExportFormat, ExportOptions
from app.services.evaluation.report_export.exporters.csv_exporter import export_csv_zip
from app.services.evaluation.report_export.exporters.docx_exporter import export_docx
from app.services.evaluation.report_export.exporters.json_exporter import export_json
from app.services.evaluation.report_export.exporters.latex_exporter import export_latex
from app.services.evaluation.report_export.exporters.markdown_exporter import export_markdown
from app.services.evaluation.report_export.exporters.pdf_exporter import export_pdf
from app.services.evaluation.report_export.exporters.xlsx_exporter import export_xlsx
from app.services.evaluation.report_export.exporters.zip_exporter import export_zip_bundle
from app.services.evaluation.report_export.report_data import REPORT_OUTPUT_ROOT, build_evaluation_report_data

logger = logging.getLogger(__name__)

FORMAT_SPECS: dict[ExportFormat, tuple[str, str]] = {
    "json": ("application/json", "report.json"),
    "markdown": ("text/markdown; charset=utf-8", "report.md"),
    "pdf": ("application/pdf", "report.pdf"),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "report.docx",
    ),
    "xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "report.xlsx",
    ),
    "csv": ("application/zip", "report_csv.zip"),
    "latex": ("application/x-tex", "report.tex"),
    "zip": ("application/zip", "evaluation_report_{job_id}.zip"),
}

EXPORTERS = {
    "json": export_json,
    "markdown": export_markdown,
    "pdf": export_pdf,
    "docx": export_docx,
    "xlsx": export_xlsx,
    "csv": export_csv_zip,
    "latex": export_latex,
    "zip": export_zip_bundle,
}


def export_evaluation_report(job_id: str, options: ExportOptions) -> tuple[Path, str, str]:
    fmt = options.format
    if fmt == "html":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "ok": False,
                "error": "不支持 HTML 导出",
            },
        )
    if fmt not in EXPORTERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported export format: {fmt}",
        )

    data = build_evaluation_report_data(job_id)
    out_dir = REPORT_OUTPUT_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "zip":
            output_path = export_zip_bundle(data, out_dir, options)
        else:
            output_path = EXPORTERS[fmt](data, out_dir, options)
    except RuntimeError as exc:
        logger.warning("evaluation report export failed job=%s format=%s error=%s", job_id, fmt, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("evaluation report export crashed job=%s format=%s", job_id, fmt)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出失败: {exc}",
        ) from exc

    media_type, filename_template = FORMAT_SPECS[fmt]
    filename = filename_template.format(job_id=job_id)
    if fmt == "zip":
        filename = output_path.name
    return output_path, media_type, filename
