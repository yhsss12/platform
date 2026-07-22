from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from app.services.evaluation.report_export.export_options import ExportOptions
from app.services.evaluation.report_export.exporters.csv_exporter import export_csv_zip
from app.services.evaluation.report_export.exporters.docx_exporter import export_docx
from app.services.evaluation.report_export.exporters.json_exporter import export_json
from app.services.evaluation.report_export.exporters.latex_exporter import export_latex
from app.services.evaluation.report_export.exporters.markdown_exporter import export_markdown
from app.services.evaluation.report_export.exporters.pdf_exporter import export_pdf
from app.services.evaluation.report_export.exporters.xlsx_exporter import export_xlsx


def _safe_export(label: str, exporter, data: dict[str, Any], out_dir: Path, options: ExportOptions) -> tuple[str, Path | None, str | None]:
    try:
        return label, exporter(data, out_dir, options), None
    except RuntimeError as exc:
        return label, None, str(exc)


def export_zip_bundle(data: dict[str, Any], out_dir: Path, options: ExportOptions) -> Path:
    job_id = str((data.get("reportMeta") or {}).get("jobId") or "evaluation")
    bundle_path = out_dir / f"evaluation_report_{job_id}.zip"
    generated: list[tuple[str, Path]] = []
    skipped: list[str] = []
    export_errors: dict[str, str] = {}

    for label, exporter in (
        ("pdf", export_pdf),
        ("docx", export_docx),
        ("json", export_json),
        ("markdown", export_markdown),
        ("xlsx", export_xlsx),
        ("latex", export_latex),
    ):
        name, path, error = _safe_export(label, exporter, data, out_dir, options)
        if path is not None:
            generated.append((name, path))
        elif error:
            skipped.append(f"{name}: {error}")
            export_errors[name] = error

    csv_path, csv_error = None, None
    name, csv_path, csv_error = _safe_export("csv", export_csv_zip, data, out_dir, options)
    if csv_error:
        skipped.append(f"csv: {csv_error}")
        export_errors["csv"] = csv_error

    readme_lines = [
        "# Evaluation Report Bundle",
        "",
        f"Job ID: {job_id}",
        "",
        "## Included files",
    ]
    for _, path in generated:
        readme_lines.append(f"- {path.name}")
    if csv_path is not None:
        readme_lines.append("- csv/ (inside report_csv.zip)")
    if skipped:
        readme_lines.extend(["", "## Skipped formats", *[f"- {item}" for item in skipped]])

    runtime_index_path = out_dir / "runtime_index.json"
    runtime_index_path.write_text(
        json.dumps(
            {
                "runtimeFiles": data.get("runtimeFiles") or [],
                "rawSources": data.get("rawSources") or {},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    export_errors_path = out_dir / "export_errors.json"
    if export_errors:
        export_errors_path.write_text(
            json.dumps({"errors": export_errors, "skipped": skipped}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for _, path in generated:
            archive.write(path, arcname=path.name)
        if csv_path is not None:
            archive.write(csv_path, arcname="report_csv.zip")
        archive.write(runtime_index_path, arcname="runtime_index.json")
        archive.write(readme_path, arcname="README.md")
        if export_errors and export_errors_path.is_file():
            archive.write(export_errors_path, arcname="export_errors.json")

    return bundle_path
