from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.evaluation.report_export.export_options import ExportOptions
from app.services.evaluation.report_export.exporters._utils import (
    EPISODE_TABLE_HEADERS,
    dash,
    episode_table_row,
    latex_escape,
    metric_display_value,
    ordered_metric_rows,
    section_enabled,
)


def export_latex(data: dict[str, Any], out_dir: Path, options: ExportOptions) -> Path:
    lines: list[str] = [
        "% Auto-generated evaluation report template",
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\begin{document}",
        "",
        "\\section*{Evaluation Report}",
    ]
    meta = data.get("reportMeta") or {}
    lines.extend(
        [
            f"Job ID: {latex_escape(meta.get('jobId'))}\\\\",
            f"Generated At: {latex_escape(meta.get('generatedAt'))}\\\\",
            "",
        ]
    )
    if data.get("legacyNotice"):
        lines.extend([latex_escape(data["legacyNotice"]) + "\\\\", ""])

    if section_enabled(options, "metricResults"):
        lines.extend(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Evaluation Metrics of the Task}",
                "\\label{tab:evaluation_metrics}",
                "\\begin{tabular}{lccc}",
                "\\toprule",
                "Metric & Value & Unit & Source \\\\",
                "\\midrule",
            ]
        )
        for entry in ordered_metric_rows(data):
            lines.append(
                " & ".join(
                    [
                        latex_escape(entry.get("displayName") or entry.get("metricId")),
                        latex_escape(metric_display_value(entry)),
                        latex_escape(entry.get("unit")),
                        latex_escape(entry.get("source")),
                    ]
                )
                + " \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

        unavailable = [entry for entry in ordered_metric_rows(data) if not entry.get("available")]
        if unavailable and options.include_unavailable_metric_reasons:
            lines.extend(
                [
                    "\\begin{table}[htbp]",
                    "\\centering",
                    "\\caption{Unavailable Metrics}",
                    "\\label{tab:unavailable_metrics}",
                    "\\begin{tabular}{ll}",
                    "\\toprule",
                    "Metric & Reason \\\\",
                    "\\midrule",
                ]
            )
            for entry in unavailable:
                lines.append(
                    f"{latex_escape(entry.get('displayName') or entry.get('metricId'))} & {latex_escape(entry.get('reason'))} \\\\"
                )
            lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if section_enabled(options, "episodeResults"):
        col_spec = "l" * len(EPISODE_TABLE_HEADERS)
        header_row = " & ".join(latex_escape(h) for h in EPISODE_TABLE_HEADERS)
        lines.extend(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Episode Results}",
                "\\label{tab:episode_results}",
                f"\\begin{{tabular}}{{{col_spec}}}",
                "\\toprule",
                f"{header_row} \\\\",
                "\\midrule",
            ]
        )
        for row in data.get("episodeResults") or []:
            cells = episode_table_row(row)
            lines.append(" & ".join(latex_escape(cell) for cell in cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend(["\\end{document}", ""])
    path = out_dir / "report.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
