from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ExportFormat = Literal[
    "pdf",
    "docx",
    "json",
    "markdown",
    "xlsx",
    "csv",
    "latex",
    "zip",
]


@dataclass
class ExportOptions:
    format: ExportFormat = "json"
    template: str = "standard"
    include_basic_info: bool = True
    include_config: bool = True
    include_metrics: bool = True
    include_episodes: bool = True
    include_video_info: bool = True
    include_diagnostics: bool = True
    include_runtime_index: bool = True
    include_unavailable_metric_reasons: bool = True
    force: bool = True

    @classmethod
    def from_payload(cls, payload: dict | None) -> "ExportOptions":
        if not payload:
            return cls()
        fmt = str(payload.get("format") or "json").strip().lower()
        return cls(
            format=fmt,  # type: ignore[arg-type]
            template=str(payload.get("template") or "standard"),
            include_basic_info=bool(payload.get("includeBasicInfo", True)),
            include_config=bool(payload.get("includeConfig", True)),
            include_metrics=bool(payload.get("includeMetrics", True)),
            include_episodes=bool(payload.get("includeEpisodes", True)),
            include_video_info=bool(payload.get("includeVideoInfo", True)),
            include_diagnostics=bool(payload.get("includeDiagnostics", True)),
            include_runtime_index=bool(payload.get("includeRuntimeIndex", True)),
            include_unavailable_metric_reasons=bool(
                payload.get("includeUnavailableMetricReasons", True)
            ),
            force=bool(payload.get("force", True)),
        )
