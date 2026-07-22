from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.evaluation.report_export.export_options import ExportOptions


def export_json(data: dict[str, Any], out_dir: Path, options: ExportOptions) -> Path:
    path = out_dir / "report.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
