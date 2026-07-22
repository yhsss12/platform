from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_live_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_root_status(job_root: Path, payload: dict[str, Any]) -> None:
    write_live_status(job_root / "status.json", payload)
