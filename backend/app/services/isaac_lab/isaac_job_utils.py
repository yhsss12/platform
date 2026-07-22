"""Isaac Lab job 通用工具（smoke / replay 共用）。"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.isaac_lab.job_paths import isaac_job_status_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_isaac_job_id(prefix: str) -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{suffix}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def finalize_status(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    existing = read_json(isaac_job_status_path(job_id))
    merged = {**existing, **dict(payload)}
    merged["updatedAt"] = utc_now_iso()
    write_json(isaac_job_status_path(job_id), merged)
    return merged
