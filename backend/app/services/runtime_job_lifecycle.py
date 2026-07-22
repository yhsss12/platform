from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def is_job_deleted(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    return payload.get("deleted") is True or payload.get("lifecycleStatus") == "deleted"


def mark_job_deleted(status_path: Path) -> dict[str, Any]:
    payload = read_json_dict(status_path)
    now = datetime.now(timezone.utc).isoformat()
    payload["deleted"] = True
    payload["deletedAt"] = now
    payload["lifecycleStatus"] = "deleted"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
