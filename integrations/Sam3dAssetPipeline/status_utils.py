from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_status(job_dir: Path) -> dict[str, Any]:
    return read_json(job_dir / "live" / "status.json")


def write_status(
    job_dir: Path,
    *,
    status: str,
    phase: str,
    progress: float,
    message: Optional[str] = None,
    error: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    job_dir = Path(job_dir)
    live_dir = job_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    current = read_status(job_dir)
    job_id = str(current.get("jobId") or read_json(job_dir / "job.json").get("jobId") or job_dir.name)

    payload: dict[str, Any] = {
        "jobId": job_id,
        "status": status,
        "phase": phase,
        "progress": float(progress),
        "message": message,
        "updatedAt": utc_now_iso(),
        "error": error,
        "extra": extra or {},
    }
    write_json_atomic(live_dir / "status.json", payload)


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def merge_job_json(job_dir: Path, patch: dict[str, Any]) -> None:
    path = Path(job_dir) / "job.json"
    data = read_json(path)
    data.update(patch)
    data["updatedAt"] = utc_now_iso()
    write_json_atomic(path, data)
