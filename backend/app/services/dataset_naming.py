"""Dataset display naming helpers for workspace manifests and list rows."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

TASK_DISPLAY_NAMES: dict[str, str] = {
    "cable_threading": "线缆穿杆",
    "dual_arm_cable_manipulation": "线缆整理",
    "block_stacking": "物块堆叠",
    "isaac_block_stacking": "物块堆叠",
    "stacking": "物块堆叠",
    "nut_assembly": "螺母装配",
}

TASK_TYPE_TO_TEMPLATE: dict[str, str] = {
    "cable_threading": "task_cable_threading_v1",
    "dual_arm_cable_manipulation": "task_dual_arm_cable_manipulation_v1",
    "block_stacking": "task_isaac_block_stacking_v1",
    "isaac_block_stacking": "task_isaac_block_stacking_v1",
    "nut_assembly": "task_nut_assembly_v1",
}

JOB_ID_TIMESTAMP_PATTERN = re.compile(
    r"(?:^|_)(?:ct_gen|dac_gen|na_gen|data_gen|isaac_import|isaac_gen|isaac_ds)_(\d{8})_(\d{6})",
    re.IGNORECASE,
)
CANONICAL_DATASET_NAME_PATTERN = re.compile(
    r"^[\u4e00-\u9fffA-Za-z0-9（）()·\-\s]+数据_\d{8}_\d{4}(?:_\d{2})?$"
)


def task_display_name(task_type: str) -> str:
    key = (task_type or "").strip()
    return TASK_DISPLAY_NAMES.get(key, key or "自定义任务")


def is_canonical_dataset_display_name(name: str) -> bool:
    candidate = (name or "").strip()
    return bool(candidate and CANONICAL_DATASET_NAME_PATTERN.fullmatch(candidate))


def _parse_created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _timestamp_from_job_id(source_job_id: str | None) -> tuple[str, str] | None:
    if not source_job_id:
        return None
    match = JOB_ID_TIMESTAMP_PATTERN.search(source_job_id)
    if not match:
        return None
    date_part, time_part = match.group(1), match.group(2)
    return date_part, time_part[:4]


def build_dataset_display_name(
    *,
    task_type: str,
    created_at: Any = None,
    source_job_id: str | None = None,
    dataset_index: int = 1,
) -> str:
    label = task_display_name(task_type)
    stamp = _timestamp_from_job_id(source_job_id)
    if stamp is None:
        dt = _parse_created_at(created_at)
        stamp = (dt.strftime("%Y%m%d"), dt.strftime("%H%M"))
    date_part, time_part = stamp
    base = f"{label}数据_{date_part}_{time_part}"
    if int(dataset_index or 1) > 1:
        return f"{base}_{int(dataset_index):02d}"
    return base


def normalize_dataset_display_name(
    *,
    task_type: str,
    display_name: str | None = None,
    name: str | None = None,
    created_at: Any = None,
    source_job_id: str | None = None,
    dataset_index: int = 1,
) -> str:
    for candidate in (display_name, name):
        value = (candidate or "").strip()
        if value and is_canonical_dataset_display_name(value):
            return value
    return build_dataset_display_name(
        task_type=task_type,
        created_at=created_at,
        source_job_id=source_job_id,
        dataset_index=dataset_index,
    )


def resolve_unique_dataset_display_name(
    *,
    task_type: str,
    created_at: Any = None,
    source_job_id: str | None = None,
    existing_names: Iterable[str] | None = None,
) -> str:
    taken = {str(item).strip() for item in (existing_names or []) if str(item).strip()}
    for index in range(1, 100):
        candidate = build_dataset_display_name(
            task_type=task_type,
            created_at=created_at,
            source_job_id=source_job_id,
            dataset_index=index,
        )
        if candidate not in taken:
            return candidate
    return build_dataset_display_name(
        task_type=task_type,
        created_at=created_at,
        source_job_id=source_job_id,
        dataset_index=99,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def persist_manifest_display_fields(
    manifest_path: Path | str,
    *,
    task_type: str,
    source_job_id: str | None = None,
    simulator_backend: str | None = None,
    dataset_format: str | None = None,
    created_at: Any = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _read_manifest(path)
    created = created_at or manifest.get("createdAt") or manifest.get("created_at")
    display_name = normalize_dataset_display_name(
        task_type=task_type,
        display_name=str(manifest.get("displayName") or ""),
        name=str(manifest.get("name") or manifest.get("datasetName") or ""),
        created_at=created,
        source_job_id=source_job_id or str(manifest.get("sourceJobId") or ""),
    )
    manifest["displayName"] = display_name
    manifest["name"] = display_name
    manifest["taskDisplayName"] = task_display_name(task_type)
    manifest["taskType"] = task_type
    manifest.setdefault("taskTemplateId", TASK_TYPE_TO_TEMPLATE.get(task_type))
    if source_job_id:
        manifest["sourceJobId"] = source_job_id
    if simulator_backend:
        manifest["simulatorBackend"] = simulator_backend
    if dataset_format:
        manifest["datasetFormat"] = dataset_format
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def apply_dataset_row_display_fields(
    row: dict[str, Any],
    *,
    task_type: str,
    manifest: dict[str, Any] | None = None,
    source_job_id: str | None = None,
) -> dict[str, Any]:
    manifest = manifest or {}
    display_name = normalize_dataset_display_name(
        task_type=task_type,
        display_name=str(row.get("displayName") or manifest.get("displayName") or ""),
        name=str(row.get("name") or manifest.get("name") or ""),
        created_at=row.get("createdAt") or manifest.get("createdAt") or manifest.get("created_at"),
        source_job_id=source_job_id or str(row.get("sourceJobId") or manifest.get("sourceJobId") or ""),
    )
    row["displayName"] = display_name
    row["name"] = display_name
    # ``datasetName`` is consumed by the training picker.  Some recovered
    # filesystem rows still contain the internal task key here (for example
    # ``nut_assembly``), even though ``name`` / ``displayName`` have already
    # been normalized.  Keep all three public display fields consistent.
    row["datasetName"] = display_name
    row["taskDisplayName"] = task_display_name(task_type)
    row.setdefault("taskType", task_type)
    row.setdefault("taskTemplateId", TASK_TYPE_TO_TEMPLATE.get(task_type))
    return row
