from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REGISTRY_ROOT = PROJECT_ROOT / "configs" / "resources"

ALLOWED_ASSET_TYPES = frozenset(
    {"robot", "end_effector", "object", "scene", "task", "metric", "policy"}
)
REQUIRED_BASE_FIELDS = ("asset_id", "asset_type", "name", "version", "status", "sim_backend", "description")

_REGISTRY_CACHE: dict[str, dict[str, Any]] = {}
_REGISTRY_INDEX: dict[str, Path] = {}
_LAST_SCAN_AT: Optional[str] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_manifest(path: Path) -> dict[str, Any]:
    from app.services.safe_file_io import safe_read_text

    text = safe_read_text(path)
    if not text:
        raise ValueError(f"manifest unreadable or empty: {path}")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    return data


def _manifest_to_resource(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    asset_id = str(data.get("asset_id") or "")
    return {
        "assetId": asset_id,
        "assetType": str(data.get("asset_type") or ""),
        "name": str(data.get("name") or ""),
        "version": str(data.get("version") or ""),
        "status": str(data.get("status") or ""),
        "simBackend": str(data.get("sim_backend") or ""),
        "description": str(data.get("description") or ""),
        "tags": list(data.get("tags") or []),
        "files": dict(data.get("files") or {}),
        "metadata": dict(data.get("metadata") or {}),
        "manifestPath": str(path.relative_to(PROJECT_ROOT)),
        "lastModifiedAt": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "taskType": data.get("task_type"),
        "requiredAssets": data.get("required_assets"),
        "metrics": list(data.get("metrics") or []),
        "runner": dict(data.get("runner") or {}),
        "defaultConfig": dict(data.get("default_config") or {}),
        "_raw": data,
    }


def _validate_manifest(data: dict[str, Any], path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for field in REQUIRED_BASE_FIELDS:
        if not data.get(field):
            errors.append(f"{path.name}: missing required field '{field}'")

    asset_type = str(data.get("asset_type") or "")
    if asset_type and asset_type not in ALLOWED_ASSET_TYPES:
        errors.append(f"{path.name}: invalid asset_type '{asset_type}'")

    files = data.get("files") or {}
    if isinstance(files, dict):
        for key, rel in files.items():
            if not rel or not str(rel).strip():
                continue
            candidate = PROJECT_ROOT / str(rel)
            if not candidate.exists():
                warnings.append(f"{path.name}: file not found [{key}]={rel}")

    return errors, warnings


def _collect_required_asset_ids(task_data: dict[str, Any]) -> list[str]:
    required = task_data.get("required_assets") or {}
    ids: list[str] = []
    if isinstance(required, dict):
        for group in ("robots", "end_effectors", "objects", "scenes", "policies"):
            for item in required.get(group) or []:
                ids.append(str(item))
    return ids


def scan_resource_registry(*, force: bool = False) -> dict[str, Any]:
    global _REGISTRY_CACHE, _REGISTRY_INDEX, _LAST_SCAN_AT

    if _REGISTRY_CACHE and not force:
        return {
            "scanned": len(_REGISTRY_INDEX),
            "valid": len(_REGISTRY_CACHE),
            "invalid": 0,
            "resourcesByType": _count_by_type(_REGISTRY_CACHE.values()),
            "errors": [],
            "warnings": [],
            "lastScanAt": _LAST_SCAN_AT,
        }

    scanned = 0
    valid = 0
    invalid = 0
    errors: list[str] = []
    warnings: list[str] = []
    cache: dict[str, dict[str, Any]] = {}
    index: dict[str, Path] = {}
    seen_ids: dict[str, str] = {}

    if not REGISTRY_ROOT.is_dir():
        errors.append(f"registry root not found: {REGISTRY_ROOT}")
        _REGISTRY_CACHE = {}
        _REGISTRY_INDEX = {}
        return {
            "scanned": 0,
            "valid": 0,
            "invalid": 0,
            "resourcesByType": {},
            "errors": errors,
            "warnings": warnings,
            "lastScanAt": _utc_now_iso(),
        }

    for path in sorted(REGISTRY_ROOT.rglob("*.yaml")):
        if not path.is_file():
            continue
        scanned += 1
        try:
            data = _read_manifest(path)
        except Exception as exc:
            invalid += 1
            errors.append(f"{path.name}: parse error: {exc}")
            continue

        file_errors, file_warnings = _validate_manifest(data, path)
        warnings.extend(file_warnings)

        asset_id = str(data.get("asset_id") or "")
        if asset_id in seen_ids:
            invalid += 1
            errors.append(f"duplicate asset_id '{asset_id}' in {path.name} and {seen_ids[asset_id]}")
            continue
        seen_ids[asset_id] = path.name

        if file_errors:
            invalid += 1
            errors.extend(file_errors)
            continue

        resource = _manifest_to_resource(path, data)
        cache[asset_id] = resource
        index[asset_id] = path
        valid += 1

    for asset_id, resource in list(cache.items()):
        if resource.get("assetType") != "task":
            continue
        raw = resource.get("_raw") or {}
        for ref_id in _collect_required_asset_ids(raw):
            if ref_id not in cache:
                warnings.append(f"{asset_id}: required asset not found: {ref_id}")
        for metric_id in raw.get("metrics") or []:
            if str(metric_id) not in cache:
                warnings.append(f"{asset_id}: metric not found: {metric_id}")

    _REGISTRY_CACHE = cache
    _REGISTRY_INDEX = index
    _LAST_SCAN_AT = _utc_now_iso()

    return {
        "scanned": scanned,
        "valid": valid,
        "invalid": invalid,
        "resourcesByType": _count_by_type(cache.values()),
        "errors": errors,
        "warnings": warnings,
        "lastScanAt": _LAST_SCAN_AT,
    }


def _count_by_type(resources: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in resources:
        t = str(item.get("assetType") or item.get("asset_type") or "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def ensure_registry_loaded() -> None:
    if not _REGISTRY_CACHE:
        scan_resource_registry(force=True)


def list_resources(
    *,
    asset_type: Optional[str] = None,
    sim_backend: Optional[str] = None,
    status: Optional[str] = None,
    task_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    ensure_registry_loaded()
    rows: list[dict[str, Any]] = []
    for resource in _REGISTRY_CACHE.values():
        if asset_type and resource.get("assetType") != asset_type:
            continue
        if sim_backend and resource.get("simBackend") != sim_backend:
            continue
        if status and resource.get("status") != status:
            continue
        if task_type and resource.get("taskType") != task_type:
            continue
        rows.append(_public_resource(resource))
    rows.sort(key=lambda r: (r.get("assetType") or "", r.get("assetId") or ""))
    return rows


def get_resource(asset_id: str) -> Optional[dict[str, Any]]:
    ensure_registry_loaded()
    resource = _REGISTRY_CACHE.get(asset_id)
    if resource is None:
        return None
    return _public_resource(resource, include_raw=True)


def list_task_configs(*, task_type: Optional[str] = None) -> list[dict[str, Any]]:
    ensure_registry_loaded()
    rows = [
        _task_config_summary(resource)
        for resource in _REGISTRY_CACHE.values()
        if resource.get("assetType") == "task"
        and (not task_type or resource.get("taskType") == task_type)
    ]
    rows.sort(key=lambda r: r.get("assetId") or "")
    return rows


def get_task_config(task_config_id: str) -> Optional[dict[str, Any]]:
    ensure_registry_loaded()
    resource = _REGISTRY_CACHE.get(task_config_id)
    if resource is None or resource.get("assetType") != "task":
        return None
    return _task_config_detail(resource)


def get_task_config_metadata_for_job(task_config_id: str) -> dict[str, Any]:
    detail = get_task_config(task_config_id)
    if not detail:
        return {}

    asset_ids: list[str] = []
    required = detail.get("requiredAssets") or {}
    if isinstance(required, dict):
        for group in ("robots", "end_effectors", "objects", "scenes", "policies"):
            for item in required.get(group) or []:
                asset_ids.append(str(item))

    metric_ids = [str(m) for m in (detail.get("metrics") or [])]
    snapshot = {
        "taskConfigId": detail.get("assetId"),
        "taskVersion": detail.get("version"),
        "simBackend": detail.get("simBackend"),
        "assetIds": asset_ids,
        "metricIds": metric_ids,
        "resourceRegistryVersion": detail.get("version"),
        "manifestSnapshot": {
            "taskConfigId": detail.get("assetId"),
            "version": detail.get("version"),
            "requiredAssets": required,
            "metrics": metric_ids,
            "runner": detail.get("runner") or {},
        },
    }
    return snapshot


def get_registry_stats() -> dict[str, Any]:
    ensure_registry_loaded()
    resources = list(_REGISTRY_CACHE.values())
    last_modified = max((r.get("lastModifiedAt") or "") for r in resources) if resources else None
    return {
        "total": len(resources),
        "byType": _count_by_type(resources),
        "byBackend": _count_by_field(resources, "simBackend"),
        "lastModifiedAt": last_modified,
        "lastScanAt": _LAST_SCAN_AT,
    }


def _count_by_field(resources: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in resources:
        key = str(item.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _public_resource(resource: dict[str, Any], *, include_raw: bool = False) -> dict[str, Any]:
    out = {
        "assetId": resource.get("assetId"),
        "assetType": resource.get("assetType"),
        "name": resource.get("name"),
        "version": resource.get("version"),
        "status": resource.get("status"),
        "simBackend": resource.get("simBackend"),
        "description": resource.get("description"),
        "tags": resource.get("tags") or [],
        "files": resource.get("files") or {},
        "metadata": resource.get("metadata") or {},
        "manifestPath": resource.get("manifestPath"),
        "lastModifiedAt": resource.get("lastModifiedAt"),
    }
    if resource.get("assetType") == "task":
        out["taskType"] = resource.get("taskType")
        out["requiredAssets"] = resource.get("requiredAssets")
        out["metrics"] = resource.get("metrics")
        out["runner"] = resource.get("runner")
        out["defaultConfig"] = resource.get("defaultConfig")
    if include_raw:
        out["rawManifest"] = resource.get("_raw")
    return out


def _task_config_summary(resource: dict[str, Any]) -> dict[str, Any]:
    required = resource.get("requiredAssets") or {}
    required_count = 0
    if isinstance(required, dict):
        for group in required.values():
            required_count += len(group or [])
    return {
        "assetId": resource.get("assetId"),
        "taskType": resource.get("taskType"),
        "name": resource.get("name"),
        "version": resource.get("version"),
        "status": resource.get("status"),
        "simBackend": resource.get("simBackend"),
        "description": resource.get("description"),
        "requiredAssetsCount": required_count,
        "metricsCount": len(resource.get("metrics") or []),
        "runner": resource.get("runner") or {},
        "tags": resource.get("tags") or [],
        "lastModifiedAt": resource.get("lastModifiedAt"),
    }


def _resolve_resources(required: Any) -> dict[str, list[dict[str, Any]]]:
    resolved: dict[str, list[dict[str, Any]]] = {
        "robots": [],
        "endEffectors": [],
        "objects": [],
        "scenes": [],
        "policies": [],
        "metrics": [],
    }
    if not isinstance(required, dict):
        return resolved

    mapping = {
        "robots": "robots",
        "end_effectors": "endEffectors",
        "objects": "objects",
        "scenes": "scenes",
        "policies": "policies",
    }
    for raw_key, out_key in mapping.items():
        for asset_id in required.get(raw_key) or []:
            item = get_resource(str(asset_id))
            if item:
                resolved[out_key].append(item)
    return resolved


def _task_config_detail(resource: dict[str, Any]) -> dict[str, Any]:
    summary = _task_config_summary(resource)
    required = resource.get("requiredAssets") or {}
    metric_ids = [str(m) for m in (resource.get("metrics") or [])]
    resolved = _resolve_resources(required)
    resolved["metrics"] = [get_resource(mid) for mid in metric_ids if get_resource(mid)]
    summary["requiredAssets"] = required
    summary["metrics"] = metric_ids
    summary["defaultConfig"] = resource.get("defaultConfig") or {}
    summary["resolvedResources"] = resolved
    summary["manifestPath"] = resource.get("manifestPath")
    return summary
