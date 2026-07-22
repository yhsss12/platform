"""资源定义目录：DB-first 读写与 registry YAML 同步。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.resource_definition import ResourceDefinition

logger = logging.getLogger(__name__)

RESOURCE_TYPES = frozenset(
    {
        "scene",
        "robot",
        "object",
        "end_effector",
        "policy",
        "metric",
        "task_config",
        "physics_proxy",
    }
)

REGISTRY_ASSET_TYPE_TO_RESOURCE_TYPE: dict[str, str] = {
    "scene": "scene",
    "robot": "robot",
    "object": "object",
    "end_effector": "end_effector",
    "policy": "policy",
    "metric": "metric",
    "task": "task_config",
}

RESOURCE_TYPE_TO_REGISTRY_ASSET_TYPE: dict[str, str] = {
    v: k for k, v in REGISTRY_ASSET_TYPE_TO_RESOURCE_TYPE.items()
}
RESOURCE_TYPE_TO_REGISTRY_ASSET_TYPE["task_config"] = "task"

USER_OWNED_SOURCES = frozenset({"user_created", "imported"})

PHYSICS_PROXY_SEEDS: list[dict[str, Any]] = [
    {
        "resource_id": "contact-force-pinn-v1",
        "name": "contact-force-pinn-v1",
        "display_name": "contact-force-pinn-v1",
        "description": "接触力 PINN 代理模型",
        "version": "v1",
        "status": "available",
        "tags": ["PINN", "接触力"],
        "metadata_json": {
            "proxyType": "接触力",
            "applicableTasks": "两次拧螺丝",
            "physicalObjects": "螺丝 / 工件 / 电批",
            "inputVariables": "位姿、速度、接触深度、材料参数",
            "outputVariables": "接触力、摩擦状态",
            "trainingMethod": "PINN-CAML",
            "errorMetric": "3.8%",
            "speedup": "12.5×",
        },
    },
    {
        "resource_id": "elastic-deform-pinn-v1",
        "name": "elastic-deform-pinn-v1",
        "display_name": "elastic-deform-pinn-v1",
        "description": "弹性形变 PINN 代理模型",
        "version": "v1",
        "status": "available",
        "tags": ["PINN", "弹性形变"],
        "metadata_json": {
            "proxyType": "弹性形变",
            "applicableTasks": "装夹任务",
            "physicalObjects": "工件 / 夹具",
            "inputVariables": "夹紧力、材料参数、接触边界",
            "outputVariables": "位移场、应力分布",
            "trainingMethod": "PINN",
            "errorMetric": "4.6%",
            "speedup": "9.2×",
        },
    },
    {
        "resource_id": "cable-shape-pinn-v1",
        "name": "cable-shape-pinn-v1",
        "display_name": "cable-shape-pinn-v1",
        "description": "线缆形状 PINN 代理模型",
        "version": "v1",
        "status": "draft",
        "tags": ["PINN", "柔性对象"],
        "metadata_json": {
            "proxyType": "柔性对象",
            "applicableTasks": "线缆插接任务",
            "physicalObjects": "线缆 / 接插件",
            "inputVariables": "端点位姿、约束点、材料刚度",
            "outputVariables": "线缆曲线、弯曲应力",
            "trainingMethod": "PINN-CAML",
            "errorMetric": "5.1%",
            "speedup": "8.7×",
            "validating": True,
        },
    },
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _table_exists(db: Session) -> bool:
    from sqlalchemy import inspect

    return "resource_definitions" in set(inspect(db.get_bind()).get_table_names())


def _registry_resource_to_row(resource: dict[str, Any]) -> dict[str, Any]:
    asset_type = str(resource.get("assetType") or "")
    resource_type = REGISTRY_ASSET_TYPE_TO_RESOURCE_TYPE.get(asset_type)
    if not resource_type:
        raise ValueError(f"unsupported asset_type: {asset_type}")

    raw = resource.get("rawManifest") or {}
    metadata = dict(resource.get("metadata") or {})
    if asset_type == "metric":
        metadata.setdefault("displayName", metadata.get("displayName") or resource.get("name"))

    return {
        "resource_id": str(resource.get("assetId") or ""),
        "resource_type": resource_type,
        "name": str(resource.get("name") or ""),
        "display_name": str(resource.get("name") or ""),
        "description": str(resource.get("description") or ""),
        "version": str(resource.get("version") or "v1"),
        "status": _normalize_status(str(resource.get("status") or "available")),
        "tags": list(resource.get("tags") or []),
        "manifest_json": raw if isinstance(raw, dict) and raw else _build_manifest_from_resource(resource),
        "metadata_json": {
            **metadata,
            "simBackend": resource.get("simBackend"),
            "taskType": resource.get("taskType"),
            "files": resource.get("files") or {},
        },
        "manifest_path": str(resource.get("manifestPath") or ""),
        "storage_uri": _extract_storage_uri(resource),
        "source": "registry",
    }


def _build_manifest_from_resource(resource: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "asset_id": resource.get("assetId"),
        "asset_type": resource.get("assetType"),
        "name": resource.get("name"),
        "version": resource.get("version"),
        "status": resource.get("status"),
        "sim_backend": resource.get("simBackend"),
        "description": resource.get("description"),
        "tags": resource.get("tags") or [],
        "files": resource.get("files") or {},
        "metadata": resource.get("metadata") or {},
    }
    if resource.get("assetType") == "task":
        manifest["task_type"] = resource.get("taskType")
        manifest["required_assets"] = resource.get("requiredAssets")
        manifest["metrics"] = resource.get("metrics")
        manifest["runner"] = resource.get("runner")
        manifest["default_config"] = resource.get("defaultConfig")
    return manifest


def _extract_storage_uri(resource: dict[str, Any]) -> Optional[str]:
    files = resource.get("files") or {}
    if not isinstance(files, dict):
        return None
    for key in ("primary", "manifest", "model", "scene"):
        value = files.get(key)
        if value:
            return str(value)
    for value in files.values():
        if value:
            return str(value)
    return None


def _normalize_status(status: str) -> str:
    if status in {"available", "draft", "disabled", "deleted", "experimental", "deprecated"}:
        if status == "experimental":
            return "draft"
        if status == "deprecated":
            return "disabled"
        return status
    return "available"


def _row_to_api_resource(row: ResourceDefinition) -> dict[str, Any]:
    manifest = row.manifest_json if isinstance(row.manifest_json, dict) else {}
    metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
    asset_type = RESOURCE_TYPE_TO_REGISTRY_ASSET_TYPE.get(row.resource_type, row.resource_type)

    updated = _iso(row.updated_at)
    resource: dict[str, Any] = {
        "assetId": row.resource_id,
        "assetType": asset_type,
        "resourceType": row.resource_type,
        "resourceId": row.resource_id,
        "name": row.display_name or row.name,
        "version": row.version,
        "status": row.status,
        "simBackend": str(metadata.get("simBackend") or manifest.get("sim_backend") or ""),
        "description": row.description or "",
        "tags": list(row.tags or []),
        "files": dict(metadata.get("files") or manifest.get("files") or {}),
        "metadata": metadata,
        "manifestPath": row.manifest_path or "",
        "lastModifiedAt": updated,
        "source": row.source,
        "storageUri": row.storage_uri,
    }

    if row.resource_type == "task_config":
        resource["taskType"] = metadata.get("taskType") or manifest.get("task_type")
        resource["requiredAssets"] = manifest.get("required_assets")
        resource["metrics"] = list(manifest.get("metrics") or [])
        resource["runner"] = manifest.get("runner") or {}
        resource["defaultConfig"] = manifest.get("default_config") or {}

    if row.resource_type == "physics_proxy":
        resource["physicsProxy"] = metadata

    return resource


def ensure_resource_catalog_seeded() -> None:
    with SessionLocal() as db:
        if not _table_exists(db):
            return
        count = (
            db.query(ResourceDefinition)
            .filter(ResourceDefinition.status != "deleted")
            .count()
        )
        proxy_count = (
            db.query(ResourceDefinition)
            .filter(
                ResourceDefinition.resource_type == "physics_proxy",
                ResourceDefinition.status != "deleted",
            )
            .count()
        )
    if count == 0:
        reindex_resource_registry_to_db()
    if proxy_count == 0:
        seed_physics_proxy_models()


def list_resource_definitions(
    *,
    resource_type: Optional[str] = None,
    asset_type: Optional[str] = None,
    status: Optional[str] = None,
    sim_backend: Optional[str] = None,
    task_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    ensure_resource_catalog_seeded()

    resolved_type = resource_type
    if not resolved_type and asset_type:
        resolved_type = REGISTRY_ASSET_TYPE_TO_RESOURCE_TYPE.get(asset_type)

    with SessionLocal() as db:
        if not _table_exists(db):
            return _fallback_list_from_registry(
                asset_type=asset_type,
                sim_backend=sim_backend,
                status=status,
                task_type=task_type,
            )

        query = db.query(ResourceDefinition).filter(ResourceDefinition.status != "deleted")
        if resolved_type:
            query = query.filter(ResourceDefinition.resource_type == resolved_type)
        if status:
            query = query.filter(ResourceDefinition.status == status)

        rows = query.order_by(
            ResourceDefinition.resource_type.asc(),
            ResourceDefinition.resource_id.asc(),
        ).all()

    resources = [_row_to_api_resource(row) for row in rows]

    if sim_backend:
        resources = [r for r in resources if r.get("simBackend") == sim_backend]
    if task_type:
        resources = [r for r in resources if r.get("taskType") == task_type]

    return resources


def get_resource_definition(
    resource_id: str,
    *,
    resource_type: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    ensure_resource_catalog_seeded()
    candidate = (resource_id or "").strip()
    if not candidate:
        return None

    with SessionLocal() as db:
        if not _table_exists(db):
            from app.services import resource_registry_service as registry

            item = registry.get_resource(candidate)
            return item

        query = db.query(ResourceDefinition).filter(
            ResourceDefinition.resource_id == candidate,
            ResourceDefinition.status != "deleted",
        )
        if resource_type:
            query = query.filter(ResourceDefinition.resource_type == resource_type)
        row = query.order_by(ResourceDefinition.updated_at.desc()).first()
        if row is None:
            return None
        return _row_to_api_resource(row)


def upsert_resource_definition(payload: dict[str, Any]) -> dict[str, Any]:
    resource_type = str(payload.get("resource_type") or payload.get("resourceType") or "")
    resource_id = str(payload.get("resource_id") or payload.get("resourceId") or "")
    version = str(payload.get("version") or "v1")
    if resource_type not in RESOURCE_TYPES or not resource_id:
        raise ValueError("resource_type and resource_id are required")

    now = _utc_now()
    with SessionLocal() as db:
        row = (
            db.query(ResourceDefinition)
            .filter(
                ResourceDefinition.resource_type == resource_type,
                ResourceDefinition.resource_id == resource_id,
                ResourceDefinition.version == version,
            )
            .one_or_none()
        )
        if row is None:
            row = ResourceDefinition(
                resource_id=resource_id,
                resource_type=resource_type,
                version=version,
                created_at=now,
            )
            db.add(row)

        row.name = str(payload.get("name") or resource_id)
        row.display_name = payload.get("display_name") or payload.get("displayName") or row.name
        row.description = payload.get("description")
        row.status = _normalize_status(str(payload.get("status") or row.status or "available"))
        row.tags = list(payload.get("tags") or [])
        row.manifest_json = dict(payload.get("manifest_json") or payload.get("manifestJson") or {})
        row.metadata_json = dict(payload.get("metadata_json") or payload.get("metadataJson") or {})
        row.manifest_path = payload.get("manifest_path") or payload.get("manifestPath")
        row.storage_uri = payload.get("storage_uri") or payload.get("storageUri")
        row.source = str(payload.get("source") or row.source or "user_created")
        row.updated_at = now
        db.commit()
        db.refresh(row)
        return _row_to_api_resource(row)


def soft_delete_resource_definition(resource_id: str, *, resource_type: Optional[str] = None) -> bool:
    with SessionLocal() as db:
        query = db.query(ResourceDefinition).filter(ResourceDefinition.resource_id == resource_id)
        if resource_type:
            query = query.filter(ResourceDefinition.resource_type == resource_type)
        row = query.one_or_none()
        if row is None:
            return False
        row.status = "deleted"
        row.updated_at = _utc_now()
        db.commit()
        return True


def _upsert_registry_row(db: Session, row_data: dict[str, Any]) -> str:
    """Return action: created | updated | skipped."""
    existing = (
        db.query(ResourceDefinition)
        .filter(
            ResourceDefinition.resource_type == row_data["resource_type"],
            ResourceDefinition.resource_id == row_data["resource_id"],
            ResourceDefinition.version == row_data["version"],
        )
        .one_or_none()
    )

    if existing and existing.source in USER_OWNED_SOURCES:
        return "skipped"

    now = _utc_now()
    if existing is None:
        db.add(
            ResourceDefinition(
                **row_data,
                created_at=now,
                updated_at=now,
            )
        )
        return "created"

    existing.name = row_data["name"]
    existing.display_name = row_data["display_name"]
    existing.description = row_data["description"]
    existing.status = row_data["status"]
    existing.tags = row_data["tags"]
    existing.manifest_json = row_data["manifest_json"]
    existing.metadata_json = row_data["metadata_json"]
    existing.manifest_path = row_data["manifest_path"]
    existing.storage_uri = row_data.get("storage_uri")
    existing.source = row_data["source"]
    existing.updated_at = now
    return "updated"


def reindex_resource_registry_to_db() -> dict[str, Any]:
    from app.services import resource_registry_service as registry

    scan_result = registry.scan_resource_registry(force=True)
    created = updated = skipped = synced = 0
    errors: list[str] = list(scan_result.get("errors") or [])
    warnings: list[str] = list(scan_result.get("warnings") or [])

    with SessionLocal() as db:
        if not _table_exists(db):
            return {
                "scanned": scan_result.get("scanned", 0),
                "synced": 0,
                "created": 0,
                "updated": 0,
                "skipped": 0,
                "valid": scan_result.get("valid", 0),
                "invalid": scan_result.get("invalid", 0),
                "resourcesByType": scan_result.get("resourcesByType") or {},
                "errors": errors + ["resource_definitions table not found"],
                "warnings": warnings,
                "lastScanAt": scan_result.get("lastScanAt"),
            }

        for summary in registry.list_resources():
            try:
                asset_id = str(summary.get("assetId") or "")
                item = registry.get_resource(asset_id)
                if not item:
                    continue
                row_data = _registry_resource_to_row(item)
                action = _upsert_registry_row(db, row_data)
                synced += 1
                if action == "created":
                    created += 1
                elif action == "updated":
                    updated += 1
                else:
                    skipped += 1
            except Exception as exc:
                errors.append(f"{asset_id}: {exc}")

        db.commit()

    resources_by_type = count_resource_definitions_by_type()
    return {
        "scanned": scan_result.get("scanned", 0),
        "synced": synced,
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "valid": scan_result.get("valid", 0),
        "invalid": scan_result.get("invalid", 0),
        "resourcesByType": resources_by_type,
        "errors": errors,
        "warnings": warnings,
        "lastScanAt": scan_result.get("lastScanAt"),
    }


def seed_physics_proxy_models() -> dict[str, int]:
    created = updated = skipped = 0
    with SessionLocal() as db:
        if not _table_exists(db):
            return {"created": 0, "updated": 0, "skipped": 0}

        for item in PHYSICS_PROXY_SEEDS:
            row_data = {
                "resource_id": item["resource_id"],
                "resource_type": "physics_proxy",
                "name": item["name"],
                "display_name": item.get("display_name") or item["name"],
                "description": item.get("description"),
                "version": item.get("version", "v1"),
                "status": item.get("status", "available"),
                "tags": list(item.get("tags") or []),
                "manifest_json": {},
                "metadata_json": dict(item.get("metadata_json") or {}),
                "manifest_path": None,
                "storage_uri": None,
                "source": "seeded",
            }
            existing = (
                db.query(ResourceDefinition)
                .filter(
                    ResourceDefinition.resource_type == "physics_proxy",
                    ResourceDefinition.resource_id == item["resource_id"],
                    ResourceDefinition.version == row_data["version"],
                )
                .one_or_none()
            )
            if existing and existing.source in USER_OWNED_SOURCES:
                skipped += 1
                continue
            action = _upsert_registry_row(db, row_data)
            if action == "created":
                created += 1
            elif action == "updated":
                updated += 1
            else:
                skipped += 1
        db.commit()

    return {"created": created, "updated": updated, "skipped": skipped}


def count_resource_definitions_by_type(*, include_deleted: bool = False) -> dict[str, int]:
    ensure_resource_catalog_seeded()
    with SessionLocal() as db:
        if not _table_exists(db):
            return {}
        query = db.query(
            ResourceDefinition.resource_type,
            func.count(ResourceDefinition.id),
        )
        if not include_deleted:
            query = query.filter(ResourceDefinition.status != "deleted")
        rows = query.group_by(ResourceDefinition.resource_type).all()
    return {str(resource_type): int(count) for resource_type, count in rows}


def count_resource_definitions(
    resource_types: list[str],
    *,
    include_deleted: bool = False,
) -> int:
    ensure_resource_catalog_seeded()
    with SessionLocal() as db:
        if not _table_exists(db):
            return 0
        query = db.query(ResourceDefinition).filter(
            ResourceDefinition.resource_type.in_(resource_types)
        )
        if not include_deleted:
            query = query.filter(ResourceDefinition.status != "deleted")
        return int(query.count())


def get_resource_overview_counts() -> dict[str, Any]:
    ensure_resource_catalog_seeded()

    from app.services.model_asset_db_service import list_model_assets_from_db
    from app.services.model_type_service import list_model_types
    from app.services.sam3d_asset_paths import count_sim_asset_jobs
    from app.services.task_template_catalog_service import count_task_templates

    by_type = count_resource_definitions_by_type()
    warnings: list[dict[str, Any]] = []
    counts: dict[str, Optional[int]] = {}

    def _count_category(category: str, fn) -> None:
        try:
            counts[category] = int(fn())
        except Exception as exc:
            logger.warning("resource overview category %s failed: %s", category, exc, exc_info=True)
            warnings.append({"category": category, "message": str(exc)})
            counts[category] = None

    _count_category("taskTemplates", count_task_templates)
    _count_category("modelAssets", lambda: len(list_model_assets_from_db()))
    counts["metrics"] = int(by_type.get("metric", 0))
    counts["scenes"] = int(by_type.get("scene", 0))
    counts["robots"] = int(by_type.get("robot", 0))
    counts["objects"] = int(by_type.get("object", 0)) + int(by_type.get("end_effector", 0))
    counts["policyAssets"] = int(by_type.get("policy", 0))
    counts["physicsProxies"] = int(by_type.get("physics_proxy", 0))
    _count_category("modelTypes", lambda: len(list_model_types()))
    counts["craftConfig"] = int(by_type.get("task_config", 0))
    # simAssets: count all asset pipeline job folders (any stage except missing dirs).
    _count_category("simAssets", lambda: count_sim_asset_jobs(reconstructed_only=False))

    return {
        **counts,
        "warnings": warnings,
        "partialFailure": bool(warnings),
    }


def list_task_configs_from_db(*, task_type: Optional[str] = None) -> list[dict[str, Any]]:
    resources = list_resource_definitions(resource_type="task_config", task_type=task_type)
    rows: list[dict[str, Any]] = []
    for resource in resources:
        required = resource.get("requiredAssets") or {}
        required_count = 0
        if isinstance(required, dict):
            for group in required.values():
                required_count += len(group or [])
        rows.append(
            {
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
        )
    return rows


def get_task_config_detail_from_db(task_config_id: str) -> Optional[dict[str, Any]]:
    resource = get_resource_definition(task_config_id, resource_type="task_config")
    if not resource:
        return None

    required = resource.get("requiredAssets") or {}
    metric_ids = [str(m) for m in (resource.get("metrics") or [])]
    resolved = _resolve_resources_from_db(required)
    resolved["metrics"] = [
        item for mid in metric_ids if (item := get_resource_definition(mid)) is not None
    ]

    return {
        "assetId": resource.get("assetId"),
        "taskType": resource.get("taskType"),
        "name": resource.get("name"),
        "version": resource.get("version"),
        "status": resource.get("status"),
        "simBackend": resource.get("simBackend"),
        "description": resource.get("description"),
        "requiredAssetsCount": sum(len(v or []) for v in required.values()) if isinstance(required, dict) else 0,
        "metricsCount": len(metric_ids),
        "runner": resource.get("runner") or {},
        "tags": resource.get("tags") or [],
        "lastModifiedAt": resource.get("lastModifiedAt"),
        "requiredAssets": required,
        "metrics": metric_ids,
        "defaultConfig": resource.get("defaultConfig") or {},
        "resolvedResources": resolved,
        "manifestPath": resource.get("manifestPath"),
    }


def _resolve_resources_from_db(required: Any) -> dict[str, list[dict[str, Any]]]:
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
        "robots": ("robots", "robot"),
        "end_effectors": ("endEffectors", "end_effector"),
        "objects": ("objects", "object"),
        "scenes": ("scenes", "scene"),
        "policies": ("policies", "policy"),
    }
    for raw_key, (out_key, resource_type) in mapping.items():
        for asset_id in required.get(raw_key) or []:
            item = get_resource_definition(str(asset_id), resource_type=resource_type)
            if item:
                resolved[out_key].append(item)
    return resolved


def _fallback_list_from_registry(
    *,
    asset_type: Optional[str],
    sim_backend: Optional[str],
    status: Optional[str],
    task_type: Optional[str],
) -> list[dict[str, Any]]:
    from app.services import resource_registry_service as registry

    registry.ensure_registry_loaded()
    items = registry.list_resources(
        asset_type=asset_type,
        sim_backend=sim_backend,
        status=status,
        task_type=task_type,
    )
    return items
