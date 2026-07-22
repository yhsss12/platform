"""任务模板目录：DB-first 读写与默认 seed。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.task_template_catalog import TaskTemplateCatalog
from app.services.workspace_task_template_service import DEFAULT_TASK_TEMPLATES

logger = logging.getLogger(__name__)


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

    return "task_template_catalog" in set(inspect(db.get_bind()).get_table_names())


def _merge_registry_fields(template: dict[str, Any]) -> dict[str, Any]:
    from app.services import resource_definition_service as resource_svc

    registry_id = template.get("registryTaskConfigId") or template.get("task_config_id")
    if not registry_id:
        return template

    try:
        resource = resource_svc.get_resource_definition(str(registry_id), resource_type="task_config")
        if not resource:
            return template
    except Exception as exc:
        logger.debug("task template registry merge skipped for %s: %s", registry_id, exc)
        return template

    merged = dict(template)
    if resource.get("name"):
        merged.setdefault("name", resource["name"])
    if resource.get("description"):
        merged["description"] = resource["description"]
    if resource.get("status"):
        merged["status"] = resource["status"]
    if resource.get("lastModifiedAt"):
        merged["updatedAt"] = resource["lastModifiedAt"]
    runner = resource.get("runner") or {}
    if isinstance(runner, dict):
        merged["runner"] = runner
    return merged


def _row_to_template(row: TaskTemplateCatalog) -> dict[str, Any]:
    metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
    payload = {
        "id": row.template_id,
        "name": row.display_name or row.name,
        "description": row.description or "",
        "status": row.status,
        "createdAt": _iso(row.created_at),
        "updatedAt": _iso(row.updated_at),
        "registryTaskConfigId": row.task_config_id,
        **metadata,
    }
    return _merge_registry_fields(payload)


def seed_default_task_templates() -> dict[str, int]:
    created = updated = skipped = 0
    now = _utc_now()

    with SessionLocal() as db:
        if not _table_exists(db):
            return {"created": 0, "updated": 0, "skipped": 0}

        for item in DEFAULT_TASK_TEMPLATES:
            template_id = str(item.get("id") or "")
            if not template_id:
                continue

            metadata = {k: v for k, v in item.items() if k not in {"id", "name", "description", "status"}}
            registry_id = item.get("registryTaskConfigId")

            existing = (
                db.query(TaskTemplateCatalog)
                .filter(TaskTemplateCatalog.template_id == template_id)
                .one_or_none()
            )

            if existing and not existing.is_builtin:
                skipped += 1
                continue

            if existing is None:
                db.add(
                    TaskTemplateCatalog(
                        template_id=template_id,
                        name=str(item.get("name") or template_id),
                        display_name=str(item.get("name") or template_id),
                        description=str(item.get("description") or ""),
                        category=str(item.get("taskFamily") or item.get("category") or ""),
                        simulator=str(item.get("simulatorBackend") or item.get("simulatorType") or ""),
                        robot_type=", ".join(item.get("supportedRobotTypes") or []),
                        task_config_id=str(registry_id) if registry_id else None,
                        metadata_json=metadata,
                        status=str(item.get("status") or "available"),
                        is_builtin=True,
                        created_at=now,
                        updated_at=now,
                    )
                )
                created += 1
            else:
                existing.name = str(item.get("name") or existing.name)
                existing.display_name = str(item.get("name") or existing.display_name)
                existing.description = str(item.get("description") or existing.description or "")
                existing.category = str(item.get("taskFamily") or existing.category or "")
                existing.simulator = str(
                    item.get("simulatorBackend") or item.get("simulatorType") or existing.simulator or ""
                )
                existing.robot_type = ", ".join(item.get("supportedRobotTypes") or [])
                existing.task_config_id = str(registry_id) if registry_id else existing.task_config_id
                existing.metadata_json = metadata
                existing.status = str(item.get("status") or existing.status)
                existing.is_builtin = True
                existing.updated_at = now
                updated += 1

        db.commit()

    return {"created": created, "updated": updated, "skipped": skipped}


def ensure_task_template_catalog_seeded() -> None:
    with SessionLocal() as db:
        if not _table_exists(db):
            return
        count = (
            db.query(TaskTemplateCatalog)
            .filter(TaskTemplateCatalog.status != "deleted")
            .count()
        )
    if count == 0:
        seed_default_task_templates()


def list_task_templates() -> list[dict[str, Any]]:
    ensure_task_template_catalog_seeded()

    with SessionLocal() as db:
        if not _table_exists(db):
            from app.services.workspace_task_template_service import list_task_templates_fallback

            return list_task_templates_fallback()

        rows = (
            db.query(TaskTemplateCatalog)
            .filter(TaskTemplateCatalog.status != "deleted")
            .order_by(TaskTemplateCatalog.is_builtin.desc(), TaskTemplateCatalog.template_id.asc())
            .all()
        )

    if not rows:
        seed_default_task_templates()
        return list_task_templates()

    now = _iso(_utc_now())
    results: list[dict[str, Any]] = []
    for row in rows:
        item = _row_to_template(row)
        item.setdefault("createdAt", item.get("createdAt") or now)
        item.setdefault("updatedAt", item.get("updatedAt") or now)
        results.append(item)
    return results


def count_task_templates() -> int:
    ensure_task_template_catalog_seeded()
    with SessionLocal() as db:
        if not _table_exists(db):
            from app.services.workspace_task_template_service import list_task_templates_fallback

            return len(list_task_templates_fallback())
        return (
            db.query(TaskTemplateCatalog)
            .filter(TaskTemplateCatalog.status != "deleted")
            .count()
        )


def get_task_template(template_id: str) -> Optional[dict[str, Any]]:
    ensure_task_template_catalog_seeded()
    candidate = (template_id or "").strip()
    if not candidate:
        return None

    with SessionLocal() as db:
        if not _table_exists(db):
            from app.services.workspace_task_template_service import list_task_templates_fallback

            for item in list_task_templates_fallback():
                if item.get("id") == candidate:
                    return item
            return None

        row = (
            db.query(TaskTemplateCatalog)
            .filter(
                TaskTemplateCatalog.template_id == candidate,
                TaskTemplateCatalog.status != "deleted",
            )
            .one_or_none()
        )
        if row is None:
            return None
        return _row_to_template(row)

