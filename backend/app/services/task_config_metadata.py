"""Task config metadata helpers for workspace job records."""

from __future__ import annotations

from typing import Any, Optional

from app.services.dataset_naming import TASK_TYPE_TO_TEMPLATE, task_display_name
from app.services.resource_registry_service import get_task_config_metadata_for_job


def build_job_resource_metadata(
    *,
    task_type: str,
    task_config_id: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "taskType": task_type,
        "taskTemplateId": TASK_TYPE_TO_TEMPLATE.get(task_type),
        "taskDisplayName": task_display_name(task_type),
    }
    if task_config_id:
        metadata.update(get_task_config_metadata_for_job(task_config_id))
    if extra:
        metadata.update(extra)
    return metadata
