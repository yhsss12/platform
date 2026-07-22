from __future__ import annotations

import asyncio
import logging

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.core.api_timing import log_api_duration, paginate_rows

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.workspace_task_template import TaskTemplateListResponse, TaskTemplateResponse
from app.services import task_template_catalog_service as svc
from app.services.workspace_task_template_service import enrich_task_template_derived_fields

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/task-templates", response_model=TaskTemplateListResponse)
async def list_task_templates(
    limit: Optional[int] = Query(None, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: User = Depends(get_current_user),
) -> TaskTemplateListResponse:
    with log_api_duration("GET /workspace/task-templates", limit=limit, offset=offset):
        rows = await asyncio.to_thread(svc.list_task_templates)
        total = len(rows)
        page_rows = paginate_rows(rows, limit=limit, offset=offset)
        templates = [
            TaskTemplateResponse(**enrich_task_template_derived_fields(row)) for row in page_rows
        ]
    return TaskTemplateListResponse(taskTemplates=templates, total=total)
