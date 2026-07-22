from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.resource_registry import (
    ResourceListResponse,
    ResourceOverviewResponse,
    ResourceReindexResponse,
    ResourceSummary,
    TaskConfigDetail,
    TaskConfigListResponse,
    TaskConfigSummary,
)
from app.services import resource_definition_service as resource_svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/resources/overview", response_model=ResourceOverviewResponse)
async def get_resources_overview(_: User = Depends(get_current_user)) -> ResourceOverviewResponse:
    try:
        counts = await asyncio.to_thread(resource_svc.get_resource_overview_counts)
        return ResourceOverviewResponse(**counts, source="database")
    except Exception as exc:
        logger.exception("get_resources_overview failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("/resources", response_model=ResourceListResponse)
async def list_resources(
    assetType: str | None = Query(default=None),
    resourceType: str | None = Query(default=None),
    simBackend: str | None = Query(default=None),
    status: str | None = Query(default=None),
    taskType: str | None = Query(default=None),
    includeMock: bool = Query(default=False),
    _: User = Depends(get_current_user),
) -> ResourceListResponse:
    del includeMock  # DB-first: mock 不再合并
    try:
        resources = await asyncio.to_thread(
            resource_svc.list_resource_definitions,
            resource_type=resourceType,
            asset_type=assetType,
            status=status,
            sim_backend=simBackend,
            task_type=taskType,
        )
        by_type = await asyncio.to_thread(resource_svc.count_resource_definitions_by_type)
        stats = {
            "total": len(resources),
            "byType": {
                resource_svc.RESOURCE_TYPE_TO_REGISTRY_ASSET_TYPE.get(k, k): v
                for k, v in by_type.items()
            },
            "byResourceType": by_type,
            "lastScanAt": None,
        }
        return ResourceListResponse(
            resources=[ResourceSummary(**item) for item in resources],
            total=len(resources),
            source="database",
            stats=stats,
        )
    except Exception as exc:
        logger.exception("list_resources failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("/resources/{resource_type}/{resource_id}", response_model=ResourceSummary)
async def get_resource_by_type(
    resource_type: str,
    resource_id: str,
    _: User = Depends(get_current_user),
) -> ResourceSummary:
    item = await asyncio.to_thread(
        resource_svc.get_resource_definition,
        resource_id,
        resource_type=resource_type,
    )
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="resource not found")
    return ResourceSummary(**item)


@router.get("/resources/{asset_id}", response_model=ResourceSummary)
async def get_resource(
    asset_id: str,
    _: User = Depends(get_current_user),
) -> ResourceSummary:
    item = await asyncio.to_thread(resource_svc.get_resource_definition, asset_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="resource not found")
    return ResourceSummary(**item)


@router.post("/resources/reindex", response_model=ResourceReindexResponse)
async def reindex_resources(
    _: User = Depends(get_current_user),
) -> ResourceReindexResponse:
    try:
        result = await asyncio.to_thread(resource_svc.reindex_resource_registry_to_db)
        proxy_seed = await asyncio.to_thread(resource_svc.seed_physics_proxy_models)
        result.setdefault("warnings", []).append(
            f"physics_proxy seed: created={proxy_seed.get('created', 0)} updated={proxy_seed.get('updated', 0)}"
        )
        return ResourceReindexResponse(**result, source="database")
    except Exception as exc:
        logger.exception("reindex_resources failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get("/task-configs", response_model=TaskConfigListResponse)
async def list_task_configs(
    taskType: str | None = Query(default=None),
    _: User = Depends(get_current_user),
) -> TaskConfigListResponse:
    rows = await asyncio.to_thread(resource_svc.list_task_configs_from_db, task_type=taskType)
    return TaskConfigListResponse(
        taskConfigs=[TaskConfigSummary(**row) for row in rows],
        total=len(rows),
    )


@router.get("/task-configs/{task_config_id}", response_model=TaskConfigDetail)
async def get_task_config(
    task_config_id: str,
    _: User = Depends(get_current_user),
) -> TaskConfigDetail:
    detail = await asyncio.to_thread(resource_svc.get_task_config_detail_from_db, task_config_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task config not found")
    return TaskConfigDetail(**detail)
