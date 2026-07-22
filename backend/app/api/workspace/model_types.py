from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.model_type import (
    CreateModelTypeRequest,
    ModelTypeDefinitionResponse,
    ModelTypeDeleteResponse,
    ModelTypeListResponse,
    ModelTypeProbeRefreshResponse,
    ModelTypeValidateResponse,
    UpdateModelTypeRequest,
)
from app.services import model_type_service as svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/model-types", response_model=ModelTypeListResponse)
async def list_model_types(
    status: str | None = Query(default=None),
    _: User = Depends(get_current_user),
) -> ModelTypeListResponse:
    try:
        rows = await asyncio.to_thread(svc.list_model_types, status=status)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("list_model_types failed")
        raise HTTPException(status_code=500, detail=f"加载模型类型失败: {exc}") from exc
    items = [ModelTypeDefinitionResponse(**row) for row in rows]
    return ModelTypeListResponse(modelTypes=items, total=len(items))


@router.post("/model-types/probe/refresh", response_model=ModelTypeProbeRefreshResponse)
async def refresh_model_type_probe(
    _: User = Depends(get_current_user),
) -> ModelTypeProbeRefreshResponse:
    await asyncio.to_thread(svc.schedule_model_type_readiness_refresh, force=True)
    return ModelTypeProbeRefreshResponse(accepted=True)


@router.post("/model-types", response_model=ModelTypeDefinitionResponse)
async def create_model_type(
    payload: CreateModelTypeRequest,
    _: User = Depends(get_current_user),
) -> ModelTypeDefinitionResponse:
    row = await asyncio.to_thread(svc.create_model_type, payload.model_dump())
    return ModelTypeDefinitionResponse(**row)


@router.get("/model-types/{model_type_id}", response_model=ModelTypeDefinitionResponse)
async def get_model_type(
    model_type_id: str,
    _: User = Depends(get_current_user),
) -> ModelTypeDefinitionResponse:
    row = await asyncio.to_thread(svc.get_model_type, model_type_id)
    if not row:
        raise HTTPException(status_code=404, detail="模型类型不存在")
    return ModelTypeDefinitionResponse(**row)


@router.put("/model-types/{model_type_id}", response_model=ModelTypeDefinitionResponse)
async def update_model_type(
    model_type_id: str,
    payload: UpdateModelTypeRequest,
    _: User = Depends(get_current_user),
) -> ModelTypeDefinitionResponse:
    updates = payload.model_dump(exclude_unset=True)
    row = await asyncio.to_thread(svc.update_model_type, model_type_id, updates)
    return ModelTypeDefinitionResponse(**row)


@router.delete("/model-types/{model_type_id}", response_model=ModelTypeDeleteResponse)
async def delete_model_type(
    model_type_id: str,
    _: User = Depends(get_current_user),
) -> ModelTypeDeleteResponse:
    result = await asyncio.to_thread(svc.delete_model_type, model_type_id)
    return ModelTypeDeleteResponse(**result)


@router.post("/model-types/{model_type_id}/validate", response_model=ModelTypeValidateResponse)
async def validate_model_type(
    model_type_id: str,
    _: User = Depends(get_current_user),
) -> ModelTypeValidateResponse:
    result = await asyncio.to_thread(svc.validate_model_type_definition, model_type_id)
    return ModelTypeValidateResponse(**result)
