from __future__ import annotations

import asyncio
import logging
import time

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.core.api_timing import log_api_duration

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.workspace_benchmark import (
    ModelAssetDeleteResponse,
    ModelAssetFilterOptionsResponse,
    ModelAssetImportResponse,
    ModelAssetListResponse,
    ModelAssetResponse,
    TrainingJobModelAssetItemResponse,
    TrainingJobModelAssetListResponse,
)
from app.services import workspace_model_asset_service as svc
from app.services.model_asset_import_service import import_pretrained_model_asset

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/model-assets", response_model=ModelAssetListResponse)
async def list_model_assets(
    forEvaluation: bool = Query(False, alias="forEvaluation"),
    taskType: Optional[str] = Query(None, alias="taskType"),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    modelType: Optional[str] = Query(None, alias="modelType"),
    trainingJobId: Optional[str] = Query(None, alias="trainingJobId"),
    datasetId: Optional[str] = Query(None, alias="datasetId"),
    source: Optional[str] = Query(None),
    dataset: Optional[str] = Query(None),
    sourceTask: Optional[str] = Query(None, alias="sourceTask"),
    limit: Optional[int] = Query(None, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: User = Depends(get_current_user),
) -> ModelAssetListResponse:
    with log_api_duration(
        "GET /workspace/model-assets",
        forEvaluation=forEvaluation,
        taskType=taskType,
        search=search,
        status=status,
        modelType=modelType,
        trainingJobId=trainingJobId,
        datasetId=datasetId,
        source=source,
        limit=limit,
        offset=offset,
    ):
        page_rows, total, timing = await asyncio.to_thread(
            svc.list_model_assets_for_list_api,
            for_evaluation=forEvaluation,
            evaluation_task_type=taskType,
            search=search,
            status=status,
            model_type=modelType,
            training_job_id=trainingJobId,
            dataset_id=datasetId,
            source=source,
            dataset_label=dataset,
            source_task=sourceTask,
            limit=limit,
            offset=offset,
        )
        serialize_started = time.perf_counter()
        assets = [ModelAssetResponse(**row) for row in page_rows]
        timing.json_serialize_ms = (time.perf_counter() - serialize_started) * 1000
        svc.log_model_asset_list_timing(
            timing,
            forEvaluation=forEvaluation,
            taskType=taskType,
            limit=limit,
            offset=offset,
        )
    return ModelAssetListResponse(assets=assets, total=total)


@router.get("/model-assets/filter-options", response_model=ModelAssetFilterOptionsResponse)
async def list_model_asset_filter_options(
    _: User = Depends(get_current_user),
) -> ModelAssetFilterOptionsResponse:
    with log_api_duration("GET /workspace/model-assets/filter-options"):
        payload = await asyncio.to_thread(svc.list_model_asset_filter_options)
    return ModelAssetFilterOptionsResponse(**payload)


@router.get("/model-assets/by-training-job/{train_job_id}", response_model=ModelAssetListResponse)
async def list_model_assets_by_training_job(
    train_job_id: str,
    _: User = Depends(get_current_user),
) -> ModelAssetListResponse:
    rows = await asyncio.to_thread(svc.list_model_assets_for_training_job, train_job_id)
    assets = [ModelAssetResponse(**row) for row in rows]
    return ModelAssetListResponse(assets=assets, total=len(assets))


@router.get(
    "/model-assets/by-training-job/{train_job_id}/detail",
    response_model=TrainingJobModelAssetListResponse,
)
async def list_training_job_model_assets_detail(
    train_job_id: str,
    _: User = Depends(get_current_user),
) -> TrainingJobModelAssetListResponse:
    payload = await asyncio.to_thread(svc.list_training_job_model_assets_detail, train_job_id)
    assets = [TrainingJobModelAssetItemResponse(**row) for row in payload.get("modelAssets", [])]
    return TrainingJobModelAssetListResponse(
        assets=assets,
        total=int(payload.get("total") or len(assets)),
        listMessage=payload.get("listMessage"),
    )


@router.get("/model-assets/{model_asset_id}", response_model=ModelAssetResponse)
async def get_model_asset(
    model_asset_id: str,
    _: User = Depends(get_current_user),
) -> ModelAssetResponse:
    row = await asyncio.to_thread(svc.get_model_asset_by_id, model_asset_id)
    if not row:
        raise HTTPException(status_code=404, detail="Model asset not found")
    return ModelAssetResponse(**row)


@router.delete("/model-assets/{model_asset_id}", response_model=ModelAssetDeleteResponse)
async def delete_model_asset(
    model_asset_id: str,
    _: User = Depends(get_current_user),
) -> ModelAssetDeleteResponse:
    result = await asyncio.to_thread(svc.delete_model_asset, model_asset_id)
    return ModelAssetDeleteResponse(**result)


@router.post("/model-assets/import", response_model=ModelAssetImportResponse)
async def import_model_asset(
    modelName: str = Form(...),
    modelType: str = Form(...),
    taskType: str = Form(...),
    datasetId: str = Form(...),
    checkpoint: UploadFile = File(...),
    metadata: Optional[UploadFile] = File(None),
    note: Optional[str] = Form(None),
    _: User = Depends(get_current_user),
) -> ModelAssetImportResponse:
    result = await import_pretrained_model_asset(
        model_name=modelName,
        model_type=modelType,
        task_type=taskType,
        dataset_id=datasetId,
        checkpoint_file=checkpoint,
        metadata_file=metadata,
        note=note,
    )
    return ModelAssetImportResponse(**result)
