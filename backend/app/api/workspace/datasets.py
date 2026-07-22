from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.workspace_benchmark import (
    BuildDatasetFromImportRequest,
    BuildDatasetFromImportResponse,
    DatasetImportUploadResponse,
    DatasetListResponse,
    DatasetResponse,
)
from app.services import workspace_dataset_build_service as build_svc
from app.services import workspace_dataset_import_service as import_svc
from app.services import workspace_dataset_service as svc
from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    limit: int = Query(10, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    format: Optional[str] = Query(None, alias="format"),
    user: User = Depends(get_current_user),
) -> DatasetListResponse:
    started = time.perf_counter()
    page_rows, total, cache_hit = await asyncio.to_thread(
        svc.list_datasets_for_list_api,
        user_id=user.id,
        search=search,
        task=task,
        source=source,
        format_filter=format,
        limit=limit,
        offset=offset,
    )
    datasets = [DatasetResponse(**row) for row in page_rows]
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "[api-timing] GET /workspace/datasets %.1fms cache=%s limit=%s offset=%s search=%s task=%s source=%s format=%s total=%s",
        elapsed_ms,
        "hit" if cache_hit else "miss",
        limit,
        offset,
        search or "",
        task or "",
        source or "",
        format or "",
        total,
    )
    return DatasetListResponse(datasets=datasets, total=total)


@router.post("/datasets/import/upload", response_model=DatasetImportUploadResponse)
async def upload_import_dataset(
    name: str = Form(...),
    dataSource: str = Form(...),
    taskType: str = Form(...),
    robotType: str = Form(...),
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
) -> DatasetImportUploadResponse:
    result = await import_svc.import_hdf5_dataset_upload(
        name=name,
        data_source=dataSource,
        task_type=taskType,
        robot_type=robotType,
        file=file,
    )
    invalidate_workspace_dataset_list_cache()
    dataset = DatasetResponse(**result["dataset"])
    return DatasetImportUploadResponse(
        dataset=dataset,
        datasetId=str(result["datasetId"]),
        status=str(result["status"]),
        validationReport=result.get("validationReport") or {},
    )


@router.delete("/datasets/import/{dataset_id}")
async def delete_imported_dataset(
    dataset_id: str,
    _: User = Depends(get_current_user),
) -> dict:
    try:
        result = await asyncio.to_thread(import_svc.delete_imported_dataset, dataset_id)
        invalidate_workspace_dataset_list_cache()
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/schema")
async def get_dataset_schema(
    dataset_id: str,
    _: User = Depends(get_current_user),
) -> dict:
    try:
        return await asyncio.to_thread(build_svc.get_import_dataset_schema, dataset_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/datasets/build/from-import", response_model=BuildDatasetFromImportResponse)
async def build_dataset_from_import(
    body: BuildDatasetFromImportRequest,
    _: User = Depends(get_current_user),
) -> BuildDatasetFromImportResponse:
    try:
        result = await asyncio.to_thread(
            build_svc.build_dataset_from_import,
            {
                **body.model_dump(),
                "fieldMapping": (
                    body.fieldMapping.model_dump(exclude_none=True)
                    if body.fieldMapping is not None
                    else {}
                ),
            },
        )
        invalidate_workspace_dataset_list_cache()
        dataset_row = result.get("dataset")
        dataset = DatasetResponse(**dataset_row) if dataset_row else None
        return BuildDatasetFromImportResponse(
            builtDatasetId=str(result["builtDatasetId"]),
            status=str(result["status"]),
            trainable=bool(result.get("trainable")),
            directTrainable=bool(result.get("directTrainable", True)),
            dataset=dataset,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("build dataset from import failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/datasets/built/{dataset_id}")
async def delete_built_dataset(
    dataset_id: str,
    _: User = Depends(get_current_user),
) -> dict:
    try:
        result = await asyncio.to_thread(build_svc.delete_built_dataset, dataset_id)
        invalidate_workspace_dataset_list_cache()
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
