from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.responses import FileResponse

from app.core.api_timing import log_api_duration, paginate_rows
from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.evaluation import (
    DatasetEvaluateRequest,
    EvaluateAsyncRequest,
    EvaluateAsyncResponse,
    EvaluationCapabilitiesResponse,
    EvaluationJobBatchDeleteRequest,
    EvaluationJobBatchDeleteResponse,
    EvaluationJobDeleteResponse,
    EvaluationJobListItem,
    EvaluationJobListResponse,
    EvaluationJobStatusResponse,
    EvaluationLogResponse,
    EvaluationPendingRecordDeleteResponse,
    EvaluationReportExportRequest,
)
from app.services.benchmark_adapters.registry import get_benchmark_adapter, list_benchmark_adapters
from app.services.evaluation import evaluation_service as svc
from app.services.evaluation.report_export.export_options import ExportOptions
from app.services.evaluation.report_export.service import export_evaluation_report

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/capabilities", response_model=list[EvaluationCapabilitiesResponse])
async def list_evaluation_capabilities(
    _: User = Depends(get_current_user),
) -> list[EvaluationCapabilitiesResponse]:
    return [
        EvaluationCapabilitiesResponse(**adapter.get_capabilities().to_api_dict())
        for adapter in list_benchmark_adapters()
    ]


@router.get("/capabilities/{task_type}", response_model=EvaluationCapabilitiesResponse)
async def get_evaluation_capabilities(
    task_type: str,
    _: User = Depends(get_current_user),
) -> EvaluationCapabilitiesResponse:
    adapter = get_benchmark_adapter(task_type)
    return EvaluationCapabilitiesResponse(**adapter.get_capabilities().to_api_dict())


@router.post("/evaluate-async", response_model=EvaluateAsyncResponse)
async def evaluate_async(
    payload: EvaluateAsyncRequest,
    _: User = Depends(get_current_user),
) -> EvaluateAsyncResponse:
    result = svc.start_evaluate_async(payload)
    logger.info(
        "evaluation evaluate-async evalJobId=%s taskType=%s mode=%s",
        result.get("evalJobId"),
        result.get("taskType"),
        result.get("evaluationMode"),
    )
    return EvaluateAsyncResponse(**result)


@router.post("/dataset-evaluate-async", response_model=EvaluateAsyncResponse)
async def dataset_evaluate_async(
    payload: DatasetEvaluateRequest,
    _: User = Depends(get_current_user),
) -> EvaluateAsyncResponse:
    result = svc.start_dataset_evaluate_async(payload)
    logger.info(
        "evaluation dataset-evaluate-async evalJobId=%s datasetId=%s metrics=%s",
        result.get("evalJobId"),
        payload.config.datasetId,
        payload.config.metrics,
    )
    return EvaluateAsyncResponse(**result)


@router.get("/jobs", response_model=EvaluationJobListResponse)
async def list_evaluation_jobs(
    limit: int = Query(10, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    mode: Optional[str] = Query(None),
    backend: Optional[str] = Query(None),
    _: User = Depends(get_current_user),
) -> EvaluationJobListResponse:
    started = time.perf_counter()
    rows = svc.list_evaluation_jobs(sync_stale=False)
    q = (search or "").strip().lower()
    status_val = (status or "").strip()
    mode_val = (mode or "").strip()
    backend_val = (backend or "").strip()
    filtered = rows
    if status_val:
        filtered = [row for row in filtered if str(row.get("status") or "") == status_val]
    if mode_val:
        filtered = [
            row
            for row in filtered
            if mode_val in str(row.get("evaluationMode") or "") or mode_val in str(row.get("evaluationTypeLabel") or "")
        ]
    if backend_val:
        filtered = [row for row in filtered if backend_val in str(row.get("runner") or "")]
    if q:
        filtered = [
            row
            for row in filtered
            if q
            in " ".join(
                str(row.get(key) or "")
                for key in (
                    "evalJobId",
                    "jobId",
                    "taskName",
                    "taskType",
                    "evaluationMode",
                    "status",
                    "message",
                )
            ).lower()
        ]
    total = len(filtered)
    page_rows = paginate_rows(filtered, limit=limit, offset=offset)
    jobs = [EvaluationJobListItem(**row) for row in page_rows]
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "[api-timing] GET /workspace/evaluation/jobs %.1fms cache=n/a limit=%s offset=%s total=%s",
        elapsed_ms,
        limit,
        offset,
        total,
    )
    return EvaluationJobListResponse(jobs=jobs, total=total)


@router.delete("/jobs/{eval_job_id}", response_model=EvaluationJobDeleteResponse)
async def delete_evaluation_job(
    eval_job_id: str,
    _: User = Depends(get_current_user),
) -> EvaluationJobDeleteResponse:
    result = svc.delete_evaluation_job(eval_job_id)
    logger.info("evaluation job deleted id=%s", result.get("evalJobId"))
    return EvaluationJobDeleteResponse(**result)


@router.delete("/records/{workspace_job_id}", response_model=EvaluationPendingRecordDeleteResponse)
async def delete_pending_evaluation_record(
    workspace_job_id: int,
    _: User = Depends(get_current_user),
) -> EvaluationPendingRecordDeleteResponse:
    result = svc.delete_pending_evaluation_record(workspace_job_id)
    logger.info(
        "evaluation pending record deleted workspaceJobId=%s jobId=%s",
        result.get("workspaceJobId"),
        result.get("jobId"),
    )
    return EvaluationPendingRecordDeleteResponse(**result)


@router.post("/jobs/batch-delete", response_model=EvaluationJobBatchDeleteResponse)
async def batch_delete_evaluation_jobs(
    payload: EvaluationJobBatchDeleteRequest,
    _: User = Depends(get_current_user),
) -> EvaluationJobBatchDeleteResponse:
    if not payload.evalJobIds and not payload.workspaceJobIds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="evalJobIds 或 workspaceJobIds 至少提供一个",
        )
    result = svc.batch_delete_evaluation_jobs(
        payload.evalJobIds,
        workspace_job_ids=payload.workspaceJobIds,
    )
    logger.info(
        "evaluation jobs batch deleted count=%s failed=%s",
        result.get("deletedCount"),
        len(result.get("failed") or []),
    )
    return EvaluationJobBatchDeleteResponse(**result)


@router.get("/jobs/{eval_job_id}/status", response_model=EvaluationJobStatusResponse)
async def get_evaluation_job_status(
    eval_job_id: str,
    _: User = Depends(get_current_user),
) -> EvaluationJobStatusResponse:
    result = svc.get_evaluation_status(eval_job_id)
    return EvaluationJobStatusResponse(**result)


@router.get("/jobs/{eval_job_id}/log", response_model=EvaluationLogResponse)
async def get_evaluation_job_log(
    eval_job_id: str,
    _: User = Depends(get_current_user),
) -> EvaluationLogResponse:
    tail = svc.read_evaluation_log_tail(eval_job_id)
    return EvaluationLogResponse(evalJobId=eval_job_id, tail=tail)


@router.get("/jobs/{eval_job_id}/result")
async def get_evaluation_job_result(
    eval_job_id: str,
    _: User = Depends(get_current_user),
) -> dict:
    return svc.get_evaluation_result(eval_job_id)


@router.get("/jobs/{eval_job_id}/video")
async def get_evaluation_job_video(
    eval_job_id: str,
    episode: Optional[int] = Query(default=None, ge=0, le=99),
    _: User = Depends(get_current_user),
):
    video_path = svc.resolve_evaluation_video_path(eval_job_id, episode=episode)
    if video_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation video not found",
        )
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


def _resolve_export_options(
    *,
    format: str,
    template: str = "standard",
    include_basic_info: bool = True,
    include_config: bool = True,
    include_metrics: bool = True,
    include_episodes: bool = True,
    include_video_info: bool = True,
    include_diagnostics: bool = True,
    include_runtime_index: bool = True,
    include_unavailable_metric_reasons: bool = True,
    force: bool = True,
) -> ExportOptions:
    return ExportOptions.from_payload(
        {
            "format": format,
            "template": template,
            "includeBasicInfo": include_basic_info,
            "includeConfig": include_config,
            "includeMetrics": include_metrics,
            "includeEpisodes": include_episodes,
            "includeVideoInfo": include_video_info,
            "includeDiagnostics": include_diagnostics,
            "includeRuntimeIndex": include_runtime_index,
            "includeUnavailableMetricReasons": include_unavailable_metric_reasons,
            "force": force,
        }
    )


@router.get("/jobs/{eval_job_id}/report/export")
async def export_evaluation_job_report_get(
    eval_job_id: str,
    format: str = Query(default="json", alias="format"),
    template: str = Query(default="standard"),
    includeBasicInfo: bool = Query(default=True),
    includeConfig: bool = Query(default=True),
    includeMetrics: bool = Query(default=True),
    includeEpisodes: bool = Query(default=True),
    includeVideoInfo: bool = Query(default=True),
    includeDiagnostics: bool = Query(default=True),
    includeRuntimeIndex: bool = Query(default=True),
    includeUnavailableMetricReasons: bool = Query(default=True),
    force: bool = Query(default=True),
    _: User = Depends(get_current_user),
):
    options = _resolve_export_options(
        format=format,
        template=template,
        include_basic_info=includeBasicInfo,
        include_config=includeConfig,
        include_metrics=includeMetrics,
        include_episodes=includeEpisodes,
        include_video_info=includeVideoInfo,
        include_diagnostics=includeDiagnostics,
        include_runtime_index=includeRuntimeIndex,
        include_unavailable_metric_reasons=includeUnavailableMetricReasons,
        force=force,
    )
    output_path, media_type, filename = export_evaluation_report(eval_job_id, options)
    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=filename,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.post("/jobs/{eval_job_id}/report/export")
async def export_evaluation_job_report_post(
    eval_job_id: str,
    payload: EvaluationReportExportRequest,
    _: User = Depends(get_current_user),
):
    options = ExportOptions.from_payload(payload.model_dump())
    output_path, media_type, filename = export_evaluation_report(eval_job_id, options)
    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=filename,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
