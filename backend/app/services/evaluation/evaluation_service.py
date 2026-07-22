from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.schemas.evaluation import DatasetEvaluateRequest, EvaluateAsyncRequest
from app.services.benchmark_adapters.registry import (
    list_benchmark_adapters,
    resolve_benchmark_adapter,
    resolve_benchmark_adapter_for_eval_job,
)
from app.services.evaluation.job_paths import (
    EVAL_OUTPUT_ROOT,
    eval_job_dir,
    make_eval_job_id,
    PROJECT_ROOT,
    resolve_eval_status_path,
    validate_eval_job_id,
)
from app.services.evaluation.registry import get_evaluation_adapter
from app.services.runtime_job_lifecycle import is_job_deleted, mark_job_deleted
from app.services.workspace_job_service import record_workspace_job_start, sync_workspace_job_from_runtime

TERMINAL_EVAL_STATUSES = frozenset({"completed", "failed", "canceled"})

_REPLAY_INFO_KEYS = (
    "requestedEpisodes",
    "completedEpisodes",
    "successfulEpisodes",
    "failedEpisodes",
    "successRate",
    "recordedVideoCount",
    "replayUri",
    "replayUris",
    "videoAvailable",
    "videoSourceKind",
    "evaluationMode",
    "isRepresentativeVideo",
    "warning",
    "currentEpisodeIndex",
    "recordCamera",
    "cameraFallbackUsed",
)

_EVAL_JOB_ID_RE = re.compile(
    r"^(?:ct_eval_|eval_|isaac_eval_)\d{8}_\d{6}_[a-f0-9]{4}$",
    re.IGNORECASE,
)


def _evaluation_job_roots() -> tuple[Path, ...]:
    """Configured evaluation job roots."""
    return (EVAL_OUTPUT_ROOT / "jobs",)


def _resolve_list_eval_job_id(job_dir: Path, status_data: dict[str, Any]) -> str:
    """列表项 evalJobId 必须使用真实 job id，不能误用 taskName。"""
    for candidate in (status_data.get("evalJobId"), job_dir.name):
        if isinstance(candidate, str):
            text = candidate.strip()
            if _EVAL_JOB_ID_RE.match(text):
                return text
    return job_dir.name


def _attach_workbench_basic_info(
    payload: dict[str, Any],
    *,
    eval_job_id: str,
    job_root: Path,
) -> dict[str, Any]:
    from app.services.evaluation_workbench_basic_info import attach_workbench_basic_info

    return attach_workbench_basic_info(payload, eval_job_id=eval_job_id, job_root=job_root)


def _attach_runtime_replay_info(
    payload: dict[str, Any],
    *,
    eval_job_id: str,
    job_root: Path,
) -> dict[str, Any]:
    if not job_root.is_dir():
        return payload
    if isinstance(payload.get("replayUris"), list) and payload["replayUris"]:
        return payload

    from app.services.evaluation_replay_info import (
        build_evaluation_replay_info,
        resolve_replay_api_prefix,
    )

    status_data = {}
    status_path = job_root / "status.json"
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                status_data = loaded
        except (OSError, json.JSONDecodeError):
            pass

    aggregate = {}
    for rel in ("results/aggregate_result.json", "results/eval.results.json"):
        aggregate_path = job_root / rel
        if aggregate_path.is_file():
            try:
                loaded = json.loads(aggregate_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    aggregate = loaded
                    break
            except (OSError, json.JSONDecodeError):
                pass

    replay_info = build_evaluation_replay_info(
        eval_job_id,
        job_root,
        live=status_data,
        results_data=aggregate,
        aggregate_file=aggregate,
        status_value=str(status_data.get("status") or payload.get("status") or ""),
        api_prefix=resolve_replay_api_prefix(eval_job_id),
    )
    merged = dict(payload)
    for key in _REPLAY_INFO_KEYS:
        value = replay_info.get(key)
        if value is not None:
            merged[key] = value
    return _attach_workbench_basic_info(merged, eval_job_id=eval_job_id, job_root=job_root)


def _load_legacy_request_or_404(job_root: Path) -> dict[str, Any]:
    if not job_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation job not found",
        )
    request_path = job_root / "metadata" / "evaluation_request.json"
    if not request_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation job not found",
        )
    return json.loads(request_path.read_text(encoding="utf-8"))


def start_evaluate_async(request: EvaluateAsyncRequest) -> dict[str, Any]:
    adapter = resolve_benchmark_adapter(request)
    normalized = adapter.normalize_request(request)
    return adapter.start_evaluation(request, normalized)


def start_dataset_evaluate_async(request: DatasetEvaluateRequest) -> dict[str, Any]:
    """离线数据集评测：登记 workspace job 与请求元数据（指标流水线可后续扩展）。"""
    eval_job_id = make_eval_job_id()
    job_root = eval_job_dir(eval_job_id)
    metadata_dir = job_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    request_path = metadata_dir / "dataset_evaluation_request.json"
    request_path.write_text(
        json.dumps(request.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "status.json").write_text(
        json.dumps(
            {
                "evalJobId": eval_job_id,
                "taskType": "dataset_offline",
                "evaluationMode": "dataset_offline",
                "status": "completed",
                "message": "离线数据集评测请求已登记",
                "metrics": {"selectedMetrics": request.config.metrics},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    record_workspace_job_start(
        job_id=eval_job_id,
        job_type="evaluation",
        task_type="dataset_offline",
        runtime_path=str(job_root.relative_to(PROJECT_ROOT) if job_root.is_relative_to(PROJECT_ROOT) else job_root),
        runner="dataset_offline_eval",
        task_name=f"离线数据集评测 · {request.config.datasetName or request.config.datasetId}",
        metadata={
            "evaluationType": "dataset",
            "datasetId": request.config.datasetId,
            "datasetName": request.config.datasetName,
            "metrics": request.config.metrics,
        },
        status="completed",
    )
    runtime_rel = str(job_root.relative_to(PROJECT_ROOT)) if job_root.is_relative_to(PROJECT_ROOT) else str(job_root)
    return {
        "evalJobId": eval_job_id,
        "taskType": "dataset_offline",
        "evaluationMode": "dataset_offline",
        "status": "completed",
        "runtimePath": runtime_rel,
        "statusUrl": f"/api/workspace/evaluation/jobs/{eval_job_id}/status",
        "logUrl": f"/api/workspace/evaluation/jobs/{eval_job_id}/log",
        "resultUrl": f"/api/workspace/evaluation/jobs/{eval_job_id}/result",
    }


def get_evaluation_status(eval_job_id: str) -> dict[str, Any]:
    from app.services.benchmark_adapters.base import is_ct_eval_job_id
    from app.services.eval_job_db_service import get_evaluation_job_from_db
    from app.services.training_job_sync_service import sync_eval_job_from_runtime

    candidate = (eval_job_id or "").strip()
    validated = validate_eval_job_id(candidate)

    sync_eval_job_from_runtime(validated)

    if is_ct_eval_job_id(validated):
        adapter = resolve_benchmark_adapter_for_eval_job(validated)
        if adapter is not None:
            status_payload = adapter.get_status(validated)
            if not isinstance(status_payload, dict):
                status_payload = status_payload.to_dict()
            return _attach_runtime_replay_info(
                status_payload,
                eval_job_id=validated,
                job_root=eval_job_dir(validated),
            )

    job_root = eval_job_dir(validated)

    db_row = get_evaluation_job_from_db(validated)
    if db_row and str(db_row.get("status") or "").lower() in TERMINAL_EVAL_STATUSES:
        metrics = db_row.get("metrics") if isinstance(db_row.get("metrics"), dict) else {}
        payload = {
            "evalJobId": validated,
            "taskType": db_row.get("taskType") or "",
            "evaluationMode": db_row.get("evaluationMode") or "",
            "status": db_row.get("status") or "unknown",
            "message": metrics.get("message") or "",
            "metrics": metrics,
            "artifacts": {
                "reportUri": db_row.get("reportUri"),
                "replayUri": db_row.get("replayUri"),
                "videoAvailable": db_row.get("videoAvailable"),
            },
            "updatedAt": db_row.get("updatedAt"),
            "startedAt": db_row.get("startedAt"),
            "finishedAt": db_row.get("finishedAt"),
        }
        return _attach_runtime_replay_info(payload, eval_job_id=validated, job_root=job_root)

    dataset_status_path = job_root / "metadata" / "status.json"
    if dataset_status_path.is_file():
        try:
            payload = json.loads(dataset_status_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("taskType") == "dataset_offline":
                return payload
        except (OSError, json.JSONDecodeError):
            pass

    adapter = resolve_benchmark_adapter_for_eval_job(eval_job_id)
    if adapter is not None:
        status_payload = adapter.get_status(eval_job_id)
        if not isinstance(status_payload, dict):
            status_payload = status_payload.to_dict()
        return _attach_runtime_replay_info(status_payload, eval_job_id=validated, job_root=job_root)

    sync_workspace_job_from_runtime(eval_job_id)
    job_root = eval_job_dir(eval_job_id)
    request_data = _load_legacy_request_or_404(job_root)
    legacy_adapter = get_evaluation_adapter(str(request_data.get("taskType")))
    return legacy_adapter.get_status(eval_job_id, job_root).to_dict()


def get_evaluation_result(eval_job_id: str) -> dict[str, Any]:
    from app.services.benchmark_adapters.base import is_ct_eval_job_id
    from app.services.eval_job_db_service import get_evaluation_result_from_db
    from app.services.training_job_sync_service import sync_eval_job_from_runtime

    candidate = (eval_job_id or "").strip()
    validated = validate_eval_job_id(candidate)

    if is_ct_eval_job_id(validated):
        sync_eval_job_from_runtime(validated)
        adapter = resolve_benchmark_adapter_for_eval_job(validated)
        if adapter is not None:
            return adapter.get_result(validated)

    job_root = eval_job_dir(validated)
    if job_root.is_dir():
        sync_eval_job_from_runtime(validated)

    db_result = get_evaluation_result_from_db(validated)
    if db_result:
        return _attach_runtime_replay_info(db_result, eval_job_id=validated, job_root=job_root)

    adapter = resolve_benchmark_adapter_for_eval_job(validated)
    if adapter is not None:
        result = adapter.get_result(eval_job_id)
        return _attach_runtime_replay_info(result, eval_job_id=validated, job_root=job_root)

    job_root = eval_job_dir(eval_job_id)
    request_data = _load_legacy_request_or_404(job_root)
    legacy_adapter = get_evaluation_adapter(str(request_data.get("taskType")))
    result = legacy_adapter.get_result(eval_job_id, job_root)
    return _attach_runtime_replay_info(result, eval_job_id=validated, job_root=job_root)


def read_evaluation_log_tail(eval_job_id: str, lines: int = 80) -> str:
    adapter = resolve_benchmark_adapter_for_eval_job(eval_job_id)
    if adapter is not None:
        tail = adapter.get_log(eval_job_id)
        if not tail.strip():
            return ""
        content = tail.splitlines()
        return "\n".join(content[-lines:])

    job_root = eval_job_dir(eval_job_id)
    request_data = _load_legacy_request_or_404(job_root)
    legacy_adapter = get_evaluation_adapter(str(request_data.get("taskType")))
    tail = legacy_adapter.get_log(eval_job_id, job_root)
    if not tail:
        return ""
    content = tail.splitlines()
    return "\n".join(content[-lines:])


def resolve_evaluation_video_path(eval_job_id: str, episode: Optional[int] = None) -> Optional[Path]:
    from app.services.benchmark_adapters.base import is_ct_eval_job_id
    from app.services.evaluation_replay_info import resolve_episode_video_path
    from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id, resolve_eval_job_root

    adapter = resolve_benchmark_adapter_for_eval_job(eval_job_id)
    if adapter is not None and is_ct_eval_job_id(eval_job_id) and not is_imported_workspace_eval_job_id(eval_job_id):
        return adapter.get_video(eval_job_id, episode_id=episode)

    job_root = resolve_eval_job_root(eval_job_id) or eval_job_dir(eval_job_id)
    if job_root.is_dir():
        path = resolve_episode_video_path(job_root, episode)
        if path is not None and path.is_file():
            browser = path.with_name(f"{path.stem}.browser{path.suffix}")
            if browser.is_file() and browser.stat().st_size > 0:
                return browser
            return path

    if adapter is not None:
        return adapter.get_video(eval_job_id, episode_id=episode)

    request_data = _load_legacy_request_or_404(job_root)
    legacy_adapter = get_evaluation_adapter(str(request_data.get("taskType")))
    return legacy_adapter.get_video_path(eval_job_id, job_root, episode=episode)


def list_evaluation_capabilities() -> list[dict[str, Any]]:
    return [adapter.get_capabilities().to_api_dict() for adapter in list_benchmark_adapters()]


def _eval_video_available(job_root: Path) -> bool:
    videos_dir = job_root / "videos"
    if not videos_dir.is_dir():
        return False
    return any(videos_dir.glob("*.mp4"))


def _runtime_path_relative(job_root: Path) -> str:
    try:
        return str(job_root.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(job_root)


def _read_eval_job_display_name(job_dir: Path) -> Optional[str]:
    for rel in (
        "metadata/evaluation_request.json",
        "metadata/evaluation_context.json",
        "metadata/dataset_evaluation_request.json",
    ):
        path = job_dir / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("modelName", "taskName", "name", "title"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for block_key in ("cableThreading", "dualArmCable", "modelEvaluationConfig", "config"):
            block = data.get(block_key)
            if isinstance(block, dict):
                for key in ("modelName", "taskName", "name"):
                    value = block.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    return None


def list_evaluation_jobs(*, sync_stale: bool = True) -> list[dict[str, Any]]:
    from app.services.eval_job_db_service import (
        is_valid_evaluation_list_item,
        list_evaluation_jobs_from_db,
    )
    from app.services.training_job_sync_service import reindex_runtime_jobs, sync_eval_job_from_runtime

    rows = list_evaluation_jobs_from_db(sync_stale=sync_stale)
    if rows:
        return rows

    for jobs_root in _evaluation_job_roots():
        if not jobs_root.is_dir():
            continue
        for job_dir in jobs_root.iterdir():
            if job_dir.is_dir():
                try:
                    validate_eval_job_id(job_dir.name)
                    sync_eval_job_from_runtime(job_dir.name)
                except HTTPException:
                    continue

    rows = list_evaluation_jobs_from_db(sync_stale=False)
    if rows:
        return rows

    reindex_runtime_jobs(job_type="evaluation", dry_run=False)
    rows = list_evaluation_jobs_from_db(sync_stale=False)
    if rows:
        return rows

    return _list_evaluation_jobs_from_runtime()


def _list_evaluation_jobs_from_runtime() -> list[dict[str, Any]]:
    from app.services.eval_job_db_service import is_valid_evaluation_list_item

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    job_dirs = sorted(
        (
            path
            for root in _evaluation_job_roots()
            if root.is_dir()
            for path in root.iterdir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for job_dir in job_dirs:
        if not job_dir.is_dir():
            continue
        if job_dir.name in seen:
            continue
        seen.add(job_dir.name)
        try:
            validate_eval_job_id(job_dir.name)
        except HTTPException:
            continue

        status_path = resolve_eval_status_path(job_dir)
        if not status_path.is_file():
            continue

        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(status_data, dict) or is_job_deleted(status_data):
            continue

        task_type = str(status_data.get("taskType") or "")
        metrics = status_data.get("metrics") if isinstance(status_data.get("metrics"), dict) else {}
        runner = "dataset_offline_eval" if task_type == "dataset_offline" else None
        task_name = status_data.get("taskName")
        if not task_name:
            task_name = _read_eval_job_display_name(job_dir)
        if not task_name and task_type == "dataset_offline":
            dataset_name = metrics.get("datasetName") or metrics.get("datasetId") or ""
            task_name = f"离线数据集评测 · {dataset_name}" if dataset_name else "离线数据集评测"
        if task_name:
            metrics = dict(metrics or {})
            metrics.setdefault("modelName", task_name)

        from app.services.evaluation.evaluation_type import resolve_evaluation_type_from_sources

        eval_request: dict[str, Any] = {}
        for candidate in (
            job_dir / "metadata" / "evaluation_request.json",
            job_dir / "metadata" / "evaluation_context.json",
        ):
            if not candidate.is_file():
                continue
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    nested = raw.get("evaluationRequest")
                    eval_request = nested if isinstance(nested, dict) else raw
                    break
            except (OSError, json.JSONDecodeError):
                continue

        type_resolution = resolve_evaluation_type_from_sources(
            evaluation_mode=status_data.get("evaluationMode"),
            model_asset_id=metrics.get("modelAssetId") or eval_request.get("modelAssetId"),
            model_asset_name=metrics.get("modelName") or eval_request.get("modelName"),
            dataset_id=metrics.get("datasetId") or eval_request.get("datasetId"),
            dataset_name=metrics.get("datasetName") or eval_request.get("datasetName"),
            task_type=task_type,
            runner=runner,
            task_name=task_name,
            metrics=metrics,
            evaluation_request=eval_request,
        )

        rows.append(
            {
                "evalJobId": _resolve_list_eval_job_id(job_dir, status_data),
                "jobId": _resolve_list_eval_job_id(job_dir, status_data),
                "taskType": task_type or None,
                "evaluationMode": status_data.get("evaluationMode"),
                "evaluationObject": type_resolution["evaluationObject"],
                "evaluationType": type_resolution["evaluationType"],
                "evaluationTypeLabel": type_resolution["evaluationTypeLabel"],
                "status": status_data.get("status") or "queued",
                "createdAt": status_data.get("createdAt") or status_data.get("startedAt"),
                "updatedAt": status_data.get("updatedAt"),
                "startedAt": status_data.get("startedAt"),
                "finishedAt": status_data.get("finishedAt"),
                "taskName": task_name,
                "runner": runner,
                "runtimePath": _runtime_path_relative(job_dir),
                "metrics": metrics,
                "videoAvailable": _eval_video_available(job_dir),
            }
        )
    return [row for row in rows if is_valid_evaluation_list_item(row)]


def _resolve_evaluation_job_root(eval_job_id: str) -> Optional[Path]:
    from app.services.eval_job_db_service import get_evaluation_job_from_db
    from app.services.training_job_sync_service import _resolve_eval_job_dir

    runtime_path: Optional[str] = None
    db_item = get_evaluation_job_from_db(eval_job_id)
    if db_item:
        runtime_path = db_item.get("runtimePath")

    resolved = _resolve_eval_job_dir(eval_job_id, runtime_path)
    if resolved is not None:
        return resolved

    legacy_root = eval_job_dir(eval_job_id)
    if legacy_root.is_dir():
        return legacy_root
    return None


def _mark_evaluation_runtime_deleted(job_root: Path) -> Optional[str]:
    deleted_at: Optional[str] = None
    candidates = [
        job_root / "live" / "status.json",
        job_root / "status.json",
        job_root / "metadata" / "status.json",
    ]
    for status_path in candidates:
        if status_path.is_file():
            payload = mark_job_deleted(status_path)
            deleted_at = str(payload.get("deletedAt") or deleted_at or "")
    return deleted_at


def _read_eval_status_payload(job_root: Path) -> dict[str, Any]:
    for candidate in (
        job_root / "live" / "status.json",
        job_root / "status.json",
        job_root / "metadata" / "status.json",
    ):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def delete_evaluation_job(eval_job_id: str) -> dict[str, Any]:
    from app.services.eval_job_db_service import (
        delete_evaluation_job_from_db,
        get_evaluation_job_from_db,
    )

    validated = validate_eval_job_id(eval_job_id)

    from app.core.database import SessionLocal
    from app.models.workspace_job import WorkspaceJob

    db_exists = False
    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == validated).one_or_none()
            if row is not None:
                if row.job_type != "evaluation":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="只能删除评测任务",
                    )
                if row.status != "deleted":
                    db_exists = True
    except HTTPException:
        raise
    except Exception:
        pass

    if not db_exists:
        db_exists = get_evaluation_job_from_db(validated) is not None

    job_root = _resolve_evaluation_job_root(validated)

    if job_root is None and not db_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation job not found",
        )

    warning: Optional[str] = None
    deleted_at: Optional[str] = None

    if job_root is not None and job_root.is_dir():
        status_payload = _read_eval_status_payload(job_root)
        raw_status = str(status_payload.get("status") or "").lower()
        if raw_status in {"running", "queued", "pending"}:
            warning = "任务可能仍在后台运行，已从评测中心移除"
        deleted_at = _mark_evaluation_runtime_deleted(job_root)

    deleted_in_db = delete_evaluation_job_from_db(validated)
    if not deleted_in_db and not db_exists and (job_root is None or not job_root.is_dir()):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation job not found or not an evaluation task",
        )

    result: dict[str, Any] = {
        "success": True,
        "evalJobId": validated,
        "deleted": True,
        "deletedAt": deleted_at,
    }
    if warning:
        result["warning"] = warning
    return result


def batch_delete_evaluation_jobs(
    eval_job_ids: list[str],
    *,
    workspace_job_ids: list[int] | None = None,
) -> dict[str, Any]:
    deleted: list[str] = []
    deleted_records: list[int] = []
    failed: list[dict[str, Any]] = []
    warnings: list[str] = []

    for raw_id in eval_job_ids:
        job_id = (raw_id or "").strip()
        if not job_id:
            continue
        try:
            result = delete_evaluation_job(job_id)
            deleted.append(job_id)
            warning = result.get("warning")
            if isinstance(warning, str) and warning:
                warnings.append(f"{job_id}: {warning}")
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            failed.append({"evalJobId": job_id, "reason": detail})
        except Exception as exc:
            failed.append({"evalJobId": job_id, "reason": str(exc)})

    from app.services.eval_job_db_service import delete_pending_evaluation_record

    for raw_id in workspace_job_ids or []:
        try:
            record_id = int(raw_id)
        except (TypeError, ValueError):
            failed.append({"workspaceJobId": raw_id, "reason": "workspaceJobId 无效"})
            continue
        try:
            delete_pending_evaluation_record(record_id)
            deleted_records.append(record_id)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            failed.append({"workspaceJobId": record_id, "reason": detail})
        except Exception as exc:
            failed.append({"workspaceJobId": record_id, "reason": str(exc)})

    payload: dict[str, Any] = {
        "success": len(failed) == 0,
        "deletedCount": len(deleted) + len(deleted_records),
        "deleted": deleted,
        "deletedRecordIds": deleted_records,
        "failed": failed,
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def delete_pending_evaluation_record(workspace_job_id: int | str) -> dict[str, Any]:
    from app.services.eval_job_db_service import delete_pending_evaluation_record as _delete

    return _delete(workspace_job_id)
