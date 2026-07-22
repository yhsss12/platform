
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request

from app.core.deps import get_current_user
from app.core.label_conversion_access import (
    assert_conversion_analyze_allowed_sync,
    assert_conversion_job_in_scope_sync,
    assert_conversion_manage_or_execute_sync,
    assert_platform_task_manage_project_sync,
    assert_platform_task_execute_project_sync,
    scoped_project_ids_sync,
)
from app.models.user import User
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func, text
from app.services.mcap_converter import convert_mcap_to_hdf5, analyze_mcap_frequency
from app.services.lerobot_service import convert_mcap_to_lerobot_task
from app.models.data_asset import DataAsset, ConversionJobAsset, ConversionBatchJob, TaskJob
from app.services.conversion_batch_service import recompute_conversion_batch_stats
from app.services.asset_registration_service import upsert_converted_asset, DataAssetsSyncSessionLocal
from app.models.project_asset import Project
from app.services.minio_service import delete_by_minio_uri, download_by_minio_uri, MinioBucketError
from app.db.data_assets_session import DATA_ASSETS_ROOT
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple, Set
import anyio
import uuid
from datetime import datetime, timedelta
import os
import shutil
import logging
import json
import yaml
from pathlib import Path
import threading
from queue import Queue, Empty
from app.api.routes_fs import validate_path_whitelist
from app.schemas.common import ApiResponse
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import log_audit_safe, sync_resolve_user_for_audit, request_client_ip, request_user_agent
from app.services.dispatcher import RedisDispatchError, dispatch_task
from app.services.queue_backpressure import enforce_queue_dispatch_allowed
from app.services.task_profiles import get_task_profile
import asyncio
import time
import tempfile
from app.services.task_job_store import delete_task_job, is_cancelled, update_task_status
from app.services.task_queue import redis_conn
from rq.job import Job
from rq.exceptions import NoSuchJobError

# Set up logging
logger = logging.getLogger(__name__)
DEFAULT_GRID_FPS = float(os.getenv("CONVERSION_DEFAULT_GRID_FPS", "15"))

router = APIRouter()

# 无 batch_id 的旧单文件任务在列表接口中的合成父任务 ID 前缀（避免与真实 UUID 冲突）
LEGACY_SINGLE_BATCH_PREFIX = "legacy-job-"


def _cleanup_local_conversion_output(path: str) -> None:
    """上传 MinIO 成功后删除本地中转产物，最终仅保留 MinIO 数据。"""
    p = (path or "").strip()
    if not p or p.startswith("minio://"):
        return
    try:
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=False)
    except FileNotFoundError:
        return
    except Exception as e:
        logger.warning(f"Cleanup local conversion output failed: {p}, error={e}")

# 当数据资产登记的是“目录”（如采集产物为 episode_xxx/），转换实际需要目录内的 .mcap 文件。
def _resolve_mcap_input_path(path: str) -> str:
    p = os.path.normpath((path or "").strip())
    if not p:
        return p
    if os.path.isfile(p):
        return p
    if not os.path.isdir(p):
        return p
    best = None
    best_size = -1
    try:
        with os.scandir(p) as it:
            for entry in it:
                try:
                    if not entry.is_file():
                        continue
                    name = (entry.name or "").lower()
                    if not name.endswith(".mcap"):
                        continue
                    st = entry.stat()
                    sz = int(getattr(st, "st_size", 0) or 0)
                    if sz > best_size:
                        best_size = sz
                        best = entry.path
                except Exception:
                    continue
    except Exception:
        return p
    return os.path.normpath(best) if best else p

# --- Models ---

class ConversionJobStageProgress(BaseModel):
    stage: str
    status: str  # 'pending' | 'running' | 'done' | 'error'
    progressPercent: float
    durationMs: Optional[float] = None

class ConversionJob(BaseModel):
    jobId: str
    shortCode: str
    taskNo: str
    taskName: Optional[str] = None
    outputFileName: str
    fileName: str
    assetId: str
    assetName: str
    projectId: str
    projectName: str
    deviceName: str
    outputFormat: str
    fileFormat: str
    outputLocation: str
    outputPath: str
    status: str  # 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled'
    progressPercent: float
    currentStage: Optional[str] = None
    stages: List[ConversionJobStageProgress]
    logs: List[str]
    createdAt: str
    updatedAt: str
    artifactReady: bool
    errorMessage: Optional[str] = None
    operatorName: Optional[str] = None
    batchId: Optional[str] = None

class CreateConversionInput(BaseModel):
    projectId: str
    inputDatasetId: str
    outputFormat: str
    outputLocation: str
    taskName: Optional[str] = None
    outputFileName: Optional[str] = None
    # 可为空：空则后端生成 MinIO 目录前缀（与单文件默认规则一致），无需用户选择输出目录
    outputPath: str = ""
    # assetId is deprecated but might be sent by older clients
    assetId: Optional[str] = None
    frequency: Optional[float] = None  # User-specified sampling frequency (Hz)
    topics: Optional[List[str]] = None


class McapAnalysisResult(BaseModel):
    topic: str
    count: int
    frequency: float
    period_ms: float
    min_delta_ms: float
    max_delta_ms: float

class LeRobotConversionInput(BaseModel):
    projectId: str
    inputDatasetId: str
    outputRepoId: str
    # Flexible config dictionary containing 'topics', 'alignment', 'lerobot'
    config: Dict[str, Any]


class CreateConversionBatchInput(BaseModel):
    """一次提交多条子任务输入；可选父级任务名。"""

    taskName: Optional[str] = None
    items: List[CreateConversionInput]


class ConversionBatchSummary(BaseModel):
    batchId: str
    taskName: Optional[str] = None
    targetFormat: str
    projectId: str
    projectName: str
    # creatorId / creatorName：父任务创建人；旧单文件任务的 creatorName 来自 conversion_jobs.operator_name
    creatorId: Optional[str] = None
    creatorName: Optional[str] = None
    totalCount: int
    successCount: int
    failedCount: int
    runningCount: int
    pendingCount: int
    progressPercent: float
    overallStatus: str
    createdAt: str
    updatedAt: str
    legacySingleFile: bool = False


class ConversionBatchChildItem(BaseModel):
    jobId: str
    sourceFileName: str
    outputFileName: str
    itemStatus: str
    itemStage: Optional[str] = None
    errorMessage: Optional[str] = None
    createdAt: str
    updatedAt: str


class ConversionBatchDetail(BaseModel):
    batch: ConversionBatchSummary
    children: List[ConversionBatchChildItem]


# --- In-Memory Store ---

# Store jobs in memory for now. 
# Key: jobId, Value: ConversionJob
jobs_store: Dict[str, ConversionJob] = {}

_schema_checked = False
def _ensure_conversion_jobs_schema() -> None:
    """Best-effort: add new columns without requiring manual migration."""
    global _schema_checked
    if _schema_checked:
        return
    _schema_checked = True
    try:
        # PostgreSQL: add column if missing
        from app.services.asset_registration_service import data_assets_sync_engine
        with data_assets_sync_engine.begin() as conn:
            conn.execute(text("ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS task_name VARCHAR(256)"))
            conn.execute(text("ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS operator_name VARCHAR(256)"))
            conn.execute(text("ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS batch_id VARCHAR(64)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conversion_jobs_batch_id ON conversion_jobs (batch_id)"))
            # conversion_batch_jobs.canceled_count：与 NOT NULL 表结构对齐（迁移外兜底）
            conn.execute(
                text(
                    "ALTER TABLE conversion_batch_jobs ADD COLUMN IF NOT EXISTS canceled_count INTEGER DEFAULT 0"
                )
            )
            conn.execute(text("UPDATE conversion_batch_jobs SET canceled_count = 0 WHERE canceled_count IS NULL"))
            conn.execute(text("ALTER TABLE conversion_batch_jobs ALTER COLUMN canceled_count SET DEFAULT 0"))
            conn.execute(text("ALTER TABLE conversion_batch_jobs ALTER COLUMN canceled_count SET NOT NULL"))
    except Exception:
        # ignore migration errors (e.g. permissions); app can still run without the field
        pass

MAX_CONVERSION_CONCURRENCY = max(1, int(os.getenv("CONVERSION_MAX_CONCURRENCY", "1")))
_conversion_semaphore = threading.Semaphore(MAX_CONVERSION_CONCURRENCY)
_asset_locks: Dict[str, threading.Lock] = {}
_asset_locks_guard = threading.Lock()
_task_no_lock = threading.Lock()

CONVERSION_QUEUE_WORKERS = max(1, int(os.getenv("CONVERSION_QUEUE_WORKERS", "1")))
CONVERSION_QUEUE_MAX_SIZE = max(1, int(os.getenv("CONVERSION_QUEUE_MAX_SIZE", "200")))
_conversion_job_queue: "Queue[tuple[str, callable]]" = Queue()
_conversion_workers_started = False
_conversion_workers_guard = threading.Lock()

def _get_asset_lock(asset_id: str) -> threading.Lock:
    with _asset_locks_guard:
        lock = _asset_locks.get(asset_id)
        if lock is None:
            lock = threading.Lock()
            _asset_locks[asset_id] = lock
        return lock


def _next_task_no_from_db() -> str:
    session = DataAssetsSyncSessionLocal()
    try:
        rec = session.query(func.max(ConversionJobAsset.id)).one_or_none()
        max_id = int((rec[0] if rec else 0) or 0)
        return f"{max_id + 1:04d}"
    finally:
        session.close()

# --- Helper Functions & DB Setup ---

def generate_task_no() -> str:
    # 以数据库自增主键近似生成展示序号，并加锁避免并发重复
    with _task_no_lock:
        return _next_task_no_from_db()


def _conversion_worker_loop() -> None:
    while True:
        try:
            _, task = _conversion_job_queue.get(timeout=1)
        except Empty:
            continue
        try:
            task()
        except Exception:
            logger.exception("conversion queue worker task failed")
        finally:
            _conversion_job_queue.task_done()


def _ensure_conversion_workers() -> None:
    global _conversion_workers_started
    if _conversion_workers_started:
        return
    with _conversion_workers_guard:
        if _conversion_workers_started:
            return
        for i in range(CONVERSION_QUEUE_WORKERS):
            t = threading.Thread(
                target=_conversion_worker_loop,
                name=f"conversion-queue-worker-{i}",
                daemon=True,
            )
            t.start()
        _conversion_workers_started = True


def _enqueue_conversion_task(task_name: str, task_fn) -> None:
    _ensure_conversion_workers()
    if _conversion_job_queue.qsize() >= CONVERSION_QUEUE_MAX_SIZE:
        raise HTTPException(
            status_code=429,
            detail=f"转换任务过多，请稍后重试（队列上限 {CONVERSION_QUEUE_MAX_SIZE}）",
        )
    _conversion_job_queue.put((task_name, task_fn))

def get_dataset_info(dataset_id: str) -> DataAsset:
    """
    根据 inputDatasetId 从数据资产表（PostgreSQL）中读取输入资产信息。
    目前转换输入统一使用 DataAsset 作为来源。
    """
    session = DataAssetsSyncSessionLocal()
    try:
        asset = session.query(DataAsset).filter(DataAsset.id == int(dataset_id)).one_or_none()
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset with ID {dataset_id} not found")
        return asset
    finally:
        session.close()

def _is_minio_uri(raw: str) -> bool:
    return (raw or "").strip().startswith("minio://")


def _ensure_input_mcap_exists_or_404(path: str, *, dataset_id: Optional[str] = None) -> str:
    """
    将 MCAP 输入路径解析为本地可读文件路径。

    - dataset.file_path 可能是本地绝对路径，也可能是 minio://bucket/xxx（同步后会覆盖 file_path）。
    - 统一在本地缓存后再执行 _resolve_mcap_input_path（支持目录型资产，自动选取目录中最大的 .mcap）。
    """
    p = (path or "").strip()
    if not p:
        raise HTTPException(status_code=404, detail="Dataset file not found")

    local_candidate = p
    if _is_minio_uri(p):
        minio_uri = p
        cache_root = DATA_ASSETS_ROOT / "_minio_conversion_cache" / (str(dataset_id or "unknown").strip() or "unknown")
        try:
            local_candidate = download_by_minio_uri(minio_uri, str(cache_root))
        except MinioBucketError:
            raise HTTPException(status_code=404, detail="Dataset file not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    resolved = _resolve_mcap_input_path(local_candidate)
    if not resolved or not os.path.exists(resolved):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    if os.path.isdir(resolved):
        raise HTTPException(status_code=404, detail="Dataset is a directory without any .mcap file")
    return resolved


def resolve_executor_conversion_mcap_path(task_params: Dict[str, Any]) -> str:
    """
    供 RQ worker / 与创建任务的 API 不在同一文件系统时使用。

    API 在创建任务时会把 MinIO 资源下载到 API 进程本机路径并写入 mcap_path；独立 worker 容器内
    该路径往往不存在。此时应使用任务载荷中的 DB 原始 file_path（input_dataset_file_path）在 worker
    本机重新下载或解析。宿主机脚本（restart.sh）同机同盘时 mcap_path 通常已存在，则直接沿用，避免重复拉取。
    """
    p = task_params or {}
    mcap_path = (p.get("mcap_path") or "").strip()
    raw_fp = (p.get("input_dataset_file_path") or "").strip()
    ds_id = ((p.get("input_dataset_id") or "").strip() or None)

    if mcap_path:
        if os.path.isfile(mcap_path):
            return mcap_path
        if os.path.isdir(mcap_path):
            resolved = _resolve_mcap_input_path(mcap_path)
            if resolved and os.path.isfile(resolved):
                return resolved

    if raw_fp:
        return _ensure_input_mcap_exists_or_404(raw_fp, dataset_id=ds_id)

    if mcap_path:
        return _ensure_input_mcap_exists_or_404(mcap_path, dataset_id=ds_id)

    raise ValueError("conversion task missing mcap_path and input_dataset_file_path")


def _assert_project_writable_sync(project_id: str) -> None:
    pid = (project_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="projectId is required")
    session = DataAssetsSyncSessionLocal()
    try:
        p = session.query(Project).filter(Project.id == pid).one_or_none()
        if not p:
            raise HTTPException(status_code=404, detail="Project not found")
        if (p.status or "").strip() == "已归档":
            raise HTTPException(status_code=403, detail="Project is archived")
    finally:
        session.close()

def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

def _persist_conversion_job(job: ConversionJob, *, dataset: Optional[DataAsset] = None) -> None:
    _ensure_conversion_jobs_schema()
    session = DataAssetsSyncSessionLocal()
    try:
        rec = session.query(ConversionJobAsset).filter(ConversionJobAsset.job_id == job.jobId).one_or_none()
        if rec is None:
            rec = ConversionJobAsset(job_id=job.jobId)
            created_at = _parse_dt(getattr(job, "createdAt", None))
            if created_at:
                rec.created_at = created_at
        rec.short_code = getattr(job, "shortCode", None)
        rec.task_no = getattr(job, "taskNo", None)
        rec.task_name = getattr(job, "taskName", None)

        rec.input_dataset_id = str(getattr(job, "assetId", None) or "") or None
        rec.input_asset_name = getattr(job, "assetName", None)
        if dataset is not None:
            rec.input_file_path = getattr(dataset, "file_path", None)

        rec.project_id = getattr(job, "projectId", None)
        rec.project_name = getattr(job, "projectName", None)
        rec.device_name = getattr(job, "deviceName", None)

        rec.output_format = getattr(job, "outputFormat", None)
        rec.file_format = getattr(job, "fileFormat", None)
        rec.output_location = getattr(job, "outputLocation", None)
        rec.output_file_name = getattr(job, "outputFileName", None)
        rec.output_path = getattr(job, "outputPath", None)

        rec.status = getattr(job, "status", None) or rec.status
        rec.progress_percent = float(getattr(job, "progressPercent", 0) or 0)
        rec.current_stage = getattr(job, "currentStage", None)
        rec.artifact_ready = bool(getattr(job, "artifactReady", False))
        rec.error_message = getattr(job, "errorMessage", None)
        rec.operator_name = (
            (getattr(job, "operatorName", None) or getattr(job, "operator_name", None) or "").strip() or None
        )
        bid = (getattr(job, "batchId", None) or getattr(job, "batch_id", None) or "").strip() or None
        rec.batch_id = bid

        try:
            stages = getattr(job, "stages", None) or []
            rec.stages_json = json.dumps(
                [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in stages],
                ensure_ascii=False,
            )
        except Exception:
            rec.stages_json = rec.stages_json
        try:
            logs = getattr(job, "logs", None) or []
            rec.logs_json = json.dumps(list(logs), ensure_ascii=False)
        except Exception:
            rec.logs_json = rec.logs_json

        updated_at = _parse_dt(getattr(job, "updatedAt", None))
        if updated_at:
            rec.updated_at = updated_at

        session.add(rec)
        session.commit()
        if bid:
            recompute_conversion_batch_stats(bid)
    except Exception:
        session.rollback()
    finally:
        session.close()

def _record_to_job(rec: ConversionJobAsset) -> ConversionJob:
    try:
        stages = json.loads(rec.stages_json) if rec.stages_json else []
    except Exception:
        stages = []
    try:
        logs = json.loads(rec.logs_json) if rec.logs_json else []
    except Exception:
        logs = []
    stages_objs: List[ConversionJobStageProgress] = []
    for s in stages or []:
        try:
            stages_objs.append(ConversionJobStageProgress(**s))
        except Exception:
            continue
    return ConversionJob(
        jobId=rec.job_id,
        shortCode=rec.short_code or (rec.job_id[:8] if rec.job_id else ""),
        taskNo=rec.task_no or "",
        taskName=getattr(rec, "task_name", None),
        outputFileName=rec.output_file_name or "",
        fileName=rec.output_file_name or "",
        assetId=rec.input_dataset_id or "",
        assetName=rec.input_asset_name or "",
        projectId=rec.project_id or "",
        projectName=rec.project_name or (rec.project_id or "Unknown Project"),
        deviceName=rec.device_name or "Unknown Device",
        outputFormat=rec.output_format or "",
        fileFormat=rec.file_format or "",
        outputLocation=rec.output_location or "",
        outputPath=rec.output_path or "",
        status=rec.status or "queued",
        progressPercent=float(rec.progress_percent or 0),
        currentStage=rec.current_stage,
        stages=stages_objs,
        logs=list(logs or []),
        createdAt=(rec.created_at.isoformat() if rec.created_at else ""),
        updatedAt=(rec.updated_at.isoformat() if rec.updated_at else ""),
        artifactReady=bool(rec.artifact_ready),
        errorMessage=rec.error_message,
        operatorName=(getattr(rec, "operator_name", None) or "").strip() or None,
        batchId=(getattr(rec, "batch_id", None) or None),
    )

def _load_jobs_from_db() -> List[ConversionJobAsset]:
    session = DataAssetsSyncSessionLocal()
    try:
        return list(session.query(ConversionJobAsset).order_by(ConversionJobAsset.created_at.asc()).all())
    finally:
        session.close()

def _get_job_from_db(job_id: str) -> Optional[ConversionJobAsset]:
    session = DataAssetsSyncSessionLocal()
    try:
        return session.query(ConversionJobAsset).filter(ConversionJobAsset.job_id == job_id).one_or_none()
    finally:
        session.close()

def _delete_job_from_db(job_id: str) -> bool:
    session = DataAssetsSyncSessionLocal()
    try:
        rec = session.query(ConversionJobAsset).filter(ConversionJobAsset.job_id == job_id).one_or_none()
        if not rec:
            return False
        session.delete(rec)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


def run_conversion_task(job_id: str, mcap_path: str, output_path: str, config: Dict[str, Any]):
    job = jobs_store.get(job_id)
    if not job:
        return

    def _apply_cancelled_state(msg: str = "Task cancelled") -> None:
        job.status = "canceled"
        job.errorMessage = msg
        job.logs.append(f"[{datetime.now().isoformat()}] {msg}")
        job.updatedAt = datetime.now().isoformat()
        _persist_conversion_job(job)

    asset_lock = None
    lock_held = False
    sem_held = False
    try:
        if is_cancelled(job_id):
            _apply_cancelled_state()
            return

        asset_id = str(getattr(job, "assetId", "") or "")
        if asset_id:
            asset_lock = _get_asset_lock(asset_id)
            job.logs.append(f"[{datetime.now().isoformat()}] Waiting for same-dataset lock (assetId={asset_id}) ...")
            job.updatedAt = datetime.now().isoformat()
            while True:
                if is_cancelled(job_id):
                    _apply_cancelled_state()
                    return
                if asset_lock.acquire(blocking=False):
                    lock_held = True
                    break
                time.sleep(0.2)
            job.logs.append(f"[{datetime.now().isoformat()}] Same-dataset lock acquired (assetId={asset_id})")
            job.updatedAt = datetime.now().isoformat()

        if MAX_CONVERSION_CONCURRENCY > 0:
            job.logs.append(f"[{datetime.now().isoformat()}] Waiting for resources (max concurrency={MAX_CONVERSION_CONCURRENCY}) ...")
            job.updatedAt = datetime.now().isoformat()
            while True:
                if is_cancelled(job_id):
                    if lock_held and asset_lock is not None:
                        try:
                            asset_lock.release()
                        except Exception:
                            pass
                        lock_held = False
                    _apply_cancelled_state()
                    return
                if _conversion_semaphore.acquire(timeout=1.0):
                    sem_held = True
                    break
            job.logs.append(f"[{datetime.now().isoformat()}] Resource slot acquired, starting ...")

        # Update status to running
        job.status = "running"
        job.currentStage = "Parse"
        job.progressPercent = 10
        job.updatedAt = datetime.now().isoformat()
        job.logs.append(f"[{datetime.now().isoformat()}] Starting conversion...")
        _persist_conversion_job(job)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Log paths
        job.logs.append(f"Input: {mcap_path}")
        job.logs.append(f"Output: {output_path}")

        last_persist_at = 0.0
        def _progress_cb(stage: str, percent: float) -> None:
            if is_cancelled(job_id):
                raise RuntimeError("Task cancelled")
            job.currentStage = stage
            job.progressPercent = min(99, int(percent))
            job.updatedAt = datetime.now().isoformat()
            nonlocal last_persist_at
            now = time.monotonic()
            if now - last_persist_at >= 0.8:
                last_persist_at = now
                _persist_conversion_job(job)

        success, conversion_error = convert_mcap_to_hdf5(
            mcap_path=mcap_path,
            output_path=output_path,
            config=config,
            progress_callback=_progress_cb,
        )

        if success:
            if is_cancelled(job_id):
                job.status = "canceled"
                job.errorMessage = "Task cancelled"
                _persist_conversion_job(job)
                return
            job.currentStage = "Validate"
            job.progressPercent = 95
            job.logs.append(f"[{datetime.now().isoformat()}] Converting finished, registering output asset...")
            job.updatedAt = datetime.now().isoformat()

            try:
                upsert_converted_asset(job, mcap_path, output_path)
                _cleanup_local_conversion_output(output_path)
            except Exception as e:
                job.status = "failed"
                job.errorMessage = f"Asset registration failed: {e}"
                job.logs.append(f"[{datetime.now().isoformat()}] Error: Asset registration failed: {e}")
                job.updatedAt = datetime.now().isoformat()
                return

            job.status = "succeeded"
            job.progressPercent = 100
            job.artifactReady = True
            job.logs.append(f"[{datetime.now().isoformat()}] Conversion completed successfully.")
            for stage in job.stages:
                stage.status = "done"
                stage.progressPercent = 100
            _persist_conversion_job(job)
        else:
            job.status = "failed"
            job.errorMessage = conversion_error or "转换失败（未返回具体原因）"
            job.logs.append(f"[{datetime.now().isoformat()}] Conversion failed: {job.errorMessage}")
            _persist_conversion_job(job)

    except Exception as e:
        logger.exception(f"Conversion failed for job {job_id}")
        if is_cancelled(job_id):
            job.status = "canceled"
            job.errorMessage = "Task cancelled"
            job.logs.append(f"[{datetime.now().isoformat()}] Task cancelled")
        else:
            job.status = "failed"
            job.errorMessage = str(e)
            job.logs.append(f"[{datetime.now().isoformat()}] Error: {str(e)}")
        _persist_conversion_job(job)
    finally:
        try:
            if sem_held and MAX_CONVERSION_CONCURRENCY > 0:
                _conversion_semaphore.release()
        except Exception:
            pass
        try:
            if lock_held and asset_lock is not None:
                asset_lock.release()
        except Exception:
            pass
        job.updatedAt = datetime.now().isoformat()
        _persist_conversion_job(job)

async def run_lerobot_with_limit(
    job_id: str,
    mcap_path: str,
    output_repo_id: str,
    config: Dict[str, Any],
    jobs_store: Dict[str, Any],
):
    job = jobs_store.get(job_id)
    if is_cancelled(job_id):
        if job:
            job.status = "canceled"
            job.errorMessage = "Task cancelled"
            _persist_conversion_job(job)
        return
    asset_lock = None
    if job:
        asset_id = str(getattr(job, "assetId", "") or "")
        if asset_id:
            asset_lock = _get_asset_lock(asset_id)
            job.logs.append(f"[{datetime.now().isoformat()}] Waiting for same-dataset lock (assetId={asset_id}) ...")
            job.updatedAt = datetime.now().isoformat()
        job.logs.append(f"[{datetime.now().isoformat()}] Waiting for resources (max concurrency={MAX_CONVERSION_CONCURRENCY}) ...")
        job.updatedAt = datetime.now().isoformat()
    loop = asyncio.get_running_loop()
    if asset_lock is not None:
        await loop.run_in_executor(None, asset_lock.acquire)
        if job:
            job.logs.append(f"[{datetime.now().isoformat()}] Same-dataset lock acquired")
            job.updatedAt = datetime.now().isoformat()
    await loop.run_in_executor(None, _conversion_semaphore.acquire)
    try:
        await convert_mcap_to_lerobot_task(
            job_id=job_id,
            mcap_path=mcap_path,
            output_repo_id=output_repo_id,
            config=config,
            jobs_store=jobs_store,
        )
    finally:
        try:
            _conversion_semaphore.release()
        except Exception:
            pass
        try:
            if asset_lock is not None:
                asset_lock.release()
        except Exception:
            pass
    if is_cancelled(job_id) and job:
        job.status = "canceled"
        job.errorMessage = "Task cancelled"
        _persist_conversion_job(job)

# --- Endpoints ---

@router.get("/analyze", response_model=List[McapAnalysisResult])
def analyze_dataset_frequency(
    datasetId: str,
    current_user: User = Depends(get_current_user),
):
    """
    Analyze the frequency statistics of topics in an MCAP dataset.
    """
    dataset = get_dataset_info(datasetId)
    assert_conversion_analyze_allowed_sync(
        current_user,
        dataset_project_id=getattr(dataset, "project_id", None),
    )
    if (dataset.format or "").lower() != "mcap":
        raise HTTPException(status_code=400, detail="Only MCAP format is supported for analysis")
    
    input_path = _ensure_input_mcap_exists_or_404(dataset.file_path, dataset_id=datasetId)
        
    try:
        results = analyze_mcap_frequency(input_path)
        return results
    except Exception as e:
        logger.exception(f"Failed to analyze dataset {datasetId}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _create_conversion_job_impl(
    input_data: CreateConversionInput,
    request: Request,
    current_user: User,
    *,
    batch_id: Optional[str] = None,
) -> ConversionJob:
    assert_conversion_manage_or_execute_sync(current_user)
    _ensure_conversion_jobs_schema()
    # Ensure new task-name columns exist in data_assets for binding
    try:
        from app.services.asset_registration_service import data_assets_sync_engine
        with data_assets_sync_engine.begin() as conn:
            conn.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS conversion_task_name VARCHAR(256)"))
    except Exception:
        pass
    # Resolve dataset（从数据资产表 backend/data/assets 读取输入资产）
    dataset_id = input_data.inputDatasetId or input_data.assetId
    if not dataset_id:
        raise HTTPException(status_code=400, detail="inputDatasetId is required")
    
    dataset = get_dataset_info(dataset_id)
    if (getattr(dataset, "sync_status", "synced") or "synced").strip().lower() != "synced":
        raise HTTPException(status_code=409, detail="该数据尚未同步，暂不可转换")
    # 目前转换仅支持 MCAP → HDF5 / LeRobot
    if (dataset.format or "").lower() != "mcap":
        raise HTTPException(status_code=400, detail="当前仅支持 MCAP 格式数据的转换")

    input_mcap_path = _ensure_input_mcap_exists_or_404(dataset.file_path, dataset_id=dataset_id)

    output_path_prefix = (input_data.outputPath or "").strip()
    output_location_eff = (input_data.outputLocation or "local").strip()
    if not output_path_prefix:
        ts_ms = int(time.time() * 1000)
        proj_key = (dataset.project_id or input_data.projectId or "").strip() or "unknown"
        output_path_prefix = os.path.join(
            tempfile.gettempdir(),
            "eai-conversion",
            "project",
            proj_key,
            "assets",
            str(dataset_id),
            "converted",
            str(ts_ms),
        )
        output_location_eff = "cloud"
    
    # Determine default output filename if not provided
    if not input_data.outputFileName:
        base_name = os.path.splitext(dataset.filename)[0]
        if input_data.outputFormat == "LeRobot":
             input_data.outputFileName = base_name
        else:
             input_data.outputFileName = f"{base_name}.hdf5"

    # Create Job ID
    job_id = str(uuid.uuid4())
    short_code = job_id[:8]
    task_no = generate_task_no()
    
    # Initialize stages
    stages = [
        ConversionJobStageProgress(stage="Parse", status="pending", progressPercent=0),
        ConversionJobStageProgress(stage="Align", status="pending", progressPercent=0),
        ConversionJobStageProgress(stage="Write", status="pending", progressPercent=0),
        ConversionJobStageProgress(stage="Validate", status="pending", progressPercent=0),
    ]

    # Create Job Object
    job = ConversionJob(
        jobId=job_id,
        shortCode=short_code,
        taskNo=task_no,
        taskName=(input_data.taskName.strip() if input_data.taskName else None),
        outputFileName=input_data.outputFileName,
        fileName=input_data.outputFileName, # fallback
        assetId=dataset_id,
        assetName=dataset.filename,
        projectId=dataset.project_id or input_data.projectId,
        projectName=dataset.project_name or dataset.project_id or input_data.projectId or "Unknown Project",
        deviceName="Unknown Device",
        outputFormat=input_data.outputFormat,
        fileFormat=(dataset.format or "mcap").upper(),
        outputLocation=output_location_eff,
        outputPath=output_path_prefix,
        status="queued",
        progressPercent=0,
        currentStage=None,
        stages=stages,
        logs=[],
        createdAt=datetime.now().isoformat(),
        updatedAt=datetime.now().isoformat(),
        artifactReady=False,
        operatorName=(getattr(current_user, "username", None) or "").strip() or None,
        batchId=batch_id,
    )
    _assert_project_writable_sync(job.projectId)
    assert_platform_task_execute_project_sync(current_user, job.projectId)

    enforce_queue_dispatch_allowed(get_task_profile("conversion").queue)

    # Save to store
    jobs_store[job_id] = job
    _persist_conversion_job(job, dataset=dataset)

    uid, uname, role = sync_resolve_user_for_audit(request)
    ip = request_client_ip(request)
    ua = request_user_agent(request)
    log_audit_safe(
        user_id=uid,
        username=uname,
        role=role,
        action_type=AA.CREATE_TASK,
        project_id=job.projectId,
        project_name=job.projectName,
        resource_type=AR.CONVERT_JOB,
        resource_id=job_id,
        resource_name=input_data.taskName or job.outputFileName,
        ip=ip,
        user_agent=ua,
        detail_json={"outputFormat": input_data.outputFormat, "assetId": dataset_id, "domain": "convert"},
    )

    # 转换任务名称只写在「输出」资产上，由转换完成后 upsert_converted_asset 写入，不写回输入资产
    if input_data.outputFormat == "LeRobot":
        # Remove .zip extension if present, as LeRobot dataset is a directory structure
        if input_data.outputFileName and input_data.outputFileName.lower().endswith('.zip'):
             input_data.outputFileName = input_data.outputFileName[:-4]

        # Load default configs
        # Assuming scripts/relman is at project root
        project_root = Path(__file__).resolve().parents[3]
        relman_path = project_root / "scripts" / "relman"
        
        # Load aloha config for topics
        with open(relman_path / "config_aloha.yaml", "r") as f:
            aloha_config = yaml.safe_load(f)
            
        # Load lerobot config
        lerobot_config = {}
        try:
            with open(relman_path / "config_lerobot.yaml", "r") as f:
                lerobot_config = yaml.safe_load(f) or {}
        except Exception:
            lerobot_config = {}
            
        # Filter topics if provided
        topics_config = aloha_config.get("topics", [])
        if input_data.topics:
             # Only include topics that are in input_data.topics
             topics_config = [t for t in topics_config if t.get("topic_name") in input_data.topics]
        
        # Override output_repo_id
        # Combine outputPath and outputFileName to form the full path
        repo_id = os.path.join(output_path_prefix, input_data.outputFileName)
        
        # Remove .zip extension if present, as LeRobot dataset is a directory structure
        if repo_id.lower().endswith('.zip'):
            repo_id = repo_id[:-4]
        
        # If user provides path-like string, use it. Otherwise, maybe prefix with local?
        # LeRobotDataset.create supports local paths.
        
        # Construct task config
        lerobot_cfg = lerobot_config.get("lerobot", {}) if isinstance(lerobot_config, dict) else {}
        if not isinstance(lerobot_cfg, dict):
            lerobot_cfg = {}
        lerobot_cfg["use_videos"] = True
        lerobot_cfg["mode"] = "video"
        lerobot_cfg.setdefault("vcodec", "h264")
        lerobot_cfg.setdefault("encoder_threads", 1)
        lerobot_cfg.setdefault("batch_encoding_size", 1)
        lerobot_cfg.setdefault("streaming_encoding", False)
        lerobot_cfg.setdefault("encoder_queue_maxsize", 10)
        lerobot_cfg.setdefault("image_writer_processes", 1)
        lerobot_cfg.setdefault("image_writer_threads", 1)

        task_config = {
            "topics": topics_config,
            "alignment": {
                "strategy": "backfill_on_grid",
                "grid_fps": input_data.frequency if input_data.frequency else DEFAULT_GRID_FPS
            },
            "lerobot": lerobot_cfg
        }
        
        def _runner():
            log_audit_safe(
                user_id=uid,
                username=uname,
                role=role,
                action_type=AA.START_TASK,
                project_id=job.projectId,
                project_name=job.projectName,
                resource_type=AR.CONVERT_JOB,
                resource_id=job_id,
                resource_name=input_data.taskName or job.outputFileName,
                ip=ip,
                user_agent=ua,
                detail_json={"domain": "convert"},
            )
            anyio.from_thread.run(
                run_lerobot_with_limit,
                job_id=job_id,
                mcap_path=input_mcap_path,
                output_repo_id=repo_id,
                config=task_config,
                jobs_store=jobs_store,
            )
        _conversion_dispatch = {
            "type": "conversion",
            "task_id": job_id,
            "mode": "lerobot_with_limit",
            "task_name": f"conversion-lerobot-{job_id[:8]}",
            "job": job.model_dump(),
            "job_id": job_id,
            "mcap_path": input_mcap_path,
            "input_dataset_id": str(dataset_id),
            "input_dataset_file_path": (dataset.file_path or "").strip(),
            "output_repo_id": repo_id,
            "config": task_config,
        }
    else:
        # Construct output path (ensure it ends with correct extension)
        # The frontend usually sends a directory as outputPath and a filename as outputFileName
        # But here we assume outputPath is the directory.
        full_output_path = os.path.join(output_path_prefix, input_data.outputFileName)
        if input_data.outputFormat == "HDF5" and not full_output_path.endswith(".hdf5"):
            full_output_path += ".hdf5"
        
        # Create task config
        config = {
            "topics": input_data.topics or [],
            "alignment": {
                "strategy": "backfill_on_grid",
                "grid_fps": input_data.frequency if input_data.frequency else DEFAULT_GRID_FPS
            }
        }

        def _runner():
            log_audit_safe(
                user_id=uid,
                username=uname,
                role=role,
                action_type=AA.START_TASK,
                project_id=job.projectId,
                project_name=job.projectName,
                resource_type=AR.CONVERT_JOB,
                resource_id=job_id,
                resource_name=input_data.taskName or job.outputFileName,
                ip=ip,
                user_agent=ua,
                detail_json={"domain": "convert"},
            )
            run_conversion_task(
                job_id=job_id,
                mcap_path=input_mcap_path,
                output_path=full_output_path,
                config=config,
            )
        _conversion_dispatch = {
            "type": "conversion",
            "task_id": job_id,
            "mode": "hdf5",
            "task_name": f"conversion-hdf5-{job_id[:8]}",
            "job": job.model_dump(),
            "job_id": job_id,
            "mcap_path": input_mcap_path,
            "input_dataset_id": str(dataset_id),
            "input_dataset_file_path": (dataset.file_path or "").strip(),
            "output_path": full_output_path,
            "config": config,
        }

    try:
        dispatch_task(_conversion_dispatch)
    except RedisDispatchError as e:
        job.status = "failed"
        job.errorMessage = str(e)[:2000]
        job.updatedAt = datetime.now().isoformat()
        job.logs.append(f"[{job.updatedAt}] {job.errorMessage}")
        _persist_conversion_job(job, dataset=dataset)
        raise HTTPException(status_code=503, detail=str(e)) from e

    return job


def _fmt_ts(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    try:
        if getattr(dt, "tzinfo", None) is not None:
            return dt.isoformat()
        return dt.isoformat()
    except Exception:
        return ""


def _creator_display_from_user(user: User) -> Optional[str]:
    """优先 users.username，空则 account_id；均空返回 None。"""
    un = (getattr(user, "username", None) or "").strip()
    if un:
        return un
    aid = (getattr(user, "account_id", None) or "").strip()
    return aid or None


def _batch_row_to_summary(batch: ConversionBatchJob, *, creator_name: Optional[str] = None) -> ConversionBatchSummary:
    cid = (str(batch.creator_id).strip() if getattr(batch, "creator_id", None) else None) or None
    cname = (creator_name or "").strip() or None
    return ConversionBatchSummary(
        batchId=batch.batch_id,
        taskName=batch.task_name,
        targetFormat=(batch.target_format or "").strip(),
        projectId=(batch.project_id or "").strip(),
        projectName=(batch.project_name or batch.project_id or "").strip() or "Unknown Project",
        creatorId=cid,
        creatorName=cname,
        totalCount=int(batch.total_count or 0),
        successCount=int(batch.success_count or 0),
        failedCount=int(batch.failed_count or 0),
        runningCount=int(batch.running_count or 0),
        pendingCount=int(batch.pending_count or 0),
        progressPercent=float(batch.progress_percent or 0),
        overallStatus=(batch.overall_status or "PENDING").strip(),
        createdAt=_fmt_ts(batch.created_at),
        updatedAt=_fmt_ts(batch.updated_at),
        legacySingleFile=False,
    )


def _legacy_job_to_summary(rec: ConversionJobAsset) -> ConversionBatchSummary:
    st = (rec.status or "queued").strip().lower()
    pending = running = succ = failc = 0
    if st == "queued":
        pending = 1
    elif st == "running":
        running = 1
    elif st == "succeeded":
        succ = 1
    elif st in ("failed", "canceled"):
        failc = 1
    else:
        running = 1
    ended = succ + failc
    prog = 100.0 * float(ended)
    omap = {
        "queued": "PENDING",
        "running": "RUNNING",
        "succeeded": "SUCCESS",
        "failed": "FAILED",
        "canceled": "CANCELED",
    }
    overall = omap.get(st, "RUNNING")
    nm = (rec.task_name or rec.output_file_name or rec.input_asset_name or rec.job_id or "").strip()
    op_name = (getattr(rec, "operator_name", None) or "").strip() or None
    return ConversionBatchSummary(
        batchId=f"{LEGACY_SINGLE_BATCH_PREFIX}{rec.job_id}",
        taskName=nm or None,
        targetFormat=(rec.output_format or "").strip(),
        projectId=(rec.project_id or "").strip(),
        projectName=(rec.project_name or rec.project_id or "").strip() or "Unknown Project",
        creatorId=None,
        creatorName=op_name,
        totalCount=1,
        successCount=succ,
        failedCount=failc,
        runningCount=running,
        pendingCount=pending,
        progressPercent=prog,
        overallStatus=overall,
        createdAt=_fmt_ts(rec.created_at),
        updatedAt=_fmt_ts(rec.updated_at),
        legacySingleFile=True,
    )


def _rec_to_child_item(rec: ConversionJobAsset) -> ConversionBatchChildItem:
    return ConversionBatchChildItem(
        jobId=rec.job_id,
        sourceFileName=(rec.input_asset_name or "").strip(),
        outputFileName=(rec.output_file_name or "").strip(),
        itemStatus=(rec.status or "queued").strip(),
        itemStage=rec.current_stage,
        errorMessage=rec.error_message,
        createdAt=_fmt_ts(rec.created_at),
        updatedAt=_fmt_ts(rec.updated_at),
    )


def _repair_stale_conversion_jobs(*, batch_id: Optional[str] = None) -> None:
    """
    自愈卡死的 conversion 子任务状态：
    - task_jobs 已明确 failed/cancelled/success：同步到 conversion_jobs
    - task_jobs 仍 running 且 RQ job 丢失/失败，且超过超时阈值：标记 failed
    """
    stale_timeout = timedelta(minutes=3)
    now = datetime.utcnow()
    session = DataAssetsSyncSessionLocal()
    changed_batch_ids: Set[str] = set()
    try:
        q = session.query(ConversionJobAsset).filter(ConversionJobAsset.status.in_(["queued", "running"]))
        if batch_id:
            q = q.filter(ConversionJobAsset.batch_id == batch_id)
        rows = q.order_by(ConversionJobAsset.updated_at.asc()).limit(200).all()
        if not rows:
            return

        for rec in rows:
            jid = (rec.job_id or "").strip()
            if not jid:
                continue

            tj = session.query(TaskJob).filter(TaskJob.id == jid).one_or_none()
            tjs = ((tj.status or "").strip().lower() if tj is not None else "")
            changed = False
            rec_status = (rec.status or "").strip().lower()

            if tjs in ("failed", "cancelled", "success"):
                if tjs == "failed":
                    rec.status = "failed"
                    rec.error_message = (tj.error or rec.error_message or "转换任务失败")
                elif tjs == "cancelled":
                    rec.status = "canceled"
                    rec.error_message = (tj.error or rec.error_message or "Task cancelled")
                elif tjs == "success" and rec_status in ("queued", "running"):
                    # 执行已完成但转换表未写回 succeeded，按异常失败兜底，避免永远 running。
                    rec.status = "failed"
                    rec.error_message = "转换任务状态异常（执行完成但结果未回写），请重试"
                changed = True
            elif rec_status == "running":
                updated_at = getattr(rec, "updated_at", None)
                age_ok = bool(updated_at and (now - updated_at) >= stale_timeout)
                if age_ok:
                    rq_status = None
                    try:
                        job = Job.fetch(jid, connection=redis_conn)
                        rq_status = (job.get_status(refresh=True) or "").strip().lower()
                    except NoSuchJobError:
                        rq_status = "missing"
                    except Exception:
                        rq_status = None
                    if rq_status in ("failed", "stopped", "canceled", "cancelled", "missing"):
                        rec.status = "failed"
                        rec.error_message = "转换进程异常退出，请重试（worker 可能被系统终止）"
                        changed = True

            if changed:
                session.add(rec)
                if rec.batch_id:
                    changed_batch_ids.add(str(rec.batch_id))

        if changed_batch_ids:
            session.commit()
        else:
            session.rollback()
    except Exception:
        session.rollback()
    finally:
        session.close()

    for bid in changed_batch_ids:
        try:
            recompute_conversion_batch_stats(bid)
        except Exception:
            pass


def _preflight_conversion_item(input_data: CreateConversionInput) -> DataAsset:
    dataset_id = input_data.inputDatasetId or input_data.assetId
    if not dataset_id:
        raise HTTPException(status_code=400, detail="inputDatasetId is required")
    dataset = get_dataset_info(dataset_id)
    if (getattr(dataset, "sync_status", "synced") or "synced").strip().lower() != "synced":
        raise HTTPException(status_code=409, detail="该数据尚未同步，暂不可转换")
    if (dataset.format or "").lower() != "mcap":
        raise HTTPException(status_code=400, detail="当前仅支持 MCAP 格式数据的转换")
    _ensure_input_mcap_exists_or_404(dataset.file_path, dataset_id=dataset_id)
    return dataset


@router.post("/jobs", response_model=ConversionJob)
def create_conversion_job(
    input_data: CreateConversionInput,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    return _create_conversion_job_impl(input_data, request, current_user, batch_id=None)


@router.post("/batches", response_model=ConversionBatchDetail)
def create_conversion_batch(
    body: CreateConversionBatchInput,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    assert_conversion_manage_or_execute_sync(current_user)
    _ensure_conversion_jobs_schema()
    if not body.items:
        raise HTTPException(status_code=400, detail="items 不能为空")
    if len(body.items) > 200:
        raise HTTPException(status_code=400, detail="单次批量不得超过 200 条")

    validated: List[Tuple[CreateConversionInput, DataAsset]] = []
    primary_fmt: Optional[str] = None
    for it in body.items:
        ds = _preflight_conversion_item(it)
        of = (it.outputFormat or "").strip().upper()
        if primary_fmt is None:
            primary_fmt = of
        elif of != primary_fmt:
            raise HTTPException(status_code=400, detail="同一批量任务内输出格式必须一致")
        validated.append((it, ds))

    pid0 = (validated[0][1].project_id or validated[0][0].projectId or "").strip()
    for it, ds in validated:
        pid = (ds.project_id or it.projectId or "").strip()
        if pid != pid0:
            raise HTTPException(status_code=400, detail="同一批量任务内项目必须一致")

    batch_id = str(uuid.uuid4())
    d0 = validated[0][1]
    it0 = validated[0][0]
    creator_id = str(getattr(current_user, "id", "") or "").strip() or None

    sess = DataAssetsSyncSessionLocal()
    try:
        row = ConversionBatchJob(
            batch_id=batch_id,
            task_name=(body.taskName or "").strip() or None,
            source_format=(d0.format or "mcap").upper(),
            target_format=(it0.outputFormat or "").strip(),
            project_id=pid0 or None,
            project_name=(d0.project_name or d0.project_id or it0.projectId or "").strip() or None,
            creator_id=creator_id,
            total_count=len(validated),
            success_count=0,
            failed_count=0,
            canceled_count=0,
            running_count=0,
            pending_count=len(validated),
            progress_percent=0.0,
            overall_status="PENDING",
        )
        sess.add(row)
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()

    uid, uname, role = sync_resolve_user_for_audit(request)
    ip = request_client_ip(request)
    ua = request_user_agent(request)
    log_audit_safe(
        user_id=uid,
        username=uname,
        role=role,
        action_type=AA.CREATE_TASK,
        project_id=pid0,
        project_name=d0.project_name or pid0,
        resource_type=AR.CONVERT_JOB,
        resource_id=batch_id,
        resource_name=body.taskName or f"conversion-batch-{batch_id[:8]}",
        ip=ip,
        user_agent=ua,
        detail_json={"domain": "convert", "kind": "batch", "childCount": len(validated)},
    )

    try:
        for it, _ds in validated:
            _create_conversion_job_impl(it, request, current_user, batch_id=batch_id)
    except HTTPException as he:
        if he.status_code == 503:
            try:
                recompute_conversion_batch_stats(batch_id)
            except Exception:
                pass
        raise

    recompute_conversion_batch_stats(batch_id)

    brec = DataAssetsSyncSessionLocal()
    try:
        batch = brec.query(ConversionBatchJob).filter(ConversionBatchJob.batch_id == batch_id).one_or_none()
        if not batch:
            raise HTTPException(status_code=500, detail="父任务创建失败")
        assert_conversion_job_in_scope_sync(current_user, project_id=batch.project_id)
        rows = (
            brec.query(ConversionJobAsset)
            .filter(ConversionJobAsset.batch_id == batch_id)
            .order_by(ConversionJobAsset.id.asc())
            .all()
        )
        cn = _creator_display_from_user(current_user)
        return ConversionBatchDetail(
            batch=_batch_row_to_summary(batch, creator_name=cn),
            children=[_rec_to_child_item(r) for r in rows],
        )
    finally:
        brec.close()


@router.get("/batches", response_model=List[ConversionBatchSummary])
def list_conversion_batches(current_user: User = Depends(get_current_user)):
    _ensure_conversion_jobs_schema()
    _repair_stale_conversion_jobs()
    scoped = scoped_project_ids_sync(current_user)
    if scoped is not None and len(scoped) == 0:
        return []

    session = DataAssetsSyncSessionLocal()
    out: List[ConversionBatchSummary] = []
    try:
        qb = session.query(ConversionBatchJob)
        if scoped is not None:
            qb = qb.filter(ConversionBatchJob.project_id.in_(list(scoped)))
        batch_rows = qb.order_by(ConversionBatchJob.created_at.desc()).all()
        creator_ids: Set[str] = {
            str(x.creator_id).strip()
            for x in batch_rows
            if getattr(x, "creator_id", None) and str(x.creator_id).strip()
        }
        creator_name_by_id: Dict[str, str] = {}
        if creator_ids:
            for u in session.query(User).filter(User.id.in_(list(creator_ids))).all():
                dn = _creator_display_from_user(u)
                if dn:
                    creator_name_by_id[str(u.id)] = dn
        for b in batch_rows:
            cid = (str(b.creator_id).strip() if getattr(b, "creator_id", None) else "") or ""
            cname = creator_name_by_id.get(cid) if cid else None
            out.append(_batch_row_to_summary(b, creator_name=cname))

        qj = session.query(ConversionJobAsset).filter(ConversionJobAsset.batch_id.is_(None))
        if scoped is not None:
            qj = qj.filter(ConversionJobAsset.project_id.in_(list(scoped)))
        for rec in qj.order_by(ConversionJobAsset.created_at.desc()).all():
            out.append(_legacy_job_to_summary(rec))

        out.sort(key=lambda x: x.createdAt or "", reverse=True)
        return out
    finally:
        session.close()


@router.get("/batches/{batch_id}", response_model=ConversionBatchDetail)
def get_conversion_batch_detail(batch_id: str, current_user: User = Depends(get_current_user)):
    _ensure_conversion_jobs_schema()
    if batch_id.startswith(LEGACY_SINGLE_BATCH_PREFIX):
        jid = batch_id[len(LEGACY_SINGLE_BATCH_PREFIX) :]
        rec = _get_job_from_db(jid)
        if not rec:
            raise HTTPException(status_code=404, detail="Batch not found")
        assert_conversion_job_in_scope_sync(current_user, project_id=rec.project_id)
        return ConversionBatchDetail(batch=_legacy_job_to_summary(rec), children=[_rec_to_child_item(rec)])

    _repair_stale_conversion_jobs(batch_id=batch_id)
    session = DataAssetsSyncSessionLocal()
    try:
        batch = session.query(ConversionBatchJob).filter(ConversionBatchJob.batch_id == batch_id).one_or_none()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        assert_conversion_job_in_scope_sync(current_user, project_id=batch.project_id)
        rows = (
            session.query(ConversionJobAsset)
            .filter(ConversionJobAsset.batch_id == batch_id)
            .order_by(ConversionJobAsset.id.asc())
            .all()
        )
        cname: Optional[str] = None
        cid = (str(batch.creator_id).strip() if getattr(batch, "creator_id", None) else "") or ""
        if cid:
            u = session.query(User).filter(User.id == cid).one_or_none()
            if u is not None:
                cname = _creator_display_from_user(u)
        return ConversionBatchDetail(
            batch=_batch_row_to_summary(batch, creator_name=cname),
            children=[_rec_to_child_item(r) for r in rows],
        )
    finally:
        session.close()


def _cancel_single_child_conversion_job(job_id: str) -> Optional[str]:
    """协作式取消一个子任务；返回其 batch_id（若有）。"""
    jid = (job_id or "").strip()
    if not jid:
        return None
    rec = _get_job_from_db(jid)
    bid = ((rec.batch_id or "").strip() or None) if rec else None
    st = ((rec.status if rec else None) or "").lower()
    if st in ("succeeded", "failed", "canceled"):
        return bid
    try:
        update_task_status(jid, "cancelled")
    except Exception:
        pass
    session = DataAssetsSyncSessionLocal()
    try:
        r = session.query(ConversionJobAsset).filter(ConversionJobAsset.job_id == jid).one_or_none()
        if r is not None and (r.status or "").lower() in ("queued", "running"):
            r.status = "canceled"
            r.error_message = "Task cancelled"
            session.add(r)
            session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()
    job = jobs_store.get(jid)
    if job is not None:
        job.status = "canceled"
        job.errorMessage = "Task cancelled"
        job.updatedAt = datetime.now().isoformat()
        _persist_conversion_job(job)
    elif rec is not None:
        j2 = _record_to_job(rec)
        j2.status = "canceled"
        j2.errorMessage = "Task cancelled"
        j2.updatedAt = datetime.now().isoformat()
        _persist_conversion_job(j2)
    return bid


@router.post("/batches/{batch_id}/cancel")
def cancel_conversion_batch(batch_id: str, current_user: User = Depends(get_current_user)):
    assert_conversion_manage_or_execute_sync(current_user)
    _ensure_conversion_jobs_schema()
    logger.info(
        "conversion_batch_cancel_requested",
        extra={"batch_id": (batch_id or "")[:64], "user_id": str(getattr(current_user, "id", "") or "")},
    )
    if batch_id.startswith(LEGACY_SINGLE_BATCH_PREFIX):
        jid = batch_id[len(LEGACY_SINGLE_BATCH_PREFIX) :].strip()
        rec = _get_job_from_db(jid)
        if not rec:
            raise HTTPException(status_code=404, detail="Batch not found")
        assert_conversion_job_in_scope_sync(current_user, project_id=rec.project_id)
        assert_platform_task_execute_project_sync(current_user, rec.project_id or "")
        st = (rec.status or "").lower()
        if st in ("succeeded", "failed", "canceled"):
            raise HTTPException(status_code=400, detail="任务已结束，无法取消")
        _cancel_single_child_conversion_job(jid)
        return {"ok": True}

    sess = DataAssetsSyncSessionLocal()
    try:
        batch = sess.query(ConversionBatchJob).filter(ConversionBatchJob.batch_id == batch_id).one_or_none()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        assert_conversion_job_in_scope_sync(current_user, project_id=batch.project_id)
        assert_platform_task_execute_project_sync(current_user, batch.project_id or "")
        ovs = (batch.overall_status or "").strip().upper()
        if ovs in ("SUCCESS", "FAILED", "CANCELED", "PARTIAL_SUCCESS"):
            raise HTTPException(status_code=400, detail="批量任务已结束，无法取消")
        rows = (
            sess.query(ConversionJobAsset)
            .filter(ConversionJobAsset.batch_id == batch_id)
            .order_by(ConversionJobAsset.id.asc())
            .all()
        )
    finally:
        sess.close()

    for r in rows:
        st = (r.status or "").lower()
        if st in ("queued", "running"):
            _cancel_single_child_conversion_job(r.job_id)
    recompute_conversion_batch_stats(batch_id)
    return {"ok": True}


@router.delete("/batches/{batch_id}")
def delete_conversion_batch(
    batch_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    assert_conversion_manage_or_execute_sync(current_user)
    _ensure_conversion_jobs_schema()
    if batch_id.startswith(LEGACY_SINGLE_BATCH_PREFIX):
        jid = batch_id[len(LEGACY_SINGLE_BATCH_PREFIX) :].strip()
        rec = _get_job_from_db(jid)
        job_snap = jobs_store.get(jid)
        if not rec and not job_snap:
            raise HTTPException(status_code=404, detail="Batch not found")
        pid = (rec.project_id or "").strip() if rec else str(getattr(job_snap, "projectId", None) or "").strip()
        assert_platform_task_manage_project_sync(current_user, pid)
        if jid in jobs_store:
            del jobs_store[jid]
        deleted = _delete_job_from_db(jid)
        if deleted:
            delete_task_job(jid)
            uid, uname, role = sync_resolve_user_for_audit(request)
            log_audit_safe(
                user_id=uid,
                username=uname,
                role=role,
                action_type=AA.DELETE_TASK,
                resource_type=AR.CONVERT_JOB,
                resource_id=jid,
                resource_name=(getattr(job_snap, "taskName", None) or batch_id[:48]),
                detail_json={"domain": "convert", "kind": "legacy_batch"},
                request=request,
            )
            return {"success": True}
        raise HTTPException(status_code=404, detail="Batch not found")

    sess = DataAssetsSyncSessionLocal()
    batch_name_for_audit: Optional[str] = None
    try:
        batch = sess.query(ConversionBatchJob).filter(ConversionBatchJob.batch_id == batch_id).one_or_none()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        assert_platform_task_manage_project_sync(current_user, batch.project_id or "")
        batch_name_for_audit = batch.task_name
        rows = (
            sess.query(ConversionJobAsset)
            .filter(ConversionJobAsset.batch_id == batch_id)
            .order_by(ConversionJobAsset.id.asc())
            .all()
        )
        job_ids = [r.job_id for r in rows]
    finally:
        sess.close()

    for jid in job_ids:
        if jid in jobs_store:
            del jobs_store[jid]
        _delete_job_from_db(jid)
        delete_task_job(jid)

    sess2 = DataAssetsSyncSessionLocal()
    try:
        b = sess2.query(ConversionBatchJob).filter(ConversionBatchJob.batch_id == batch_id).one_or_none()
        if b:
            sess2.delete(b)
            sess2.commit()
    except Exception:
        sess2.rollback()
    finally:
        sess2.close()

    uid, uname, role = sync_resolve_user_for_audit(request)
    log_audit_safe(
        user_id=uid,
        username=uname,
        role=role,
        action_type=AA.DELETE_TASK,
        resource_type=AR.CONVERT_JOB,
        resource_id=batch_id,
        resource_name=(batch_name_for_audit or batch_id[:12]),
        detail_json={"domain": "convert", "kind": "batch", "childCount": len(job_ids)},
        request=request,
    )
    return {"success": True}


@router.post("/mcap-to-lerobot", response_model=ConversionJob)
def create_lerobot_conversion_job(
    input_data: LeRobotConversionInput,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Start a background job to convert MCAP to LeRobot dataset.
    """
    assert_conversion_manage_or_execute_sync(current_user)
    job_id = str(uuid.uuid4())
    task_no = generate_task_no()
    
    # Get dataset info to find file path
    asset = get_dataset_info(input_data.inputDatasetId)
    if not asset:
        raise HTTPException(status_code=404, detail="Input dataset not found")
    if (getattr(asset, "sync_status", "synced") or "synced").strip().lower() != "synced":
        raise HTTPException(status_code=409, detail="该数据尚未同步，暂不可转换")
        
    input_path = _ensure_input_mcap_exists_or_404(asset.file_path, dataset_id=input_data.inputDatasetId)

    # Create job entry
    job = ConversionJob(
        jobId=job_id,
        shortCode=job_id[:8],
        taskNo=task_no,
        outputFileName=input_data.outputRepoId, # Use repo ID as output name
        fileName=os.path.basename(input_path),
        assetId=input_data.inputDatasetId,
        assetName=asset.filename,
        projectId=input_data.projectId,
        projectName=asset.project_name or "Unknown",
        deviceName="Unknown Device",
        outputFormat="LeRobot",
        fileFormat=(asset.format or "mcap").upper(),
        outputLocation="local", 
        outputPath=input_data.outputRepoId,
        status="queued",
        progressPercent=0,
        currentStage="init",
        stages=[],
        logs=[],
        createdAt=datetime.now().isoformat(),
        updatedAt=datetime.now().isoformat(),
        artifactReady=False,
        operatorName=(getattr(current_user, "username", None) or "").strip() or None,
    )
    _assert_project_writable_sync(job.projectId)
    assert_platform_task_execute_project_sync(current_user, job.projectId)

    enforce_queue_dispatch_allowed(get_task_profile("conversion").queue)

    jobs_store[job_id] = job
    _persist_conversion_job(job, dataset=asset)

    uid, uname, role = sync_resolve_user_for_audit(request)
    ip = request_client_ip(request)
    ua = request_user_agent(request)
    log_audit_safe(
        user_id=uid,
        username=uname,
        role=role,
        action_type=AA.CREATE_TASK,
        project_id=job.projectId,
        project_name=job.projectName,
        resource_type=AR.CONVERT_JOB,
        resource_id=job_id,
        resource_name=input_data.outputRepoId,
        ip=ip,
        user_agent=ua,
        detail_json={"assetId": input_data.inputDatasetId, "domain": "convert"},
    )
    
    def _runner():
        log_audit_safe(
            user_id=uid,
            username=uname,
            role=role,
            action_type=AA.START_TASK,
            project_id=job.projectId,
            project_name=job.projectName,
            resource_type=AR.CONVERT_JOB,
            resource_id=job_id,
            resource_name=input_data.outputRepoId,
            ip=ip,
            user_agent=ua,
            detail_json={"domain": "convert"},
        )
        anyio.from_thread.run(
            convert_mcap_to_lerobot_task,
            job_id=job_id,
            mcap_path=input_path,
            output_repo_id=input_data.outputRepoId,
            config=input_data.config,
            jobs_store=jobs_store,
        )
    dispatch_task({
        "type": "conversion",
        "task_id": job_id,
        "mode": "lerobot_direct",
        "task_name": f"conversion-lerobot-{job_id[:8]}",
        "job": job.model_dump(),
        "job_id": job_id,
        "mcap_path": input_path,
        "input_dataset_id": str(input_data.inputDatasetId),
        "input_dataset_file_path": (asset.file_path or "").strip(),
        "output_repo_id": input_data.outputRepoId,
        "config": input_data.config,
    })
    
    return job

@router.get("/jobs", response_model=List[ConversionJob])
def list_conversion_jobs(current_user: User = Depends(get_current_user)):
    _ensure_conversion_jobs_schema()
    # DB 为真源：避免 API 与 worker 不同进程时，内存缓存 jobs_store 造成状态倒退（长期 queued）。
    records = _load_jobs_from_db()

    scoped = scoped_project_ids_sync(current_user)
    if scoped is not None:
        records = [r for r in records if (r.project_id or "").strip() in scoped]
    return [_record_to_job(rec) for rec in records]

@router.get("/jobs/{job_id}", response_model=ConversionJob)
def get_conversion_job(job_id: str, current_user: User = Depends(get_current_user)):
    # DB 为真源：单任务查询也统一从数据库读，保证与 worker 执行状态一致。
    rec = _get_job_from_db(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Job not found")
    assert_conversion_job_in_scope_sync(current_user, project_id=rec.project_id)
    return _record_to_job(rec)

@router.delete("/jobs/{job_id}")
def delete_conversion_job(
    job_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    rec = _get_job_from_db(job_id)
    job_snap = jobs_store.get(job_id)
    if not rec and not job_snap:
        raise HTTPException(status_code=404, detail="Job not found")
    assert_conversion_manage_or_execute_sync(current_user)
    pid = (
        (rec.project_id or "").strip()
        if rec
        else str(getattr(job_snap, "projectId", None) or "").strip()
    )
    assert_platform_task_manage_project_sync(current_user, pid)
    if job_id in jobs_store:
        del jobs_store[job_id]
    deleted = _delete_job_from_db(job_id)
    if deleted:
        uid, uname, role = sync_resolve_user_for_audit(request)
        rname = None
        if job_snap is not None:
            rname = (
                getattr(job_snap, "taskName", None)
                or getattr(job_snap, "outputFileName", None)
                or getattr(job_snap, "fileName", None)
            )
        log_audit_safe(
            user_id=uid,
            username=uname,
            role=role,
            action_type=AA.DELETE_TASK,
            resource_type=AR.CONVERT_JOB,
            resource_id=job_id,
            resource_name=(rname or job_id[:12]),
            detail_json={"domain": "convert"},
            request=request,
        )
        return {"success": True}
    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/jobs/{job_id}/delete-result", response_model=ApiResponse)
def delete_conversion_result(job_id: str, current_user: User = Depends(get_current_user)):
    """删除转换任务产物（输出文件或目录）。仅允许删除白名单内的路径。删除成功后任务记录移除。"""
    rec0 = _get_job_from_db(job_id)
    job = jobs_store.get(job_id)
    if not job:
        if not rec0:
            return ApiResponse(ok=False, error="转换任务不存在或已过期")
        job = _record_to_job(rec0)
    pid = (
        (rec0.project_id or "").strip()
        if rec0
        else str(getattr(job, "projectId", None) or "").strip()
    )
    assert_conversion_manage_or_execute_sync(current_user)
    assert_platform_task_manage_project_sync(current_user, pid)

    jst = (getattr(job, "status", None) or "").strip().lower()
    if jst in ("queued", "running"):
        return ApiResponse(ok=False, error="进行中任务请先在任务中心点击「取消」，勿使用删除")

    output_path = (getattr(job, "outputPath", None) or "").strip()
    output_file = getattr(job, "outputFileName", None) or getattr(job, "fileName", None) or ""
    output_format = getattr(job, "outputFormat", "") or ""
    if not output_path:
        jobs_store.pop(job_id, None)
        _delete_job_from_db(job_id)
        delete_task_job(job_id)
        return ApiResponse(ok=True, data=None)

    to_delete = output_path
    if not output_path.startswith("minio://"):
        if output_format == "LeRobot" or (isinstance(output_path, str) and not output_path.endswith(".hdf5")):
            to_delete = output_path.rstrip("/\\")
        else:
            to_delete = os.path.join(output_path, output_file) if output_file else output_path
        to_delete = (to_delete or "").strip()
        if not to_delete:
            jobs_store.pop(job_id, None)
            _delete_job_from_db(job_id)
            delete_task_job(job_id)
            return ApiResponse(ok=True, data=None)
        try:
            resolved = validate_path_whitelist(to_delete)
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            return ApiResponse(ok=False, error=detail[:200])
        try:
            if os.path.isfile(resolved):
                os.remove(resolved)
            elif os.path.isdir(resolved):
                shutil.rmtree(resolved, ignore_errors=False)
        except FileNotFoundError:
            pass
        except PermissionError:
            return ApiResponse(ok=False, error="没有权限删除该路径")
        except OSError as e:
            return ApiResponse(ok=False, error=(str(e) or "删除失败")[:200])
    else:
        try:
            delete_by_minio_uri(output_path)
        except MinioBucketError as e:
            return ApiResponse(ok=False, error=f"删除 MinIO 产物失败: {str(e)[:180]}")

    # 同步清理转换产物对应的 data_assets 记录（按 file_path=outputPath）
    try:
        session = DataAssetsSyncSessionLocal()
        try:
            rows = session.query(DataAsset).filter(DataAsset.file_path == output_path).all()
            for row in rows:
                session.delete(row)
            session.commit()
        finally:
            session.close()
    except Exception:
        pass

    jobs_store.pop(job_id, None)
    _delete_job_from_db(job_id)
    delete_task_job(job_id)
    return ApiResponse(ok=True, data={"deletedPath": output_path})
