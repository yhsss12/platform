import asyncio
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Union, Dict, Any
from uuid import UUID
import os
import time
from datetime import datetime
import logging
import threading
from collections import deque
from sqlalchemy.exc import IntegrityError

from app.core.deps import get_current_user, get_current_user_ws
from app.core.roles import is_super_admin
from sqlalchemy import select, func, text
from app.db.data_assets_session import get_data_assets_db, DataAssetsSessionLocal
from app.models.user import User
from app.schemas.common import ApiResponse
from app.schemas.job import JobCreate, JobUpdate
from app.schemas.data_asset import DataAssetCreate
from app.models.data_asset import CollectionJobAsset, CollectionTaskAsset, DataAsset
from app.models.project_asset import Project
from app.crud.project import get_visible_project_ids, get_project_by_id, is_project_visible_to_user
from app.crud.data_asset import (
    get_asset_by_file_path,
    create_asset,
    next_code as next_asset_code,
    update_asset,
)
from app.services.asset_meta_parser import parse_meta_for_asset
from app.services.collect_storage_layout import resolve_collect_job_workspace_path
from app.services.agent_collect_proxy import delete_collect_job_workspace_remote
from app.services.collect_disk_reconcile import reconcile_collection_job_progress_from_agent_disk
from app.realtime.job_ws import manager as job_ws_manager
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import enqueue_audit_log

router = APIRouter()
logger = logging.getLogger(__name__)


async def _canonical_project_name(
    db: AsyncSession,
    project_id: Optional[str],
    *fallbacks: Optional[str],
) -> Optional[str]:
    """人类可读项目名称：优先 projects.name，其次非空 fallback，最后退回 project_id。"""
    pid = (project_id or "").strip()
    if pid:
        try:
            proj = await get_project_by_id(db, pid)
            if proj is not None:
                nm = (getattr(proj, "name", None) or "").strip()
                if nm:
                    return nm
        except Exception:
            pass
    for fb in fallbacks:
        s = (fb or "").strip()
        if s:
            return s
    return pid or None

_collection_jobs_device_id_migrated = False
_collection_jobs_validation_report_json_migrated = False
_progress_anomalies_lock = threading.Lock()
_progress_anomalies: deque[dict] = deque(maxlen=300)
_progress_regression_count = 0


def _normalize_collect_mcap_filename(raw_name: Optional[str]) -> str:
    """采集资产文件名统一保留 .mcap 后缀。"""
    name = (raw_name or "").strip() or "data.mcap"
    if not name.lower().endswith(".mcap"):
        name = f"{name}.mcap"
    return name


def _ensure_collection_jobs_device_id_column() -> None:
    """PostgreSQL：为 collection_jobs 增加 device_id（领取作业时写入平台设备 id）。"""
    global _collection_jobs_device_id_migrated
    if _collection_jobs_device_id_migrated:
        return
    _collection_jobs_device_id_migrated = True
    try:
        from app.services.asset_registration_service import data_assets_sync_engine

        with data_assets_sync_engine.begin() as conn:
            conn.execute(text("ALTER TABLE collection_jobs ADD COLUMN IF NOT EXISTS device_id VARCHAR(64)"))
            # 避免并发创建作业时出现同 task 下 job_number 重复
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_collection_jobs_task_job_number "
                    "ON collection_jobs (task_id, job_number)"
                )
            )
    except Exception:
        pass


def _ensure_collection_jobs_validation_report_json_column() -> None:
    """PostgreSQL：为 collection_jobs 增加 validation_report_json（质检报告 JSON 文本）。"""
    global _collection_jobs_validation_report_json_migrated
    if _collection_jobs_validation_report_json_migrated:
        return
    _collection_jobs_validation_report_json_migrated = True
    try:
        from app.services.asset_registration_service import data_assets_sync_engine

        with data_assets_sync_engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE collection_jobs ADD COLUMN IF NOT EXISTS validation_report_json TEXT")
            )
    except Exception:
        pass


# Global dictionary to track job cancellation requests
# Key: job_id (str), Value: bool (True if cancelled)
job_cancellation_flags: Dict[str, bool] = {}


async def _backfill_collect_asset_size_by_path(
    mcap_path: str,
    *,
    retry_count: int = 6,
    retry_interval_sec: float = 1.0,
) -> None:
    """延迟重试回填采集资产大小，处理“先落库后落盘”的时间窗。"""
    p = os.path.normpath((mcap_path or "").replace("\\", "/").strip())
    if not p:
        return

    for _ in range(max(1, int(retry_count))):
        size = 0
        try:
            if os.path.isfile(p):
                size = int(os.path.getsize(p))
        except Exception:
            size = 0

        if size > 0:
            try:
                async with DataAssetsSessionLocal() as s:
                    asset = await get_asset_by_file_path(s, p)
                    if not asset:
                        return
                    if int(getattr(asset, "file_size_bytes", 0) or 0) > 0:
                        return
                    asset.file_size_bytes = size
                    await s.commit()
            except Exception:
                return
            return

        await asyncio.sleep(max(0.1, float(retry_interval_sec)))

async def run_collection_service(job_id: str, duration: int, job_ws):
    """
    Simulate backend collection service running asynchronously.
    In a real scenario, this would interact with the hardware/ROS2 nodes.
    Now supports cancellation.
    """
    print(f"Starting collection service for job {job_id}, duration {duration}s")
    
    # Notify start
    await job_ws.broadcast_log(job_id, f"Collection service started for job {job_id}")
    await job_ws.broadcast_log(job_id, "Initializing sensors...")
    await asyncio.sleep(2)
    await job_ws.broadcast_log(job_id, "Sensors ready. Starting recording...")
    
    start_time = time.time()
    
    try:
        # Simulate main loop
        while True:
            # Check for cancellation
            if job_cancellation_flags.get(job_id, False):
                await job_ws.broadcast_log(job_id, "WARNING: Emergency Stop received! Stopping collection...")
                await asyncio.sleep(1)
                await job_ws.broadcast_log(job_id, "Collection stopped by user.")
                
                # Update DB status to CANCELLED (or DONE if preferred for now)
                # We need a new DB session here since we are in a background task
                # For simplicity in this simulation, we rely on the main thread or next request to see status
                # But best practice is to update it here. 
                # Leaving DB update to the caller or separate mechanism for now to avoid complexity with async sessions in bg tasks
                
                return # Exit the loop and function
                
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
                
            remaining = max(0, duration - elapsed)
            
            # Broadcast progress
            # Calculate progress percentage (20% to 100%)
            # We start at 20% (setup done) and go to 100%
            progress_percent = 20 + (elapsed / duration) * 80
            
            await job_ws.broadcast_progress(job_id, {
                "current": int(elapsed),
                "total": duration,
                "percent": min(100, int(progress_percent)),
                "status": "RUNNING",
                "remaining_time": int(remaining)
            })
            
            # Simulate random log events
            if int(elapsed) % 5 == 0:
                await job_ws.broadcast_log(job_id, f"Recording data chunk {int(elapsed)}... [OK]")
                
            await asyncio.sleep(1)
            
        # Completion
        await job_ws.broadcast_log(job_id, "Collection completed successfully.")
        await job_ws.broadcast_progress(job_id, {
            "current": duration,
            "total": duration,
            "percent": 100,
            "status": "DONE",
            "remaining_time": 0
        })
        
    except Exception as e:
        print(f"Error in collection service: {e}")
        await job_ws.broadcast_log(job_id, f"Error: {str(e)}")
    finally:
        # Cleanup
        if job_id in job_cancellation_flags:
            del job_cancellation_flags[job_id]

def _dt_iso(dt) -> str:
    if not dt:
        return ""
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)

def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None

def _job_progress_payload(percent: int, current: int, total: int) -> Dict[str, Any]:
    return {"percent": int(percent or 0), "current": int(current or 0), "total": int(total or 0)}

def _record_progress_anomaly(event: dict) -> None:
    global _progress_regression_count
    with _progress_anomalies_lock:
        _progress_anomalies.append(event)
        _progress_regression_count += 1

def _is_explicit_progress_reset_payload(update_data: Dict[str, Any]) -> bool:
    s = str(update_data.get("status") or "").strip().upper()
    c = update_data.get("completed_count")
    p = update_data.get("progress")
    mp = update_data.get("mcap_path")
    if s == "PENDING" and (c is None or int(c or 0) == 0):
        if (p is None or (isinstance(p, (int, float)) and int(p or 0) == 0)) and (mp is None or str(mp or "") == ""):
            return True
    if s in ("CANCELED", "FAILED"):
        return True
    return False

def _apply_progress_guard(
    *,
    existing_current: int,
    existing_total: int,
    existing_percent: int,
    desired_current: Optional[int],
    desired_total: Optional[int],
    allow_reset: bool,
    protect_total_regression: bool = True,
) -> tuple[int, int, int, bool, bool]:
    from app.services.collect_progress import apply_progress_guard

    next_current, next_total, next_percent, blocked_current, blocked_total = apply_progress_guard(
        existing_current=existing_current,
        existing_total=existing_total,
        existing_percent=existing_percent,
        desired_current=desired_current,
        desired_total=desired_total,
        allow_reset=allow_reset,
        protect_total_regression=protect_total_regression,
    )
    return next_current, next_total, next_percent, bool(blocked_current), bool(blocked_total)

def _job_to_dict(j: CollectionJobAsset) -> Dict[str, Any]:
    completed = int(getattr(j, "completed_count", 0) or 0)
    total = int(getattr(j, "collection_quantity", 0) or 0)
    percent = int(getattr(j, "progress", 0) or 0)
    return {
        "id": str(j.id),
        "type": "collection",
        "taskId": str(j.task_id),
        "task_id": str(j.task_id),
        "target": {"taskId": str(j.task_id)},
        "job_number": int(j.job_number or 0),
        "jobNumber": int(j.job_number or 0),
        "operator_name": j.operator_name,
        "operatorName": j.operator_name,
        "status": j.status,
        "collection_quantity": total,
        "collectionQuantity": total,
        "completed_count": completed,
        "completedCount": completed,
        "project_id": getattr(j, "project_id", None),
        "projectId": getattr(j, "project_id", None),
        "project_name": getattr(j, "project_name", None),
        "projectName": getattr(j, "project_name", None),
        "mcap_path": j.mcap_path,
        "mcapPath": j.mcap_path,
        "mcap_size_bytes": j.mcap_size_bytes,
        "mcapSizeBytes": j.mcap_size_bytes,
        "validation_report_json": getattr(j, "validation_report_json", None),
        "validationReportJson": getattr(j, "validation_report_json", None),
        "duration_sec": j.duration_sec,
        "durationSec": j.duration_sec,
        "started_at": _dt_iso(j.started_at),
        "startedAt": _dt_iso(j.started_at),
        "finished_at": _dt_iso(j.finished_at),
        "finishedAt": _dt_iso(j.finished_at),
        "progress": _job_progress_payload(percent, completed, total),
        "progressPercent": percent,
        "created_at": _dt_iso(getattr(j, "created_at", None)),
        "createdAt": _dt_iso(getattr(j, "created_at", None)),
        "updated_at": _dt_iso(getattr(j, "updated_at", None)),
        "updatedAt": _dt_iso(getattr(j, "updated_at", None)),
        "device_id": getattr(j, "device_id", None) or None,
        "deviceId": getattr(j, "device_id", None) or None,
    }

async def _assert_project_writable(db: AsyncSession, project_id: str) -> None:
    pid = (project_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="请选择所属项目")
    p = (await db.execute(select(Project).where(Project.id == pid))).scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="项目不存在")
    if (p.status or "").strip() == "已归档":
        raise HTTPException(status_code=403, detail="项目已归档，禁止该操作")

@router.get("", response_model=ApiResponse)
async def read_jobs(
    task_id: Optional[UUID] = None,
    skip: int = 0,
    limit: int = 100,
    reconcile_disk: bool = Query(
        False,
        description="为 true 且采集端隧道在线时，按作业 workspace 下 episode 目录数回写进度（与磁盘对齐）",
    ),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """获取作业列表

    - 若带 task_id：只返回该任务下的作业，任意已登录用户可查（任务可见性由任务列表接口控制）
    - 若不带 task_id（查全部）：仅管理员可查，避免跨项目泄露
    - reconcile_disk：可选，将作业 completed_count 与采集端磁盘 episode 目录数对齐（需 Agent 在线）
    """
    base_stmt = select(CollectionJobAsset)

    if task_id:
        task = await db.get(CollectionTaskAsset, str(task_id))
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if not is_super_admin(getattr(current_user, "role", None)):
            pid = (getattr(task, "project_id", None) or "").strip()
            visible = await is_project_visible_to_user(
                db,
                project_id=pid,
                user_id=str(getattr(current_user, "id", "") or ""),
                include_owner_projects=True,
            )
            if not visible:
                raise HTTPException(status_code=404, detail="Task not found")
        base_stmt = base_stmt.where(CollectionJobAsset.task_id == str(task_id))
    else:
        if not is_super_admin(getattr(current_user, "role", None)):
            allowed = await get_visible_project_ids(
                db,
                user_id=str(getattr(current_user, "id", "") or ""),
                include_owner_projects=True,
            )
            if not allowed:
                return ApiResponse(ok=True, data=[])
            base_stmt = base_stmt.where(CollectionJobAsset.project_id.in_(allowed))

    _ensure_collection_jobs_device_id_column()
    _ensure_collection_jobs_validation_report_json_column()
    stmt = base_stmt.order_by(CollectionJobAsset.updated_at.desc()).offset(skip).limit(limit)
    r = await db.execute(stmt)
    jobs = list(r.scalars().all())
    if reconcile_disk and jobs:
        for db_job in jobs:
            try:
                tid0 = str(getattr(db_job, "task_id", "") or "").strip()
                task_row = await db.get(CollectionTaskAsset, tid0) if tid0 else None
                task_desc = getattr(task_row, "description", None) if task_row else None
                ok_rec, _err = await reconcile_collection_job_progress_from_agent_disk(
                    db,
                    job_row=db_job,
                    task_description_json=task_desc,
                )
                if ok_rec:
                    await db.refresh(db_job)
            except Exception as exc:
                logger.warning(
                    "read_jobs reconcile_disk failed job_id=%s err=%s",
                    getattr(db_job, "id", ""),
                    exc,
                )
    return ApiResponse(
        ok=True,
        data=[_job_to_dict(j) for j in jobs]
    )

@router.get("/progress-anomalies", response_model=ApiResponse)
async def list_progress_anomalies(current_user: User = Depends(get_current_user)):
    if not is_super_admin(getattr(current_user, "role", None)):
        raise HTTPException(status_code=403, detail="仅管理员可查看")
    with _progress_anomalies_lock:
        rows = list(_progress_anomalies)
        count = int(_progress_regression_count)
    return ApiResponse(ok=True, data={"count": count, "items": rows})

def _audit_collection_job_lifecycle(
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: User,
    db_job: CollectionJobAsset,
    old_status: str,
    new_status: str,
) -> None:
    o = (old_status or "").strip().upper()
    n = (new_status or "").strip().upper()
    if o == n:
        return
    pid = (getattr(db_job, "project_id", None) or "").strip() or None
    pname = getattr(db_job, "project_name", None)
    jn = getattr(db_job, "job_number", None)
    rname = f"采集作业 #{jn}" if jn is not None else f"采集作业 {db_job.id[:8]}"
    base = dict(
        user=current_user,
        request=request,
        project_id=pid,
        project_name=pname,
        resource_type=AR.COLLECTION_JOB,
        resource_id=str(db_job.id),
        resource_name=rname,
        detail_json={"task_id": str(getattr(db_job, "task_id", "") or ""), "job_number": jn},
    )
    if n == "CANCELED":
        enqueue_audit_log(background_tasks, action_type=AA.STOP_TASK, **base)
        return
    if n == "PAUSED":
        enqueue_audit_log(
            background_tasks,
            action_type=AA.UPDATE_TASK,
            detail_json={**(base.get("detail_json") or {}), "control": "pause"},
            **{k: v for k, v in base.items() if k != "detail_json"},
        )
        return
    if n == "RUNNING":
        if o == "PAUSED":
            enqueue_audit_log(
                background_tasks,
                action_type=AA.UPDATE_TASK,
                detail_json={**(base.get("detail_json") or {}), "control": "resume"},
                **{k: v for k, v in base.items() if k != "detail_json"},
            )
        elif o == "PENDING":
            # 创建作业时已记 START_TASK，避免 PATCH 首次置 RUNNING 重复
            return
        else:
            enqueue_audit_log(background_tasks, action_type=AA.START_TASK, **base)


@router.post("", response_model=ApiResponse)
async def create_new_job(
    job_in: JobCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """创建作业"""
    # Validate device online (mock check)
    # In real world, call routes_devices check_device_health
    
    # Create local directory (mock)
    # In real world, os.makedirs(job_in.storage_path)
    
    task = await db.get(CollectionTaskAsset, str(job_in.task_id))
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not is_super_admin(getattr(current_user, "role", None)):
        pid = (getattr(task, "project_id", None) or "").strip()
        visible = await is_project_visible_to_user(
            db,
            project_id=pid,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            raise HTTPException(status_code=404, detail="Task not found")
    await _assert_project_writable(db, getattr(task, "project_id", "") or "")

    _ensure_collection_jobs_device_id_column()
    _ensure_collection_jobs_validation_report_json_column()
    job_device_id = (getattr(job_in, "device_id", None) or "").strip() or None

    # 并发安全创建：依赖(task_id, job_number)唯一索引 + 失败重试
    job = None
    for _ in range(5):
        max_stmt = select(func.max(CollectionJobAsset.job_number)).where(CollectionJobAsset.task_id == str(job_in.task_id))
        max_r = await db.execute(max_stmt)
        next_no = int(max_r.scalar() or 0) + 1

        task_pid = (getattr(task, "project_id", None) or "").strip()
        task_pname = (getattr(task, "project_name", None) or "").strip() or None
        resolved_job_pname = await _canonical_project_name(db, task_pid, task_pname)
        job = CollectionJobAsset(
            id=str(uuid.uuid4()),
            task_id=str(job_in.task_id),
            job_number=next_no,
            operator_name=job_in.operator_name,
            status=job_in.status or "PENDING",
            collection_quantity=job_in.collection_quantity or 0,
            completed_count=job_in.completed_count or 0,
            # 后台执行层：作业归属必须继承“任务绑定项目”，不要信任前端可注入的 project_id/project_name。
            project_id=task_pid,
            project_name=resolved_job_pname,
            progress=0,
            device_id=job_device_id,
        )
        db.add(job)
        try:
            await db.commit()
            await db.refresh(job)
            break
        except IntegrityError:
            await db.rollback()
            job = None
    if job is None:
        raise HTTPException(status_code=409, detail="作业创建冲突，请重试")

    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.START_TASK,
        project_id=job.project_id,
        project_name=job.project_name,
        resource_type=AR.COLLECTION_JOB,
        resource_id=str(job.id),
        resource_name=f"采集作业 #{job.job_number or ''}".strip(),
        detail_json={"task_id": str(job.task_id), "job_number": job.job_number},
    )
    return ApiResponse(
        ok=True,
        data=_job_to_dict(job)
    )

@router.post("/{job_id}/cancel", response_model=ApiResponse[Dict[str, Any]])
async def cancel_job(
    job_id: UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """取消/急停作业"""
    job = await db.get(CollectionJobAsset, str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not is_super_admin(getattr(current_user, "role", None)):
        pid = (getattr(job, "project_id", None) or "").strip()
        visible = await is_project_visible_to_user(
            db,
            project_id=pid,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            raise HTTPException(status_code=404, detail="Job not found")
        
    # Set cancellation flag
    job_cancellation_flags[str(job_id)] = True
    
    # Update status in DB
    job.status = "CANCELED"
    await db.commit()
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.STOP_TASK,
        project_id=getattr(job, "project_id", None),
        project_name=getattr(job, "project_name", None),
        resource_type=AR.COLLECTION_JOB,
        resource_id=str(job_id),
        resource_name=f"采集作业 #{getattr(job, 'job_number', '') or ''}".strip(),
        detail_json={"task_id": str(getattr(job, "task_id", "") or "")},
    )
    await job_ws_manager.broadcast_log(str(job_id), "User requested Emergency Stop.")
    
    return ApiResponse(
        ok=True,
        data={"id": job_id, "status": "CANCELED"}
    )

@router.websocket("/{job_id}/ws")
async def websocket_job_progress(
    websocket: WebSocket,
    job_id: str,
    current_user: User = Depends(get_current_user_ws),
    db: AsyncSession = Depends(get_data_assets_db),
):
    job = await db.get(CollectionJobAsset, str(job_id))
    if not job:
        await websocket.close(code=1008)
        return
    if not is_super_admin(getattr(current_user, "role", None)):
        pid = (getattr(job, "project_id", None) or "").strip()
        visible = await is_project_visible_to_user(
            db,
            project_id=pid,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            await websocket.close(code=1008)
            return

    await job_ws_manager.connect(websocket, job_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        job_ws_manager.disconnect(websocket, job_id)

@router.get("/{job_id}/files", response_model=ApiResponse[List[Dict[str, Any]]])
async def list_job_files(
    job_id: UUID,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """获取作业生成的文件列表"""
    job = await db.get(CollectionJobAsset, str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not is_super_admin(getattr(current_user, "role", None)):
        pid = (getattr(job, "project_id", None) or "").strip()
        visible = await is_project_visible_to_user(
            db,
            project_id=pid,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            raise HTTPException(status_code=404, detail="Job not found")
    # Mock file listing
    # In real world, list files in job.storage_path
    
    files = [
        {"name": "metadata.json", "size": 1024, "type": "json"},
        {"name": "data.mcap", "size": 1024 * 1024 * 50, "type": "mcap"},
        {"name": "sensor_config.yaml", "size": 2048, "type": "yaml"}
    ]
    
    return ApiResponse(
        ok=True,
        data=files
    )

@router.get("/{job_id}", response_model=ApiResponse)
async def read_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """获取作业详情"""
    _ensure_collection_jobs_device_id_column()
    _ensure_collection_jobs_validation_report_json_column()
    job = await db.get(CollectionJobAsset, str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not is_super_admin(getattr(current_user, "role", None)):
        pid = (getattr(job, "project_id", None) or "").strip()
        visible = await is_project_visible_to_user(
            db,
            project_id=pid,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            raise HTTPException(status_code=404, detail="Job not found")
    return ApiResponse(
        ok=True,
        data=_job_to_dict(job)
    )

@router.patch("/{job_id}", response_model=ApiResponse)
async def update_job_by_id(
    job_id: UUID,
    job_update: JobUpdate,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """更新作业"""
    db_job = await db.get(CollectionJobAsset, str(job_id))
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    old_status = (db_job.status or "").strip().upper()
    if not is_super_admin(getattr(current_user, "role", None)):
        pid0 = (getattr(db_job, "project_id", None) or "").strip()
        visible0 = await is_project_visible_to_user(
            db,
            project_id=pid0,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible0:
            raise HTTPException(status_code=404, detail="Job not found")

    update_data = job_update.model_dump(exclude_unset=True)

    allow_reset = _is_explicit_progress_reset_payload(update_data)
    desired_current: Optional[int] = None
    desired_total: Optional[int] = None

    if isinstance(update_data.get("progress"), dict):
        raw = update_data.get("progress") or {}
        current = raw.get("current")
        total = raw.get("total")
        desired_current = int(current) if current is not None else None
        desired_total = int(total) if total is not None else None
        update_data.pop("progress", None)

    if "completed_count" in update_data:
        try:
            desired_current = int(update_data.get("completed_count") or 0)
        except Exception:
            desired_current = 0
    if "collection_quantity" in update_data:
        try:
            desired_total = int(update_data.get("collection_quantity") or 0)
        except Exception:
            desired_total = 0

    existing_current = int(getattr(db_job, "completed_count", 0) or 0)
    existing_total = int(getattr(db_job, "collection_quantity", 0) or 0)
    attempt_current = existing_current if desired_current is None else int(desired_current)
    attempt_total = existing_total if desired_total is None else int(desired_total)

    protect_total_regression = (not allow_reset) and (
        existing_current > 0 or old_status in ("RUNNING", "COMPLETED", "SUCCEEDED", "FAILED", "CANCELED")
    )
    next_current, next_total, percent, blocked_current, blocked_total = _apply_progress_guard(
        existing_current=existing_current,
        existing_total=existing_total,
        existing_percent=int(getattr(db_job, "progress", 0) or 0),
        desired_current=desired_current,
        desired_total=desired_total,
        allow_reset=allow_reset,
        protect_total_regression=protect_total_regression,
    )

    if blocked_current:
        logger.error(
            "COLLECT_PROGRESS_REGRESSION job_id=%s task_id=%s old=%s/%s new=%s/%s user_id=%s",
            str(db_job.id),
            str(getattr(db_job, "task_id", "") or ""),
            existing_current,
            existing_total,
            int(attempt_current),
            int(attempt_total),
            str(getattr(current_user, "id", "") or ""),
        )
        _record_progress_anomaly(
            {
                "ts": datetime.utcnow().isoformat(),
                "jobId": str(db_job.id),
                "taskId": str(getattr(db_job, "task_id", "") or ""),
                "projectId": (getattr(db_job, "project_id", None) or "").strip() or None,
                "old": {"current": existing_current, "total": existing_total},
                "attempt": {"current": int(attempt_current), "total": int(attempt_total)},
                "userId": str(getattr(current_user, "id", "") or "") or None,
            }
        )
        enqueue_audit_log(
            background_tasks,
            user=current_user,
            request=request,
            action_type=AA.UPDATE_TASK,
            project_id=(getattr(db_job, "project_id", None) or "").strip() or None,
            project_name=getattr(db_job, "project_name", None),
            resource_type=AR.COLLECTION_JOB,
            resource_id=str(db_job.id),
            resource_name=f"采集作业 #{getattr(db_job, 'job_number', '')}".strip(),
            detail_json={
                "anomaly": "progress_regression_blocked",
                "task_id": str(getattr(db_job, "task_id", "") or ""),
                "old": {"current": existing_current, "total": existing_total},
                "attempt": {"current": int(attempt_current), "total": int(attempt_total)},
            },
        )

    if blocked_total:
        logger.warning(
            "COLLECT_TOTAL_REGRESSION_BLOCKED job_id=%s task_id=%s old_total=%s attempt_total=%s user_id=%s",
            str(db_job.id),
            str(getattr(db_job, "task_id", "") or ""),
            existing_total,
            int(attempt_total),
            str(getattr(current_user, "id", "") or ""),
        )

    if percent >= 100 and str(update_data.get("status") or "").strip().upper() not in ("COMPLETED", "SUCCEEDED"):
        update_data["status"] = "COMPLETED"

    update_data["completed_count"] = next_current
    update_data["collection_quantity"] = next_total
    update_data["progress"] = percent

    _ensure_collection_jobs_device_id_column()
    _ensure_collection_jobs_validation_report_json_column()
    for field in [
        "operator_name",
        "status",
        "mcap_path",
        "mcap_size_bytes",
        "validation_report_json",
        "duration_sec",
        "progress",
        "collection_quantity",
        "completed_count",
        "project_id",
        "project_name",
        "device_id",
    ]:
        if field in update_data:
            setattr(db_job, field, update_data[field])
    if "project_id" in update_data:
        pid = (update_data.get("project_id") or "").strip()
        if pid:
            await _assert_project_writable(db, pid)
            if not is_super_admin(getattr(current_user, "role", None)):
                visible = await is_project_visible_to_user(
                    db,
                    project_id=pid,
                    user_id=str(getattr(current_user, "id", "") or ""),
                    include_owner_projects=True,
                )
                if not visible:
                    raise HTTPException(status_code=404, detail="项目不存在")
    if "started_at" in update_data:
        db_job.started_at = _parse_dt(update_data.get("started_at"))
    if "finished_at" in update_data:
        db_job.finished_at = _parse_dt(update_data.get("finished_at"))
    pid_final = (getattr(db_job, "project_id", None) or "").strip()
    if pid_final:
        pn_resolved = await _canonical_project_name(
            db,
            pid_final,
            getattr(db_job, "project_name", None),
        )
        if pn_resolved:
            db_job.project_name = pn_resolved
    new_status = (db_job.status or "").strip().upper()
    await db.commit()
    await db.refresh(db_job)
    _audit_collection_job_lifecycle(
        background_tasks, request, current_user, db_job, old_status, new_status
    )
    
    # 登记采集数据资产：仅当客户端显式 register_collect_asset=true 且 mcap_path 非空（用户点击「保存数据」等）。
    # Agent /OUTPUT_PATH 仅写 jobs.mcap_path，不会走此处；避免中途中断或多次 PATCH 误登记。
    reg_flag = bool(getattr(job_update, "register_collect_asset", False))
    raw_mcap_for_asset = (job_update.mcap_path or "").strip() if job_update.mcap_path is not None else ""
    if reg_flag and raw_mcap_for_asset:
        new_asset_created = False
        mcap_path_for_count: str | None = None
        # 远程采集场景：完全信任前端/Agent 上传的路径和大小信息，
        # 不再在平台本机访问文件系统或解析 MCAP 内容。
        # 同时，为兼容部分采集端仅上传“目录路径”的场景，这里会在本机可见时
        # 尝试从目录中解析唯一的 .mcap 文件作为真正的文件路径写入 data_assets.file_path。
        try:
            raw_path = raw_mcap_for_asset
            mcap_path_raw = os.path.normpath(raw_path.replace("\\", "/").strip())
            mcap_path = mcap_path_raw
            # 标记：当传入的是「目录且包含多个 .mcap 文件」时，避免把目录本身登记成一条伪“文件”资产。
            skip_asset_registration = False

            # 如果路径指向目录，且目录中只有一个 .mcap 文件，则自动解析为该文件路径
            try:
                if os.path.isdir(mcap_path):
                    mcap_files = []
                    for entry in os.listdir(mcap_path):
                        full = os.path.join(mcap_path, entry)
                        if os.path.isfile(full) and entry.lower().endswith(".mcap"):
                            mcap_files.append(full)
                    # 仅在存在且唯一的情况下自动替换，避免多文件目录带来歧义
                    if len(mcap_files) == 1:
                        mcap_path = os.path.normpath(mcap_files[0])
                    elif len(mcap_files) > 1:
                        # 目录下存在多个 .mcap 文件时，不再把「上一级目录」当作单个资产登记，
                        # 否则会在数据资产列表中出现一条名称为目录名的“伪文件”记录。
                        # 此类场景通常由采集端自行针对每个 episode 上报具体文件路径。
                        skip_asset_registration = True
            except Exception:
                # 解析目录失败时保持原始路径，后续逻辑仍按原值处理
                pass

            if skip_asset_registration:
                # 已完成作业状态更新，但本次不创建/更新数据资产记录。
                return ApiResponse(
                    ok=True,
                    data=_job_to_dict(db_job),
                )

            filename = _normalize_collect_mcap_filename(os.path.basename(mcap_path))
            file_size = int(job_update.mcap_size_bytes or 0)
            # 兜底：前端未上报大小时，尽量从本机文件系统读取实际大小。
            # 远程采集路径在平台不可见时会保持 0（列表展示为 "—"）。
            if file_size <= 0:
                try:
                    if os.path.isfile(mcap_path):
                        file_size = int(os.path.getsize(mcap_path))
                except Exception:
                    pass
            if file_size <= 0:
                # 再兜底：文件可能还在落盘，异步重试回填，避免连续采集前两条大小为 "—"。
                background_tasks.add_task(_backfill_collect_asset_size_by_path, mcap_path)
            # 后台产物写 data_assets 时，project_id 必须继承“作业绑定项目”
            # 避免出现已登录但 job/project 上下文缺失导致 project_id 落空的问题。
            project_id = (getattr(job_update, "project_id", None) or "").strip() or (getattr(db_job, "project_id", None) or "").strip()
            project_name = await _canonical_project_name(
                db,
                project_id,
                getattr(job_update, "project_name", None),
                getattr(db_job, "project_name", None),
            ) or project_id
            await _assert_project_writable(db, project_id)

            collect_task_name_for_asset: Optional[str] = None
            _ctask_id = (getattr(db_job, "task_id", None) or "").strip()
            if _ctask_id:
                try:
                    ct_row = await db.get(CollectionTaskAsset, _ctask_id)
                    if ct_row is not None:
                        tnm = (getattr(ct_row, "name", None) or "").strip()
                        if tnm:
                            collect_task_name_for_asset = tnm
                except Exception:
                    collect_task_name_for_asset = None

            hw_collect = (
                (getattr(job_update, "hardware_uuid", None) or "") or (getattr(job_update, "mac_address", None) or "")
            ).strip()
            payload_dev = (getattr(job_update, "device_id", None) or "").strip()
            job_dev = (getattr(db_job, "device_id", None) or "").strip()
            # 保存数据资产时优先用本次 PATCH 的 device_id，否则用作业领取时落库的平台设备 id（避免沿用任务/采集端侧的旧编号）
            dev_collect = payload_dev or job_dev

            # 解析设备名称，便于在 data_assets.meta.collect 中持久化人类可读字段
            device_name: Optional[str] = None
            if dev_collect:
                try:
                    dev_id_int = int(dev_collect)
                except ValueError:
                    dev_id_int = None
                if dev_id_int is not None:
                    try:
                        from app.crud.device import get_device_by_id  # 局部导入避免循环依赖

                        dev_obj = await get_device_by_id(db, dev_id_int)
                        if dev_obj is not None:
                            device_name = getattr(dev_obj, "name", None)
                    except Exception:
                        # 设备查询失败不影响采集资产登记
                        device_name = None

            def _collect_block() -> dict:
                c = {
                    "job_id": str(job_id),
                    "task_id": str(getattr(db_job, "task_id", "")),
                    "operator_name": getattr(db_job, "operator_name", None),
                    "project_id": project_id,
                    "project_name": project_name,
                }
                if hw_collect:
                    c["hardware_uuid"] = hw_collect
                if dev_collect:
                    c["device_id"] = dev_collect
                if device_name:
                    c["device_name"] = device_name
                return c

            mcap_path_for_count = mcap_path
            existing_asset = await get_asset_by_file_path(db, mcap_path)
            if existing_asset:
                asset_id = existing_asset.id
                existing_asset.filename = _normalize_collect_mcap_filename(filename) or existing_asset.filename
                existing_asset.file_size_bytes = file_size
                existing_asset.format = "mcap"
                existing_asset.file_path = mcap_path
                # 如果同一 file_path 已绑定到其它项目，拒绝重绑定，避免把资产写进不该写的项目。
                existing_pid = (getattr(existing_asset, "project_id", None) or "").strip()
                if existing_pid and existing_pid != project_id:
                    raise HTTPException(status_code=409, detail="数据路径已绑定其他项目，拒绝重绑定")
                existing_asset.project_id = project_id
                existing_asset.project_name = project_name or existing_asset.project_name
                existing_asset.source = "collect"
                existing_asset.sync_status = "unsynced"
                existing_asset.sync_error = "采集数据尚未同步到一体机存储"
                if not (getattr(existing_asset, "operator_name", None) or "").strip():
                    existing_asset.operator_name = (
                        (getattr(db_job, "operator_name", None) or "").strip() or None
                    )
                if collect_task_name_for_asset:
                    existing_asset.collect_task_name = collect_task_name_for_asset
                await db.commit()
            else:
                new_asset_created = True
                code = await next_asset_code(db)
                create_data = DataAssetCreate(
                    code=code,
                    filename=_normalize_collect_mcap_filename(filename),
                    format="mcap",
                    source="collect",
                    project_id=project_id,
                    project_name=project_name,
                    file_path=mcap_path,
                    file_size_bytes=file_size,
                    meta=None,
                    parse_status="未解析",
                    error_msg=None,
                    sync_status="unsynced",
                    sync_error="采集数据尚未同步到一体机存储",
                    device_id=dev_collect or None,
                    operator_name=(getattr(db_job, "operator_name", None) or "").strip() or None,
                    collect_task_name=collect_task_name_for_asset,
                )
                asset = await create_asset(db, create_data)
                asset_id = asset.id

            # 远端采集时路径往往在采集端本机，平台进程上可能不存在该文件
            file_exists = os.path.isfile(mcap_path)
            if file_exists:
                meta_json, parse_status, err_msg = parse_meta_for_asset(mcap_path, "mcap")
                try:
                    base_meta = json.loads(meta_json) if meta_json else {}
                except Exception:
                    base_meta = {}
                base_meta["collect"] = _collect_block()
                await update_asset(
                    db,
                    asset_id,
                    parse_status=parse_status,
                    error_msg=err_msg,
                    meta=json.dumps(base_meta, ensure_ascii=False),
                    device_id=dev_collect if dev_collect else None,
                    collect_task_name=collect_task_name_for_asset,
                )
            else:
                await update_asset(
                    db,
                    asset_id,
                    parse_status="未解析",
                    error_msg="文件不存在，等待落盘后再解析",
                    meta=json.dumps(
                        {"collect": _collect_block()},
                        ensure_ascii=False,
                    ),
                    device_id=dev_collect if dev_collect else None,
                    collect_task_name=collect_task_name_for_asset,
                )
        except HTTPException:
            raise
        except Exception as e:
            print(f"ERROR: Failed to register data asset for job {job_id}: {e}")

        # 仅当本次 PATCH 未携带进度字段时，才在「新建资产」后自动 +1；避免与上文已提交的 completed_count 重复累加。
        if new_asset_created and desired_current is None and desired_total is None:
            try:
                prev_current = int(getattr(db_job, "completed_count", 0) or 0)
                prev_total = int(getattr(db_job, "collection_quantity", 0) or 0)
                desired_inc = prev_current + 1
                next_current2, next_total2, percent2, _, _ = _apply_progress_guard(
                    existing_current=prev_current,
                    existing_total=prev_total,
                    existing_percent=int(getattr(db_job, "progress", 0) or 0),
                    desired_current=desired_inc,
                    desired_total=None,
                    allow_reset=False,
                    protect_total_regression=True,
                )
                db_job.completed_count = next_current2
                db_job.collection_quantity = next_total2
                db_job.progress = percent2
                if percent2 >= 100 and (db_job.status or "").strip().upper() not in ("COMPLETED", "SUCCEEDED"):
                    db_job.status = "COMPLETED"
                await db.commit()
                await db.refresh(db_job)
            except Exception:
                await db.rollback()

        if mcap_path_for_count:
            try:
                probe_id = str(job_id)
                actual_count = (
                    await db.execute(
                        select(func.count(DataAsset.id)).where(
                            DataAsset.source == "collect",
                            DataAsset.meta.is_not(None),
                            DataAsset.meta.contains("job_id"),
                            DataAsset.meta.contains(probe_id),
                        )
                    )
                ).scalar_one() or 0
                actual_count = int(actual_count)
                cur = int(getattr(db_job, "completed_count", 0) or 0)
                if actual_count > cur:
                    prev_total = int(getattr(db_job, "collection_quantity", 0) or 0)
                    next_current3, next_total3, percent3, _, _ = _apply_progress_guard(
                        existing_current=cur,
                        existing_total=prev_total,
                        existing_percent=int(getattr(db_job, "progress", 0) or 0),
                        desired_current=actual_count,
                        desired_total=None,
                        allow_reset=False,
                        protect_total_regression=True,
                    )
                    db_job.completed_count = next_current3
                    db_job.collection_quantity = next_total3
                    db_job.progress = percent3
                    if percent3 >= 100 and (db_job.status or "").strip().upper() not in ("COMPLETED", "SUCCEEDED"):
                        db_job.status = "COMPLETED"
                    await db.commit()
                    await db.refresh(db_job)
                elif actual_count < cur:
                    logger.error(
                        "COLLECT_COUNT_MISMATCH job_id=%s current=%s actual=%s",
                        str(job_id),
                        cur,
                        actual_count,
                    )
            except Exception:
                await db.rollback()

    return ApiResponse(
        ok=True,
        data=_job_to_dict(db_job)
    )

@router.delete("/{job_id}", response_model=ApiResponse)
async def delete_job_by_id(
    job_id: UUID,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """删除作业；并在采集端删除该作业对应输出目录（任务 storagePath + 四位作业编号）。"""
    job = await db.get(CollectionJobAsset, str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not is_super_admin(getattr(current_user, "role", None)):
        pid = (getattr(job, "project_id", None) or "").strip()
        visible = await is_project_visible_to_user(
            db,
            project_id=pid,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            raise HTTPException(status_code=404, detail="Job not found")

    tid = str(getattr(job, "task_id", "") or "").strip()
    task_row = await db.get(CollectionTaskAsset, tid) if tid else None
    task_desc = getattr(task_row, "description", None) if task_row else None
    workspace_path = resolve_collect_job_workspace_path(
        task_desc,
        int(getattr(job, "job_number", 0) or 0),
    )
    dev_raw = getattr(job, "device_id", None)
    dev_str = str(dev_raw).strip() if dev_raw is not None else ""
    try:
        ok_del, err_del = await delete_collect_job_workspace_remote(
            device_id_str=dev_str or None,
            workspace_path=workspace_path,
        )
        if not ok_del:
            logger.warning(
                "delete_job: 采集端作业目录删除失败 job_id=%s workspace=%s device_id=%s err=%s",
                job_id,
                workspace_path,
                dev_str or None,
                err_del,
            )
    except Exception as exc:
        logger.warning(
            "delete_job: 采集端作业目录删除异常 job_id=%s workspace=%s err=%s",
            job_id,
            workspace_path,
            exc,
        )

    await db.delete(job)
    await db.commit()
    return ApiResponse(
        ok=True,
        data={"id": str(job_id)}
    )
