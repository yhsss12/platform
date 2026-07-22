import logging
import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.core.deps import get_current_user
from app.core.roles import is_super_admin_or_team_admin, is_super_admin
from app.core.project_permissions import can_manage_project_tasks
from sqlalchemy import select, func
from app.db.data_assets_session import get_data_assets_db
from app.core.data_asset_access import data_assets_allowed_project_ids
from app.models.data_asset import CollectionTaskAsset
from app.models.data_asset import CollectionJobAsset
from app.models.project_asset import Project
from app.crud.project import get_visible_project_ids, is_project_visible_to_user
from app.schemas.task import TaskCreate, TaskUpdate, TaskResponse
from app.schemas.common import ApiResponse
from app.models.user import User
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import enqueue_audit_log
from app.models.data_asset import TaskJob
from app.services.task_job_store import update_task_status
from app.services.task_queue import redis_conn
from rq.job import Job
from app.api.routes_conversion import jobs_store as conversion_jobs_store
from app.models.data_asset import ConversionJobAsset
from app.api.routes_data_assets import _export_jobs, _export_jobs_lock
from app.services.collect_storage_layout import resolve_task_storage_full_path

router = APIRouter()
logger = logging.getLogger(__name__)

def _dt_iso(dt) -> str:
    if not dt:
        return ""
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)

def _task_to_response_obj(t: CollectionTaskAsset) -> TaskResponse:
    payload: dict = {
        "id": str(t.id),
        "name": t.name,
        "status": t.status,
        "created_at": _dt_iso(getattr(t, "created_at", None)),
        "updated_at": _dt_iso(getattr(t, "updated_at", None)),
        "createdAt": _dt_iso(getattr(t, "created_at", None)),
        "updatedAt": _dt_iso(getattr(t, "updated_at", None)),
        "description": t.description,
        "projectId": getattr(t, "project_id", None),
        "projectName": getattr(t, "project_name", None),
    }
    if t.description:
        try:
            cfg = json.loads(t.description)
            if isinstance(cfg, dict):
                if "_text" in cfg:
                    payload["description"] = cfg.get("_text")
                for k, v in cfg.items():
                    if k != "_text":
                        payload[k] = v
        except Exception:
            pass
    return TaskResponse.model_validate(payload)

def _merge_creator_into_description(desc_json: str | None, user: User) -> str | None:
    cfg: dict = {}
    if desc_json:
        try:
            loaded = json.loads(desc_json)
            if isinstance(loaded, dict):
                cfg = loaded
        except Exception:
            cfg = {}
    uid = str(getattr(user, "id", "") or "").strip()
    if uid and "creatorId" not in cfg:
        cfg["creatorId"] = uid
    uname = (getattr(user, "username", None) or "").strip()
    if uname and "creatorUsername" not in cfg:
        cfg["creatorUsername"] = uname
    acc = (getattr(user, "account_id", None) or "").strip()
    if acc and "creatorAccountId" not in cfg:
        cfg["creatorAccountId"] = acc
    return json.dumps(cfg, ensure_ascii=False) if cfg else None

def _build_task_description(task: TaskCreate | TaskUpdate, existing_json: str | None = None) -> str | None:
    cfg: dict = {}
    if existing_json:
        try:
            loaded = json.loads(existing_json)
            if isinstance(loaded, dict):
                cfg = loaded
        except Exception:
            cfg = {}
    if getattr(task, "description", None) is not None:
        desc_value = getattr(task, "description", None)
        if desc_value:
            try:
                parsed = json.loads(desc_value)
                if isinstance(parsed, dict):
                    cfg.update(parsed)
                else:
                    cfg["_text"] = str(parsed)
            except Exception:
                cfg["_text"] = desc_value
        else:
            if "_text" in cfg:
                del cfg["_text"]

    for key in [
        "owner",
        "deviceId",
        "deviceName",
        "episodeCount",
        "durationSec",
        "storagePath",
        "storageTypes",
        "remark",
        "projectId",
        "projectName",
        "cameraDataFormat",
        "frequencyConfig",
    ]:
        if hasattr(task, key):
            value = getattr(task, key)
            if value is not None:
                cfg[key] = value

    return json.dumps(cfg, ensure_ascii=False) if cfg else None

async def _assert_project_task_manage(db: AsyncSession, current_user: User, project_id: str) -> Project:
    pid = (project_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="请选择所属项目")
    p = (await db.execute(select(Project).where(Project.id == pid))).scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="项目不存在")
    if (p.status or "").strip() == "已归档":
        raise HTTPException(status_code=403, detail="项目已归档，禁止该操作")
    if not await can_manage_project_tasks(db, current_user, p):
        raise HTTPException(status_code=403, detail="无权操作该项目下的任务")
    return p

@router.get("", response_model=ApiResponse)
async def list_tasks(
    skip: int = 0,
    limit: int = 100,
    project_id: str | None = None,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """获取任务列表

    - 默认返回所有采集任务（按更新时间倒序）
    - 如果提供 project_id，仅返回该项目下的采集任务
    - 当带 project_id 查询时，需满足：
      - 平台管理员，或
      - 项目 owner，或
      - 当前用户在该项目成员列表中（前端会把成员写入 Project 表的扩展字段）
    """
    stmt = select(CollectionTaskAsset).order_by(CollectionTaskAsset.updated_at.desc())

    pid = (project_id or "").strip()
    if pid:
        # 校验项目是否存在
        p = (await db.execute(select(Project).where(Project.id == pid))).scalar_one_or_none()
        if not p:
            raise HTTPException(status_code=404, detail="项目不存在")

        # 管理员直接放行
        if not is_super_admin_or_team_admin(getattr(current_user, "role", None)):
            visible = await is_project_visible_to_user(
                db,
                project_id=pid,
                user_id=str(getattr(current_user, "id", "") or ""),
                include_owner_projects=True,
            )
            if not visible:
                raise HTTPException(status_code=404, detail="项目不存在")

        stmt = stmt.where(CollectionTaskAsset.project_id == pid)
    else:
        if not is_super_admin_or_team_admin(getattr(current_user, "role", None)):
            allowed = await get_visible_project_ids(
                db,
                user_id=str(getattr(current_user, "id", "") or ""),
                include_owner_projects=True,
            )
            if not allowed:
                return ApiResponse(ok=True, data=[])
            stmt = stmt.where(CollectionTaskAsset.project_id.in_(allowed))

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    rows = list(result.scalars().all())

    allowed = await data_assets_allowed_project_ids(db, current_user)

    if is_super_admin(getattr(current_user, "role", None)):
        visible_rows = rows
    else:
        allowed_set = {str(x) for x in (allowed or [])}
        visible_rows = []
        for row in rows:
            row_project_id = getattr(row, "project_id", None)
            if row_project_id and str(row_project_id) in allowed_set:
                visible_rows.append(row)

    return ApiResponse(
        ok=True,
        data=[_task_to_response_obj(t) for t in visible_rows],
    )


@router.post("", response_model=ApiResponse)
async def create_new_task(
    task: TaskCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """创建任务"""
    normalized = (task.name or "").strip()
    if not normalized:
        return ApiResponse(ok=False, error="任务名称不能为空")

    # 强制要求绑定项目，防止产生“游离任务”
    project_id = (task.projectId or "").strip()
    if not project_id:
        raise HTTPException(status_code=400, detail="采集任务必须绑定所属项目")

    logger.info(
        "create_task request user=%s name=%s projectId=%s deviceId=%s",
        getattr(current_user, "username", None),
        getattr(task, "name", None),
        getattr(task, "projectId", None),
        getattr(task, "deviceId", None),
    )
    await _assert_project_task_manage(db, current_user, project_id)
    existing_stmt = select(CollectionTaskAsset).where(
        func.lower(CollectionTaskAsset.name) == func.lower(normalized)
    )
    existing_r = await db.execute(existing_stmt)
    existing = existing_r.scalar_one_or_none()
    if existing is not None:
        logger.info(
            "create_task rejected duplicate name=%s existing_id=%s",
            normalized,
            getattr(existing, "id", None),
        )
        return ApiResponse(ok=False, error="任务名称已存在，请修改后再创建")
    full_storage = resolve_task_storage_full_path(
        normalized,
        user_parent=(getattr(task, "storagePath", None) or "").strip() or None,
        existing_description_json=None,
        previous_task_name=None,
    )
    merged_fields = task.model_dump()
    merged_fields["storagePath"] = full_storage
    task_for_desc = TaskCreate(**merged_fields)
    db_task = CollectionTaskAsset(
        id=str(uuid.uuid4()),
        name=normalized,
        description=_merge_creator_into_description(_build_task_description(task_for_desc), current_user),
        status=task.status,
        project_id=project_id,
        project_name=(task.projectName or "").strip() or None,
    )
    db.add(db_task)
    await db.commit()
    await db.refresh(db_task)
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.CREATE_TASK,
        project_id=db_task.project_id,
        project_name=getattr(db_task, "project_name", None),
        resource_type=AR.TASK,
        resource_id=str(db_task.id),
        resource_name=db_task.name,
        detail_json={"domain": "collection"},
    )
    return ApiResponse(
        ok=True,
        data=_task_to_response_obj(db_task)
    )


def _task_job_to_dict(t: TaskJob) -> dict:
    return {
        "task_id": t.id,
        "rq_job_id": t.rq_job_id,
        "task_type": t.task_type,
        "status": t.status,
        "user_id": t.user_id,
        "queue_name": t.queue_name,
        "payload": t.payload,
        "result": t.result,
        "error": t.error,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "finished_at": t.finished_at.isoformat() if t.finished_at else None,
    }


@router.get("/jobs/{task_id}", response_model=ApiResponse)
async def get_task_job_status(
    task_id: str,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    stmt = select(TaskJob).where(TaskJob.id == task_id)
    row = (await db.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Task job not found")
    if not is_super_admin_or_team_admin(getattr(current_user, "role", None)):
        if str(getattr(row, "user_id", "") or "") != str(getattr(current_user, "id", "") or ""):
            raise HTTPException(status_code=404, detail="Task job not found")
    data = _task_job_to_dict(row)
    return ApiResponse(ok=True, data=data)


@router.get("/jobs", response_model=ApiResponse)
async def list_task_jobs_status(
    user_id: str | None = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    stmt = select(TaskJob).order_by(TaskJob.created_at.desc()).offset(skip).limit(limit)
    if is_super_admin_or_team_admin(getattr(current_user, "role", None)):
        if user_id:
            stmt = stmt.where(TaskJob.user_id == user_id)
    else:
        stmt = stmt.where(TaskJob.user_id == str(getattr(current_user, "id", "") or ""))
    rows = list((await db.execute(stmt)).scalars().all())
    return ApiResponse(ok=True, data=[_task_job_to_dict(x) for x in rows])


@router.post("/jobs/{task_id}/cancel", response_model=ApiResponse)
async def cancel_task_job(
    task_id: str,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    row = (await db.execute(select(TaskJob).where(TaskJob.id == task_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Task job not found")

    if not is_super_admin_or_team_admin(getattr(current_user, "role", None)):
        if str(getattr(row, "user_id", "") or "") != str(getattr(current_user, "id", "") or ""):
            raise HTTPException(status_code=403, detail="No permission")

    st = (row.status or "").strip().lower()
    if st not in ("pending", "queued", "running"):
        raise HTTPException(status_code=400, detail="Cannot cancel this task")

    update_task_status(task_id, "cancelled")
    print(f"[Cancel] Task {task_id} cancelled")

    # 同步更新各业务真源状态，避免前端刷新后被恢复为 running
    task_type = (row.task_type or "").strip().lower()
    if task_type == "export":
        with _export_jobs_lock:
            job = _export_jobs.get(task_id)
            if isinstance(job, dict):
                job["status"] = "cancelled"
                job["progress"] = 0
                if not (job.get("errorMessage") or "").strip():
                    job["errorMessage"] = "Task cancelled"
                if not (job.get("currentStep") or "").strip():
                    job["currentStep"] = "Task cancelled"
    elif task_type == "conversion":
        rec = (await db.execute(select(ConversionJobAsset).where(ConversionJobAsset.job_id == task_id))).scalar_one_or_none()
        if rec is not None:
            rec.status = "canceled"
            rec.error_message = "Task cancelled"
            rec.updated_at = datetime.now(timezone.utc)
            await db.commit()
        job = conversion_jobs_store.get(task_id)
        if job is not None:
            try:
                job.status = "canceled"
                job.errorMessage = "Task cancelled"
                job.updatedAt = datetime.now().isoformat()
            except Exception:
                pass

    # 队列中尚未执行时，尝试从 RQ 取消
    try:
        job = Job.fetch(task_id, connection=redis_conn)
        job.cancel()
    except Exception:
        pass

    return ApiResponse(ok=True, data={"task_id": task_id, "status": "cancelled"})


@router.get("/{task_id}", response_model=ApiResponse)
async def get_task_by_id(
    task_id: UUID,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """根据 ID 获取任务"""
    db_task = await db.get(CollectionTaskAsset, str(task_id))
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if not is_super_admin_or_team_admin(getattr(current_user, "role", None)):
        pid = (getattr(db_task, "project_id", None) or "").strip()
        visible = await is_project_visible_to_user(
            db,
            project_id=pid,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(
        ok=True,
        data=_task_to_response_obj(db_task)
    )


@router.patch("/{task_id}", response_model=ApiResponse)
async def update_task_by_id(
    task_id: UUID,
    task_update: TaskUpdate,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """更新任务"""
    db_task = await db.get(CollectionTaskAsset, str(task_id))
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    pid0 = (getattr(db_task, "project_id", None) or "").strip()
    await _assert_project_task_manage(db, current_user, pid0)
    update_data = task_update.model_dump(exclude_unset=True)
    old_task_name = db_task.name
    if "name" in update_data and update_data["name"] is not None:
        db_task.name = str(update_data["name"]).strip()
    if "status" in update_data and update_data["status"] is not None:
        db_task.status = str(update_data["status"])

    if "projectId" in update_data:
        pid = (update_data.get("projectId") or "").strip()
        # 不允许把任务更新为“无项目”
        if not pid:
            raise HTTPException(status_code=400, detail="采集任务必须绑定所属项目")
        await _assert_project_task_manage(db, current_user, pid)
        db_task.project_id = pid
    if "projectName" in update_data:
        pname = update_data.get("projectName")
        db_task.project_name = (str(pname).strip() if pname is not None else None)

    if "description" in update_data or any(
        k in update_data
        for k in [
            "owner",
            "deviceId",
            "deviceName",
            "episodeCount",
            "durationSec",
            "storagePath",
            "storageTypes",
            "remark",
            "projectId",
            "projectName",
            "cameraDataFormat",
            "frequencyConfig",
        ]
    ):
        if "name" in update_data or "storagePath" in update_data:
            user_parent = None
            if "storagePath" in update_data:
                sp = update_data.get("storagePath")
                user_parent = (str(sp).strip() if sp is not None else "") or None
            full_storage = resolve_task_storage_full_path(
                db_task.name,
                user_parent=user_parent,
                existing_description_json=db_task.description,
                previous_task_name=old_task_name,
            )
            patch = task_update.model_dump(exclude_unset=True)
            patch["storagePath"] = full_storage
            db_task.description = _build_task_description(
                TaskUpdate(**patch), existing_json=db_task.description
            )
        else:
            db_task.description = _build_task_description(task_update, existing_json=db_task.description)

    await db.commit()
    await db.refresh(db_task)
    return ApiResponse(
        ok=True,
        data=_task_to_response_obj(db_task)
    )


@router.delete("/{task_id}", response_model=ApiResponse)
async def delete_task_by_id(
    task_id: UUID,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user)
):
    """删除任务"""
    db_task = await db.get(CollectionTaskAsset, str(task_id))
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    pid = (getattr(db_task, "project_id", None) or "").strip()
    await _assert_project_task_manage(db, current_user, pid)

    await db.execute(
        CollectionJobAsset.__table__.delete().where(CollectionJobAsset.task_id == str(task_id))
    )
    await db.delete(db_task)
    await db.commit()
    return ApiResponse(ok=True, data=None)
