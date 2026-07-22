from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models.data_asset import TaskJob
from app.services.asset_registration_service import DataAssetsSyncSessionLocal


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_task_job(
    *,
    task_id: str,
    task_type: str,
    status: str,
    user_id: Optional[str],
    queue_name: Optional[str],
    payload: Dict[str, Any],
) -> None:
    session = DataAssetsSyncSessionLocal()
    try:
        row = TaskJob(
            id=str(task_id),
            task_type=str(task_type),
            status=str(status),
            user_id=(str(user_id) if user_id else None),
            queue_name=(str(queue_name) if queue_name else None),
            payload=payload,
        )
        session.add(row)
        session.commit()
    except IntegrityError:
        # 同一 task_id 已存在（例如 API 重启后 recover 再次 dispatch）
        session.rollback()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def update_task_status(
    task_id: str,
    status: str,
    *,
    rq_job_id: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
) -> None:
    session = DataAssetsSyncSessionLocal()
    try:
        row = session.query(TaskJob).filter(TaskJob.id == str(task_id)).one_or_none()
        if row is None:
            return
        # 取消态优先：除非本次就是写 cancelled，否则不允许覆盖 cancelled
        if (row.status or "").strip().lower() == "cancelled" and str(status).strip().lower() != "cancelled":
            return
        row.status = str(status)
        if rq_job_id is not None:
            row.rq_job_id = str(rq_job_id)
        if result is not None:
            row.result = result
        if error is not None:
            row.error = str(error)[:2000]
        if started_at is not None:
            row.started_at = started_at
        if finished_at is not None:
            row.finished_at = finished_at
        row.updated_at = _now()
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def is_cancelled(task_id: str) -> bool:
    row = get_task_job(task_id)
    if row is None:
        return False
    return (row.status or "").strip().lower() == "cancelled"


def get_task_job(task_id: str) -> Optional[TaskJob]:
    session = DataAssetsSyncSessionLocal()
    try:
        return session.query(TaskJob).filter(TaskJob.id == str(task_id)).one_or_none()
    finally:
        session.close()


def delete_task_job(task_id: str) -> None:
    """物理删除 TaskJob 行（导出/转换删除结果时与内存任务一并清理）。"""
    tid = str(task_id or "").strip()
    if not tid:
        return
    session = DataAssetsSyncSessionLocal()
    try:
        row = session.query(TaskJob).filter(TaskJob.id == tid).one_or_none()
        if row is None:
            return
        session.delete(row)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_task_jobs(*, user_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[TaskJob]:
    session = DataAssetsSyncSessionLocal()
    try:
        stmt = select(TaskJob).order_by(TaskJob.created_at.desc()).offset(max(0, int(offset))).limit(max(1, int(limit)))
        if user_id:
            stmt = stmt.where(TaskJob.user_id == str(user_id))
        rows = session.execute(stmt).scalars().all()
        return list(rows)
    finally:
        session.close()

