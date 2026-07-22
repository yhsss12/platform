from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from app.models.job import Job
from app.schemas.job import JobCreate, JobUpdate
from app.models.task import Task


async def get_jobs(
    db: AsyncSession,
    task_id: Optional[UUID] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Job]:
    """获取作业列表"""
    query = select(Job)
    if task_id:
        query = query.where(Job.task_id == task_id)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_job(db: AsyncSession, job_id: UUID) -> Job | None:
    """根据 ID 获取作业"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def create_job(db: AsyncSession, job: JobCreate) -> Job:
    """创建作业"""
    now = datetime.utcnow().isoformat()
    
    # Calculate next job number for this task
    stmt = select(func.max(Job.job_number)).where(Job.task_id == job.task_id)
    result = await db.execute(stmt)
    max_num = result.scalar() or 0
    new_job_number = max_num + 1

    db_job = Job(
        task_id=job.task_id,
        job_number=new_job_number,
        operator_name=job.operator_name,
        status=job.status,
        collection_quantity=job.collection_quantity,
        completed_count=job.completed_count,
        project_id=(job.project_id or "").strip() or None,
        project_name=(job.project_name or "").strip() or None,
        created_at=now,
        updated_at=now
    )
    if not db_job.project_id:
        task_obj = await db.get(Task, job.task_id)
        if task_obj:
            db_job.project_id = getattr(task_obj, "project_id", None)
            db_job.project_name = getattr(task_obj, "project_name", None)
    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)
    return db_job


async def delete_job(db: AsyncSession, job_id: UUID) -> bool:
    """删除作业"""
    db_job = await get_job(db, job_id)
    if db_job is None:
        return False
    await db.delete(db_job)
    await db.commit()
    return True


async def update_job(db: AsyncSession, job_id: UUID, job_update: JobUpdate) -> Job | None:
    """更新作业"""
    db_job = await get_job(db, job_id)
    if db_job is None:
        return None
    
    update_data = job_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_job, field, value)
    db_job.updated_at = datetime.utcnow().isoformat()
    
    await db.commit()
    await db.refresh(db_job)
    return db_job

