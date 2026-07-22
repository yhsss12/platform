from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from uuid import UUID
from datetime import datetime
from app.models.run import Run
from app.schemas.run import RunCreate, RunUpdate


async def get_runs(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Run]:
    """获取运行列表"""
    result = await db.execute(select(Run).offset(skip).limit(limit))
    return list(result.scalars().all())


async def get_run(db: AsyncSession, run_id: UUID) -> Run | None:
    """根据 ID 获取运行"""
    result = await db.execute(select(Run).where(Run.id == run_id))
    return result.scalar_one_or_none()


async def create_run(db: AsyncSession, run: RunCreate) -> Run:
    """创建运行"""
    now = datetime.utcnow().isoformat()
    db_run = Run(
        task_id=UUID(run.task_id),
        status=run.status,
        created_at=now,
        updated_at=now
    )
    db.add(db_run)
    await db.commit()
    await db.refresh(db_run)
    return db_run


async def update_run(db: AsyncSession, run_id: UUID, run_update: RunUpdate) -> Run | None:
    """更新运行"""
    db_run = await get_run(db, run_id)
    if db_run is None:
        return None
    
    update_data = run_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_run, field, value)
    db_run.updated_at = datetime.utcnow().isoformat()
    
    await db.commit()
    await db.refresh(db_run)
    return db_run


async def delete_run(db: AsyncSession, run_id: UUID) -> bool:
    """删除运行"""
    db_run = await get_run(db, run_id)
    if db_run is None:
        return False
    
    await db.delete(db_run)
    await db.commit()
    return True


