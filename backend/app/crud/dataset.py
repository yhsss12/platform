from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from uuid import UUID
from datetime import datetime
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUpdate


async def get_datasets(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Dataset]:
    """获取数据集列表"""
    result = await db.execute(select(Dataset).offset(skip).limit(limit))
    return list(result.scalars().all())


async def get_dataset(db: AsyncSession, dataset_id: UUID) -> Dataset | None:
    """根据 ID 获取数据集"""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    return result.scalar_one_or_none()


async def create_dataset(db: AsyncSession, dataset: DatasetCreate) -> Dataset:
    """创建数据集"""
    now = datetime.utcnow().isoformat()
    db_dataset = Dataset(
        name=dataset.name,
        status=dataset.status,
        created_at=now,
        updated_at=now
    )
    db.add(db_dataset)
    await db.commit()
    await db.refresh(db_dataset)
    return db_dataset


async def update_dataset(db: AsyncSession, dataset_id: UUID, dataset_update: DatasetUpdate) -> Dataset | None:
    """更新数据集"""
    db_dataset = await get_dataset(db, dataset_id)
    if db_dataset is None:
        return None
    
    update_data = dataset_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_dataset, field, value)
    db_dataset.updated_at = datetime.utcnow().isoformat()
    
    await db.commit()
    await db.refresh(db_dataset)
    return db_dataset


async def delete_dataset(db: AsyncSession, dataset_id: UUID) -> bool:
    """删除数据集"""
    db_dataset = await get_dataset(db, dataset_id)
    if db_dataset is None:
        return False
    
    await db.delete(db_dataset)
    await db.commit()
    return True


