"""
HDF5 数据集 CRUD 操作
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_, update
from typing import List, Optional
from app.models.hdf5_dataset import HDF5Dataset
from app.schemas.hdf5_dataset import HDF5DatasetCreate, HDF5DatasetUpdate, DatasetQueryParams


async def get_datasets(
    db: AsyncSession,
    params: DatasetQueryParams
) -> tuple[List[HDF5Dataset], int]:
    """获取数据集列表（支持筛选和分页）"""
    # 构建查询
    query = select(HDF5Dataset)
    count_query = select(func.count()).select_from(HDF5Dataset)
    
    conditions = []
    
    # 关键词搜索（文件名模糊匹配）
    if params.keyword:
        conditions.append(HDF5Dataset.name.like(f"%{params.keyword}%"))
    
    # 设备筛选
    if params.device:
        conditions.append(HDF5Dataset.device == params.device)
    
    # 项目筛选：精确匹配 project 字段（与项目管理 projectId / 项目名一致）
    if params.project:
        pv = (params.project or "").strip()
        if pv:
            conditions.append(HDF5Dataset.project == pv)
    
    # 格式筛选（hdf5 / mcap / lerobot，兼容大小写）
    if params.format:
        fmt = (params.format or "").strip().lower()
        if fmt:
            conditions.append(HDF5Dataset.format.ilike(fmt))
    
    # 状态筛选
    if params.qc_status:
        conditions.append(HDF5Dataset.qc_status == params.qc_status)
    if params.label_status:
        conditions.append(HDF5Dataset.label_status == params.label_status)
    if params.assign_status:
        conditions.append(HDF5Dataset.assign_status == params.assign_status)
    
    # 应用条件
    if conditions:
        where_clause = and_(*conditions)
        query = query.where(where_clause)
        count_query = count_query.where(where_clause)
    
    # 获取总数
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # 分页
    skip = (params.page - 1) * params.page_size
    query = query.order_by(HDF5Dataset.created_at.desc())
    query = query.offset(skip).limit(params.page_size)
    
    # 执行查询
    result = await db.execute(query)
    datasets = list(result.scalars().all())
    
    return datasets, total


async def get_dataset_by_id(db: AsyncSession, dataset_id: int) -> Optional[HDF5Dataset]:
    """根据 ID 获取数据集"""
    result = await db.execute(select(HDF5Dataset).where(HDF5Dataset.id == dataset_id))
    return result.scalar_one_or_none()


async def get_dataset_by_uri(db: AsyncSession, storage_uri: str) -> Optional[HDF5Dataset]:
    """根据存储路径获取数据集"""
    result = await db.execute(select(HDF5Dataset).where(HDF5Dataset.storage_uri == storage_uri))
    return result.scalar_one_or_none()


async def create_dataset(db: AsyncSession, dataset: HDF5DatasetCreate) -> HDF5Dataset:
    """创建数据集"""
    db_dataset = HDF5Dataset(**dataset.model_dump())
    db.add(db_dataset)
    await db.commit()
    await db.refresh(db_dataset)
    return db_dataset


async def update_dataset(
    db: AsyncSession,
    dataset_id: int,
    dataset_update: HDF5DatasetUpdate
) -> Optional[HDF5Dataset]:
    """更新数据集"""
    db_dataset = await get_dataset_by_id(db, dataset_id)
    if db_dataset is None:
        return None
    
    update_data = dataset_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_dataset, field, value)
    
    await db.commit()
    await db.refresh(db_dataset)
    return db_dataset


async def delete_dataset(db: AsyncSession, dataset_id: int) -> bool:
    """删除数据集"""
    db_dataset = await get_dataset_by_id(db, dataset_id)
    if db_dataset is None:
        return False
    
    await db.delete(db_dataset)
    await db.commit()
    return True


async def get_distinct_devices(db: AsyncSession) -> List[str]:
    """获取所有不同的设备列表"""
    result = await db.execute(
        select(HDF5Dataset.device)
        .where(HDF5Dataset.device.isnot(None))
        .distinct()
        .order_by(HDF5Dataset.device)
    )
    devices = [d for d in result.scalars().all() if d]
    return devices


async def migrate_project_binding(
    db: AsyncSession,
    mappings: List[tuple[str, str]],
) -> int:
    """
    将 project 字段从项目名改为项目ID。
    mappings: [(project_name, project_id), ...]
    返回更新行数。
    """
    total = 0
    for name, project_id in mappings:
        if not (name and name.strip()) or not (project_id and project_id.strip()):
            continue
        nv = name.strip()
        pv = project_id.strip()
        result = await db.execute(
            update(HDF5Dataset).where(HDF5Dataset.project == nv).values(project=pv)
        )
        total += result.rowcount or 0
    await db.commit()
    return total

