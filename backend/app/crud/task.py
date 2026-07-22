from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List
from uuid import UUID
from datetime import datetime
from app.models.task import Task
from app.schemas.task import TaskCreate, TaskUpdate


async def get_tasks(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Task]:
    """获取任务列表"""
    result = await db.execute(select(Task).offset(skip).limit(limit))
    return list(result.scalars().all())


async def get_task(db: AsyncSession, task_id: UUID) -> Task | None:
    """根据 ID 获取任务"""
    result = await db.execute(select(Task).where(Task.id == task_id))
    return result.scalar_one_or_none()


async def get_task_by_name(db: AsyncSession, name: str) -> Task | None:
    normalized = (name or "").strip()
    if not normalized:
        return None
    result = await db.execute(
        select(Task).where(func.lower(Task.name) == func.lower(normalized))
    )
    return result.scalar_one_or_none()



async def create_task(db: AsyncSession, task: TaskCreate) -> Task:
    """创建任务"""
    import json
    now = datetime.utcnow().isoformat()
    
    
    # 将扩展字段序列化为 JSON 存储在 description 中
    # 如果 description 已经存在，则合并
    task_config = {}
    if task.description:
        try:
            loaded_config = json.loads(task.description)
            if isinstance(loaded_config, dict):
                task_config = loaded_config
            else:
                task_config = {"_text": task.description}
        except (json.JSONDecodeError, TypeError):
            # 如果不是 JSON，作为普通文本存储
            task_config = {"_text": task.description}
    
    # 添加扩展字段到配置中
    if task.owner:
        task_config["owner"] = task.owner
    if task.deviceId:
        task_config["deviceId"] = task.deviceId
    if task.deviceName:
        task_config["deviceName"] = task.deviceName
    if task.episodeCount is not None:
        task_config["episodeCount"] = task.episodeCount
    if task.durationSec is not None:
        task_config["durationSec"] = task.durationSec
    if task.storagePath:
        task_config["storagePath"] = task.storagePath
    if task.storageTypes:
        task_config["storageTypes"] = task.storageTypes
    if task.remark:
        task_config["remark"] = task.remark
    if task.projectId:
        task_config["projectId"] = task.projectId
    if task.projectName:
        task_config["projectName"] = task.projectName
    if task.cameraDataFormat:
        task_config["cameraDataFormat"] = task.cameraDataFormat
    if task.frequencyConfig:
        task_config["frequencyConfig"] = task.frequencyConfig
    
    # 将配置序列化为 JSON
    description_json = json.dumps(task_config, ensure_ascii=False) if task_config else None
    
    db_task = Task(
        name=task.name.strip(),
        description=description_json,
        status=task.status,
        project_id=(task.projectId or "").strip() or None,
        project_name=(task.projectName or "").strip() or None,
        created_at=now,
        updated_at=now
    )
    db.add(db_task)
    await db.commit()
    await db.refresh(db_task)
    return db_task


async def update_task(db: AsyncSession, task_id: UUID, task_update: TaskUpdate) -> Task | None:
    """更新任务"""
    import json
    
    db_task = await get_task(db, task_id)
    if db_task is None:
        return None
    
    update_data = task_update.model_dump(exclude_unset=True)
    
    # 扩展字段列表
    extended_fields = ["owner", "deviceId", "deviceName", "episodeCount", "durationSec", "storagePath", "storageTypes", "remark", "projectId", "projectName", "cameraDataFormat", "frequencyConfig"]
    
    # 如果更新了 description 或扩展字段，需要合并到 JSON 中
    if "description" in update_data or any(k in update_data for k in extended_fields):
        # 解析现有的 description
        task_config = {}
        if db_task.description:
            try:
                loaded_config = json.loads(db_task.description)
                if isinstance(loaded_config, dict):
                    task_config = loaded_config
                else:
                    task_config = {"_text": db_task.description}
            except (json.JSONDecodeError, TypeError):
                task_config = {"_text": db_task.description}
        
        # 处理 description 更新
        if "description" in update_data:
            desc_value = update_data.pop("description")
            if desc_value:
                try:
                    desc_json = json.loads(desc_value)
                    # 如果是 JSON，合并到 config
                    if isinstance(desc_json, dict):
                        task_config.update(desc_json)
                    else:
                        task_config["_text"] = str(desc_json)
                except (json.JSONDecodeError, TypeError):
                    # 如果不是 JSON，作为文本存储
                    task_config["_text"] = desc_value
            else:
                # description 被清空，是否要清空 _text?
                if "_text" in task_config:
                    del task_config["_text"]

        # 处理扩展字段更新
        for key in extended_fields:
            if key in update_data:
                value = update_data.pop(key) # 从 update_data 中移除，避免写入 DB 字段
                if value is not None:
                    task_config[key] = value
                elif key in task_config:
                    # 如果显式设为 None，则删除该字段
                    del task_config[key]
                if key == "projectId":
                    db_task.project_id = (str(value).strip() if value is not None else None)
                if key == "projectName":
                    db_task.project_name = (str(value).strip() if value is not None else None)
        
        # 序列化回 JSON
        if task_config:
            # 如果只有 _text 且没有其他字段，可以直接存为字符串吗？
            # 为了保持一致性，还是存为 JSON
            update_data["description"] = json.dumps(task_config, ensure_ascii=False)
        else:
            update_data["description"] = None
    
    # 更新 DB 字段
    for field, value in update_data.items():
        if hasattr(db_task, field):
            setattr(db_task, field, value)
    
    db_task.updated_at = datetime.utcnow().isoformat()
    
    await db.commit()
    await db.refresh(db_task)
    return db_task


async def delete_task(db: AsyncSession, task_id: UUID) -> bool:
    """删除任务"""
    db_task = await get_task(db, task_id)
    if db_task is None:
        return False
    
    await db.delete(db_task)
    await db.commit()
    return True
