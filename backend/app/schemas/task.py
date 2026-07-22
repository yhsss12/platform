from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from datetime import datetime
import json


class TaskBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: str = "DRAFT"
    # 扩展字段（前端传递，但不直接存储到数据库）
    owner: Optional[str] = None
    deviceId: Optional[str] = None
    deviceName: Optional[str] = None
    episodeCount: Optional[int] = None
    durationSec: Optional[int] = None
    storagePath: Optional[str] = None
    storageTypes: Optional[List[str]] = None
    remark: Optional[str] = None
    projectId: Optional[str] = None
    projectName: Optional[str] = None
    cameraDataFormat: Optional[str] = None
    frequencyConfig: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"  # 允许额外字段


class TaskCreate(TaskBase):
    pass


class TaskUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    owner: Optional[str] = None
    deviceId: Optional[str] = None
    deviceName: Optional[str] = None
    episodeCount: Optional[int] = None
    durationSec: Optional[int] = None
    storagePath: Optional[str] = None
    storageTypes: Optional[List[str]] = None
    remark: Optional[str] = None
    projectId: Optional[str] = None
    projectName: Optional[str] = None
    cameraDataFormat: Optional[str] = None
    frequencyConfig: Optional[Dict[str, Any]] = None

class TaskResponse(TaskBase):
    id: str
    created_at: str
    updated_at: str
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

    class Config:
        from_attributes = True

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        # 如果是从 ORM 对象转换
        if hasattr(obj, "description") and obj.description:
            try:
                config = json.loads(obj.description)
                if isinstance(config, dict):
                    # 将 config 中的字段注入到 obj 中（如果是 dict）或创建一个新 dict
                    # 由于 obj 是 ORM 对象，我们不能直接修改它，所以先转换为 dict
                    if not isinstance(obj, dict):
                        # 将 ORM 对象转换为 dict
                        obj_dict = {
                            "id": str(obj.id),
                            "name": obj.name,
                            "status": obj.status,
                            "created_at": obj.created_at,
                            "updated_at": obj.updated_at,
                            "createdAt": obj.created_at,
                            "updatedAt": obj.updated_at,
                            "description": obj.description, # 保留原始 JSON 字符串
                        }
                        
                        # 尝试提取 _text 作为 description
                        if "_text" in config:
                            obj_dict["description"] = config["_text"]
                        
                        # 注入其他字段
                        for key, value in config.items():
                            if key != "_text":
                                obj_dict[key] = value
                        if getattr(obj, "project_id", None) and "projectId" not in obj_dict:
                            obj_dict["projectId"] = getattr(obj, "project_id", None)
                        if getattr(obj, "project_name", None) and "projectName" not in obj_dict:
                            obj_dict["projectName"] = getattr(obj, "project_name", None)
                        
                        return super().model_validate(obj_dict)
            except (json.JSONDecodeError, TypeError):
                pass

        if hasattr(obj, "id") and not isinstance(obj, dict):
            obj_dict = {
                "id": str(getattr(obj, "id", "")),
                "name": getattr(obj, "name", ""),
                "status": getattr(obj, "status", "DRAFT"),
                "created_at": getattr(obj, "created_at", ""),
                "updated_at": getattr(obj, "updated_at", ""),
                "createdAt": getattr(obj, "created_at", ""),
                "updatedAt": getattr(obj, "updated_at", ""),
                "description": getattr(obj, "description", None),
                "projectId": getattr(obj, "project_id", None),
                "projectName": getattr(obj, "project_name", None),
            }
            return super().model_validate(obj_dict)

        return super().model_validate(obj)
