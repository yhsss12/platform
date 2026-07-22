from typing import Any, Optional, TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    """统一 API 返回结构"""
    ok: bool
    data: Optional[T] = None
    error: Optional[str] = None
    # 成功时的提示（如采集端路径已不存在仍删除平台记录），便于前端展示，不占用 error
    warning: Optional[str] = None
