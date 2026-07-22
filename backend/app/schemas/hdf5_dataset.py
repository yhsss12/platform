"""
HDF5 数据集 Schema
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class HDF5DatasetBase(BaseModel):
    name: str
    project: Optional[str] = None
    task: Optional[str] = None
    device: Optional[str] = None
    uploader: Optional[str] = None
    source: Optional[str] = None  # local | collect | label | convert，缺失/null 视为 local
    file_size_bytes: int
    duration_sec: Optional[float] = None
    format: str = "HDF5"
    storage_type: str = "local"
    storage_uri: str
    qc_status: str = "pending"
    label_status: str = "unlabeled"
    assign_status: str = "unassigned"
    tags: Optional[str] = None


class HDF5DatasetCreate(HDF5DatasetBase):
    pass


class HDF5DatasetUpdate(BaseModel):
    project: Optional[str] = None
    task: Optional[str] = None
    device: Optional[str] = None
    uploader: Optional[str] = None
    source: Optional[str] = None
    duration_sec: Optional[float] = None
    qc_status: Optional[str] = None
    label_status: Optional[str] = None
    assign_status: Optional[str] = None
    tags: Optional[str] = None


class HDF5DatasetResponse(HDF5DatasetBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# 简化的导入接口 Schema
class ImportRequest(BaseModel):
    file_path: str
    project_id: str
    tags: Optional[str] = None
    uploader: Optional[str] = None


class ImportResponse(BaseModel):
    success: bool
    message: str
    dataset_id: Optional[int] = None


class BatchImportResult(BaseModel):
    filename: str
    success: bool
    message: Optional[str] = None
    dataset_id: Optional[int] = None


class BatchImportResponse(BaseModel):
    success: bool
    message: str
    results: list[BatchImportResult]


# 查询参数
class DatasetQueryParams(BaseModel):
    keyword: Optional[str] = None
    device: Optional[str] = None
    project: Optional[str] = None
    """数据格式：hdf5 / mcap / lerobot（小写），不传则不过滤"""
    format: Optional[str] = None
    qc_status: Optional[str] = None
    label_status: Optional[str] = None
    assign_status: Optional[str] = None
    page: int = 1
    page_size: int = 20


class DatasetListResponse(BaseModel):
    items: list[HDF5DatasetResponse]
    total: int
    page: int
    page_size: int

