from pydantic import BaseModel, Field
from typing import Optional, Union, Dict
from datetime import datetime
from uuid import UUID

class JobBase(BaseModel):
    task_id: UUID
    job_number: Optional[int] = 0
    operator_name: Optional[str] = None
    status: str = "PENDING"
    collection_quantity: Optional[int] = 0
    completed_count: Optional[int] = 0
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    # 领取作业时传入：平台 devices.id（权威，区别于任务/采集端侧旧 ID）
    device_id: Optional[str] = None

class JobCreate(JobBase):
    pass

class JobUpdate(BaseModel):
    operator_name: Optional[str] = None
    status: Optional[str] = None
    mcap_path: Optional[str] = None
    # 为 True 且 mcap_path 非空时，才写入/更新 data_assets（正式「保存」）；默认 False，避免仅更新作业字段时误登记。
    register_collect_asset: bool = False
    mcap_size_bytes: Optional[int] = None
    validation_report_json: Optional[str] = None
    duration_sec: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: Optional[Union[int, Dict[str, int]]] = None
    collection_quantity: Optional[int] = None
    completed_count: Optional[int] = None
    # 采集所属项目信息（从前端 SaveDataDialog 传入，用于写入 data_assets meta.collect）
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    # 当前连接采集端的 hardware_uuid（与设备表一致），用于数据资产同步时定位 Agent
    hardware_uuid: Optional[str] = None
    # 新字段：mac_address 作为硬件标识，等价于 hardware_uuid 的用途（兼容“不要传 hardware_uuid，直接传 MAC”）
    mac_address: Optional[str] = None
    # 平台设备主键 devices.id，写入 data_assets.device_id，列表区分采集端
    device_id: Optional[str] = None

class JobResponse(JobBase):
    id: UUID
    job_number: int = 0
    mcap_path: Optional[str] = None
    mcap_size_bytes: Optional[int] = None
    duration_sec: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: int
    created_at: str
    updated_at: str
    collection_quantity: int = 0
    completed_count: int = 0

    class Config:
        from_attributes = True
