"""
标注任务表（PostgreSQL，与数据资产同库）
dataset_ids 存储 data_assets 表的 ID 列表；当 data_assets 中对应记录被删除时，dataset_ids 失效，表示数据不存在。
"""
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Index
from sqlalchemy.sql import func
from app.models.data_asset import Base


class LabelTask(Base):
    """标注任务表"""
    __tablename__ = "label_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(32), unique=True, nullable=False, index=True, comment="任务业务 ID，如 dec33e26")
    name = Column(String(256), nullable=False, comment="任务名称")
    dataset_path = Column(Text, nullable=False, comment="数据集路径（服务器路径，可为多路径拼接）")
    dataset_ids = Column(Text, nullable=True, comment="继承自 data_assets.dataset_id 列表（JSON 如 [\"DS000001\"]），执行时据此校验数据是否存在")
    dataset_source = Column(String(32), nullable=True, comment="来源: data_assets | null")
    data_count = Column(Integer, nullable=True, comment="数据数量")
    device_type = Column(String(64), nullable=True, comment="设备类型（已废弃，保留兼容）")
    project_id = Column(String(128), nullable=True, comment="所属项目 ID")
    labeler = Column(String(128), nullable=True, comment="标注员")
    reviewer = Column(String(128), nullable=True, comment="审核员")
    collector = Column(String(128), nullable=True, comment="采集员")
    completed = Column(Boolean, default=False, nullable=False, comment="是否已完成（标注完成）")
    verified = Column(Boolean, default=False, nullable=False, comment="是否已校验（审核通过）")
    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")

    __table_args__ = (
        Index("idx_label_tasks_project", "project_id"),
        Index("idx_label_tasks_created", "created_at"),
        Index("idx_label_tasks_updated", "updated_at"),
        Index("idx_label_tasks_collector", "collector"),
    )
