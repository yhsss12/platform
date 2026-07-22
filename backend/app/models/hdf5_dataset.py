"""
HDF5 数据集模型（PostgreSQL 统一库）
"""
from sqlalchemy import Column, Integer, BigInteger, String, Float, DateTime, Index
from sqlalchemy.sql import func
from app.db.base import Base


class HDF5Dataset(Base):
    """HDF5 数据集表"""
    __tablename__ = "hdf5_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, comment="文件名")
    project = Column(String, nullable=True, comment="项目")
    task = Column(String, nullable=True, comment="任务")
    device = Column(String, nullable=True, comment="机器人/设备")
    uploader = Column(String, nullable=True, comment="上传者（已废弃，保留兼容）")
    source = Column(String, nullable=True, default="local", comment="来源: local/collect/label/convert")
    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="入库时间")
    file_size_bytes = Column(BigInteger, nullable=False, comment="文件大小（字节），大文件可超 int32")
    duration_sec = Column(Float, nullable=True, comment="时长（秒）")
    format = Column(String, default="HDF5", nullable=False, comment="格式")
    storage_type = Column(String, default="local", nullable=False, comment="存储类型")
    storage_uri = Column(String, nullable=False, unique=True, comment="文件绝对路径")
    qc_status = Column(String, default="pending", nullable=False, comment="质检状态: pending/passed/failed")
    label_status = Column(String, default="unlabeled", nullable=False, comment="标注状态: unlabeled/labeled/partial")
    assign_status = Column(String, default="unassigned", nullable=False, comment="分配状态: unassigned/assigned")
    tags = Column(String, nullable=True, comment="标签（逗号分隔）")

    # 创建索引
    __table_args__ = (
        Index("idx_name", "name"),
        Index("idx_device", "device"),
        Index("idx_created_at", "created_at"),
        Index("idx_status", "qc_status", "label_status", "assign_status"),
        Index("idx_storage_uri", "storage_uri"),
    )























