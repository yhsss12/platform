from sqlalchemy import Column, String, Integer, BigInteger, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
import enum
from app.db.base import Base


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    job_number = Column(Integer, nullable=False, default=0)
    status = Column(SQLEnum(JobStatus), nullable=False, default=JobStatus.PENDING)
    operator_name = Column(String, nullable=True)
    project_id = Column(String, nullable=True)
    project_name = Column(String, nullable=True)
    mcap_path = Column(String, nullable=True)
    mcap_size_bytes = Column(BigInteger, nullable=True, comment="MCAP 文件大小（字节）")
    duration_sec = Column(Integer, nullable=True)
    started_at = Column(String, nullable=True)  # ISO format string
    finished_at = Column(String, nullable=True)  # ISO format string
    progress = Column(Integer, nullable=False, default=0)  # 0-100
    collection_quantity = Column(Integer, nullable=True, default=0)  # Target count
    completed_count = Column(Integer, nullable=True, default=0)  # Actual count
    created_at = Column(String, nullable=False)  # ISO format string
    updated_at = Column(String, nullable=False)  # ISO format string

    # Relationships
    task = relationship("Task", back_populates="jobs")

