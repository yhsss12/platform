from sqlalchemy import Column, String, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
import enum
from app.db.base import Base


class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    DRAFT = "DRAFT"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    status = Column(SQLEnum(TaskStatus), nullable=False, default=TaskStatus.PENDING)
    project_id = Column(String, nullable=True)
    project_name = Column(String, nullable=True)
    created_at = Column(String, nullable=False)  # ISO format string
    updated_at = Column(String, nullable=False)  # ISO format string

    # Relationships
    jobs = relationship("Job", back_populates="task", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="task", cascade="all, delete-orphan")
