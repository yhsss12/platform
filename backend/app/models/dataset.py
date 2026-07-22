from sqlalchemy import Column, String, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum
from app.db.base import Base


class DatasetStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    status = Column(SQLEnum(DatasetStatus), nullable=False, default=DatasetStatus.ACTIVE)
    created_at = Column(String, nullable=False)  # ISO format string
    updated_at = Column(String, nullable=False)  # ISO format string


