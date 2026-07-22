from pydantic import BaseModel
from typing import Optional


class DatasetBase(BaseModel):
    name: str
    status: str = "ACTIVE"


class DatasetCreate(DatasetBase):
    pass


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None


class DatasetResponse(DatasetBase):
    id: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


