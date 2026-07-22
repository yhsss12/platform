from pydantic import BaseModel
from typing import Optional


class RunBase(BaseModel):
    task_id: str
    status: str = "QUEUED"


class RunCreate(RunBase):
    pass


class RunUpdate(BaseModel):
    status: Optional[str] = None


class RunResponse(RunBase):
    id: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


