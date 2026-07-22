"""统一资产查询 API。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.services.asset_query_service import search_assets

router = APIRouter()


class AssetSummary(BaseModel):
    model_config = {"extra": "allow"}


class AssetItem(BaseModel):
    id: str
    type: str
    job_id: Optional[str] = None
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    storage_uri: Optional[str] = None
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class AssetSearchResponse(BaseModel):
    items: list[AssetItem]
    total: int
    limit: int
    offset: int


@router.get("/search", response_model=AssetSearchResponse)
def search_assets_api(
    type: Optional[str] = Query(default=None, description="model | dataset | eval | checkpoint"),
    project_id: Optional[str] = Query(default=None),
    job_id: Optional[str] = Query(default=None),
    dataset_id: Optional[str] = Query(default=None),
    min_score: Optional[float] = Query(default=None),
    time_from: Optional[datetime] = Query(default=None),
    time_to: Optional[datetime] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AssetSearchResponse:
    items, total = search_assets(
        db,
        asset_type=type,
        project_id=project_id,
        job_id=job_id,
        dataset_id=dataset_id,
        min_score=min_score,
        time_from=time_from,
        time_to=time_to,
        limit=limit,
        offset=offset,
    )
    return AssetSearchResponse(
        items=[AssetItem(**row) for row in items],
        total=total,
        limit=limit,
        offset=offset,
    )
