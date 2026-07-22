from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from app.core.deps import get_current_user
from app.db.session import get_db
from app.crud.run import (
    get_runs,
    get_run,
    create_run,
    update_run,
    delete_run
)
from app.schemas.run import RunCreate, RunUpdate, RunResponse
from app.schemas.common import ApiResponse
from app.models.user import User

router = APIRouter()


@router.get("", response_model=ApiResponse)
async def list_runs(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取运行列表"""
    runs = await get_runs(db, skip=skip, limit=limit)
    return ApiResponse(
        ok=True,
        data=[RunResponse.model_validate(r) for r in runs]
    )


@router.post("", response_model=ApiResponse)
async def create_new_run(
    run: RunCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建运行"""
    db_run = await create_run(db, run)
    return ApiResponse(
        ok=True,
        data=RunResponse.model_validate(db_run)
    )


@router.get("/{run_id}", response_model=ApiResponse)
async def get_run_by_id(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """根据 ID 获取运行"""
    db_run = await get_run(db, run_id)
    if db_run is None:
        return ApiResponse(
            ok=False,
            error="Run not found"
        )
    return ApiResponse(
        ok=True,
        data=RunResponse.model_validate(db_run)
    )


@router.patch("/{run_id}", response_model=ApiResponse)
async def update_run_by_id(
    run_id: UUID,
    run_update: RunUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新运行"""
    db_run = await update_run(db, run_id, run_update)
    if db_run is None:
        return ApiResponse(
            ok=False,
            error="Run not found"
        )
    return ApiResponse(
        ok=True,
        data=RunResponse.model_validate(db_run)
    )


@router.delete("/{run_id}", response_model=ApiResponse)
async def delete_run_by_id(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除运行"""
    success = await delete_run(db, run_id)
    if not success:
        return ApiResponse(
            ok=False,
            error="Run not found"
        )
    return ApiResponse(ok=True, data=None)


