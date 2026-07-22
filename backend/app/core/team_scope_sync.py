"""
通过 asyncio.run 在独立事件循环中访问数据资产库 AsyncSession。

⚠️ 禁止在 FastAPI/Starlette 已有运行中事件循环的请求处理路径内调用（会触发 loop 混用）。
用户管理、审计列表等 HTTP 接口已改为 async + Depends(get_data_assets_db) + team_crud；
本模块仅保留给脚本、同步 CLI 等无活跃事件循环的场景按需使用。
"""
from __future__ import annotations

from typing import List, Set

import anyio
from app.crud import team as team_crud
from app.db.data_assets_session import DataAssetsSessionLocal


def list_team_ids_where_user_is_team_admin_sync(user_id: str) -> List[str]:
    async def inner() -> List[str]:
        async with DataAssetsSessionLocal() as session:
            return await team_crud.list_team_ids_where_user_is_team_admin(session, user_id)

    return anyio.from_thread.run(inner)


def list_user_ids_in_teams_administered_by_sync(admin_user_id: str) -> Set[str]:
    async def inner() -> Set[str]:
        async with DataAssetsSessionLocal() as session:
            return await team_crud.list_user_ids_in_teams_administered_by(session, admin_user_id)

    return anyio.from_thread.run(inner)


def list_teams_minimal_for_audit_filter_sync(*, all_teams: bool, team_ids: List[str] | None) -> List[dict]:
    """返回 [{\"id\", \"name\"}]，供审计页团队下拉使用。"""

    async def inner() -> List[dict]:
        async with DataAssetsSessionLocal() as session:
            if all_teams:
                rows, _ = await team_crud.list_teams(session)
            else:
                rows = await team_crud.list_teams_by_ids(session, team_ids or [])
            return [{"id": r.id, "name": r.name} for r in rows]

    return anyio.from_thread.run(inner)
