"""
用户账号创建：统一分配 account_id 并写入 users（平台 Pibot#### / 团队 {code}####）。
所有新建 users 行应通过此处或显式传入非空 account_id 的 crud.create_user。
"""
from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.crud import team as team_crud
from app.models.user import User, UserRole
from app.services.account_id_allocate import (
    allocate_displaced_account_id,
    allocate_team_scoped_account_id,
)


class UserProvisionError(ValueError):
    """业务可映射为 400 的创建失败（团队不存在、停用等）"""


async def create_user_with_allocated_account_id(
    main_db: AsyncSession,
    assets_db: AsyncSession,
    *,
    display_username: str,
    password: str,
    role: UserRole,
    team_id_for_account: str | None,
) -> User:
    """
    生成唯一 account_id 并插入 users。team_id_for_account 非空时按团队流水号；否则平台 Pibot####。
    展示名 username 允许与现有用户重复；唯一性由 account_id 保证。
    """
    uname = (display_username or "").strip()
    if not uname:
        raise UserProvisionError("USERNAME_EMPTY")
    if not (password or "").strip():
        raise UserProvisionError("PASSWORD_EMPTY")

    tid = (team_id_for_account or "").strip() or None
    if tid:
        trow = await team_crud.get_team_by_id(assets_db, tid)
        if trow is None:
            raise UserProvisionError("TEAM_NOT_FOUND")
        if str(getattr(trow, "status", "") or "").lower() == "inactive":
            raise UserProvisionError("TEAM_INACTIVE")
        account_id = await allocate_team_scoped_account_id(main_db, assets_db, tid)
    else:
        account_id = await allocate_displaced_account_id(main_db)

    aid = (account_id or "").strip()
    if not aid:
        raise UserProvisionError("ACCOUNT_ID_ALLOCATION_FAILED")

    user = User(
        account_id=aid,
        username=uname,
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    main_db.add(user)
    try:
        await main_db.commit()
        await main_db.refresh(user)
    except IntegrityError:
        await main_db.rollback()
        raise UserProvisionError("USER_CREATE_CONFLICT") from None
    return user
