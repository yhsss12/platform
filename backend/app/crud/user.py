from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.roles import CanonicalUserRole, normalize_role
from app.core.security import get_password_hash
from app.models.user import User, UserRole

# 平台唯一超级管理员：登录账号固定 Pibot0001；展示名默认 Pibot（可与账号不同）
DEFAULT_SUPER_ADMIN_ACCOUNT_ID = "Pibot0001"
DEFAULT_SUPER_ADMIN_DISPLAY_NAME = "Pibot"
DEFAULT_SUPER_ADMIN_PASSWORD = "jinlian1234"
LEGACY_SUPER_ADMIN_USERNAME = "admin"
LEGACY_PIBOT_ACCOUNT_ID = "Pibot"


async def get_user_by_account_id(db: AsyncSession, account_id: str) -> User | None:
    """根据登录账号 account_id 获取用户"""
    aid = (account_id or "").strip()
    if not aid:
        return None
    result = await db.execute(select(User).where(User.account_id == aid))
    return result.scalar_one_or_none()


async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    """
    按展示名精确匹配返回一行（若存在多行同名，返回任意一条，顺序未定义）。
    不应用于「用户名唯一」校验；登录请用 get_user_by_account_id。
    """
    un = (username or "").strip()
    if not un:
        return None
    result = await db.execute(select(User).where(User.username == un).limit(1))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str) -> User | None:
    """根据用户 ID 获取用户"""
    uid = (user_id or "").strip()
    if not uid:
        return None
    result = await db.execute(select(User).where(User.id == uid))
    return result.scalar_one_or_none()


def _parse_role(role: str) -> UserRole:
    """将字符串转换为 UserRole，默认 USER"""
    try:
        return UserRole(role)
    except ValueError:
        return UserRole.USER


async def create_user(
    db: AsyncSession,
    *,
    account_id: str,
    username: str,
    password: str,
    role: str = "USER",
) -> User:
    """创建用户（须由调用方生成唯一 account_id）"""
    aid = (account_id or "").strip()
    if not aid:
        raise ValueError("account_id is required and must be non-empty")
    hashed_password = get_password_hash(password)
    user = User(
        account_id=aid,
        username=(username or "").strip(),
        password_hash=hashed_password,
        role=_parse_role(role),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def _demote_other_super_admins(db: AsyncSession, keep_user_id: str) -> None:
    res = await db.execute(select(User).where(User.role == UserRole.SUPER_ADMIN))
    for u in res.scalars():
        if str(u.id) != keep_user_id:
            u.role = UserRole.USER
            db.add(u)
    await db.commit()


async def _evict_non_super_from_account(db: AsyncSession, account_id: str) -> None:
    """若 account_id 被非超管占用，迁出到 _rel_{uuid}，为超管账号让路。"""
    aid = (account_id or "").strip()
    if not aid:
        return
    u = await get_user_by_account_id(db, aid)
    if u is None:
        return
    if normalize_role(u.role) == CanonicalUserRole.SUPER_ADMIN:
        return
    u.account_id = "_rel_" + str(u.id).replace("-", "")
    db.add(u)
    await db.flush()


async def get_or_create_admin_user(db: AsyncSession) -> User:
    """
    幂等确保存在唯一启用中的平台超级管理员：
    - 登录账号固定为 Pibot0001
    - 展示名默认 Pibot（username），可与 account_id 不同
    - 兼容存量 account_id=Pibot / admin 的超管行（迁移到 Pibot0001）
    """
    admin_user = await get_user_by_account_id(db, LEGACY_SUPER_ADMIN_USERNAME)
    pibot_user = await get_user_by_account_id(db, LEGACY_PIBOT_ACCOUNT_ID)
    canon_user = await get_user_by_account_id(db, DEFAULT_SUPER_ADMIN_ACCOUNT_ID)

    for u in (admin_user, pibot_user, canon_user):
        if u is None:
            continue
        rv = str(getattr(u.role, "value", u.role) or "").upper()
        if rv == "ADMINISTRATOR":
            u.role = UserRole.SUPER_ADMIN
            db.add(u)
    await db.commit()

    await _evict_non_super_from_account(db, DEFAULT_SUPER_ADMIN_ACCOUNT_ID)

    admin_user = await get_user_by_account_id(db, LEGACY_SUPER_ADMIN_USERNAME)
    pibot_user = await get_user_by_account_id(db, LEGACY_PIBOT_ACCOUNT_ID)
    canon_user = await get_user_by_account_id(db, DEFAULT_SUPER_ADMIN_ACCOUNT_ID)

    if canon_user is not None and canon_user.role == UserRole.SUPER_ADMIN:
        await _demote_other_super_admins(db, str(canon_user.id))
        await db.refresh(canon_user)
        return canon_user

    if pibot_user is not None and pibot_user.role == UserRole.SUPER_ADMIN:
        await _evict_non_super_from_account(db, DEFAULT_SUPER_ADMIN_ACCOUNT_ID)
        pibot_user.account_id = DEFAULT_SUPER_ADMIN_ACCOUNT_ID
        db.add(pibot_user)
        await db.commit()
        await db.refresh(pibot_user)
        await _demote_other_super_admins(db, str(pibot_user.id))
        return pibot_user

    if admin_user is not None and admin_user.role == UserRole.SUPER_ADMIN:
        await _evict_non_super_from_account(db, LEGACY_PIBOT_ACCOUNT_ID)
        await _evict_non_super_from_account(db, DEFAULT_SUPER_ADMIN_ACCOUNT_ID)
        admin_user.account_id = DEFAULT_SUPER_ADMIN_ACCOUNT_ID
        admin_user.username = DEFAULT_SUPER_ADMIN_DISPLAY_NAME
        admin_user.password_hash = get_password_hash(DEFAULT_SUPER_ADMIN_PASSWORD)
        admin_user.role = UserRole.SUPER_ADMIN
        db.add(admin_user)
        await db.commit()
        await db.refresh(admin_user)
        await _demote_other_super_admins(db, str(admin_user.id))
        return admin_user

    res = await db.execute(select(User).where(User.role == UserRole.SUPER_ADMIN))
    supers = list(res.scalars().all())
    if not supers:
        return await create_user(
            db,
            account_id=DEFAULT_SUPER_ADMIN_ACCOUNT_ID,
            username=DEFAULT_SUPER_ADMIN_DISPLAY_NAME,
            password=DEFAULT_SUPER_ADMIN_PASSWORD,
            role=UserRole.SUPER_ADMIN.value,
        )

    for u in supers:
        u.role = UserRole.USER
        db.add(u)
    await db.commit()
    return await create_user(
        db,
        account_id=DEFAULT_SUPER_ADMIN_ACCOUNT_ID,
        username=DEFAULT_SUPER_ADMIN_DISPLAY_NAME,
        password=DEFAULT_SUPER_ADMIN_PASSWORD,
        role=UserRole.SUPER_ADMIN.value,
    )
