from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models import User, UserRole

CANON_SUPER_ACCOUNT = "Pibot0001"
DISPLAY_SUPER_NAME = "Pibot"


def get_user_by_account_id(db: Session, account_id: str) -> User | None:
    """同步 Session：按登录账号查用户（JWT sub = account_id）"""
    aid = (account_id or "").strip()
    if not aid:
        return None
    stmt = select(User).where(User.account_id == aid)
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_username(db: Session, username: str) -> User | None:
    """按展示名匹配一行（多行同名时返回任意一条）。非登录键；不可用于唯一性判断。"""
    un = (username or "").strip()
    if not un:
        return None
    stmt = select(User).where(User.username == un).limit(1)
    return db.execute(stmt).scalar_one_or_none()


def create_user(db: Session, account_id: str, username: str, password: str, role: str) -> User:
    aid = (account_id or "").strip()
    if not aid:
        raise ValueError("account_id is required and must be non-empty")
    user_role = UserRole(role)
    user = User(
        account_id=aid,
        username=(username or "").strip(),
        password_hash=hash_password(password),
        role=user_role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def ensure_default_super_admin(db: Session) -> None:
    """
    同步 Session 下的默认超管兜底（与异步 get_or_create_admin_user 一致思路）。
    仍以 main.py 中 get_or_create_admin_user 为准。
    """
    canon = db.execute(select(User).where(User.account_id == CANON_SUPER_ACCOUNT)).scalar_one_or_none()
    if canon and canon.role == UserRole.SUPER_ADMIN:
        return
    legacy = db.execute(select(User).where(User.account_id == "admin")).scalar_one_or_none()
    if legacy and legacy.role == UserRole.SUPER_ADMIN:
        pibot = db.execute(select(User).where(User.account_id == "Pibot")).scalar_one_or_none()
        if pibot is not None and pibot.id != legacy.id and pibot.role != UserRole.SUPER_ADMIN:
            pibot.account_id = "_rel_" + str(pibot.id).replace("-", "")
        block = db.execute(select(User).where(User.account_id == CANON_SUPER_ACCOUNT)).scalar_one_or_none()
        if block is not None and block.id != legacy.id and block.role != UserRole.SUPER_ADMIN:
            block.account_id = "_rel_" + str(block.id).replace("-", "")
        legacy.account_id = CANON_SUPER_ACCOUNT
        legacy.username = DISPLAY_SUPER_NAME
        legacy.password_hash = hash_password("jinlian1234")
        legacy.role = UserRole.SUPER_ADMIN
        db.commit()
        print("Migrated legacy admin to Pibot0001.")
        return
    has_super = db.execute(
        select(User.id).where(User.role == UserRole.SUPER_ADMIN).limit(1)
    ).scalar_one_or_none()
    if has_super is None:
        user = create_user(
            db,
            CANON_SUPER_ACCOUNT,
            DISPLAY_SUPER_NAME,
            "jinlian1234",
            UserRole.SUPER_ADMIN.value,
        )
        print(f"Default super admin created: {user.account_id}")
