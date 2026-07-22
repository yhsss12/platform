from collections.abc import Iterator

import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models import Base


# 仅支持 PostgreSQL，使用同步驱动（psycopg2）
def _create_main_engine():
    connect_timeout = int(os.getenv("MAIN_DB_CONNECT_TIMEOUT", "5"))
    statement_timeout_ms = int(os.getenv("MAIN_DB_STATEMENT_TIMEOUT_MS", "30000"))
    lock_timeout_ms = int(os.getenv("MAIN_DB_LOCK_TIMEOUT_MS", "5000"))
    pool_timeout = float(os.getenv("MAIN_DB_POOL_TIMEOUT", "10"))
    try:
        return create_engine(
            settings.sync_database_url,
            echo=False,
            pool_size=int(os.getenv("MAIN_DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("MAIN_DB_MAX_OVERFLOW", "20")),
            pool_timeout=pool_timeout,
            pool_pre_ping=True,
            pool_recycle=int(os.getenv("MAIN_DB_POOL_RECYCLE", "1800")),
            pool_reset_on_return="rollback",
            connect_args={
                "connect_timeout": connect_timeout,
                "options": (
                    f"-c statement_timeout={statement_timeout_ms} "
                    f"-c lock_timeout={lock_timeout_ms}"
                ),
            },
        )
    except ModuleNotFoundError as e:
        if ("pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST")) and "psycopg2" in str(e):
            return create_engine("sqlite+pysqlite:///:memory:", echo=False)
        raise


engine = _create_main_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    # 确保模型已导入以注册到元数据
    import app.models.user  # noqa: F401
    import app.models.refresh_token  # noqa: F401
    import app.models.auth_session  # noqa: F401
    import app.models.audit_log  # noqa: F401

    Base.metadata.create_all(bind=engine)
