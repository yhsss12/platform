import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    future=True,
    echo=False,
    # 保守连接池：避免 Postgres 连接数被打满
    pool_size=int(os.getenv("MAIN_DB_POOL_SIZE", "20")),
    max_overflow=int(os.getenv("MAIN_DB_MAX_OVERFLOW", "40")),
    pool_timeout=float(os.getenv("MAIN_DB_POOL_TIMEOUT", "120")),
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    """获取数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


