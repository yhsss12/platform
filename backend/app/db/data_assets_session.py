"""
数据资产/标注任务库会话（与主库共用同一 PostgreSQL）。
DATA_ASSETS_ROOT 仅用于资产文件目录（HDF5/MCAP 等），不用于数据库连接。
"""
import os
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
from app.core.config import settings
from app.core.platform_paths import platform_paths

_DATA_ASSETS_ROOT_DEFAULT = platform_paths.datasets
DATA_ASSETS_ROOT = Path(os.getenv("DATA_ASSETS_ROOT", str(_DATA_ASSETS_ROOT_DEFAULT))).resolve()
DATA_ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = None  # 保留字段名兼容旧代码；数据库仅 PostgreSQL

data_assets_engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    # 保守连接池：避免 Postgres 连接数被打满
    pool_size=int(os.getenv("DATA_ASSETS_DB_POOL_SIZE", "20")),
    max_overflow=int(os.getenv("DATA_ASSETS_DB_MAX_OVERFLOW", "40")),
    pool_timeout=float(os.getenv("DATA_ASSETS_DB_POOL_TIMEOUT", "120")),
    pool_pre_ping=True,
)
print("✓ 数据资产/标注任务库: PostgreSQL (统一库)")

DataAssetsSessionLocal = async_sessionmaker(
    data_assets_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

_data_assets_columns_checked = False


async def _ensure_data_assets_optional_columns(session: AsyncSession) -> None:
    """轻量自愈：补齐历史库缺失的可选列。"""
    global _data_assets_columns_checked
    if _data_assets_columns_checked:
        return
    await session.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS operator_name VARCHAR(256)"))
    await session.commit()
    _data_assets_columns_checked = True


async def get_data_assets_db() -> AsyncSession:
    async with DataAssetsSessionLocal() as session:
        try:
            await _ensure_data_assets_optional_columns(session)
            yield session
        finally:
            await session.close()
