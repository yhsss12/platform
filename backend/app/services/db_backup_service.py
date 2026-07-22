from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = (os.getenv(name, "") or "").strip()
    try:
        value = int(raw) if raw else default
    except Exception:
        value = default
    return max(minimum, value)


def resolve_dated_backup_path(project_root: Path) -> Path:
    """
    每次备份生成独立文件：eai_ide_backup_YYYY-MM-DD.sql（按运行时刻的本地日期）。

    DB_AUTO_BACKUP_OUTPUT_PATH：
    - 未设置：写入项目根目录 project_root /
    - 为目录（已存在且为目录，或以 / 结尾）：在该目录下写入
    - 以 .sql 结尾（旧版单文件配置）：写入其父目录，文件名仍按日期
    - 其它路径：视为目录（若不存在会在 pg_dump 前 mkdir）
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    fname = f"eai_ide_backup_{date_str}.sql"
    raw = (os.getenv("DB_AUTO_BACKUP_OUTPUT_PATH", "") or "").strip()
    if not raw:
        return project_root / fname
    p = Path(raw).expanduser()
    if raw.endswith(("/", "\\")):
        return p / fname
    if p.exists() and p.is_dir():
        return p / fname
    if p.suffix.lower() == ".sql":
        return p.parent / fname
    return p / fname


def _seconds_until_next_daily_time(hour: int, minute: int) -> int:
    now = datetime.now()
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run = next_run + timedelta(days=1)
    delta = next_run - now
    return max(1, int(delta.total_seconds()))


def _build_pg_dump_target() -> tuple[str, dict]:
    """
    将 DATABASE_URL 转换为 pg_dump 可用连接串与环境变量。
    """
    db_url = (os.getenv("DATABASE_URL", "") or "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL 未配置")

    # asyncpg/psycopg 适配为 pg_dump 可识别 scheme
    normalized = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )
    parsed = urlparse(normalized)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise RuntimeError("DATABASE_URL 不是 PostgreSQL 连接串")

    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise RuntimeError("DATABASE_URL 缺少主机或数据库名")

    db_name = parsed.path.lstrip("/")
    user = parsed.username or ""
    password = parsed.password or ""
    host = parsed.hostname
    port = str(parsed.port or 5432)

    conn = f"postgresql://{user}@{host}:{port}/{db_name}" if user else f"postgresql://{host}:{port}/{db_name}"
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password
    return conn, env


def run_database_backup(output_path: Path) -> None:
    """
    执行一次 SQL 备份，输出为 plain SQL 文件。
    """
    if shutil.which("pg_dump") is None:
        raise RuntimeError("未找到 pg_dump，请先安装 PostgreSQL client 工具")

    conn, env = _build_pg_dump_target()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".sql.tmp")

    cmd = [
        "pg_dump",
        "--no-owner",
        "--no-privileges",
        "--encoding=UTF8",
        "--file",
        str(tmp_path),
        conn,
    ]
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"pg_dump 失败: {stderr or completed.returncode}")

    tmp_path.replace(output_path)


async def periodic_db_backup_loop(stop_event: asyncio.Event) -> None:
    """
    每日自动备份（默认开启）。每次写入独立文件：eai_ide_backup_YYYY-MM-DD.sql。

    环境变量：
    - DB_AUTO_BACKUP_ENABLED: 是否开启（默认 true）
    - DB_AUTO_BACKUP_DAILY_HOUR: 每天备份小时（0-23，默认 3）
    - DB_AUTO_BACKUP_DAILY_MINUTE: 每天备份分钟（0-59，默认 0）
    - DB_AUTO_BACKUP_OUTPUT_PATH: 备份目录或旧版 .sql 路径（默认项目根目录）
    - DB_AUTO_BACKUP_RUN_ON_STARTUP: 启动后立即备份一次（默认 true）
    """
    if not _env_bool("DB_AUTO_BACKUP_ENABLED", True):
        logger.info("db_backup: auto backup disabled")
        return

    backend_root = Path(__file__).resolve().parents[2]
    project_root = backend_root.parent
    daily_hour = min(23, _env_int("DB_AUTO_BACKUP_DAILY_HOUR", 3, minimum=0))
    daily_minute = min(59, _env_int("DB_AUTO_BACKUP_DAILY_MINUTE", 0, minimum=0))
    run_on_startup = _env_bool("DB_AUTO_BACKUP_RUN_ON_STARTUP", True)

    logger.info(
        "db_backup: started, schedule=%02d:%02d pattern=eai_ide_backup_<YYYY-MM-DD>.sql project_root=%s DB_AUTO_BACKUP_OUTPUT_PATH=%r",
        daily_hour,
        daily_minute,
        project_root,
        (os.getenv("DB_AUTO_BACKUP_OUTPUT_PATH", "") or "").strip(),
    )

    if run_on_startup and not stop_event.is_set():
        try:
            out = resolve_dated_backup_path(project_root)
            await asyncio.to_thread(run_database_backup, out)
            logger.info("db_backup: startup backup finished: %s", out)
        except Exception as e:
            logger.warning("db_backup: startup backup failed: %s", e)

    while not stop_event.is_set():
        wait_seconds = _seconds_until_next_daily_time(daily_hour, daily_minute)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
            break
        except asyncio.TimeoutError:
            pass

        try:
            out = resolve_dated_backup_path(project_root)
            await asyncio.to_thread(run_database_backup, out)
            logger.info("db_backup: periodic backup finished: %s", out)
        except Exception as e:
            logger.warning("db_backup: periodic backup failed: %s", e)

    logger.info("db_backup: stopped")
