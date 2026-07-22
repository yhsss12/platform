from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from app.services.data_asset_path_resolver import (
    clear_all_minio_view_cache,
    clear_minio_conversion_download_cache,
)

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = (os.getenv(name, "") or "").strip()
    try:
        val = int(raw) if raw else default
    except Exception:
        val = default
    return max(minimum, val)


def clear_project_cache() -> Dict[str, int]:
    """
    清理 backend/project 目录内容（保留目录本身）。
    """
    backend_root = Path(__file__).resolve().parents[2]
    target_dir = backend_root / "project"
    removed_files = 0
    removed_dirs = 0
    removed_bytes = 0
    if not target_dir.exists() or not target_dir.is_dir():
        return {"removed_files": 0, "removed_dirs": 0, "removed_bytes": 0}
    for child in target_dir.iterdir():
        try:
            if child.is_dir():
                for root, dirs, files in os.walk(child, topdown=False):
                    for name in files:
                        fp = Path(root) / name
                        try:
                            removed_bytes += fp.stat().st_size
                        except Exception:
                            pass
                        try:
                            fp.unlink(missing_ok=True)
                            removed_files += 1
                        except Exception:
                            pass
                    for name in dirs:
                        dp = Path(root) / name
                        try:
                            dp.rmdir()
                            removed_dirs += 1
                        except Exception:
                            pass
                try:
                    child.rmdir()
                    removed_dirs += 1
                except Exception:
                    pass
            else:
                try:
                    removed_bytes += child.stat().st_size
                except Exception:
                    pass
                child.unlink(missing_ok=True)
                removed_files += 1
        except Exception:
            continue
    return {
        "removed_files": int(removed_files),
        "removed_dirs": int(removed_dirs),
        "removed_bytes": int(removed_bytes),
    }


def clear_data_disk_caches() -> Dict[str, Any]:
    """
    清理 DATA_ASSETS_ROOT 下与对象仓库相关的可丢弃磁盘缓存：
    - .minio_view_cache
    - _minio_conversion_cache
    """
    return {
        "minio_view_cache": clear_all_minio_view_cache(),
        "minio_conversion_cache": clear_minio_conversion_download_cache(),
    }


def run_startup_cache_cleanup_from_env() -> Dict[str, Any]:
    """
    应用启动时执行一次缓存清理（由 main lifespan 调用）。

    环境变量：
    - CLEAR_CACHE_ON_STARTUP: 是否启用（默认 true）；设为 false 可关闭
    - CLEAR_CACHE_ON_STARTUP_SCOPE: all | data | project（默认 all）
      - data: data 资产根下 .minio_view_cache + _minio_conversion_cache
      - project: 仅 backend/project 下内容
      - all: 两者都清理（与 POST /api/data-assets/cache/clear?scope=all 语义一致）
    """
    result: Dict[str, Any] = {"ran": False}
    if not _env_bool("CLEAR_CACHE_ON_STARTUP", True):
        result["skipped"] = "CLEAR_CACHE_ON_STARTUP disabled"
        return result

    scope = (os.getenv("CLEAR_CACHE_ON_STARTUP_SCOPE") or "all").strip().lower()
    if scope not in {"all", "data", "project"}:
        scope = "all"

    data_cache: Optional[Dict[str, Any]] = None
    project_cache: Optional[Dict[str, int]] = None

    if scope in {"all", "data"}:
        data_cache = clear_data_disk_caches()
    if scope in {"all", "project"}:
        project_cache = clear_project_cache()

    result["ran"] = True
    result["scope"] = scope
    result["data_cache"] = data_cache
    result["project_cache"] = project_cache
    logger.info("cache_cleanup: startup cleanup scope=%s data=%s project=%s", scope, data_cache, project_cache)
    return result


async def periodic_cache_cleanup_loop(stop_event: asyncio.Event) -> None:
    """
    自动缓存清理后台循环。

    环境变量：
    - CACHE_AUTO_CLEANUP_ENABLED: 是否开启自动清理（默认 true）
    - CACHE_AUTO_CLEANUP_INTERVAL_MINUTES: 清理周期分钟数（默认 1440）
    - CACHE_AUTO_CLEANUP_DATA_ENABLED: 是否清理 data 缓存（默认 true）
    - CACHE_AUTO_CLEANUP_PROJECT_ENABLED: 是否清理 project 缓存（默认 true）
    """
    enabled = _env_bool("CACHE_AUTO_CLEANUP_ENABLED", True)
    if not enabled:
        logger.info("cache_cleanup: auto cleanup disabled")
        return

    interval_minutes = _env_int("CACHE_AUTO_CLEANUP_INTERVAL_MINUTES", 1440, minimum=1)
    run_data = _env_bool("CACHE_AUTO_CLEANUP_DATA_ENABLED", True)
    run_project = _env_bool("CACHE_AUTO_CLEANUP_PROJECT_ENABLED", True)
    interval_seconds = interval_minutes * 60

    logger.info(
        "cache_cleanup: auto cleanup started, interval=%sm data=%s project=%s",
        interval_minutes,
        run_data,
        run_project,
    )

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            break
        except asyncio.TimeoutError:
            pass

        try:
            if run_data:
                data_stats = await asyncio.to_thread(clear_data_disk_caches)
                logger.info("cache_cleanup: data cache cleaned: %s", data_stats)
            if run_project:
                project_stats = await asyncio.to_thread(clear_project_cache)
                logger.info("cache_cleanup: project cache cleaned: %s", project_stats)
        except Exception as e:
            logger.warning("cache_cleanup: periodic cleanup failed: %s", e)

    logger.info("cache_cleanup: auto cleanup stopped")
