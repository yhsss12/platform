from __future__ import annotations

from contextvars import ContextVar
import logging
from typing import Any, Dict, Optional


# 请求级/连接级缓存：同一请求内同一 episode 只 resolve 一次
# ⚠️ Do not access abs_path / warehouse_path directly.
# Use EpisodeStorage instead.
_resolve_cache_ctx: ContextVar[Optional[dict]] = ContextVar("episode_resolve_cache", default=None)

logger = logging.getLogger(__name__)


def _get_cache() -> dict:
    cache = _resolve_cache_ctx.get()
    if cache is None:
        cache = {}
        _resolve_cache_ctx.set(cache)
    return cache


def clear_episode_cache():
    """
    清空当前请求/连接的 episode resolve 缓存，避免内存泄漏。
    """
    cache = _resolve_cache_ctx.get()
    if cache is not None:
        cache.clear()


def get_current_resolve_cache():
    cache = _resolve_cache_ctx.get()
    return dict(cache) if cache else {}


class EpisodeStorage:
    def __init__(self, episode: Dict[str, Any]):
        ep = episode or {}
        self.episode_id: Optional[str] = ep.get("episode_id")
        # 批量标注等接口的结果字典使用 path 表示已解析本地文件路径，与 abs_path 等价
        self.abs_path: Optional[str] = ep.get("abs_path") or ep.get("path")
        self.warehouse_path: Optional[str] = ep.get("warehouse_path")

    def resolve_local_path(self) -> str:
        from app.services.storage_resolver import resolve_episode_file

        key = self._make_cache_key()
        cache = _get_cache()
        if key in cache:
            logger.debug(f"[resolve_cache] hit=True key={key}")
            return cache[key]
        logger.debug(f"[resolve_cache] hit=False key={key}")
        path = resolve_episode_file(self)
        cache[key] = path
        return path

    def _make_cache_key(self):
        eid = self.episode_id or ""
        wh = self.warehouse_path or ""
        ap = self.abs_path or ""
        return f"{eid}|{wh}|{ap}"

    def get_storage_key(self) -> str:
        return (self.warehouse_path or self.abs_path or "").strip()

