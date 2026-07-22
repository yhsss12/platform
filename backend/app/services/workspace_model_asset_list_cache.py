"""TTL cache for workspace model asset list rows."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 30.0


@dataclass
class _CacheEntry:
    rows: list[dict[str, Any]]
    expires_at: float


_lock = threading.Lock()
_list_cache: _CacheEntry | None = None


def invalidate_model_asset_list_cache() -> None:
    global _list_cache
    with _lock:
        if _list_cache is not None:
            _list_cache = None
            logger.info("workspace model asset list cache cleared")


def get_or_load_model_asset_list_rows(
    *,
    loader: Callable[[], list[dict[str, Any]]],
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> tuple[list[dict[str, Any]], bool]:
    global _list_cache
    now = time.monotonic()
    with _lock:
        if _list_cache is not None and _list_cache.expires_at > now:
            return list(_list_cache.rows), True

    rows = loader()
    with _lock:
        _list_cache = _CacheEntry(rows=list(rows), expires_at=now + ttl_seconds)
    return rows, False
