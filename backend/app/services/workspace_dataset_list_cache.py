"""TTL cache for workspace dataset list scan results."""

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
_scan_cache: dict[str, _CacheEntry] = {}


def _scan_cache_key(user_id: int | str, workspace: str = "default") -> str:
    return f"workspace:{workspace}:user:{user_id}:datasets:scan"


def invalidate_workspace_dataset_list_cache(
    *,
    user_id: int | str | None = None,
    workspace: str = "default",
) -> None:
    """Drop cached dataset scan rows after mutations."""
    with _lock:
        if user_id is None:
            before = len(_scan_cache)
            _scan_cache.clear()
            if before:
                logger.info("workspace dataset list cache cleared all entries=%s", before)
            return
        key = _scan_cache_key(user_id, workspace=workspace)
        if _scan_cache.pop(key, None) is not None:
            logger.info("workspace dataset list cache invalidated key=%s", key)


def get_or_load_dataset_scan_rows(
    *,
    user_id: int | str,
    loader: Callable[[], list[dict[str, Any]]],
    workspace: str = "default",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (rows, cache_hit). loader runs only on miss/expired."""
    key = _scan_cache_key(user_id, workspace=workspace)
    now = time.monotonic()
    with _lock:
        entry = _scan_cache.get(key)
        if entry and entry.expires_at > now:
            return list(entry.rows), True

    rows = loader()
    with _lock:
        _scan_cache[key] = _CacheEntry(rows=list(rows), expires_at=now + ttl_seconds)
    return rows, False
