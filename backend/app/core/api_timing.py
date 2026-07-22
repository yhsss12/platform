"""Lightweight timing helpers for workspace list API endpoints."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def log_api_duration(endpoint: str, **context: object) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        ctx = " ".join(f"{key}={value}" for key, value in context.items() if value is not None)
        suffix = f" {ctx}" if ctx else ""
        logger.info("[api-timing] %s %.1fms%s", endpoint, elapsed_ms, suffix)


def paginate_rows(rows: list[T], *, limit: int | None = None, offset: int = 0) -> list[T]:
    sliced = rows[offset:] if offset else rows
    if limit is not None and limit >= 0:
        return sliced[:limit]
    return sliced
