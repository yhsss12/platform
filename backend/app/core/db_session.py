"""同步 DB Session 辅助：短事务、超时、显式 rollback/close。"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, TypeVar

from sqlalchemy.orm import Session

from app.core.database import SessionLocal

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def db_session_scope(*, label: str = "db") -> Iterator[Session]:
    """开启短生命周期 session；异常 rollback，结束 close。"""
    db = SessionLocal()
    started = time.perf_counter()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.debug("%s session closed elapsed_ms=%s", label, elapsed_ms)


def run_db_with_timeout(
    fn: Callable[[Session], T],
    *,
    timeout_sec: float = 5.0,
    label: str = "db-op",
    default: Optional[T] = None,
) -> tuple[Optional[T], Optional[str]]:
    """
    在独立短 session 中执行 fn(db)。
    若连接池等待或执行超时，返回 (default, error_message)。
    """
    import concurrent.futures

    def _runner() -> T:
        with db_session_scope(label=label) as db:
            return fn(db)

    pool_timeout = max(1.0, float(timeout_sec))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_runner)
        try:
            return future.result(timeout=pool_timeout), None
        except concurrent.futures.TimeoutError:
            logger.warning("%s timed out after %.1fs", label, pool_timeout)
            return default, "DB_TIMEOUT"
        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)
            return default, f"{type(exc).__name__}: {exc}"
