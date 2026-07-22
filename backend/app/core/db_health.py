"""PostgreSQL 连接健康检查与诊断（只读）。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from app.core.config import settings


@dataclass
class DbHealthReport:
    ok: bool
    level: str  # DB_HEALTH_OK | DB_HEALTH_WARN | DB_HEALTH_FAIL
    host: str = ""
    port: Optional[int] = None
    database: str = ""
    user: str = ""
    connect_ms: Optional[float] = None
    select1_ms: Optional[float] = None
    total_connections: Optional[int] = None
    idle_in_transaction: Optional[int] = None
    blocking_locks: Optional[int] = None
    table_checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    activity_sample: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "ok": self.ok,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "connectMs": self.connect_ms,
            "select1Ms": self.select1_ms,
            "totalConnections": self.total_connections,
            "idleInTransaction": self.idle_in_transaction,
            "blockingLocks": self.blocking_locks,
            "tableChecks": self.table_checks,
            "activitySample": self.activity_sample,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def parse_database_target(database_url: Optional[str] = None) -> dict[str, Any]:
    url = (database_url or settings.sync_database_url or "").strip()
    parsed = urlparse(url.replace("postgresql+psycopg2", "postgresql"))
    return {
        "host": parsed.hostname or "",
        "port": parsed.port,
        "database": (parsed.path or "").lstrip("/"),
        "user": parsed.username or "",
    }


def create_health_engine(
    *,
    connect_timeout: float = 5.0,
    statement_timeout_ms: int = 30_000,
    database_url: Optional[str] = None,
) -> Engine:
    url = database_url or settings.sync_database_url
    return create_engine(
        url,
        echo=False,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": int(max(1, round(connect_timeout))),
            "options": f"-c statement_timeout={int(statement_timeout_ms)}",
        },
    )


def check_db_health(
    *,
    connect_timeout: float = 5.0,
    statement_timeout_ms: int = 30_000,
    database_url: Optional[str] = None,
) -> DbHealthReport:
    target = parse_database_target(database_url)
    report = DbHealthReport(
        ok=False,
        level="DB_HEALTH_FAIL",
        host=str(target.get("host") or ""),
        port=target.get("port"),
        database=str(target.get("database") or ""),
        user=str(target.get("user") or ""),
    )

    engine: Optional[Engine] = None
    try:
        engine = create_health_engine(
            connect_timeout=connect_timeout,
            statement_timeout_ms=statement_timeout_ms,
            database_url=database_url,
        )
        t0 = time.perf_counter()
        conn = engine.connect()
        report.connect_ms = round((time.perf_counter() - t0) * 1000, 1)
        try:
            t1 = time.perf_counter()
            scalar = conn.execute(text("SELECT 1")).scalar()
            report.select1_ms = round((time.perf_counter() - t1) * 1000, 1)
            if scalar != 1:
                report.errors.append(f"SELECT 1 returned unexpected value: {scalar!r}")
        finally:
            conn.close()

        with engine.connect() as conn:
            report.total_connections = int(
                conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                    )
                ).scalar()
                or 0
            )
            report.idle_in_transaction = int(
                conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE datname = current_database() AND state = 'idle in transaction'"
                    )
                ).scalar()
                or 0
            )
            report.blocking_locks = int(
                conn.execute(
                    text(
                        """
                        SELECT count(*)
                        FROM pg_locks blocked_locks
                        JOIN pg_locks blocking_locks
                          ON blocking_locks.locktype = blocked_locks.locktype
                         AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
                         AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
                         AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
                         AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
                         AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
                         AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
                         AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
                         AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
                         AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
                         AND blocking_locks.pid != blocked_locks.pid
                        WHERE NOT blocked_locks.granted
                        """
                    )
                ).scalar()
                or 0
            )

            rows = conn.execute(
                text(
                    """
                    SELECT pid, state, wait_event_type, wait_event,
                           now() - query_start AS age,
                           left(query, 120) AS query
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                    ORDER BY query_start NULLS LAST
                    LIMIT 20
                    """
                )
            ).mappings()
            report.activity_sample = [
                {
                    "pid": row["pid"],
                    "state": row["state"],
                    "waitEventType": row["wait_event_type"],
                    "waitEvent": row["wait_event"],
                    "age": str(row["age"]),
                    "query": row["query"],
                }
                for row in rows
            ]

            for table in (
                "workspace_jobs",
                "training_metric_summary",
                "model_assets",
                "evaluation_jobs",
                "data_assets",
            ):
                t_table = time.perf_counter()
                try:
                    with engine.connect() as table_conn:
                        count = table_conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
                    report.table_checks[table] = {
                        "ok": True,
                        "count": int(count or 0),
                        "ms": round((time.perf_counter() - t_table) * 1000, 1),
                    }
                except Exception as exc:
                    report.table_checks[table] = {
                        "ok": False,
                        "error": str(exc),
                        "ms": round((time.perf_counter() - t_table) * 1000, 1),
                    }
                    report.warnings.append(f"table {table} unreadable: {exc}")

        if report.idle_in_transaction and report.idle_in_transaction > 0:
            report.warnings.append(
                f"idle in transaction connections: {report.idle_in_transaction}"
            )
        if report.blocking_locks and report.blocking_locks > 0:
            report.warnings.append(f"blocking lock pairs: {report.blocking_locks}")

        if report.errors:
            report.level = "DB_HEALTH_FAIL"
            report.ok = False
        elif report.warnings:
            report.level = "DB_HEALTH_WARN"
            report.ok = True
        else:
            report.level = "DB_HEALTH_OK"
            report.ok = True
    except Exception as exc:
        report.errors.append(f"{type(exc).__name__}: {exc}")
        report.level = "DB_HEALTH_FAIL"
        report.ok = False
    finally:
        if engine is not None:
            engine.dispose()

    return report


def ping_db(*, connect_timeout: float = 5.0, statement_timeout_ms: int = 10_000) -> tuple[bool, str]:
    report = check_db_health(
        connect_timeout=connect_timeout,
        statement_timeout_ms=statement_timeout_ms,
    )
    if report.ok and report.level != "DB_HEALTH_FAIL":
        return True, report.level
    if report.errors:
        return False, report.errors[0]
    return False, report.level
