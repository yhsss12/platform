#!/usr/bin/env python3
"""PostgreSQL 健康检查（只读）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded

ensure_dotenv_loaded()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PostgreSQL health for EAI IDE backend")
    parser.add_argument("--connect-timeout", type=float, default=5.0, help="connection timeout seconds")
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=30_000,
        help="per-query statement timeout in milliseconds",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    from app.core.db_health import check_db_health

    report = check_db_health(
        connect_timeout=args.connect_timeout,
        statement_timeout_ms=args.statement_timeout_ms,
    )
    payload = report.to_dict()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(report.level)
        print(f"host={report.host} port={report.port} database={report.database} user={report.user}")
        print(f"connect_ms={report.connect_ms} select1_ms={report.select1_ms}")
        print(
            f"connections={report.total_connections} "
            f"idle_in_transaction={report.idle_in_transaction} "
            f"blocking_locks={report.blocking_locks}"
        )
        if report.warnings:
            print("warnings:")
            for item in report.warnings:
                print(f"  - {item}")
        if report.errors:
            print("errors:")
            for item in report.errors:
                print(f"  - {item}")
        if report.table_checks:
            print("tables:")
            for name, info in report.table_checks.items():
                print(f"  - {name}: {info}")
        if report.activity_sample:
            print("activity_sample:")
            for row in report.activity_sample[:10]:
                print(f"  pid={row.get('pid')} state={row.get('state')} age={row.get('age')} query={row.get('query')}")

    return 0 if report.level == "DB_HEALTH_OK" else (1 if report.level == "DB_HEALTH_FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
