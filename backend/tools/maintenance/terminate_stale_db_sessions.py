#!/usr/bin/env python3
"""终止长时间 idle in transaction 的 PostgreSQL 会话（不删表数据）。"""

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
    parser = argparse.ArgumentParser(description="Terminate stale idle-in-transaction PostgreSQL sessions")
    parser.add_argument("--min-age-sec", type=int, default=600, help="minimum idle-in-transaction age")
    parser.add_argument("--apply", action="store_true", help="actually terminate sessions (default dry-run)")
    args = parser.parse_args()

    from sqlalchemy import text

    from app.core.db_health import create_health_engine

    engine = create_health_engine(connect_timeout=5.0)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT pid, state, now() - state_change AS idle_age, left(query, 120) AS query
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                      AND (
                        (state = 'idle in transaction' AND now() - state_change > make_interval(secs => :min_age))
                        OR (state = 'active' AND wait_event_type = 'Lock' AND now() - query_start > make_interval(secs => :min_age))
                      )
                    ORDER BY state_change ASC
                    """
                ),
                {"min_age": args.min_age_sec},
            ).mappings().all()
            candidates = [dict(row) for row in rows]
            terminated: list[int] = []
            if args.apply:
                for row in candidates:
                    pid = int(row["pid"])
                    conn.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
                    terminated.append(pid)
            payload = {
                "apply": bool(args.apply),
                "minAgeSec": args.min_age_sec,
                "candidates": candidates,
                "terminated": terminated,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
