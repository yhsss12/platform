#!/usr/bin/env python3
"""实时监控 RQ：队列深度 + 已注册 worker（Ctrl+C 退出）。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

INTERVAL = float(os.getenv("RQ_WATCH_INTERVAL", "2"))


def main() -> None:
    backend = Path(__file__).resolve().parents[2]
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    os.chdir(backend)

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv:
        for p in (backend / ".env", backend.parent / ".env"):
            if p.exists():
                load_dotenv(p, override=False)

    import redis
    from app.core.config import settings

    pwd = settings.REDIS_PASSWORD
    if pwd is not None and str(pwd).strip() == "":
        pwd = None
    r = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=pwd,
        decode_responses=True,
    )

    queues = ["cpu_queue", "io_queue", "gpu_queue", "collect_queue", "default"]
    print(f"监控 Redis {settings.REDIS_HOST}:{settings.REDIS_PORT} db={settings.REDIS_DB}，间隔 {INTERVAL}s（Ctrl+C 退出）\n")

    try:
        while True:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            header = f"=== RQ 监控 {ts} ==="
            if sys.stdout.isatty():
                header = f"\033[2J\033[H{header}"
            lines = [header]
            try:
                r.ping()
                lines.append("Redis: OK")
            except Exception as e:
                lines.append(f"Redis: FAIL {e}")
                print("\n".join(lines), flush=True)
                time.sleep(INTERVAL)
                continue

            total_pending = 0
            for q in queues:
                n = int(r.llen(f"rq:queue:{q}"))
                total_pending += n
                lines.append(f"  rq:queue:{q}\tdepth={n}")
            lines.append(f"  (pending sum)\t{total_pending}")

            workers = sorted(r.smembers("rq:workers") or [])
            lines.append(f"\nrq:workers count={len(workers)}")
            for w in workers[:16]:
                lines.append(f"  {w}")
            if len(workers) > 16:
                lines.append(f"  ... +{len(workers) - 16} more")

            print("\n".join(lines), flush=True)
            time.sleep(max(0.5, INTERVAL))
    except KeyboardInterrupt:
        print("\n已退出监控。")


if __name__ == "__main__":
    main()
