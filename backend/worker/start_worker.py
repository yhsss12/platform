#!/usr/bin/env python3
"""
RQ Worker 启动入口。

用法示例：
- 单队列: python worker/start_worker.py --queues gpu_queue
- 多队列: python worker/start_worker.py --queues gpu_queue,cpu_queue
- 指定 Redis: python worker/start_worker.py --queues io_queue --redis-host 127.0.0.1 --redis-port 6379
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import sys
from typing import List

from redis import Redis
from rq import Queue, Worker

try:
    from dotenv import load_dotenv

    _backend_dir = Path(__file__).resolve().parent.parent
    # Ensure backend root is importable for RQ job targets (app.*, worker.py).
    if str(_backend_dir) not in sys.path:
        sys.path.insert(0, str(_backend_dir))
    for _env in (_backend_dir / ".env", _backend_dir.parent / ".env"):
        if _env.exists():
            load_dotenv(_env, override=False)
except Exception:
    pass


def _parse_queues(raw: str) -> List[str]:
    parts = [x.strip() for x in (raw or "").split(",")]
    queues = [x for x in parts if x]
    if not queues:
        raise ValueError("at least one queue is required")
    return queues


def main() -> None:
    parser = argparse.ArgumentParser(description="Start RQ worker for selected queues")
    parser.add_argument(
        "--queues",
        required=True,
        help="Comma-separated queue names, e.g. gpu_queue,cpu_queue",
    )
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--redis-db", type=int, default=int(os.getenv("REDIS_DB", "0")))
    parser.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD", ""))
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Run in burst mode (quit when queues are empty)",
    )
    args = parser.parse_args()

    queue_names = _parse_queues(args.queues)

    # Ensure backend/ is on PYTHONPATH so RQ can import `worker.py`
    # (backend/worker.py defines `execute_task` which tasks enqueue as "worker.execute_task").
    backend_dir = Path(__file__).resolve().parent.parent
    backend_dir_str = str(backend_dir)
    if backend_dir_str not in sys.path:
        sys.path.insert(0, backend_dir_str)

    redis_conn = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=(args.redis_password or None),
    )

    worker_name = f"rq-worker-{'-'.join(queue_names)}"
    queues = [Queue(name, connection=redis_conn) for name in queue_names]
    worker = Worker(queues, connection=redis_conn, name=worker_name)
    worker.work(burst=args.burst, with_scheduler=False)


if __name__ == "__main__":
    main()

