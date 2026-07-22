#!/usr/bin/env python3
"""
RQ 高负载模拟：经 dispatch_task 真实入队 type=stress（仅 sleep + 可选短 CPU，无 MinIO 写入）。

用法（在 backend 目录或任意位置）：
  conda activate IDE
  python tools/operations/stress_queue.py --queue cpu_queue --count 80 --sleep-ms 60 --cpu-ms 15

前置：Redis 可达、PostgreSQL 可达（会写 task_jobs）、对应队列的 worker 已启动（且已加载含 stress 的 worker.py）。
可选：export QUEUE_STRESS=cpu_queue 与 --queue 二选一（脚本 --queue 会覆盖当前进程的 QUEUE_STRESS）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress-test RQ via stress tasks")
    parser.add_argument("--queue", default=os.getenv("QUEUE_STRESS", "io_queue"), help="目标队列名")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--sleep-ms", type=float, default=80.0)
    parser.add_argument("--cpu-ms", type=float, default=20.0)
    parser.add_argument("--burst-workers", type=int, default=24, help="并发入队线程数")
    parser.add_argument("--wait-sec", type=int, default=0, help="入队后等待队列清空，0 表示不等待")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="等待时轮询 Redis 队列长度间隔（秒）",
    )
    args = parser.parse_args()

    backend = Path(__file__).resolve().parents[2]
    os.environ["QUEUE_STRESS"] = str(args.queue)
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    os.chdir(backend)

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore
    if load_dotenv:
        for p in (backend / ".env", backend.parent / ".env"):
            if p.exists():
                load_dotenv(p, override=False)

    from redis import Redis
    from app.core.config import settings
    from app.services.dispatcher import USE_QUEUE, dispatch_task
    from app.services.task_profiles import TASK_PROFILES

    if not USE_QUEUE:
        print("错误: USE_QUEUE 未开启，dispatch_task 不会入 RQ。请在 .env 设置 USE_QUEUE=true。", file=sys.stderr)
        return 2

    prof = TASK_PROFILES.get("stress")
    resolved_queue = prof.queue if prof else args.queue
    print(
        f"stress -> 队列 {resolved_queue} | 任务数={args.count} sleep_ms={args.sleep_ms} cpu_ms={args.cpu_ms} "
        f"入队并发={args.burst_workers} USE_QUEUE={USE_QUEUE}"
    )

    pwd = settings.REDIS_PASSWORD
    if pwd is not None and str(pwd).strip() == "":
        pwd = None
    r = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB, password=pwd, decode_responses=True)
    r.ping()

    def qlen(name: str) -> int:
        return int(r.llen(f"rq:queue:{name}"))

    def one_dispatch(_: int) -> str:
        tid = str(uuid.uuid4())
        dispatch_task(
            {
                "type": "stress",
                "task_id": tid,
                "sleep_ms": args.sleep_ms,
                "cpu_ms": args.cpu_ms,
            }
        )
        return tid

    t0 = time.perf_counter()
    errors: list[str] = []
    ids: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.burst_workers)) as ex:
        futs = [ex.submit(one_dispatch, i) for i in range(args.count)]
        for fut in as_completed(futs):
            try:
                ids.append(fut.result())
            except Exception as e:
                errors.append(str(e))
    t_enqueue = time.perf_counter() - t0

    print(f"入队完成: {len(ids)} 成功, {len(errors)} 失败, 耗时 {t_enqueue:.2f}s")
    if errors:
        print("首条错误:", errors[0][:500], file=sys.stderr)
    print(f"Redis rq:queue:{resolved_queue} 长度 = {qlen(resolved_queue)}")

    if args.wait_sec > 0:
        deadline = time.monotonic() + args.wait_sec
        while time.monotonic() < deadline:
            n = qlen(resolved_queue)
            print(f"  [{time.strftime('%H:%M:%S')}] queue_depth={n}")
            if n == 0:
                print(f"队列 {resolved_queue} 已清空（在 {args.wait_sec}s 预算内）。")
                break
            time.sleep(max(0.2, args.poll_interval))
        else:
            print(f"超时: {args.wait_sec}s 后 rq:queue:{resolved_queue} 仍 depth={qlen(resolved_queue)}", file=sys.stderr)
            return 1

    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
