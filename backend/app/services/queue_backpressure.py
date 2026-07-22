"""
RQ 队列背压：限制单队列「排队 + 已开始执行」的任务数量，避免无节制提交导致 CPU/磁盘被拖垮。

环境变量（0 或未设置表示不限制）：
  RQ_MAX_PENDING_<队列名大写>，例如：
  RQ_MAX_PENDING_CPU_QUEUE=24
  RQ_MAX_PENDING_IO_QUEUE=48
"""
from __future__ import annotations

import os
from typing import Optional

from rq import Queue
from rq.registry import StartedJobRegistry

from app.services.task_queue import redis_conn


class QueueBackpressureError(RuntimeError):
    """队列积压超过配置上限，拒绝新任务入队。"""

    def __init__(self, queue_name: str, depth: int, limit: int) -> None:
        self.queue_name = queue_name
        self.depth = depth
        self.limit = limit
        super().__init__(
            f"队列 {queue_name} 任务积压已达上限（当前约 {depth}，上限 {limit}），请稍后再提交"
        )


def _parse_nonneg_int(raw: Optional[str], default: int) -> int:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return max(0, int(str(raw).strip()))
    except Exception:
        return default


def max_pending_for_queue(queue_name: str) -> int:
    q = (queue_name or "").strip()
    if not q:
        return 0
    env_key = "RQ_MAX_PENDING_" + q.upper().replace("-", "_")
    return _parse_nonneg_int(os.getenv(env_key), 0)


def count_pending_and_started(queue_name: str) -> int:
    """Redis 队列长度 + RQ StartedJobRegistry（已被 worker 取出执行中的 job）。"""
    q = (queue_name or "").strip()
    if not q:
        return 0
    try:
        queued = int(redis_conn.llen(f"rq:queue:{q}"))
    except Exception:
        queued = 0
    try:
        rq_queue = Queue(q, connection=redis_conn)
        started = int(StartedJobRegistry(queue=rq_queue).count)
    except Exception:
        started = 0
    return max(0, queued + started)


def enforce_queue_dispatch_allowed(queue_name: str) -> None:
    limit = max_pending_for_queue(queue_name)
    if limit <= 0:
        return
    depth = count_pending_and_started(queue_name)
    if depth >= limit:
        raise QueueBackpressureError(queue_name, depth, limit)
