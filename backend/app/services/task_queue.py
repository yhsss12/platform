"""
Redis Queue (RQ) 任务队列适配层。

默认用于 dispatcher 的可选双轨能力：
- 队列可用时：入 RQ
- 队列不可用时：由 dispatcher 回退旧逻辑
"""
from __future__ import annotations

from typing import Any, Dict

from redis import Redis
from rq import Queue, Retry

from app.core.config import settings
from app.services.task_profiles import TASK_PROFILES, get_task_profile

_pwd = settings.REDIS_PASSWORD
if _pwd is not None and str(_pwd).strip() == "":
    _pwd = None

redis_conn = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=_pwd,
)

_queue_names = sorted({p.queue for p in TASK_PROFILES.values()} | {"gpu_queue", "cpu_queue", "io_queue", "collect_queue"})
queues = {name: Queue(name, connection=redis_conn) for name in _queue_names}


def get_timeout(task_type: str) -> int:
    return int(get_task_profile(task_type).timeout_seconds)


def _is_retryable(task_type: str) -> bool:
    # collect 任务通常是实时交互链路，默认不自动重试，避免重复动作
    return bool(get_task_profile(task_type).retryable)


def enqueue_task(queue_name: str, task: Dict[str, Any]) -> str:
    if queue_name not in queues:
        queues[queue_name] = Queue(queue_name, connection=redis_conn)
    queue = queues[queue_name]
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("task.task_id is required for enqueue")
    task_type = str(task.get("type") or "")
    timeout = get_timeout(task_type)
    retry = Retry(max=3, interval=[10, 30, 60]) if _is_retryable(task_type) else None
    job = queue.enqueue(
        "app.services.rq_executor.execute_task",
        task,
        job_id=task_id,
        job_timeout=timeout,
        retry=retry,
    )
    return str(job.id)
