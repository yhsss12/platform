"""
统一任务分发入口（双轨制）：
- USE_QUEUE=False：走旧执行链路（old_dispatch）
- USE_QUEUE=True：入 RQ，由 worker.execute_task 消费执行
- 数据资产导出：由 routes_data_assets.create_export_job 内 asyncio.create_task 在 API 事件循环执行，不走本模块。
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from app.core.config import settings
from app.services.task_profiles import get_task_profile


class RedisDispatchError(RuntimeError):
    """USE_QUEUE 模式下 RQ 入队失败（通常为 Redis 不可达或认证失败）。"""


def _is_redis_related_enqueue_error(exc: BaseException) -> bool:
    """判断是否为队列/Redis 侧连接或协议错误（避免把业务异常误判为 Redis 故障）。"""
    if isinstance(exc, (ConnectionRefusedError, TimeoutError, BrokenPipeError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (111, 110, 104, 32, 9):
        # 111 ECONNREFUSED, 110 ETIMEDOUT, 104 ECONNRESET, 32 EPIPE, 9 EBADF
        return True
    mod = (getattr(type(exc), "__module__", None) or "").lower()
    if "redis" in mod:
        return True
    if mod.startswith("rq."):
        return True
    return False


def enqueue(queue: str, task: Dict[str, Any]) -> str:
    """
    执行“入队”动作（兼容旧逻辑）：
    1) enqueue_fn：复用模块原有队列函数（如 conversion 队列）
    2) coroutine：提交 asyncio 任务
    3) callable：直接调用（由调用方决定其内部是否再入队）
    """
    enqueue_fn = task.get("enqueue_fn")
    if callable(enqueue_fn):
        enqueue_fn()
        return str(task.get("task_id") or task.get("job_id") or "")

    coroutine = task.get("coroutine")
    if coroutine is not None:
        import asyncio
        asyncio.create_task(coroutine)
        return str(task.get("task_id") or task.get("job_id") or "")

    runner = task.get("callable")
    if callable(runner):
        runner()
        return str(task.get("task_id") or task.get("job_id") or "")

    # 统一任务描述（type + payload）时，复用 worker 执行入口（本地直调）
    from worker import execute_task
    import threading
    threading.Thread(target=execute_task, args=(task,), daemon=True).start()
    return str(task.get("task_id") or task.get("job_id") or "")

    raise ValueError(f"任务缺少可执行入口，queue={queue}, type={task.get('type')}")


def route_task(task: Dict[str, Any]) -> str:
    task_type = str(task.get("type") or "").strip().lower()
    if not task_type:
        raise ValueError("task.type is required")
    return get_task_profile(task_type).queue


def old_dispatch(task: Dict[str, Any]) -> str:
    queue = route_task(task)
    return enqueue(queue, task)


# 默认关闭；可通过环境变量 USE_QUEUE=true 开启
USE_QUEUE = os.getenv("USE_QUEUE", "false").strip().lower() in ("1", "true", "yes", "on")


def dispatch_task(task: Dict[str, Any]) -> str:
    from app.services.task_job_store import create_task_job, update_task_status

    task_id = str(task.get("task_id") or "").strip() or str(uuid.uuid4())
    task_type = str(task.get("type") or "").strip().lower()
    queue_name = route_task(task)
    if USE_QUEUE:
        from app.services.queue_backpressure import enforce_queue_dispatch_allowed

        enforce_queue_dispatch_allowed(queue_name)

    normalized_task = {
        "task_id": task_id,
        "type": task_type,
        "user_id": task.get("user_id"),
        "params": dict(task),
    }
    normalized_task["params"]["task_id"] = task_id

    create_task_job(
        task_id=task_id,
        task_type=task_type,
        status="pending",
        user_id=(str(task.get("user_id")) if task.get("user_id") is not None else None),
        queue_name=queue_name,
        payload=normalized_task,
    )

    if USE_QUEUE:
        from app.services.task_queue import enqueue_task

        try:
            rq_job_id = enqueue_task(queue_name, normalized_task)
        except Exception as e:
            if _is_redis_related_enqueue_error(e):
                try:
                    update_task_status(
                        task_id,
                        "failed",
                        error=f"Redis/RQ 入队失败: {type(e).__name__}: {e}",
                    )
                except Exception:
                    pass
                raise RedisDispatchError(
                    f"任务队列不可用：无法连接 Redis（REDIS_HOST={settings.REDIS_HOST} "
                    f"REDIS_PORT={settings.REDIS_PORT}）。请确认 Redis 已启动，且与 API 进程网络一致"
                    f"（例如 compose 同网用服务名 redis；本机直连用 127.0.0.1）。详情: {e}"
                ) from e
            raise
        update_task_status(task_id, "queued", rq_job_id=rq_job_id)
        return task_id

    update_task_status(task_id, "queued", rq_job_id=task_id)
    old_dispatch(normalized_task)
    return task_id
