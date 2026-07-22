from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict


@dataclass(frozen=True)
class TaskProfile:
    queue: str
    timeout_seconds: int
    retryable: bool
    concurrency_limit: int


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(str(raw).strip())
    except Exception:
        value = default
    return max(minimum, value)


TASK_PROFILES: Dict[str, TaskProfile] = {
    "collect": TaskProfile(
        queue=os.getenv("QUEUE_COLLECT", "collect_queue"),
        timeout_seconds=_env_int("TASK_TIMEOUT_COLLECT_SECONDS", 1800),
        retryable=False,
        concurrency_limit=_env_int("TASK_CONCURRENCY_COLLECT", 2),
    ),
    "conversion": TaskProfile(
        queue=os.getenv("QUEUE_CONVERSION", "cpu_queue"),
        timeout_seconds=_env_int("TASK_TIMEOUT_CONVERSION_SECONDS", 3600),
        retryable=True,
        concurrency_limit=_env_int("TASK_CONCURRENCY_CONVERSION", 3),
    ),
    "annotation": TaskProfile(
        queue=os.getenv("QUEUE_ANNOTATION", "gpu_queue"),
        timeout_seconds=_env_int("TASK_TIMEOUT_ANNOTATION_SECONDS", 2400),
        retryable=True,
        concurrency_limit=_env_int("TASK_CONCURRENCY_ANNOTATION", 2),
    ),
    "batch": TaskProfile(
        queue=os.getenv("QUEUE_BATCH", "io_queue"),
        timeout_seconds=_env_int("TASK_TIMEOUT_BATCH_SECONDS", 1800),
        retryable=True,
        concurrency_limit=_env_int("TASK_CONCURRENCY_BATCH", 2),
    ),
    "export": TaskProfile(
        queue=os.getenv("QUEUE_EXPORT", "io_queue"),
        timeout_seconds=_env_int("TASK_TIMEOUT_EXPORT_SECONDS", 2400),
        retryable=True,
        concurrency_limit=_env_int("TASK_CONCURRENCY_EXPORT", 2),
    ),
    # 压测/健康检查：仅消耗 worker 时间片，不写业务数据（队列由 QUEUE_STRESS 指定）
    "stress": TaskProfile(
        queue=os.getenv("QUEUE_STRESS", "io_queue"),
        timeout_seconds=_env_int("TASK_TIMEOUT_STRESS_SECONDS", 300),
        retryable=False,
        concurrency_limit=_env_int("TASK_CONCURRENCY_STRESS", 32),
    ),
}


def get_task_profile(task_type: str) -> TaskProfile:
    t = (task_type or "").strip().lower()
    profile = TASK_PROFILES.get(t)
    if profile is not None:
        return profile
    return TaskProfile(
        queue=os.getenv("QUEUE_DEFAULT", "io_queue"),
        timeout_seconds=_env_int("TASK_TIMEOUT_DEFAULT_SECONDS", 900),
        retryable=False,
        concurrency_limit=_env_int("TASK_CONCURRENCY_DEFAULT", 2),
    )

