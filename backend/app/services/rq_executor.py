"""
RQ task execution entrypoint.

使用稳定的 `import worker`（加载 /app/backend/worker.py），避免：
1. backend/worker/ 目录与 worker.py 同名冲突（勿用含糊的 import 路径）
2. PyArmor 混淆后 `importlib.util.spec_from_file_location` 无法暴露 execute_task
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_backend_s = str(_BACKEND_ROOT)
if _backend_s not in sys.path:
    sys.path.insert(0, _backend_s)

_worker_mod = importlib.import_module("worker")


def execute_task(task: Dict[str, Any]):
    fn = getattr(_worker_mod, "execute_task", None)
    if fn is None:
        raise RuntimeError(
            "worker.execute_task 不可用；请确认 /app/backend/worker.py 已正确加载（PyArmor 场景勿用 spec_from_file_location）"
        )
    return fn(task)
