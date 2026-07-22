from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from app.services.evaluation.evaluation_type import resolve_evaluation_type_label as resolve_product_type_label


def resolve_task_display_name(task_type: str) -> str:
    t = (task_type or "").strip()
    if t == "cable_threading":
        return "线缆穿杆"
    if t == "dual_arm_cable_manipulation":
        return "线缆整理"
    if t in {"block_stacking", "isaac_block_stacking"}:
        return "物块堆叠"
    return t or "评测任务"


def resolve_evaluation_type_label(evaluation_mode: str) -> str:
    return resolve_product_type_label(evaluation_mode)


def build_evaluation_display_name(task_type: str, evaluation_mode: str, now: datetime | None = None) -> str:
    """正式评测展示名：{任务名称}{评测类型}_{YYYYMMDD}_{HHmm}"""
    dt = now or datetime.now()
    ts = dt.strftime("%Y%m%d_%H%M")
    return f"{resolve_task_display_name(task_type)}{resolve_evaluation_type_label(evaluation_mode)}_{ts}"


def extract_user_evaluation_task_name(request: Any) -> Optional[str]:
    """从评测请求中提取用户输入的任务名称（兼容 taskName / modelName 及嵌套块）。"""
    if request is None:
        return None

    if isinstance(request, dict):
        for value in (
            request.get("taskName"),
            request.get("task_name"),
            request.get("name"),
            request.get("evaluationTaskName"),
            request.get("modelName"),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        for block_key in ("cableThreading", "dualArmCable"):
            block = request.get(block_key)
            if isinstance(block, dict):
                for key in ("taskName", "modelName"):
                    nested = block.get(key)
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
        return None

    for value in (
        getattr(request, "taskName", None),
        getattr(request, "task_name", None),
        getattr(request, "name", None),
        getattr(request, "evaluationTaskName", None),
        getattr(request, "modelName", None),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    for block in (getattr(request, "cableThreading", None), getattr(request, "dualArmCable", None)):
        if isinstance(block, dict):
            for key in ("taskName", "modelName"):
                nested = block.get(key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return None


def resolve_evaluation_task_name(
    request: Any,
    task_type: str,
    evaluation_mode: str,
    now: datetime | None = None,
) -> tuple[str, str]:
    """返回 (最终任务名称, 自动生成展示名)。用户输入优先于自动生成名。"""
    generated_display_name = build_evaluation_display_name(task_type, evaluation_mode, now=now)
    user_task_name = extract_user_evaluation_task_name(request)
    task_name = user_task_name or generated_display_name
    return task_name, generated_display_name
