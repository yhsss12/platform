from __future__ import annotations

from fastapi import HTTPException, status

from app.services.evaluation.base import BaseEvaluationAdapter
from app.services.evaluation.cable_threading_adapter import CableThreadingEvaluationAdapter
from app.services.evaluation.dual_arm_cable_adapter import DualArmCableEvaluationAdapter

_ADAPTERS: dict[str, BaseEvaluationAdapter] = {
    CableThreadingEvaluationAdapter.task_type: CableThreadingEvaluationAdapter(),
    DualArmCableEvaluationAdapter.task_type: DualArmCableEvaluationAdapter(),
}


def get_evaluation_adapter(task_type: str) -> BaseEvaluationAdapter:
    adapter = _ADAPTERS.get(task_type)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported taskType for evaluation: {task_type}",
        )
    return adapter


def list_registered_task_types() -> list[str]:
    return sorted(_ADAPTERS.keys())


def get_supported_modes(task_type: str) -> list[str]:
    return list(get_evaluation_adapter(task_type).supported_modes)
