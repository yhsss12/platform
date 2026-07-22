from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, status

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.benchmark_adapters.base import BenchmarkTaskAdapter, is_ct_eval_job_id
from app.services.isaac_lab.job_paths import is_isaac_eval_job_id
from app.services.benchmark_adapters.cable_threading_adapter import CableThreadingTaskAdapter
from app.services.benchmark_adapters.dual_arm_cable_adapter import DualArmCableTaskAdapter
from app.services.benchmark_adapters.isaac_lab_adapter import IsaacLabBlockStackingAdapter
from app.services.benchmark_adapters.nut_assembly_adapter import NutAssemblyTaskAdapter

_TEMPLATE_ADAPTERS: dict[str, BenchmarkTaskAdapter] = {
    CableThreadingTaskAdapter.task_template_id: CableThreadingTaskAdapter(),
    DualArmCableTaskAdapter.task_template_id: DualArmCableTaskAdapter(),
    IsaacLabBlockStackingAdapter.task_template_id: IsaacLabBlockStackingAdapter(),
    NutAssemblyTaskAdapter.task_template_id: NutAssemblyTaskAdapter(),
}

_TASK_TYPE_ADAPTERS: dict[str, BenchmarkTaskAdapter] = {
    CableThreadingTaskAdapter.task_type: _TEMPLATE_ADAPTERS[CableThreadingTaskAdapter.task_template_id],
    DualArmCableTaskAdapter.task_type: _TEMPLATE_ADAPTERS[DualArmCableTaskAdapter.task_template_id],
    IsaacLabBlockStackingAdapter.task_type: _TEMPLATE_ADAPTERS[IsaacLabBlockStackingAdapter.task_template_id],
    NutAssemblyTaskAdapter.task_type: _TEMPLATE_ADAPTERS[NutAssemblyTaskAdapter.task_template_id],
    # 模板 ID 别名
    "cable_threading_single_arm": _TEMPLATE_ADAPTERS[CableThreadingTaskAdapter.task_template_id],
    "isaac_block_stacking": _TEMPLATE_ADAPTERS[IsaacLabBlockStackingAdapter.task_template_id],
}

_CABLE_ADAPTER = _TEMPLATE_ADAPTERS[CableThreadingTaskAdapter.task_template_id]
_DUAL_ARM_ADAPTER = _TEMPLATE_ADAPTERS[DualArmCableTaskAdapter.task_template_id]
_ISAAC_ADAPTER = _TEMPLATE_ADAPTERS[IsaacLabBlockStackingAdapter.task_template_id]
_NUT_ASSEMBLY_ADAPTER = _TEMPLATE_ADAPTERS[NutAssemblyTaskAdapter.task_template_id]


def list_benchmark_adapters() -> list[BenchmarkTaskAdapter]:
    return list(_TEMPLATE_ADAPTERS.values())


def get_benchmark_adapter(key: str) -> BenchmarkTaskAdapter:
    candidate = (key or "").strip()
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="taskTemplateId or taskType is required",
        )
    adapter = _TEMPLATE_ADAPTERS.get(candidate) or _TASK_TYPE_ADAPTERS.get(candidate)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No benchmark adapter registered for: {candidate}",
        )
    return adapter


def resolve_benchmark_adapter(request: EvaluateAsyncRequest) -> BenchmarkTaskAdapter:
    template_id = (request.taskTemplateId or "").strip()
    task_type = (request.taskType or "").strip()
    if template_id:
        return get_benchmark_adapter(template_id)
    if task_type:
        return get_benchmark_adapter(task_type)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="taskTemplateId or taskType is required",
    )


def resolve_benchmark_adapter_for_eval_job(eval_job_id: str) -> Optional[BenchmarkTaskAdapter]:
    """根据 evalJobId 解析 benchmark adapter；无法识别时返回 None（走 legacy 路径）。"""
    from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id

    if is_ct_eval_job_id(eval_job_id) or is_imported_workspace_eval_job_id(eval_job_id):
        return _CABLE_ADAPTER
    if _DUAL_ARM_ADAPTER.recognizes_eval_job_id(eval_job_id):
        return _DUAL_ARM_ADAPTER
    if _NUT_ASSEMBLY_ADAPTER.recognizes_eval_job_id(eval_job_id):
        return _NUT_ASSEMBLY_ADAPTER
    if is_isaac_eval_job_id(eval_job_id):
        return _ISAAC_ADAPTER
    return None
