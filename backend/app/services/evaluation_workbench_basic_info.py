"""评测工作台右侧「基础信息」字段统一解析。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.services.evaluation.evaluation_type import (
    EVALUATION_TYPE_LABELS,
    EvaluationTypeKey,
    resolve_evaluation_type_from_sources,
)

TASK_TYPE_ASSOCIATED_NAMES: dict[str, str] = {
    "cable_threading": "线缆穿杆",
    "cable_threading_single_arm": "线缆穿杆",
    "dual_arm_cable_manipulation": "线缆整理",
    "nut_assembly": "螺母装配",
    "block_stacking": "物块堆叠",
    "isaac_block_stacking": "物块堆叠",
    "isaaclab_franka_stack_cube": "物块堆叠",
    "stacking": "物块堆叠",
    "dataset_offline": "离线数据集",
}

TASK_TEMPLATE_ASSOCIATED_NAMES: dict[str, str] = {
    "cable_threading_single_arm": "线缆穿杆",
    "dual_arm_cable_manipulation": "线缆整理",
    "nut_assembly_single_arm": "螺母装配",
    "isaac_block_stacking": "物块堆叠",
    "isaaclab_franka_stack_cube": "物块堆叠",
}

SIMULATION_PLATFORM_LABELS: dict[str, str] = {
    "mujoco": "MuJoCo",
    "isaac_lab": "Isaac Lab",
    "isaaclab": "Isaac Lab",
    "isaac_sim": "Isaac Sim",
    "isaacsim": "Isaac Sim",
    "isaac": "Isaac Lab",
}

EVALUATION_OBJECT_LABELS: dict[str, str] = {
    "expert_policy": "专家策略",
    "expert": "专家策略",
    "trained_model": "已训练模型",
    "model": "已训练模型",
    "dataset": "数据集",
}

STATUS_LABELS: dict[str, str] = {
    "completed": "已完成",
    "running": "评测中",
    "failed": "失败",
    "queued": "待评测",
    "pending": "待评测",
    "canceled": "已取消",
    "cancelled": "已取消",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick_str(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_eval_metadata(job_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    context: dict[str, Any] = {}
    eval_request: dict[str, Any] = {}
    for rel in ("metadata/evaluation_context.json", "metadata/evaluation_request.json"):
        data = _read_json(job_root / rel)
        if not data:
            continue
        if not context:
            context = data
        nested = data.get("evaluationRequest")
        if isinstance(nested, dict) and nested:
            eval_request = nested
            break
        if not eval_request:
            eval_request = data
    if not eval_request:
        nested = context.get("evaluationRequest")
        if isinstance(nested, dict):
            eval_request = nested
    return context, eval_request


def _lookup_model_asset_display_name(
    *,
    model_asset_id: str,
    checkpoint_path: str,
    task_name: str,
) -> str:
    """从训练任务 registry / checkpoint manifest 解析模型资产展示名。"""
    from app.services.checkpoint_registry import read_registry

    search_roots: list[Path] = []
    if checkpoint_path:
        cp = Path(checkpoint_path)
        parts = cp.parts
        if "jobs" in parts:
            job_idx = parts.index("jobs")
            if job_idx + 1 < len(parts):
                search_roots.append(Path(*parts[: job_idx + 2]))

    for train_job_dir in search_roots:
        manifest_file = (
            train_job_dir / "artifacts" / "checkpoint_manifests" / f"{model_asset_id}.json"
        )
        manifest_data = _read_json(manifest_file)
        resolved = _pick_str(manifest_data.get("displayName"), manifest_data.get("name"))
        if resolved and resolved != task_name:
            return resolved

        registry = read_registry(train_job_dir)
        for asset in registry.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            if str(asset.get("modelAssetId") or "") != model_asset_id:
                continue
            resolved = _pick_str(asset.get("displayName"), asset.get("name"))
            if resolved and resolved != task_name:
                return resolved
    return ""


def _resolve_model_asset_name(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    eval_request: dict[str, Any],
    metrics: dict[str, Any],
    task_name: str,
) -> str:
    model_asset_id = _pick_str(
        payload.get("modelAssetId"),
        context.get("modelAssetId"),
        eval_request.get("modelAssetId"),
        metrics.get("modelAssetId"),
    )
    checkpoint_path = _pick_str(
        eval_request.get("checkpointPath"),
        context.get("checkpointPath"),
        metrics.get("checkpointPath"),
    )

    explicit = _pick_str(
        payload.get("modelAssetName"),
        eval_request.get("modelAssetName"),
        context.get("modelAssetName"),
        metrics.get("modelAssetName"),
    )
    if explicit and explicit != task_name:
        return explicit

    from_registry = _lookup_model_asset_display_name(
        model_asset_id=model_asset_id,
        checkpoint_path=checkpoint_path,
        task_name=task_name,
    )
    if from_registry:
        return from_registry

    fallback = _pick_str(
        eval_request.get("modelName"),
        context.get("modelName"),
        metrics.get("modelName"),
    )
    if fallback and fallback != task_name:
        return fallback
    return model_asset_id or ""


def _resolve_task_name(
    *,
    eval_job_id: str,
    status_payload: dict[str, Any],
    context: dict[str, Any],
    eval_request: dict[str, Any],
) -> str:
    config = _as_dict(context.get("config") or eval_request.get("config"))
    cable = _as_dict(context.get("cableThreading") or eval_request.get("cableThreading"))
    dual_arm = _as_dict(context.get("dualArmCable") or eval_request.get("dualArmCable"))

    return _pick_str(
        status_payload.get("taskName"),
        status_payload.get("name"),
        eval_request.get("taskName"),
        eval_request.get("evaluationTaskName"),
        eval_request.get("modelName"),
        context.get("taskName"),
        context.get("displayName"),
        context.get("templateDisplayName"),
        context.get("modelName"),
        cable.get("taskName"),
        cable.get("modelName"),
        dual_arm.get("taskName"),
        dual_arm.get("modelName"),
        config.get("taskName"),
        config.get("name"),
        config.get("modelName"),
        eval_job_id,
    )


def _format_simulation_platform(raw: str, task_type: str) -> str:
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in SIMULATION_PLATFORM_LABELS:
        return SIMULATION_PLATFORM_LABELS[normalized]
    if normalized:
        return raw.strip()
    if task_type in {"block_stacking", "isaac_block_stacking", "isaaclab_franka_stack_cube", "stacking"}:
        return "Isaac Lab"
    if task_type in {"cable_threading", "dual_arm_cable_manipulation"}:
        return "MuJoCo"
    return ""


def _format_robot_label(raw: str, task_type: str) -> str:
    value = raw.strip()
    if not value:
        if task_type == "dual_arm_cable_manipulation":
            return "Dual FR3"
        if task_type in {"block_stacking", "isaac_block_stacking", "isaaclab_franka_stack_cube", "stacking"}:
            return "Franka Panda"
        return ""
    lowered = value.lower()
    if lowered in {"panda", "franka_panda"}:
        return "Panda" if task_type == "cable_threading" else "Franka Panda"
    if lowered in {"franka", "franka_panda"}:
        return "Franka Panda"
    if lowered in {"dual_fr3", "dual fr3", "fr3"}:
        return "Dual FR3"
    if lowered == "ur5e":
        return "UR5e"
    if task_type == "dual_arm_cable_manipulation" and "dual" not in lowered and "双臂" not in value:
        return f"Dual {value}"
    return value


def _associated_task_name(task_type: str, task_template_id: str) -> str:
    template = task_template_id.strip()
    if template and template in TASK_TEMPLATE_ASSOCIATED_NAMES:
        return TASK_TEMPLATE_ASSOCIATED_NAMES[template]
    normalized = task_type.strip()
    if normalized in TASK_TYPE_ASSOCIATED_NAMES:
        return TASK_TYPE_ASSOCIATED_NAMES[normalized]
    return ""


def _status_label(status: str) -> str:
    normalized = (status or "").strip().lower()
    return STATUS_LABELS.get(normalized, status or "—")


def _evaluation_object_label(evaluation_object: str, evaluation_type: EvaluationTypeKey) -> str:
    explicit = EVALUATION_OBJECT_LABELS.get(evaluation_object.lower())
    if explicit:
        return explicit
    if evaluation_type == "model":
        return "已训练模型"
    if evaluation_type == "dataset":
        return "数据集"
    return "专家策略"


def build_evaluation_workbench_basic_info(
    eval_job_id: str,
    job_root: Path,
    status_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = dict(status_payload or {})
    context, eval_request = _load_eval_metadata(job_root)
    metrics = _as_dict(payload.get("metrics"))
    live = _as_dict(payload.get("live"))
    config = _as_dict(context.get("config") or eval_request.get("config"))
    cable = _as_dict(context.get("cableThreading") or eval_request.get("cableThreading"))
    dual_arm = _as_dict(context.get("dualArmCable") or eval_request.get("dualArmCable"))

    task_type = _pick_str(
        payload.get("taskType"),
        context.get("taskType"),
        eval_request.get("taskType"),
        config.get("taskType"),
    )
    task_template_id = _pick_str(
        payload.get("taskTemplateId"),
        context.get("taskTemplateId"),
        eval_request.get("taskTemplateId"),
        config.get("taskTemplateId"),
    )

    type_resolution = resolve_evaluation_type_from_sources(
        evaluation_object=_pick_str(
            payload.get("evaluationObject"),
            context.get("evaluationObject"),
            eval_request.get("evaluationObject"),
        ),
        evaluation_mode=_pick_str(
            payload.get("evaluationMode"),
            context.get("evaluationMode"),
            eval_request.get("evaluationMode"),
            eval_request.get("productEvaluationMode"),
            metrics.get("evaluationMode"),
        ),
        product_evaluation_mode=_pick_str(
            context.get("productEvaluationMode"),
            eval_request.get("productEvaluationMode"),
        ),
        model_asset_id=_pick_str(
            payload.get("modelAssetId"),
            context.get("modelAssetId"),
            eval_request.get("modelAssetId"),
            metrics.get("modelAssetId"),
        ),
        model_asset_name=_pick_str(
            payload.get("modelAssetName"),
            context.get("modelAssetName"),
            eval_request.get("modelAssetName"),
            eval_request.get("modelName"),
            context.get("modelName"),
            metrics.get("modelAssetName"),
            metrics.get("modelName"),
        ),
        dataset_id=_pick_str(
            payload.get("datasetId"),
            context.get("datasetId"),
            eval_request.get("datasetId"),
            metrics.get("datasetId"),
        ),
        dataset_name=_pick_str(
            payload.get("datasetName"),
            context.get("datasetName"),
            eval_request.get("datasetName"),
            metrics.get("datasetName"),
        ),
        task_type=task_type,
        task_name=_resolve_task_name(
            eval_job_id=eval_job_id,
            status_payload=payload,
            context=context,
            eval_request=eval_request,
        ),
        metrics=metrics,
        metadata=context,
        evaluation_request=eval_request,
    )

    evaluation_type = type_resolution["evaluationType"]
    evaluation_type_label = type_resolution["evaluationTypeLabel"]
    evaluation_object = type_resolution["evaluationObject"]

    simulation_platform = _format_simulation_platform(
        _pick_str(
            payload.get("simulationPlatform"),
            config.get("simulationPlatform"),
            config.get("simulatorBackend"),
            context.get("simulatorBackend"),
            context.get("simulationPlatform"),
        ),
        task_type,
    )

    robot_type = _format_robot_label(
        _pick_str(
            payload.get("robotType"),
            config.get("robotType"),
            config.get("robot"),
            cable.get("robot"),
            dual_arm.get("robot"),
            context.get("robot"),
            eval_request.get("robot"),
            live.get("robot"),
        ),
        task_type,
    )

    task_name = _resolve_task_name(
        eval_job_id=eval_job_id,
        status_payload=payload,
        context=context,
        eval_request=eval_request,
    )

    model_asset_name = _resolve_model_asset_name(
        payload=payload,
        context=context,
        eval_request=eval_request,
        metrics=metrics,
        task_name=task_name,
    )
    dataset_name = _pick_str(
        payload.get("datasetName"),
        eval_request.get("datasetName"),
        context.get("datasetName"),
        metrics.get("datasetName"),
        eval_request.get("datasetId"),
        context.get("datasetId"),
    )

    info = {
        "taskName": task_name,
        "evaluationTypeLabel": evaluation_type_label,
        "evaluationObjectLabel": _evaluation_object_label(evaluation_object, evaluation_type),
        "simulationPlatform": simulation_platform or "—",
        "statusLabel": _status_label(str(payload.get("status") or "")),
        "robotType": robot_type or None,
        "modelAssetName": model_asset_name or None,
        "datasetName": dataset_name or None,
        "associatedTaskName": _associated_task_name(task_type, task_template_id) or None,
        "evaluationType": evaluation_type,
        "evaluationObject": evaluation_object,
    }
    return info


def attach_workbench_basic_info(
    payload: dict[str, Any],
    *,
    eval_job_id: str,
    job_root: Path,
) -> dict[str, Any]:
    if not job_root.is_dir():
        return payload
    info = build_evaluation_workbench_basic_info(eval_job_id, job_root, payload)
    merged = dict(payload)
    merged["workbenchBasicInfo"] = info
    for key, value in info.items():
        if key.endswith("Label"):
            continue
        if value is not None and merged.get(key) in (None, ""):
            merged[key] = value
    merged.setdefault("taskName", info.get("taskName"))
    merged.setdefault("evaluationTypeLabel", info.get("evaluationTypeLabel"))
    merged.setdefault("evaluationObject", info.get("evaluationObject"))
    merged.setdefault("simulationPlatform", info.get("simulationPlatform"))
    if info.get("robotType"):
        merged.setdefault("robotType", info.get("robotType"))
    return merged
