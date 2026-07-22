"""评测类型（专家策略 / 模型 / 数据集）统一解析。"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

EvaluationTypeKey = Literal["expert_policy", "model", "dataset"]

EVALUATION_TYPE_LABELS: dict[EvaluationTypeKey, str] = {
    "expert_policy": "专家策略评测",
    "model": "模型评测",
    "dataset": "数据集评测",
}

PRODUCT_EVALUATION_OBJECTS: dict[EvaluationTypeKey, str] = {
    "expert_policy": "expert_policy",
    "model": "trained_model",
    "dataset": "dataset",
}

PRODUCT_EVALUATION_MODES: dict[EvaluationTypeKey, str] = {
    "expert_policy": "expert_policy_evaluation",
    "model": "model_evaluation",
    "dataset": "dataset_evaluation",
}


class EvaluationTypeResolution(TypedDict):
    evaluationType: EvaluationTypeKey
    evaluationTypeLabel: str
    evaluationObject: str
    evaluationMode: str
    confidence: str
    basis: str


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


def _object_implies_type(evaluation_object: str) -> Optional[EvaluationTypeKey]:
    normalized = evaluation_object.lower()
    if normalized in {"trained_model", "model"}:
        return "model"
    if normalized == "dataset":
        return "dataset"
    if normalized in {"expert_policy", "expert"}:
        return "expert_policy"
    return None


def _mode_implies_model(mode: str) -> bool:
    normalized = mode.lower()
    return normalized in {
        "trained_model_evaluation",
        "model_evaluation",
        "model",
        "robomimic",
        "robomimic_bc",
    }


def _mode_implies_expert(mode: str) -> bool:
    normalized = mode.lower()
    return normalized in {
        "expert_policy_evaluation",
        "expert_policy",
        "expert",
        "policy_evaluation",
        "policy",
        "episode_stability",
        "scripted",
    }


def _mode_implies_dataset(mode: str) -> bool:
    normalized = mode.lower()
    return normalized in {
        "dataset_evaluation",
        "dataset_offline",
        "dataset_offline_evaluation",
        "offline_dataset_evaluation",
        "dataset",
    }


def resolve_evaluation_type_from_sources(
    *,
    evaluation_object: Optional[str] = None,
    evaluation_mode: Optional[str] = None,
    product_evaluation_mode: Optional[str] = None,
    model_asset_id: Optional[str] = None,
    model_asset_name: Optional[str] = None,
    dataset_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
    runner: Optional[str] = None,
    task_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, Any]] = None,
    evaluation_request: Optional[dict[str, Any]] = None,
) -> EvaluationTypeResolution:
    meta = _as_dict(metadata)
    metric_data = _as_dict(metrics)
    eval_request = _as_dict(evaluation_request)
    if not eval_request:
        eval_request = _as_dict(meta.get("evaluationRequest"))

    eval_object = _pick_str(
        evaluation_object,
        eval_request.get("evaluationObject"),
        meta.get("evaluationObject"),
        metric_data.get("evaluationObject"),
    )
    product_mode = _pick_str(
        product_evaluation_mode,
        eval_request.get("productEvaluationMode"),
        meta.get("productEvaluationMode"),
    )
    eval_mode = _pick_str(
        evaluation_mode,
        product_mode,
        eval_request.get("evaluationMode"),
        meta.get("evaluationMode"),
        metric_data.get("evaluationMode"),
    )
    model_id = _pick_str(
        model_asset_id,
        eval_request.get("modelAssetId"),
        meta.get("modelAssetId"),
        metric_data.get("modelAssetId"),
    )
    model_name = _pick_str(
        model_asset_name,
        eval_request.get("modelAssetName"),
        eval_request.get("modelName"),
        meta.get("modelAssetName"),
        metric_data.get("modelAssetName"),
        metric_data.get("modelName"),
    )
    ds_id = _pick_str(
        dataset_id,
        eval_request.get("datasetId"),
        meta.get("datasetId"),
        metric_data.get("datasetId"),
    )
    ds_name = _pick_str(
        dataset_name,
        eval_request.get("datasetName"),
        meta.get("datasetName"),
        metric_data.get("datasetName"),
    )
    runner_name = _pick_str(runner, meta.get("runner"))
    task_type_name = _pick_str(task_type, eval_request.get("taskType"), meta.get("taskType"))
    name = _pick_str(task_name, eval_request.get("taskName"), eval_request.get("modelName"))

    explicit_type = _pick_str(
        eval_request.get("evaluationType"),
        meta.get("evaluationType"),
        metric_data.get("evaluationType"),
    )
    if explicit_type in {"expert_policy", "model", "dataset"}:
        key = explicit_type  # type: ignore[assignment]
        return _build_resolution(key, "high", f"evaluationType={explicit_type}")

    explicit_label = _pick_str(
        eval_request.get("evaluationTypeLabel"),
        meta.get("evaluationTypeLabel"),
        metric_data.get("evaluationTypeLabel"),
    )
    if explicit_label in EVALUATION_TYPE_LABELS.values():
        key = next(k for k, v in EVALUATION_TYPE_LABELS.items() if v == explicit_label)
        return _build_resolution(key, "high", f"evaluationTypeLabel={explicit_label}")

    object_type = _object_implies_type(eval_object)

    if object_type == "model" or _mode_implies_model(eval_mode) or model_id:
        return _build_resolution("model", "high", _basis("model", eval_object, eval_mode, model_id))

    if (
        object_type == "dataset"
        or _mode_implies_dataset(eval_mode)
        or task_type_name == "dataset_offline"
        or runner_name == "dataset_offline_eval"
        or "离线数据集评测" in name
        or ((ds_id or ds_name) and not model_id)
    ):
        return _build_resolution("dataset", "high", _basis("dataset", eval_object, eval_mode, ds_id, ds_name))

    if object_type == "expert_policy" or _mode_implies_expert(eval_mode) or "专家策略" in name:
        confidence = "high" if eval_object or eval_mode else "medium"
        return _build_resolution("expert_policy", confidence, _basis("expert_policy", eval_object, eval_mode))

    if model_name:
        return _build_resolution("model", "low", "modelName without explicit mode")

    return _build_resolution("expert_policy", "low", "fallback default expert_policy")


def _basis(*parts: Any) -> str:
    rendered = [str(part).strip() for part in parts if str(part or "").strip()]
    return ", ".join(rendered) if rendered else "fallback"


def _build_resolution(key: EvaluationTypeKey, confidence: str, basis: str) -> EvaluationTypeResolution:
    return {
        "evaluationType": key,
        "evaluationTypeLabel": EVALUATION_TYPE_LABELS[key],
        "evaluationObject": PRODUCT_EVALUATION_OBJECTS[key],
        "evaluationMode": PRODUCT_EVALUATION_MODES[key],
        "confidence": confidence,
        "basis": basis,
    }


def resolve_evaluation_type_label(evaluation_mode: Optional[str]) -> str:
    return resolve_evaluation_type_from_sources(evaluation_mode=evaluation_mode)["evaluationTypeLabel"]


def enrich_evaluation_request_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(request_payload)
    resolution = resolve_evaluation_type_from_sources(
        evaluation_mode=str(payload.get("evaluationMode") or ""),
        model_asset_id=_pick_str(payload.get("modelAssetId")),
        dataset_id=_pick_str(payload.get("datasetId")),
        evaluation_request=payload,
    )
    payload["evaluationObject"] = resolution["evaluationObject"]
    payload["productEvaluationMode"] = resolution["evaluationMode"]
    payload["evaluationType"] = resolution["evaluationType"]
    payload["evaluationTypeLabel"] = resolution["evaluationTypeLabel"]
    return payload
