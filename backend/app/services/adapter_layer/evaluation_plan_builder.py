from __future__ import annotations

from typing import Any

from app.services.adapter_layer.model_capability_registry import get_model_capability


DEFAULT_NUM_EPISODES = 10
DEFAULT_SEED = 0


def _resolve_from_training_plan(training_plan: dict[str, Any]) -> dict[str, Any]:
    model_type = str(training_plan.get("modelType") or training_plan.get("trainingBackend") or "")
    capability = get_model_capability(model_type)
    task_template_id = (
        training_plan.get("taskTemplateId")
        or (capability.taskTemplateId if capability else None)
        or "cable_threading_single_arm"
    )
    policy_type = (
        training_plan.get("evaluationPolicyType")
        or (capability.evaluationPolicyType if capability else None)
        or "robomimic_bc"
    )
    metrics = list(
        training_plan.get("metrics")
        or (capability.defaultMetrics if capability else ())
        or ("metric_cable_success_rate_v1",)
    )
    return {
        "modelType": model_type,
        "taskTemplateId": task_template_id,
        "policyType": policy_type,
        "metrics": metrics,
        "taskName": training_plan.get("taskName") or "",
        "simulator": training_plan.get("simulator") or "",
        "robotType": training_plan.get("robotType") or "",
        "datasetId": training_plan.get("datasetId"),
        "downstreamModelType": training_plan.get("downstreamModelType"),
        "trainingBackend": training_plan.get("trainingBackend"),
    }


def _resolve_from_model_asset(model_asset: dict[str, Any]) -> dict[str, Any]:
    model_type = str(model_asset.get("modelType") or model_asset.get("trainingBackend") or "")
    capability = get_model_capability(model_type)
    task_template_id = str(
        model_asset.get("taskTemplateId")
        or (capability.taskTemplateId if capability else None)
        or "cable_threading_single_arm"
    )
    policy_type = str(
        model_asset.get("policyType")
        or (capability.evaluationPolicyType if capability else None)
        or "robomimic_bc"
    )
    metrics = list(
        model_asset.get("metrics")
        or (capability.defaultMetrics if capability else ())
        or ("metric_cable_success_rate_v1",)
    )
    return {
        "modelType": model_type,
        "taskTemplateId": task_template_id,
        "policyType": policy_type,
        "metrics": metrics,
        "taskName": model_asset.get("taskName") or model_asset.get("name") or "",
        "simulator": model_asset.get("simulator") or "",
        "robotType": model_asset.get("robotType") or "",
        "datasetId": model_asset.get("sourceDatasetId") or model_asset.get("datasetId"),
        "modelAssetId": model_asset.get("id") or model_asset.get("modelAssetId"),
        "checkpointPath": model_asset.get("checkpointPath"),
        "downstreamModelType": model_asset.get("framework") or model_asset.get("downstreamModelType"),
        "trainingBackend": model_asset.get("trainingBackend") or model_asset.get("backendType"),
    }


def build_evaluation_plan(model_asset_or_training_plan: dict[str, Any]) -> dict[str, Any]:
    """根据模型资产或训练计划生成评测计划（不启动真实评测）。"""
    payload = dict(model_asset_or_training_plan or {})
    is_training_plan = any(
        key in payload
        for key in ("epochs", "batchSize", "learningRate", "advancedConfig", "savePolicy")
    )
    if is_training_plan:
        resolved = _resolve_from_training_plan(payload)
    else:
        resolved = _resolve_from_model_asset(payload)

    evaluation_mode = "trained_model_evaluation"
    if resolved.get("policyType") == "scripted":
        evaluation_mode = "expert_policy_evaluation"

    return {
        "evaluationMode": evaluation_mode,
        "taskTemplateId": resolved["taskTemplateId"],
        "taskName": resolved["taskName"],
        "simulator": resolved["simulator"],
        "robotType": resolved["robotType"],
        "policyType": resolved["policyType"],
        "modelType": resolved["modelType"],
        "numEpisodes": int(payload.get("numEpisodes") or DEFAULT_NUM_EPISODES),
        "seed": int(payload.get("seed") if payload.get("seed") is not None else DEFAULT_SEED),
        "record": bool(payload.get("record", True)),
        "headless": bool(payload.get("headless", True)),
        "metrics": resolved["metrics"],
        "datasetId": resolved.get("datasetId"),
        "modelAssetId": resolved.get("modelAssetId") or payload.get("modelAssetId"),
        "checkpointPath": resolved.get("checkpointPath") or payload.get("checkpointPath"),
        "downstreamModelType": resolved.get("downstreamModelType"),
        "trainingBackend": resolved.get("trainingBackend"),
        "adapterLayerVersion": "1.0",
    }
