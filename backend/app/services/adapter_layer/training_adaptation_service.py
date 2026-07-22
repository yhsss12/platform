from __future__ import annotations

from typing import Any

from app.services.adapter_layer.adaptation_validator import (
    AdaptationValidation,
    build_explanation,
    validate_adaptation,
)
from app.services.adapter_layer.dataset_profiler import build_dataset_profile
from app.services.adapter_layer.model_adaptation_builder import build_model_adaptation_plan


def build_training_adaptation_plan(
    *,
    dataset_id: str | None = None,
    raw_manifest: dict[str, Any] | None = None,
    model_type: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    核心流程：datasetId/modelType → datasetProfile → modelAdaptation → validation。
    """
    from app.services.adapter_layer.adapter_service import resolve_manifest_by_dataset_id

    manifest = dict(raw_manifest or {})
    if dataset_id and not manifest:
        resolved = resolve_manifest_by_dataset_id(dataset_id)
        if resolved is None:
            raise LookupError(f"数据集未找到: {dataset_id}")
        manifest = resolved
    elif dataset_id:
        manifest.setdefault("datasetId", dataset_id)

    if not manifest:
        raise ValueError("需要提供 datasetId 或 datasetManifest")

    profile = build_dataset_profile(manifest)
    plan = build_model_adaptation_plan(profile, model_type, overrides)
    validation = validate_adaptation(profile, plan)
    explanation = build_explanation(profile, plan, validation)

    return {
        "datasetProfile": profile.to_dict(),
        "modelAdaptation": plan.to_dict(),
        "validation": {
            "adaptable": validation.adaptable,
            "warnings": validation.warnings,
            "errors": validation.errors,
        },
        "explanation": explanation,
        "configPatch": build_training_job_config_patch(profile, plan, validation),
        "adapterLayerVersion": "2.0",
    }


def build_training_job_config_patch(
    profile: Any,
    plan: Any,
    validation: AdaptationValidation,
) -> dict[str, Any]:
    """生成可直接合并到 create_training_job 请求的配置片段。"""
    training = dict(plan.trainingConfig)
    patch: dict[str, Any] = {
        "datasetId": profile.datasetId,
        "downstreamModelType": plan.downstreamModelType,
        "trainingBackend": plan.trainingBackend,
        "dataFormat": profile.format,
        "epochs": training.get("epochs", 5),
        "batchSize": training.get("batchSize", 16),
        "learningRate": training.get("learningRate", 1e-4),
        "device": training.get("device", "cuda"),
        "seed": training.get("seed", 1),
        "advancedEnabled": bool(training.get("advancedEnabled", True)),
        "modelParams": dict(plan.advancedConfig),
        "saveFinal": bool(training.get("saveFinal", True)),
        "saveBest": bool(training.get("saveBest", False)),
        "checkpointIntervalEpochs": training.get("checkpointIntervalEpochs"),
        "architectureConfig": dict(plan.architectureConfig),
        "dataLoaderConfig": dict(plan.dataLoaderConfig),
        "normalizationConfig": dict(plan.normalizationConfig),
        "inputConfig": dict(plan.inputConfig),
        "outputConfig": dict(plan.outputConfig),
    }
    if not validation.adaptable:
        patch["_adaptationBlocked"] = True
        patch["_adaptationErrors"] = list(validation.errors)
    return patch
