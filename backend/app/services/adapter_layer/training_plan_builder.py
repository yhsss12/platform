from __future__ import annotations

from typing import Any

from app.services.adapter_layer.manifest_schema import DatasetManifest, normalize_dataset_manifest
from app.services.adapter_layer.model_capability_registry import get_model_capability


DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = 1


def _advanced_config_for_model(model_type: str) -> dict[str, Any]:
    key = (model_type or "").strip().lower()
    if key in {"robomimic_bc", "robomimic"}:
        return {
            "actor_hidden_dims": "512,512",
            "l2_regularization": 0.0,
        }
    if key == "diffusion_policy":
        return {
            "observation_horizon": 2,
            "action_horizon": 8,
            "num_inference_steps": 16,
        }
    if key == "act":
        return {
            "chunk_size": 100,
            "kl_weight": 10.0,
            "hidden_dim": 512,
        }
    if key == "isaac_robomimic_bc":
        return {
            "algo_name": "bc",
            "obs_mode": "low_dim",
        }
    if key == "torch_bc":
        return {
            "hidden_dims": [256, 256],
            "dropout": 0.1,
        }
    return {}


def build_training_plan(dataset_manifest: DatasetManifest | dict[str, Any], model_type: str) -> dict[str, Any]:
    """根据 manifest 与模型类型生成训练计划（不启动真实训练）。"""
    manifest = (
        dataset_manifest
        if isinstance(dataset_manifest, DatasetManifest)
        else normalize_dataset_manifest(dataset_manifest)
    )
    capability = get_model_capability(model_type)
    if capability is None:
        raise ValueError(f"未知模型类型: {model_type}")

    from app.services.adapter_layer.compatibility_checker import analyze_dataset_compatibility

    analysis = analyze_dataset_compatibility(manifest)
    model_result = next((item for item in analysis.results if item.modelType == capability.modelType), None)
    if model_result is None or not model_result.compatible:
        reason = "; ".join(model_result.reasons if model_result else ["模型未注册"])
        raise ValueError(f"数据集与 {capability.displayName} 不兼容: {reason}")

    save_policy = {
        "robomimic_bc": {"saveFinal": True, "saveBest": True, "checkpointIntervalEpochs": None},
        "diffusion_policy": {"saveFinal": True, "saveBest": True, "checkpointIntervalEpochs": None},
        "act": {"saveFinal": True, "saveBest": False, "checkpointIntervalEpochs": None},
        "isaac_robomimic_bc": {"saveFinal": True, "saveBest": False, "checkpointIntervalEpochs": 10},
        "torch_bc": {"saveFinal": True, "saveBest": False, "checkpointIntervalEpochs": None},
    }.get(capability.modelType, {"saveFinal": True, "saveBest": False, "checkpointIntervalEpochs": None})

    return {
        "datasetId": manifest.datasetId,
        "datasetName": manifest.datasetName,
        "modelType": capability.modelType,
        "downstreamModelType": capability.downstreamModelType,
        "trainingBackend": capability.backendKey,
        "dataFormat": manifest.dataFormat,
        "epochs": DEFAULT_EPOCHS,
        "batchSize": DEFAULT_BATCH_SIZE,
        "learningRate": DEFAULT_LEARNING_RATE,
        "device": DEFAULT_DEVICE,
        "seed": DEFAULT_SEED,
        "advancedEnabled": bool(_advanced_config_for_model(capability.modelType)),
        "advancedConfig": _advanced_config_for_model(capability.modelType),
        "savePolicy": save_policy,
        "storageUri": manifest.storageUri,
        "taskName": manifest.taskName,
        "simulator": manifest.simulator,
        "robotType": manifest.robotType,
        "manifestVersion": manifest.manifestVersion,
        "adapterLayerVersion": "1.0",
        "notes": capability.notes,
    }
