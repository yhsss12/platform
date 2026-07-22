"""从 model_type_definition 生成训练配置，桥接标准适配层。"""

from __future__ import annotations

from typing import Any, Optional

from app.services.model_type_service import (
    ADAPTER_BACKEND_MAP,
    ADAPTER_DOWNSTREAM_MAP,
    get_available_model_type,
    resolve_legacy_model_type_id,
    validate_structure_config,
)


def adapter_model_type_from_definition(defn: dict[str, Any]) -> str:
    adapter_key = str(defn.get("adapterKey") or "")
    return ADAPTER_BACKEND_MAP.get(adapter_key, str(defn.get("baseAlgorithm") or ""))


def structure_config_to_model_params(base_algorithm: str, structure_config: dict[str, Any]) -> dict[str, Any]:
    """将 model_type_definitions.structure_config 转为适配层 advancedConfig / modelParams。"""
    config = dict(structure_config or {})
    algo = (base_algorithm or "").strip()

    if algo == "robomimic_bc":
        return {
            "actor_hidden_dims": str(
                config.get("actor_hidden_dims") or config.get("hidden_dims") or "512,512"
            ),
            "l2_regularization": float(
                config.get("l2_regularization") if config.get("l2_regularization") is not None else config.get("weight_decay", 0.0)
            ),
        }

    if algo == "act":
        return {
            "chunk_size": int(config.get("chunk_size") or config.get("num_queries") or 100),
            "n_action_steps": int(config.get("n_action_steps") or config.get("chunk_size") or 100),
            "kl_weight": float(config.get("kl_weight") or 10.0),
            "latent_dim": int(config.get("latent_dim") or 32),
            "hidden_dim": int(config.get("hidden_dim") or 512),
            "dim_feedforward": int(config.get("dim_feedforward") or 2048),
            "enc_layers": int(config.get("enc_layers") or 4),
            "dec_layers": int(config.get("dec_layers") or 4),
            "nheads": int(config.get("nheads") or 8),
            "dropout": float(config.get("dropout") or 0.1),
        }

    if algo == "diffusion_policy":
        params: dict[str, Any] = {
            "n_obs_steps": int(config.get("n_obs_steps") or 2),
            "horizon": int(config.get("horizon") or 16),
            "n_action_steps": int(config.get("n_action_steps") or 8),
            "num_inference_steps": int(config.get("num_inference_steps") or 20),
            "weight_decay": float(config.get("weight_decay") or 1e-4),
        }
        if config.get("vision_encoder"):
            params["vision_encoder"] = str(config["vision_encoder"])
        if config.get("noise_scheduler"):
            params["noise_scheduler"] = str(config["noise_scheduler"])
        return params

    if algo == "pi0":
        return {
            "context_window": int(config.get("context_window") or 256),
            "action_horizon": int(config.get("action_horizon") or 16),
            "vision_encoder": str(config.get("vision_encoder") or "siglip"),
            "language_conditioning": bool(config.get("language_conditioning", True)),
            "action_head": str(config.get("action_head") or "flow_matching"),
            "tokenizer_or_processor": str(config.get("tokenizer_or_processor") or "default"),
        }

    return config


def build_training_config_from_model_type(
    model_type_definition: dict[str, Any],
    dataset_manifest: dict[str, Any],
    training_params: dict[str, Any],
    initialization_weight: Optional[dict[str, Any]] = None,
    save_policy: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    读取模型类型结构配置，合并训练任务参数，生成 create_training_job 所需字段。

    结构参数仅来自 model_type_definition；training_params 可覆盖 epochs/batch/lr/seed。
    """
    base_algorithm = str(model_type_definition.get("baseAlgorithm") or "")
    structure_config = dict(model_type_definition.get("structureConfig") or {})
    errors = validate_structure_config(base_algorithm, structure_config)
    if errors:
        raise ValueError(errors[0])

    adapter_key = str(model_type_definition.get("adapterKey") or "")
    training_backend = ADAPTER_BACKEND_MAP.get(adapter_key, base_algorithm)
    downstream_model_type = ADAPTER_DOWNSTREAM_MAP.get(adapter_key, model_type_definition.get("name") or "Robomimic")
    defaults = dict(model_type_definition.get("trainingDefaults") or {})
    save_policy = dict(save_policy or {})
    training_params = dict(training_params or {})

    model_params = structure_config_to_model_params(base_algorithm, structure_config)
    adapter_model_type = adapter_model_type_from_definition(model_type_definition)

    merged: dict[str, Any] = {
        "modelTypeId": model_type_definition.get("modelTypeId"),
        "downstreamModelType": downstream_model_type,
        "trainingBackend": training_backend,
        "epochs": int(training_params.get("epochs") or defaults.get("default_epochs") or 5),
        "batchSize": int(training_params.get("batchSize") or defaults.get("default_batch_size") or 16),
        "learningRate": float(
            training_params.get("learningRate") or defaults.get("default_learning_rate") or 1e-4
        ),
        "seed": int(training_params.get("seed") if training_params.get("seed") is not None else 1),
        "advancedEnabled": True,
        "modelParams": model_params,
        "saveFinal": bool(save_policy.get("saveFinal", True)),
        "saveBest": bool(save_policy.get("saveBest", False)),
        "checkpointIntervalEpochs": save_policy.get("checkpointIntervalEpochs"),
        "datasetId": dataset_manifest.get("datasetId"),
        "_adapterModelType": adapter_model_type,
        "_modelTypeDefinition": model_type_definition,
    }

    if initialization_weight:
        merged["pretrained"] = initialization_weight

    if training_params.get("device"):
        merged["device"] = training_params["device"]
    if training_params.get("deviceLabel"):
        merged["deviceLabel"] = training_params["deviceLabel"]
    if training_params.get("taskName"):
        merged["taskName"] = training_params["taskName"]
    if training_params.get("seedMode"):
        merged["seedMode"] = training_params["seedMode"]

    return merged


def resolve_training_payload_from_model_type(payload: dict[str, Any]) -> dict[str, Any]:
    """
    若请求携带 modelTypeId，解析模型定义并合并到 payload。
    兼容历史请求：仅有 downstreamModelType / trainingBackend 时映射到默认内置模型类型。
    """
    requested_training_backend = str(payload.get("trainingBackend") or "").strip()
    model_type_id = str(payload.get("modelTypeId") or "").strip()
    if not model_type_id:
        model_type_id = resolve_legacy_model_type_id(
            downstream_model_type=str(payload.get("downstreamModelType") or ""),
            training_backend=str(payload.get("trainingBackend") or ""),
        ) or ""

    if not model_type_id:
        return payload

    defn = get_available_model_type(model_type_id)
    manifest = dict(payload.get("datasetManifest") or {})
    training_params = {
        "epochs": payload.get("epochs"),
        "batchSize": payload.get("batchSize"),
        "learningRate": payload.get("learningRate"),
        "seed": payload.get("seed"),
        "seedMode": payload.get("seedMode"),
        "device": payload.get("device"),
        "deviceLabel": payload.get("deviceLabel"),
        "taskName": payload.get("taskName"),
    }
    save_policy = {
        "saveFinal": payload.get("saveFinal", True),
        "saveBest": payload.get("saveBest", False),
        "checkpointIntervalEpochs": payload.get("checkpointIntervalEpochs"),
    }

    merged = build_training_config_from_model_type(
        model_type_definition=defn,
        dataset_manifest=manifest,
        training_params=training_params,
        initialization_weight=payload.get("pretrained"),
        save_policy=save_policy,
    )

    result = dict(payload)
    result.update(merged)
    result["modelTypeId"] = model_type_id
    # A generic Robomimic model definition is shared by MuJoCo and Isaac.
    # Preserve the dataset-aware Isaac backend selected by the client instead
    # of overwriting it with the model definition's generic backend.
    if requested_training_backend == "isaac_robomimic_bc":
        result["trainingBackend"] = requested_training_backend
    # 训练任务不再接受外部结构参数覆盖
    result.pop("architectureConfig", None)
    if payload.get("modelParams") and not payload.get("modelTypeId"):
        # 仅 legacy 无 modelTypeId 时保留用户提交的 modelParams
        pass
    else:
        result["modelParams"] = merged["modelParams"]
    return result
