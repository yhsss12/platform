from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.services.adapter_layer.dataset_profiler import DatasetProfile
from app.services.adapter_layer.model_capability_registry import get_model_capability

SIDE_CHANNEL_OBS_KEYS = frozenset({"attachment_enabled"})


def _attachment_training_metadata(
    profile: DatasetProfile,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    include = bool(
        overrides.get("includeAttachmentObs")
        or overrides.get("include_attachment_obs")
        or profile.includeAttachmentObs
    )
    return {
        "attachmentSideChannel": bool(profile.attachmentSideChannel),
        "attachmentInputMode": "low_dim_obs" if include else "not_used_by_policy",
        "attachmentControlMode": profile.attachmentControlMode or "eval_controller",
        "includeAttachmentObs": include,
    }


def _policy_low_dim_keys(profile: DatasetProfile, overrides: dict[str, Any]) -> list[str]:
    camera_keys = list(profile.cameraKeys)
    keys = [k for k in profile.observationKeys if k not in camera_keys]
    meta = _attachment_training_metadata(profile, overrides)
    if not meta["includeAttachmentObs"]:
        keys = [k for k in keys if k not in SIDE_CHANNEL_OBS_KEYS]
    return keys


@dataclass
class ModelAdaptationPlan:
    modelType: str
    displayName: str
    inputConfig: dict[str, Any] = field(default_factory=dict)
    outputConfig: dict[str, Any] = field(default_factory=dict)
    architectureConfig: dict[str, Any] = field(default_factory=dict)
    normalizationConfig: dict[str, Any] = field(default_factory=dict)
    dataLoaderConfig: dict[str, Any] = field(default_factory=dict)
    trainingConfig: dict[str, Any] = field(default_factory=dict)
    advancedConfig: dict[str, Any] = field(default_factory=dict)
    downstreamModelType: str = ""
    trainingBackend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelType": self.modelType,
            "displayName": self.displayName,
            "inputConfig": self.inputConfig,
            "outputConfig": self.outputConfig,
            "architectureConfig": self.architectureConfig,
            "normalizationConfig": self.normalizationConfig,
            "dataLoaderConfig": self.dataLoaderConfig,
            "trainingConfig": self.trainingConfig,
            "advancedConfig": self.advancedConfig,
            "downstreamModelType": self.downstreamModelType,
            "trainingBackend": self.trainingBackend,
        }


def _default_training_config(profile: DatasetProfile) -> dict[str, Any]:
    epochs = 5
    if profile.episodeCount >= 50:
        epochs = 10
    batch_size = 16
    if profile.observationType == "image":
        batch_size = 8
    return {
        "epochs": epochs,
        "batchSize": batch_size,
        "learningRate": 1e-4,
        "device": "cuda",
        "seed": 1,
        "saveFinal": True,
        "saveBest": True,
        "checkpointIntervalEpochs": None,
        "advancedEnabled": True,
    }


def _shape_meta_for_keys(profile: DatasetProfile) -> dict[str, Any]:
    shape_meta: dict[str, Any] = {"obs": {}, "action": {"shape": [profile.actionDim]}}
    for key in profile.observationKeys:
        if key in profile.cameraKeys:
            shape_meta["obs"][key] = {"type": "rgb", "shape": [3, 84, 84]}
        else:
            shape_meta["obs"][key] = {"type": "low_dim"}
    return shape_meta


def _normalization_for_keys(keys: list[str]) -> dict[str, Any]:
    return {
        "mode": "min_max",
        "obs_keys": {key: {"enabled": True} for key in keys},
        "action": {"enabled": True},
    }


def adapt_robomimic_bc(profile: DatasetProfile, overrides: dict[str, Any]) -> ModelAdaptationPlan:
    capability = get_model_capability("robomimic_bc")
    low_dim_keys = _policy_low_dim_keys(profile, overrides)
    obs_type = "low_dim" if profile.observationType in {"low_dim", "unknown"} else profile.observationType
    attach_meta = _attachment_training_metadata(profile, overrides)

    plan = ModelAdaptationPlan(
        modelType="robomimic_bc",
        displayName=capability.displayName if capability else "Robomimic BC",
        downstreamModelType=capability.downstreamModelType if capability else "Robomimic",
        trainingBackend="robomimic_bc",
        inputConfig={
            "obs_keys": low_dim_keys or profile.observationKeys,
            "observation_type": obs_type,
            "shape_meta": _shape_meta_for_keys(profile),
        },
        outputConfig={
            "action_dim": profile.actionDim,
            "action_space": profile.actionSpace,
        },
        architectureConfig={
            "encoder_type": "low_dim" if obs_type != "image" else "visual",
            "actor_hidden_dims": [512, 512],
            "policy_class": "GaussianActorNetwork",
            "input_dim": profile.stateDim or None,
            "output_dim": profile.actionDim,
        },
        normalizationConfig=_normalization_for_keys(low_dim_keys or profile.observationKeys),
        dataLoaderConfig={
            "frame_stack": 1,
            "seq_length": 1,
            "pad_frame_stack": True,
            "pad_seq_length": True,
            "dataset_keys": ["actions"] + (low_dim_keys or profile.observationKeys),
            "hdf5_cache_mode": "low_dim",
        },
        trainingConfig={**_default_training_config(profile), **attach_meta},
        advancedConfig={
            "actor_hidden_dims": "512,512",
            "l2_regularization": 0.0,
        },
    )

    if profile.horizon > 1:
        plan.dataLoaderConfig["seq_length"] = min(profile.horizon, 10)
    return _apply_overrides(plan, overrides)


def adapt_diffusion_policy(profile: DatasetProfile, overrides: dict[str, Any]) -> ModelAdaptationPlan:
    from app.services.dp_schema_resolver import resolve_dp_training_schema

    capability = get_model_capability("diffusion_policy")
    obs_horizon = 2
    pred_horizon = 16
    action_horizon = 8

    raw_profile = profile.to_dict()
    dp_schema = resolve_dp_training_schema(raw_profile, profile=raw_profile)

    camera_keys = list(dp_schema.image_keys or profile.cameraKeys)
    low_dim_keys = list(dp_schema.low_dim_keys) or _policy_low_dim_keys(profile, overrides)
    policy_obs_keys = list(camera_keys) + low_dim_keys
    has_images = bool(camera_keys) or profile.observationType in {"image", "mixed"}
    obs_encoder = "multi_image" if has_images and camera_keys else "low_dim"
    attach_meta = _attachment_training_metadata(profile, overrides)
    action_dataset_key = dp_schema.action_key
    if dp_schema.gripper_action_key:
        action_dataset_key = dp_schema.action_key

    plan = ModelAdaptationPlan(
        modelType="diffusion_policy",
        displayName=capability.displayName if capability else "Diffusion Policy",
        downstreamModelType=capability.downstreamModelType if capability else "Diffusion Policy",
        trainingBackend="diffusion_policy",
        inputConfig={
            "obs_keys": policy_obs_keys or profile.observationKeys,
            "camera_keys": camera_keys,
            "low_dim_keys": low_dim_keys,
            "observation_type": profile.observationType,
            "shape_meta": {
                "obs": _shape_meta_for_keys(profile)["obs"],
                "action": {"shape": [dp_schema.action_dim]},
            },
        },
        outputConfig={
            "action_dim": dp_schema.action_dim,
            "action_shape": [dp_schema.action_dim],
            "action_space": profile.actionSpace,
            "action_key": dp_schema.action_key,
            "gripper_action_key": dp_schema.gripper_action_key,
            "action_mode": dp_schema.action_mode,
            "controller_type": dp_schema.controller_type,
            "eval_executor": dp_schema.eval_executor,
            "trained_action_mode": dp_schema.trained_action_mode,
        },
        architectureConfig={
            "obs_encoder": obs_encoder,
            "obs_horizon": obs_horizon,
            "action_horizon": action_horizon,
            "pred_horizon": pred_horizon,
            "image_encoder": {
                "type": "resnet18" if camera_keys else "tiny_cnn",
                "camera_keys": camera_keys,
            },
        },
        normalizationConfig={
            "mode": "min_max",
            "obs_keys": {key: {"enabled": True, "mode": "min_max"} for key in profile.observationKeys},
            "action": {"enabled": True, "mode": "min_max"},
            "image": {"scale": 255.0},
        },
        dataLoaderConfig={
            "n_obs_steps": obs_horizon,
            "n_action_steps": action_horizon,
            "horizon": pred_horizon,
            "pad_before": obs_horizon - 1,
            "pad_after": action_horizon - 1,
            "dataset_keys": policy_obs_keys + [action_dataset_key],
        },
        trainingConfig={
            **_default_training_config(profile),
            **attach_meta,
            "batchSize": 8,
            "saveBest": True,
            "actionMode": dp_schema.trained_action_mode,
            "controllerType": dp_schema.controller_type,
            "evalExecutor": dp_schema.eval_executor,
        },
        advancedConfig={
            "observation_horizon": obs_horizon,
            "action_horizon": action_horizon,
            "pred_horizon": pred_horizon,
            "num_inference_steps": 16,
            "n_obs_steps": obs_horizon,
            "n_action_steps": action_horizon,
            "horizon": pred_horizon,
        },
    )
    return _apply_overrides(plan, overrides)


def adapt_act(profile: DatasetProfile, overrides: dict[str, Any]) -> ModelAdaptationPlan:
    from app.services.policy_schema_resolver import resolve_act_training_schema

    capability = get_model_capability("act")
    chunk_size = 20
    if profile.horizon > 1:
        chunk_size = min(max(profile.horizon // 4, 10), 100)

    raw_profile = profile.to_dict()
    act_schema = resolve_act_training_schema(raw_profile, profile=raw_profile)

    camera_keys = list(act_schema.image_keys or profile.cameraKeys)
    low_dim_keys = list(act_schema.low_dim_keys) or _policy_low_dim_keys(profile, overrides)
    policy_obs_keys = list(camera_keys) + low_dim_keys
    attach_meta = _attachment_training_metadata(profile, overrides)

    plan = ModelAdaptationPlan(
        modelType="act",
        displayName=capability.displayName if capability else "ACT",
        downstreamModelType=capability.downstreamModelType if capability else "ACT",
        trainingBackend="act",
        inputConfig={
            "state_dim": profile.stateDim,
            "camera_names": camera_keys,
            "camera_keys": camera_keys,
            "image_keys": camera_keys,
            "low_dim_keys": low_dim_keys,
            "obs_keys": policy_obs_keys or profile.observationKeys,
            "observation_type": profile.observationType,
            "act_variant": "image_proprio",
        },
        outputConfig={
            "action_dim": act_schema.action_dim,
            "action_space": profile.actionSpace,
            "action_key": act_schema.action_key,
            "gripper_action_key": act_schema.gripper_action_key,
            "action_mode": act_schema.action_mode,
            "controller_type": act_schema.controller_type,
            "eval_executor": act_schema.eval_executor,
            "trained_action_mode": act_schema.trained_action_mode,
            "chunk_size": chunk_size,
        },
        architectureConfig={
            "hidden_dim": 512,
            "dim_feedforward": 2048,
            "enc_layers": 4,
            "dec_layers": 4,
            "nheads": 8,
            "dropout": 0.1,
            "chunk_size": chunk_size,
            "kl_weight": 10.0,
            "latent_dim": 32,
            "backbone": "tiny_cnn",
        },
        normalizationConfig=_normalization_for_keys(profile.observationKeys),
        dataLoaderConfig={
            "chunk_size": chunk_size,
            "camera_names": camera_keys,
            "image_keys": camera_keys,
            "state_keys": low_dim_keys,
            "val_ratio": 0.1,
        },
        trainingConfig={
            **_default_training_config(profile),
            **attach_meta,
            "batchSize": 8,
            "saveBest": True,
        },
        advancedConfig={
            "chunk_size": chunk_size,
            "kl_weight": 10.0,
            "hidden_dim": 512,
            "latent_dim": 32,
            "enc_layers": 4,
            "dec_layers": 4,
            "nheads": 8,
            "dropout": 0.1,
            "dim_feedforward": 2048,
        },
    )
    return _apply_overrides(plan, overrides)


def adapt_torch_bc(profile: DatasetProfile, overrides: dict[str, Any]) -> ModelAdaptationPlan:
    capability = get_model_capability("torch_bc")
    low_dim_keys = _policy_low_dim_keys(profile, overrides)
    attach_meta = _attachment_training_metadata(profile, overrides)
    input_dim = profile.stateDim or profile.actionDim * 2 or 32

    plan = ModelAdaptationPlan(
        modelType="torch_bc",
        displayName=capability.displayName if capability else "BC (PyTorch)",
        downstreamModelType=capability.downstreamModelType if capability else "Robomimic",
        trainingBackend="torch_bc",
        inputConfig={
            "input_dim": input_dim,
            "obs_keys": low_dim_keys,
            "observation_type": "low_dim",
        },
        outputConfig={
            "action_dim": profile.actionDim,
            "action_space": profile.actionSpace,
        },
        architectureConfig={
            "hidden_dims": [256, 256],
            "dropout": 0.1,
            "activation": "relu",
            "output_dim": profile.actionDim,
        },
        normalizationConfig=_normalization_for_keys(low_dim_keys),
        dataLoaderConfig={
            "batch_keys": ["obs", "actions"],
            "shuffle": True,
        },
        trainingConfig={
            **_default_training_config(profile),
            **attach_meta,
            "saveBest": False,
        },
        advancedConfig={
            "hidden_dims": [256, 256],
            "dropout": 0.1,
        },
    )
    return _apply_overrides(plan, overrides)


def adapt_isaac_robomimic_bc(profile: DatasetProfile, overrides: dict[str, Any]) -> ModelAdaptationPlan:
    plan = adapt_robomimic_bc(profile, overrides)
    plan.modelType = "isaac_robomimic_bc"
    plan.displayName = "Isaac Robomimic BC"
    plan.trainingBackend = "isaac_robomimic_bc"
    plan.architectureConfig["algo_name"] = "bc"
    plan.architectureConfig["obs_mode"] = "low_dim"
    plan.trainingConfig["checkpointIntervalEpochs"] = 10
    plan.trainingConfig["saveBest"] = False
    return plan


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_overrides(plan: ModelAdaptationPlan, overrides: dict[str, Any]) -> ModelAdaptationPlan:
    if not overrides:
        return plan
    sections = {
        "inputConfig": plan.inputConfig,
        "outputConfig": plan.outputConfig,
        "architectureConfig": plan.architectureConfig,
        "normalizationConfig": plan.normalizationConfig,
        "dataLoaderConfig": plan.dataLoaderConfig,
        "trainingConfig": plan.trainingConfig,
        "advancedConfig": plan.advancedConfig,
    }
    for section_name, section in sections.items():
        patch = overrides.get(section_name)
        if isinstance(patch, dict):
            sections[section_name] = _deep_merge(section, patch)
    for key, value in overrides.items():
        if key in sections:
            continue
        if key in plan.trainingConfig and not isinstance(value, dict):
            plan.trainingConfig[key] = value
        elif key in plan.advancedConfig and not isinstance(value, dict):
            plan.advancedConfig[key] = value
    plan.inputConfig = sections["inputConfig"]
    plan.outputConfig = sections["outputConfig"]
    plan.architectureConfig = sections["architectureConfig"]
    plan.normalizationConfig = sections["normalizationConfig"]
    plan.dataLoaderConfig = sections["dataLoaderConfig"]
    plan.trainingConfig = sections["trainingConfig"]
    plan.advancedConfig = sections["advancedConfig"]
    return plan


def adapt_pi0(profile: DatasetProfile, overrides: dict[str, Any]) -> ModelAdaptationPlan:
    camera_keys = list(profile.cameraKeys)
    low_dim_keys = _policy_low_dim_keys(profile, overrides)
    policy_obs_keys = list(camera_keys) + low_dim_keys
    attach_meta = _attachment_training_metadata(profile, overrides)
    action_horizon = 16
    if profile.horizon > 1:
        action_horizon = min(max(profile.horizon, 4), 32)

    plan = ModelAdaptationPlan(
        modelType="pi0",
        displayName="pi0",
        downstreamModelType="pi0",
        trainingBackend="pi0",
        inputConfig={
            "obs_keys": policy_obs_keys or profile.observationKeys,
            "camera_keys": camera_keys,
            "low_dim_keys": low_dim_keys,
            "observation_type": profile.observationType,
            "language_conditioning": True,
        },
        outputConfig={
            "action_dim": profile.actionDim,
            "action_space": profile.actionSpace,
            "action_horizon": action_horizon,
        },
        architectureConfig={
            "context_window": 256,
            "action_horizon": action_horizon,
            "vision_encoder": "siglip",
            "language_conditioning": True,
            "action_head": "flow_matching",
            "tokenizer_or_processor": "default",
        },
        normalizationConfig=_normalization_for_keys(profile.observationKeys),
        dataLoaderConfig={
            "camera_keys": camera_keys,
            "low_dim_keys": low_dim_keys,
            "action_horizon": action_horizon,
            "val_ratio": 0.1,
        },
        trainingConfig={
            **_default_training_config(profile),
            **attach_meta,
            "batchSize": 8,
            "saveBest": False,
        },
        advancedConfig={
            "context_window": 256,
            "action_horizon": action_horizon,
            "vision_encoder": "siglip",
            "language_conditioning": True,
            "action_head": "flow_matching",
            "tokenizer_or_processor": "default",
        },
    )
    return _apply_overrides(plan, overrides)


MODEL_ADAPTERS: dict[str, Callable[[DatasetProfile, dict[str, Any]], ModelAdaptationPlan]] = {
    "robomimic_bc": adapt_robomimic_bc,
    "robomimic": adapt_robomimic_bc,
    "diffusion_policy": adapt_diffusion_policy,
    "act": adapt_act,
    "torch_bc": adapt_torch_bc,
    "isaac_robomimic_bc": adapt_isaac_robomimic_bc,
    "pi0": adapt_pi0,
}


def build_model_adaptation_plan(
    profile: DatasetProfile,
    model_type: str,
    overrides: dict[str, Any] | None = None,
) -> ModelAdaptationPlan:
    key = (model_type or "").strip().lower()
    adapter = MODEL_ADAPTERS.get(key)
    if adapter is None:
        capability = get_model_capability(key)
        if capability:
            adapter = MODEL_ADAPTERS.get(capability.modelType)
    if adapter is None:
        raise ValueError(f"未知或不支持的模型类型: {model_type}")
    return adapter(profile, overrides or {})
