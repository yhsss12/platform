from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.adapter_layer.dataset_profiler import DatasetProfile
from app.services.adapter_layer.model_adaptation_builder import ModelAdaptationPlan


@dataclass
class AdaptationValidation:
    adaptable: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_adaptation(profile: DatasetProfile, plan: ModelAdaptationPlan) -> AdaptationValidation:
    """校验数据集 profile 与模型适配方案是否可训练。"""
    result = AdaptationValidation()
    result.warnings.extend(profile.warnings)

    if not profile.storageUri:
        result.errors.append("缺少数据集存储路径 (storageUri / artifacts.hdf5)")

    if profile.successCount <= 0 and profile.episodeCount <= 0:
        result.errors.append("数据集无可用 episode / 成功轨迹")

    if profile.actionDim <= 0:
        result.errors.append("无法确定 action_dim，无法构建 policy 输出层")

    model_type = plan.modelType

    if model_type == "robomimic_bc":
        obs_keys = plan.inputConfig.get("obs_keys") or []
        if not obs_keys:
            result.errors.append("Robomimic BC 需要至少一个 observation key")
        if profile.observationType == "image" and not profile.cameraKeys:
            result.warnings.append("数据集为纯图像观测，Robomimic BC 通常需要 low_dim 状态键")

    elif model_type == "diffusion_policy":
        if not profile.observationKeys and profile.actionDim <= 0:
            result.errors.append("Diffusion Policy 需要可推断的 observation keys 或 action_dim")
        elif profile.observationType == "low_dim" or not profile.cameraKeys:
            result.warnings.append(
                "数据集无图像观测，将使用 low_dim Diffusion Policy 配置（image_keys 为空）"
            )
        elif profile.observationType not in {"image", "mixed", "low_dim", "unknown"}:
            result.warnings.append(f"观测类型 {profile.observationType} 可能需人工确认 DP 配置")

    elif model_type == "act":
        if not profile.cameraKeys:
            result.errors.append(
                "ACT requires image observations, but dataset has only low_dim observations."
            )
            result.adaptable = False
        if profile.stateDim <= 0 and profile.cameraKeys:
            result.warnings.append("ACT state_dim 未能推断，将仅使用图像特征")

    elif model_type == "pi0":
        if not profile.cameraKeys:
            result.errors.append("pi0 需要图像观测，但数据集仅有 low_dim observations")
            result.adaptable = False
        if profile.actionDim <= 0:
            result.errors.append("pi0 无法确定 action_dim")
            result.adaptable = False
        language_on = bool(
            (plan.architectureConfig or {}).get("language_conditioning", True)
        )
        if language_on and not profile.taskType:
            result.warnings.append("pi0 启用 language_conditioning，建议 manifest 提供 taskType / taskDescription")

    elif model_type == "torch_bc":
        if profile.observationType == "image" and not profile.stateDim:
            result.warnings.append("torch_bc 更适合 low_dim 观测，当前数据集主要为图像")
        if profile.robotType not in {"dual_fr3", "dual_arm", "unknown"} and "dual" not in profile.robotType.lower():
            result.warnings.append(f"torch_bc 通常用于双臂数据集，当前 robotType={profile.robotType}")

    elif model_type == "isaac_robomimic_bc":
        if "isaac" not in profile.simulator.lower():
            result.warnings.append(
                f"Isaac Robomimic BC 通常用于 Isaac 数据，当前 simulator={profile.simulator}"
            )

    if result.errors:
        result.adaptable = False

    return result


def build_explanation(profile: DatasetProfile, plan: ModelAdaptationPlan, validation: AdaptationValidation) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"数据集 {profile.datasetId}：{profile.observationType} 观测，"
        f"action_dim={profile.actionDim}，episodes={profile.episodeCount}"
    )
    if profile.inferenceSources:
        lines.append(f"profile 推断来源: {', '.join(profile.inferenceSources)}")

    if plan.modelType == "robomimic_bc":
        lines.append(
            f"Robomimic BC：obs_keys={plan.inputConfig.get('obs_keys')}，"
            f"hidden_dims={plan.architectureConfig.get('actor_hidden_dims')}"
        )
    elif plan.modelType == "diffusion_policy":
        lines.append(
            f"Diffusion Policy：camera_keys={plan.inputConfig.get('camera_keys')}，"
            f"obs_horizon={plan.architectureConfig.get('obs_horizon')}，"
            f"action_horizon={plan.architectureConfig.get('action_horizon')}"
        )
    elif plan.modelType == "act":
        lines.append(
            f"ACT：camera_names={plan.inputConfig.get('camera_names')}，"
            f"chunk_size={plan.architectureConfig.get('chunk_size')}"
        )
    elif plan.modelType == "torch_bc":
        lines.append(
            f"torch_bc：input_dim={plan.inputConfig.get('input_dim')}，"
            f"hidden_dims={plan.architectureConfig.get('hidden_dims')}"
        )

    lines.extend(validation.warnings)
    return lines
