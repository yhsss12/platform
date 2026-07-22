from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.adapter_layer.manifest_schema import DatasetManifest, normalize_dataset_manifest
from app.services.adapter_layer.model_capability_registry import (
    ModelCapability,
    list_model_capabilities,
)


@dataclass
class ModelCompatibilityResult:
    modelType: str
    displayName: str
    compatible: bool
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "available"


@dataclass
class CompatibilityAnalysis:
    datasetId: str
    compatible: bool
    manifestVersion: str
    results: list[ModelCompatibilityResult] = field(default_factory=list)
    recommendedModels: list[str] = field(default_factory=list)
    blockingReasons: list[str] = field(default_factory=list)


def _robot_matches(required: tuple[str, ...], actual: str) -> bool:
    if not required:
        return True
    actual_norm = actual.lower().replace("-", "_")
    for robot in required:
        if robot.lower().replace("-", "_") in actual_norm or actual_norm in robot.lower().replace("-", "_"):
            return True
    return False


def _simulator_matches(required: tuple[str, ...], actual: str) -> bool:
    if not required:
        return True
    actual_norm = actual.lower()
    for sim in required:
        if sim in actual_norm or actual_norm in sim:
            return True
    return False


def _data_format_matches(required: tuple[str, ...], actual: str) -> bool:
    actual_norm = actual.upper()
    return any(fmt.upper() in actual_norm or actual_norm in fmt.upper() for fmt in required)


def _observation_type_matches(required: tuple[str, ...], actual: str) -> bool:
    if actual in required:
        return True
    if actual == "mixed":
        return bool(required)
    return False


def _check_model_compatibility(manifest: DatasetManifest, capability: ModelCapability) -> ModelCompatibilityResult:
    reasons: list[str] = []
    warnings: list[str] = []
    score = 100.0

    if capability.status == "unavailable":
        reasons.append(f"{capability.displayName} 当前不可用")
        score = 0.0
    elif capability.status == "experimental":
        warnings.append(f"{capability.displayName} 为实验性后端，尚未接入主流程")
        score -= 50.0

    if manifest.successCount < capability.minSuccessEpisodes:
        reasons.append(
            f"成功轨迹数不足：需要至少 {capability.minSuccessEpisodes} 条，"
            f"当前 {manifest.successCount} 条"
        )
        score = 0.0

    if manifest.episodeCount < capability.minEpisodeCount:
        reasons.append(
            f"episode 数不足：需要至少 {capability.minEpisodeCount}，当前 {manifest.episodeCount}"
        )
        score = 0.0

    if not _simulator_matches(capability.requiredSimulators, manifest.simulator):
        reasons.append(
            f"仿真环境不匹配：需要 {', '.join(capability.requiredSimulators)}，"
            f"当前为 {manifest.simulator or '未知'}"
        )
        score = 0.0

    if capability.requiredRobotTypes and not _robot_matches(capability.requiredRobotTypes, manifest.robotType):
        reasons.append(
            f"机器人类型不匹配：需要 {', '.join(capability.requiredRobotTypes)}，"
            f"当前为 {manifest.robotType or '未知'}"
        )
        score = 0.0

    if not _data_format_matches(capability.requiredDataFormats, manifest.dataFormat):
        reasons.append(
            f"数据格式不匹配：需要 {', '.join(capability.requiredDataFormats)}，"
            f"当前为 {manifest.dataFormat or '未知'}"
        )
        score = 0.0

    if not _observation_type_matches(capability.requiredObservationTypes, manifest.observationSpace.type):
        reasons.append(
            f"观测空间类型不匹配：需要 {', '.join(capability.requiredObservationTypes)}，"
            f"当前为 {manifest.observationSpace.type}"
        )
        score = 0.0

    if capability.requiredObservationKeys:
        available_keys = {key.lower() for key in manifest.observationSpace.keys}
        missing = [key for key in capability.requiredObservationKeys if key.lower() not in available_keys]
        if missing and manifest.observationSpace.type != "image":
            quality = manifest.raw.get("quality") if isinstance(manifest.raw.get("quality"), dict) else {}
            if not quality.get("hasImage"):
                reasons.append(
                    f"缺少必要观测字段：{', '.join(missing)}"
                )
                score = 0.0
            else:
                warnings.append(f"manifest 未声明观测键 {', '.join(missing)}，但 quality.hasImage=true")
                score -= 10.0
        elif missing:
            reasons.append(f"缺少必要观测字段：{', '.join(missing)}")
            score = 0.0

    if not manifest.storageUri:
        reasons.append("缺少 storageUri / artifacts 存储路径")
        score = 0.0

    if capability.prefersSequenceActions:
        if manifest.actionSpace.supportsSequence:
            score += 15.0
        else:
            warnings.append(f"{capability.displayName} 更适合支持序列动作的数据集")
            score -= 20.0

    if capability.modelType == "robomimic_bc" and manifest.observationSpace.type == "image":
        if not manifest.observationSpace.keys or all("image" in k.lower() for k in manifest.observationSpace.keys):
            warnings.append("纯图像观测数据集更推荐使用 Diffusion Policy")
            score -= 15.0

    compatible = score > 0 and not reasons
    return ModelCompatibilityResult(
        modelType=capability.modelType,
        displayName=capability.displayName,
        compatible=compatible,
        score=max(score, 0.0),
        reasons=reasons,
        warnings=warnings,
        status=capability.status,
    )


def analyze_dataset_compatibility(dataset_manifest: DatasetManifest | dict[str, Any]) -> CompatibilityAnalysis:
    """分析数据集与各模型的兼容性。"""
    manifest = (
        dataset_manifest
        if isinstance(dataset_manifest, DatasetManifest)
        else normalize_dataset_manifest(dataset_manifest)
    )

    results: list[ModelCompatibilityResult] = []
    for capability in list_model_capabilities():
        results.append(_check_model_compatibility(manifest, capability))

    recommended = recommend_training_models(manifest, results)
    compatible_any = any(item.compatible for item in results)
    blocking = []
    if manifest.successCount <= 0:
        blocking.append("数据集无成功轨迹，无法训练")
    if not manifest.storageUri:
        blocking.append("缺少数据集存储路径")

    return CompatibilityAnalysis(
        datasetId=manifest.datasetId,
        compatible=compatible_any and not blocking,
        manifestVersion=manifest.manifestVersion,
        results=results,
        recommendedModels=recommended,
        blockingReasons=blocking,
    )


def recommend_training_models(
    dataset_manifest: DatasetManifest | dict[str, Any],
    precomputed_results: list[ModelCompatibilityResult] | None = None,
) -> list[str]:
    """根据兼容性分析推荐模型，按 score 降序。"""
    if isinstance(dataset_manifest, dict):
        manifest = normalize_dataset_manifest(dataset_manifest)
    else:
        manifest = dataset_manifest

    if precomputed_results is None:
        precomputed_results = [
            _check_model_compatibility(manifest, cap) for cap in list_model_capabilities()
        ]

    ranked = sorted(
        [item for item in precomputed_results if item.compatible],
        key=lambda item: (
            -item.score,
            next(
                (cap.priority for cap in list_model_capabilities() if cap.modelType == item.modelType),
                999,
            ),
            item.modelType,
        ),
    )
    return [item.modelType for item in ranked]
