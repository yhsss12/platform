"""标准适配层（Adapter Layer）— 独立模块，用于数据集 manifest 兼容性分析与计划生成。"""

from app.services.adapter_layer.adapter_service import (
    analyze_dataset_compatibility,
    build_evaluation_plan,
    build_training_adaptation_plan,
    build_training_plan,
    normalize_dataset_manifest,
    recommend_training_models,
    resolve_manifest_by_dataset_id,
)

__all__ = [
    "normalize_dataset_manifest",
    "analyze_dataset_compatibility",
    "recommend_training_models",
    "build_training_plan",
    "build_training_adaptation_plan",
    "build_evaluation_plan",
    "resolve_manifest_by_dataset_id",
]
