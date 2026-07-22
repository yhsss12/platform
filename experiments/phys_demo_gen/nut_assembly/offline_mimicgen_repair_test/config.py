"""Offline MimicGen Repair Test 配置。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _EXPERIMENT_DIR.parents[2]

DEFAULT_FAILED_HDF5 = _REPO_ROOT / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_SUCCESS_HDF5 = _REPO_ROOT / "mnt" / "data" / "demo.hdf5"
DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_PINN_MODEL = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "model.pt"
DEFAULT_V1F_MODEL = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "model_v1f.pt"
DEFAULT_OUTPUT_DIR = _EXPERIMENT_DIR / "outputs" / "offline_mimicgen_repair_test"
DEFAULT_V1F_OUTPUT_DIR = _EXPERIMENT_DIR / "outputs" / "offline_mimicgen_repair_test_v1f"

DEMO_REPAIR_CONFIGS: dict[str, dict[str, Any]] = {
    "demo_4": {
        "failure_type": "insertion_failed",
        "active": "insertion",
        "search_kind": "insertion",
        "label": "failed",
    },
    "demo_2": {
        "failure_type": "grasp_failed",
        "active": "grasp",
        "search_kind": "grasp",
        "label": "failed",
    },
    "demo_3": {
        "failure_type": "transport_failed",
        "secondary_failure_type": "lift_underdeveloped",
        "legacy_failure_type": "lift_failed",
        "active": "lift",
        "search_kind": "lift",
        "label": "failed",
    },
}

SELECTION_METHODS = ("pinn_top_k", "explicit_top_k", "random_top_k")
V1F_SELECTION_METHODS = (
    "v1e_pinn_top_k",
    "v1f_pinn_top_k",
    "v1f_plain_top_k",
    "v1f_diverse_top_k",
    "explicit_top_k",
    "random_top_k",
    "physics_residual_top_k",
    "physics_residual_gated_top_k",
    "physics_residual_p1p2_gated_top_k",
    "physics_residual_insertion_gated_top_k",
)
CONTEXT_SOURCES = ("original_failed_context", "cem_refined_context")
DEFAULT_CONTEXT_ALIGNMENT_OUTPUT_DIR = _EXPERIMENT_DIR / "outputs" / "context_alignment_ablation"
DEFAULT_V1F_ALIGNED_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_repair_parameter_model"

# PINN 模型资产注册表（default_baseline + experimental candidates）
DEFAULT_MODEL_ASSETS_REGISTRY = _EXPERIMENT_DIR / "model_assets" / "registry.json"
DEFAULT_BASELINE_MODEL = (
    _EXPERIMENT_DIR
    / "outputs"
    / "v1f_aligned_repair_parameter_model"
    / "original_failed"
    / "trained_model"
    / "model_v1f_aligned_original.pt"
)
V1G_STAGE1_LITE_P1P2_MODEL = (
    _EXPERIMENT_DIR / "outputs" / "v1g_stage1_lite_p1p2" / "trained_model" / "model_v1g_stage1_lite_p1p2.pt"
)
V1G_STAGE1_LITE_P1P2_METADATA = (
    _EXPERIMENT_DIR / "outputs" / "v1g_stage1_lite_p1p2" / "model_asset_metadata.json"
)
V1G_STAGE1_LITE_P1P2_MODEL_CARD = (
    _EXPERIMENT_DIR / "outputs" / "v1g_stage1_lite_p1p2" / "model_card_v1g_stage1_lite_p1p2.md"
)

# Physics residual repair（独立层，不覆盖 aligned-original checkpoint）
ENABLE_PHYSICS_RESIDUAL_REPAIR = False
ENABLE_INSERTION_STAGE_REPAIR = False
DEFAULT_SUCCESS_REFERENCE_JSONL = (
    _EXPERIMENT_DIR / "outputs" / "v1f_100base" / "success_reference_samples.jsonl"
)
DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR = _EXPERIMENT_DIR / "outputs" / "physics_residual_repair"

# demo_3：当前链路不可修复，V1-G-lite 训练/主验收排除，仅保留诊断
DEMO_3_V1G_LITE_DIAGNOSTIC: dict[str, Any] = {
    "repairability": "non_repairable_under_current_pipeline",
    "include_in_v1g_lite_training": False,
    "include_in_v1g_lite_validation": False,
    "keep_for_diagnostic": True,
    "failure_stage": "insertion_contact",
    "failure_reason": "axis_misalignment / jamming",
}
DEFAULT_RESIDUAL_BREAKDOWN_JSON = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "residual_breakdown.json"
DEFAULT_ROLLOUT_VALIDATION_OUTPUT_DIR = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "rollout_validation"
DEFAULT_ROLLOUT_VALIDATION_JSON = DEFAULT_ROLLOUT_VALIDATION_OUTPUT_DIR / "rollout_validation_report.json"
DEFAULT_ROLLOUT_VALIDATION_MD = DEFAULT_ROLLOUT_VALIDATION_OUTPUT_DIR / "rollout_validation_report.md"
