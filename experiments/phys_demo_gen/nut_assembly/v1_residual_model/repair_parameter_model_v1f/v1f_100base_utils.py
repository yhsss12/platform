"""V1-F-100Base：Square_D0 100 条数据 + aligned-original 基座。"""
from __future__ import annotations

from pathlib import Path

from v1f_plus_utils import (  # noqa: E402
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_ALIGNED_NPZ,
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_SUCCESS_HDF5,
    audit_category_to_coarse,
    list_demo_keys,
    load_failure_map,
    load_theta_or_default,
    sampler_plan_for_failure,
    search_kind_for_failure,
)

_V1F_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1F_DIR.parent.parent

DEFAULT_100BASE_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_100base"
DEFAULT_AUDIT_DIR = DEFAULT_100BASE_OUTPUT / "audit"
DEFAULT_AUDIT_REPORT = DEFAULT_AUDIT_DIR / "new_demo_audit_report.json"
DEFAULT_FAILED_CONTEXTS = DEFAULT_100BASE_OUTPUT / "failed_contexts.jsonl"
DEFAULT_FAILED_ROLLOUT = DEFAULT_100BASE_OUTPUT / "failed_rollout_samples.jsonl"
DEFAULT_TARGETED_ROLLOUT = DEFAULT_100BASE_OUTPUT / "targeted_rollout_samples.jsonl"
DEFAULT_SUCCESS_REFERENCE = DEFAULT_100BASE_OUTPUT / "success_reference_samples.jsonl"
DEFAULT_REPAIRABILITY = DEFAULT_100BASE_OUTPUT / "repairability_audit" / "repairability_report.json"
DEFAULT_DATASET_NPZ = DEFAULT_100BASE_OUTPUT / "repair_parameter_dataset_v1f_100base.npz"
DEFAULT_TRAINED_MODEL = DEFAULT_100BASE_OUTPUT / "trained_model" / "model_v1f_100base.pt"
DEFAULT_EVAL_REPORT = DEFAULT_100BASE_OUTPUT / "evaluation" / "quick_validation_report.json"
DEFAULT_SANITY_REPORT = DEFAULT_100BASE_OUTPUT / "pretrain_sanity_report.json"
DEFAULT_SANITY_SUMMARY = DEFAULT_100BASE_OUTPUT / "pretrain_sanity_summary.md"

OLD_DEMO_KEYS = ("demo_0", "demo_1", "demo_2", "demo_3", "demo_4")
OLD_STABLE_SOURCE_PREFIXES = ("aligned_original", "v1e_import:", "v1f_rollout_sampling")
NEW_FAILED_SOURCE_PREFIXES = ("v1f_100base_failed", "v1f_100base_targeted", "v1f_plus_rollout")
SUCCESS_REFERENCE_SOURCES = ("v1f_100base_success_reference", "v1f_100base_success_perturb")
V2_FORBIDDEN_TOKENS = (
    "balanced_v2",
    "aligned-plus-balanced-v2",
    "model_v1f_aligned_plus_balanced_v2",
    "deprecated",
)
ALIGNED_ORIGINAL_JSONL = (
    _EXPERIMENT_DIR
    / "outputs"
    / "v1f_aligned_repair_parameter_model"
    / "original_failed"
    / "repair_parameter_dataset_v1f.jsonl"
)

REPAIRABILITY_BUDGET = {
    "repairable": (160, 220),
    "hard_but_improvable": (120, 170),
    "no_positive_candidate": (80, 120),
    "default": (100, 150),
}

TARGETED_EXTRA_BUDGET = {
    "repairable": 80,
    "hard_but_improvable": 60,
}
