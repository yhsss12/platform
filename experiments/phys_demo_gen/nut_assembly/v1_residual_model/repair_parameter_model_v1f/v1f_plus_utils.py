"""V1-F-aligned-plus：Square_D0 新数据 pipeline 共享工具。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import h5py

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sim_in_loop_refiner import load_best_theta  # noqa: E402

DEFAULT_SUCCESS_HDF5 = _EXPERIMENT_DIR.parents[2] / "demo(1).hdf5"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "demo_failed(1).hdf5"
DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_ALIGNED_NPZ = (
    _EXPERIMENT_DIR
    / "outputs"
    / "v1f_aligned_repair_parameter_model"
    / "original_failed"
    / "repair_parameter_dataset_v1f.npz"
)
DEFAULT_ALIGNED_MODEL = (
    _EXPERIMENT_DIR
    / "outputs"
    / "v1f_aligned_repair_parameter_model"
    / "original_failed"
    / "trained_model"
    / "model_v1f_aligned_original.pt"
)
DEFAULT_PLUS_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus"

NEUTRAL_TRANSPORT_THETA: dict[str, Any] = {
    "transport_xy_offset": [0.0, 0.0],
    "pre_align_xy_offset": [0.0, 0.0],
    "align_yaw_offset": 0.0,
    "insert_z_offset": 0.0,
    "speed_scale": 1.0,
    "gripper_close_shift": 0.0,
    "release_shift": 0.0,
}


def list_demo_keys(hdf5_path: str | Path) -> list[str]:
    with h5py.File(hdf5_path, "r") as handle:
        return sorted(handle["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def audit_category_to_coarse(category: str) -> str:
    """将 residual audit 分类映射到 PINN coarse failure mode。"""
    mapping = {
        "transport_failed": "transport_failed",
        "insertion_failed": "insertion_failed",
        "alignment_failed": "transport_failed",
        "smoothness_issue": "insertion_failed",
        "grasp_failed": "grasp_failed",
        "lift_failed": "lift_failed",
        "success": "success",
    }
    return mapping.get(category, "transport_failed")


def sampler_plan_for_failure(category: str) -> str:
    """transport / insertion / mixed。"""
    if category == "transport_failed":
        return "transport"
    if category == "insertion_failed":
        return "insertion"
    return "mixed"


def search_kind_for_failure(coarse: str) -> str:
    mapping = {
        "transport_failed": "transport",
        "insertion_failed": "insertion",
        "grasp_failed": "grasp",
        "lift_failed": "lift",
    }
    return mapping.get(coarse, "insertion")


def load_theta_or_default(
    cem_report: str | Path,
    demo_key: str,
    *,
    search_kind: str,
) -> dict[str, Any]:
    try:
        return load_best_theta(str(cem_report), demo_key)
    except KeyError:
        if search_kind == "insertion":
            return load_best_theta(str(cem_report), "demo_4")
        if search_kind == "transport":
            try:
                return load_best_theta(str(cem_report), "demo_0")
            except KeyError:
                return dict(NEUTRAL_TRANSPORT_THETA)
        return dict(NEUTRAL_TRANSPORT_THETA)


def load_failure_map(audit_report_path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in report.get("per_demo_residuals", []):
        if row.get("file_label") != "failed":
            continue
        category = str(row.get("failure_category", "alignment_failed"))
        coarse = audit_category_to_coarse(category)
        out[row["demo_key"]] = {
            "rough_failure_type": category,
            "coarse_failure_type": coarse,
            "sampler": sampler_plan_for_failure(category),
            "search_kind": search_kind_for_failure(coarse),
            "final_nut_peg_xy": row.get("final_nut_peg_xy_distance"),
            "min_nut_peg_xy": row.get("min_nut_peg_xy_distance"),
            "final_z_diff": row.get("final_nut_peg_z_difference"),
            "grasp_signal_index": row.get("grasp_signal_index"),
        }
    if out:
        return out
    for demo_key, category in report.get("failed_demo_classification", {}).items():
        coarse = audit_category_to_coarse(str(category))
        out[str(demo_key)] = {
            "rough_failure_type": str(category),
            "coarse_failure_type": coarse,
            "sampler": sampler_plan_for_failure(str(category)),
            "search_kind": search_kind_for_failure(coarse),
        }
    return out
