"""V1-A / V1-B / V1-C：PyTorch Dataset for residual energy model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore

FAILURE_TYPES = [
    "success",
    "transport_failed",
    "alignment_failed",
    "insertion_failed",
    "grasp_failed",
    "lift_failed",
    "smoothness_issue",
    "unknown_failed",
]
FAILURE_TYPE_TO_IDX = {name: i for i, name in enumerate(FAILURE_TYPES)}

OUTCOME_TYPES = [
    "success",
    "refined_success",
    "improved_but_failed",
    "grasp_improved_but_failed",
    "no_improvement",
    "grasp_no_improvement",
    "baseline",
    "search_candidate",
    "candidate_ready",
    "failed",
    "unknown_outcome",
]
OUTCOME_TO_IDX = {name: i for i, name in enumerate(OUTCOME_TYPES)}

DEMO_KEYS = ["demo_0", "demo_1", "demo_2", "demo_3", "demo_4"]
COARSE_FAILURE_MODES = [
    "success",
    "insertion_failed",
    "transport_failed",
    "grasp_failed",
    "lift_failed",
]

BASE_FEATURE_NAMES = [
    "final_nut_peg_xy",
    "min_nut_peg_xy",
    "final_z_diff",
    "min_yaw_error",
    "final_yaw_error",
    "action_accel_mean",
    "action_accel_max",
    "eef_velocity_mean",
    "gripper_closed_fraction",
    "grasp_index_norm",
]

SIM_PARAM_KEYS = [
    "insert_z_offset",
    "z_gain",
    "insertion_steps",
    "hold_steps",
    "insertion_speed_scale",
    "release_shift",
    "pre_insert_pause",
]

TRANSPORT_PARAM_KEYS = [
    "transport_xy_gain",
    "transport_xy_offset_scale",
    "pre_align_height",
    "lift_height",
    "approach_steps",
    "transport_steps",
    "transport_hold_steps",
    "gripper_close_shift",
    "speed_scale",
]

DIAGNOSTIC_FEATURE_NAMES = [
    "nut_displacement_total",
    "nut_displacement_after_grasp",
    "eef_nut_distance_at_grasp",
    "min_eef_nut_distance",
    "grasp_phase_duration",
    "transport_improvement_ratio",
    "xy_improvement_ratio",
]

GRASP_PARAM_KEYS = [
    "grasp_xy_offset_x",
    "grasp_xy_offset_y",
    "pre_grasp_height",
    "approach_height",
    "grasp_gripper_close_shift",
    "grasp_gripper_hold_steps",
    "grasp_lift_height",
    "grasp_lift_steps",
    "grasp_speed_scale",
]

GRASP_EXTRA_FEATURE_NAMES = [
    "nut_lift_delta",
    "grasp_success_proxy_feat",
    "lift_success_proxy_feat",
]

FEATURE_NAMES_V1A = BASE_FEATURE_NAMES + SIM_PARAM_KEYS
FEATURE_NAMES_V1B = (
    BASE_FEATURE_NAMES + SIM_PARAM_KEYS + TRANSPORT_PARAM_KEYS + DIAGNOSTIC_FEATURE_NAMES
)
FEATURE_NAMES_V1C = FEATURE_NAMES_V1B + GRASP_PARAM_KEYS + GRASP_EXTRA_FEATURE_NAMES
FEATURE_NAMES = FEATURE_NAMES_V1C

TARGET_COMPONENT_NAMES = [
    "E_xy_norm",
    "E_transport_norm",
    "E_yaw_norm",
    "E_z_norm",
    "E_smooth_norm",
]

DEFAULT_SIM_PARAMS = {
    "insert_z_offset": 0.0,
    "z_gain": 0.55,
    "insertion_steps": 30.0,
    "hold_steps": 10.0,
    "insertion_speed_scale": 1.0,
    "release_shift": 0.0,
    "pre_insert_pause": 0.0,
}

DEFAULT_TRANSPORT_PARAMS = {
    "transport_xy_gain": 1.0,
    "transport_xy_offset_scale": 1.0,
    "pre_align_height": 0.06,
    "lift_height": 0.06,
    "approach_steps": 20.0,
    "transport_steps": 40.0,
    "transport_hold_steps": 10.0,
    "gripper_close_shift": 0.0,
    "speed_scale": 1.0,
}

DEFAULT_GRASP_PARAMS = {
    "grasp_xy_offset_x": 0.0,
    "grasp_xy_offset_y": 0.0,
    "pre_grasp_height": 0.05,
    "approach_height": 0.02,
    "grasp_gripper_close_shift": 0.0,
    "grasp_gripper_hold_steps": 20.0,
    "grasp_lift_height": 0.06,
    "grasp_lift_steps": 20.0,
    "grasp_speed_scale": 1.0,
}

DEFAULT_GRASP_EXTRA = {
    "nut_lift_delta": 0.0,
    "grasp_success_proxy_feat": 0.0,
    "lift_success_proxy_feat": 0.0,
}


def _as_bool_float(value: Any) -> float:
    if isinstance(value, str):
        return float(value.lower() in ("true", "1", "yes"))
    return float(bool(value))


def normalize_failure_type(name: str) -> str:
    mapping = {
        "success": "success",
        "transport_failed": "transport_failed",
        "alignment_failed": "alignment_failed",
        "insertion_failed": "insertion_failed",
        "grasp_failed": "grasp_failed",
        "lift_failed": "lift_failed",
        "smoothness_issue": "smoothness_issue",
        "unknown_failed": "unknown_failed",
        "candidate_ready": "unknown_failed",
        "near_success_but_not_task_success": "alignment_failed",
        "nut_not_picked": "grasp_failed",
        "misaligned_grasp": "grasp_failed",
        "grasp_improved_transport_blocked": "transport_failed",
        "transport_not_started": "transport_failed",
        "transport_not_enough": "transport_failed",
        "grasp_partial_improvement": "grasp_failed",
    }
    return mapping.get(name, "unknown_failed")


def collapse_failure_mode(name: str) -> str:
    """Map fine-grained failure_type to coarse source_failure_mode for group splits."""
    ft = normalize_failure_type(name)
    if ft == "success":
        return "success"
    if ft in ("insertion_failed", "smoothness_issue"):
        return "insertion_failed"
    if ft in ("transport_failed", "alignment_failed", "unknown_failed"):
        return "transport_failed"
    if ft == "grasp_failed":
        return "grasp_failed"
    if ft == "lift_failed":
        return "lift_failed"
    return "transport_failed"


def enrich_meta_record(record: dict[str, Any]) -> dict[str, Any]:
    demo_key = str(record.get("source_demo") or record.get("demo_key") or "")
    failure_type = str(record.get("failure_type") or "unknown_failed")
    return {
        **record,
        "source_demo": demo_key,
        "source_failure_mode": str(record.get("source_failure_mode") or collapse_failure_mode(failure_type)),
        "sample_source": str(record.get("sample_source") or record.get("source") or ""),
    }


def normalize_outcome(name: str | None, *, success_flag: bool = False) -> str:
    if not name:
        return "refined_success" if success_flag else "unknown_outcome"
    mapping = {
        "success": "success",
        "refined_success": "refined_success",
        "improved_but_failed": "improved_but_failed",
        "grasp_improved_but_failed": "grasp_improved_but_failed",
        "no_improvement": "no_improvement",
        "grasp_no_improvement": "grasp_no_improvement",
        "baseline": "baseline",
        "search_candidate": "search_candidate",
        "candidate_ready": "candidate_ready",
        "failed": "failed",
    }
    if name in mapping:
        return mapping[name]
    if success_flag:
        return "refined_success"
    return "unknown_outcome"


def failure_idx(name: str) -> int:
    return FAILURE_TYPE_TO_IDX.get(normalize_failure_type(name), FAILURE_TYPE_TO_IDX["unknown_failed"])


def outcome_idx(name: str) -> int:
    return OUTCOME_TO_IDX.get(name, OUTCOME_TO_IDX["unknown_outcome"])


def build_feature_vector(
    *,
    dataset_version: str,
    base: dict[str, float],
    sim_params: dict[str, float] | None = None,
    transport_params: dict[str, float] | None = None,
    diagnostic: dict[str, float] | None = None,
    grasp_params: dict[str, float] | None = None,
    grasp_extra: dict[str, float] | None = None,
) -> np.ndarray:
    sim = {**DEFAULT_SIM_PARAMS, **(sim_params or {})}
    transport = {**DEFAULT_TRANSPORT_PARAMS, **(transport_params or {})}
    if "transport_hold_steps" not in (transport_params or {}) and "hold_steps" in (transport_params or {}):
        transport["transport_hold_steps"] = float(transport_params["hold_steps"])
    diag = {
        "nut_displacement_total": 0.0,
        "nut_displacement_after_grasp": 0.0,
        "eef_nut_distance_at_grasp": 0.0,
        "min_eef_nut_distance": 0.0,
        "grasp_phase_duration": 0.0,
        "transport_improvement_ratio": 0.0,
        "xy_improvement_ratio": 0.0,
        **(diagnostic or {}),
    }
    grasp = {**DEFAULT_GRASP_PARAMS, **(grasp_params or {})}
    extra = {**DEFAULT_GRASP_EXTRA, **(grasp_extra or {})}

    values = [float(base[name]) for name in BASE_FEATURE_NAMES]
    values += [float(sim[k]) for k in SIM_PARAM_KEYS]
    if dataset_version in ("v1b", "v1c"):
        values += [float(transport[k]) for k in TRANSPORT_PARAM_KEYS]
        values += [float(diag[k]) for k in DIAGNOSTIC_FEATURE_NAMES]
    if dataset_version == "v1c":
        values += [float(grasp[k]) for k in GRASP_PARAM_KEYS]
        values += [float(extra[k]) for k in GRASP_EXTRA_FEATURE_NAMES]
    return np.array(values, dtype=np.float32)


def load_npz_dataset(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta_json"]))
    bundle: dict[str, Any] = {
        "features": data["features"].astype(np.float32),
        "targets_components": data["targets_components"].astype(np.float32),
        "target_E_total": data["target_E_total"].astype(np.float32),
        "success_flag": data["success_flag"].astype(np.float32),
        "failure_type_idx": data["failure_type_idx"].astype(np.int64),
        "meta": meta,
    }
    optional_keys = (
        "outcome_idx",
        "level_1_pass",
        "level_2_pass",
        "level_3_pass",
        "improvement_ratio",
        "refined_success_flag",
        "grasp_success_flag",
        "lift_success_flag",
        "nut_lift_delta",
        "nut_displacement_after_grasp",
    )
    for key in optional_keys:
        if key in data:
            bundle[key] = data[key]
    return bundle


class ResidualEnergyDataset(Dataset):
    def __init__(self, npz_path: str | Path, indices: np.ndarray | None = None):
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        bundle = load_npz_dataset(npz_path)
        self.features = torch.from_numpy(bundle["features"])
        self.targets_components = torch.from_numpy(bundle["targets_components"])
        self.target_E_total = torch.from_numpy(bundle["target_E_total"])
        self.success_flag = torch.from_numpy(bundle["success_flag"])
        self.failure_type_idx = torch.from_numpy(bundle["failure_type_idx"])
        self.outcome_idx = torch.from_numpy(
            bundle.get("outcome_idx", np.zeros(len(self.features), dtype=np.int64))
        )
        self.grasp_success_flag = torch.from_numpy(
            bundle.get("grasp_success_flag", np.zeros(len(self.features), dtype=np.float32))
        )
        self.lift_success_flag = torch.from_numpy(
            bundle.get("lift_success_flag", np.zeros(len(self.features), dtype=np.float32))
        )
        self.meta = bundle["meta"]
        self.indices = np.arange(len(self.features)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        return {
            "features": self.features[i],
            "targets_components": self.targets_components[i],
            "target_E_total": self.target_E_total[i],
            "success_flag": self.success_flag[i],
            "failure_type_idx": self.failure_type_idx[i],
            "outcome_idx": self.outcome_idx[i],
            "grasp_success_flag": self.grasp_success_flag[i],
            "lift_success_flag": self.lift_success_flag[i],
        }
