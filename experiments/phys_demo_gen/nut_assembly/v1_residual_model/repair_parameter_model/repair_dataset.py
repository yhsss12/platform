"""V1-E：Repair-parameter residual field dataset utilities."""
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

import sys

_V1E_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1E_DIR.parent
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))

from residual_dataset import (  # noqa: E402
    COARSE_FAILURE_MODES,
    DEMO_KEYS,
    FAILURE_TYPES,
    OUTCOME_TYPES,
    TARGET_COMPONENT_NAMES,
    _as_bool_float,
    collapse_failure_mode,
    failure_idx,
    normalize_failure_type,
    normalize_outcome,
    outcome_idx,
)

INSERTION_PARAM_KEYS = [
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
    "gripper_close_shift",
    "speed_scale",
]

GRASP_LIFT_PARAM_KEYS = [
    "grasp_xy_offset_x",
    "grasp_xy_offset_y",
    "pre_grasp_height",
    "approach_height",
    "gripper_hold_steps",
    "lift_steps",
    "lift_speed_scale",
    "micro_lift_height",
    "reclose_after_contact",
]

ALL_THETA_KEYS = INSERTION_PARAM_KEYS + TRANSPORT_PARAM_KEYS + GRASP_LIFT_PARAM_KEYS

CONTEXT_NUMERIC_KEYS = [
    "original_final_xy",
    "original_min_xy",
    "original_final_z_diff",
    "original_min_yaw_error",
    "original_eef_nut_distance",
    "original_nut_lift_delta",
    "original_nut_displacement_after_grasp",
    "original_E_total_norm",
]

INPUT_FEATURE_NAMES = (
    [f"demo_{k.split('_')[1]}" for k in DEMO_KEYS]
    + [f"fail_{m}" for m in COARSE_FAILURE_MODES if m != "success"]
    + CONTEXT_NUMERIC_KEYS
    + ALL_THETA_KEYS
    + [f"mask_{k}" for k in ALL_THETA_KEYS]
)

TARGET_ROLLOUT_KEYS = [
    "rollout_success_flag",
    "rollout_E_total_norm",
    "rollout_E_xy_norm",
    "rollout_E_transport_norm",
    "rollout_E_yaw_norm",
    "rollout_E_z_norm",
    "rollout_grasp_success_proxy",
    "rollout_lift_success_proxy",
    "rollout_failure_type",
    "rollout_outcome",
]

DEFAULT_INSERTION_PARAMS = {k: 0.0 for k in INSERTION_PARAM_KEYS}
DEFAULT_INSERTION_PARAMS.update(
    {"z_gain": 0.55, "insertion_steps": 30.0, "hold_steps": 10.0, "insertion_speed_scale": 1.0}
)

DEFAULT_TRANSPORT_PARAMS = {k: 0.0 for k in TRANSPORT_PARAM_KEYS}
DEFAULT_TRANSPORT_PARAMS.update(
    {
        "transport_xy_gain": 1.0,
        "transport_xy_offset_scale": 1.0,
        "pre_align_height": 0.06,
        "lift_height": 0.06,
        "approach_steps": 20.0,
        "transport_steps": 40.0,
        "speed_scale": 1.0,
    }
)

DEFAULT_GRASP_LIFT_PARAMS = {k: 0.0 for k in GRASP_LIFT_PARAM_KEYS}
DEFAULT_GRASP_LIFT_PARAMS.update(
    {
        "pre_grasp_height": 0.05,
        "approach_height": 0.02,
        "gripper_hold_steps": 20.0,
        "lift_steps": 20.0,
        "lift_speed_scale": 1.0,
        "micro_lift_height": 0.06,
    }
)

COARSE_FAILURE_TO_IDX = {name: i for i, name in enumerate(COARSE_FAILURE_MODES)}
DEMO_TO_IDX = {name: i for i, name in enumerate(DEMO_KEYS)}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def infer_coarse_failure_mode(*, demo_key: str, failure_type: str | None = None) -> str:
    if failure_type:
        coarse = collapse_failure_mode(failure_type)
        if coarse != "success":
            return coarse
    mapping = {
        "demo_4": "insertion_failed",
        "demo_0": "transport_failed",
        "demo_1": "transport_failed",
        "demo_2": "grasp_failed",
        "demo_3": "transport_failed",
    }
    return mapping.get(demo_key, "transport_failed")


def extract_failed_context(original: dict[str, Any], *, demo_key: str, failure_type: str | None = None) -> dict[str, Any]:
    ft = failure_type or original.get("failure_guess") or original.get("failure_reason") or "unknown_failed"
    coarse = infer_coarse_failure_mode(demo_key=demo_key, failure_type=str(ft))
    return {
        "source_demo": demo_key,
        "source_failure_type": coarse,
        "original_final_xy": float(original.get("final_nut_peg_xy", 0.0)),
        "original_min_xy": float(original.get("min_nut_peg_xy", 0.0)),
        "original_final_z_diff": float(original.get("final_z_diff", -0.02)),
        "original_min_yaw_error": float(original.get("min_yaw_error", 0.0)),
        "original_eef_nut_distance": float(
            original.get("eef_nut_distance_at_grasp", original.get("min_eef_nut_distance", 0.1))
        ),
        "original_nut_lift_delta": float(original.get("nut_lift_delta", 0.0)),
        "original_nut_displacement_after_grasp": float(original.get("nut_displacement_after_grasp", 0.0)),
        "original_E_total_norm": float(original.get("E_total_norm", 0.0)),
    }


def parse_insertion_params(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("sim_params") or {}
    if not raw and row.get("best_params"):
        try:
            raw = json.loads(row["best_params"]) if isinstance(row["best_params"], str) else row["best_params"]
        except (json.JSONDecodeError, TypeError):
            raw = {}
    if not raw:
        for key in INSERTION_PARAM_KEYS:
            flat = row.get(f"sim_{key}")
            if flat is not None:
                raw[key] = flat
    out = {**DEFAULT_INSERTION_PARAMS}
    for key in INSERTION_PARAM_KEYS:
        if key in raw:
            out[key] = float(raw[key])
    return out


def parse_transport_params(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("transport_params") or {}
    if not raw:
        raw = {}
        for key in TRANSPORT_PARAM_KEYS:
            for prefix in (f"transport_{key}", key):
                if prefix in row and row[prefix] not in ("", None):
                    raw[key] = row[prefix]
                    break
    out = {**DEFAULT_TRANSPORT_PARAMS}
    alias = {"hold_steps": "transport_steps"}  # legacy csv may use hold_steps differently
    for key in TRANSPORT_PARAM_KEYS:
        if key in raw:
            out[key] = float(raw[key])
        elif key == "transport_steps" and "hold_steps" in raw:
            out[key] = float(raw["hold_steps"])
    return out


def parse_grasp_lift_params(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("grasp_params") or {}
    if not raw:
        mapping = {
            "grasp_xy_offset_x": "grasp_grasp_xy_offset_x",
            "grasp_xy_offset_y": "grasp_grasp_xy_offset_y",
            "pre_grasp_height": "grasp_pre_grasp_height",
            "approach_height": "grasp_approach_height",
            "gripper_hold_steps": "grasp_gripper_hold_steps",
            "lift_steps": "grasp_lift_steps",
            "lift_speed_scale": "grasp_speed_scale",
            "micro_lift_height": "grasp_lift_height",
        }
        for dst, src in mapping.items():
            if src in row and row[src] not in ("", None):
                raw[dst] = row[src]
        for key in GRASP_LIFT_PARAM_KEYS:
            prefixed = f"grasp_{key}"
            if prefixed in row and row[prefixed] not in ("", None):
                raw[key] = row[prefixed]
    out = {**DEFAULT_GRASP_LIFT_PARAMS}
    legacy = {
        "grasp_xy_offset_x": raw.get("grasp_xy_offset_x"),
        "grasp_xy_offset_y": raw.get("grasp_xy_offset_y"),
        "pre_grasp_height": raw.get("pre_grasp_height"),
        "approach_height": raw.get("approach_height"),
        "gripper_hold_steps": raw.get("gripper_hold_steps", raw.get("grasp_gripper_hold_steps")),
        "lift_steps": raw.get("lift_steps", raw.get("grasp_lift_steps")),
        "lift_speed_scale": raw.get("lift_speed_scale", raw.get("speed_scale", raw.get("grasp_speed_scale"))),
        "micro_lift_height": raw.get("micro_lift_height", raw.get("lift_height", raw.get("grasp_lift_height"))),
        "reclose_after_contact": raw.get("reclose_after_contact", raw.get("gripper_close_shift", 0.0)),
    }
    for key, val in legacy.items():
        if val is not None:
            out[key] = float(val)
    return out


def build_param_mask(*, active: str) -> np.ndarray:
    mask = np.zeros(len(ALL_THETA_KEYS), dtype=np.float32)
    if active == "insertion":
        for key in INSERTION_PARAM_KEYS:
            mask[ALL_THETA_KEYS.index(key)] = 1.0
    elif active == "transport":
        for key in TRANSPORT_PARAM_KEYS:
            mask[ALL_THETA_KEYS.index(key)] = 1.0
    elif active in ("grasp", "lift", "grasp_lift"):
        for key in GRASP_LIFT_PARAM_KEYS:
            mask[ALL_THETA_KEYS.index(key)] = 1.0
    return mask


def build_theta_vector(
    *,
    insertion: dict[str, float] | None = None,
    transport: dict[str, float] | None = None,
    grasp_lift: dict[str, float] | None = None,
) -> np.ndarray:
    merged = {
        **DEFAULT_INSERTION_PARAMS,
        **(insertion or {}),
        **DEFAULT_TRANSPORT_PARAMS,
        **(transport or {}),
        **DEFAULT_GRASP_LIFT_PARAMS,
        **(grasp_lift or {}),
    }
    return np.array([float(merged[k]) for k in ALL_THETA_KEYS], dtype=np.float32)


def build_input_vector(context: dict[str, Any], theta: np.ndarray, param_mask: np.ndarray) -> np.ndarray:
    demo_key = str(context["source_demo"])
    demo_onehot = np.zeros(len(DEMO_KEYS), dtype=np.float32)
    if demo_key in DEMO_TO_IDX:
        demo_onehot[DEMO_TO_IDX[demo_key]] = 1.0

    coarse = str(context["source_failure_type"])
    fail_onehot = np.zeros(len(COARSE_FAILURE_MODES) - 1, dtype=np.float32)
    if coarse in COARSE_FAILURE_TO_IDX and coarse != "success":
        idx = COARSE_FAILURE_TO_IDX[coarse] - 1
        if 0 <= idx < len(fail_onehot):
            fail_onehot[idx] = 1.0

    numeric = np.array([float(context[k]) for k in CONTEXT_NUMERIC_KEYS], dtype=np.float32)
    masked_theta = theta * param_mask
    return np.concatenate([demo_onehot, fail_onehot, numeric, masked_theta, param_mask.astype(np.float32)])


def extract_rollout_targets(row: dict[str, Any]) -> dict[str, Any]:
    success = _as_bool(row.get("success_flag"))
    outcome = normalize_outcome(str(row.get("outcome") or row.get("outcome_label") or ""), success_flag=success)
    failure = normalize_failure_type(
        str(row.get("failure_type") or row.get("failure_reason") or row.get("failure_guess") or "unknown_failed")
    )
    return {
        "rollout_success_flag": float(success),
        "rollout_E_total_norm": float(row.get("E_total_norm", 0.0)),
        "rollout_E_xy_norm": float(row.get("E_xy_norm", 0.0)),
        "rollout_E_transport_norm": float(row.get("E_transport_norm", 0.0)),
        "rollout_E_yaw_norm": float(row.get("E_yaw_norm", 0.0)),
        "rollout_E_z_norm": float(row.get("E_z_norm", 0.0)),
        "rollout_grasp_success_proxy": _as_bool_float(row.get("grasp_success_proxy", False)),
        "rollout_lift_success_proxy": _as_bool_float(row.get("lift_success_proxy", False)),
        "rollout_failure_type": failure,
        "rollout_outcome": outcome,
        "rollout_failure_type_idx": failure_idx(failure),
        "rollout_outcome_idx": outcome_idx(outcome),
        "refined_success_flag": float(outcome == "refined_success" or success),
        "E_smooth_norm": float(row.get("E_smooth_norm", 0.0)),
    }


def make_sample(
    *,
    context: dict[str, Any],
    theta: np.ndarray,
    param_mask: np.ndarray,
    targets: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    targets_components = np.array(
        [
            targets["rollout_E_xy_norm"],
            targets["rollout_E_transport_norm"],
            targets["rollout_E_yaw_norm"],
            targets["rollout_E_z_norm"],
            float(targets.get("E_smooth_norm", 0.0)),
        ],
        dtype=np.float32,
    )

    return {
        "features": build_input_vector(context, theta, param_mask),
        "theta": theta.astype(np.float32),
        "param_mask": param_mask.astype(np.float32),
        "context": context,
        "targets_components": targets_components,
        "target_E_total": float(targets["rollout_E_total_norm"]),
        "success_flag": float(targets["rollout_success_flag"]),
        "failure_type_idx": int(targets["rollout_failure_type_idx"]),
        "outcome_idx": int(targets["rollout_outcome_idx"]),
        "grasp_success_flag": float(targets["rollout_grasp_success_proxy"]),
        "lift_success_flag": float(targets["rollout_lift_success_proxy"]),
        "refined_success_flag": float(targets["refined_success_flag"]),
        "meta": meta,
    }


def load_repair_npz(path: Path | str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    summary = json.loads(str(data["meta_json"]))
    bundle = {k: data[k] for k in data.files if k != "meta_json"}
    bundle["meta"] = summary
    return bundle


class RepairParameterDataset(Dataset):
    def __init__(self, npz_path: Path | str, indices: np.ndarray | None = None):
        if torch is None:
            raise ImportError("torch required for RepairParameterDataset")
        bundle = load_repair_npz(npz_path)
        self.features = bundle["features"]
        idx = indices if indices is not None else np.arange(len(self.features))
        self.indices = idx
        self.targets_components = bundle["targets_components"][idx]
        self.target_E_total = bundle["target_E_total"][idx]
        self.success_flag = bundle["success_flag"][idx]
        self.failure_type_idx = bundle["failure_type_idx"][idx]
        self.outcome_idx = bundle["outcome_idx"][idx]
        self.grasp_success_flag = bundle["grasp_success_flag"][idx]
        self.lift_success_flag = bundle["lift_success_flag"][idx]
        self.refined_success_flag = bundle["refined_success_flag"][idx]
        self.param_mask = bundle["param_mask"][idx]
        self.original_E_total = bundle["original_E_total"][idx]
        self.source_failure_mode_idx = bundle["source_failure_mode_idx"][idx]
        self.features = self.features[idx]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[i]),
            "targets_components": torch.from_numpy(self.targets_components[i]),
            "target_E_total": torch.tensor(self.target_E_total[i], dtype=torch.float32),
            "success_flag": torch.tensor(self.success_flag[i], dtype=torch.float32),
            "failure_type_idx": torch.tensor(self.failure_type_idx[i], dtype=torch.long),
            "outcome_idx": torch.tensor(self.outcome_idx[i], dtype=torch.long),
            "grasp_success_flag": torch.tensor(self.grasp_success_flag[i], dtype=torch.float32),
            "lift_success_flag": torch.tensor(self.lift_success_flag[i], dtype=torch.float32),
            "refined_success_flag": torch.tensor(self.refined_success_flag[i], dtype=torch.float32),
            "param_mask": torch.from_numpy(self.param_mask[i]),
            "original_E_total": torch.tensor(self.original_E_total[i], dtype=torch.float32),
            "source_failure_mode_idx": torch.tensor(self.source_failure_mode_idx[i], dtype=torch.long),
        }
