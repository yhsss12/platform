"""V1-F：Uncertainty-aware PINN Repair Parameter Model dataset utilities."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_V1E_DIR = _V1_DIR / "repair_parameter_model"
for path in (_V1_DIR, _V1E_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from residual_dataset import (  # noqa: E402
    COARSE_FAILURE_MODES,
    DEMO_KEYS,
    FAILURE_TYPES,
    OUTCOME_TYPES,
    _as_bool_float,
    failure_idx,
    normalize_failure_type,
    normalize_outcome,
    outcome_idx,
)
from repair_dataset import (  # noqa: E402
    CONTEXT_NUMERIC_KEYS,
    DEFAULT_GRASP_LIFT_PARAMS,
    DEFAULT_INSERTION_PARAMS,
    DEFAULT_TRANSPORT_PARAMS,
    GRASP_LIFT_PARAM_KEYS,
    INSERTION_PARAM_KEYS,
    TRANSPORT_PARAM_KEYS,
    extract_failed_context,
    infer_coarse_failure_mode,
)

# V1-F 新增 lift-aware 参数（micro_lift_height / lift_speed_scale 已在 grasp_lift 中，此处补全独立键）
LIFT_EXTRA_PARAM_KEYS = [
    "micro_lift_steps",
    "regrasp_shift",
    "gripper_extra_close",
    "lift_pause_steps",
    "contact_hold_steps",
    "post_grasp_settle_steps",
    "lift_direction_bias",
    "nut_follow_threshold",
]

ALL_THETA_KEYS_V1F = INSERTION_PARAM_KEYS + TRANSPORT_PARAM_KEYS + GRASP_LIFT_PARAM_KEYS + LIFT_EXTRA_PARAM_KEYS

# PINN 主输出 7 分量
V1F_COMPONENT_NAMES = [
    "E_xy",
    "E_transport",
    "E_yaw",
    "E_z",
    "E_grasp",
    "E_lift",
    "E_smooth",
]

# lift 专项监督残差
LIFT_RESIDUAL_NAMES = [
    "E_lift_follow",
    "E_grasp_contact",
    "E_object_displacement",
    "E_eef_nut_coupling",
    "E_lift_stability",
]

INPUT_FEATURE_NAMES_V1F = (
    [f"demo_{k.split('_')[1]}" for k in DEMO_KEYS]
    + [f"fail_{m}" for m in COARSE_FAILURE_MODES if m != "success"]
    + CONTEXT_NUMERIC_KEYS
    + ALL_THETA_KEYS_V1F
    + [f"mask_{k}" for k in ALL_THETA_KEYS_V1F]
)

DEFAULT_LIFT_EXTRA_PARAMS = {k: 0.0 for k in LIFT_EXTRA_PARAM_KEYS}
DEFAULT_LIFT_EXTRA_PARAMS.update(
    {
        "micro_lift_steps": 20.0,
        "contact_hold_steps": 20.0,
        "post_grasp_settle_steps": 5.0,
        "nut_follow_threshold": 0.05,
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


def parse_lift_extra_params(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("lift_params") or row.get("lift_extra_params") or {}
    if not raw and row.get("grasp_params"):
        raw = row["grasp_params"]
    out = {**DEFAULT_LIFT_EXTRA_PARAMS}
    for key in LIFT_EXTRA_PARAM_KEYS:
        for src in (key, f"lift_{key}", f"grasp_{key}"):
            if src in raw and raw[src] not in ("", None):
                out[key] = float(raw[src])
                break
        if key in row and row[key] not in ("", None):
            out[key] = float(row[key])
    return out


def build_param_mask_v1f(*, active: str) -> np.ndarray:
    mask = np.zeros(len(ALL_THETA_KEYS_V1F), dtype=np.float32)
    groups: dict[str, list[str]] = {
        "insertion": INSERTION_PARAM_KEYS,
        "transport": TRANSPORT_PARAM_KEYS,
        "grasp": GRASP_LIFT_PARAM_KEYS,
        "lift": GRASP_LIFT_PARAM_KEYS + LIFT_EXTRA_PARAM_KEYS,
        "grasp_lift": GRASP_LIFT_PARAM_KEYS,
    }
    for key in groups.get(active, []):
        if key in ALL_THETA_KEYS_V1F:
            mask[ALL_THETA_KEYS_V1F.index(key)] = 1.0
    return mask


def build_theta_vector_v1f(
    *,
    insertion: dict[str, float] | None = None,
    transport: dict[str, float] | None = None,
    grasp_lift: dict[str, float] | None = None,
    lift_extra: dict[str, float] | None = None,
) -> np.ndarray:
    merged = {
        **DEFAULT_INSERTION_PARAMS,
        **(insertion or {}),
        **DEFAULT_TRANSPORT_PARAMS,
        **(transport or {}),
        **DEFAULT_GRASP_LIFT_PARAMS,
        **(grasp_lift or {}),
        **DEFAULT_LIFT_EXTRA_PARAMS,
        **(lift_extra or {}),
    }
    return np.array([float(merged[k]) for k in ALL_THETA_KEYS_V1F], dtype=np.float32)


def build_input_vector_v1f(context: dict[str, Any], theta: np.ndarray, param_mask: np.ndarray) -> np.ndarray:
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


def extract_rollout_targets_v1f(row: dict[str, Any]) -> dict[str, Any]:
    success = _as_bool(row.get("success_flag"))
    outcome = normalize_outcome(str(row.get("outcome") or row.get("outcome_label") or ""), success_flag=success)
    failure = normalize_failure_type(
        str(row.get("failure_type") or row.get("failure_reason") or row.get("failure_guess") or "unknown_failed")
    )
    components = np.array(
        [
            float(row.get("rollout_E_xy_norm", row.get("E_xy_norm", 0.0))),
            float(row.get("rollout_E_transport_norm", row.get("E_transport_norm", 0.0))),
            float(row.get("rollout_E_yaw_norm", row.get("E_yaw_norm", 0.0))),
            float(row.get("rollout_E_z_norm", row.get("E_z_norm", 0.0))),
            float(row.get("rollout_E_grasp_norm", row.get("E_grasp_norm", 0.0))),
            float(row.get("rollout_E_lift_norm", row.get("E_lift_norm", 0.0))),
            float(row.get("rollout_E_smooth_norm", row.get("E_smooth_norm", 0.0))),
        ],
        dtype=np.float32,
    )
    lift_residuals = np.array(
        [
            float(row.get("E_lift_follow", row.get("rollout_E_lift_follow", 0.0))),
            float(row.get("E_grasp_contact", row.get("rollout_E_grasp_contact", 0.0))),
            float(row.get("E_object_displacement", row.get("rollout_E_object_displacement", 0.0))),
            float(row.get("E_eef_nut_coupling", row.get("rollout_E_eef_nut_coupling", 0.0))),
            float(row.get("E_lift_stability", row.get("rollout_E_lift_stability", 0.0))),
        ],
        dtype=np.float32,
    )
    return {
        "targets_components": components,
        "lift_residuals": lift_residuals,
        "rollout_success_flag": float(success),
        "rollout_E_total_norm": float(row.get("rollout_E_total_norm", row.get("E_total_norm", 0.0))),
        "rollout_grasp_success_proxy": _as_bool_float(row.get("grasp_success_proxy", False)),
        "rollout_lift_success_proxy": _as_bool_float(row.get("lift_success_proxy", False)),
        "rollout_failure_type": failure,
        "rollout_outcome": outcome,
        "rollout_failure_type_idx": failure_idx(failure),
        "rollout_outcome_idx": outcome_idx(outcome),
        "refined_success_flag": float(outcome == "refined_success" or success),
    }


def make_sample_v1f(
    *,
    context: dict[str, Any],
    theta: np.ndarray,
    param_mask: np.ndarray,
    targets: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "features": build_input_vector_v1f(context, theta, param_mask),
        "theta": theta.astype(np.float32),
        "param_mask": param_mask.astype(np.float32),
        "context": context,
        "targets_components": targets["targets_components"],
        "lift_residuals": targets["lift_residuals"],
        "target_E_total": float(targets["rollout_E_total_norm"]),
        "success_flag": float(targets["rollout_success_flag"]),
        "failure_type_idx": int(targets["rollout_failure_type_idx"]),
        "outcome_idx": int(targets["rollout_outcome_idx"]),
        "grasp_success_flag": float(targets["rollout_grasp_success_proxy"]),
        "lift_success_flag": float(targets["rollout_lift_success_proxy"]),
        "refined_success_flag": float(targets["refined_success_flag"]),
        "meta": meta,
    }


def load_v1f_npz(path: Path | str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    summary = json.loads(str(data["meta_json"]))
    bundle = {k: data[k] for k in data.files if k != "meta_json"}
    bundle["meta"] = summary
    return bundle


class V1FRepairDataset(Dataset):
    def __init__(self, npz_path: Path | str, indices: np.ndarray | None = None):
        if torch is None:
            raise ImportError("torch required for V1FRepairDataset")
        bundle = load_v1f_npz(npz_path)
        self.features = bundle["features"]
        idx = indices if indices is not None else np.arange(len(self.features))
        self.indices = idx
        self.targets_components = bundle["targets_components"][idx]
        self.lift_residuals = bundle["lift_residuals"][idx]
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
        self.demo_idx = bundle["demo_idx"][idx]
        self.features = self.features[idx]
        full_n = len(bundle["features"])
        if "sample_weight" in bundle:
            self.sample_weight = bundle["sample_weight"][idx]
        else:
            self.sample_weight = np.ones(len(idx), dtype=np.float32)
        if "ranking_supervision_eligible" in bundle:
            self.ranking_supervision_eligible = bundle["ranking_supervision_eligible"][idx]
        else:
            self.ranking_supervision_eligible = np.ones(len(idx), dtype=np.float32)
        if "demo_group_id" in bundle:
            self.demo_group_id = bundle["demo_group_id"][idx]
        else:
            self.demo_group_id = np.full(len(idx), -1, dtype=np.int64)
        if "old_demo_retention" in bundle:
            self.old_demo_retention = bundle["old_demo_retention"][idx]
        else:
            self.old_demo_retention = np.zeros(len(idx), dtype=np.float32)
        if "is_success_reference" in bundle:
            self.is_success_reference = bundle["is_success_reference"][idx]
        else:
            self.is_success_reference = np.zeros(len(idx), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[i]),
            "targets_components": torch.from_numpy(self.targets_components[i]),
            "lift_residuals": torch.from_numpy(self.lift_residuals[i]),
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
            "demo_idx": torch.tensor(self.demo_idx[i], dtype=torch.long),
            "sample_weight": torch.tensor(self.sample_weight[i], dtype=torch.float32),
            "ranking_supervision_eligible": torch.tensor(
                self.ranking_supervision_eligible[i], dtype=torch.float32
            ),
            "demo_group_id": torch.tensor(self.demo_group_id[i], dtype=torch.long),
            "old_demo_retention": torch.tensor(self.old_demo_retention[i], dtype=torch.float32),
            "is_success_reference": torch.tensor(self.is_success_reference[i], dtype=torch.float32),
        }


# Re-export for rollout sampling
__all__ = [
    "ALL_THETA_KEYS_V1F",
    "V1F_COMPONENT_NAMES",
    "LIFT_RESIDUAL_NAMES",
    "INPUT_FEATURE_NAMES_V1F",
    "extract_failed_context",
    "infer_coarse_failure_mode",
    "parse_lift_extra_params",
    "build_param_mask_v1f",
    "build_theta_vector_v1f",
    "extract_rollout_targets_v1f",
    "make_sample_v1f",
    "load_v1f_npz",
    "V1FRepairDataset",
]
