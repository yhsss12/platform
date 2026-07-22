"""V1-A / V1-B / V1-C：从 V0/V0.5 energy + V2-B sim rollout 构建训练数据集。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from energy_model import compute_total_energy
from extract_features import (
    NutAssemblyFeatures,
    action_acceleration_stats,
    extract_demo_features,
    grasp_signal_index,
    load_features_from_hdf5,
)
from residual_dataset import (
    BASE_FEATURE_NAMES,
    DEFAULT_GRASP_EXTRA,
    DEFAULT_GRASP_PARAMS,
    DEFAULT_SIM_PARAMS,
    DEFAULT_TRANSPORT_PARAMS,
    DIAGNOSTIC_FEATURE_NAMES,
    FEATURE_NAMES_V1A,
    FEATURE_NAMES_V1B,
    FEATURE_NAMES_V1C,
    GRASP_PARAM_KEYS,
    SIM_PARAM_KEYS,
    TARGET_COMPONENT_NAMES,
    TRANSPORT_PARAM_KEYS,
    _as_bool_float,
    build_feature_vector,
    failure_idx,
    normalize_failure_type,
    normalize_outcome,
    outcome_idx,
    collapse_failure_mode,
)
from trajectory_parameterization import load_trajectory_proxy

DEFAULT_OUTPUT_V1A = _EXPERIMENT_DIR / "outputs" / "v1_residual_model"
DEFAULT_OUTPUT_V1B = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1b"
DEFAULT_OUTPUT_V1C = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c"
DEFAULT_SUCCESS_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo.hdf5"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"


def _extended_hdf5_features(hdf5_path: str, demo_key: str) -> dict[str, float]:
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle[f"data/{demo_key}"]
        actions = demo["actions"][:]
        eef = demo["datagen_info/eef_pose"][:]
        gripper = demo["datagen_info/gripper_action"][:].reshape(-1)
        grasp = demo["datagen_info/subtask_term_signals/grasp"][:]
        length = len(actions)

    acc_mean, acc_max = action_acceleration_stats(actions)
    eef_pos = eef[:, :3, 3]
    if len(eef_pos) > 1:
        eef_vel = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)
        eef_velocity_mean = float(np.mean(eef_vel))
    else:
        eef_velocity_mean = 0.0
    gripper_closed_fraction = float(np.mean(gripper < 0.0))
    grasp_idx = grasp_signal_index(grasp)
    grasp_index_norm = float(grasp_idx / max(1, length - 1)) if grasp_idx is not None else 0.0

    return {
        "action_accel_mean": acc_mean,
        "action_accel_max": acc_max,
        "eef_velocity_mean": eef_velocity_mean,
        "gripper_closed_fraction": gripper_closed_fraction,
        "grasp_index_norm": grasp_index_norm,
        "length": float(length),
    }


def _diagnostic_features(
    hdf5_path: str,
    demo_key: str,
    label: str,
    *,
    baseline_min_xy: float | None = None,
    baseline_final_xy: float | None = None,
    current_min_xy: float | None = None,
    current_final_xy: float | None = None,
) -> dict[str, float]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    with h5py.File(hdf5_path, "r") as handle:
        eef = handle[f"data/{demo_key}/datagen_info/eef_pose"][:, :3, 3]

    nut_pos = proxy.nut_pos
    if len(nut_pos) > 1:
        nut_displacement_total = float(np.sum(np.linalg.norm(np.diff(nut_pos, axis=0), axis=1)))
    else:
        nut_displacement_total = 0.0

    grasp_idx = proxy.phases.grasp_index
    t_min_xy = proxy.phases.t_min_xy
    grasp_phase_duration = float(max(0, t_min_xy - grasp_idx))

    if grasp_idx is not None and grasp_idx < len(nut_pos):
        after = nut_pos[grasp_idx:]
        if len(after) > 1:
            nut_displacement_after_grasp = float(np.sum(np.linalg.norm(np.diff(after, axis=0), axis=1)))
        else:
            nut_displacement_after_grasp = 0.0
        eef_nut_distance_at_grasp = float(np.linalg.norm(eef[grasp_idx] - nut_pos[grasp_idx]))
        if grasp_idx < len(eef):
            dists = np.linalg.norm(eef[grasp_idx:] - nut_pos[grasp_idx:], axis=1)
            min_eef_nut_distance = float(np.min(dists))
        else:
            min_eef_nut_distance = eef_nut_distance_at_grasp
    else:
        nut_displacement_after_grasp = 0.0
        eef_nut_distance_at_grasp = 0.0
        min_eef_nut_distance = 0.0

    cur_min = current_min_xy if current_min_xy is not None else float(np.min(np.linalg.norm(nut_pos[:, :2] - proxy.peg_pos[:, :2], axis=1)))
    cur_final = current_final_xy if current_final_xy is not None else float(
        np.linalg.norm(nut_pos[-1, :2] - proxy.peg_pos[-1, :2])
    )
    transport_improvement_ratio = 0.0
    xy_improvement_ratio = 0.0
    if baseline_min_xy is not None and baseline_min_xy > 1e-6:
        transport_improvement_ratio = float((baseline_min_xy - cur_min) / baseline_min_xy)
    if baseline_final_xy is not None and baseline_final_xy > 1e-6:
        xy_improvement_ratio = float((baseline_final_xy - cur_final) / baseline_final_xy)

    return {
        "nut_displacement_total": nut_displacement_total,
        "nut_displacement_after_grasp": nut_displacement_after_grasp,
        "eef_nut_distance_at_grasp": eef_nut_distance_at_grasp,
        "min_eef_nut_distance": min_eef_nut_distance,
        "grasp_phase_duration": grasp_phase_duration,
        "transport_improvement_ratio": transport_improvement_ratio,
        "xy_improvement_ratio": xy_improvement_ratio,
    }


def _base_feature_dict(
    *,
    final_nut_peg_xy: float,
    min_nut_peg_xy: float,
    final_z_diff: float,
    min_yaw_error: float,
    final_yaw_error: float,
    ext: dict[str, float],
) -> dict[str, float]:
    return {
        "final_nut_peg_xy": final_nut_peg_xy,
        "min_nut_peg_xy": min_nut_peg_xy,
        "final_z_diff": final_z_diff,
        "min_yaw_error": min_yaw_error,
        "final_yaw_error": final_yaw_error,
        "action_accel_mean": ext["action_accel_mean"],
        "action_accel_max": ext["action_accel_max"],
        "eef_velocity_mean": ext["eef_velocity_mean"],
        "gripper_closed_fraction": ext["gripper_closed_fraction"],
        "grasp_index_norm": ext["grasp_index_norm"],
    }


def _parse_transport_params(row: dict[str, Any]) -> dict[str, float]:
    params = dict(DEFAULT_TRANSPORT_PARAMS)
    raw = row.get("transport_params") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        for key, val in raw.items():
            if key == "hold_steps":
                params["transport_hold_steps"] = float(val)
            elif key in params:
                params[key] = float(val)
    for key in TRANSPORT_PARAM_KEYS:
        csv_key = f"transport_{key}"
        if csv_key in row and str(row[csv_key]).strip() != "":
            params[key] = float(row[csv_key])
    return params


def _parse_sim_params(row: dict[str, Any]) -> dict[str, float]:
    params = dict(DEFAULT_SIM_PARAMS)
    raw = row.get("sim_params") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        params.update({k: float(raw[k]) for k in SIM_PARAM_KEYS if k in raw})
    return params


def _parse_grasp_params(row: dict[str, Any]) -> dict[str, float]:
    params = dict(DEFAULT_GRASP_PARAMS)
    raw = row.get("grasp_params") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        mapping = {
            "grasp_xy_offset_x": "grasp_xy_offset_x",
            "grasp_xy_offset_y": "grasp_xy_offset_y",
            "pre_grasp_height": "pre_grasp_height",
            "approach_height": "approach_height",
            "gripper_close_shift": "grasp_gripper_close_shift",
            "gripper_hold_steps": "grasp_gripper_hold_steps",
            "lift_height": "grasp_lift_height",
            "lift_steps": "grasp_lift_steps",
            "speed_scale": "grasp_speed_scale",
        }
        for src, dst in mapping.items():
            if src in raw:
                params[dst] = float(raw[src])
    csv_mapping = {
        "grasp_grasp_xy_offset_x": "grasp_xy_offset_x",
        "grasp_grasp_xy_offset_y": "grasp_xy_offset_y",
        "grasp_pre_grasp_height": "pre_grasp_height",
        "grasp_approach_height": "approach_height",
        "grasp_gripper_close_shift": "grasp_gripper_close_shift",
        "grasp_gripper_hold_steps": "grasp_gripper_hold_steps",
        "grasp_lift_height": "grasp_lift_height",
        "grasp_lift_steps": "grasp_lift_steps",
        "grasp_speed_scale": "grasp_speed_scale",
    }
    for csv_key, param_key in csv_mapping.items():
        if csv_key in row and str(row[csv_key]).strip() != "":
            params[param_key] = float(row[csv_key])
    return params


def _grasp_extra_from_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "nut_lift_delta": float(row.get("nut_lift_delta", 0.0)),
        "grasp_success_proxy_feat": _as_bool_float(row.get("grasp_success_proxy", False)),
        "lift_success_proxy_feat": _as_bool_float(row.get("lift_success_proxy", False)),
    }


def _grasp_diagnostic_from_row(row: dict[str, Any], *, baseline: dict[str, float] | None = None) -> dict[str, float]:
    baseline = baseline or {}
    cur_min = float(row.get("min_nut_peg_xy", 0.0))
    cur_final = float(row.get("final_nut_peg_xy", cur_min))
    baseline_min = baseline.get("baseline_min_xy")
    baseline_final = baseline.get("baseline_final_xy")
    transport_improvement_ratio = 0.0
    xy_improvement_ratio = 0.0
    if baseline_min is not None and baseline_min > 1e-6:
        transport_improvement_ratio = float((baseline_min - cur_min) / baseline_min)
    if baseline_final is not None and baseline_final > 1e-6:
        xy_improvement_ratio = float((baseline_final - cur_final) / baseline_final)
    return {
        "nut_displacement_total": float(row.get("nut_displacement_total", 0.0)),
        "nut_displacement_after_grasp": float(row.get("nut_displacement_after_grasp", 0.0)),
        "eef_nut_distance_at_grasp": float(row.get("eef_nut_distance_at_grasp", 0.0)),
        "min_eef_nut_distance": float(row.get("min_eef_nut_distance", row.get("eef_nut_distance_at_grasp", 0.0))),
        "grasp_phase_duration": float(row.get("grasp_phase_duration", 0.0)),
        "transport_improvement_ratio": transport_improvement_ratio,
        "xy_improvement_ratio": xy_improvement_ratio,
    }


def _level_flags_grasp(acceptance: dict[str, Any] | None) -> dict[str, float]:
    if not acceptance:
        return {"level_g1_pass": 0.0, "level_g2_pass": 0.0, "level_g3_pass": 0.0}
    return {
        "level_g1_pass": float(acceptance.get("level_g1_nut_motion_improved_50pct", False)),
        "level_g2_pass": float(acceptance.get("level_g2_lift_or_grasp_proxy", False)),
        "level_g3_pass": float(acceptance.get("level_g3_transport_improved_30pct", False)),
    }


def _resolve_failure_type(row: dict[str, Any], *, default: str = "unknown_failed") -> str:
    for key in ("failure_reason", "failure_type", "failure_guess"):
        val = row.get(key)
        if val and str(val).strip():
            resolved = normalize_failure_type(str(val))
            if resolved != "unknown_failed" or key == "failure_reason":
                return resolved
    return normalize_failure_type(default)


def _resolve_outcome(row: dict[str, Any], *, success_flag: bool) -> str:
    label = row.get("outcome") or row.get("outcome_label")
    if label:
        return normalize_outcome(str(label), success_flag=success_flag)
    if success_flag:
        return "refined_success" if row.get("rollout_kind", "").startswith("transport") else "success"
    return "unknown_outcome"


def _level_flags(
    *,
    original_min_xy: float | None,
    original_final_xy: float | None,
    min_xy: float,
    final_xy: float,
    success_flag: bool,
    acceptance: dict[str, Any] | None = None,
) -> dict[str, float]:
    if acceptance:
        return {
            "level_1_pass": float(acceptance.get("level_1_final_xy_reduction_50pct", False)),
            "level_2_pass": float(acceptance.get("level_2_min_xy_under_0.08", False)),
            "level_3_pass": float(acceptance.get("level_3_near_success_or_success", False)),
        }
    l1 = l2 = l3 = 0.0
    if original_final_xy is not None and original_final_xy > 1e-6:
        l1 = float((original_final_xy - final_xy) / original_final_xy >= 0.5)
    l2 = float(min_xy < 0.08)
    l3 = float(min_xy < 0.03 or success_flag)
    return {"level_1_pass": l1, "level_2_pass": l2, "level_3_pass": l3}


def _make_sample(
    *,
    dataset_version: str,
    source: str,
    demo_key: str,
    base: dict[str, float],
    ext: dict[str, float],
    targets: dict[str, Any],
    sim_params: dict[str, float] | None = None,
    transport_params: dict[str, float] | None = None,
    diagnostic: dict[str, float] | None = None,
    grasp_params: dict[str, float] | None = None,
    grasp_extra: dict[str, float] | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failure_type = str(targets["failure_type"])
    outcome = str(targets.get("outcome", "unknown_outcome"))
    success_flag = bool(targets["success_flag"])
    components = np.array(
        [
            float(targets["E_xy_norm"]),
            float(targets["E_transport_norm"]),
            float(targets["E_yaw_norm"]),
            float(targets["E_z_norm"]),
            float(targets["E_smooth_norm"]),
        ],
        dtype=np.float32,
    )
    levels = targets.get("levels") or {"level_1_pass": 0.0, "level_2_pass": 0.0, "level_3_pass": 0.0}
    grasp_levels = targets.get("grasp_levels") or {"level_g1_pass": 0.0, "level_g2_pass": 0.0, "level_g3_pass": 0.0}
    improvement_ratio = float(
        targets.get(
            "improvement_ratio",
            diagnostic.get("transport_improvement_ratio", 0.0) if diagnostic else 0.0,
        )
    )
    refined_success_flag = float(
        targets.get("refined_success_flag", outcome in ("success", "refined_success") or success_flag)
    )
    nut_lift_delta = float(
        targets.get("nut_lift_delta", (grasp_extra or {}).get("nut_lift_delta", 0.0))
    )
    nut_disp_after_grasp = float(
        targets.get(
            "nut_displacement_after_grasp",
            (diagnostic or {}).get("nut_displacement_after_grasp", 0.0),
        )
    )
    grasp_success_proxy = float(
        targets.get("grasp_success_proxy", (grasp_extra or {}).get("grasp_success_proxy_feat", 0.0))
    )
    lift_success_proxy = float(
        targets.get("lift_success_proxy", (grasp_extra or {}).get("lift_success_proxy_feat", 0.0))
    )

    features = build_feature_vector(
        dataset_version=dataset_version,
        base=base,
        sim_params=sim_params,
        transport_params=transport_params,
        diagnostic=diagnostic if dataset_version in ("v1b", "v1c") else None,
        grasp_params=grasp_params if dataset_version == "v1c" else None,
        grasp_extra=grasp_extra if dataset_version == "v1c" else None,
    )
    return {
        "features": features,
        "targets_components": components,
        "target_E_total": float(targets["E_total_norm"]),
        "success_flag": float(success_flag),
        "failure_type_idx": failure_idx(failure_type),
        "outcome_idx": outcome_idx(outcome),
        "level_1_pass": float(levels["level_1_pass"]),
        "level_2_pass": float(levels["level_2_pass"]),
        "level_3_pass": float(levels["level_3_pass"]),
        "level_g1_pass": float(grasp_levels["level_g1_pass"]),
        "level_g2_pass": float(grasp_levels["level_g2_pass"]),
        "level_g3_pass": float(grasp_levels["level_g3_pass"]),
        "improvement_ratio": improvement_ratio,
        "refined_success_flag": refined_success_flag,
        "grasp_success_flag": grasp_success_proxy,
        "lift_success_flag": lift_success_proxy,
        "nut_lift_delta": nut_lift_delta,
        "nut_displacement_after_grasp": nut_disp_after_grasp,
        "meta": {
            "source": source,
            "sample_source": source,
            "demo_key": demo_key,
            "source_demo": demo_key,
            "source_failure_mode": collapse_failure_mode(failure_type),
            "failure_type": failure_type,
            "outcome": outcome,
            **(meta_extra or {}),
        },
    }


def _sim_energy_targets(
    row: dict[str, Any],
    *,
    demo_key: str,
    hdf5_path: str,
    ext: dict[str, float],
) -> dict[str, Any]:
    final_xy = float(row.get("final_nut_peg_xy", row.get("final_nut_peg_xy_distance", 0)))
    min_xy = float(row.get("min_nut_peg_xy", row.get("min_nut_peg_xy_distance", final_xy)))
    final_z = float(row.get("final_z_diff", row.get("final_nut_peg_z_difference", 0)))
    min_yaw = float(row.get("min_yaw_error", row.get("min_nut_peg_yaw_error", 0)))
    final_yaw = float(row.get("final_yaw_error", min_yaw))
    acc_max = float(row.get("action_acceleration_max", ext["action_accel_max"]))
    acc_mean = float(row.get("action_acceleration_mean", ext["action_accel_mean"]))
    success_flag = bool(row.get("success_flag", False))

    if all(k in row for k in ("E_xy_norm", "E_transport_norm", "E_yaw_norm", "E_z_norm", "E_smooth_norm")):
        failure_type = _resolve_failure_type(row)
        if success_flag:
            failure_type = "success"
        return {
            "E_xy_norm": float(row["E_xy_norm"]),
            "E_transport_norm": float(row["E_transport_norm"]),
            "E_yaw_norm": float(row["E_yaw_norm"]),
            "E_z_norm": float(row["E_z_norm"]),
            "E_smooth_norm": float(row["E_smooth_norm"]),
            "E_total_norm": float(row["E_total_norm"]),
            "failure_type": failure_type,
            "success_flag": success_flag,
            "outcome": _resolve_outcome(row, success_flag=success_flag),
            "grasp_success_proxy": _as_bool_float(row.get("grasp_success_proxy", False)),
            "lift_success_proxy": _as_bool_float(row.get("lift_success_proxy", False)),
            "nut_lift_delta": float(row.get("nut_lift_delta", 0.0)),
            "nut_displacement_after_grasp": float(row.get("nut_displacement_after_grasp", 0.0)),
        }

    feat = NutAssemblyFeatures(
        demo_key=demo_key,
        label=str(row.get("label", "failed")),
        source_file=str(row.get("source_file", hdf5_path)),
        length=int(row.get("num_steps", ext.get("length", 100))),
        final_nut_peg_xy_distance=final_xy,
        min_nut_peg_xy_distance=min_xy,
        final_nut_peg_z_difference=final_z,
        min_nut_peg_yaw_error=min_yaw,
        final_nut_peg_yaw_error=final_yaw,
        action_acceleration_mean=acc_mean,
        action_acceleration_max=acc_max,
        grasp_signal_index=None,
    )
    energy = compute_total_energy(feat)
    failure_type = "success" if success_flag else energy.failure_type
    failure_type = _resolve_failure_type({"failure_reason": row.get("failure_reason"), "failure_guess": failure_type})
    if success_flag:
        failure_type = "success"
    return {
        "E_xy_norm": energy.E_xy_norm,
        "E_transport_norm": energy.E_transport_norm,
        "E_yaw_norm": energy.E_yaw_norm,
        "E_z_norm": energy.E_z_norm,
        "E_smooth_norm": energy.E_smooth_norm,
        "E_total_norm": float(row.get("E_total_norm", energy.E_total_norm)),
        "failure_type": failure_type,
        "success_flag": success_flag,
        "outcome": _resolve_outcome(row, success_flag=success_flag),
    }


def _sample_from_features(
    feat: NutAssemblyFeatures,
    ext: dict[str, float],
    *,
    dataset_version: str,
    source: str,
    energy_targets: dict[str, Any],
    sim_params: dict[str, float] | None = None,
    transport_params: dict[str, float] | None = None,
    diagnostic: dict[str, float] | None = None,
    demo_key: str = "",
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = _base_feature_dict(
        final_nut_peg_xy=feat.final_nut_peg_xy_distance,
        min_nut_peg_xy=feat.min_nut_peg_xy_distance,
        final_z_diff=feat.final_nut_peg_z_difference,
        min_yaw_error=feat.min_nut_peg_yaw_error,
        final_yaw_error=feat.final_nut_peg_yaw_error,
        ext=ext,
    )
    diag = diagnostic
    if dataset_version in ("v1b", "v1c") and diag is None:
        diag = _diagnostic_features(
            feat.source_file,
            feat.demo_key,
            feat.label,
            current_min_xy=feat.min_nut_peg_xy_distance,
            current_final_xy=feat.final_nut_peg_xy_distance,
        )
    grasp_params = dict(DEFAULT_GRASP_PARAMS) if dataset_version == "v1c" else None
    grasp_extra = dict(DEFAULT_GRASP_EXTRA) if dataset_version == "v1c" else None
    return _make_sample(
        dataset_version=dataset_version,
        source=source,
        demo_key=demo_key or feat.demo_key,
        base=base,
        ext=ext,
        targets=energy_targets,
        sim_params=sim_params,
        transport_params=transport_params,
        diagnostic=diag,
        grasp_params=grasp_params,
        grasp_extra=grasp_extra,
        meta_extra={"label": feat.label, **(meta_extra or {})},
    )


def _sample_from_sim_rollout(
    row: dict[str, Any],
    *,
    dataset_version: str,
    source: str,
    demo_key: str,
    hdf5_path: str,
    ext_base: dict[str, float] | None = None,
    transport_params: dict[str, float] | None = None,
    diagnostic: dict[str, float] | None = None,
    levels: dict[str, float] | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ext = ext_base or _extended_hdf5_features(hdf5_path, demo_key)
    sim_params = _parse_sim_params(row)
    transport = transport_params or _parse_transport_params(row)
    targets = _sim_energy_targets(row, demo_key=demo_key, hdf5_path=hdf5_path, ext=ext)
    if levels:
        targets["levels"] = levels

    final_xy = float(row.get("final_nut_peg_xy", 0))
    min_xy = float(row.get("min_nut_peg_xy", final_xy))
    action_accel_max = float(row.get("action_acceleration_max", ext["action_accel_max"]))
    ext_rollout = {**ext, "action_accel_max": action_accel_max}

    diag = diagnostic
    if dataset_version in ("v1b", "v1c") and diag is None:
        diag = _diagnostic_features(
            hdf5_path,
            demo_key,
            str(row.get("label", "failed")),
            baseline_min_xy=meta_extra.get("baseline_min_xy") if meta_extra else None,
            baseline_final_xy=meta_extra.get("baseline_final_xy") if meta_extra else None,
            current_min_xy=min_xy,
            current_final_xy=final_xy,
        )
        targets["improvement_ratio"] = max(diag["transport_improvement_ratio"], diag["xy_improvement_ratio"])

    grasp_params = dict(DEFAULT_GRASP_PARAMS) if dataset_version == "v1c" else None
    grasp_extra = dict(DEFAULT_GRASP_EXTRA) if dataset_version == "v1c" else None
    if "gripper_closed_fraction" in row:
        ext_rollout["gripper_closed_fraction"] = float(row["gripper_closed_fraction"])

    base = _base_feature_dict(
        final_nut_peg_xy=final_xy,
        min_nut_peg_xy=min_xy,
        final_z_diff=float(row.get("final_z_diff", 0)),
        min_yaw_error=float(row.get("min_yaw_error", 0)),
        final_yaw_error=float(row.get("final_yaw_error", row.get("min_yaw_error", 0))),
        ext=ext_rollout,
    )
    return _make_sample(
        dataset_version=dataset_version,
        source=source,
        demo_key=demo_key,
        base=base,
        ext=ext_rollout,
        targets=targets,
        sim_params=sim_params,
        transport_params=transport,
        diagnostic=diag,
        grasp_params=grasp_params,
        grasp_extra=grasp_extra,
        meta_extra=meta_extra,
    )


def _sample_from_grasp_rollout(
    row: dict[str, Any],
    *,
    dataset_version: str,
    source: str,
    demo_key: str,
    hdf5_path: str,
    ext_base: dict[str, float] | None = None,
    diagnostic: dict[str, float] | None = None,
    grasp_levels: dict[str, float] | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ext = ext_base or _extended_hdf5_features(hdf5_path, demo_key)
    sim_params = _parse_sim_params(row)
    transport = _parse_transport_params(row)
    grasp_params = _parse_grasp_params(row)
    grasp_extra = _grasp_extra_from_row(row)
    targets = _sim_energy_targets(row, demo_key=demo_key, hdf5_path=hdf5_path, ext=ext)
    if grasp_levels:
        targets["grasp_levels"] = grasp_levels

    final_xy = float(row.get("final_nut_peg_xy", 0))
    min_xy = float(row.get("min_nut_peg_xy", final_xy))
    action_accel_max = float(row.get("action_acceleration_max", ext["action_accel_max"]))
    ext_rollout = {**ext, "action_accel_max": action_accel_max}
    if "gripper_closed_fraction" in row:
        ext_rollout["gripper_closed_fraction"] = float(row["gripper_closed_fraction"])

    baseline_meta = meta_extra or {}
    diag = diagnostic or _grasp_diagnostic_from_row(row, baseline=baseline_meta)
    targets["improvement_ratio"] = max(diag["transport_improvement_ratio"], diag["xy_improvement_ratio"])

    base = _base_feature_dict(
        final_nut_peg_xy=final_xy,
        min_nut_peg_xy=min_xy,
        final_z_diff=float(row.get("final_z_diff", 0)),
        min_yaw_error=float(row.get("min_yaw_error", 0)),
        final_yaw_error=float(row.get("final_yaw_error", row.get("min_yaw_error", 0))),
        ext=ext_rollout,
    )
    return _make_sample(
        dataset_version=dataset_version,
        source=source,
        demo_key=demo_key,
        base=base,
        ext=ext_rollout,
        targets=targets,
        sim_params=sim_params,
        transport_params=transport,
        diagnostic=diag,
        grasp_params=grasp_params,
        grasp_extra=grasp_extra,
        meta_extra=meta_extra,
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _collect_hdf5_samples(
    success_hdf5: Path,
    failed_hdf5: Path,
    *,
    dataset_version: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path, label in [(success_hdf5, "success"), (failed_hdf5, "failed")]:
        feats = load_features_from_hdf5(str(path), label)
        for feat in feats:
            ext = _extended_hdf5_features(str(path), feat.demo_key)
            energy = compute_total_energy(feat)
            targets = {
                "E_xy_norm": energy.E_xy_norm,
                "E_transport_norm": energy.E_transport_norm,
                "E_yaw_norm": energy.E_yaw_norm,
                "E_z_norm": energy.E_z_norm,
                "E_smooth_norm": energy.E_smooth_norm,
                "E_total_norm": energy.E_total_norm,
                "failure_type": energy.failure_type if label == "failed" else "success",
                "success_flag": label == "success",
                "outcome": "success" if label == "success" else "baseline",
            }
            samples.append(
                _sample_from_features(
                    feat,
                    ext,
                    dataset_version=dataset_version,
                    source="hdf5_baseline",
                    energy_targets=targets,
                )
            )
    return samples


def _collect_cem_samples(
    cem_report_path: Path,
    failed_hdf5: Path,
    *,
    dataset_version: str,
) -> list[dict[str, Any]]:
    if not cem_report_path.exists():
        return []
    report = json.loads(cem_report_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    with h5py.File(str(failed_hdf5), "r") as handle:
        for item in report.get("results", []):
            demo_key = item["demo_key"]
            ext = _extended_hdf5_features(str(failed_hdf5), demo_key)
            feat = extract_demo_features(handle[f"data/{demo_key}"], demo_key, "failed", str(failed_hdf5))
            before = item["components_before"]
            after = item["components_after"]
            res_a = item["residual_after"]
            theta = item.get("best_theta") or {}

            samples.append(
                _sample_from_features(
                    feat,
                    ext,
                    dataset_version=dataset_version,
                    source="cem_proxy_before",
                    demo_key=demo_key,
                    energy_targets={
                        "E_xy_norm": before["xy"],
                        "E_transport_norm": before["transport"],
                        "E_yaw_norm": before["yaw"],
                        "E_z_norm": before["z"],
                        "E_smooth_norm": before["smooth"],
                        "E_total_norm": item["energy_before"],
                        "failure_type": item["failure_type_before"],
                        "success_flag": False,
                        "outcome": "baseline",
                    },
                )
            )
            feat_after = NutAssemblyFeatures(
                demo_key=demo_key,
                label="failed",
                source_file=str(failed_hdf5),
                length=feat.length,
                final_nut_peg_xy_distance=float(res_a["final_xy"]),
                min_nut_peg_xy_distance=float(res_a["min_xy"]),
                final_nut_peg_z_difference=float(res_a["final_z"]),
                min_nut_peg_yaw_error=float(res_a["min_yaw"]),
                final_nut_peg_yaw_error=float(res_a["min_yaw"]),
                action_acceleration_mean=feat.action_acceleration_mean,
                action_acceleration_max=feat.action_acceleration_max,
                grasp_signal_index=feat.grasp_signal_index,
            )
            sim_params = {
                "insert_z_offset": float(theta.get("insert_z_offset", 0.0)),
                "release_shift": float(theta.get("release_shift", 0.0)),
                "insertion_speed_scale": float(theta.get("speed_scale", 1.0)),
            }
            samples.append(
                _sample_from_features(
                    feat_after,
                    ext,
                    dataset_version=dataset_version,
                    source="cem_proxy_after",
                    demo_key=demo_key,
                    sim_params=sim_params,
                    energy_targets={
                        "E_xy_norm": after["xy"],
                        "E_transport_norm": after["transport"],
                        "E_yaw_norm": after["yaw"],
                        "E_z_norm": after["z"],
                        "E_smooth_norm": after["smooth"],
                        "E_total_norm": item["energy_after"],
                        "failure_type": normalize_failure_type(
                            item.get("failure_type_after_raw", item["failure_type_after"])
                        ),
                        "success_flag": False,
                        "outcome": "search_candidate",
                    },
                )
            )
    return samples


def _collect_sim_in_loop_samples(
    report_path: Path,
    failed_hdf5: Path,
    demo_key: str,
    *,
    dataset_version: str,
) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ext = _extended_hdf5_features(str(failed_hdf5), demo_key)
    samples: list[dict[str, Any]] = []
    for key in ["original_waypoint_rollout", "refined_waypoint_rollout_default", "best_sim_in_loop"]:
        row = report.get(key)
        if row:
            samples.append(
                _sample_from_sim_rollout(
                    row,
                    dataset_version=dataset_version,
                    source=f"sim_in_loop_{key}",
                    demo_key=demo_key,
                    hdf5_path=str(failed_hdf5),
                    ext_base=ext,
                )
            )
    for row in report.get("top_10_candidates", []):
        samples.append(
            _sample_from_sim_rollout(
                row,
                dataset_version=dataset_version,
                source="sim_in_loop_search",
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
            )
        )
    return samples


def _collect_repeatability_samples(
    report_path: Path,
    failed_hdf5: Path,
    demo_key: str,
    *,
    dataset_version: str,
) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ext = _extended_hdf5_features(str(failed_hdf5), demo_key)
    samples: list[dict[str, Any]] = []
    for row in report.get("runs", []):
        sim_params = row.get("sim_params")
        if sim_params is None and row.get("best_params"):
            sim_params = json.loads(row["best_params"])
        payload = {**row, "sim_params": sim_params or {}}
        samples.append(
            _sample_from_sim_rollout(
                payload,
                dataset_version=dataset_version,
                source="sim_repeatability",
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
            )
        )
    return samples


def _collect_ablation_samples(
    report_path: Path,
    failed_hdf5: Path,
    demo_key: str,
    *,
    dataset_version: str,
) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ext = _extended_hdf5_features(str(failed_hdf5), demo_key)
    samples: list[dict[str, Any]] = []
    for method_id, row in report.get("method_results", {}).items():
        payload = {**row, "method_id": method_id}
        samples.append(
            _sample_from_sim_rollout(
                payload,
                dataset_version=dataset_version,
                source=f"sim_ablation_{method_id}",
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
            )
        )
    return samples


def _transport_source_name(outcome: str) -> str:
    mapping = {
        "refined_success": "transport_refined_success",
        "improved_but_failed": "transport_improved_but_failed",
        "no_improvement": "transport_no_improvement",
        "baseline": "transport_original_waypoint",
    }
    return mapping.get(outcome, "transport_top_candidate")


def _collect_transport_refinement_samples(
    transport_dir: Path,
    failed_hdf5: Path,
    *,
    dataset_version: str,
) -> list[dict[str, Any]]:
    report_path = transport_dir / "transport_refinement_report.json"
    if not report_path.exists():
        return []

    if (transport_dir / "transport_refinement_summary.csv").exists():
        _load_csv(transport_dir / "transport_refinement_summary.csv")
    if (transport_dir / "per_demo_best.json").exists():
        json.loads((transport_dir / "per_demo_best.json").read_text(encoding="utf-8"))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_demo = report.get("per_demo", {})
    samples: list[dict[str, Any]] = []

    for demo_key, demo_result in per_demo.items():
        ext = _extended_hdf5_features(str(failed_hdf5), demo_key)
        original = demo_result["original_waypoint_rollout"]
        best = demo_result["best_transport_refined"]
        acceptance = demo_result.get("acceptance_levels") or {}
        baseline_min_xy = float(original["min_nut_peg_xy"])
        baseline_final_xy = float(original["final_nut_peg_xy"])
        baseline_meta = {"baseline_min_xy": baseline_min_xy, "baseline_final_xy": baseline_final_xy}

        orig_outcome = "baseline"
        orig_row = dict(original)
        orig_row["outcome"] = orig_outcome
        orig_row["failure_reason"] = demo_result.get("failure_reason", original.get("failure_guess"))
        samples.append(
            _sample_from_sim_rollout(
                orig_row,
                dataset_version=dataset_version,
                source="transport_original_waypoint",
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
                levels=_level_flags(
                    original_min_xy=baseline_min_xy,
                    original_final_xy=baseline_final_xy,
                    min_xy=float(original["min_nut_peg_xy"]),
                    final_xy=float(original["final_nut_peg_xy"]),
                    success_flag=bool(original.get("success_flag", False)),
                ),
                meta_extra=baseline_meta,
            )
        )

        best_outcome = demo_result.get("outcome", best.get("outcome_label", "unknown_outcome"))
        best_row = dict(best)
        best_row["outcome"] = best_outcome
        best_row["failure_reason"] = demo_result.get("failure_reason", best.get("failure_reason"))
        samples.append(
            _sample_from_sim_rollout(
                best_row,
                dataset_version=dataset_version,
                source=_transport_source_name(str(best_outcome)),
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
                levels=_level_flags(
                    original_min_xy=baseline_min_xy,
                    original_final_xy=baseline_final_xy,
                    min_xy=float(best["min_nut_peg_xy"]),
                    final_xy=float(best["final_nut_peg_xy"]),
                    success_flag=bool(best.get("success_flag", False)),
                    acceptance=acceptance,
                ),
                meta_extra=baseline_meta,
            )
        )

        for row in demo_result.get("top_10_candidates", []):
            row = dict(row)
            row_outcome = row.get("outcome") or "search_candidate"
            if not row_outcome or row_outcome == "":
                row_outcome = "search_candidate"
            row["outcome"] = normalize_outcome(str(row_outcome), success_flag=bool(row.get("success_flag", False)))
            if row.get("failure_reason"):
                row["failure_type"] = row["failure_reason"]
            source = (
                _transport_source_name(str(row["outcome"]))
                if row["outcome"] in ("refined_success", "improved_but_failed", "no_improvement")
                else "transport_top_candidate"
            )
            samples.append(
                _sample_from_sim_rollout(
                    row,
                    dataset_version=dataset_version,
                    source=source,
                    demo_key=demo_key,
                    hdf5_path=str(failed_hdf5),
                    ext_base=ext,
                    meta_extra=baseline_meta,
                )
            )

    top_csv = transport_dir / "top_candidates.csv"
    if top_csv.exists():
        seen = {(s["meta"]["demo_key"], s["meta"].get("search_index"), s["meta"]["source"]) for s in samples if "search_index" in s["meta"]}
        for row in _load_csv(top_csv):
            demo_key = row["demo_key"]
            ext = _extended_hdf5_features(str(failed_hdf5), demo_key)
            original = per_demo.get(demo_key, {}).get("original_waypoint_rollout", {})
            baseline_meta = {
                "baseline_min_xy": float(original.get("min_nut_peg_xy", row.get("min_nut_peg_xy", 0))),
                "baseline_final_xy": float(original.get("final_nut_peg_xy", row.get("final_nut_peg_xy", 0))),
            }
            payload = dict(row)
            if payload.get("failure_reason"):
                payload["failure_type"] = payload["failure_reason"]
            outcome = normalize_outcome(
                str(payload.get("outcome") or "search_candidate"),
                success_flag=bool(payload.get("success_flag") == "True" or payload.get("success_flag") is True),
            )
            payload["success_flag"] = payload.get("success_flag") in (True, "True", "true", "1", 1)
            payload["outcome"] = outcome
            source = (
                _transport_source_name(outcome)
                if outcome in ("refined_success", "improved_but_failed", "no_improvement")
                else "transport_top_candidate"
            )
            sample = _sample_from_sim_rollout(
                payload,
                dataset_version=dataset_version,
                source=source,
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
                meta_extra={**baseline_meta, "search_index": payload.get("search_index"), "rank": payload.get("rank")},
            )
            key = (demo_key, payload.get("search_index"), source)
            if key not in seen:
                samples.append(sample)
                seen.add(key)

    return samples


def _grasp_source_name(outcome: str, *, failure_reason: str = "") -> str:
    mapping = {
        "refined_success": "grasp_refined_success",
        "grasp_improved_but_failed": "grasp_improved_but_failed",
        "no_improvement": "grasp_no_improvement",
        "grasp_no_improvement": "grasp_no_improvement",
        "baseline": "grasp_original_waypoint",
    }
    if failure_reason == "lift_failed" and outcome in ("grasp_improved_but_failed", "failed", ""):
        return "lift_failed_candidate"
    return mapping.get(outcome, "grasp_top_candidate")


def _collect_grasp_refinement_samples(
    grasp_dir: Path,
    failed_hdf5: Path,
    *,
    dataset_version: str,
) -> list[dict[str, Any]]:
    report_path = grasp_dir / "grasp_refinement_report.json"
    if not report_path.exists():
        return []

    if (grasp_dir / "grasp_refinement_summary.csv").exists():
        _load_csv(grasp_dir / "grasp_refinement_summary.csv")
    if (grasp_dir / "per_demo_best.json").exists():
        json.loads((grasp_dir / "per_demo_best.json").read_text(encoding="utf-8"))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_demo = report.get("per_demo", {})
    samples: list[dict[str, Any]] = []

    for demo_key, demo_result in per_demo.items():
        ext = _extended_hdf5_features(str(failed_hdf5), demo_key)
        original = demo_result["original_waypoint_rollout"]
        best = demo_result["best_grasp_refined"]
        acceptance = demo_result.get("acceptance_levels") or {}
        baseline_min_xy = float(original["min_nut_peg_xy"])
        baseline_final_xy = float(original["final_nut_peg_xy"])
        baseline_meta = {"baseline_min_xy": baseline_min_xy, "baseline_final_xy": baseline_final_xy}

        orig_row = dict(original)
        orig_row["outcome"] = "baseline"
        orig_row["failure_reason"] = demo_result.get(
            "failure_reason",
            original.get("failure_reason", original.get("failure_guess", "grasp_failed")),
        )
        samples.append(
            _sample_from_grasp_rollout(
                orig_row,
                dataset_version=dataset_version,
                source="grasp_original_waypoint",
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
                grasp_levels=_level_flags_grasp(None),
                meta_extra=baseline_meta,
            )
        )

        best_outcome = demo_result.get("outcome", best.get("outcome_label", "unknown_outcome"))
        best_failure = demo_result.get("failure_reason", best.get("failure_reason", ""))
        best_row = dict(best)
        best_row["outcome"] = best_outcome
        best_row["failure_reason"] = best_failure
        samples.append(
            _sample_from_grasp_rollout(
                best_row,
                dataset_version=dataset_version,
                source=_grasp_source_name(str(best_outcome), failure_reason=str(best_failure)),
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
                grasp_levels=_level_flags_grasp(acceptance),
                meta_extra=baseline_meta,
            )
        )

        for row in demo_result.get("top_10_candidates", []):
            row = dict(row)
            row_outcome = row.get("outcome") or row.get("outcome_label") or "search_candidate"
            if not row_outcome or row_outcome == "":
                row_outcome = "search_candidate"
            row["outcome"] = normalize_outcome(str(row_outcome), success_flag=bool(row.get("success_flag", False)))
            failure_reason = str(row.get("failure_reason") or row.get("failure_guess") or "")
            if failure_reason:
                row["failure_type"] = failure_reason
            source = _grasp_source_name(str(row["outcome"]), failure_reason=failure_reason)
            if source == "grasp_top_candidate" and row_outcome == "search_candidate":
                source = "grasp_top_candidate"
            samples.append(
                _sample_from_grasp_rollout(
                    row,
                    dataset_version=dataset_version,
                    source=source,
                    demo_key=demo_key,
                    hdf5_path=str(failed_hdf5),
                    ext_base=ext,
                    meta_extra=baseline_meta,
                )
            )

    top_csv = grasp_dir / "top_candidates.csv"
    if top_csv.exists():
        seen = {
            (s["meta"]["demo_key"], s["meta"].get("rank"), s["meta"]["source"])
            for s in samples
            if "rank" in s["meta"]
        }
        for row in _load_csv(top_csv):
            demo_key = row["demo_key"]
            ext = _extended_hdf5_features(str(failed_hdf5), demo_key)
            original = per_demo.get(demo_key, {}).get("original_waypoint_rollout", {})
            baseline_meta = {
                "baseline_min_xy": float(original.get("min_nut_peg_xy", row.get("min_nut_peg_xy", 0))),
                "baseline_final_xy": float(original.get("final_nut_peg_xy", row.get("final_nut_peg_xy", 0))),
            }
            payload = dict(row)
            payload["success_flag"] = payload.get("success_flag") in (True, "True", "true", "1", 1)
            outcome = normalize_outcome(
                str(payload.get("outcome") or "search_candidate"),
                success_flag=bool(payload["success_flag"]),
            )
            payload["outcome"] = outcome
            failure_reason = str(payload.get("failure_reason") or payload.get("failure_guess") or "")
            source = _grasp_source_name(outcome, failure_reason=failure_reason)
            if outcome == "search_candidate" and not payload["success_flag"]:
                source = "grasp_top_candidate"
            sample = _sample_from_grasp_rollout(
                payload,
                dataset_version=dataset_version,
                source=source,
                demo_key=demo_key,
                hdf5_path=str(failed_hdf5),
                ext_base=ext,
                meta_extra={**baseline_meta, "rank": payload.get("rank")},
            )
            key = (demo_key, payload.get("rank"), source)
            if key not in seen:
                samples.append(sample)
                seen.add(key)

    return samples


def build_dataset(
    *,
    experiment_dir: Path,
    success_hdf5: Path,
    failed_hdf5: Path,
    dataset_version: str = "v1b",
    include_transport: bool = True,
    include_grasp: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = experiment_dir / "outputs"
    if (outputs / "energy_report.json").exists():
        json.loads((outputs / "energy_report.json").read_text(encoding="utf-8"))
    if (outputs / "energy_summary.csv").exists():
        _load_csv(outputs / "energy_summary.csv")

    samples: list[dict[str, Any]] = []
    samples.extend(_collect_hdf5_samples(success_hdf5, failed_hdf5, dataset_version=dataset_version))
    samples.extend(
        _collect_cem_samples(outputs / "cem_refinement" / "cem_refinement_report.json", failed_hdf5, dataset_version=dataset_version)
    )
    samples.extend(
        _collect_sim_in_loop_samples(
            outputs / "sim_in_loop_refinement" / "sim_in_loop_refinement_report.json",
            failed_hdf5,
            "demo_4",
            dataset_version=dataset_version,
        )
    )
    samples.extend(
        _collect_repeatability_samples(
            outputs / "sim_in_loop_repeatability" / "repeatability_report.json",
            failed_hdf5,
            "demo_4",
            dataset_version=dataset_version,
        )
    )
    samples.extend(
        _collect_ablation_samples(
            outputs / "sim_in_loop_ablation" / "ablation_report.json",
            failed_hdf5,
            "demo_4",
            dataset_version=dataset_version,
        )
    )
    if include_transport or dataset_version in ("v1b", "v1c"):
        samples.extend(
            _collect_transport_refinement_samples(
                outputs / "transport_refinement",
                failed_hdf5,
                dataset_version=dataset_version,
            )
        )
    if include_grasp or dataset_version == "v1c":
        samples.extend(
            _collect_grasp_refinement_samples(
                outputs / "grasp_refinement",
                failed_hdf5,
                dataset_version=dataset_version,
            )
        )

    source_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    for s in samples:
        src = s["meta"]["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
        ft = s["meta"]["failure_type"]
        failure_counts[ft] = failure_counts.get(ft, 0) + 1
        oc = s["meta"]["outcome"]
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1

    if dataset_version == "v1a":
        feature_names = FEATURE_NAMES_V1A
    elif dataset_version == "v1b":
        feature_names = FEATURE_NAMES_V1B
    else:
        feature_names = FEATURE_NAMES_V1C

    version_notes = {
        "v1a": [
            "V1-A insertion-focused residual model dataset.",
            "Not a final PINN / PINA model.",
        ],
        "v1b": [
            "V1-B multi-failure-mode residual model dataset.",
            "Includes V0/V0.5, V2-A, V2-B2.6 demo_4, and V2-B3 transport_refinement rollouts.",
            "no_improvement / grasp_failed samples are retained intentionally.",
            "Not a final PINN / PINA model.",
        ],
        "v1c": [
            "V1-C grasp-aware multi-failure residual model dataset.",
            "Includes V1-B sources plus V2-B4 grasp_refinement rollouts.",
            "Retains grasp_failed, lift_failed, grasp_improved_but_failed, refined_success.",
            "Not a final PINN / PINA model.",
        ],
    }
    summary = {
        "dataset_version": dataset_version,
        "num_samples": len(samples),
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "target_component_names": TARGET_COMPONENT_NAMES,
        "source_counts": source_counts,
        "failure_type_counts": failure_counts,
        "outcome_counts": outcome_counts,
        "success_count": int(sum(s["success_flag"] for s in samples)),
        "failed_count": int(sum(1 - s["success_flag"] for s in samples)),
        "inputs": {
            "energy_report": str(outputs / "energy_report.json"),
            "energy_summary": str(outputs / "energy_summary.csv"),
            "cem_refinement": str(outputs / "cem_refinement" / "cem_refinement_report.json"),
            "sim_in_loop": str(outputs / "sim_in_loop_refinement" / "sim_in_loop_refinement_report.json"),
            "repeatability": str(outputs / "sim_in_loop_repeatability" / "repeatability_report.json"),
            "ablation": str(outputs / "sim_in_loop_ablation" / "ablation_report.json"),
            "transport_refinement": str(outputs / "transport_refinement" / "transport_refinement_report.json"),
            "grasp_refinement": str(outputs / "grasp_refinement" / "grasp_refinement_report.json"),
            "success_hdf5": str(success_hdf5),
            "failed_hdf5": str(failed_hdf5),
        },
        "notes": version_notes.get(dataset_version, version_notes["v1b"]),
    }
    return samples, summary


def save_dataset(samples: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = summary["feature_names"]
    summary["meta_records"] = [s["meta"] for s in samples]
    npz_path = output_dir / "training_dataset.npz"
    np.savez_compressed(
        npz_path,
        features=np.stack([s["features"] for s in samples], axis=0),
        targets_components=np.stack([s["targets_components"] for s in samples], axis=0),
        target_E_total=np.array([s["target_E_total"] for s in samples], dtype=np.float32),
        success_flag=np.array([s["success_flag"] for s in samples], dtype=np.float32),
        failure_type_idx=np.array([s["failure_type_idx"] for s in samples], dtype=np.int64),
        outcome_idx=np.array([s["outcome_idx"] for s in samples], dtype=np.int64),
        level_1_pass=np.array([s["level_1_pass"] for s in samples], dtype=np.float32),
        level_2_pass=np.array([s["level_2_pass"] for s in samples], dtype=np.float32),
        level_3_pass=np.array([s["level_3_pass"] for s in samples], dtype=np.float32),
        improvement_ratio=np.array([s["improvement_ratio"] for s in samples], dtype=np.float32),
        refined_success_flag=np.array([s["refined_success_flag"] for s in samples], dtype=np.float32),
        grasp_success_flag=np.array([s.get("grasp_success_flag", 0.0) for s in samples], dtype=np.float32),
        lift_success_flag=np.array([s.get("lift_success_flag", 0.0) for s in samples], dtype=np.float32),
        nut_lift_delta=np.array([s.get("nut_lift_delta", 0.0) for s in samples], dtype=np.float32),
        nut_displacement_after_grasp=np.array(
            [s.get("nut_displacement_after_grasp", 0.0) for s in samples], dtype=np.float32
        ),
        meta_json=json.dumps(summary),
    )

    jsonl_path = output_dir / "training_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "features": {feature_names[i]: float(sample["features"][i]) for i in range(len(feature_names))},
                "targets": {
                    **{TARGET_COMPONENT_NAMES[i]: float(sample["targets_components"][i]) for i in range(5)},
                    "E_total_norm": float(sample["target_E_total"]),
                    "success_flag": bool(sample["success_flag"]),
                    "failure_type_idx": int(sample["failure_type_idx"]),
                    "outcome_idx": int(sample["outcome_idx"]),
                    "level_1_pass": bool(sample["level_1_pass"]),
                    "level_2_pass": bool(sample["level_2_pass"]),
                    "level_3_pass": bool(sample["level_3_pass"]),
                    "improvement_ratio": float(sample["improvement_ratio"]),
                    "refined_success_flag": bool(sample["refined_success_flag"]),
                    "grasp_success_proxy": bool(sample.get("grasp_success_flag", 0.0)),
                    "lift_success_proxy": bool(sample.get("lift_success_flag", 0.0)),
                    "level_g1_pass": bool(sample.get("level_g1_pass", 0.0)),
                    "level_g2_pass": bool(sample.get("level_g2_pass", 0.0)),
                    "level_g3_pass": bool(sample.get("level_g3_pass", 0.0)),
                    "nut_lift_delta": float(sample.get("nut_lift_delta", 0.0)),
                    "nut_displacement_after_grasp": float(sample.get("nut_displacement_after_grasp", 0.0)),
                },
                "meta": sample["meta"],
            }
            handle.write(json.dumps(record) + "\n")

    (output_dir / "training_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return npz_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-A / V1-B / V1-C training dataset")
    parser.add_argument("--experiment-dir", type=Path, default=_EXPERIMENT_DIR)
    parser.add_argument("--success-hdf5", type=Path, default=DEFAULT_SUCCESS_HDF5)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--dataset-version", choices=["v1a", "v1b", "v1c"], default="v1b")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-transport", action="store_true")
    parser.add_argument("--no-grasp", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        if args.dataset_version == "v1a":
            output_dir = DEFAULT_OUTPUT_V1A
        elif args.dataset_version == "v1c":
            output_dir = DEFAULT_OUTPUT_V1C
        else:
            output_dir = DEFAULT_OUTPUT_V1B

    samples, summary = build_dataset(
        experiment_dir=args.experiment_dir,
        success_hdf5=args.success_hdf5,
        failed_hdf5=args.failed_hdf5,
        dataset_version=args.dataset_version,
        include_transport=not args.no_transport and args.dataset_version in ("v1b", "v1c"),
        include_grasp=not args.no_grasp and args.dataset_version == "v1c",
    )
    npz_path = save_dataset(samples, summary, output_dir)
    print(json.dumps({"npz_path": str(npz_path), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
