"""V2-B5.1：contact-aware demo_3 lift rollout + 搜索。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from grasp_sim_search import compute_grasp_proxies, get_sim_nut_pos
from lift_contact_diagnostics import LiftContactTracker
from lift_contact_energy_model import compute_contact_aware_lift_energies
from lift_energy_model import compute_lift_residual_energies
from lift_v2b51_refiner import (
    CONTACT_AWARE_SEED_PARAMS,
    LIFT_V2B51_SEARCH_SPACE,
    LiftV2B51Params,
    build_lift_v2b51_waypoints_from_hdf5,
)
from osc_action_converter import compute_closed_loop_waypoint_action
from robosuite_env_loader import (
    create_env_from_metadata,
    extract_sim_features,
    get_sim_eef_pose4,
    load_demo_rollout_data,
    read_env_metadata,
    reset_env_to_demo_state,
    rollout_metrics_to_features_dict,
)

PARTIAL_LIFT_DELTA_THRESH = 0.005
LIFT_SUCCESS_DELTA_THRESH = 0.02


def _perturb_params(base: LiftV2B51Params, rng: random.Random) -> LiftV2B51Params:
    raw = base.to_dict()
    for key, choices in LIFT_V2B51_SEARCH_SPACE.items():
        if rng.random() < 0.6:
            raw[key] = rng.choice(choices)
    return LiftV2B51Params(**raw)


def iter_lift_v2b51_candidates(
    *, max_evals: int, seed: int, local_fraction: float = 0.65
) -> Iterator[LiftV2B51Params]:
    rng = random.Random(seed)
    keys = list(LIFT_V2B51_SEARCH_SPACE.keys())
    n_local = int(max_evals * local_fraction)
    for i in range(max_evals):
        if i < n_local:
            yield _perturb_params(CONTACT_AWARE_SEED_PARAMS, rng)
        else:
            yield LiftV2B51Params(**{k: rng.choice(LIFT_V2B51_SEARCH_SPACE[k]) for k in keys})


def classify_lift_outcome(result: dict[str, Any]) -> str:
    if result.get("success_flag"):
        return "refined_success"
    if result.get("lift_success_proxy"):
        return "lift_success"
    if result.get("partial_lift_success"):
        return "partial_lift_success"
    if result.get("grasp_success_proxy"):
        return "grasp_improved_but_failed"
    if int(result.get("bilateral_contact_steps", 0)) > 0:
        return "contact_but_no_lift"
    return "lift_no_improvement"


def apply_lift_v2b51_step_overlay(
    action: np.ndarray,
    step: int,
    params: LiftV2B51Params,
    phases: dict[str, int],
    *,
    stage1_lift_delta: float = 0.0,
) -> np.ndarray:
    out = action.copy()
    enabled = params.enabled_templates()
    grasp_idx = phases["grasp_index"]
    lift_begin = phases["lift_begin"]
    stage2_end = phases["stage2_end"]
    contact_end = phases.get("contact_window_end", grasp_idx + int(params.contact_settle_steps))
    pause_end = phases.get("pause_end", phases["stage1_end"])

    if grasp_idx - 10 <= step <= contact_end:
        squeeze = float(params.squeeze_close_gain) if "squeeze_close" in enabled else 0.0
        out[2] *= 1.08 + abs(float(params.gripper_extra_close)) * 0.6 + squeeze * 0.15
        out[6] = min(out[6], -0.90 - abs(float(params.gripper_extra_close)) - squeeze * 0.08)

    if lift_begin <= step <= stage2_end:
        speed = float(params.lift_speed_scale)
        if "slow_lift" in enabled:
            speed = min(speed, 0.30)
        out[2] *= 1.30 + float(params.lift_direction_bias_z) * 6.0
        out[2] *= max(0.35, speed)
        out[6] = min(out[6], -0.99)

    if (
        "two_stage_lift" in enabled
        and float(params.enable_two_stage_lift) > 0.5
        and step > pause_end
        and stage1_lift_delta < float(params.micro_lift_check_threshold)
    ):
        out[2] *= 1.40

    return np.clip(out, -1.0, 1.0)


def execute_lift_v2b51_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: LiftV2B51Params,
    *,
    rollout_kind: str = "lift_v2b51",
    partial_lift_delta_thresh: float = PARTIAL_LIFT_DELTA_THRESH,
) -> dict[str, Any]:
    proxy, _orig, target_eef, gripper, phases = build_lift_v2b51_waypoints_from_hdf5(
        hdf5_path, demo_key, label, params
    )
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    env_args = read_env_metadata(hdf5_path)
    grasp_idx = phases["grasp_index"]
    lift_begin = phases["lift_begin"]
    stage2_end = phases["stage2_end"]
    contact_window_end = phases.get("contact_window_end", grasp_idx + int(params.contact_settle_steps))

    build = create_env_from_metadata(env_args, for_video=False)
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)
    tracker = LiftContactTracker(env)

    length = len(target_eef)
    grip = np.asarray(gripper, dtype=float).reshape(-1)
    actions = np.zeros((length, 7), dtype=float)
    nut_z_trace: list[float] = []
    nut_positions: list[np.ndarray] = []
    lift_window_distances: list[float] = []
    eef_nut_distance_at_grasp = float("inf")

    nut_positions.append(get_sim_nut_pos(env).copy())
    stage1_end = phases["stage1_end"]

    for step in range(length):
        nut_z_before_stage1 = float(get_sim_nut_pos(env)[2]) if step == stage1_end else 0.0
        target_idx = min(step + 1, length - 1)
        action = compute_closed_loop_waypoint_action(
            env,
            target_eef[target_idx],
            grip[step],
            env_args,
            speed_scale=float(params.lift_speed_scale),
        )
        stage1_lift_delta = 0.0
        if step == stage1_end and lift_begin < len(nut_z_trace):
            nut_z_at_lift = nut_z_trace[lift_begin] if lift_begin < len(nut_z_trace) else nut_z_trace[-1]
            stage1_lift_delta = float(nut_z_before_stage1 - nut_z_at_lift)
        action = apply_lift_v2b51_step_overlay(
            action, step, params, phases, stage1_lift_delta=stage1_lift_delta
        )
        actions[step] = action
        env.step(action)

        tracker.observe_step(
            env,
            step=step,
            grasp_idx=grasp_idx,
            lift_begin=lift_begin,
            lift_end=stage2_end,
            contact_window_end=contact_window_end,
        )

        nut_pos = get_sim_nut_pos(env)
        nut_positions.append(nut_pos.copy())
        nut_z_trace.append(float(nut_pos[2]))
        if step == grasp_idx:
            eef_pos = get_sim_eef_pose4(env)[:3, 3]
            eef_nut_distance_at_grasp = float(np.linalg.norm(eef_pos - nut_pos))
        if lift_begin <= step <= stage2_end:
            eef_pos = get_sim_eef_pose4(env)[:3, 3]
            lift_window_distances.append(float(np.linalg.norm(eef_pos - nut_pos)))

    nut_positions_arr = np.asarray(nut_positions)
    grasp_step = min(grasp_idx, len(nut_positions_arr) - 1)
    if grasp_step < len(nut_positions_arr) - 1:
        after = nut_positions_arr[grasp_step + 1 :]
        nut_displacement_after_grasp = (
            float(np.sum(np.linalg.norm(np.diff(after, axis=0), axis=1))) if len(after) > 1 else 0.0
        )
    else:
        nut_displacement_after_grasp = 0.0

    nut_z_at_grasp = nut_z_trace[grasp_step] if grasp_step < len(nut_z_trace) else nut_z_trace[-1]
    lift_end = min(len(nut_z_trace) - 1, stage2_end)
    lift_begin_idx = min(max(0, lift_begin), len(nut_z_trace) - 1)
    nut_z_at_lift_begin = nut_z_trace[lift_begin_idx]
    if lift_end > lift_begin_idx:
        lift_z_window = nut_z_trace[lift_begin_idx : lift_end + 1]
        nut_lift_delta = float(max(lift_z_window) - nut_z_at_grasp)
        nut_lift_phase_delta = float(max(lift_z_window) - nut_z_at_lift_begin)
    else:
        nut_lift_delta = 0.0
        nut_lift_phase_delta = 0.0
    nut_z_std = float(np.std(nut_z_trace[lift_begin : lift_end + 1])) if lift_end > lift_begin else 0.0

    follow_thresh = float(params.nut_follow_threshold)
    lift_follow_score = (
        float(np.clip(1.0 - float(np.mean(lift_window_distances)) / max(follow_thresh, 1e-6), 0.0, 1.0))
        if lift_window_distances
        else 0.0
    )

    contact_diag = tracker.finalize(
        env,
        lift_begin=lift_begin,
        lift_end=stage2_end,
        nut_z_trace=nut_z_trace,
        partial_lift_delta_thresh=partial_lift_delta_thresh,
    )

    proxies = compute_grasp_proxies(
        nut_displacement_after_grasp=nut_displacement_after_grasp,
        nut_lift_delta=nut_lift_delta,
        eef_nut_distance_at_grasp=eef_nut_distance_at_grasp,
    )
    partial_lift_success = bool(contact_diag.partial_lift_success) or nut_lift_phase_delta >= partial_lift_delta_thresh
    lift_success_proxy = nut_lift_phase_delta >= LIFT_SUCCESS_DELTA_THRESH or bool(proxies["lift_success_proxy"])

    final_metrics = extract_sim_features(env)
    acc_mean, acc_max = action_acceleration_stats(actions)
    feat_dict = rollout_metrics_to_features_dict(
        demo_key, label, hdf5_path, final_metrics, acc_max, len(actions)
    )
    features = NutAssemblyFeatures(**feat_dict)
    energy = compute_total_energy(features)
    success_flag = bool(env._check_success())
    env.close()

    target_lift = float(params.micro_lift_height_stage1 + params.micro_lift_height_stage2)
    result: dict[str, Any] = {
        "demo_name": demo_key,
        "label": label,
        "source_file": hdf5_path,
        "rollout_kind": rollout_kind,
        "lift_v2b51_params": params.to_dict(),
        "enabled_templates": sorted(params.enabled_templates()),
        "success_flag": success_flag,
        "outcome_label": "",
        "partial_lift_success": partial_lift_success,
        "lift_success_proxy": lift_success_proxy,
        "grasp_success_proxy": proxies["grasp_success_proxy"],
        "nut_lift_delta": nut_lift_delta,
        "nut_lift_phase_delta": nut_lift_phase_delta,
        "nut_z_std_during_lift": nut_z_std,
        "lift_follow_score": lift_follow_score,
        "nut_displacement_after_grasp": nut_displacement_after_grasp,
        "eef_nut_distance_at_grasp": eef_nut_distance_at_grasp,
        "target_micro_lift_height": target_lift,
        "lift_speed_scale": float(params.lift_speed_scale),
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "min_nut_peg_xy": final_metrics.get("min_nut_peg_xy", final_metrics["final_nut_peg_xy"]),
        "final_z_diff": final_metrics["final_z_diff"],
        "failure_guess": classify_failure_type(features, energy.E_smooth),
        "E_total_norm": energy.E_total_norm,
        "E_xy_norm": energy.E_xy_norm,
        "E_transport_norm": energy.E_transport_norm,
        "E_yaw_norm": energy.E_yaw_norm,
        "E_z_norm": energy.E_z_norm,
        "E_smooth_norm": energy.E_smooth_norm,
        "object_poses_modified": False,
        "reset_info": reset_info,
        "env_warnings": build.warnings,
    }
    result.update(contact_diag.to_dict())
    result.update(compute_lift_residual_energies(result))
    result.update(compute_contact_aware_lift_energies(result))
    result["outcome_label"] = classify_lift_outcome(result)
    return result
