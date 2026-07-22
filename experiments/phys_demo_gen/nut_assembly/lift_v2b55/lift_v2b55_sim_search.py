"""V2-B5.5 pre-lift reclose + slow vertical lift rollout."""
from __future__ import annotations

from typing import Any

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from grasp_sim_search import compute_grasp_proxies, get_sim_nut_pos
from lift_contact_diagnostics import LiftContactTracker
from lift_contact_energy_model import compute_contact_aware_lift_energies
from lift_energy_model import compute_lift_residual_energies
from lift_v2b51_sim_search import classify_lift_outcome
from lift_v2b54_sim_search import apply_lift_v2b54_step_overlay
from lift_v2b55_objective import compute_prelift_slow_lift_score
from lift_v2b55_refiner import LiftV2B55Params, build_lift_v2b55_waypoints_from_hdf5
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

PARTIAL_THRESH = 0.005
LIFT_SUCCESS_THRESH = 0.02


def apply_lift_v2b55_step_overlay(
    action: np.ndarray,
    step: int,
    params: LiftV2B55Params,
    phases: dict[str, int],
    *,
    stage1_lift_delta: float = 0.0,
    nut_z_lift_so_far: float = 0.0,
) -> np.ndarray:
    out = apply_lift_v2b54_step_overlay(
        action, step, params, phases,
        stage1_lift_delta=stage1_lift_delta,
        nut_z_lift_so_far=nut_z_lift_so_far,
    )
    lift_begin = phases["lift_begin"]
    stage2_end = phases["stage2_end"]
    second_end = phases.get("second_reclose_end", lift_begin)
    vert_end = phases.get("vertical_only_lift_end", lift_begin)
    slow_end = phases.get("slow_vertical_lift_end", vert_end)
    weak_gate = float(phases.get("weak_lift_gate_m", params.weak_lift_before_transport_m))

    if lift_begin <= step <= second_end:
        out[6] = min(out[6], -0.98 - float(params.second_reclose_strength) * 0.1)

    if lift_begin <= step <= slow_end:
        out[0] *= 0.05
        out[1] *= 0.05
        out[2] *= 1.45
        out[2] *= max(0.2, min(float(params.lift_speed_scale), 0.1))
        out[6] = min(out[6], -0.99)

    if lift_begin <= step <= vert_end:
        out[0] = 0.0
        out[1] = 0.0

    transport_start = phases.get("transport_start", stage2_end)
    if step >= transport_start and nut_z_lift_so_far < weak_gate:
        out[0] *= 0.02
        out[1] *= 0.02
        out[2] *= 1.3

    return np.clip(out, -1.0, 1.0)


def execute_lift_v2b55_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: LiftV2B55Params,
    *,
    rollout_kind: str = "lift_v2b55_cem",
) -> dict[str, Any]:
    proxy, _orig, target_eef, gripper, phases = build_lift_v2b55_waypoints_from_hdf5(
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
    lift_dists: list[float] = []
    eef_nut_at_grasp = float("inf")
    min_xy = float("inf")
    min_yaw = float("inf")
    stage1_end = phases["stage1_end"]
    nut_z_at_grasp = 0.0

    for step in range(length):
        nut_z_before = float(get_sim_nut_pos(env)[2]) if step == stage1_end else 0.0
        action = compute_closed_loop_waypoint_action(
            env, target_eef[min(step + 1, length - 1)], grip[step], env_args,
            speed_scale=min(float(params.lift_speed_scale), 0.1),
        )
        s1_delta = 0.0
        if step == stage1_end and lift_begin < len(nut_z_trace):
            s1_delta = float(nut_z_before - nut_z_trace[lift_begin])
        nz_so_far = 0.0
        if nut_z_trace and grasp_idx < len(nut_z_trace):
            nz_so_far = max(nut_z_trace[grasp_idx:]) - nut_z_trace[grasp_idx]
        action = apply_lift_v2b55_step_overlay(
            action, step, params, phases, stage1_lift_delta=s1_delta, nut_z_lift_so_far=nz_so_far
        )
        actions[step] = action
        env.step(action)

        tracker.observe_step(env, step=step, grasp_idx=grasp_idx, lift_begin=lift_begin,
                             lift_end=stage2_end, contact_window_end=contact_window_end)
        nut_pos = get_sim_nut_pos(env)
        nut_z_trace.append(float(nut_pos[2]))
        if step == grasp_idx:
            nut_z_at_grasp = float(nut_pos[2])
            eef_nut_at_grasp = float(np.linalg.norm(get_sim_eef_pose4(env)[:3, 3] - nut_pos))
        if lift_begin <= step <= stage2_end:
            lift_dists.append(float(np.linalg.norm(get_sim_eef_pose4(env)[:3, 3] - nut_pos)))
        m = extract_sim_features(env)
        min_xy = min(min_xy, float(m["final_nut_peg_xy"]))
        min_yaw = min(min_yaw, float(m["min_yaw_error"]))

    gs = min(grasp_idx, len(nut_z_trace) - 1)
    nut_z_at_grasp = nut_z_trace[gs] if gs < len(nut_z_trace) else 0.0
    le = min(len(nut_z_trace) - 1, stage2_end)
    lb = min(max(0, lift_begin), len(nut_z_trace) - 1)
    if le > lb:
        w = nut_z_trace[lb : le + 1]
        nut_lift_delta = float(max(w) - nut_z_at_grasp)
        nut_phase_delta = float(max(w) - nut_z_trace[lb])
    else:
        nut_lift_delta = nut_phase_delta = 0.0
    nut_z_std = float(np.std(nut_z_trace[lift_begin : le + 1])) if le > lb else 0.0
    follow = float(np.clip(1.0 - np.mean(lift_dists) / max(float(params.nut_follow_threshold), 1e-6), 0, 1)) if lift_dists else 0.0

    contact_diag = tracker.finalize(env, lift_begin=lift_begin, lift_end=stage2_end,
                                    nut_z_trace=nut_z_trace, partial_lift_delta_thresh=PARTIAL_THRESH)
    proxies = compute_grasp_proxies(
        nut_displacement_after_grasp=float(contact_diag.nut_xy_slip),
        nut_lift_delta=nut_lift_delta,
        eef_nut_distance_at_grasp=eef_nut_at_grasp,
    )
    partial = bool(contact_diag.partial_lift_success) or nut_phase_delta >= PARTIAL_THRESH

    final_metrics = extract_sim_features(env)
    final_metrics["min_nut_peg_xy"] = min_xy if min_xy != float("inf") else final_metrics["final_nut_peg_xy"]
    final_metrics["min_yaw_error"] = min(min_yaw, final_metrics["min_yaw_error"])
    acc_mean, acc_max = action_acceleration_stats(actions)
    feat = NutAssemblyFeatures(**rollout_metrics_to_features_dict(demo_key, label, hdf5_path, final_metrics, acc_max, len(actions)))
    energy = compute_total_energy(feat)
    success_flag = bool(env._check_success())
    env.close()

    result: dict[str, Any] = {
        "demo_name": demo_key, "label": label, "source_file": hdf5_path,
        "rollout_kind": rollout_kind, "lift_v2b55_params": params.to_dict(),
        "training_eligible": False, "exclude_from_lift_success_training": True,
        "success_flag": success_flag, "partial_lift_success": partial,
        "lift_success_proxy": nut_phase_delta >= LIFT_SUCCESS_THRESH or bool(proxies["lift_success_proxy"]),
        "grasp_success_proxy": proxies["grasp_success_proxy"],
        "nut_lift_delta": nut_lift_delta, "nut_lift_phase_delta": nut_phase_delta,
        "nut_z_std_during_lift": nut_z_std, "lift_follow_score": follow,
        "eef_nut_distance_at_grasp": eef_nut_at_grasp,
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "min_nut_peg_xy": final_metrics["min_nut_peg_xy"],
        "final_z_diff": final_metrics["final_z_diff"],
        "failure_guess": classify_failure_type(feat, energy.E_smooth),
        "E_total_norm": energy.E_total_norm, "E_xy_norm": energy.E_xy_norm,
        "E_transport_norm": energy.E_transport_norm, "object_poses_modified": False,
        "reset_info": reset_info, "env_warnings": build.warnings,
    }
    result.update(contact_diag.to_dict())
    result.update(compute_lift_residual_energies(result))
    result.update(compute_contact_aware_lift_energies(result))
    result["outcome_label"] = classify_lift_outcome(result)
    result.update(compute_prelift_slow_lift_score(result))
    return result
