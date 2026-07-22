"""V1-F：lift_failed demo sim-in-loop rollout（lift-aware refiner）。"""
from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from grasp_sim_search import compute_grasp_proxies, get_sim_nut_pos
from lift_contact_diagnostics import LiftContactTracker
from lift_contact_energy_model import compute_contact_aware_lift_energies
from lift_energy_model import compute_lift_residual_energies, merge_v1f_energy_targets
from lift_waypoint_refiner import (
    LIFT_REPAIR_SEARCH_SPACE,
    LiftRepairParams,
    build_lift_waypoints_from_hdf5,
)
from osc_action_converter import SimLoopParams, compute_closed_loop_waypoint_action
from robosuite_env_loader import (
    capture_camera_frame,
    create_env_from_metadata,
    extract_sim_features,
    get_sim_eef_pose4,
    load_demo_rollout_data,
    read_env_metadata,
    reset_env_to_demo_state,
    rollout_metrics_to_features_dict,
    write_mp4,
)


def iter_lift_candidates(
    *,
    mode: str = "random",
    max_evals: int = 80,
    seed: int = 0,
) -> Iterator[LiftRepairParams]:
    keys = list(LIFT_REPAIR_SEARCH_SPACE.keys())
    if mode == "grid":
        combos = list(itertools.product(*(LIFT_REPAIR_SEARCH_SPACE[k] for k in keys)))
        rng = random.Random(seed)
        if len(combos) > max_evals:
            combos = rng.sample(combos, max_evals)
        for combo in combos:
            yield LiftRepairParams(**dict(zip(keys, combo)))
    else:
        rng = random.Random(seed)
        for _ in range(max_evals):
            yield LiftRepairParams(**{k: rng.choice(LIFT_REPAIR_SEARCH_SPACE[k]) for k in keys})


def apply_lift_step_overlay(
    action: np.ndarray,
    step: int,
    proxy: Any,
    params: LiftRepairParams,
) -> np.ndarray:
    out = action.copy()
    grasp_idx = proxy.phases.grasp_index
    settle_end = grasp_idx + int(params.post_grasp_settle_steps)
    lift_start = settle_end + int(params.lift_pause_steps) + 1
    lift_end = min(proxy.length - 1, lift_start + int(params.micro_lift_steps))

    if grasp_idx - 5 <= step <= grasp_idx + int(params.contact_hold_steps):
        out[2] *= 1.15 + float(params.gripper_extra_close) * 0.5
    if lift_start <= step <= lift_end:
        out[2] *= 1.25 + float(params.lift_direction_bias) * 5.0
        out[2] *= float(params.lift_speed_scale)
    return np.clip(out, -1.0, 1.0)


def execute_lift_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: LiftRepairParams,
    *,
    rollout_kind: str = "lift_repair",
    video_path: str | Path | None = None,
    record_video: bool = False,
    control_freq: int = 20,
) -> dict[str, Any]:
    proxy, _original_eef, target_eef, gripper = build_lift_waypoints_from_hdf5(
        hdf5_path, demo_key, label, params
    )
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    env_args = read_env_metadata(hdf5_path)
    base_speed = float(params.lift_speed_scale)
    grasp_idx = proxy.phases.grasp_index

    build = create_env_from_metadata(
        env_args,
        for_video=record_video and video_path is not None,
    )
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)
    contact_tracker = LiftContactTracker(env) if demo_key == "demo_3" else None

    frames: list[np.ndarray] = []
    if record_video and video_path is not None:
        frames.append(capture_camera_frame(env))

    length = len(target_eef)
    grip = np.asarray(gripper, dtype=float).reshape(-1)
    actions = np.zeros((length, 7), dtype=float)
    min_xy = float("inf")
    min_yaw = float("inf")

    nut_positions: list[np.ndarray] = []
    nut_z_trace: list[float] = []
    eef_nut_distance_at_grasp = float("inf")
    gripper_closed_flags: list[float] = []
    lift_window_distances: list[float] = []

    initial_nut = get_sim_nut_pos(env)
    nut_positions.append(initial_nut.copy())

    lift_start = min(length - 1, grasp_idx + int(params.post_grasp_settle_steps) + int(params.lift_pause_steps) + 1)
    lift_end = min(length - 1, lift_start + int(params.micro_lift_steps))
    contact_window_end = min(length - 1, grasp_idx + int(params.contact_hold_steps))

    for step in range(length):
        target_idx = min(step + 1, length - 1)
        action = compute_closed_loop_waypoint_action(
            env,
            target_eef[target_idx],
            grip[step],
            env_args,
            speed_scale=base_speed,
        )
        action = apply_lift_step_overlay(action, step, proxy, params)
        actions[step] = action
        env.step(action)

        nut_pos = get_sim_nut_pos(env)
        nut_positions.append(nut_pos.copy())
        nut_z_trace.append(float(nut_pos[2]))
        gripper_closed_flags.append(float(grip[step] < 0.0))

        eef_pos = get_sim_eef_pose4(env)[:3, 3]
        eef_nut_dist = float(np.linalg.norm(eef_pos - nut_pos))
        if step == grasp_idx:
            eef_nut_distance_at_grasp = eef_nut_dist
        if lift_start <= step <= lift_end:
            lift_window_distances.append(eef_nut_dist)
        if contact_tracker is not None:
            contact_tracker.observe_step(
                env,
                step=step,
                grasp_idx=grasp_idx,
                lift_begin=lift_start,
                lift_end=lift_end,
                contact_window_end=contact_window_end,
            )

        metrics = extract_sim_features(env)
        min_xy = min(min_xy, metrics["final_nut_peg_xy"])
        min_yaw = min(min_yaw, metrics["min_yaw_error"])
        if record_video and video_path is not None and step % 2 == 0:
            frames.append(capture_camera_frame(env))

    nut_positions_arr = np.asarray(nut_positions)
    if len(nut_positions_arr) > 1:
        nut_displacement_total = float(
            np.sum(np.linalg.norm(np.diff(nut_positions_arr, axis=0), axis=1))
        )
    else:
        nut_displacement_total = 0.0

    grasp_step = min(grasp_idx, len(nut_positions_arr) - 1)
    if grasp_step < len(nut_positions_arr) - 1:
        after = nut_positions_arr[grasp_step + 1 :]
        if len(after) > 1:
            nut_displacement_after_grasp = float(
                np.sum(np.linalg.norm(np.diff(after, axis=0), axis=1))
            )
        elif len(after) == 1:
            nut_displacement_after_grasp = float(
                np.linalg.norm(after[0] - nut_positions_arr[grasp_step])
            )
        else:
            nut_displacement_after_grasp = 0.0
    else:
        nut_displacement_after_grasp = 0.0

    nut_z_at_grasp = nut_z_trace[grasp_step] if grasp_step < len(nut_z_trace) else nut_z_trace[-1]
    if lift_end > lift_start and lift_start < len(nut_z_trace):
        nut_lift_delta = float(max(nut_z_trace[lift_start : lift_end + 1]) - nut_z_at_grasp)
        nut_z_during_lift = nut_z_trace[lift_start : lift_end + 1]
        nut_z_std_during_lift = float(np.std(nut_z_during_lift)) if len(nut_z_during_lift) > 1 else 0.0
    else:
        nut_lift_delta = 0.0
        nut_z_std_during_lift = 0.0

    min_eef_nut_distance = eef_nut_distance_at_grasp
    if grasp_step < len(nut_positions_arr):
        for step in range(grasp_step, len(nut_positions_arr)):
            eef_step = target_eef[min(step, len(target_eef) - 1), :3, 3]
            min_eef_nut_distance = min(
                min_eef_nut_distance,
                float(np.linalg.norm(eef_step - nut_positions_arr[step])),
            )

    follow_thresh = float(params.nut_follow_threshold)
    if lift_window_distances:
        mean_lift_dist = float(np.mean(lift_window_distances))
        lift_follow_score = float(np.clip(1.0 - mean_lift_dist / max(follow_thresh, 1e-6), 0.0, 1.0))
    else:
        lift_follow_score = 0.0

    proxies = compute_grasp_proxies(
        nut_displacement_after_grasp=nut_displacement_after_grasp,
        nut_lift_delta=nut_lift_delta,
        eef_nut_distance_at_grasp=eef_nut_distance_at_grasp,
    )

    final_metrics = extract_sim_features(env)
    final_metrics["min_nut_peg_xy"] = min_xy
    final_metrics["min_yaw_error"] = min(min_yaw, final_metrics["min_yaw_error"])

    acc_mean, acc_max = action_acceleration_stats(actions)
    feat_dict = rollout_metrics_to_features_dict(
        demo_key,
        label,
        hdf5_path,
        final_metrics,
        acc_max,
        len(actions),
    )
    features = NutAssemblyFeatures(**feat_dict)
    energy = compute_total_energy(features)
    failure_guess = classify_failure_type(features, energy.E_smooth)
    success_flag = bool(env._check_success())

    result: dict[str, Any] = {
        "demo_name": demo_key,
        "label": label,
        "source_file": hdf5_path,
        "rollout_kind": rollout_kind,
        "lift_params": params.to_dict(),
        "grasp_params": params.to_dict(),
        "sim_params": SimLoopParams().to_dict(),
        "success_flag": success_flag,
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "min_nut_peg_xy": final_metrics["min_nut_peg_xy"],
        "final_z_diff": final_metrics["final_z_diff"],
        "min_yaw_error": final_metrics["min_yaw_error"],
        "nut_displacement_total": nut_displacement_total,
        "nut_displacement_after_grasp": nut_displacement_after_grasp,
        "eef_nut_distance_at_grasp": eef_nut_distance_at_grasp,
        "min_eef_nut_distance": min_eef_nut_distance,
        "nut_lift_delta": nut_lift_delta,
        "nut_z_std_during_lift": nut_z_std_during_lift,
        "lift_follow_score": lift_follow_score,
        "micro_lift_height": float(params.micro_lift_height),
        "nut_follow_threshold": follow_thresh,
        "target_micro_lift_height": float(params.micro_lift_height),
        "grasp_success_proxy": proxies["grasp_success_proxy"],
        "lift_success_proxy": proxies["lift_success_proxy"],
        "failure_guess": failure_guess,
        "E_total_norm": energy.E_total_norm,
        "E_xy_norm": energy.E_xy_norm,
        "E_transport_norm": energy.E_transport_norm,
        "E_yaw_norm": energy.E_yaw_norm,
        "E_z_norm": energy.E_z_norm,
        "E_smooth_norm": energy.E_smooth_norm,
        "action_acceleration_max": acc_max,
        "num_steps": len(actions),
        "reset_info": reset_info,
        "env_warnings": build.warnings,
        "object_poses_modified": False,
        "video_path": str(video_path) if video_path else None,
    }
    result.update(compute_lift_residual_energies(result))
    result.update(merge_v1f_energy_targets(result))
    if contact_tracker is not None:
        contact_diag = contact_tracker.finalize(
            env,
            lift_begin=lift_start,
            lift_end=lift_end,
            nut_z_trace=nut_z_trace,
            partial_lift_delta_thresh=0.005,
        )
        result.update(contact_diag.to_dict())
        result.update(compute_contact_aware_lift_energies(result))
        result["lift_diagnostics_enabled"] = True

    if record_video and video_path is not None and frames:
        write_mp4(frames, video_path, fps=control_freq)
        result["video_path"] = str(video_path)

    env.close()
    return result
