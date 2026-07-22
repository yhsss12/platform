"""Offline MimicGen Repair：带轨迹记录的 rollout（不修改 HDF5 object_poses）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_OFFLINE_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _OFFLINE_DIR.parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from grasp_sim_search import apply_grasp_step_overlay, compute_grasp_proxies, get_sim_nut_pos
from grasp_waypoint_builder import GraspSearchParams, build_grasp_waypoints_from_hdf5
from lift_sim_search import apply_lift_step_overlay
from lift_waypoint_refiner import LiftRepairParams, build_lift_waypoints_from_hdf5, lift_params_from_dict
from osc_action_converter import SimLoopParams, apply_sim_loop_step_overlay, compute_closed_loop_waypoint_action
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
from sim_in_loop_refiner import _json_safe_theta, build_refined_waypoints_from_hdf5, load_best_theta, load_best_theta_or_fallback, score_rollout_result


def _record_state(env) -> np.ndarray:
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64)


def execute_insertion_repair_rollout(
    *,
    failed_hdf5: str | Path,
    demo_key: str,
    cem_report: str | Path,
    insertion: dict[str, float],
    rollout_kind: str = "offline_mimicgen_repair",
    record_video: bool = False,
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    theta = load_best_theta_or_fallback(str(cem_report), demo_key)
    proxy, _, target_eef, shifted_gripper = build_refined_waypoints_from_hdf5(
        str(failed_hdf5), demo_key, "failed", theta, rollout_safe=True
    )
    sim_params = SimLoopParams(**insertion)
    demo = load_demo_rollout_data(str(failed_hdf5), demo_key, "failed")
    env_args = read_env_metadata(str(failed_hdf5))
    base_speed = float(theta.get("speed_scale", 1.0))

    build = create_env_from_metadata(env_args, for_video=record_video and video_path is not None)
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    frames: list[np.ndarray] = []
    if record_video and video_path is not None:
        frames.append(capture_camera_frame(env))

    length = len(target_eef)
    grip = np.asarray(shifted_gripper, dtype=float).reshape(-1)
    actions = np.zeros((length, 7), dtype=np.float64)
    states = np.zeros((length + 1, demo.states.shape[1]), dtype=np.float64)
    states[0] = _record_state(env)

    min_xy = float("inf")
    min_yaw = float("inf")
    for step in range(length):
        target_idx = min(step + 1, length - 1)
        action = compute_closed_loop_waypoint_action(
            env, target_eef[target_idx], grip[step], env_args, speed_scale=base_speed
        )
        action = apply_sim_loop_step_overlay(action, step, proxy, grip, env_args, sim_params)
        actions[step] = action
        env.step(action)
        states[step + 1] = _record_state(env)
        metrics = extract_sim_features(env)
        min_xy = min(min_xy, metrics["final_nut_peg_xy"])
        min_yaw = min(min_yaw, metrics["min_yaw_error"])
        if record_video and video_path is not None and step % 2 == 0:
            frames.append(capture_camera_frame(env))

    final_metrics = extract_sim_features(env)
    final_metrics["min_nut_peg_xy"] = min_xy
    final_metrics["min_yaw_error"] = min(min_yaw, final_metrics["min_yaw_error"])
    acc_mean, acc_max = action_acceleration_stats(actions)
    feat_dict = rollout_metrics_to_features_dict(
        demo_key, "failed", str(failed_hdf5), final_metrics, acc_max, len(actions)
    )
    features = NutAssemblyFeatures(**feat_dict)
    energy = compute_total_energy(features)
    success_flag = bool(env._check_success())
    env.close()

    if record_video and video_path is not None and frames:
        write_mp4(frames, video_path, fps=20)

    return {
        "demo_name": demo_key,
        "source_file": str(failed_hdf5),
        "rollout_kind": rollout_kind,
        "sim_params": sim_params.to_dict(),
        "theta": _json_safe_theta(theta),
        "success_flag": success_flag,
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "min_nut_peg_xy": final_metrics["min_nut_peg_xy"],
        "final_z_diff": final_metrics["final_z_diff"],
        "min_yaw_error": final_metrics["min_yaw_error"],
        "failure_guess": classify_failure_type(features, energy.E_smooth),
        "E_total_norm": energy.E_total_norm,
        "E_xy_norm": energy.E_xy_norm,
        "E_transport_norm": energy.E_transport_norm,
        "E_yaw_norm": energy.E_yaw_norm,
        "E_z_norm": energy.E_z_norm,
        "E_smooth_norm": energy.E_smooth_norm,
        "score": score_rollout_result({"E_total_norm": energy.E_total_norm}),
        "action_acceleration_max": acc_max,
        "num_steps": len(actions),
        "reset_info": reset_info,
        "env_warnings": build.warnings,
        "object_poses_modified": False,
        "recorded_actions": actions,
        "recorded_states": states,
        "recorded_eef_pose": np.asarray(target_eef, dtype=np.float64),
        "recorded_gripper_action": grip.reshape(-1, 1),
        "repair_insertion_params": insertion,
    }


def execute_grasp_repair_rollout(
    *,
    failed_hdf5: str | Path,
    demo_key: str,
    grasp_lift: dict[str, float],
    rollout_kind: str = "offline_mimicgen_repair",
    record_video: bool = False,
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    params = GraspSearchParams(
        grasp_xy_offset_x=grasp_lift["grasp_xy_offset_x"],
        grasp_xy_offset_y=grasp_lift["grasp_xy_offset_y"],
        pre_grasp_height=grasp_lift["pre_grasp_height"],
        approach_height=grasp_lift["approach_height"],
        gripper_close_shift=int(grasp_lift.get("reclose_after_contact", 0)),
        gripper_hold_steps=int(grasp_lift["gripper_hold_steps"]),
        lift_height=grasp_lift["micro_lift_height"],
        lift_steps=int(grasp_lift["lift_steps"]),
        speed_scale=grasp_lift["lift_speed_scale"],
    )
    proxy, _original_eef, target_eef, gripper = build_grasp_waypoints_from_hdf5(
        str(failed_hdf5), demo_key, "failed", params
    )
    demo = load_demo_rollout_data(str(failed_hdf5), demo_key, "failed")
    env_args = read_env_metadata(str(failed_hdf5))
    grasp_idx = proxy.phases.grasp_index

    build = create_env_from_metadata(env_args, for_video=record_video and video_path is not None)
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    length = len(target_eef)
    grip = np.asarray(gripper, dtype=float).reshape(-1)
    actions = np.zeros((length, 7), dtype=np.float64)
    states = np.zeros((length + 1, demo.states.shape[1]), dtype=np.float64)
    states[0] = _record_state(env)

    nut_positions: list[np.ndarray] = []
    nut_z_trace: list[float] = []
    eef_nut_distance_at_grasp = float("inf")
    gripper_closed_flags: list[float] = []
    nut_positions.append(get_sim_nut_pos(env).copy())

    min_xy = float("inf")
    min_yaw = float("inf")
    for step in range(length):
        target_idx = min(step + 1, length - 1)
        action = compute_closed_loop_waypoint_action(
            env, target_eef[target_idx], grip[step], env_args, speed_scale=float(params.speed_scale)
        )
        action = apply_grasp_step_overlay(action, step, proxy, params)
        actions[step] = action
        env.step(action)
        states[step + 1] = _record_state(env)

        nut_pos = get_sim_nut_pos(env)
        nut_positions.append(nut_pos.copy())
        nut_z_trace.append(float(nut_pos[2]))
        gripper_closed_flags.append(float(grip[step] < 0.0))
        if step == grasp_idx:
            eef_pos = get_sim_eef_pose4(env)[:3, 3]
            eef_nut_distance_at_grasp = float(np.linalg.norm(eef_pos - nut_pos))
        metrics = extract_sim_features(env)
        min_xy = min(min_xy, metrics["final_nut_peg_xy"])
        min_yaw = min(min_yaw, metrics["min_yaw_error"])

    nut_positions_arr = np.asarray(nut_positions)
    nut_displacement_total = (
        float(np.sum(np.linalg.norm(np.diff(nut_positions_arr, axis=0), axis=1)))
        if len(nut_positions_arr) > 1
        else 0.0
    )
    grasp_step = min(grasp_idx, len(nut_positions_arr) - 1)
    if grasp_step < len(nut_positions_arr) - 1:
        after = nut_positions_arr[grasp_step + 1 :]
        nut_displacement_after_grasp = (
            float(np.sum(np.linalg.norm(np.diff(after, axis=0), axis=1)))
            if len(after) > 1
            else float(np.linalg.norm(after[0] - nut_positions_arr[grasp_step]))
            if len(after) == 1
            else 0.0
        )
    else:
        nut_displacement_after_grasp = 0.0

    lift_start = min(len(nut_z_trace) - 1, grasp_idx + 1)
    lift_end = min(len(nut_z_trace) - 1, grasp_idx + int(params.lift_steps))
    nut_z_at_grasp = nut_z_trace[grasp_step] if grasp_step < len(nut_z_trace) else nut_z_trace[-1]
    nut_lift_delta = (
        float(max(nut_z_trace[lift_start : lift_end + 1]) - nut_z_at_grasp) if lift_end > lift_start else 0.0
    )
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
        demo_key, "failed", str(failed_hdf5), final_metrics, acc_max, len(actions)
    )
    features = NutAssemblyFeatures(**feat_dict)
    energy = compute_total_energy(features)
    success_flag = bool(env._check_success())
    env.close()

    return {
        "demo_name": demo_key,
        "source_file": str(failed_hdf5),
        "rollout_kind": rollout_kind,
        "grasp_params": params.to_dict(),
        "success_flag": success_flag,
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "min_nut_peg_xy": final_metrics["min_nut_peg_xy"],
        "final_z_diff": final_metrics["final_z_diff"],
        "min_yaw_error": final_metrics["min_yaw_error"],
        "failure_guess": classify_failure_type(features, energy.E_smooth),
        "E_total_norm": energy.E_total_norm,
        "E_xy_norm": energy.E_xy_norm,
        "E_transport_norm": energy.E_transport_norm,
        "E_yaw_norm": energy.E_yaw_norm,
        "E_z_norm": energy.E_z_norm,
        "E_smooth_norm": energy.E_smooth_norm,
        "nut_displacement_total": nut_displacement_total,
        "nut_displacement_after_grasp": nut_displacement_after_grasp,
        "nut_lift_delta": nut_lift_delta,
        "eef_nut_distance_at_grasp": eef_nut_distance_at_grasp,
        "grasp_success_proxy": proxies["grasp_success_proxy"],
        "lift_success_proxy": proxies["lift_success_proxy"],
        "object_poses_modified": False,
        "recorded_actions": actions,
        "recorded_states": states,
        "recorded_eef_pose": np.asarray(target_eef, dtype=np.float64),
        "recorded_gripper_action": grip.reshape(-1, 1),
        "repair_grasp_lift_params": grasp_lift,
    }


def execute_lift_repair_rollout(
    *,
    failed_hdf5: str | Path,
    demo_key: str,
    grasp_lift: dict[str, float],
    lift_extra: dict[str, float] | None = None,
    rollout_kind: str = "offline_mimicgen_repair",
) -> dict[str, Any]:
    merged = {**grasp_lift, **(lift_extra or {})}
    params = lift_params_from_dict(merged)
    proxy, _original_eef, target_eef, gripper = build_lift_waypoints_from_hdf5(
        str(failed_hdf5), demo_key, "failed", params
    )
    demo = load_demo_rollout_data(str(failed_hdf5), demo_key, "failed")
    env_args = read_env_metadata(str(failed_hdf5))
    grasp_idx = proxy.phases.grasp_index

    build = create_env_from_metadata(env_args, for_video=False)
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    length = len(target_eef)
    grip = np.asarray(gripper, dtype=float).reshape(-1)
    actions = np.zeros((length, 7), dtype=np.float64)
    states = np.zeros((length + 1, demo.states.shape[1]), dtype=np.float64)
    states[0] = _record_state(env)

    nut_positions: list[np.ndarray] = []
    nut_z_trace: list[float] = []
    eef_nut_distance_at_grasp = float("inf")
    nut_positions.append(get_sim_nut_pos(env).copy())

    min_xy = float("inf")
    min_yaw = float("inf")
    for step in range(length):
        target_idx = min(step + 1, length - 1)
        action = compute_closed_loop_waypoint_action(
            env, target_eef[target_idx], grip[step], env_args, speed_scale=float(params.lift_speed_scale)
        )
        action = apply_lift_step_overlay(action, step, proxy, params)
        actions[step] = action
        env.step(action)
        states[step + 1] = _record_state(env)

        nut_pos = get_sim_nut_pos(env)
        nut_positions.append(nut_pos.copy())
        nut_z_trace.append(float(nut_pos[2]))
        if step == grasp_idx:
            eef_pos = get_sim_eef_pose4(env)[:3, 3]
            eef_nut_distance_at_grasp = float(np.linalg.norm(eef_pos - nut_pos))
        metrics = extract_sim_features(env)
        min_xy = min(min_xy, metrics["final_nut_peg_xy"])
        min_yaw = min(min_yaw, metrics["min_yaw_error"])

    nut_positions_arr = np.asarray(nut_positions)
    grasp_step = min(grasp_idx, len(nut_positions_arr) - 1)
    if grasp_step < len(nut_positions_arr) - 1:
        after = nut_positions_arr[grasp_step + 1 :]
        nut_displacement_after_grasp = (
            float(np.sum(np.linalg.norm(np.diff(after, axis=0), axis=1))) if len(after) > 1 else 0.0
        )
    else:
        nut_displacement_after_grasp = 0.0

    settle_end = grasp_idx + int(params.post_grasp_settle_steps)
    lift_start = min(len(nut_z_trace) - 1, settle_end + int(params.lift_pause_steps) + 1)
    lift_end = min(len(nut_z_trace) - 1, lift_start + int(params.micro_lift_steps))
    nut_z_at_grasp = nut_z_trace[grasp_step] if grasp_step < len(nut_z_trace) else nut_z_trace[-1]
    nut_lift_delta = (
        float(max(nut_z_trace[lift_start : lift_end + 1]) - nut_z_at_grasp) if lift_end > lift_start else 0.0
    )
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
        demo_key, "failed", str(failed_hdf5), final_metrics, acc_max, len(actions)
    )
    features = NutAssemblyFeatures(**feat_dict)
    energy = compute_total_energy(features)
    success_flag = bool(env._check_success())
    env.close()

    return {
        "demo_name": demo_key,
        "source_file": str(failed_hdf5),
        "rollout_kind": rollout_kind,
        "lift_params": params.to_dict(),
        "success_flag": success_flag,
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "min_nut_peg_xy": final_metrics["min_nut_peg_xy"],
        "final_z_diff": final_metrics["final_z_diff"],
        "min_yaw_error": final_metrics["min_yaw_error"],
        "failure_guess": classify_failure_type(features, energy.E_smooth),
        "E_total_norm": energy.E_total_norm,
        "E_xy_norm": energy.E_xy_norm,
        "E_transport_norm": energy.E_transport_norm,
        "E_yaw_norm": energy.E_yaw_norm,
        "E_z_norm": energy.E_z_norm,
        "E_smooth_norm": energy.E_smooth_norm,
        "nut_displacement_after_grasp": nut_displacement_after_grasp,
        "nut_lift_delta": nut_lift_delta,
        "eef_nut_distance_at_grasp": eef_nut_distance_at_grasp,
        "grasp_success_proxy": proxies["grasp_success_proxy"],
        "lift_success_proxy": proxies["lift_success_proxy"],
        "object_poses_modified": False,
        "recorded_actions": actions,
        "recorded_states": states,
        "recorded_eef_pose": np.asarray(target_eef, dtype=np.float64),
        "recorded_gripper_action": grip.reshape(-1, 1),
        "repair_grasp_lift_params": grasp_lift,
        "repair_lift_extra_params": lift_extra or {},
    }


def run_repair_rollout(
    *,
    failed_hdf5: Path,
    demo_key: str,
    search_kind: str,
    cem_report: Path | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if search_kind == "insertion":
        if cem_report is None:
            raise ValueError("cem_report required for insertion repair")
        return execute_insertion_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            cem_report=cem_report,
            insertion=candidate["insertion"],
        )
    if search_kind == "transport":
        from transport_sim_search import execute_transport_rollout
        from transport_waypoint_builder import TransportSearchParams

        if cem_report is None:
            raise ValueError("cem_report required for transport repair")
        theta = load_best_theta_or_fallback(str(cem_report), demo_key, fallback_demo_key="demo_0")
        params = TransportSearchParams(**candidate.get("transport", {}))
        return execute_transport_rollout(
            str(failed_hdf5),
            demo_key,
            "failed",
            theta,
            params,
            rollout_kind="offline_mimicgen_transport_repair",
        )
    if search_kind == "lift":
        return execute_lift_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            grasp_lift=candidate["grasp_lift"],
            lift_extra=candidate.get("lift_extra"),
        )
    return execute_grasp_repair_rollout(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        grasp_lift=candidate["grasp_lift"],
    )
