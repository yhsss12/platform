"""V2-B2：在 RoboSuite 中 rollout refined theta 轨迹。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from osc_action_converter import build_refined_actions
from refined_waypoint_builder import build_refined_waypoints_from_hdf5
from robosuite_env_loader import (
    capture_camera_frame,
    create_env_from_metadata,
    extract_sim_features,
    load_demo_rollout_data,
    read_env_metadata,
    reset_env_to_demo_state,
    rollout_metrics_to_features_dict,
    write_mp4,
)


def rollout_refined_demo(
    hdf5_path: str,
    demo_key: str,
    label: str,
    theta: dict[str, Any],
    *,
    video_path: str | Path | None = None,
    camera_name: str = "agentview",
    control_freq: int = 20,
    record_video: bool = True,
) -> dict[str, Any]:
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    proxy, original_eef, refined_eef, shifted_gripper = build_refined_waypoints_from_hdf5(
        hdf5_path,
        demo_key,
        label,
        theta,
    )
    env_args = read_env_metadata(hdf5_path)
    refined_actions = build_refined_actions(
        proxy,
        original_eef,
        refined_eef,
        demo.actions,
        shifted_gripper,
        proxy.phases,
        env_args,
        theta,
    )

    build = create_env_from_metadata(
        env_args,
        for_video=record_video and video_path is not None,
        camera_name=camera_name,
    )
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    frames: list[np.ndarray] = []
    if record_video and video_path is not None:
        frames.append(capture_camera_frame(env, camera_name))

    min_xy = float("inf")
    min_yaw = float("inf")
    for step, action in enumerate(refined_actions):
        env.step(action)
        metrics = extract_sim_features(env)
        min_xy = min(min_xy, metrics["final_nut_peg_xy"])
        min_yaw = min(min_yaw, metrics["min_yaw_error"])
        if record_video and video_path is not None:
            frames.append(capture_camera_frame(env, camera_name))

    final_metrics = extract_sim_features(env)
    final_metrics["min_nut_peg_xy"] = min_xy
    final_metrics["min_yaw_error"] = min(min_yaw, final_metrics["min_yaw_error"])

    acc_mean, acc_max = action_acceleration_stats(refined_actions)
    feat_dict = rollout_metrics_to_features_dict(
        demo.demo_key,
        demo.label,
        demo.source_file,
        final_metrics,
        acc_max,
        len(refined_actions),
    )
    features = NutAssemblyFeatures(**feat_dict)
    energy = compute_total_energy(features)
    failure_guess = classify_failure_type(features, energy.E_smooth)
    success_flag = bool(env._check_success())

    result: dict[str, Any] = {
        "demo_name": demo_key,
        "source_file": hdf5_path,
        "label": label,
        "rollout_type": "refined_theta",
        "best_theta": theta,
        "success_flag": success_flag,
        "refined_success": success_flag,
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "final_z_diff": final_metrics["final_z_diff"],
        "min_nut_peg_xy": final_metrics["min_nut_peg_xy"],
        "min_yaw_error": final_metrics["min_yaw_error"],
        "failure_guess": failure_guess,
        "E_total_norm": energy.E_total_norm,
        "energy_breakdown": energy.to_dict(),
        "action_acceleration_max": acc_max,
        "num_steps": len(refined_actions),
        "reset_info": reset_info,
        "env_warnings": build.warnings,
        "video_path": str(video_path) if video_path else None,
        "object_poses_modified": False,
    }

    if video_path is not None and frames:
        write_mp4(frames, video_path, fps=control_freq)
        result["video_path"] = str(video_path)

    env.close()
    return result
