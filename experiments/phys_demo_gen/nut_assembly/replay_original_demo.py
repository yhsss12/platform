"""V2-B1：在 RoboSuite 中重放原始 demo actions（不修改 object_poses）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import action_acceleration_stats
from robosuite_env_loader import (
    capture_camera_frame,
    create_env_from_metadata,
    extract_sim_features,
    load_demo_rollout_data,
    reset_env_to_demo_state,
    rollout_metrics_to_features_dict,
    write_mp4,
)


def replay_demo_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    *,
    video_path: str | Path | None = None,
    camera_name: str = "agentview",
    control_freq: int = 20,
    record_video: bool = True,
) -> dict[str, Any]:
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    build = create_env_from_metadata(
        demo.env_args,
        for_video=record_video and video_path is not None,
        camera_name=camera_name,
    )
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    frames: list[np.ndarray] = []
    if record_video and video_path is not None:
        frames.append(capture_camera_frame(env, camera_name))

    replay_errors: list[float] = []
    min_xy = float("inf")
    min_yaw = float("inf")

    for step, action in enumerate(demo.actions):
        env.step(action)
        metrics = extract_sim_features(env)
        min_xy = min(min_xy, metrics["final_nut_peg_xy"])
        min_yaw = min(min_yaw, metrics["min_yaw_error"])
        if step < len(demo.actions) - 1:
            playback = env.sim.get_state().flatten()
            replay_errors.append(float(np.linalg.norm(demo.states[step + 1] - playback)))
        if record_video and video_path is not None:
            frames.append(capture_camera_frame(env, camera_name))

    final_metrics = extract_sim_features(env)
    final_metrics["min_nut_peg_xy"] = min_xy
    final_metrics["min_yaw_error"] = min(min_yaw, final_metrics["min_yaw_error"])

    acc_mean, acc_max = action_acceleration_stats(demo.actions)
    feat_dict = rollout_metrics_to_features_dict(
        demo.demo_key,
        demo.label,
        demo.source_file,
        final_metrics,
        acc_max,
        len(demo.actions),
    )
    from extract_features import NutAssemblyFeatures

    features = NutAssemblyFeatures(**feat_dict)
    energy = compute_total_energy(features)
    failure_guess = classify_failure_type(features, energy.E_smooth)

    success_flag = bool(env._check_success())
    replay_success = success_flag

    result: dict[str, Any] = {
        "demo_name": demo_key,
        "source_file": hdf5_path,
        "label": label,
        "replay_success": replay_success,
        "success_flag": success_flag,
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "final_z_diff": final_metrics["final_z_diff"],
        "min_nut_peg_xy": final_metrics["min_nut_peg_xy"],
        "min_yaw_error": final_metrics["min_yaw_error"],
        "failure_guess": failure_guess,
        "E_total_norm": energy.E_total_norm,
        "replay_state_error_mean": float(np.mean(replay_errors)) if replay_errors else 0.0,
        "replay_state_error_max": float(np.max(replay_errors)) if replay_errors else 0.0,
        "action_acceleration_max": acc_max,
        "num_steps": len(demo.actions),
        "reset_info": reset_info,
        "env_warnings": build.warnings,
        "video_path": str(video_path) if video_path else None,
        "metrics_near_success": bool(
            final_metrics["final_nut_peg_xy"] < 0.03 and final_metrics["final_z_diff"] < 0.0
        ),
    }

    if video_path is not None and frames:
        write_mp4(frames, video_path, fps=control_freq)
        result["video_path"] = str(video_path)

    env.close()
    return result
