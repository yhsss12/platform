"""V2-B1.5：Replay Fidelity 检查工具（final-state / state-sequence / action-replay）。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from extract_features import square_yaw_error
from robosuite_env_loader import (
    capture_camera_frame,
    create_env_from_metadata,
    extract_sim_features,
    load_demo_rollout_data,
    reset_env_to_demo_state,
    write_mp4,
)


@dataclass
class DatagenPoses:
    eef_pose: np.ndarray
    nut_pose: np.ndarray
    peg_pose: np.ndarray
    gripper_action: np.ndarray


def load_datagen_poses(hdf5_path: str, demo_key: str) -> DatagenPoses:
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle[f"data/{demo_key}"]
        return DatagenPoses(
            eef_pose=demo["datagen_info/eef_pose"][:].astype(float),
            nut_pose=demo["datagen_info/object_poses/square_nut"][:].astype(float),
            peg_pose=demo["datagen_info/object_poses/square_peg"][:].astype(float),
            gripper_action=demo["datagen_info/gripper_action"][:].astype(float).reshape(-1),
        )


def mat4_from_pos_rot(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.asarray(rot, dtype=float).reshape(3, 3)
    mat[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return mat


def pose_position_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def pose_rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    rot_diff = a[:3, :3] @ b[:3, :3].T
    trace = np.trace(rot_diff)
    angle = float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    return angle


def pose_combined_error(a: np.ndarray, b: np.ndarray) -> float:
    return pose_position_error(a, b) + pose_rotation_error(a, b)


def get_sim_eef_pose4(env: Any, arm: str = "right") -> np.ndarray:
    robot = env.robots[0]
    site_id = robot.eef_site_id[arm]
    pos = env.sim.data.site_xpos[site_id].copy()
    rot = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
    return mat4_from_pos_rot(pos, rot)


def get_sim_object_pose4(env: Any, body_id: int) -> np.ndarray:
    pos = env.sim.data.body_xpos[body_id].copy()
    rot = env.sim.data.body_xmat[body_id].reshape(3, 3).copy()
    return mat4_from_pos_rot(pos, rot)


def get_sim_poses(env: Any) -> dict[str, np.ndarray]:
    nut_name = env.nuts[env.nut_id].name
    nut_body = env.obj_body_id[nut_name]
    peg_body = env.peg1_body_id
    return {
        "eef": get_sim_eef_pose4(env),
        "nut": get_sim_object_pose4(env, nut_body),
        "peg": get_sim_object_pose4(env, peg_body),
    }


def frame_metrics(env: Any) -> dict[str, Any]:
    sim = extract_sim_features(env)
    success = bool(env._check_success())
    return {
        "nut_peg_xy": sim["final_nut_peg_xy"],
        "z_diff": sim["final_z_diff"],
        "yaw_error": sim["min_yaw_error"],
        "success_flag": success,
    }


def compare_poses_to_datagen(env: Any, datagen: DatagenPoses, step: int) -> dict[str, float]:
    sim_poses = get_sim_poses(env)
    step = min(step, len(datagen.eef_pose) - 1)
    return {
        "eef_pose_error": pose_combined_error(sim_poses["eef"], datagen.eef_pose[step]),
        "eef_position_error": pose_position_error(sim_poses["eef"], datagen.eef_pose[step]),
        "nut_pose_error": pose_combined_error(sim_poses["nut"], datagen.nut_pose[step]),
        "peg_pose_error": pose_combined_error(sim_poses["peg"], datagen.peg_pose[step]),
    }


def extract_runtime_controller_info(env: Any) -> dict[str, Any]:
    robot = env.robots[0]
    right_cfg = dict(robot.part_controller_config.get("right", {}))
    for key in list(right_cfg.keys()):
        val = right_cfg[key]
        if isinstance(val, np.ndarray):
            right_cfg[key] = val.tolist()
    composite_cfg = getattr(robot, "composite_controller_config", {}) or {}
    return {
        "action_dim": int(env.action_dim),
        "control_freq": int(getattr(env, "control_freq", -1)),
        "composite_controller_type": composite_cfg.get("type", "unknown"),
        "right_arm_controller": {
            "type": right_cfg.get("type"),
            "control_delta": right_cfg.get("control_delta"),
            "input_max": right_cfg.get("input_max"),
            "input_min": right_cfg.get("input_min"),
            "output_max": right_cfg.get("output_max"),
            "output_min": right_cfg.get("output_min"),
            "kp": right_cfg.get("kp"),
            "damping": right_cfg.get("damping"),
            "uncouple_pos_ori": right_cfg.get("uncouple_pos_ori"),
            "interpolation": right_cfg.get("interpolation"),
            "ramp_ratio": right_cfg.get("ramp_ratio"),
        },
    }


def compare_controller_configs(hdf5_env_args: dict[str, Any], runtime_info: dict[str, Any]) -> dict[str, Any]:
    recorded = dict(hdf5_env_args.get("env_kwargs", hdf5_env_args).get("controller_configs", {}))
    runtime = runtime_info["right_arm_controller"]
    diffs: list[str] = []

    if recorded.get("type") == "OSC_POSE" and runtime_info.get("composite_controller_type"):
        diffs.append(
            "HDF5 uses flat OSC_POSE; runtime uses CompositeController wrapper (refactored at env load)"
        )

    for key in ["control_delta", "kp", "damping", "uncouple_pos_ori", "ramp_ratio"]:
        if key in recorded and recorded[key] != runtime.get(key):
            diffs.append(f"{key}: hdf5={recorded[key]!r} runtime={runtime.get(key)!r}")

    for key in ["output_max", "output_min"]:
        rec = np.array(recorded.get(key, []), dtype=float)
        run = np.array(runtime.get(key, []), dtype=float)
        if rec.size and run.size and not np.allclose(rec, run, rtol=1e-5, atol=1e-8):
            diffs.append(f"{key}: hdf5={rec.tolist()} runtime={run.tolist()}")

    hdf5_freq = hdf5_env_args.get("env_kwargs", hdf5_env_args).get("control_freq")
    if hdf5_freq is not None and int(hdf5_freq) != int(runtime_info.get("control_freq", -1)):
        diffs.append(f"control_freq: hdf5={hdf5_freq} runtime={runtime_info.get('control_freq')}")

    return {
        "hdf5_controller_configs": recorded,
        "runtime_controller_info": runtime_info,
        "differences": diffs,
        "action_shape_expected": "(T, 7)",
        "action_range_observed_hdf5": "[-1, 1] per OSC normalized input",
    }


def run_final_state_check(
    hdf5_path: str,
    demo_key: str,
    label: str,
) -> dict[str, Any]:
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    build = create_env_from_metadata(demo.env_args, for_video=False)
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    env.sim.set_state_from_flattened(demo.states[-1])
    env.sim.forward()
    residuals = extract_sim_features(env)
    success = bool(env._check_success())
    datagen = load_datagen_poses(hdf5_path, demo_key)
    pose_err = compare_poses_to_datagen(env, datagen, len(demo.actions) - 1)

    result = {
        "check": "final_state",
        "demo_name": demo_key,
        "label": label,
        "source_file": hdf5_path,
        "final_state_success_flag": success,
        "final_state_residuals": residuals,
        "state_vs_datagen_pose_error": pose_err,
        "reset_info": reset_info,
        "env_warnings": build.warnings,
    }
    env.close()
    return result


def run_state_sequence_check(
    hdf5_path: str,
    demo_key: str,
    label: str,
    *,
    video_path: Path | None = None,
    control_freq: int = 20,
    record_video: bool = False,
    subsample_video: int = 2,
) -> dict[str, Any]:
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    datagen = load_datagen_poses(hdf5_path, demo_key)
    build = create_env_from_metadata(
        demo.env_args,
        for_video=record_video and video_path is not None,
    )
    env = build.env
    reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    frames: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    min_xy = float("inf")
    min_yaw = float("inf")
    any_success = False

    for step, state in enumerate(demo.states):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        metrics = frame_metrics(env)
        pose_err = compare_poses_to_datagen(env, datagen, step)
        min_xy = min(min_xy, metrics["nut_peg_xy"])
        min_yaw = min(min_yaw, metrics["yaw_error"])
        any_success = any_success or metrics["success_flag"]
        timeline.append(
            {
                "step": step,
                **metrics,
                **pose_err,
                "state_roundtrip_error": float(
                    np.linalg.norm(state - env.sim.get_state().flatten())
                ),
            }
        )
        if record_video and video_path is not None and step % subsample_video == 0:
            frames.append(capture_camera_frame(env))

    final = timeline[-1] if timeline else {}
    result = {
        "check": "state_sequence",
        "demo_name": demo_key,
        "label": label,
        "source_file": hdf5_path,
        "num_steps": len(timeline),
        "final_success_flag": final.get("success_flag", False),
        "any_success_flag": any_success,
        "final_residuals": {
            "nut_peg_xy": final.get("nut_peg_xy"),
            "z_diff": final.get("z_diff"),
            "yaw_error": final.get("yaw_error"),
        },
        "min_nut_peg_xy": float(min_xy),
        "min_yaw_error": float(min_yaw),
        "max_state_roundtrip_error": float(max(t["state_roundtrip_error"] for t in timeline)),
        "timeline": timeline,
        "video_path": str(video_path) if video_path else None,
    }

    if record_video and video_path is not None and frames:
        write_mp4(frames, video_path, fps=control_freq)
        result["video_path"] = str(video_path)

    env.close()
    return result


def run_action_replay_check(
    hdf5_path: str,
    demo_key: str,
    label: str,
    *,
    video_path: Path | None = None,
    control_freq: int = 20,
    record_video: bool = False,
    subsample_video: int = 2,
) -> dict[str, Any]:
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    datagen = load_datagen_poses(hdf5_path, demo_key)
    build = create_env_from_metadata(
        demo.env_args,
        for_video=record_video and video_path is not None,
    )
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    frames: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    replay_state_errors: list[float] = []
    eef_errors: list[float] = []
    nut_errors: list[float] = []
    peg_errors: list[float] = []

    if record_video and video_path is not None:
        frames.append(capture_camera_frame(env))

    for step, action in enumerate(demo.actions):
        env.step(action)
        metrics = frame_metrics(env)
        pose_err = compare_poses_to_datagen(env, datagen, step)
        eef_errors.append(pose_err["eef_pose_error"])
        nut_errors.append(pose_err["nut_pose_error"])
        peg_errors.append(pose_err["peg_pose_error"])

        state_err = None
        if step < len(demo.actions) - 1:
            playback = env.sim.get_state().flatten()
            state_err = float(np.linalg.norm(demo.states[step + 1] - playback))
            replay_state_errors.append(state_err)

        timeline.append(
            {
                "step": step,
                "action": action.tolist(),
                **metrics,
                **pose_err,
                "replay_state_error": state_err,
            }
        )
        if record_video and video_path is not None and step % subsample_video == 0:
            frames.append(capture_camera_frame(env))

    final_metrics = frame_metrics(env)
    result = {
        "check": "action_replay",
        "demo_name": demo_key,
        "label": label,
        "source_file": hdf5_path,
        "num_steps": len(demo.actions),
        "action_shape": list(demo.actions.shape),
        "action_min": float(demo.actions.min()),
        "action_max": float(demo.actions.max()),
        "action_mean_abs": float(np.mean(np.abs(demo.actions))),
        "final_success_flag": final_metrics["success_flag"],
        "final_residuals": {
            "nut_peg_xy": final_metrics["nut_peg_xy"],
            "z_diff": final_metrics["z_diff"],
            "yaw_error": final_metrics["yaw_error"],
        },
        "replay_state_error_mean": float(np.mean(replay_state_errors)) if replay_state_errors else 0.0,
        "replay_state_error_max": float(np.max(replay_state_errors)) if replay_state_errors else 0.0,
        "replay_state_error_final": replay_state_errors[-1] if replay_state_errors else None,
        "eef_pose_error_mean": float(np.mean(eef_errors)),
        "eef_pose_error_max": float(np.max(eef_errors)),
        "nut_pose_error_mean": float(np.mean(nut_errors)),
        "nut_pose_error_max": float(np.max(nut_errors)),
        "peg_pose_error_mean": float(np.mean(peg_errors)),
        "peg_pose_error_max": float(np.max(peg_errors)),
        "reset_info": reset_info,
        "env_warnings": build.warnings,
        "timeline": timeline,
        "video_path": str(video_path) if video_path else None,
    }

    if record_video and video_path is not None and frames:
        write_mp4(frames, video_path, fps=control_freq)
        result["video_path"] = str(video_path)

    env.close()
    return result


def diagnose_demo_fidelity(
    final_state: dict[str, Any],
    state_sequence: dict[str, Any],
    action_replay: dict[str, Any],
    controller_comparison: dict[str, Any],
) -> dict[str, Any]:
    """根据三类检查结果给出明确诊断结论。"""
    fs_ok = bool(final_state.get("final_state_success_flag"))
    ss_ok = bool(state_sequence.get("final_success_flag"))
    ar_ok = bool(action_replay.get("final_success_flag"))
    label = final_state.get("label", "unknown")
    expected_success = label == "success"

    issues: list[str] = []
    replay_err = float(action_replay.get("replay_state_error_mean", 0.0))
    replay_err_high = replay_err > 0.15

    fs_residuals = final_state.get("final_state_residuals", {})
    ar_residuals = action_replay.get("final_residuals", {})
    xy_drift = abs(
        float(ar_residuals.get("nut_peg_xy", 0.0)) - float(fs_residuals.get("final_nut_peg_xy", 0.0))
    )
    z_drift = abs(
        float(ar_residuals.get("z_diff", 0.0)) - float(fs_residuals.get("final_z_diff", 0.0))
    )
    residual_drift_notable = xy_drift > 0.02 or z_drift > 0.02

    if expected_success:
        if fs_ok:
            issues.append("HDF5 final state is successful in current env (state vector compatible).")
        else:
            issues.append("HDF5 final state does NOT pass env success checker → env/state compatibility issue.")

        if ss_ok or state_sequence.get("any_success_flag"):
            issues.append("State-sequence replay reproduces success residuals from HDF5 states.")
        else:
            issues.append("State-sequence replay fails to reproduce success → state compatibility issue.")

        if replay_err_high:
            issues.append(
                f"Action replay state trajectory diverges from HDF5 (mean L2 error={replay_err:.4f} > 0.15)."
            )
        elif replay_err > 0.05:
            issues.append(
                f"Action replay shows moderate state drift (mean L2 error={replay_err:.4f})."
            )

        if residual_drift_notable:
            issues.append(
                f"Action replay final residuals drift from final-state check (Δxy={xy_drift:.4f}, Δz={z_drift:.4f})."
            )

        action_replay_unreliable = (not ar_ok) or replay_err_high or residual_drift_notable

        if fs_ok and action_replay_unreliable:
            primary = "controller_action_replay_fidelity_issue"
            summary = (
                "Final-state success=true but open-loop action replay is unreliable: "
                "HDF5 states are valid (state-sequence reproduces success), yet action replay "
                f"either fails success_flag ({ar_ok}) or diverges from recorded states "
                f"(replay_state_error_mean={replay_err:.4f}). "
                "Root cause: legacy OSC_POSE → CompositeController refactor and closed-loop vs open-loop mismatch."
            )
        elif not fs_ok:
            primary = "env_state_compatibility_issue"
            summary = (
                "Final-state success=false for a success demo: prioritize fixing env loader / XML / "
                "success checker / state dimension before trusting action replay or refined rollout."
            )
        elif fs_ok and ar_ok:
            primary = "replay_fidelity_ok"
            summary = "Both final-state and action replay agree on success."
        else:
            primary = "mixed_fidelity_issue"
            summary = "Outcome mismatch across checks; inspect per-step timeline."

        refined_rollout_interpretation = (
            "residual_improvement_validation"
            if primary == "controller_action_replay_fidelity_issue"
            else (
                "full_success_validation"
                if primary == "replay_fidelity_ok"
                else "blocked_pending_env_fix"
            )
        )
        continue_refined_rollout = primary in {
            "controller_action_replay_fidelity_issue",
            "replay_fidelity_ok",
            "mixed_fidelity_issue",
        }
    else:
        # failed demo：final-state 应为 false；检查三类检查是否一致失败
        fs_failed = not fs_ok
        ss_failed = not ss_ok
        ar_failed = not ar_ok

        if fs_failed:
            issues.append("HDF5 final state correctly evaluates as failed in current env.")
        else:
            issues.append("Unexpected: failed demo final-state passes success checker.")

        if ss_failed:
            issues.append("State-sequence replay shows failed residuals at final frame.")
        else:
            issues.append("State-sequence unexpectedly shows success at final frame.")

        if replay_err > 0.05:
            issues.append(
                f"Action replay diverges from recorded states (mean L2 error={replay_err:.4f})."
            )

        consistent_failure = fs_failed and ss_failed and ar_failed
        if consistent_failure:
            primary = "consistent_failure_as_expected"
            summary = (
                "Failed demo: final-state, state-sequence, and action replay all agree on failure. "
                "HDF5 states reliably encode the failed outcome."
            )
        else:
            primary = "mixed_failure_fidelity"
            summary = "Failed demo: checks disagree on failure mode; inspect per-step timeline."

        refined_rollout_interpretation = "residual_improvement_validation"
        continue_refined_rollout = True

    return {
        "primary_diagnosis": primary,
        "summary": summary,
        "issues": issues,
        "checks_agreement": {
            "final_state_success": fs_ok,
            "state_sequence_final_success": ss_ok,
            "action_replay_final_success": ar_ok,
            "expected_label": label,
            "expected_success": expected_success,
        },
        "controller_differences": controller_comparison.get("differences", []),
        "replay_state_error_mean": replay_err,
        "residual_drift": {"xy": xy_drift, "z": z_drift},
        "continue_refined_rollout": continue_refined_rollout,
        "refined_rollout_interpretation": refined_rollout_interpretation,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def timeline_summary_rows(demo_key: str, label: str, checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    mapping = [
        ("final_state", "final_state"),
        ("state_sequence", "state_sequence"),
        ("action_replay", "action_replay"),
    ]
    for check_name, _ in mapping:
        payload = checks[check_name]
        residuals = payload.get("final_state_residuals") or payload.get("final_residuals") or {}
        rows.append(
            {
                "demo_key": demo_key,
                "label": label,
                "check": payload.get("check", check_name),
                "final_success_flag": payload.get("final_state_success_flag")
                if check_name == "final_state"
                else payload.get("final_success_flag"),
                "final_nut_peg_xy": residuals.get("final_nut_peg_xy") or residuals.get("nut_peg_xy"),
                "final_z_diff": residuals.get("final_z_diff") or residuals.get("z_diff"),
                "replay_state_error_mean": payload.get("replay_state_error_mean"),
                "eef_pose_error_mean": payload.get("eef_pose_error_mean"),
            }
        )
    return rows
