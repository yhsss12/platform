"""V2-B2.5：Simulator-in-the-loop 局部参数搜索与 rollout 打分。"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from osc_action_converter import (
    SEARCH_SPACE,
    SimLoopParams,
)
from refined_waypoint_builder import (
    build_refined_waypoints_from_hdf5,
    load_eef_pose_sequence,
)
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
from trajectory_parameterization import empty_theta, load_trajectory_proxy

SUCCESS_Z_TARGET = -0.021


def _json_safe_theta(theta: dict | None) -> dict | None:
    if theta is None:
        return None
    out: dict = {}
    for key, val in theta.items():
        if isinstance(val, np.ndarray):
            out[key] = val.tolist()
        elif isinstance(val, (list, tuple)):
            out[key] = [float(x) for x in val]
        else:
            out[key] = float(val) if isinstance(val, (np.floating, np.integer)) else val
    return out


def load_best_theta(cem_report_path: str | Path, demo_key: str = "demo_4") -> dict[str, Any]:
    report = json.loads(Path(cem_report_path).read_text(encoding="utf-8"))
    for item in report.get("results", []):
        if item.get("demo_key") == demo_key:
            return item["best_theta"]
    raise KeyError(f"{demo_key} not found in {cem_report_path}")


def load_best_theta_or_fallback(
    cem_report_path: str | Path,
    demo_key: str,
    *,
    fallback_demo_key: str = "demo_4",
) -> dict[str, Any]:
    try:
        return load_best_theta(cem_report_path, demo_key)
    except KeyError:
        return load_best_theta(cem_report_path, fallback_demo_key)


def iter_search_candidates(
    *,
    mode: str = "random",
    max_evals: int = 150,
    seed: int = 0,
) -> Iterator[SimLoopParams]:
    keys = list(SEARCH_SPACE.keys())
    if mode == "grid":
        combos = list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
        rng = random.Random(seed)
        if len(combos) > max_evals:
            combos = rng.sample(combos, max_evals)
        for combo in combos:
            yield SimLoopParams(**dict(zip(keys, combo)))
    else:
        rng = random.Random(seed)
        for _ in range(max_evals):
            yield SimLoopParams(**{k: rng.choice(SEARCH_SPACE[k]) for k in keys})


def score_rollout_result(result: dict[str, Any]) -> float:
    """与 E_total_norm 一致：3*E_xy + 3*E_transport + 2*E_yaw + 2*E_z + 0.2*E_smooth。"""
    return float(result["E_total_norm"])


SCORING_MODES = {
    "energy_full",
    "explicit_energy",
    "energy_without_z",
    "energy_without_xy_transport",
    "energy_without_yaw",
    "energy_without_smooth",
    "random",
    "pinn_energy",
}


def compute_search_score(result: dict[str, Any], scoring_mode: str) -> float:
    """Sim-in-loop 搜索排序分数（越小越好）；random 模式不使用 energy。"""
    if scoring_mode == "random":
        return float(result.get("_random_score", 0.0))
    if scoring_mode in ("energy_full", "explicit_energy"):
        xy = float(result.get("E_xy_norm", 0.0))
        transport = float(result.get("E_transport_norm", 0.0))
        yaw = float(result.get("E_yaw_norm", 0.0))
        z = float(result.get("E_z_norm", 0.0))
        smooth = float(result.get("E_smooth_norm", 0.0))
        return 3 * xy + 3 * transport + 2 * yaw + 2 * z + 0.2 * smooth
    if scoring_mode == "pinn_energy":
        import sys
        from pathlib import Path

        v1_dir = Path(__file__).resolve().parent / "v1_residual_model"
        if str(v1_dir) not in sys.path:
            sys.path.insert(0, str(v1_dir))
        from pinn_inference import score_rollout_with_pinn

        return score_rollout_with_pinn(result)
    if scoring_mode == "energy_without_z":
        return 3 * xy + 3 * transport + 2 * yaw + 0.2 * smooth
    if scoring_mode == "energy_without_xy_transport":
        return 2 * yaw + 2 * z + 0.2 * smooth
    if scoring_mode == "energy_without_yaw":
        return 3 * xy + 3 * transport + 2 * z + 0.2 * smooth
    if scoring_mode == "energy_without_smooth":
        return 3 * xy + 3 * transport + 2 * yaw + 2 * z
    raise ValueError(f"unknown scoring_mode: {scoring_mode}")


def pick_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    scoring_mode: str = "energy_full",
) -> dict[str, Any]:
    for row in candidates:
        row["search_score"] = compute_search_score(row, scoring_mode)
    return min(
        candidates,
        key=lambda row: (not row.get("success_flag", False), row["search_score"]),
    )


def classify_outcome(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    success_z_target: float = SUCCESS_Z_TARGET,
) -> str:
    if candidate.get("success_flag"):
        return "refined_success"
    z_improved = abs(candidate["final_z_diff"] - success_z_target) < abs(
        baseline["final_z_diff"] - success_z_target
    )
    energy_improved = candidate["E_total_norm"] < baseline["E_total_norm"]
    if z_improved and energy_improved:
        return "improved_but_failed"
    return "no_improvement"


def execute_waypoint_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    target_eef: np.ndarray,
    gripper: np.ndarray,
    sim_params: SimLoopParams,
    *,
    rollout_kind: str,
    theta: dict[str, Any] | None = None,
    video_path: str | Path | None = None,
    record_video: bool = False,
    control_freq: int = 20,
) -> dict[str, Any]:
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    env_args = read_env_metadata(hdf5_path)
    base_speed = float(theta.get("speed_scale", 1.0)) if theta else 1.0

    build = create_env_from_metadata(
        env_args,
        for_video=record_video and video_path is not None,
    )
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    frames: list[np.ndarray] = []
    if record_video and video_path is not None:
        frames.append(capture_camera_frame(env))

    from osc_action_converter import (
        apply_sim_loop_step_overlay,
        compute_closed_loop_waypoint_action,
    )

    length = len(target_eef)
    grip = np.asarray(gripper, dtype=float).reshape(-1)
    actions = np.zeros((length, 7), dtype=float)
    min_xy = float("inf")
    min_yaw = float("inf")

    for step in range(length):
        target_idx = min(step + 1, length - 1)
        action = compute_closed_loop_waypoint_action(
            env,
            target_eef[target_idx],
            grip[step],
            env_args,
            speed_scale=base_speed,
        )
        action = apply_sim_loop_step_overlay(
            action, step, proxy, grip, env_args, sim_params
        )
        actions[step] = action
        env.step(action)
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
        "sim_params": sim_params.to_dict(),
        "theta": _json_safe_theta(theta),
        "success_flag": success_flag,
        "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
        "min_nut_peg_xy": final_metrics["min_nut_peg_xy"],
        "final_z_diff": final_metrics["final_z_diff"],
        "min_yaw_error": final_metrics["min_yaw_error"],
        "failure_guess": failure_guess,
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
        "video_path": str(video_path) if video_path else None,
    }

    if record_video and video_path is not None and frames:
        write_mp4(frames, video_path, fps=control_freq)
        result["video_path"] = str(video_path)

    env.close()
    return result


def run_original_waypoint_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    *,
    video_path: str | Path | None = None,
    record_video: bool = False,
) -> dict[str, Any]:
    proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
    original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
    gripper = proxy.gripper_action
    return execute_waypoint_rollout(
        hdf5_path,
        demo_key,
        label,
        original_eef,
        gripper,
        SimLoopParams(),
        rollout_kind="original_waypoint_rollout",
        theta=empty_theta(),
        video_path=video_path,
        record_video=record_video,
    )


def run_refined_waypoint_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    theta: dict[str, Any],
    sim_params: SimLoopParams | None = None,
    *,
    video_path: str | Path | None = None,
    record_video: bool = False,
) -> dict[str, Any]:
    proxy, original_eef, refined_eef, shifted_gripper = build_refined_waypoints_from_hdf5(
        hdf5_path,
        demo_key,
        label,
        theta,
        rollout_safe=True,
    )
    return execute_waypoint_rollout(
        hdf5_path,
        demo_key,
        label,
        refined_eef,
        shifted_gripper,
        sim_params or SimLoopParams(),
        rollout_kind="refined_waypoint_rollout",
        theta=theta,
        video_path=video_path,
        record_video=record_video,
    )


def run_sim_in_loop_search(
    hdf5_path: str,
    demo_key: str,
    label: str,
    theta: dict[str, Any],
    *,
    mode: str = "random",
    max_evals: int = 150,
    seed: int = 0,
    scoring_mode: str = "energy_full",
    record_videos: bool = False,
    video_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proxy, _, refined_eef, shifted_gripper = build_refined_waypoints_from_hdf5(
        hdf5_path,
        demo_key,
        label,
        theta,
        rollout_safe=True,
    )
    rng = random.Random(seed + 7919 if scoring_mode == "random" else seed)
    results: list[dict[str, Any]] = []
    best_so_far: dict[str, Any] | None = None
    eval_count_to_best = 0

    for index, sim_params in enumerate(iter_search_candidates(mode=mode, max_evals=max_evals, seed=seed)):
        video_path = None
        if record_videos and video_dir is not None:
            video_path = video_dir / f"search_{index:04d}.mp4"
        result = execute_waypoint_rollout(
            hdf5_path,
            demo_key,
            label,
            refined_eef,
            shifted_gripper,
            sim_params,
            rollout_kind="sim_in_loop_search",
            theta=theta,
            video_path=video_path,
            record_video=record_videos and video_path is not None,
        )
        result["search_index"] = index
        if scoring_mode == "random":
            result["_random_score"] = rng.random()
        result["search_score"] = compute_search_score(result, scoring_mode)
        results.append(result)

        if best_so_far is None:
            best_so_far = result
            eval_count_to_best = index + 1
        else:
            candidate_key = (not result.get("success_flag", False), result["search_score"])
            best_key = (not best_so_far.get("success_flag", False), best_so_far["search_score"])
            if candidate_key < best_key:
                best_so_far = result
                eval_count_to_best = index + 1

    meta = {
        "eval_count_to_best": eval_count_to_best,
        "total_evals": len(results),
        "scoring_mode": scoring_mode,
        "seed": seed,
        "max_evals": max_evals,
    }
    return results, meta


def summarize_repeatability_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    energies = [float(r["E_total_norm"]) for r in runs]
    z_diffs = [float(r["final_z_diff"]) for r in runs]
    successes = [bool(r.get("success_flag")) for r in runs]
    return {
        "n_runs": len(runs),
        "success_count": sum(successes),
        "success_rate": float(sum(successes) / len(runs)),
        "E_total_norm_mean": float(np.mean(energies)),
        "E_total_norm_std": float(np.std(energies)),
        "E_total_norm_min": float(np.min(energies)),
        "E_total_norm_max": float(np.max(energies)),
        "final_z_diff_mean": float(np.mean(z_diffs)),
        "final_z_diff_std": float(np.std(z_diffs)),
        "final_z_diff_distance_to_target_mean": float(np.mean([abs(z - SUCCESS_Z_TARGET) for z in z_diffs])),
        "outcomes": [r.get("outcome_label") for r in runs],
    }


def result_to_summary_row(result: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row = {
        "success_flag": result.get("success_flag"),
        "outcome": result.get("outcome_label"),
        "E_total_norm": result.get("E_total_norm"),
        "final_nut_peg_xy": result.get("final_nut_peg_xy"),
        "final_z_diff": result.get("final_z_diff"),
        "min_yaw_error": result.get("min_yaw_error"),
        "best_params": json.dumps(result.get("sim_params") or {}),
        "eval_count_to_best": result.get("eval_count_to_best"),
        "video_path": result.get("video_path"),
        "rollout_kind": result.get("rollout_kind"),
        "search_score": result.get("search_score"),
        "failure_guess": result.get("failure_guess"),
    }
    row.update(extra)
    return row
