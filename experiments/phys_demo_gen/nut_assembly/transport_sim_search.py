"""V2-B3：transport_failed demo sim-in-loop 随机搜索与打分。"""
from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from osc_action_converter import SimLoopParams, compute_closed_loop_waypoint_action
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
from sim_in_loop_refiner import _json_safe_theta
from transport_waypoint_builder import (
    TransportSearchParams,
    build_transport_waypoints_from_hdf5,
)

TRANSPORT_SEARCH_SPACE: dict[str, list[float | int]] = {
    "transport_xy_gain": [0.4, 0.6, 0.8, 1.0],
    "transport_xy_offset_scale": [0.5, 0.75, 1.0, 1.25],
    "pre_align_height": [0.04, 0.06, 0.08, 0.10],
    "lift_height": [0.04, 0.06, 0.08],
    "approach_steps": [10, 20, 30],
    "transport_steps": [20, 40, 60],
    "hold_steps": [5, 10, 20],
    "gripper_close_shift": [-10, -5, 0, 5],
    "speed_scale": [0.5, 0.75, 1.0],
}

LEVEL2_MIN_XY = 0.08
LEVEL3_MIN_XY = 0.03


def iter_transport_candidates(
    *,
    mode: str = "random",
    max_evals: int = 80,
    seed: int = 0,
) -> Iterator[TransportSearchParams]:
    keys = list(TRANSPORT_SEARCH_SPACE.keys())
    if mode == "grid":
        combos = list(itertools.product(*(TRANSPORT_SEARCH_SPACE[k] for k in keys)))
        rng = random.Random(seed)
        if len(combos) > max_evals:
            combos = rng.sample(combos, max_evals)
        for combo in combos:
            yield TransportSearchParams(**dict(zip(keys, combo)))
    else:
        rng = random.Random(seed)
        for _ in range(max_evals):
            yield TransportSearchParams(**{k: rng.choice(TRANSPORT_SEARCH_SPACE[k]) for k in keys})


def compute_transport_score(result: dict[str, Any]) -> float:
    """Transport 导向评分（越小越好）。"""
    xy = float(result.get("E_xy_norm", 0.0))
    transport = float(result.get("E_transport_norm", 0.0))
    yaw = float(result.get("E_yaw_norm", 0.0))
    z = float(result.get("E_z_norm", 0.0))
    smooth = float(result.get("E_smooth_norm", 0.0))
    return 4 * transport + 4 * xy + 2 * yaw + 1 * z + 0.2 * smooth


def apply_transport_step_overlay(
    action: np.ndarray,
    step: int,
    proxy: Any,
    params: TransportSearchParams,
) -> np.ndarray:
    """transport 窗口内增强 xy 跟踪增益（不改 object_poses）。"""
    out = action.copy()
    phases = proxy.phases
    grasp_idx = phases.grasp_index
    t_min_xy = phases.t_min_xy
    t1 = min(proxy.length - 1, max(t_min_xy, grasp_idx + int(params.transport_steps)))
    if grasp_idx <= step <= t1 and float(params.transport_xy_gain) != 1.0:
        gain = float(params.transport_xy_gain)
        out[0] *= gain
        out[1] *= gain
    return np.clip(out, -1.0, 1.0)


def execute_transport_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    base_theta: dict[str, Any],
    params: TransportSearchParams,
    *,
    rollout_kind: str,
    video_path: str | Path | None = None,
    record_video: bool = False,
    control_freq: int = 20,
) -> dict[str, Any]:
    proxy, _original_eef, target_eef, gripper, effective_theta = build_transport_waypoints_from_hdf5(
        hdf5_path, demo_key, label, base_theta, params
    )
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    env_args = read_env_metadata(hdf5_path)
    base_speed = float(effective_theta.get("speed_scale", 1.0))

    build = create_env_from_metadata(
        env_args,
        for_video=record_video and video_path is not None,
    )
    env = build.env
    reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

    frames: list[np.ndarray] = []
    if record_video and video_path is not None:
        frames.append(capture_camera_frame(env))

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
        action = apply_transport_step_overlay(action, step, proxy, params)
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
        "transport_params": params.to_dict(),
        "theta": _json_safe_theta(effective_theta),
        "sim_params": SimLoopParams().to_dict(),
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
        "score": compute_transport_score(
            {
                "E_xy_norm": energy.E_xy_norm,
                "E_transport_norm": energy.E_transport_norm,
                "E_yaw_norm": energy.E_yaw_norm,
                "E_z_norm": energy.E_z_norm,
                "E_smooth_norm": energy.E_smooth_norm,
            }
        ),
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


def pick_best_transport_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for row in candidates:
        row["search_score"] = compute_transport_score(row)
    return min(
        candidates,
        key=lambda row: (not row.get("success_flag", False), row["search_score"]),
    )


def classify_transport_outcome(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    if candidate.get("success_flag"):
        return "refined_success"
    final_improved = candidate["final_nut_peg_xy"] < baseline["final_nut_peg_xy"] * 0.5
    min_improved = candidate["min_nut_peg_xy"] < baseline["min_nut_peg_xy"]
    energy_improved = candidate["E_total_norm"] < baseline["E_total_norm"]
    if (final_improved or min_improved) and energy_improved:
        return "improved_but_failed"
    return "no_improvement"


def evaluate_acceptance_levels(
    original: dict[str, Any],
    best: dict[str, Any],
) -> dict[str, Any]:
    orig_final = float(original["final_nut_peg_xy"])
    best_final = float(best["final_nut_peg_xy"])
    orig_min = float(original["min_nut_peg_xy"])
    best_min = float(best["min_nut_peg_xy"])
    reduction = (orig_final - best_final) / max(orig_final, 1e-6)
    min_reduction = (orig_min - best_min) / max(orig_min, 1e-6)

    level1 = reduction >= 0.5
    level2 = best_min < LEVEL2_MIN_XY
    level3 = best_min < LEVEL3_MIN_XY or bool(best.get("success_flag"))

    return {
        "level_1_final_xy_reduction_50pct": level1,
        "level_2_min_xy_under_0.08": level2,
        "level_3_near_success_or_success": level3,
        "final_xy_reduction_ratio": float(reduction),
        "min_xy_reduction_ratio": float(min_reduction),
        "original_final_nut_peg_xy": orig_final,
        "best_final_nut_peg_xy": best_final,
        "original_min_nut_peg_xy": orig_min,
        "best_min_nut_peg_xy": best_min,
    }


def diagnose_failure_reason(result: dict[str, Any], *, baseline_min_xy: float | None = None) -> str:
    if result.get("success_flag"):
        return "success"

    min_xy = float(result["min_nut_peg_xy"])
    final_xy = float(result["final_nut_peg_xy"])
    min_yaw = float(result["min_yaw_error"])
    final_z = float(result["final_z_diff"])
    baseline_min = baseline_min_xy if baseline_min_xy is not None else 0.32

    if min_xy > 0.25 and min_xy >= baseline_min * 0.95:
        return "grasp_failed"

    if min_xy > LEVEL2_MIN_XY:
        return "transport_not_enough"

    if min_xy <= LEVEL2_MIN_XY and final_xy > max(LEVEL2_MIN_XY, min_xy + 0.03):
        return "nut_slip"

    if min_xy <= LEVEL2_MIN_XY and min_yaw > 0.05:
        return "alignment_failed"

    if min_xy <= LEVEL2_MIN_XY and final_z > 0.02:
        return "insertion_failed"

    if min_xy <= LEVEL3_MIN_XY:
        return "near_success_but_not_task_success"

    return str(result.get("failure_guess", "unknown_failed"))


def summarize_effective_params(candidates: list[dict[str, Any]], top_k: int = 10) -> dict[str, Any]:
    if not candidates:
        return {}
    ranked = sorted(
        candidates,
        key=lambda row: (not row.get("success_flag", False), row.get("search_score", row.get("score", 1e9))),
    )
    top = ranked[:top_k]
    counts: dict[str, dict[str, int]] = {}
    for row in top:
        params = row.get("transport_params") or {}
        for key, val in params.items():
            bucket = counts.setdefault(key, {})
            sval = str(val)
            bucket[sval] = bucket.get(sval, 0) + 1
    most_common = {
        key: max(vals.items(), key=lambda item: item[1])[0]
        for key, vals in counts.items()
        if vals
    }
    return {"top_k": top_k, "most_common_values_in_top_k": most_common, "counts_in_top_k": counts}


def compute_transport_search_score(result: dict[str, Any], scoring_mode: str = "transport_heuristic") -> float:
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

        return score_rollout_with_pinn(result, stage="transport")
    return compute_transport_score(result)


def run_transport_search(
    hdf5_path: str,
    demo_key: str,
    label: str,
    base_theta: dict[str, Any],
    *,
    mode: str = "random",
    max_evals: int = 80,
    seed: int = 0,
    scoring_mode: str = "transport_heuristic",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import random as random_mod

    results: list[dict[str, Any]] = []
    best_so_far: dict[str, Any] | None = None
    eval_count_to_best = 0
    rng = random_mod.Random(seed + 7919 if scoring_mode == "random" else seed)

    for index, params in enumerate(iter_transport_candidates(mode=mode, max_evals=max_evals, seed=seed)):
        result = execute_transport_rollout(
            hdf5_path,
            demo_key,
            label,
            base_theta,
            params,
            rollout_kind="transport_sim_search",
        )
        result["search_index"] = index
        if scoring_mode == "random":
            result["_random_score"] = rng.random()
        result["search_score"] = compute_transport_search_score(result, scoring_mode)
        result["seed"] = seed
        result["scoring_mode"] = scoring_mode
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
        "seed": seed,
        "max_evals": max_evals,
        "mode": mode,
        "scoring_mode": scoring_mode,
    }
    return results, meta
