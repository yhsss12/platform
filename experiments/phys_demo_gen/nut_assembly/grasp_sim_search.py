"""V2-B4：grasp_failed demo sim-in-loop 随机搜索与打分。"""
from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from grasp_waypoint_builder import GraspSearchParams, build_grasp_waypoints_from_hdf5
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

GRASP_SEARCH_SPACE: dict[str, list[float | int]] = {
    "grasp_xy_offset_x": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "grasp_xy_offset_y": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "pre_grasp_height": [0.03, 0.05, 0.07, 0.09],
    "approach_height": [0.01, 0.02, 0.03],
    "gripper_close_shift": [-15, -10, -5, 0, 5],
    "gripper_hold_steps": [10, 20, 30, 40],
    "lift_height": [0.04, 0.06, 0.08, 0.10],
    "lift_steps": [10, 20, 30],
    "speed_scale": [0.4, 0.6, 0.8, 1.0],
}

TRANSPORT_IMPROVED_MIN_XY_RATIO = 0.70


def iter_grasp_candidates(
    *,
    mode: str = "random",
    max_evals: int = 80,
    seed: int = 0,
) -> Iterator[GraspSearchParams]:
    keys = list(GRASP_SEARCH_SPACE.keys())
    if mode == "grid":
        combos = list(itertools.product(*(GRASP_SEARCH_SPACE[k] for k in keys)))
        rng = random.Random(seed)
        if len(combos) > max_evals:
            combos = rng.sample(combos, max_evals)
        for combo in combos:
            yield GraspSearchParams(**dict(zip(keys, combo)))
    else:
        rng = random.Random(seed)
        for _ in range(max_evals):
            yield GraspSearchParams(**{k: rng.choice(GRASP_SEARCH_SPACE[k]) for k in keys})


def get_sim_nut_pos(env: Any) -> np.ndarray:
    nut = env.nuts[env.nut_id]
    nut_name = nut.name
    return env.sim.data.body_xpos[env.obj_body_id[nut_name]].copy()


def compute_grasp_proxies(
    *,
    nut_displacement_after_grasp: float,
    nut_lift_delta: float,
    eef_nut_distance_at_grasp: float,
) -> dict[str, bool]:
    cond_motion = nut_displacement_after_grasp > 0.03
    cond_lift = nut_lift_delta > 0.02
    cond_distance = eef_nut_distance_at_grasp < 0.05
    grasp_success_proxy = sum([cond_motion, cond_lift, cond_distance]) >= 2
    lift_success_proxy = cond_lift
    return {
        "grasp_success_proxy": grasp_success_proxy,
        "lift_success_proxy": lift_success_proxy,
        "cond_nut_motion": cond_motion,
        "cond_lift": cond_lift,
        "cond_grasp_distance": cond_distance,
    }


def compute_grasp_score(result: dict[str, Any]) -> float:
    grasp_distance_energy = float(result.get("eef_nut_distance_at_grasp", 0.05)) / 0.05
    nut_lift_delta = float(result.get("nut_lift_delta", 0.0))
    nut_disp_after = float(result.get("nut_displacement_after_grasp", 0.0))
    no_lift_penalty = max(0.0, 0.02 - nut_lift_delta) / 0.02
    no_nut_motion_penalty = max(0.0, 0.03 - nut_disp_after) / 0.03
    e_transport = float(result.get("E_transport_norm", 0.0))
    e_xy = float(result.get("E_xy_norm", 0.0))
    e_smooth = float(result.get("E_smooth_norm", 0.0))
    return (
        4.0 * grasp_distance_energy
        + 4.0 * no_lift_penalty
        + 3.0 * no_nut_motion_penalty
        + 2.0 * e_transport
        + 1.0 * e_xy
        + 0.2 * e_smooth
    )


def apply_grasp_step_overlay(
    action: np.ndarray,
    step: int,
    proxy: Any,
    params: GraspSearchParams,
) -> np.ndarray:
    """grasp / lift 窗口内略增 z 跟踪（不改 object_poses）。"""
    out = action.copy()
    grasp_idx = proxy.phases.grasp_index
    lift_end = min(proxy.length - 1, grasp_idx + int(params.lift_steps))
    if grasp_idx - 5 <= step <= grasp_idx + int(params.gripper_hold_steps):
        out[2] *= 1.15
    if grasp_idx + 1 <= step <= lift_end:
        out[2] *= 1.25
    return np.clip(out, -1.0, 1.0)


def execute_grasp_rollout(
    hdf5_path: str,
    demo_key: str,
    label: str,
    params: GraspSearchParams,
    *,
    rollout_kind: str,
    video_path: str | Path | None = None,
    record_video: bool = False,
    control_freq: int = 20,
) -> dict[str, Any]:
    proxy, _original_eef, target_eef, gripper = build_grasp_waypoints_from_hdf5(
        hdf5_path, demo_key, label, params
    )
    demo = load_demo_rollout_data(hdf5_path, demo_key, label)
    env_args = read_env_metadata(hdf5_path)
    base_speed = float(params.speed_scale)
    grasp_idx = proxy.phases.grasp_index

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

    nut_positions: list[np.ndarray] = []
    nut_z_trace: list[float] = []
    eef_nut_distance_at_grasp = float("inf")
    gripper_closed_flags: list[float] = []

    initial_nut = get_sim_nut_pos(env)
    nut_positions.append(initial_nut.copy())

    for step in range(length):
        target_idx = min(step + 1, length - 1)
        action = compute_closed_loop_waypoint_action(
            env,
            target_eef[target_idx],
            grip[step],
            env_args,
            speed_scale=base_speed,
        )
        action = apply_grasp_step_overlay(action, step, proxy, params)
        actions[step] = action
        env.step(action)

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

    lift_start = min(len(nut_z_trace) - 1, grasp_idx + 1)
    lift_end = min(len(nut_z_trace) - 1, grasp_idx + int(params.lift_steps))
    nut_z_at_grasp = nut_z_trace[grasp_step] if grasp_step < len(nut_z_trace) else nut_z_trace[-1]
    if lift_end > lift_start:
        nut_lift_delta = float(max(nut_z_trace[lift_start : lift_end + 1]) - nut_z_at_grasp)
    else:
        nut_lift_delta = 0.0

    min_eef_nut_distance = eef_nut_distance_at_grasp
    if grasp_step < len(nut_positions_arr):
        for step in range(grasp_step, len(nut_positions_arr)):
            eef_step = target_eef[min(step, len(target_eef) - 1), :3, 3]
            min_eef_nut_distance = min(
                min_eef_nut_distance,
                float(np.linalg.norm(eef_step - nut_positions_arr[step])),
            )

    gripper_closed_fraction = float(np.mean(gripper_closed_flags)) if gripper_closed_flags else 0.0
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
        "gripper_closed_fraction": gripper_closed_fraction,
        "grasp_success_proxy": proxies["grasp_success_proxy"],
        "lift_success_proxy": proxies["lift_success_proxy"],
        "failure_guess": failure_guess,
        "E_total_norm": energy.E_total_norm,
        "E_xy_norm": energy.E_xy_norm,
        "E_transport_norm": energy.E_transport_norm,
        "E_yaw_norm": energy.E_yaw_norm,
        "E_z_norm": energy.E_z_norm,
        "E_smooth_norm": energy.E_smooth_norm,
        "score": compute_grasp_score(
            {
                "eef_nut_distance_at_grasp": eef_nut_distance_at_grasp,
                "nut_lift_delta": nut_lift_delta,
                "nut_displacement_after_grasp": nut_displacement_after_grasp,
                "E_transport_norm": energy.E_transport_norm,
                "E_xy_norm": energy.E_xy_norm,
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


def pick_best_grasp_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for row in candidates:
        row["search_score"] = compute_grasp_score(row)
    return min(
        candidates,
        key=lambda row: (
            not row.get("success_flag", False),
            not row.get("grasp_success_proxy", False),
            row["search_score"],
        ),
    )


def classify_grasp_outcome(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    if candidate.get("success_flag"):
        return "refined_success"
    grasp_improved = (
        float(candidate.get("nut_displacement_after_grasp", 0.0))
        >= float(baseline.get("nut_displacement_after_grasp", 0.0)) * 1.5
        or bool(candidate.get("grasp_success_proxy"))
    )
    transport_improved = float(candidate.get("min_nut_peg_xy", 1.0)) < float(
        baseline.get("min_nut_peg_xy", 1.0)
    ) * TRANSPORT_IMPROVED_MIN_XY_RATIO
    if grasp_improved or transport_improved:
        return "grasp_improved_but_failed"
    return "grasp_no_improvement"


def evaluate_grasp_levels(
    original: dict[str, Any],
    best: dict[str, Any],
) -> dict[str, Any]:
    orig_disp = float(original.get("nut_displacement_after_grasp", 0.0))
    best_disp = float(best.get("nut_displacement_after_grasp", 0.0))
    orig_min_xy = float(original.get("min_nut_peg_xy", 1.0))
    best_min_xy = float(best.get("min_nut_peg_xy", 1.0))
    disp_ratio = (best_disp - orig_disp) / max(orig_disp, 1e-6) if orig_disp > 1e-6 else (
        1.0 if best_disp > 0.0 else 0.0
    )
    min_xy_reduction = (orig_min_xy - best_min_xy) / max(orig_min_xy, 1e-6)

    level_g1 = best_disp >= orig_disp * 1.5 if orig_disp > 1e-6 else best_disp > 0.015
    level_g2 = float(best.get("nut_lift_delta", 0.0)) > 0.02 or bool(best.get("grasp_success_proxy"))
    level_g3 = min_xy_reduction >= 0.30 or (
        best_min_xy < orig_min_xy * TRANSPORT_IMPROVED_MIN_XY_RATIO
    )

    return {
        "level_g1_nut_motion_improved_50pct": level_g1,
        "level_g2_lift_or_grasp_proxy": level_g2,
        "level_g3_transport_improved_30pct": level_g3,
        "nut_displacement_after_grasp_improvement_ratio": float(disp_ratio),
        "min_nut_peg_xy_reduction_ratio": float(min_xy_reduction),
        "original_nut_displacement_after_grasp": orig_disp,
        "best_nut_displacement_after_grasp": best_disp,
        "original_min_nut_peg_xy": orig_min_xy,
        "best_min_nut_peg_xy": best_min_xy,
        "original_nut_lift_delta": float(original.get("nut_lift_delta", 0.0)),
        "best_nut_lift_delta": float(best.get("nut_lift_delta", 0.0)),
    }


def diagnose_grasp_failure_reason(result: dict[str, Any], baseline: dict[str, Any]) -> str:
    if result.get("success_flag"):
        return "success"
    if bool(result.get("grasp_success_proxy")):
        return "grasp_improved_transport_blocked"
    if float(result.get("nut_lift_delta", 0.0)) < 0.01:
        return "lift_failed"
    if float(result.get("eef_nut_distance_at_grasp", 1.0)) > 0.08:
        return "misaligned_grasp"
    if float(result.get("nut_displacement_after_grasp", 0.0)) <= float(
        baseline.get("nut_displacement_after_grasp", 0.0)
    ) * 1.05:
        return "nut_not_picked"
    if float(result.get("min_nut_peg_xy", 1.0)) >= float(baseline.get("min_nut_peg_xy", 1.0)) * 0.95:
        return "transport_not_started"
    return "grasp_partial_improvement"


def summarize_effective_grasp_params(candidates: list[dict[str, Any]], top_k: int = 10) -> dict[str, Any]:
    if not candidates:
        return {}
    ranked = sorted(
        candidates,
        key=lambda row: (
            not row.get("success_flag", False),
            not row.get("grasp_success_proxy", False),
            row.get("search_score", row.get("score", 1e9)),
        ),
    )
    top = ranked[:top_k]
    counts: dict[str, dict[str, int]] = {}
    for row in top:
        params = row.get("grasp_params") or {}
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


def compute_grasp_search_score(result: dict[str, Any], scoring_mode: str = "grasp_heuristic") -> float:
    if scoring_mode == "random":
        return float(result.get("_random_score", 0.0))
    if scoring_mode in ("energy_full", "explicit_energy"):
        return explicit_energy_from_result(result)
    if scoring_mode == "pinn_energy":
        import sys
        from pathlib import Path

        v1_dir = Path(__file__).resolve().parent / "v1_residual_model"
        if str(v1_dir) not in sys.path:
            sys.path.insert(0, str(v1_dir))
        from pinn_inference import score_rollout_with_pinn

        return score_rollout_with_pinn(result, stage="grasp")
    return compute_grasp_score(result)


def explicit_energy_from_result(result: dict[str, Any]) -> float:
    xy = float(result.get("E_xy_norm", 0.0))
    transport = float(result.get("E_transport_norm", 0.0))
    yaw = float(result.get("E_yaw_norm", 0.0))
    z = float(result.get("E_z_norm", 0.0))
    smooth = float(result.get("E_smooth_norm", 0.0))
    return 3 * xy + 3 * transport + 2 * yaw + 2 * z + 0.2 * smooth


def run_grasp_search(
    hdf5_path: str,
    demo_key: str,
    label: str,
    *,
    mode: str = "random",
    max_evals: int = 80,
    seed: int = 0,
    scoring_mode: str = "grasp_heuristic",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import random as random_mod

    results: list[dict[str, Any]] = []
    best_so_far: dict[str, Any] | None = None
    eval_count_to_best = 0
    rng = random_mod.Random(seed + 7919 if scoring_mode == "random" else seed)

    for index, params in enumerate(iter_grasp_candidates(mode=mode, max_evals=max_evals, seed=seed)):
        result = execute_grasp_rollout(
            hdf5_path,
            demo_key,
            label,
            params,
            rollout_kind="grasp_sim_search",
        )
        result["search_index"] = index
        if scoring_mode == "random":
            result["_random_score"] = rng.random()
        result["search_score"] = compute_grasp_search_score(result, scoring_mode)
        result["seed"] = seed
        result["scoring_mode"] = scoring_mode
        results.append(result)

        if best_so_far is None:
            best_so_far = result
            eval_count_to_best = index + 1
        else:
            candidate_key = (
                not result.get("success_flag", False),
                not result.get("grasp_success_proxy", False),
                result["search_score"],
            )
            best_key = (
                not best_so_far.get("success_flag", False),
                not best_so_far.get("grasp_success_proxy", False),
                best_so_far["search_score"],
            )
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
