"""demo_3 insertion/contact 阶段局部 repair（partial success 后，不改 grasp/transport 前段）。"""
from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from grasp_sim_search import compute_grasp_proxies, get_sim_nut_pos
from insertion_residuals import INSERTION_RESIDUAL_KEYS, compute_insertion_residuals
from lift_contact_diagnostics import LiftContactTracker
from lift_sim_search import apply_lift_step_overlay
from lift_waypoint_refiner import LiftRepairParams, build_lift_waypoints_from_hdf5, lift_params_from_dict
from osc_action_converter import SimLoopParams, apply_sim_loop_step_overlay, compute_closed_loop_waypoint_action
from refined_waypoint_builder import _rotate_z_mat
from robosuite_env_loader import (
    create_env_from_metadata,
    extract_sim_features,
    get_sim_eef_pose4,
    load_demo_rollout_data,
    read_env_metadata,
    reset_env_to_demo_state,
    rollout_metrics_to_features_dict,
)
from rollout_outcome_evaluator import evaluate_rollout_outcome

FEATURE_FLAG_ENV = "enable_insertion_stage_repair"
FEATURE_FLAG_VALUE = "true"

INSERTION_STAGE_REPAIR_WEIGHTS = {
    "E_insert_depth": 3.0,
    "E_axis_alignment": 2.5,
    "E_vertical_approach": 2.0,
    "E_final_pose": 3.0,
    "E_jamming": 1.5,
    "contact_stability": 1.0,
}


def is_insertion_stage_repair_enabled(explicit: bool | None = None) -> bool:
    import os

    if explicit is not None:
        return explicit
    return os.environ.get(FEATURE_FLAG_ENV, "").strip().lower() == FEATURE_FLAG_VALUE


@dataclass
class InsertionStageRepairParams:
    """insertion 窗口局部搜索参数（不改 grasp/transport 前段）。"""

    approach_height: float = 0.02
    vertical_insertion_delta: float = -0.06
    xy_micro_x: float = 0.0
    xy_micro_y: float = 0.0
    axis_alignment: float = 0.0
    insertion_speed: float = 1.0
    settling_steps: int = 10
    z_gain: float = 0.55
    insertion_steps: int = 30
    pre_insert_pause: int = 5

    def to_sim_loop_params(self) -> SimLoopParams:
        return SimLoopParams(
            insert_z_offset=float(self.vertical_insertion_delta),
            z_gain=float(self.z_gain),
            insertion_steps=int(self.insertion_steps),
            hold_steps=int(self.settling_steps),
            insertion_speed_scale=float(self.insertion_speed),
            release_shift=0.0,
            pre_insert_pause=int(self.pre_insert_pause),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


INSERTION_STAGE_SEARCH_SPACE: dict[str, list[float | int]] = {
    "approach_height": [0.0, 0.01, 0.02, 0.03, 0.04],
    "vertical_insertion_delta": [-0.04, -0.06, -0.08, -0.10, -0.12],
    "xy_micro_x": [-0.015, -0.008, 0.0, 0.008, 0.015],
    "xy_micro_y": [-0.015, -0.008, 0.0, 0.008, 0.015],
    "axis_alignment": [-0.08, -0.04, 0.0, 0.04, 0.08],
    "insertion_speed": [0.5, 0.75, 1.0],
    "settling_steps": [5, 10, 15, 20],
    "z_gain": [0.45, 0.55, 0.70, 0.85],
    "insertion_steps": [15, 20, 30, 40],
    "pre_insert_pause": [0, 5, 10],
}


def iter_insertion_stage_candidates(
    *,
    mode: str = "random",
    max_evals: int = 40,
    seed: int = 0,
) -> Iterator[InsertionStageRepairParams]:
    keys = list(INSERTION_STAGE_SEARCH_SPACE.keys())
    if mode == "grid":
        combos = list(itertools.product(*(INSERTION_STAGE_SEARCH_SPACE[k] for k in keys)))
        rng = random.Random(seed)
        if len(combos) > max_evals:
            combos = rng.sample(combos, max_evals)
        for combo in combos:
            yield InsertionStageRepairParams(**dict(zip(keys, combo)))
    else:
        rng = random.Random(seed)
        for _ in range(max_evals):
            yield InsertionStageRepairParams(**{k: rng.choice(INSERTION_STAGE_SEARCH_SPACE[k]) for k in keys})


def apply_insertion_stage_waypoint_offsets(
    target_eef: np.ndarray,
    proxy: Any,
    params: InsertionStageRepairParams,
) -> np.ndarray:
    """仅修改 insertion 窗口及 approach 段 waypoint，不动 grasp/transport 前段。"""
    out = np.asarray(target_eef, dtype=float).copy()
    t_min = int(proxy.phases.t_min_xy)
    final_idx = int(proxy.phases.final_index)
    approach_start = max(int(proxy.phases.grasp_index), t_min - 10)

    for step in range(approach_start, t_min):
        alpha = (step - approach_start) / max(1, t_min - approach_start)
        out[step, 2, 3] += float(params.approach_height) * alpha

    denom = max(1, final_idx - t_min)
    for step in range(t_min, len(out)):
        gamma = (step - t_min) / denom
        out[step, 0, 3] += float(params.xy_micro_x) * gamma
        out[step, 1, 3] += float(params.xy_micro_y) * gamma
        out[step, :3, :3] = _rotate_z_mat(out[step, :3, :3], float(params.axis_alignment) * gamma)
    return out


def _contact_stability_component(
    rollout: dict[str, Any],
) -> tuple[float, bool, dict[str, Any]]:
    """返回 (penalty, diagnostic_only, meta)。measured contact 才进入目标函数。"""
    left = int(rollout.get("left_finger_contact_count", 0) or 0)
    right = int(rollout.get("right_finger_contact_count", 0) or 0)
    bilateral = int(rollout.get("bilateral_contact_steps", 0) or 0)
    measured = left > 0 or right > 0 or bilateral > 0
    meta = {
        "left_finger_contact_count": left,
        "right_finger_contact_count": right,
        "bilateral_contact_steps": bilateral,
        "contact_measured": measured,
    }
    if not measured:
        return 0.0, True, meta
    bilateral_target = 3.0
    penalty = float(max(0.0, 1.0 - bilateral / bilateral_target))
    penalty += float(max(0.0, 0.5 - (left + right) / 12.0))
    return penalty, False, meta


def compute_insertion_stage_objective(
    rollout: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ibr = compute_insertion_residuals(rollout, context)
    contact_penalty, contact_diagnostic_only, contact_meta = _contact_stability_component(rollout)
    residual_score = float(
        sum(INSERTION_STAGE_REPAIR_WEIGHTS[k] * ibr["residuals"][k]["normalized"] for k in INSERTION_RESIDUAL_KEYS)
        / sum(INSERTION_STAGE_REPAIR_WEIGHTS[k] for k in INSERTION_RESIDUAL_KEYS)
    )
    objective = residual_score
    if not contact_diagnostic_only:
        objective += INSERTION_STAGE_REPAIR_WEIGHTS["contact_stability"] * contact_penalty
    if rollout.get("success_flag"):
        objective -= 2.0
    return {
        "objective_score": float(objective),
        "residual_score": residual_score,
        "contact_penalty": contact_penalty,
        "contact_diagnostic_only": contact_diagnostic_only,
        "contact_meta": contact_meta,
        "insertion_residuals": ibr["residuals"],
        "insertion_total_score": ibr["insertion_total_score"],
    }


def classify_insertion_stage_failure(
    rollout: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> str:
    if rollout.get("success_flag"):
        return "success"
    ibr = compute_insertion_residuals(rollout, context)
    r = ibr["residuals"]
    if r["E_jamming"]["normalized"] >= 0.75:
        return "insertion_jamming"
    if r["E_axis_alignment"]["normalized"] >= 0.45:
        return "insertion_axis_misalignment"
    if r["E_vertical_approach"]["normalized"] >= 0.35:
        return "insertion_vertical_approach_error"
    if r["E_insert_depth"]["normalized"] >= 0.40:
        return "insertion_depth_error"
    if r["E_final_pose"]["normalized"] >= 0.40:
        return "insertion_final_pose_error"
    contact_penalty, diagnostic_only, _ = _contact_stability_component(rollout)
    if not diagnostic_only and contact_penalty >= 0.6:
        return "insertion_contact_unstable"
    outcome = evaluate_rollout_outcome(rollout, context)
    if outcome.get("partial_success") and not outcome.get("final_success"):
        return "partial_success_not_final"
    return str(outcome.get("failure_reason") or rollout.get("failure_guess") or "unknown_failed")


def execute_demo3_insertion_stage_repair_rollout(
    *,
    failed_hdf5: str | Path,
    demo_key: str,
    grasp_lift: dict[str, float],
    lift_extra: dict[str, float] | None = None,
    repair_params: InsertionStageRepairParams | None = None,
    rollout_kind: str = "insertion_stage_repair",
) -> dict[str, Any]:
    """
    demo_3 二阶段 rollout：
    - grasp/transport/lift 前段：固定 lift repair waypoint + lift overlay
    - partial success 后 insertion 段：局部 waypoint 微调 + sim_loop overlay
    """
    params = repair_params or InsertionStageRepairParams()
    sim_params = params.to_sim_loop_params()
    merged = {**grasp_lift, **(lift_extra or {})}
    lift_params = lift_params_from_dict(merged)
    proxy, _original_eef, target_eef, gripper = build_lift_waypoints_from_hdf5(
        str(failed_hdf5), demo_key, "failed", lift_params
    )
    target_eef = apply_insertion_stage_waypoint_offsets(target_eef, proxy, params)

    demo = load_demo_rollout_data(str(failed_hdf5), demo_key, "failed")
    env_args = read_env_metadata(str(failed_hdf5))
    grasp_idx = int(proxy.phases.grasp_index)
    t_min = int(proxy.phases.t_min_xy)

    build = create_env_from_metadata(env_args, for_video=False)
    env = build.env
    reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)
    contact_tracker = LiftContactTracker(env)

    length = len(target_eef)
    grip = np.asarray(gripper, dtype=float).reshape(-1)
    actions = np.zeros((length, 7), dtype=np.float64)

    nut_positions: list[np.ndarray] = []
    nut_z_trace: list[float] = []
    eef_nut_distance_at_grasp = float("inf")
    nut_positions.append(get_sim_nut_pos(env).copy())

    min_xy = float("inf")
    min_yaw = float("inf")
    insertion_left = 0
    insertion_right = 0
    insertion_bilateral = 0

    lift_start = min(length - 1, grasp_idx + int(lift_params.post_grasp_settle_steps) + int(lift_params.lift_pause_steps) + 1)
    lift_end = min(length - 1, lift_start + int(lift_params.micro_lift_steps))
    contact_window_end = min(length - 1, grasp_idx + int(lift_params.contact_hold_steps))

    for step in range(length):
        target_idx = min(step + 1, length - 1)
        speed = float(lift_params.lift_speed_scale) if step < t_min else 1.0
        action = compute_closed_loop_waypoint_action(
            env, target_eef[target_idx], grip[step], env_args, speed_scale=speed
        )
        if step < t_min:
            action = apply_lift_step_overlay(action, step, proxy, lift_params)
        else:
            action = apply_sim_loop_step_overlay(action, step, proxy, grip, env_args, sim_params)
            left_c, right_c = contact_tracker._count_finger_contacts(env)
            insertion_left += left_c
            insertion_right += right_c
            if left_c and right_c:
                insertion_bilateral += 1
        actions[step] = action
        env.step(action)

        nut_pos = get_sim_nut_pos(env)
        nut_positions.append(nut_pos.copy())
        nut_z_trace.append(float(nut_pos[2]))
        if step == grasp_idx:
            eef_pos = get_sim_eef_pose4(env)[:3, 3]
            eef_nut_distance_at_grasp = float(np.linalg.norm(eef_pos - nut_pos))
        if step <= contact_window_end:
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

    grasp_step = min(grasp_idx, len(nut_positions) - 1)
    nut_z_at_grasp = nut_z_trace[grasp_step] if grasp_step < len(nut_z_trace) else nut_z_trace[-1]
    nut_lift_delta = (
        float(max(nut_z_trace[lift_start : lift_end + 1]) - nut_z_at_grasp)
        if lift_end > lift_start and lift_end < len(nut_z_trace)
        else 0.0
    )
    proxies = compute_grasp_proxies(
        nut_displacement_after_grasp=0.0,
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
    contact_diag = contact_tracker.finalize(
        env,
        lift_begin=lift_start,
        lift_end=lift_end,
        nut_z_trace=nut_z_trace,
        partial_lift_delta_thresh=0.005,
    )
    env.close()

    rollout = {
        "demo_name": demo_key,
        "source_file": str(failed_hdf5),
        "rollout_kind": rollout_kind,
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
        "nut_lift_delta": nut_lift_delta,
        "eef_nut_distance_at_grasp": eef_nut_distance_at_grasp,
        "grasp_success_proxy": proxies["grasp_success_proxy"],
        "lift_success_proxy": proxies["lift_success_proxy"],
        "left_finger_contact_count": max(contact_diag.left_finger_contact_count, insertion_left),
        "right_finger_contact_count": max(contact_diag.right_finger_contact_count, insertion_right),
        "bilateral_contact_steps": max(contact_diag.bilateral_contact_steps, insertion_bilateral),
        "contact_duration": contact_diag.contact_duration,
        "action_acceleration_max": acc_max,
        "insertion_stage_repair_params": params.to_dict(),
        "insertion_window_only": True,
        "repair_grasp_lift_params": grasp_lift,
        "repair_lift_extra_params": lift_extra or {},
    }
    rollout["insertion_failure_reason"] = classify_insertion_stage_failure(rollout)
    return rollout


def run_insertion_stage_local_search(
    *,
    failed_hdf5: str | Path,
    demo_key: str,
    grasp_lift: dict[str, float],
    lift_extra: dict[str, float] | None,
    context: dict[str, Any] | None,
    max_evals: int = 40,
    seed: int = 0,
) -> dict[str, Any]:
    """对单个 partial-success 候选做局部 CEM/随机搜索。"""
    records: list[dict[str, Any]] = []
    for idx, params in enumerate(iter_insertion_stage_candidates(mode="random", max_evals=max_evals, seed=seed)):
        rollout = execute_demo3_insertion_stage_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            grasp_lift=grasp_lift,
            lift_extra=lift_extra,
            repair_params=params,
        )
        obj = compute_insertion_stage_objective(rollout, context=context)
        outcome = evaluate_rollout_outcome(rollout, context)
        records.append(
            {
                "eval_index": idx,
                "repair_params": params.to_dict(),
                "objective": obj,
                "outcome": outcome,
                "insertion_failure_reason": rollout["insertion_failure_reason"],
                "rollout_summary": {
                    "success_flag": rollout["success_flag"],
                    "final_z_diff": rollout["final_z_diff"],
                    "min_yaw_error": rollout["min_yaw_error"],
                    "final_nut_peg_xy": rollout["final_nut_peg_xy"],
                },
            }
        )

    records.sort(
        key=lambda r: (
            not r["outcome"]["final_success"],
            r["objective"]["objective_score"],
        )
    )
    best = records[0]
    best_rollout = execute_demo3_insertion_stage_repair_rollout(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        grasp_lift=grasp_lift,
        lift_extra=lift_extra,
        repair_params=InsertionStageRepairParams(**best["repair_params"]),
    )
    return {
        "best": best,
        "best_rollout": best_rollout,
        "num_evals": len(records),
        "records_tail": records[:5],
    }
