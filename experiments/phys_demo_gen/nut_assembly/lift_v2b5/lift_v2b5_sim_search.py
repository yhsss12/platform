"""V2-B5：demo_3 lift_failed MuJoCo rollout 搜索。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from energy_model import classify_failure_type, compute_total_energy
from extract_features import NutAssemblyFeatures, action_acceleration_stats
from grasp_sim_search import compute_grasp_proxies, get_sim_nut_pos
from lift_energy_model import compute_lift_residual_energies
from lift_v2b5_refiner import (
  GRASP_SUCCESS_SEED_PARAMS,
  LIFT_V2B5_SEARCH_SPACE,
  LiftV2B5Params,
  build_lift_v2b5_waypoints_from_hdf5,
)
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


PARTIAL_LIFT_DELTA_THRESH = 0.008
PARTIAL_GRASP_DISP_THRESH = 0.04
LIFT_SUCCESS_DELTA_THRESH = 0.02


def _perturb_lift_params(base: LiftV2B5Params, rng: random.Random) -> LiftV2B5Params:
  raw = base.to_dict()
  for key, choices in LIFT_V2B5_SEARCH_SPACE.items():
    if rng.random() < 0.55:
      raw[key] = rng.choice(choices)
  return LiftV2B5Params(**raw)


def iter_lift_v2b5_candidates(
  *, max_evals: int, seed: int, local_fraction: float = 0.5
) -> Iterator[LiftV2B5Params]:
  rng = random.Random(seed)
  keys = list(LIFT_V2B5_SEARCH_SPACE.keys())
  n_local = int(max_evals * local_fraction)
  for i in range(max_evals):
    if i < n_local:
      yield _perturb_lift_params(GRASP_SUCCESS_SEED_PARAMS, rng)
    else:
      yield LiftV2B5Params(**{k: rng.choice(LIFT_V2B5_SEARCH_SPACE[k]) for k in keys})


def classify_lift_outcome(result: dict[str, Any]) -> str:
  if result.get("success_flag"):
    return "refined_success"
  if result.get("lift_success_proxy"):
    return "lift_success"
  if result.get("partial_lift_success"):
    return "partial_lift_success"
  if result.get("grasp_success_proxy"):
    return "grasp_improved_but_failed"
  return "lift_no_improvement"


def apply_lift_v2b5_step_overlay(
  action: np.ndarray,
  step: int,
  proxy: Any,
  params: LiftV2B5Params,
  phases: dict[str, int],
  *,
  stage1_lift_delta: float = 0.0,
) -> np.ndarray:
  out = action.copy()
  grasp_idx = phases["grasp_index"]
  lift_begin = phases["lift_begin"]
  stage1_end = phases["stage1_end"]
  pause_end = phases["pause_end"]
  stage2_end = phases["stage2_end"]

  if grasp_idx - 8 <= step <= grasp_idx + int(params.contact_settle_steps):
    out[2] *= 1.1 + abs(float(params.gripper_extra_close)) * 0.8
    out[6] = min(out[6], -0.92 - abs(float(params.gripper_extra_close)))

  # slow lift + force close during lift
  if lift_begin <= step <= stage2_end:
    out[2] *= 1.35 + float(params.lift_direction_bias_z) * 8.0
    out[2] *= max(0.45, float(params.lift_speed_scale))
    out[6] = min(out[6], -0.98)

  # micro-lift check: boost stage2 if stage1 failed to move nut
  if float(params.enable_two_stage_lift) > 0.5 and step > pause_end and stage1_lift_delta < float(
    params.micro_lift_check_threshold
  ):
    out[2] *= 1.35

  return np.clip(out, -1.0, 1.0)


def execute_lift_v2b5_rollout(
  hdf5_path: str,
  demo_key: str,
  label: str,
  params: LiftV2B5Params,
  *,
  rollout_kind: str = "lift_v2b5",
) -> dict[str, Any]:
  proxy, _orig, target_eef, gripper, phases = build_lift_v2b5_waypoints_from_hdf5(
    hdf5_path, demo_key, label, params
  )
  demo = load_demo_rollout_data(hdf5_path, demo_key, label)
  env_args = read_env_metadata(hdf5_path)
  grasp_idx = phases["grasp_index"]

  build = create_env_from_metadata(env_args, for_video=False)
  env = build.env
  reset_info = reset_env_to_demo_state(env, demo.states[0], model_xml=demo.model_xml)

  length = len(target_eef)
  grip = np.asarray(gripper, dtype=float).reshape(-1)
  actions = np.zeros((length, 7), dtype=float)

  nut_positions: list[np.ndarray] = []
  nut_z_trace: list[float] = []
  eef_nut_distance_at_grasp = float("inf")
  lift_window_distances: list[float] = []

  nut_positions.append(get_sim_nut_pos(env).copy())
  stage1_end = phases["stage1_end"]
  lift_begin = phases["lift_begin"]

  for step in range(length):
    nut_z_before = float(get_sim_nut_pos(env)[2]) if step == stage1_end else 0.0
    target_idx = min(step + 1, length - 1)
    action = compute_closed_loop_waypoint_action(
      env, target_eef[target_idx], grip[step], env_args, speed_scale=float(params.lift_speed_scale)
    )
    stage1_lift_delta = 0.0
    if step == stage1_end and lift_begin < len(nut_z_trace):
      nut_z_at_lift = nut_z_trace[lift_begin] if lift_begin < len(nut_z_trace) else nut_z_trace[-1]
      stage1_lift_delta = float(nut_z_before - nut_z_at_lift)
    action = apply_lift_v2b5_step_overlay(
      action, step, proxy, params, phases, stage1_lift_delta=stage1_lift_delta
    )
    actions[step] = action
    env.step(action)

    nut_pos = get_sim_nut_pos(env)
    nut_positions.append(nut_pos.copy())
    nut_z_trace.append(float(nut_pos[2]))
    if step == grasp_idx:
      eef_pos = get_sim_eef_pose4(env)[:3, 3]
      eef_nut_distance_at_grasp = float(np.linalg.norm(eef_pos - nut_pos))
    if lift_begin <= step <= phases["stage2_end"]:
      eef_pos = get_sim_eef_pose4(env)[:3, 3]
      lift_window_distances.append(float(np.linalg.norm(eef_pos - nut_pos)))

  nut_positions_arr = np.asarray(nut_positions)
  grasp_step = min(grasp_idx, len(nut_positions_arr) - 1)
  if grasp_step < len(nut_positions_arr) - 1:
    after = nut_positions_arr[grasp_step + 1 :]
    nut_displacement_after_grasp = (
      float(np.sum(np.linalg.norm(np.diff(after, axis=0), axis=1))) if len(after) > 1 else 0.0
    )
  else:
    nut_displacement_after_grasp = 0.0

  nut_z_at_grasp = nut_z_trace[grasp_step] if grasp_step < len(nut_z_trace) else nut_z_trace[-1]
  lift_end = min(len(nut_z_trace) - 1, phases["stage2_end"])
  lift_begin_idx = min(max(0, lift_begin), len(nut_z_trace) - 1)
  nut_z_at_lift_begin = nut_z_trace[lift_begin_idx]
  if lift_end > lift_begin_idx:
    lift_z_window = nut_z_trace[lift_begin_idx : lift_end + 1]
    nut_lift_delta = float(max(lift_z_window) - nut_z_at_grasp)
    nut_lift_phase_delta = float(max(lift_z_window) - nut_z_at_lift_begin)
  else:
    nut_lift_delta = 0.0
    nut_lift_phase_delta = 0.0
  nut_z_std = float(np.std(nut_z_trace[lift_begin : lift_end + 1])) if lift_end > lift_begin else 0.0

  follow_thresh = float(params.nut_follow_threshold)
  lift_follow_score = (
    float(np.clip(1.0 - float(np.mean(lift_window_distances)) / max(follow_thresh, 1e-6), 0.0, 1.0))
    if lift_window_distances
    else 0.0
  )

  proxies = compute_grasp_proxies(
    nut_displacement_after_grasp=nut_displacement_after_grasp,
    nut_lift_delta=nut_lift_delta,
    eef_nut_distance_at_grasp=eef_nut_distance_at_grasp,
  )
  partial_lift_success = nut_lift_phase_delta >= PARTIAL_LIFT_DELTA_THRESH
  partial_grasp_success = nut_displacement_after_grasp >= PARTIAL_GRASP_DISP_THRESH
  lift_success_proxy = (
    nut_lift_phase_delta >= LIFT_SUCCESS_DELTA_THRESH or bool(proxies["lift_success_proxy"])
  )

  final_metrics = extract_sim_features(env)
  acc_mean, acc_max = action_acceleration_stats(actions)
  feat_dict = rollout_metrics_to_features_dict(
    demo_key, label, hdf5_path, final_metrics, acc_max, len(actions)
  )
  features = NutAssemblyFeatures(**feat_dict)
  energy = compute_total_energy(features)
  success_flag = bool(env._check_success())
  env.close()

  result: dict[str, Any] = {
    "demo_name": demo_key,
    "label": label,
    "source_file": hdf5_path,
    "rollout_kind": rollout_kind,
    "lift_v2b5_params": params.to_dict(),
    "success_flag": success_flag,
    "outcome_label": "",
    "partial_lift_success": partial_lift_success,
    "lift_success_proxy": lift_success_proxy,
    "grasp_success_proxy": proxies["grasp_success_proxy"],
    "nut_lift_delta": nut_lift_delta,
    "nut_lift_phase_delta": nut_lift_phase_delta,
    "nut_z_std_during_lift": nut_z_std,
    "lift_follow_score": lift_follow_score,
    "nut_displacement_after_grasp": nut_displacement_after_grasp,
    "eef_nut_distance_at_grasp": eef_nut_distance_at_grasp,
    "final_nut_peg_xy": final_metrics["final_nut_peg_xy"],
    "min_nut_peg_xy": final_metrics.get("min_nut_peg_xy", final_metrics["final_nut_peg_xy"]),
    "final_z_diff": final_metrics["final_z_diff"],
    "failure_guess": classify_failure_type(features, energy.E_smooth),
    "E_total_norm": energy.E_total_norm,
    "E_xy_norm": energy.E_xy_norm,
    "E_transport_norm": energy.E_transport_norm,
    "E_yaw_norm": energy.E_yaw_norm,
    "E_z_norm": energy.E_z_norm,
    "E_smooth_norm": energy.E_smooth_norm,
    "object_poses_modified": False,
    "reset_info": reset_info,
    "env_warnings": build.warnings,
  }
  result.update(compute_lift_residual_energies(result))
  result["outcome_label"] = classify_lift_outcome(result)
  return result
