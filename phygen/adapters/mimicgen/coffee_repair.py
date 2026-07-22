#!/usr/bin/env python3
"""CoffeePreparation failed-conditioned MimicGen repair helpers and feedback generation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

MIMICGEN_ROOT = Path(os.environ.get("PHYGEN_MIMICGEN_ROOT", "third_party/mimicgen")).resolve()
DEFAULT_ROBOSUITE_PYTHON = os.environ.get(
    "PHYGEN_ROBOSUITE_PYTHON",
    "/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python",
)

# CoffeePreparation object-state layout: 6 objects × 14 dims + hinge + drawer joint.
_OBJECT_BLOCK = 14
_OBJECT_OFFSETS = {
    "coffee_pod": 0,
    "coffee_machine": 14,
    "coffee_pod_holder": 28,
    "coffee_machine_lid": 42,
    "drawer": 56,
    "mug": 70,
}
_HINGE_ANGLE_IDX = 84
_DRAWER_JOINT_IDX = 85
_OBJECT_OBS_DIM = 86

TABLE_Z = 0.805
CONTEXT_KEYS = [
    "energy",
    "pod_xy",
    "pod_z",
    "mug_xy",
    "mug_z",
    "machine_xy",
    "stage_progress",
    "drop_penalty",
    "task_order_penalty",
]


def demo_sort_key(key: str) -> int:
    m = re.search(r"(\d+)$", str(key))
    return int(m.group(1)) if m else 0


def _object_pos(obs_vec: np.ndarray, name: str) -> np.ndarray:
    start = _OBJECT_OFFSETS[name]
    return np.asarray(obs_vec[start : start + 3], dtype=np.float64)


def _blend_metrics(failed: dict[str, float], success: dict[str, float], alpha: float) -> dict[str, float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = {}
    for key in CONTEXT_KEYS:
        out[key] = float(failed.get(key, 0.0) * (1.0 - alpha) + success.get(key, 0.0) * alpha)
    return out


def compute_context_metrics(
    obs_vec: np.ndarray,
    *,
    action_delta: float = 0.0,
) -> dict[str, float]:
    """Extract coffee_preparation context metrics from a single object observation vector."""
    obs_vec = np.asarray(obs_vec, dtype=np.float64)
    if obs_vec.shape[0] < _OBJECT_OBS_DIM:
        raise ValueError(f"Expected object obs dim >= {_OBJECT_OBS_DIM}, got {obs_vec.shape[0]}")

    pod = _object_pos(obs_vec, "coffee_pod")
    holder = _object_pos(obs_vec, "coffee_pod_holder")
    machine = _object_pos(obs_vec, "coffee_machine")
    mug = _object_pos(obs_vec, "mug")
    hinge = float(obs_vec[_HINGE_ANGLE_IDX])
    drawer_joint = float(obs_vec[_DRAWER_JOINT_IDX])

    pod_xy = float(np.linalg.norm(pod[:2] - holder[:2]))
    pod_z = float(abs(pod[2] - holder[2]))
    mug_xy = float(np.linalg.norm(mug[:2] - machine[:2]))
    mug_z = float(abs(mug[2] - machine[2]))
    machine_xy = float(np.linalg.norm(machine[:2]))

    drawer_open = float(np.clip((-drawer_joint) / 0.20, 0.0, 1.0))
    mug_grasped = float(np.clip((mug[2] - TABLE_Z - 0.02) / 0.08, 0.0, 1.0))
    mug_placed = float(np.clip(1.0 - mug_xy / 0.08, 0.0, 1.0)) * float(np.clip(1.0 - mug_z / 0.05, 0.0, 1.0))
    lid_open = float(np.clip((hinge - 1.8) / 1.0, 0.0, 1.0))
    pod_near_holder = float(np.clip(1.0 - pod_xy / 0.06, 0.0, 1.0))
    pod_inserted = float(np.clip(1.0 - pod_z / 0.04, 0.0, 1.0)) * pod_near_holder

    stage_flags = [drawer_open, mug_grasped, mug_placed, lid_open, pod_near_holder, pod_inserted]
    stage_progress = float(np.mean(stage_flags))

    min_z = float(min(pod[2], mug[2], machine[2]))
    drop_penalty = float(max(0.0, TABLE_Z - 0.04 - min_z))

    task_order_penalty = 0.0
    if pod_inserted > 0.35 and mug_placed < 0.35:
        task_order_penalty += 0.45
    if lid_open > 0.35 and mug_placed < 0.25:
        task_order_penalty += 0.25
    if pod_near_holder > 0.35 and drawer_open < 0.20:
        task_order_penalty += 0.20

    raw_energy = (
        4.5 * pod_xy / 0.08
        + 3.5 * pod_z / 0.05
        + 4.0 * mug_xy / 0.10
        + 3.0 * mug_z / 0.06
        + 2.0 * machine_xy / 0.35
        + 6.0 * (1.0 - stage_progress)
        + 8.0 * drop_penalty / 0.05
        + 5.0 * task_order_penalty
        + 1.5 * action_delta
    )
    energy = float(np.clip(raw_energy, 0.0, 30.0))

    return {
        "energy": energy,
        "pod_xy": pod_xy,
        "pod_z": pod_z,
        "mug_xy": mug_xy,
        "mug_z": mug_z,
        "machine_xy": machine_xy,
        "stage_progress": stage_progress,
        "drop_penalty": drop_penalty,
        "task_order_penalty": task_order_penalty,
    }


def _action_delta(actions: np.ndarray, idx: int) -> float:
    if actions is None or len(actions) < 2:
        return 0.0
    idx = int(np.clip(idx, 1, len(actions) - 1))
    return float(np.linalg.norm(actions[idx] - actions[idx - 1]))


def _pick_failure_frame(obs: np.ndarray, actions: np.ndarray | None) -> int:
    best_idx = 0
    best_energy = -1.0
    for idx in range(len(obs)):
        metrics = compute_context_metrics(obs[idx], action_delta=_action_delta(actions, idx))
        if metrics["energy"] > best_energy:
            best_energy = metrics["energy"]
            best_idx = idx
    return int(best_idx)


def _pick_success_frame(obs: np.ndarray, actions: np.ndarray | None) -> int:
    best_idx = 0
    best_energy = float("inf")
    for idx in range(len(obs)):
        metrics = compute_context_metrics(obs[idx], action_delta=_action_delta(actions, idx))
        if metrics["energy"] < best_energy:
            best_energy = metrics["energy"]
            best_idx = idx
    return int(best_idx)


def _idealize_success_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Map near-success demo frames to a low-energy repair target for proxy supervision."""
    ideal = dict(metrics)
    for key in ("pod_xy", "pod_z", "mug_xy", "mug_z", "drop_penalty", "task_order_penalty"):
        ideal[key] = float(metrics.get(key, 0.0) * 0.20)
    ideal["stage_progress"] = float(max(metrics.get("stage_progress", 0.0), 0.88))
    ideal["machine_xy"] = float(metrics.get("machine_xy", 0.0))
    ideal["energy"] = float(
        np.clip(
            4.5 * ideal["pod_xy"] / 0.08
            + 3.5 * ideal["pod_z"] / 0.05
            + 4.0 * ideal["mug_xy"] / 0.10
            + 3.0 * ideal["mug_z"] / 0.06
            + 2.0 * ideal["machine_xy"] / 0.35
            + 6.0 * (1.0 - ideal["stage_progress"])
            + 8.0 * ideal["drop_penalty"] / 0.05
            + 5.0 * ideal["task_order_penalty"],
            0.0,
            30.0,
        )
    )
    return ideal


def score_repair_theta(theta: dict[str, Any]) -> float:
    """Heuristic quality score in [0, 1] for proxy rollout evaluation."""
    score = 0.45
    if theta.get("selection_strategy") == "nearest_neighbor_object":
        score += 0.18
    if bool(theta.get("interpolate_from_last_target_pose", True)):
        score += 0.10
    if not bool(theta.get("transform_first_robot_pose", False)):
        score += 0.05
    if bool(theta.get("select_src_per_subtask", False)):
        score += 0.05

    noise = float(theta.get("action_noise", 0.05))
    score += 0.12 * float(np.clip(1.0 - noise / 0.10, 0.0, 1.0))

    interp = int(theta.get("num_interpolation_steps", 5))
    score += 0.08 * float(np.clip(interp / 15.0, 0.0, 1.0))

    fixed = int(theta.get("num_fixed_steps", 0))
    score -= 0.05 * float(np.clip(fixed / 3.0, 0.0, 1.0))

    offset = theta.get("offset_range", [10, 20])
    width = abs(float(offset[1]) - float(offset[0]))
    score += 0.07 * float(np.clip(width / 20.0, 0.0, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def evaluate_repair_proxy(
    theta: dict[str, Any],
    failed_metrics: dict[str, float],
    success_metrics: dict[str, float],
    rng: np.random.Generator,
) -> tuple[dict[str, float], bool]:
    """Proxy repair rollout: blend failed/success metrics using repair-theta quality."""
    quality = score_repair_theta(theta)
    jitter = float(rng.normal(0.0, 0.06))
    alpha = float(np.clip(quality + jitter, 0.0, 1.0))
    metrics = _blend_metrics(failed_metrics, success_metrics, alpha)

    success = (
        metrics["energy"] <= 10.5
        and metrics["stage_progress"] >= 0.80
        and metrics["task_order_penalty"] <= 0.08
        and quality >= 0.62
    )
    if not success and quality < 0.42:
        metrics = _blend_metrics(failed_metrics, success_metrics, alpha * 0.35)
        metrics["energy"] = float(max(metrics["energy"], failed_metrics["energy"] * 0.85))
    return metrics, bool(success)


def make_candidate(candidate_index: int, rng: np.random.Generator) -> dict[str, Any]:
    offset_options = [[10, 20], [10, 15], [15, 20], [5, 20], [10, 25], [0, 20], [15, 25]]
    return {
        "candidate_index": int(candidate_index),
        "selection_strategy": str(rng.choice(["nearest_neighbor_object", "random"], p=[0.8, 0.2])),
        "select_src_per_subtask": bool(rng.random() < 0.8),
        "transform_first_robot_pose": bool(rng.random() < 0.1),
        "interpolate_from_last_target_pose": bool(rng.random() < 0.9),
        "action_noise": float(rng.choice([0.0, 0.01, 0.02, 0.05, 0.08])),
        "num_interpolation_steps": int(rng.choice([3, 5, 8, 10, 15])),
        "num_fixed_steps": int(rng.choice([0, 1, 2])),
        "offset_range": offset_options[int(rng.integers(0, len(offset_options)))],
        "nn_k": int(rng.choice([1, 3, 5, 10])),
    }


def load_demo_states_actions(hdf5_path: str | Path, demo_key: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        states = np.asarray(demo["states"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float64)
    return states, actions


def check_sim_runtime(mimicgen_root: str | Path | None = None) -> dict[str, Any]:
    """Probe robosuite + MimicGen coffee_preparation runtime availability."""
    root = Path(mimicgen_root or MIMICGEN_ROOT)
    info: dict[str, Any] = {
        "mimicgen_root": str(root),
        "mimicgen_root_exists": root.is_dir(),
        "robosuite_importable": False,
        "mimicgen_importable": False,
        "coffee_preparation_env_loadable": False,
        "recommended_python": DEFAULT_ROBOSUITE_PYTHON,
        "missing_packages": [],
        "notes": [],
    }
    if str(root) not in sys.path and root.is_dir():
        sys.path.insert(0, str(root))

    try:
        import robosuite  # noqa: F401

        info["robosuite_importable"] = True
        info["notes"].append(f"robosuite={robosuite.__version__}")
    except Exception as exc:
        info["missing_packages"].append("robosuite")
        info["notes"].append(f"robosuite import failed: {exc}")

    try:
        import mimicgen  # noqa: F401

        info["mimicgen_importable"] = True
    except Exception as exc:
        info["missing_packages"].append("mimicgen")
        info["notes"].append(f"mimicgen import failed: {exc}")

    if info["robosuite_importable"] and info["mimicgen_importable"]:
        try:
            os.environ.setdefault("MUJOCO_GL", "egl")
            from mimicgen.envs.robosuite.coffee import CoffeePreparation_D0

            env = CoffeePreparation_D0(
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                ignore_done=True,
                horizon=2000,
            )
            env.reset()
            info["coffee_preparation_env_loadable"] = True
            info["notes"].append(f"coffee env action_dim={env.action_dim}")
            env.close()
        except Exception as exc:
            info["notes"].append(f"CoffeePreparation env load failed: {exc}")

    info["ready_for_true_rollout"] = bool(info["coffee_preparation_env_loadable"])
    if not info["ready_for_true_rollout"]:
        info["recommended_install"] = (
            "Use conda env with robosuite 1.4.x, then: "
            f"pip install -e {root} && pip install robosuite==1.4.1 robomimic h5py mujoco"
        )
    return info


def _ensure_mimicgen_path(mimicgen_root: str | Path | None = None) -> Path:
    root = Path(mimicgen_root or MIMICGEN_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(f"MimicGen root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _build_coffee_env():
    os.environ.setdefault("MUJOCO_GL", "egl")
    _ensure_mimicgen_path()
    from mimicgen.envs.robosuite.coffee import CoffeePreparation_D0

    return CoffeePreparation_D0(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        horizon=2000,
    )


def _pad_action(env: Any, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] == env.action_dim:
        return action
    if action.shape[0] == env.action_dim - 1:
        return np.concatenate([action, np.zeros(1, dtype=np.float64)])
    raise ValueError(f"Unexpected action dim {action.shape[0]} for env dim {env.action_dim}")


def _reset_env_to_state(env: Any, state_vec: np.ndarray) -> None:
    state_vec = np.asarray(state_vec, dtype=np.float64)
    sim_dim = len(env.sim.get_state().flatten())
    if sim_dim != len(state_vec):
        raise ValueError(f"state dimension mismatch: demo={len(state_vec)} sim={sim_dim}")
    env.sim.reset()
    env.sim.set_state_from_flattened(state_vec)
    env.sim.forward()


def _collect_sim_aux_metrics(env: Any, action_deltas: list[float]) -> dict[str, float]:
    out = {
        "contact_residual": 0.0,
        "penetration": 0.0,
        "object_slip": 0.0,
        "velocity_jump": float(max(action_deltas) if action_deltas else 0.0),
        "action_smoothness": float(np.mean(action_deltas) if action_deltas else 0.0),
    }
    try:
        robot = env.robots[0]
        robot_geoms = set(getattr(robot.robot_model, "contact_geoms", []) or [])
        contact_count = 0
        max_pen = 0.0
        ncon = int(env.sim.data.ncon)
        for i in range(ncon):
            con = env.sim.data.contact[i]
            g1 = env.sim.model.geom_id2name(con.geom1)
            g2 = env.sim.model.geom_id2name(con.geom2)
            if g1 in robot_geoms or g2 in robot_geoms:
                contact_count += 1
                if con.dist < 0:
                    max_pen = max(max_pen, float(-con.dist))
        out["contact_residual"] = float(contact_count)
        out["penetration"] = float(max_pen)
    except Exception:
        pass

    try:
        slip_vals = []
        for name in ("coffee_pod", "mug"):
            body_id = env.obj_body_id.get(name)
            if body_id is not None:
                vel = env.sim.data.body_xvelp[body_id]
                slip_vals.append(float(np.linalg.norm(vel[:2])))
        out["object_slip"] = float(np.mean(slip_vals) if slip_vals else 0.0)
    except Exception:
        pass
    return out


def _theta_replay_plan(theta: dict[str, Any], failure_frame: int, num_actions: int) -> dict[str, int]:
    offset = theta.get("offset_range", [10, 20])
    lo, hi = int(offset[0]), int(offset[1])
    center = int(round(0.5 * (lo + hi)))
    if theta.get("selection_strategy") == "random":
        center = int(round(center * 1.15))
    start = int(np.clip(failure_frame + max(0, center // 2), 0, max(0, num_actions - 2)))
    window = max(1, int(theta.get("num_interpolation_steps", 5) // 2))
    fixed = max(0, int(theta.get("num_fixed_steps", 0)))
    return {"start": start, "window": window, "fixed": fixed}


def run_true_mimicgen_rollout(
    hdf5_path: str | Path,
    demo_key: str,
    theta: dict[str, Any],
    *,
    failure_frame: int,
    seed: int = 0,
    max_steps: int = 400,
    mimicgen_root: str | Path | None = None,
) -> tuple[dict[str, float], bool, dict[str, Any]]:
    """Execute a robosuite replay rollout conditioned on repair theta."""
    _ensure_mimicgen_path(mimicgen_root)
    states, actions = load_demo_states_actions(hdf5_path, demo_key)
    env = _build_coffee_env()
    rng = np.random.default_rng(seed + int(theta.get("candidate_index", 0)))

    fail_idx = int(np.clip(failure_frame, 0, len(states) - 1))
    _reset_env_to_state(env, states[fail_idx])

    plan = _theta_replay_plan(theta, fail_idx, len(actions))
    start = plan["start"]
    window = plan["window"]
    fixed = plan["fixed"]
    noise = float(theta.get("action_noise", 0.05))

    base_action = _pad_action(env, actions[start])
    for _ in range(fixed):
        step_action = base_action.copy()
        if noise > 0:
            step_action = np.clip(step_action + rng.normal(0.0, noise, size=step_action.shape), -1.0, 1.0)
        env.step(step_action)

    action_deltas: list[float] = []
    prev_action = None
    end = min(len(actions), start + max_steps)
    executed = 0
    for i in range(start, end):
        lo = max(start, i - window + 1)
        chunk = actions[lo : i + 1]
        action = _pad_action(env, np.mean(chunk, axis=0))
        if noise > 0:
            action = action + rng.normal(0.0, noise, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        if prev_action is not None:
            action_deltas.append(float(np.linalg.norm(action - prev_action)))
        prev_action = action.copy()
        try:
            env.step(action)
            executed += 1
        except ValueError:
            break

    obs = env._get_observations(force_update=True)
    object_state = obs.get("object-state")
    if object_state is None:
        object_state = obs.get("object")
    metrics = compute_context_metrics(
        np.asarray(object_state, dtype=np.float64),
        action_delta=action_deltas[-1] if action_deltas else 0.0,
    )
    metrics.update(_collect_sim_aux_metrics(env, action_deltas))
    task_metrics = env._get_partial_task_metrics()
    success = bool(task_metrics.get("task", False))
    rollout_info = {
        "repair_eval_mode": "true_mimicgen_rollout",
        "failure_frame": fail_idx,
        "replay_start": start,
        "replay_window": window,
        "num_fixed_steps_applied": fixed,
        "executed_steps": executed,
        "task_metrics": {k: bool(v) if isinstance(v, (bool, np.bool_)) else float(v) for k, v in task_metrics.items()},
    }
    env.close()
    return metrics, success, rollout_info


def evaluate_repair_true_rollout(
    hdf5_path: str | Path,
    demo_key: str,
    theta: dict[str, Any],
    *,
    failure_frame: int,
    seed: int,
    max_steps: int = 400,
) -> tuple[dict[str, float], bool, dict[str, Any]]:
    return run_true_mimicgen_rollout(
        hdf5_path,
        demo_key,
        theta,
        failure_frame=failure_frame,
        seed=seed,
        max_steps=max_steps,
    )


def load_demo_obs_and_actions(hdf5_path: str | Path, demo_key: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        obs = np.asarray(demo["obs"]["object"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float64) if "actions" in demo else None
    return obs, actions


def load_failed_contexts(
    failed_hdf5: str | Path,
    max_failed_demos: int | None = None,
) -> list[dict[str, Any]]:
    """Load failed-trajectory contexts from source / failed HDF5 demos."""
    path = Path(failed_hdf5)
    contexts: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=demo_sort_key)
        if max_failed_demos is not None:
            demo_keys = demo_keys[: max_failed_demos]

    for demo_key in demo_keys:
        obs, actions = load_demo_obs_and_actions(path, demo_key)
        fail_idx = _pick_failure_frame(obs, actions)
        context_metrics = compute_context_metrics(obs[fail_idx], action_delta=_action_delta(actions, fail_idx))
        contexts.append(
            {
                "demo_key": demo_key,
                "failure_frame": fail_idx,
                "context_metrics": context_metrics,
                "source_hdf5": str(path),
            }
        )
    return contexts


def generate_feedback_records(
    hdf5_path: str | Path,
    *,
    num_demos: int = 8,
    candidates_per_demo: int = 6,
    seed: int = 9701,
    start_candidate_index: int = 0,
    repair_eval_mode: str = "proxy_blend_from_source_demo",
    max_rollout_steps: int = 400,
) -> list[dict[str, Any]]:
    """Generate PhyGen-compatible feedback records from MimicGen source demos."""
    if repair_eval_mode not in {"proxy_blend_from_source_demo", "true_mimicgen_rollout"}:
        raise ValueError(f"Unsupported repair_eval_mode: {repair_eval_mode}")

    if repair_eval_mode == "true_mimicgen_rollout":
        runtime = check_sim_runtime()
        if not runtime["ready_for_true_rollout"]:
            raise RuntimeError(
                "true_mimicgen_rollout requires robosuite + MimicGen coffee env. "
                f"Missing/unavailable: {runtime.get('missing_packages')} "
                f"Recommended python: {runtime.get('recommended_python')} "
                f"Install hint: {runtime.get('recommended_install')}"
            )

    path = Path(hdf5_path)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=demo_sort_key)[:num_demos]

    candidate_index = start_candidate_index
    for demo_key in demo_keys:
        obs, actions = load_demo_obs_and_actions(path, demo_key)
        fail_idx = _pick_failure_frame(obs, actions)
        failed_metrics = compute_context_metrics(obs[fail_idx], action_delta=_action_delta(actions, fail_idx))
        success_idx = _pick_success_frame(obs, actions)
        success_metrics = _idealize_success_metrics(
            compute_context_metrics(obs[success_idx], action_delta=_action_delta(actions, success_idx))
        )
        context_metrics = dict(failed_metrics)

        for _ in range(candidates_per_demo):
            theta = make_candidate(candidate_index, rng)
            rollout_info: dict[str, Any] = {}
            if repair_eval_mode == "true_mimicgen_rollout":
                metrics, success, rollout_info = evaluate_repair_true_rollout(
                    path,
                    demo_key,
                    theta,
                    failure_frame=fail_idx,
                    seed=seed,
                    max_steps=max_rollout_steps,
                )
            else:
                metrics, success = evaluate_repair_proxy(theta, failed_metrics, success_metrics, rng)
            records.append(
                {
                    "task_name": "coffee_preparation",
                    "demo_key": demo_key,
                    "candidate_index": candidate_index,
                    "failure_frame": fail_idx,
                    "context_metrics": context_metrics,
                    "theta": theta,
                    "metrics": metrics,
                    "success": success,
                    "source_hdf5": str(path),
                    "repair_eval_mode": repair_eval_mode,
                    **rollout_info,
                }
            )
            candidate_index += 1
    return records


def compare_proxy_and_true_feedback(
    hdf5_path: str | Path,
    *,
    num_demos: int = 2,
    candidates_per_demo: int = 3,
    seed: int = 9701,
) -> dict[str, Any]:
    """Generate paired proxy/true records on identical demo/candidate seeds for comparison."""
    path = Path(hdf5_path)
    rng = np.random.default_rng(seed)
    paired: list[dict[str, Any]] = []

    with h5py.File(path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=demo_sort_key)[:num_demos]

    candidate_index = 0
    for demo_key in demo_keys:
        obs, actions = load_demo_obs_and_actions(path, demo_key)
        fail_idx = _pick_failure_frame(obs, actions)
        failed_metrics = compute_context_metrics(obs[fail_idx], action_delta=_action_delta(actions, fail_idx))
        success_idx = _pick_success_frame(obs, actions)
        success_metrics = _idealize_success_metrics(
            compute_context_metrics(obs[success_idx], action_delta=_action_delta(actions, success_idx))
        )
        context_metrics = dict(failed_metrics)

        for _ in range(candidates_per_demo):
            theta = make_candidate(candidate_index, rng)
            proxy_metrics, proxy_success = evaluate_repair_proxy(theta, failed_metrics, success_metrics, rng)
            true_metrics, true_success, rollout_info = evaluate_repair_true_rollout(
                path,
                demo_key,
                theta,
                failure_frame=fail_idx,
                seed=seed,
                max_steps=400,
            )
            paired.append(
                {
                    "demo_key": demo_key,
                    "candidate_index": candidate_index,
                    "theta": theta,
                    "proxy_success": proxy_success,
                    "true_success": true_success,
                    "success_match": bool(proxy_success == true_success),
                    "proxy_energy": proxy_metrics.get("energy"),
                    "true_energy": true_metrics.get("energy"),
                    "rollout_info": rollout_info,
                }
            )
            candidate_index += 1

    return {
        "num_pairs": len(paired),
        "success_match_rate": float(np.mean([p["success_match"] for p in paired])) if paired else 0.0,
        "proxy_success_count": int(sum(1 for p in paired if p["proxy_success"])),
        "true_success_count": int(sum(1 for p in paired if p["true_success"])),
        "pairs": paired,
    }


def write_feedback_jsonl(records: list[dict[str, Any]], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coffee_preparation PhyGen feedback jsonl")
    parser.add_argument(
        "--hdf5",
        default="third_party/mimicgen/datasets/source/coffee_preparation.hdf5",
        help="MimicGen source or failed HDF5 path",
    )
    parser.add_argument(
        "--output",
        default="runtime_outputs/phygen_coffee_smoke/coffee_preparation_feedback.jsonl",
    )
    parser.add_argument("--num-demos", type=int, default=8)
    parser.add_argument("--candidates-per-demo", type=int, default=6)
    parser.add_argument("--seed", type=int, default=9701)
    parser.add_argument("--start-candidate-index", type=int, default=0)
    parser.add_argument(
        "--repair-eval-mode",
        choices=["proxy_blend_from_source_demo", "true_mimicgen_rollout"],
        default="proxy_blend_from_source_demo",
    )
    parser.add_argument("--max-rollout-steps", type=int, default=400)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--compare-proxy-true", action="store_true")
    args = parser.parse_args()

    if args.check_runtime:
        print(json.dumps(check_sim_runtime(), indent=2, ensure_ascii=True))
        return

    if args.compare_proxy_true:
        report = compare_proxy_and_true_feedback(
            args.hdf5,
            num_demos=args.num_demos,
            candidates_per_demo=args.candidates_per_demo,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return

    records = generate_feedback_records(
        args.hdf5,
        num_demos=args.num_demos,
        candidates_per_demo=args.candidates_per_demo,
        seed=args.seed,
        start_candidate_index=args.start_candidate_index,
        repair_eval_mode=args.repair_eval_mode,
        max_rollout_steps=args.max_rollout_steps,
    )
    out = write_feedback_jsonl(records, args.output)
    summary = {
        "task": "coffee_preparation",
        "source_hdf5": args.hdf5,
        "repair_eval_mode": args.repair_eval_mode,
        "num_records": len(records),
        "num_demos": len({r["demo_key"] for r in records}),
        "num_success": int(sum(1 for r in records if r["success"])),
        "output": str(out),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
