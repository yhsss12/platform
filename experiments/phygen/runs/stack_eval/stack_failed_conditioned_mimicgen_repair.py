#!/usr/bin/env python3
"""
Failed-demo conditioned MimicGen repair for stack.

This follows the NutAssembly V1-E style loop:
fixed failed context + repair theta -> true rollout -> success / residual label.
All candidate rollouts, including failures, are written to JSONL for later PINN
feedback training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robomimic.utils.file_utils import get_env_metadata_from_dataset

import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.pose_utils as PoseUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils
from mimicgen.configs import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory
from mimicgen.env_interfaces.base import make_interface


DEFAULT_CONFIG = "/home/zyf/mimicgen/MimicGen_physics_refine/datasets/generated/stack_d0/demo/mg_config.json"
DEFAULT_SUCCESS_HDF5 = "/home/zyf/mimicgen/MimicGen_physics_refine/datasets/generated/stack_d0/demo/demo.hdf5"
DEFAULT_FAILED_HDF5 = "/home/zyf/mimicgen/MimicGen_physics_refine/datasets/generated/stack_d0/demo/demo_failed.hdf5"


def demo_sort_key(key: str) -> int:
    try:
        return int(key.split("_")[-1])
    except Exception:
        return 10**9


def pose_to_pos(pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose)
    if arr.shape == (4, 4):
        return arr[:3, 3].astype(np.float64)
    if arr.ndim == 1 and arr.shape[0] >= 3:
        return arr[:3].astype(np.float64)
    if arr.ndim >= 2:
        return pose_to_pos(arr[-1])
    raise ValueError(f"Unsupported pose shape {arr.shape}")



def object_positions_from_hdf5_group(grp: h5py.Group) -> dict[str, np.ndarray]:
    if "datagen_info" not in grp or "object_poses" not in grp["datagen_info"]:
        raise KeyError(
            "Expected generated MimicGen demo with datagen_info/object_poses. "
            "Run MimicGen generation / preparation before repair evaluation."
        )
    obj = grp["datagen_info/object_poses"]
    return {name: pose_to_pos(obj[name][:]) for name in sorted(obj.keys())}


def object_positions_from_datagen_info(info: Any) -> dict[str, np.ndarray]:
    obj = info.object_poses
    return {name: pose_to_pos(obj[name]) for name in sorted(obj.keys())}


def stack_metrics(pos: dict[str, np.ndarray]) -> dict[str, float]:
    """Task-agnostic residual proxy from final object poses.

    This first-pass metric is intentionally generic so every MimicGen task can
    execute the same RP-RF PINN loop. It preserves the StackThree-compatible
    context schema expected by the selector. Later, a task can replace only
    this function with a stronger task-specific residual without changing the
    training, candidate selection, or rollout-feedback loop.
    """
    if not pos:
        return {
            "energy": 30.0,
            "ab_xy": 1.0,
            "ab_z": 1.0,
            "ca_xy": 1.0,
            "ca_z": 1.0,
            "cb_xy": 1.0,
            "cb_z": 1.0,
            "c_minus_a": -1.0,
            "drop_penalty": 1.0,
        }

    names = sorted(pos.keys())
    pts = [np.asarray(pos[name], dtype=np.float64) for name in names]
    pair_xy: list[float] = []
    pair_z: list[float] = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            pair_xy.append(float(np.linalg.norm(pts[i][:2] - pts[j][:2])))
            pair_z.append(float(abs(pts[i][2] - pts[j][2])))
    if not pair_xy:
        pair_xy = [0.0]
        pair_z = [0.0]

    pair_xy_sorted = sorted(pair_xy)
    pair_z_sorted = sorted(pair_z)
    ab_xy = pair_xy_sorted[0]
    ab_z = pair_z_sorted[0]
    ca_xy = pair_xy_sorted[1] if len(pair_xy_sorted) > 1 else pair_xy_sorted[0]
    ca_z = pair_z_sorted[1] if len(pair_z_sorted) > 1 else pair_z_sorted[0]
    cb_xy = pair_xy_sorted[2] if len(pair_xy_sorted) > 2 else pair_xy_sorted[-1]
    cb_z = pair_z_sorted[2] if len(pair_z_sorted) > 2 else pair_z_sorted[-1]

    xy_spread = float(np.mean(pair_xy_sorted))
    z_vals = np.array([p[2] for p in pts], dtype=np.float64)
    z_spread = float(np.max(z_vals) - np.min(z_vals)) if len(z_vals) > 1 else 0.0
    c_minus_a = z_spread
    drop_penalty = float(max(0.0, 0.015 - z_spread))
    energy = (
        ab_xy / 0.040
        + ab_z / 0.030
        + ca_xy / 0.065
        + ca_z / 0.045
        + cb_xy / 0.090
        + cb_z / 0.060
        + xy_spread / 0.080
        + drop_penalty / 0.015
    )
    return {
        "energy": float(energy),
        "ab_xy": float(ab_xy),
        "ab_z": float(ab_z),
        "ca_xy": float(ca_xy),
        "ca_z": float(ca_z),
        "cb_xy": float(cb_xy),
        "cb_z": float(cb_z),
        "c_minus_a": float(c_minus_a),
        "drop_penalty": float(drop_penalty),
    }

def count_demos(path: str | Path) -> int:
    with h5py.File(path, "r") as f:
        return len(f["data"])


def load_failed_contexts(path: str | Path, max_failed_demos: int | None) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=demo_sort_key)
        for key in keys:
            grp = f["data"][key]
            pos = object_positions_from_hdf5_group(grp)
            metrics = stack_metrics(pos)
            contexts.append(
                {
                    "demo_key": key,
                    "initial_state": {
                        "states": grp["states"][0].astype(np.float64),
                        "model": grp.attrs["model_file"],
                    },
                    "context_metrics": metrics,
                    "num_samples": int(grp.attrs.get("num_samples", grp["actions"].shape[0])),
                }
            )
    contexts.sort(key=lambda x: x["context_metrics"]["energy"])
    if max_failed_demos is not None and max_failed_demos > 0:
        contexts = contexts[:max_failed_demos]
    return contexts


def load_candidate_plan(path: str | Path | None) -> dict[str, list[dict[str, Any]]] | None:
    if path is None:
        return None
    plan: dict[str, list[dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            demo_key = row["demo_key"]
            candidates = row.get("candidates")
            if candidates is None:
                candidates = [row]
            plan.setdefault(demo_key, []).extend(candidates)
    return plan


def load_config(path: str | Path, source_override: str | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if source_override:
        cfg["experiment"]["source"]["dataset_path"] = source_override
    return cfg


def make_candidate(index: int, rng: np.random.Generator) -> dict[str, Any]:
    presets: list[dict[str, Any]] = [
        {},
        {"select_src_per_subtask": True},
        {"select_src_per_subtask": True, "action_noise": 0.02},
        {"select_src_per_subtask": True, "action_noise": 0.05, "num_interpolation_steps": 10},
        {"select_src_per_subtask": True, "selection_strategy": "nearest_neighbor_object", "nn_k": 1},
        {"select_src_per_subtask": True, "selection_strategy": "nearest_neighbor_object", "nn_k": 5},
        {"select_src_per_subtask": True, "selection_strategy": "random"},
        {"select_src_per_subtask": True, "num_interpolation_steps": 10, "num_fixed_steps": 5},
        {"select_src_per_subtask": False, "action_noise": 0.02, "num_interpolation_steps": 10},
        {"select_src_per_subtask": True, "offset_range": [5, 15]},
        {"select_src_per_subtask": True, "offset_range": [15, 25]},
        {"select_src_per_subtask": True, "interpolate_from_last_target_pose": False},
    ]
    if index < len(presets):
        theta = presets[index].copy()
    else:
        theta = {
            "select_src_per_subtask": bool(rng.random() < 0.75),
            "transform_first_robot_pose": bool(rng.random() < 0.20),
            "interpolate_from_last_target_pose": bool(rng.random() < 0.85),
            "selection_strategy": str(rng.choice(["nearest_neighbor_object", "nearest_neighbor_object", "random"])),
            "action_noise": float(rng.choice([0.0, 0.01, 0.02, 0.05, 0.08])),
            "num_interpolation_steps": int(rng.choice([3, 5, 8, 10, 15])),
            "num_fixed_steps": int(rng.choice([0, 0, 3, 5, 10])),
            "offset_range": [int(rng.choice([0, 5, 10, 15])), int(rng.choice([10, 15, 20, 25]))],
            "nn_k": int(rng.choice([1, 3, 5, 10])),
        }
    theta.setdefault("select_src_per_subtask", False)
    theta.setdefault("transform_first_robot_pose", False)
    theta.setdefault("interpolate_from_last_target_pose", True)
    theta.setdefault("selection_strategy", "nearest_neighbor_object")
    theta.setdefault("action_noise", 0.05)
    theta.setdefault("num_interpolation_steps", 5)
    theta.setdefault("num_fixed_steps", 0)
    theta.setdefault("offset_range", [10, 20])
    theta.setdefault("nn_k", 3)
    lo, hi = theta["offset_range"]
    if lo > hi:
        lo, hi = hi, lo
    theta["offset_range"] = [int(lo), int(hi)]
    theta["candidate_family"] = (
        f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}"
        f"_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}"
    )
    return theta


def apply_theta(task_spec: MG_TaskSpec, theta: dict[str, Any]) -> None:
    for i, subtask in enumerate(task_spec):
        is_final = i == len(task_spec) - 1
        subtask["selection_strategy"] = theta["selection_strategy"]
        subtask["selection_strategy_kwargs"] = None
        if theta["selection_strategy"] == "nearest_neighbor_object":
            subtask["selection_strategy_kwargs"] = {"nn_k": int(theta["nn_k"])}
        subtask["action_noise"] = float(theta["action_noise"])
        subtask["num_interpolation_steps"] = int(theta["num_interpolation_steps"])
        subtask["num_fixed_steps"] = int(theta["num_fixed_steps"])
        if is_final:
            subtask["subtask_term_offset_range"] = (0, 0)
        else:
            subtask["subtask_term_offset_range"] = tuple(theta["offset_range"])


def generate_from_initial_state(
    data_generator: DataGenerator,
    env: Any,
    env_interface: Any,
    initial_state: dict[str, Any],
    theta: dict[str, Any],
) -> dict[str, Any]:
    env.reset()
    env.reset_to(initial_state)
    new_initial_state = {"states": np.array(initial_state["states"]), "model": initial_state["model"]}

    all_subtask_inds = data_generator.randomize_subtask_boundaries()
    selected_src_demo_ind = None
    prev_executed_traj = None
    generated_states = []
    generated_obs = []
    generated_datagen_infos = []
    generated_actions = []
    generated_success = False
    generated_src_demo_inds = []
    generated_src_demo_labels = []

    for subtask_ind in range(len(data_generator.task_spec)):
        is_first_subtask = subtask_ind == 0
        cur_datagen_info = env_interface.get_datagen_info()
        subtask_object_name = data_generator.task_spec[subtask_ind]["object_ref"]
        cur_object_pose = (
            cur_datagen_info.object_poses[subtask_object_name]
            if subtask_object_name is not None
            else None
        )
        need_source_demo_selection = is_first_subtask or theta["select_src_per_subtask"]
        if need_source_demo_selection:
            selected_src_demo_ind = data_generator.select_source_demo(
                eef_pose=cur_datagen_info.eef_pose,
                object_pose=cur_object_pose,
                subtask_ind=subtask_ind,
                src_subtask_inds=all_subtask_inds[:, subtask_ind],
                subtask_object_name=subtask_object_name,
                selection_strategy_name=data_generator.task_spec[subtask_ind]["selection_strategy"],
                selection_strategy_kwargs=data_generator.task_spec[subtask_ind]["selection_strategy_kwargs"],
            )
        assert selected_src_demo_ind is not None

        selected_src_subtask_inds = all_subtask_inds[selected_src_demo_ind, subtask_ind]
        src_ep_datagen_info = data_generator.src_dataset_infos[selected_src_demo_ind]
        src_subtask_eef_poses = src_ep_datagen_info.eef_pose[
            selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
        ]
        src_subtask_target_poses = src_ep_datagen_info.target_pose[
            selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
        ]
        src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[
            selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
        ]
        src_subtask_object_pose = (
            src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_inds[0]]
            if subtask_object_name is not None
            else None
        )

        if is_first_subtask or theta["transform_first_robot_pose"]:
            src_eef_poses = np.concatenate([src_subtask_eef_poses[0:1], src_subtask_target_poses], axis=0)
        else:
            src_eef_poses = np.array(src_subtask_target_poses)
        src_subtask_gripper_actions = np.concatenate(
            [src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], axis=0
        )

        if subtask_object_name is not None:
            transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
                obj_pose=cur_object_pose,
                src_eef_poses=src_eef_poses,
                src_obj_pose=src_subtask_object_pose,
            )
        else:
            transformed_eef_poses = src_eef_poses

        traj_to_execute = WaypointTrajectory()
        if theta["interpolate_from_last_target_pose"] and (not is_first_subtask):
            assert prev_executed_traj is not None
            init_sequence = WaypointSequence(sequence=[prev_executed_traj.last_waypoint])
        else:
            init_sequence = WaypointSequence.from_poses(
                poses=cur_datagen_info.eef_pose[None],
                gripper_actions=src_subtask_gripper_actions[0:1],
                action_noise=data_generator.task_spec[subtask_ind]["action_noise"],
            )
        traj_to_execute.add_waypoint_sequence(init_sequence)

        transformed_seq = WaypointSequence.from_poses(
            poses=transformed_eef_poses,
            gripper_actions=src_subtask_gripper_actions,
            action_noise=data_generator.task_spec[subtask_ind]["action_noise"],
        )
        transformed_traj = WaypointTrajectory()
        transformed_traj.add_waypoint_sequence(transformed_seq)
        traj_to_execute.merge(
            transformed_traj,
            num_steps_interp=data_generator.task_spec[subtask_ind]["num_interpolation_steps"],
            num_steps_fixed=data_generator.task_spec[subtask_ind]["num_fixed_steps"],
            action_noise=(
                float(data_generator.task_spec[subtask_ind]["apply_noise_during_interpolation"])
                * data_generator.task_spec[subtask_ind]["action_noise"]
            ),
        )
        traj_to_execute.pop_first()

        exec_results = traj_to_execute.execute(env=env, env_interface=env_interface)
        if len(exec_results["states"]) > 0:
            generated_states += exec_results["states"]
            generated_obs += exec_results["observations"]
            generated_datagen_infos += exec_results["datagen_infos"]
            generated_actions.append(exec_results["actions"])
            generated_success = generated_success or exec_results["success"]
            generated_src_demo_inds.append(int(selected_src_demo_ind))
            generated_src_demo_labels.append(
                selected_src_demo_ind * np.ones((exec_results["actions"].shape[0], 1), dtype=int)
            )
        prev_executed_traj = traj_to_execute

    if generated_actions:
        generated_actions_np = np.concatenate(generated_actions, axis=0)
        generated_src_demo_labels_np = np.concatenate(generated_src_demo_labels, axis=0)
    else:
        generated_actions_np = np.zeros((0, 7), dtype=np.float32)
        generated_src_demo_labels_np = np.zeros((0, 1), dtype=int)

    return {
        "initial_state": new_initial_state,
        "states": generated_states,
        "observations": generated_obs,
        "datagen_infos": generated_datagen_infos,
        "actions": generated_actions_np,
        "success": bool(generated_success),
        "src_demo_inds": generated_src_demo_inds,
        "src_demo_labels": generated_src_demo_labels_np,
    }


def make_env_and_generator(cfg: dict[str, Any]) -> tuple[Any, Any, DataGenerator]:
    source_dataset_path = cfg["experiment"]["source"]["dataset_path"]
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)
    all_demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=source_dataset_path,
        filter_key=cfg["experiment"]["source"].get("filter_key"),
        start=cfg["experiment"]["source"].get("start"),
        n=cfg["experiment"]["source"].get("n"),
    )
    env = RobomimicUtils.create_env(
        env_meta=env_meta,
        env_class=None,
        env_name=cfg["experiment"]["task"].get("name"),
        robot=cfg["experiment"]["task"].get("robot"),
        gripper=cfg["experiment"]["task"].get("gripper"),
        env_meta_update_kwargs=cfg["experiment"]["task"].get("env_meta_update_kwargs", {}),
        camera_names=[],
        camera_height=cfg["obs"].get("camera_height", 84),
        camera_width=cfg["obs"].get("camera_width", 84),
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )
    env_interface_name, env_interface_type = MG_FileUtils.get_env_interface_info_from_dataset(
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
    )
    if cfg["experiment"]["task"].get("interface") is not None:
        env_interface_name = cfg["experiment"]["task"]["interface"]
    if cfg["experiment"]["task"].get("interface_type") is not None:
        env_interface_type = cfg["experiment"]["task"]["interface_type"]
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        env=env.base_env,
    )
    task_spec = MG_TaskSpec.from_json(json_dict=deepcopy(cfg["task"]["task_spec"]))
    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
    )
    return env, env_interface, data_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-hdf5", default=None)
    parser.add_argument("--success-hdf5", default=DEFAULT_SUCCESS_HDF5)
    parser.add_argument("--failed-hdf5", default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-failed-demos", type=int, default=5)
    parser.add_argument("--candidates-per-demo", type=int, default=12)
    parser.add_argument("--candidate-plan", default=None)
    parser.add_argument("--seed", type=int, default=9401)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir or (
        "/home/zyf/mimicgen/MimicGen_physics_refine/outputs/"
        f"stack_failed_conditioned_repair_{time.strftime('%Y%m%d_%H%M%S')}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = out_dir / "feedback_candidates.jsonl"
    summary_path = out_dir / "summary.json"

    cfg = load_config(args.config, args.source_hdf5)
    raw_success = count_demos(args.success_hdf5)
    raw_failed = count_demos(args.failed_hdf5)
    failed_contexts = load_failed_contexts(args.failed_hdf5, args.max_failed_demos)
    candidate_plan = load_candidate_plan(args.candidate_plan)
    if candidate_plan is not None:
        failed_contexts = [ctx for ctx in failed_contexts if ctx["demo_key"] in candidate_plan]

    env, env_interface, data_generator = make_env_and_generator(cfg)
    exceptions_to_except = tuple(env.rollout_exceptions)

    repaired: dict[str, dict[str, Any]] = {}
    per_demo: dict[str, dict[str, Any]] = {}
    num_candidates = 0
    num_candidate_success = 0
    num_problematic = 0
    start = time.time()

    with feedback_path.open("w", encoding="utf-8") as f:
        for demo_idx, ctx in enumerate(failed_contexts):
            demo_key = ctx["demo_key"]
            if candidate_plan is None:
                candidates = [
                    {"candidate_index": i, "theta": make_candidate(i, rng)}
                    for i in range(args.candidates_per_demo)
                ]
            else:
                candidates = candidate_plan.get(demo_key, [])
            per_demo[demo_key] = {
                "context_energy": ctx["context_metrics"]["energy"],
                "num_candidates": 0,
                "num_success": 0,
                "best_energy": None,
                "best_success_candidate": None,
            }
            for cand_local_idx, candidate in enumerate(candidates):
                cand_idx = int(candidate.get("candidate_index", cand_local_idx))
                theta = candidate.get("theta", candidate)
                apply_theta(data_generator.task_spec, theta)
                rollout_seed = args.seed + demo_idx * 10000 + cand_idx
                random.seed(rollout_seed)
                np.random.seed(rollout_seed)
                row: dict[str, Any] = {
                    "demo_key": demo_key,
                    "candidate_index": cand_idx,
                    "candidate_local_index": cand_local_idx,
                    "rollout_seed": rollout_seed,
                    "theta": theta,
                    "planner_score": candidate.get("planner_score"),
                    "planner_rank": candidate.get("planner_rank"),
                    "context_metrics": ctx["context_metrics"],
                    "success": False,
                    "problematic": False,
                }
                t0 = time.time()
                try:
                    traj = generate_from_initial_state(
                        data_generator=data_generator,
                        env=env,
                        env_interface=env_interface,
                        initial_state=ctx["initial_state"],
                        theta=theta,
                    )
                    if traj["datagen_infos"]:
                        final_pos = object_positions_from_datagen_info(traj["datagen_infos"][-1])
                        metrics = stack_metrics(final_pos)
                    else:
                        metrics = stack_metrics(object_positions_from_datagen_info(env_interface.get_datagen_info()))
                    success = bool(traj["success"])
                    row.update(
                        {
                            "success": success,
                            "metrics": metrics,
                            "src_demo_inds": traj["src_demo_inds"],
                            "num_steps": int(traj["actions"].shape[0]),
                            "elapsed_sec": float(time.time() - t0),
                        }
                    )
                except exceptions_to_except as exc:
                    num_problematic += 1
                    row.update(
                        {
                            "problematic": True,
                            "exception": repr(exc),
                            "elapsed_sec": float(time.time() - t0),
                        }
                    )
                except Exception as exc:
                    num_problematic += 1
                    row.update(
                        {
                            "problematic": True,
                            "exception": repr(exc),
                            "elapsed_sec": float(time.time() - t0),
                        }
                    )

                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                f.flush()
                num_candidates += 1
                per_demo[demo_key]["num_candidates"] += 1
                if row.get("metrics") is not None:
                    e = row["metrics"]["energy"]
                    best_e = per_demo[demo_key]["best_energy"]
                    if best_e is None or e < best_e:
                        per_demo[demo_key]["best_energy"] = e
                        per_demo[demo_key]["best_theta"] = theta
                if row["success"]:
                    num_candidate_success += 1
                    per_demo[demo_key]["num_success"] += 1
                    if demo_key not in repaired:
                        per_demo[demo_key]["best_success_candidate"] = cand_idx
                        repaired[demo_key] = row

            print(
                f"[demo {demo_key}] success {per_demo[demo_key]['num_success']}/"
                f"{per_demo[demo_key]['num_candidates']} best_energy={per_demo[demo_key]['best_energy']}"
            )

    total = raw_success + raw_failed
    repaired_count = len(repaired)
    summary = {
        "mode": "failed_conditioned_mimicgen_repair",
        "config": args.config,
        "source_hdf5": cfg["experiment"]["source"]["dataset_path"],
        "success_hdf5": args.success_hdf5,
        "failed_hdf5": args.failed_hdf5,
        "raw_success": raw_success,
        "raw_failed": raw_failed,
        "raw_total": total,
        "raw_success_rate": raw_success / total if total else 0.0,
        "evaluated_failed_demos": len(failed_contexts),
        "candidates_per_demo": args.candidates_per_demo,
        "candidate_plan": args.candidate_plan,
        "num_candidates": num_candidates,
        "num_candidate_success": num_candidate_success,
        "candidate_success_rate": num_candidate_success / num_candidates if num_candidates else 0.0,
        "repaired_demo_keys": sorted(repaired.keys(), key=demo_sort_key),
        "repaired_count_in_evaluated": repaired_count,
        "projected_total_success_if_unique_repairs": raw_success + repaired_count,
        "projected_total_success_rate_if_unique_repairs": (raw_success + repaired_count) / total if total else 0.0,
        "num_problematic": num_problematic,
        "per_demo": per_demo,
        "feedback_path": str(feedback_path),
        "elapsed_sec": float(time.time() - start),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
