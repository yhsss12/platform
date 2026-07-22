#!/usr/bin/env python3
"""Unified 12-task RP-RF PINN inference and real MuJoCo repair evaluation."""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch

from robomimic.utils.file_utils import get_env_metadata_from_dataset

import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.pose_utils as PoseUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils
from mimicgen.configs import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory
from mimicgen.env_interfaces.base import make_interface

import universal_rprf_pinn as pinn


SUCCESS_ANCHORS: list[dict[str, np.ndarray]] = []


def demo_sort_key(key: str) -> int:
    try:
        return int(key.split('_')[-1])
    except Exception:
        return 10 ** 9

def pose_to_pos(pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose)
    if arr.shape == (4, 4):
        return arr[:3, 3].astype(np.float64)
    if arr.ndim == 1 and arr.shape[0] >= 3:
        return arr[:3].astype(np.float64)
    if arr.ndim >= 2:
        return pose_to_pos(arr[-1])
    raise ValueError(f'Unsupported pose shape {arr.shape}')

def object_positions_from_hdf5_group(grp: h5py.Group) -> dict[str, np.ndarray]:
    if 'datagen_info' not in grp or 'object_poses' not in grp['datagen_info']:
        raise KeyError('Expected generated MimicGen demo with datagen_info/object_poses. Run MimicGen generation / preparation before repair evaluation.')
    obj = grp['datagen_info/object_poses']
    return {name: pose_to_pos(obj[name][:]) for name in sorted(obj.keys())}

def object_positions_from_datagen_info(info: Any) -> dict[str, np.ndarray]:
    obj = info.object_poses
    return {name: pose_to_pos(obj[name]) for name in sorted(obj.keys())}

def count_demos(path: str | Path) -> int:
    with h5py.File(path, 'r') as f:
        return len(f['data'])

def load_candidate_plan(path: str | Path | None) -> dict[str, list[dict[str, Any]]] | None:
    if path is None:
        return None
    plan: dict[str, list[dict[str, Any]]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            demo_key = row['demo_key']
            candidates = row.get('candidates')
            if candidates is None:
                candidates = [row]
            plan.setdefault(demo_key, []).extend(candidates)
    return plan

def load_config(path: str | Path, source_override: str | None=None) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    if source_override:
        cfg['experiment']['source']['dataset_path'] = source_override
    return cfg

def generate_from_initial_state(data_generator: DataGenerator, env: Any, env_interface: Any, initial_state: dict[str, Any], theta: dict[str, Any]) -> dict[str, Any]:
    env.reset()
    env.reset_to(initial_state)
    new_initial_state = {'states': np.array(initial_state['states']), 'model': initial_state['model']}
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
        subtask_object_name = data_generator.task_spec[subtask_ind]['object_ref']
        cur_object_pose = cur_datagen_info.object_poses[subtask_object_name] if subtask_object_name is not None else None
        need_source_demo_selection = is_first_subtask or theta['select_src_per_subtask']
        if need_source_demo_selection:
            selected_src_demo_ind = data_generator.select_source_demo(eef_pose=cur_datagen_info.eef_pose, object_pose=cur_object_pose, subtask_ind=subtask_ind, src_subtask_inds=all_subtask_inds[:, subtask_ind], subtask_object_name=subtask_object_name, selection_strategy_name=data_generator.task_spec[subtask_ind]['selection_strategy'], selection_strategy_kwargs=data_generator.task_spec[subtask_ind]['selection_strategy_kwargs'])
        assert selected_src_demo_ind is not None
        selected_src_subtask_inds = all_subtask_inds[selected_src_demo_ind, subtask_ind]
        src_ep_datagen_info = data_generator.src_dataset_infos[selected_src_demo_ind]
        src_subtask_eef_poses = src_ep_datagen_info.eef_pose[selected_src_subtask_inds[0]:selected_src_subtask_inds[1]]
        src_subtask_target_poses = src_ep_datagen_info.target_pose[selected_src_subtask_inds[0]:selected_src_subtask_inds[1]]
        src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[selected_src_subtask_inds[0]:selected_src_subtask_inds[1]]
        src_subtask_object_pose = src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_inds[0]] if subtask_object_name is not None else None
        if is_first_subtask or theta['transform_first_robot_pose']:
            src_eef_poses = np.concatenate([src_subtask_eef_poses[0:1], src_subtask_target_poses], axis=0)
        else:
            src_eef_poses = np.array(src_subtask_target_poses)
        src_subtask_gripper_actions = np.concatenate([src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], axis=0)
        if subtask_object_name is not None:
            transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(obj_pose=cur_object_pose, src_eef_poses=src_eef_poses, src_obj_pose=src_subtask_object_pose)
        else:
            transformed_eef_poses = src_eef_poses
        traj_to_execute = WaypointTrajectory()
        if theta['interpolate_from_last_target_pose'] and (not is_first_subtask):
            assert prev_executed_traj is not None
            init_sequence = WaypointSequence(sequence=[prev_executed_traj.last_waypoint])
        else:
            init_sequence = WaypointSequence.from_poses(poses=cur_datagen_info.eef_pose[None], gripper_actions=src_subtask_gripper_actions[0:1], action_noise=data_generator.task_spec[subtask_ind]['action_noise'])
        traj_to_execute.add_waypoint_sequence(init_sequence)
        transformed_seq = WaypointSequence.from_poses(poses=transformed_eef_poses, gripper_actions=src_subtask_gripper_actions, action_noise=data_generator.task_spec[subtask_ind]['action_noise'])
        transformed_traj = WaypointTrajectory()
        transformed_traj.add_waypoint_sequence(transformed_seq)
        traj_to_execute.merge(transformed_traj, num_steps_interp=data_generator.task_spec[subtask_ind]['num_interpolation_steps'], num_steps_fixed=data_generator.task_spec[subtask_ind]['num_fixed_steps'], action_noise=float(data_generator.task_spec[subtask_ind]['apply_noise_during_interpolation']) * data_generator.task_spec[subtask_ind]['action_noise'])
        traj_to_execute.pop_first()
        exec_results = traj_to_execute.execute(env=env, env_interface=env_interface)
        if len(exec_results['states']) > 0:
            generated_states += exec_results['states']
            generated_obs += exec_results['observations']
            generated_datagen_infos += exec_results['datagen_infos']
            generated_actions.append(exec_results['actions'])
            generated_success = generated_success or exec_results['success']
            generated_src_demo_inds.append(int(selected_src_demo_ind))
            generated_src_demo_labels.append(selected_src_demo_ind * np.ones((exec_results['actions'].shape[0], 1), dtype=int))
        prev_executed_traj = traj_to_execute
    if generated_actions:
        generated_actions_np = np.concatenate(generated_actions, axis=0)
        generated_src_demo_labels_np = np.concatenate(generated_src_demo_labels, axis=0)
    else:
        generated_actions_np = np.zeros((0, 7), dtype=np.float32)
        generated_src_demo_labels_np = np.zeros((0, 1), dtype=int)
    return {'initial_state': new_initial_state, 'states': generated_states, 'observations': generated_obs, 'datagen_infos': generated_datagen_infos, 'actions': generated_actions_np, 'success': bool(generated_success), 'src_demo_inds': generated_src_demo_inds, 'src_demo_labels': generated_src_demo_labels_np}

def make_env_and_generator(cfg: dict[str, Any]) -> tuple[Any, Any, DataGenerator]:
    source_dataset_path = cfg['experiment']['source']['dataset_path']
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=source_dataset_path, filter_key=cfg['experiment']['source'].get('filter_key'), start=cfg['experiment']['source'].get('start'), n=cfg['experiment']['source'].get('n'))
    env = RobomimicUtils.create_env(env_meta=env_meta, env_class=None, env_name=cfg['experiment']['task'].get('name'), robot=cfg['experiment']['task'].get('robot'), gripper=cfg['experiment']['task'].get('gripper'), env_meta_update_kwargs=cfg['experiment']['task'].get('env_meta_update_kwargs', {}), camera_names=[], camera_height=cfg['obs'].get('camera_height', 84), camera_width=cfg['obs'].get('camera_width', 84), render=False, render_offscreen=False, use_image_obs=False, use_depth_obs=False)
    (env_interface_name, env_interface_type) = MG_FileUtils.get_env_interface_info_from_dataset(dataset_path=source_dataset_path, demo_keys=all_demos)
    if cfg['experiment']['task'].get('interface') is not None:
        env_interface_name = cfg['experiment']['task']['interface']
    if cfg['experiment']['task'].get('interface_type') is not None:
        env_interface_type = cfg['experiment']['task']['interface_type']
    env_interface = make_interface(name=env_interface_name, interface_type=env_interface_type, env=env.base_env)
    task_spec = MG_TaskSpec.from_json(json_dict=deepcopy(cfg['task']['task_spec']))
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=source_dataset_path, demo_keys=all_demos)
    return (env, env_interface, data_generator)


def set_success_anchors(path: str | Path | None, max_anchors: int = 100) -> int:
    SUCCESS_ANCHORS.clear()
    if path is None:
        return 0
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=demo_sort_key)
        for key in keys[:max_anchors]:
            grp = f["data"][key]
            try:
                SUCCESS_ANCHORS.append(object_positions_from_hdf5_group(grp))
            except Exception:
                continue
    return len(SUCCESS_ANCHORS)


def stack_obs_dicts(obs_list: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not obs_list:
        return {}
    keys = sorted(obs_list[0].keys())
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if all(key in obs for obs in obs_list):
            out[key] = np.stack([np.asarray(obs[key]) for obs in obs_list], axis=0)
    return out


def stack_datagen_infos(info_list: list[Any]) -> dict[str, Any]:
    if not info_list:
        return {}
    object_names = sorted(info_list[0].object_poses.keys())
    signal_names = sorted(info_list[0].subtask_term_signals.keys())
    return {
        "eef_pose": np.stack([np.asarray(info.eef_pose) for info in info_list], axis=0),
        "target_pose": np.stack([np.asarray(info.target_pose) for info in info_list], axis=0),
        "gripper_action": np.stack([np.asarray(info.gripper_action) for info in info_list], axis=0),
        "object_poses": {
            name: np.stack([np.asarray(info.object_poses[name]) for info in info_list], axis=0)
            for name in object_names
        },
        "subtask_term_signals": {
            name: np.asarray([info.subtask_term_signals[name] for info in info_list])
            for name in signal_names
        },
    }


def write_repaired_success_hdf5(
    output_path: str | Path,
    repaired_trajs: list[dict[str, Any]],
    template_success_hdf5: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with h5py.File(template_success_hdf5, "r") as template, h5py.File(output_path, "w") as out:
        for key, value in template.attrs.items():
            out.attrs[key] = value
        data_out = out.create_group("data")
        for key, value in template["data"].attrs.items():
            data_out.attrs[key] = value
        total = 0
        for i, item in enumerate(repaired_trajs):
            traj = item["traj"]
            grp = data_out.create_group(f"demo_{i}")
            actions = np.asarray(traj["actions"])
            states = np.asarray(traj["states"])
            grp.create_dataset("actions", data=actions)
            grp.create_dataset("states", data=states)
            grp.create_dataset("src_demo_inds", data=np.asarray(traj["src_demo_inds"], dtype=np.int64))
            grp.create_dataset("src_demo_labels", data=np.asarray(traj["src_demo_labels"], dtype=np.int64))

            obs_group = grp.create_group("obs")
            for key, value in stack_obs_dicts(traj.get("observations", [])).items():
                obs_group.create_dataset(key, data=value)

            info_group = grp.create_group("datagen_info")
            info = stack_datagen_infos(traj.get("datagen_infos", []))
            for key in ["eef_pose", "target_pose", "gripper_action"]:
                if key in info:
                    info_group.create_dataset(key, data=info[key])
            obj_group = info_group.create_group("object_poses")
            for key, value in info.get("object_poses", {}).items():
                obj_group.create_dataset(key, data=value)
            sig_group = info_group.create_group("subtask_term_signals")
            for key, value in info.get("subtask_term_signals", {}).items():
                sig_group.create_dataset(key, data=value)

            grp.attrs["model_file"] = traj["initial_state"]["model"]
            grp.attrs["num_samples"] = int(actions.shape[0])
            grp.attrs["source_failed_demo_key"] = item["demo_key"]
            grp.attrs["candidate_index"] = int(item["candidate_index"])
            grp.attrs["theta_json"] = json.dumps(item["theta"], ensure_ascii=True)
            grp.attrs["task"] = item["task"]
            total += int(actions.shape[0])
        data_out.attrs["total"] = total


def merge_success_and_repaired_hdf5(
    success_hdf5: str | Path,
    repaired_hdf5: str | Path,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with h5py.File(success_hdf5, "r") as success, h5py.File(repaired_hdf5, "r") as repaired, h5py.File(output_path, "w") as out:
        for key, value in success.attrs.items():
            out.attrs[key] = value
        data_out = out.create_group("data")
        for key, value in success["data"].attrs.items():
            data_out.attrs[key] = value
        total = 0
        out_idx = 0
        for src, is_repaired in [(success, False), (repaired, True)]:
            for key in sorted(src["data"].keys(), key=demo_sort_key):
                src["data"].copy(key, data_out, name=f"demo_{out_idx}")
                grp = data_out[f"demo_{out_idx}"]
                grp.attrs["is_repaired"] = bool(is_repaired)
                total += int(grp.attrs.get("num_samples", grp["actions"].shape[0]))
                out_idx += 1
        data_out.attrs["total"] = total


def load_failed_contexts(
    path: str | Path,
    max_failed_demos: int | None,
    metrics_fn: Callable[[dict[str, np.ndarray]], dict[str, float]],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=demo_sort_key)
        for key in keys:
            grp = f["data"][key]
            pos = object_positions_from_hdf5_group(grp)
            contexts.append(
                {
                    "demo_key": key,
                    "initial_state": {
                        "states": grp["states"][0].astype(np.float64),
                        "model": grp.attrs["model_file"],
                    },
                    "context_metrics": metrics_fn(pos),
                    "num_samples": int(grp.attrs.get("num_samples", grp["actions"].shape[0])),
                }
            )
    contexts.sort(key=lambda item: item["context_metrics"]["energy"])
    if max_failed_demos is not None and max_failed_demos > 0:
        contexts = contexts[:max_failed_demos]
    return contexts


@dataclass(frozen=True)
class TaskAdapter:
    metrics: Callable[[dict[str, np.ndarray]], dict[str, float]]
    make_candidate: Callable[[int, np.random.Generator], dict[str, Any]]
    apply_theta: Callable[[MG_TaskSpec, dict[str, Any]], None]
    make_safe_candidate: Callable[[int, np.random.Generator], dict[str, Any]]



# Physics context adapter: generic
def _generic_nearest_success_anchor_residual(pos: dict[str, np.ndarray]) -> dict[str, float] | None:
    if not SUCCESS_ANCHORS:
        return None
    best: dict[str, float] | None = None
    for anchor in SUCCESS_ANCHORS:
        common = sorted(set(pos.keys()).intersection(anchor.keys()))
        if not common:
            continue
        xy = np.array([np.linalg.norm(np.asarray(pos[name])[:2] - np.asarray(anchor[name])[:2]) for name in common], dtype=np.float64)
        z = np.array([abs(float(np.asarray(pos[name])[2] - np.asarray(anchor[name])[2])) for name in common], dtype=np.float64)
        full = np.array([np.linalg.norm(np.asarray(pos[name]) - np.asarray(anchor[name])) for name in common], dtype=np.float64)
        missing_fraction = 1.0 - len(common) / max(len(anchor), 1)
        row = {'anchor_xy': float(xy.mean()), 'anchor_z': float(z.mean()), 'anchor_mean': float(full.mean()), 'anchor_max': float(full.max()), 'anchor_xy_max': float(xy.max()), 'anchor_z_max': float(z.max()), 'anchor_missing_fraction': float(missing_fraction)}
        row['anchor_energy'] = row['anchor_xy'] / 0.035 + row['anchor_z'] / 0.025 + row['anchor_max'] / 0.12 + 5.0 * row['anchor_missing_fraction']
        if best is None or row['anchor_energy'] < best['anchor_energy']:
            best = row
    return best

def _metrics_generic(pos: dict[str, np.ndarray]) -> dict[str, float]:
    """Task-agnostic residual proxy from final object poses.

    This first-pass metric is intentionally generic so every MimicGen task can
    execute the same RP-RF PINN loop. It preserves the StackThree-compatible
    context schema expected by the selector. Later, a task can replace only
    this function with a stronger task-specific residual without changing the
    training, candidate selection, or rollout-feedback loop.
    """
    if not pos:
        return {'energy': 30.0, 'ab_xy': 1.0, 'ab_z': 1.0, 'ca_xy': 1.0, 'ca_z': 1.0, 'cb_xy': 1.0, 'cb_z': 1.0, 'c_minus_a': -1.0, 'drop_penalty': 1.0}
    anchor = _generic_nearest_success_anchor_residual(pos)
    if anchor is not None:
        ab_xy = float(anchor['anchor_xy'])
        ab_z = float(anchor['anchor_z'])
        ca_xy = float(anchor['anchor_xy_max'])
        ca_z = float(anchor['anchor_z_max'])
        cb_xy = float(anchor['anchor_mean'])
        cb_z = float(anchor['anchor_max'])
        c_minus_a = float(1.0 - anchor['anchor_missing_fraction'])
        drop_penalty = float(max(0.0, anchor['anchor_max'] - 0.1) + anchor['anchor_missing_fraction'])
        energy = float(anchor['anchor_energy'] + drop_penalty / 0.05)
        return {'energy': energy, 'ab_xy': ab_xy, 'ab_z': ab_z, 'ca_xy': ca_xy, 'ca_z': ca_z, 'cb_xy': cb_xy, 'cb_z': cb_z, 'c_minus_a': c_minus_a, 'drop_penalty': drop_penalty}
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
    energy = ab_xy / 0.04 + ab_z / 0.03 + ca_xy / 0.065 + ca_z / 0.045 + cb_xy / 0.09 + cb_z / 0.06 + xy_spread / 0.08 + drop_penalty / 0.015
    return {'energy': float(energy), 'ab_xy': float(ab_xy), 'ab_z': float(ab_z), 'ca_xy': float(ca_xy), 'ca_z': float(ca_z), 'cb_xy': float(cb_xy), 'cb_z': float(cb_z), 'c_minus_a': float(c_minus_a), 'drop_penalty': float(drop_penalty)}


# Physics context adapter: coffee
def _coffee_coffee_target_delta() -> np.ndarray:
    deltas: list[np.ndarray] = []
    for anchor in SUCCESS_ANCHORS:
        pod = anchor.get('coffee_pod')
        machine = anchor.get('coffee_machine')
        if pod is not None and machine is not None:
            deltas.append(np.asarray(pod, dtype=np.float64) - np.asarray(machine, dtype=np.float64))
    if not deltas:
        return np.array([0.106, 0.0, 0.073], dtype=np.float64)
    return np.mean(np.stack(deltas, axis=0), axis=0)

def _coffee_nearest_success_anchor_residual(pos: dict[str, np.ndarray]) -> dict[str, float] | None:
    if not SUCCESS_ANCHORS:
        return None
    best: dict[str, float] | None = None
    for anchor in SUCCESS_ANCHORS:
        common = sorted(set(pos.keys()).intersection(anchor.keys()))
        if not common:
            continue
        xy = np.array([np.linalg.norm(np.asarray(pos[name])[:2] - np.asarray(anchor[name])[:2]) for name in common], dtype=np.float64)
        z = np.array([abs(float(np.asarray(pos[name])[2] - np.asarray(anchor[name])[2])) for name in common], dtype=np.float64)
        full = np.array([np.linalg.norm(np.asarray(pos[name]) - np.asarray(anchor[name])) for name in common], dtype=np.float64)
        row = {'anchor_xy': float(xy.mean()), 'anchor_z': float(z.mean()), 'anchor_mean': float(full.mean()), 'anchor_max': float(full.max())}
        row['anchor_energy'] = row['anchor_xy'] / 0.025 + row['anchor_z'] / 0.02 + row['anchor_max'] / 0.07
        if best is None or row['anchor_energy'] < best['anchor_energy']:
            best = row
    return best

def _metrics_coffee(pos: dict[str, np.ndarray]) -> dict[str, float]:
    """Coffee-specific residual using pod-machine relative pose and success anchors."""
    pod = pos.get('coffee_pod')
    machine = pos.get('coffee_machine')
    if pod is None or machine is None:
        return {'energy': 30.0, 'ab_xy': 1.0, 'ab_z': 1.0, 'ca_xy': 1.0, 'ca_z': 1.0, 'cb_xy': 1.0, 'cb_z': 1.0, 'c_minus_a': -1.0, 'drop_penalty': 1.0}
    pod = np.asarray(pod, dtype=np.float64)
    machine = np.asarray(machine, dtype=np.float64)
    delta = pod - machine
    target = _coffee_coffee_target_delta()
    rel_xy = float(np.linalg.norm(delta[:2] - target[:2]))
    rel_z = float(abs(delta[2] - target[2]))
    dist_err = float(abs(np.linalg.norm(delta[:2]) - np.linalg.norm(target[:2])))
    z_gap = float(delta[2])
    vertical_penalty = float(max(0.0, rel_z - 0.015))
    base_energy = rel_xy / 0.02 + rel_z / 0.018 + dist_err / 0.02 + vertical_penalty / 0.012
    anchor = _coffee_nearest_success_anchor_residual(pos)
    if anchor is None:
        energy = base_energy
        anchor_xy = 0.0
        anchor_z = 0.0
    else:
        energy = 0.6 * base_energy + 0.4 * anchor['anchor_energy']
        anchor_xy = anchor['anchor_xy']
        anchor_z = anchor['anchor_z']
    return {'energy': float(energy), 'ab_xy': float(rel_xy), 'ab_z': float(rel_z), 'ca_xy': float(dist_err), 'ca_z': float(abs(z_gap - target[2])), 'cb_xy': float(anchor_xy), 'cb_z': float(anchor_z), 'c_minus_a': float(z_gap), 'drop_penalty': float(vertical_penalty)}


# Physics context adapter: nut
def _nut_nearest_success_anchor_residual(pos: dict[str, np.ndarray]) -> dict[str, float] | None:
    if not SUCCESS_ANCHORS:
        return None
    best: dict[str, float] | None = None
    for anchor in SUCCESS_ANCHORS:
        common = sorted(set(pos.keys()).intersection(anchor.keys()))
        if not common:
            continue
        xy = np.array([np.linalg.norm(np.asarray(pos[name])[:2] - np.asarray(anchor[name])[:2]) for name in common], dtype=np.float64)
        z = np.array([abs(float(np.asarray(pos[name])[2] - np.asarray(anchor[name])[2])) for name in common], dtype=np.float64)
        full = np.array([np.linalg.norm(np.asarray(pos[name]) - np.asarray(anchor[name])) for name in common], dtype=np.float64)
        row = {'anchor_xy': float(xy.mean()), 'anchor_z': float(z.mean()), 'anchor_mean': float(full.mean()), 'anchor_max': float(full.max())}
        row['anchor_energy'] = row['anchor_xy'] / 0.035 + row['anchor_z'] / 0.022 + row['anchor_max'] / 0.08
        if best is None or row['anchor_energy'] < best['anchor_energy']:
            best = row
    return best

def _nut_insertion_pair_metrics(pos: dict[str, np.ndarray], nut_name: str, peg_name: str, xy_scale: float) -> dict[str, float]:
    nut = pos.get(nut_name)
    peg = pos.get(peg_name)
    if nut is None or peg is None:
        return {'xy': 1.0, 'z_err': 1.0, 'z_gap': -1.0, 'penalty': 1.0, 'energy': 30.0}
    nut = np.asarray(nut, dtype=np.float64)
    peg = np.asarray(peg, dtype=np.float64)
    xy = float(np.linalg.norm(nut[:2] - peg[:2]))
    z_gap = float(nut[2] - peg[2])
    z_err = float(abs(z_gap + 0.02))
    penalty = float(max(0.0, z_err - 0.01))
    energy = xy / xy_scale + z_err / 0.018 + penalty / 0.01
    return {'xy': xy, 'z_err': z_err, 'z_gap': z_gap, 'penalty': penalty, 'energy': energy}

def _metrics_nut(pos: dict[str, np.ndarray]) -> dict[str, float]:
    """NutAssembly residual: square-nut and round-nut insertion errors."""
    square = _nut_insertion_pair_metrics(pos, 'square_nut', 'square_peg', xy_scale=0.025)
    round_ = _nut_insertion_pair_metrics(pos, 'round_nut', 'round_peg', xy_scale=0.03)
    worst_energy = max(square['energy'], round_['energy'])
    base_energy = 0.5 * (square['energy'] + round_['energy']) + 0.25 * worst_energy
    anchor = _nut_nearest_success_anchor_residual(pos)
    if anchor is None:
        energy = base_energy
        anchor_xy = max(square['xy'], round_['xy'])
        anchor_z = max(square['z_err'], round_['z_err'])
    else:
        energy = 0.65 * base_energy + 0.35 * anchor['anchor_energy']
        anchor_xy = anchor['anchor_xy']
        anchor_z = anchor['anchor_z']
    return {'energy': float(energy), 'ab_xy': float(square['xy']), 'ab_z': float(square['z_err']), 'ca_xy': float(round_['xy']), 'ca_z': float(round_['z_err']), 'cb_xy': float(anchor_xy), 'cb_z': float(anchor_z), 'c_minus_a': float(min(square['z_gap'], round_['z_gap'])), 'drop_penalty': float(max(square['penalty'], round_['penalty']))}


# Physics context adapter: pick
def _pick_nearest_success_anchor_residual(pos: dict[str, np.ndarray]) -> dict[str, float] | None:
    if not SUCCESS_ANCHORS:
        return None
    best: dict[str, float] | None = None
    for anchor in SUCCESS_ANCHORS:
        common = sorted(set(pos.keys()).intersection(anchor.keys()))
        if not common:
            continue
        xy = np.array([np.linalg.norm(np.asarray(pos[name])[:2] - np.asarray(anchor[name])[:2]) for name in common], dtype=np.float64)
        z = np.array([abs(float(np.asarray(pos[name])[2] - np.asarray(anchor[name])[2])) for name in common], dtype=np.float64)
        full = np.array([np.linalg.norm(np.asarray(pos[name]) - np.asarray(anchor[name])) for name in common], dtype=np.float64)
        row = {'anchor_xy': float(xy.mean()), 'anchor_z': float(z.mean()), 'anchor_mean': float(full.mean()), 'anchor_max': float(full.max()), 'anchor_xy_max': float(xy.max()), 'anchor_z_max': float(z.max())}
        row['anchor_energy'] = row['anchor_xy'] / 0.035 + row['anchor_z'] / 0.025 + row['anchor_max'] / 0.12
        if best is None or row['anchor_energy'] < best['anchor_energy']:
            best = row
    return best

def _metrics_pick(pos: dict[str, np.ndarray]) -> dict[str, float]:
    """Multi-object PickPlace residual: distance to nearest successful final arrangement."""
    expected = ['bread', 'can', 'cereal', 'milk']
    present = [name for name in expected if name in pos]
    if not present:
        return {'energy': 30.0, 'ab_xy': 1.0, 'ab_z': 1.0, 'ca_xy': 1.0, 'ca_z': 1.0, 'cb_xy': 1.0, 'cb_z': 1.0, 'c_minus_a': -1.0, 'drop_penalty': 1.0}
    pts = [np.asarray(pos[name], dtype=np.float64) for name in present]
    anchor = _pick_nearest_success_anchor_residual(pos)
    if anchor is None:
        xy_vals = []
        z_vals = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                xy_vals.append(float(np.linalg.norm(pts[i][:2] - pts[j][:2])))
                z_vals.append(float(abs(pts[i][2] - pts[j][2])))
        ab_xy = float(np.mean(xy_vals)) if xy_vals else 1.0
        ab_z = float(np.mean(z_vals)) if z_vals else 1.0
        ca_xy = float(max(xy_vals)) if xy_vals else 1.0
        ca_z = float(max(z_vals)) if z_vals else 1.0
        cb_xy = ab_xy
        cb_z = ab_z
        drop_penalty = float(np.mean([max(0.0, 0.82 - p[2]) for p in pts]))
        energy = ab_xy / 0.1 + ab_z / 0.05 + ca_xy / 0.2 + drop_penalty / 0.03
    else:
        ab_xy = float(anchor['anchor_xy'])
        ab_z = float(anchor['anchor_z'])
        ca_xy = float(anchor['anchor_xy_max'])
        ca_z = float(anchor['anchor_z_max'])
        cb_xy = float(anchor['anchor_mean'])
        cb_z = float(np.mean([max(0.0, 0.82 - p[2]) for p in pts]))
        drop_penalty = float(max(0.0, anchor['anchor_max'] - 0.1))
        energy = float(anchor['anchor_energy'] + drop_penalty / 0.05)
    c_minus_a = float(0.1 - cb_xy)
    return {'energy': float(energy), 'ab_xy': float(ab_xy), 'ab_z': float(ab_z), 'ca_xy': float(ca_xy), 'ca_z': float(ca_z), 'cb_xy': float(cb_xy), 'cb_z': float(cb_z), 'c_minus_a': float(c_minus_a), 'drop_penalty': float(drop_penalty)}


# Physics context adapter: square
def _square_nearest_success_anchor_residual(pos: dict[str, np.ndarray]) -> dict[str, float] | None:
    if not SUCCESS_ANCHORS:
        return None
    names = sorted(pos.keys())
    best: dict[str, float] | None = None
    for anchor in SUCCESS_ANCHORS:
        common = [name for name in names if name in anchor]
        if not common:
            continue
        xy = np.array([np.linalg.norm(np.asarray(pos[name])[:2] - np.asarray(anchor[name])[:2]) for name in common], dtype=np.float64)
        z = np.array([abs(float(np.asarray(pos[name])[2] - np.asarray(anchor[name])[2])) for name in common], dtype=np.float64)
        full = np.array([np.linalg.norm(np.asarray(pos[name]) - np.asarray(anchor[name])) for name in common], dtype=np.float64)
        row = {'anchor_xy': float(xy.mean()), 'anchor_z': float(z.mean()), 'anchor_mean': float(full.mean()), 'anchor_max': float(full.max())}
        row['anchor_energy'] = row['anchor_xy'] / 0.03 + row['anchor_z'] / 0.02 + row['anchor_max'] / 0.07
        if best is None or row['anchor_energy'] < best['anchor_energy']:
            best = row
    return best

def _metrics_square(pos: dict[str, np.ndarray]) -> dict[str, float]:
    """Square-specific insertion residual projected to the RP-RF context schema."""
    nut = pos.get('square_nut')
    peg = pos.get('square_peg')
    if nut is None or peg is None:
        return {'energy': 30.0, 'ab_xy': 1.0, 'ab_z': 1.0, 'ca_xy': 1.0, 'ca_z': 1.0, 'cb_xy': 1.0, 'cb_z': 1.0, 'c_minus_a': -1.0, 'drop_penalty': 1.0}
    nut = np.asarray(nut, dtype=np.float64)
    peg = np.asarray(peg, dtype=np.float64)
    xy = float(np.linalg.norm(nut[:2] - peg[:2]))
    z_gap = float(nut[2] - peg[2])
    z_err = float(abs(z_gap + 0.02))
    vertical_penalty = float(max(0.0, z_err - 0.01))
    base_energy = xy / 0.025 + z_err / 0.018 + vertical_penalty / 0.01
    anchor = _square_nearest_success_anchor_residual(pos)
    if anchor is None:
        energy = base_energy
        anchor_xy = 0.0
        anchor_z = 0.0
    else:
        energy = 0.65 * base_energy + 0.35 * anchor['anchor_energy']
        anchor_xy = anchor['anchor_xy']
        anchor_z = anchor['anchor_z']
    return {'energy': float(energy), 'ab_xy': float(xy), 'ab_z': float(z_err), 'ca_xy': float(xy), 'ca_z': float(z_err), 'cb_xy': float(anchor_xy), 'cb_z': float(anchor_z), 'c_minus_a': float(z_gap), 'drop_penalty': float(vertical_penalty)}


# Physics context adapter: stack
def _stack_nearest_success_anchor_residual(pos: dict[str, np.ndarray]) -> dict[str, float] | None:
    if not SUCCESS_ANCHORS:
        return None
    best: dict[str, float] | None = None
    for anchor in SUCCESS_ANCHORS:
        common = sorted(set(pos.keys()).intersection(anchor.keys()))
        if not common:
            continue
        xy = np.array([np.linalg.norm(np.asarray(pos[name])[:2] - np.asarray(anchor[name])[:2]) for name in common], dtype=np.float64)
        z = np.array([abs(float(np.asarray(pos[name])[2] - np.asarray(anchor[name])[2])) for name in common], dtype=np.float64)
        full = np.array([np.linalg.norm(np.asarray(pos[name]) - np.asarray(anchor[name])) for name in common], dtype=np.float64)
        row = {'anchor_xy': float(xy.mean()), 'anchor_z': float(z.mean()), 'anchor_mean': float(full.mean()), 'anchor_max': float(full.max())}
        row['anchor_energy'] = row['anchor_xy'] / 0.03 + row['anchor_z'] / 0.02 + row['anchor_max'] / 0.08
        if best is None or row['anchor_energy'] < best['anchor_energy']:
            best = row
    return best

def _metrics_stack(pos: dict[str, np.ndarray]) -> dict[str, float]:
    """Two-cube Stack residual proxy projected to the shared RP-RF schema."""
    a = pos.get('cubeA')
    b = pos.get('cubeB')
    if a is None or b is None:
        return {'energy': 30.0, 'ab_xy': 1.0, 'ab_z': 1.0, 'ca_xy': 1.0, 'ca_z': 1.0, 'cb_xy': 1.0, 'cb_z': 1.0, 'c_minus_a': -1.0, 'drop_penalty': 1.0}
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab_xy = float(np.linalg.norm(a[:2] - b[:2]))
    z_diff = float(a[2] - b[2])
    ab_z = float(abs(z_diff - 0.0445))
    drop_penalty = float(max(0.0, 0.03 - z_diff))
    anchor = _stack_nearest_success_anchor_residual(pos)
    anchor_xy = 0.0 if anchor is None else float(anchor['anchor_xy'])
    anchor_z = 0.0 if anchor is None else float(anchor['anchor_z'])
    anchor_energy = 0.0 if anchor is None else float(anchor['anchor_energy'])
    ca_xy = anchor_xy
    ca_z = anchor_z
    cb_xy = float(abs(ab_xy - 0.008))
    cb_z = float(abs(z_diff - 0.0445))
    base_energy = ab_xy / 0.025 + ab_z / 0.012 + cb_xy / 0.02 + cb_z / 0.012 + drop_penalty / 0.01
    energy = base_energy if anchor is None else 0.8 * base_energy + 0.2 * anchor_energy
    return {'energy': float(energy), 'ab_xy': float(ab_xy), 'ab_z': float(ab_z), 'ca_xy': float(ca_xy), 'ca_z': float(ca_z), 'cb_xy': float(cb_xy), 'cb_z': float(cb_z), 'c_minus_a': float(z_diff), 'drop_penalty': float(drop_penalty)}


# Physics context adapter: stack_three
def _stack_three_nearest_success_anchor_residual(pos: dict[str, np.ndarray]) -> dict[str, float] | None:
    if not SUCCESS_ANCHORS:
        return None
    best: dict[str, float] | None = None
    for anchor in SUCCESS_ANCHORS:
        common = sorted(set(pos.keys()).intersection(anchor.keys()))
        if not common:
            continue
        xy = np.array([np.linalg.norm(np.asarray(pos[name])[:2] - np.asarray(anchor[name])[:2]) for name in common], dtype=np.float64)
        z = np.array([abs(float(np.asarray(pos[name])[2] - np.asarray(anchor[name])[2])) for name in common], dtype=np.float64)
        full = np.array([np.linalg.norm(np.asarray(pos[name]) - np.asarray(anchor[name])) for name in common], dtype=np.float64)
        row = {'anchor_xy': float(xy.mean()), 'anchor_z': float(z.mean()), 'anchor_mean': float(full.mean()), 'anchor_max': float(full.max())}
        row['anchor_energy'] = row['anchor_xy'] / 0.035 + row['anchor_z'] / 0.025 + row['anchor_max'] / 0.09
        if best is None or row['anchor_energy'] < best['anchor_energy']:
            best = row
    return best

def _metrics_stack_three(pos: dict[str, np.ndarray]) -> dict[str, float]:
    a = pos.get('cubeA')
    b = pos.get('cubeB')
    c = pos.get('cubeC')
    if a is None or b is None or c is None:
        return {'energy': 30.0, 'ab_xy': 1.0, 'ab_z': 1.0, 'ca_xy': 1.0, 'ca_z': 1.0, 'cb_xy': 1.0, 'cb_z': 1.0, 'c_minus_a': -1.0, 'drop_penalty': 1.0}
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    ab_xy = float(np.linalg.norm(a[:2] - b[:2]))
    ab_z = float(abs(a[2] - b[2] - 0.045))
    ca_xy = float(np.linalg.norm(c[:2] - a[:2]))
    ca_z = float(abs(c[2] - a[2] - 0.04))
    cb_xy = float(np.linalg.norm(c[:2] - b[:2]))
    cb_z = float(abs(c[2] - b[2] - 0.085))
    c_minus_a = float(c[2] - a[2])
    drop_penalty = float(max(0.0, 0.018 - c_minus_a))
    base_energy = ab_xy / 0.03 + ab_z / 0.015 + ca_xy / 0.045 + ca_z / 0.025 + cb_xy / 0.06 + cb_z / 0.025 + drop_penalty / 0.01
    anchor = _stack_three_nearest_success_anchor_residual(pos)
    energy = base_energy if anchor is None else 0.75 * base_energy + 0.25 * anchor['anchor_energy']
    return {'energy': float(energy), 'ab_xy': float(ab_xy), 'ab_z': float(ab_z), 'ca_xy': float(ca_xy), 'ca_z': float(ca_z), 'cb_xy': float(cb_xy), 'cb_z': float(cb_z), 'c_minus_a': float(c_minus_a), 'drop_penalty': float(drop_penalty)}


# Base repair candidates: generic
def _make_candidate_generic(index: int, rng: np.random.Generator) -> dict[str, Any]:
    presets: list[dict[str, Any]] = [{}, {'select_src_per_subtask': True}, {'select_src_per_subtask': True, 'action_noise': 0.02}, {'select_src_per_subtask': True, 'action_noise': 0.05, 'num_interpolation_steps': 10}, {'select_src_per_subtask': True, 'selection_strategy': 'nearest_neighbor_object', 'nn_k': 1}, {'select_src_per_subtask': True, 'selection_strategy': 'nearest_neighbor_object', 'nn_k': 5}, {'select_src_per_subtask': True, 'selection_strategy': 'random'}, {'select_src_per_subtask': True, 'num_interpolation_steps': 10, 'num_fixed_steps': 5}, {'select_src_per_subtask': False, 'action_noise': 0.02, 'num_interpolation_steps': 10}, {'select_src_per_subtask': True, 'offset_range': [5, 15]}, {'select_src_per_subtask': True, 'offset_range': [15, 25]}, {'select_src_per_subtask': True, 'interpolate_from_last_target_pose': False}]
    if index < len(presets):
        theta = presets[index].copy()
    else:
        theta = {'select_src_per_subtask': bool(rng.random() < 0.75), 'transform_first_robot_pose': bool(rng.random() < 0.2), 'interpolate_from_last_target_pose': bool(rng.random() < 0.85), 'selection_strategy': str(rng.choice(['nearest_neighbor_object', 'nearest_neighbor_object', 'random'])), 'action_noise': float(rng.choice([0.0, 0.01, 0.02, 0.05, 0.08])), 'num_interpolation_steps': int(rng.choice([3, 5, 8, 10, 15])), 'num_fixed_steps': int(rng.choice([0, 0, 3, 5, 10])), 'offset_range': [int(rng.choice([0, 5, 10, 15])), int(rng.choice([10, 15, 20, 25]))], 'nn_k': int(rng.choice([1, 3, 5, 10]))}
    theta.setdefault('select_src_per_subtask', False)
    theta.setdefault('transform_first_robot_pose', False)
    theta.setdefault('interpolate_from_last_target_pose', True)
    theta.setdefault('selection_strategy', 'nearest_neighbor_object')
    theta.setdefault('action_noise', 0.05)
    theta.setdefault('num_interpolation_steps', 5)
    theta.setdefault('num_fixed_steps', 0)
    theta.setdefault('offset_range', [10, 20])
    theta.setdefault('nn_k', 3)
    (lo, hi) = theta['offset_range']
    if lo > hi:
        (lo, hi) = (hi, lo)
    theta['offset_range'] = [int(lo), int(hi)]
    theta['candidate_family'] = f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}"
    return theta


# Base repair candidates: coffee
def _make_candidate_coffee(index: int, rng: np.random.Generator) -> dict[str, Any]:
    presets: list[dict[str, Any]] = [{}, {'select_src_per_subtask': True}, {'select_src_per_subtask': True, 'action_noise': 0.005}, {'select_src_per_subtask': True, 'action_noise': 0.01, 'num_interpolation_steps': 10}, {'select_src_per_subtask': True, 'selection_strategy': 'nearest_neighbor_object', 'nn_k': 1}, {'select_src_per_subtask': True, 'selection_strategy': 'nearest_neighbor_object', 'nn_k': 5}, {'select_src_per_subtask': True, 'selection_strategy': 'random'}, {'select_src_per_subtask': True, 'num_interpolation_steps': 15, 'num_fixed_steps': 0}, {'select_src_per_subtask': False, 'action_noise': 0.01, 'num_interpolation_steps': 10}, {'select_src_per_subtask': True, 'offset_range': [5, 10]}, {'select_src_per_subtask': True, 'offset_range': [5, 15]}, {'select_src_per_subtask': True, 'interpolate_from_last_target_pose': False}]
    if index < len(presets):
        theta = presets[index].copy()
    else:
        theta = {'select_src_per_subtask': bool(rng.random() < 0.78), 'transform_first_robot_pose': bool(rng.random() < 0.08), 'interpolate_from_last_target_pose': bool(rng.random() < 0.92), 'selection_strategy': 'random' if rng.random() < 0.65 else 'nearest_neighbor_object', 'action_noise': float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.05], p=[0.18, 0.22, 0.26, 0.22, 0.12])), 'num_interpolation_steps': int(rng.choice([5, 8, 10, 15], p=[0.22, 0.2, 0.3, 0.28])), 'num_fixed_steps': int(rng.choice([0, 0, 0, 3, 5])), 'offset_range': np.array([[5, 10], [5, 15], [0, 10], [10, 20], [0, 15], [10, 15]], dtype=int)[int(rng.integers(0, 6))].tolist(), 'nn_k': int(rng.choice([1, 3, 5, 10], p=[0.34, 0.28, 0.24, 0.14]))}
    theta.setdefault('select_src_per_subtask', False)
    theta.setdefault('transform_first_robot_pose', False)
    theta.setdefault('interpolate_from_last_target_pose', True)
    theta.setdefault('selection_strategy', 'nearest_neighbor_object')
    theta.setdefault('action_noise', 0.05)
    theta.setdefault('num_interpolation_steps', 5)
    theta.setdefault('num_fixed_steps', 0)
    theta.setdefault('offset_range', [10, 20])
    theta.setdefault('nn_k', 3)
    (lo, hi) = theta['offset_range']
    if lo > hi:
        (lo, hi) = (hi, lo)
    theta['offset_range'] = [int(lo), int(hi)]
    theta['candidate_family'] = f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}"
    return theta


# Base repair candidates: nut
def _make_candidate_nut(index: int, rng: np.random.Generator) -> dict[str, Any]:
    presets: list[dict[str, Any]] = [{}, {'select_src_per_subtask': True}, {'select_src_per_subtask': True, 'action_noise': 0.005}, {'select_src_per_subtask': True, 'action_noise': 0.01, 'num_interpolation_steps': 10}, {'select_src_per_subtask': True, 'selection_strategy': 'nearest_neighbor_object', 'nn_k': 1}, {'select_src_per_subtask': True, 'selection_strategy': 'nearest_neighbor_object', 'nn_k': 5}, {'select_src_per_subtask': True, 'selection_strategy': 'random'}, {'select_src_per_subtask': True, 'num_interpolation_steps': 15, 'num_fixed_steps': 0}, {'select_src_per_subtask': False, 'action_noise': 0.01, 'num_interpolation_steps': 10}, {'select_src_per_subtask': True, 'offset_range': [5, 10]}, {'select_src_per_subtask': True, 'offset_range': [5, 15]}, {'select_src_per_subtask': True, 'interpolate_from_last_target_pose': False}]
    if index < len(presets):
        theta = presets[index].copy()
    else:
        theta = {'select_src_per_subtask': bool(rng.random() < 0.88), 'transform_first_robot_pose': bool(rng.random() < 0.08), 'interpolate_from_last_target_pose': bool(rng.random() < 0.94), 'selection_strategy': 'nearest_neighbor_object' if rng.random() < 0.74 else 'random', 'action_noise': float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.05], p=[0.16, 0.2, 0.26, 0.26, 0.12])), 'num_interpolation_steps': int(rng.choice([5, 8, 10, 15], p=[0.2, 0.2, 0.3, 0.3])), 'num_fixed_steps': int(rng.choice([0, 0, 0, 3, 5])), 'offset_range': np.array([[5, 10], [5, 15], [0, 10], [10, 20], [0, 15], [10, 15], [15, 20]], dtype=int)[int(rng.integers(0, 7))].tolist(), 'nn_k': int(rng.choice([1, 3, 5, 10], p=[0.36, 0.28, 0.24, 0.12]))}
    theta.setdefault('select_src_per_subtask', False)
    theta.setdefault('transform_first_robot_pose', False)
    theta.setdefault('interpolate_from_last_target_pose', True)
    theta.setdefault('selection_strategy', 'nearest_neighbor_object')
    theta.setdefault('action_noise', 0.05)
    theta.setdefault('num_interpolation_steps', 5)
    theta.setdefault('num_fixed_steps', 0)
    theta.setdefault('offset_range', [10, 20])
    theta.setdefault('nn_k', 3)
    (lo, hi) = theta['offset_range']
    if lo > hi:
        (lo, hi) = (hi, lo)
    theta['offset_range'] = [int(lo), int(hi)]
    theta['candidate_family'] = f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}"
    return theta


# Base repair candidates: square
def _make_candidate_square(index: int, rng: np.random.Generator) -> dict[str, Any]:
    presets: list[dict[str, Any]] = [{}, {'select_src_per_subtask': True}, {'select_src_per_subtask': True, 'action_noise': 0.005}, {'select_src_per_subtask': True, 'action_noise': 0.01, 'num_interpolation_steps': 10}, {'select_src_per_subtask': True, 'selection_strategy': 'nearest_neighbor_object', 'nn_k': 1}, {'select_src_per_subtask': True, 'selection_strategy': 'nearest_neighbor_object', 'nn_k': 5}, {'select_src_per_subtask': True, 'selection_strategy': 'random'}, {'select_src_per_subtask': True, 'num_interpolation_steps': 15, 'num_fixed_steps': 0}, {'select_src_per_subtask': False, 'action_noise': 0.01, 'num_interpolation_steps': 10}, {'select_src_per_subtask': True, 'offset_range': [5, 10]}, {'select_src_per_subtask': True, 'offset_range': [5, 15]}, {'select_src_per_subtask': True, 'interpolate_from_last_target_pose': False}]
    if index < len(presets):
        theta = presets[index].copy()
    else:
        theta = {'select_src_per_subtask': bool(rng.random() < 0.86), 'transform_first_robot_pose': bool(rng.random() < 0.08), 'interpolate_from_last_target_pose': bool(rng.random() < 0.94), 'selection_strategy': 'nearest_neighbor_object' if rng.random() < 0.78 else 'random', 'action_noise': float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.05], p=[0.18, 0.22, 0.24, 0.24, 0.12])), 'num_interpolation_steps': int(rng.choice([5, 8, 10, 15], p=[0.22, 0.2, 0.3, 0.28])), 'num_fixed_steps': int(rng.choice([0, 0, 0, 3, 5])), 'offset_range': np.array([[5, 10], [5, 15], [0, 10], [10, 20], [0, 15], [10, 15]], dtype=int)[int(rng.integers(0, 6))].tolist(), 'nn_k': int(rng.choice([1, 3, 5, 10], p=[0.36, 0.28, 0.24, 0.12]))}
    theta.setdefault('select_src_per_subtask', False)
    theta.setdefault('transform_first_robot_pose', False)
    theta.setdefault('interpolate_from_last_target_pose', True)
    theta.setdefault('selection_strategy', 'nearest_neighbor_object')
    theta.setdefault('action_noise', 0.05)
    theta.setdefault('num_interpolation_steps', 5)
    theta.setdefault('num_fixed_steps', 0)
    theta.setdefault('offset_range', [10, 20])
    theta.setdefault('nn_k', 3)
    (lo, hi) = theta['offset_range']
    if lo > hi:
        (lo, hi) = (hi, lo)
    theta['offset_range'] = [int(lo), int(hi)]
    theta['candidate_family'] = f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}"
    return theta


# MimicGen theta application: generic
def _apply_theta_generic(task_spec: MG_TaskSpec, theta: dict[str, Any]) -> None:
    for (i, subtask) in enumerate(task_spec):
        is_final = i == len(task_spec) - 1
        has_object_ref = subtask.get('object_ref') is not None
        if has_object_ref:
            subtask['selection_strategy'] = theta['selection_strategy']
            subtask['selection_strategy_kwargs'] = None
            if theta['selection_strategy'] == 'nearest_neighbor_object':
                subtask['selection_strategy_kwargs'] = {'nn_k': int(theta['nn_k'])}
        else:
            subtask['selection_strategy'] = 'random'
            subtask['selection_strategy_kwargs'] = None
        subtask['action_noise'] = float(theta['action_noise'])
        subtask['num_interpolation_steps'] = int(theta['num_interpolation_steps'])
        subtask['num_fixed_steps'] = int(theta['num_fixed_steps'])
        if has_object_ref and (not is_final) and (subtask.get('subtask_term_signal') is not None):
            subtask['subtask_term_offset_range'] = tuple(theta['offset_range'])
        else:
            subtask['subtask_term_offset_range'] = (0, 0)


# MimicGen theta application: object_ref
def _apply_theta_object_ref(task_spec: MG_TaskSpec, theta: dict[str, Any]) -> None:
    for (i, subtask) in enumerate(task_spec):
        is_final = i == len(task_spec) - 1
        subtask['selection_strategy'] = theta['selection_strategy']
        subtask['selection_strategy_kwargs'] = None
        if theta['selection_strategy'] == 'nearest_neighbor_object':
            subtask['selection_strategy_kwargs'] = {'nn_k': int(theta['nn_k'])}
        subtask['action_noise'] = float(theta['action_noise'])
        subtask['num_interpolation_steps'] = int(theta['num_interpolation_steps'])
        subtask['num_fixed_steps'] = int(theta['num_fixed_steps'])
        if is_final:
            subtask['subtask_term_offset_range'] = (0, 0)
        else:
            subtask['subtask_term_offset_range'] = tuple(theta['offset_range'])


# MimicGen theta application: pick
def _apply_theta_pick(task_spec: MG_TaskSpec, theta: dict[str, Any]) -> None:
    for (i, subtask) in enumerate(task_spec):
        is_final = i == len(task_spec) - 1
        has_object_ref = subtask.get('object_ref') is not None
        if has_object_ref:
            subtask['selection_strategy'] = theta['selection_strategy']
            subtask['selection_strategy_kwargs'] = None
            if theta['selection_strategy'] == 'nearest_neighbor_object':
                subtask['selection_strategy_kwargs'] = {'nn_k': int(theta['nn_k'])}
        else:
            subtask['selection_strategy'] = 'random'
            subtask['selection_strategy_kwargs'] = None
        subtask['action_noise'] = float(theta['action_noise'])
        subtask['num_interpolation_steps'] = int(theta['num_interpolation_steps'])
        subtask['num_fixed_steps'] = int(theta['num_fixed_steps'])
        if has_object_ref and (not is_final) and (subtask.get('subtask_term_signal') is not None):
            subtask['subtask_term_offset_range'] = tuple(theta['offset_range'])
        else:
            subtask['subtask_term_offset_range'] = (0, 0)


# Safe repair candidates: generic
def _make_safe_candidate_generic(index: int, rng: np.random.Generator) -> dict[str, Any]:
    """Stable StackThree theta pool learned from feedback diagnostics."""
    theta = _make_candidate_generic(index, rng)
    theta['num_fixed_steps'] = 0
    if rng.random() < 0.9:
        theta['interpolate_from_last_target_pose'] = True
    if rng.random() < 0.9:
        theta['transform_first_robot_pose'] = False
    if rng.random() < 0.85:
        theta['selection_strategy'] = 'nearest_neighbor_object'
        theta['nn_k'] = int(rng.choice([1, 3, 5, 10], p=[0.35, 0.25, 0.25, 0.15]))
    else:
        theta['selection_strategy'] = 'random'
        theta['nn_k'] = int(rng.choice([1, 3, 5, 10]))
    theta['select_src_per_subtask'] = bool(rng.random() < 0.82)
    theta['action_noise'] = float(rng.choice([0.0, 0.01, 0.02, 0.05, 0.08], p=[0.12, 0.18, 0.22, 0.3, 0.18]))
    theta['num_interpolation_steps'] = int(rng.choice([3, 5, 8, 10, 15], p=[0.12, 0.28, 0.16, 0.2, 0.24]))
    offset_options = np.array([[10, 20], [10, 15], [15, 20], [5, 20], [10, 25], [0, 20], [15, 25], [0, 15], [5, 15], [10, 10], [15, 15]], dtype=int)
    offset_probs = np.array([0.2, 0.15, 0.13, 0.1, 0.09, 0.07, 0.07, 0.06, 0.06, 0.04, 0.03])
    theta['offset_range'] = offset_options[int(rng.choice(len(offset_options), p=offset_probs))].tolist()
    theta['candidate_family'] = f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}_safe"
    return theta


# Safe repair candidates: coffee
def _make_safe_candidate_coffee(index: int, rng: np.random.Generator) -> dict[str, Any]:
    """Stable Coffee theta pool; keep more random source choices from the template."""
    theta = _make_candidate_coffee(index, rng)
    theta['num_fixed_steps'] = 0
    if rng.random() < 0.92:
        theta['interpolate_from_last_target_pose'] = True
    if rng.random() < 0.92:
        theta['transform_first_robot_pose'] = False
    if rng.random() < 0.36:
        theta['selection_strategy'] = 'nearest_neighbor_object'
        theta['nn_k'] = int(rng.choice([1, 3, 5, 10], p=[0.34, 0.28, 0.24, 0.14]))
    else:
        theta['selection_strategy'] = 'random'
        theta['nn_k'] = int(rng.choice([1, 3, 5, 10]))
    theta['select_src_per_subtask'] = bool(rng.random() < 0.8)
    theta['action_noise'] = float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.05], p=[0.18, 0.22, 0.26, 0.22, 0.12]))
    theta['num_interpolation_steps'] = int(rng.choice([5, 8, 10, 15], p=[0.22, 0.2, 0.3, 0.28]))
    offset_options = np.array([[5, 10], [5, 15], [0, 10], [10, 20], [0, 15], [10, 10], [10, 15]], dtype=int)
    offset_probs = np.array([0.18, 0.2, 0.14, 0.14, 0.12, 0.08, 0.14])
    theta['offset_range'] = offset_options[int(rng.choice(len(offset_options), p=offset_probs))].tolist()
    theta['candidate_family'] = f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}_coffee_safe"
    return theta


# Safe repair candidates: nut
def _make_safe_candidate_nut(index: int, rng: np.random.Generator) -> dict[str, Any]:
    """Stable NutAssembly theta pool for two insertion subtasks."""
    theta = _make_candidate_nut(index, rng)
    theta['num_fixed_steps'] = 0
    if rng.random() < 0.94:
        theta['interpolate_from_last_target_pose'] = True
    if rng.random() < 0.92:
        theta['transform_first_robot_pose'] = False
    if rng.random() < 0.8:
        theta['selection_strategy'] = 'nearest_neighbor_object'
        theta['nn_k'] = int(rng.choice([1, 3, 5, 10], p=[0.38, 0.28, 0.22, 0.12]))
    else:
        theta['selection_strategy'] = 'random'
        theta['nn_k'] = int(rng.choice([1, 3, 5, 10]))
    theta['select_src_per_subtask'] = bool(rng.random() < 0.9)
    theta['action_noise'] = float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.05], p=[0.16, 0.2, 0.26, 0.26, 0.12]))
    theta['num_interpolation_steps'] = int(rng.choice([5, 8, 10, 15], p=[0.2, 0.2, 0.3, 0.3]))
    offset_options = np.array([[5, 10], [5, 15], [0, 10], [10, 20], [0, 15], [10, 10], [10, 15], [15, 20]], dtype=int)
    offset_probs = np.array([0.18, 0.18, 0.14, 0.14, 0.1, 0.08, 0.1, 0.08])
    theta['offset_range'] = offset_options[int(rng.choice(len(offset_options), p=offset_probs))].tolist()
    theta['candidate_family'] = f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}_nut_safe"
    return theta


# Safe repair candidates: square
def _make_safe_candidate_square(index: int, rng: np.random.Generator) -> dict[str, Any]:
    """Stable Square insertion theta pool for PINN-guided rollout."""
    theta = _make_candidate_square(index, rng)
    theta['num_fixed_steps'] = 0
    if rng.random() < 0.94:
        theta['interpolate_from_last_target_pose'] = True
    if rng.random() < 0.92:
        theta['transform_first_robot_pose'] = False
    if rng.random() < 0.84:
        theta['selection_strategy'] = 'nearest_neighbor_object'
        theta['nn_k'] = int(rng.choice([1, 3, 5, 10], p=[0.38, 0.28, 0.22, 0.12]))
    else:
        theta['selection_strategy'] = 'random'
        theta['nn_k'] = int(rng.choice([1, 3, 5, 10]))
    theta['select_src_per_subtask'] = bool(rng.random() < 0.88)
    theta['action_noise'] = float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.05], p=[0.18, 0.22, 0.24, 0.24, 0.12]))
    theta['num_interpolation_steps'] = int(rng.choice([5, 8, 10, 15], p=[0.22, 0.2, 0.3, 0.28]))
    offset_options = np.array([[5, 10], [5, 15], [0, 10], [10, 20], [0, 15], [10, 10], [10, 15]], dtype=int)
    offset_probs = np.array([0.2, 0.2, 0.16, 0.14, 0.12, 0.08, 0.1])
    theta['offset_range'] = offset_options[int(rng.choice(len(offset_options), p=offset_probs))].tolist()
    theta['candidate_family'] = f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}_square_safe"
    return theta


TASK_REGISTRY: dict[str, TaskAdapter] = {
    "square": TaskAdapter(_metrics_square, _make_candidate_square, _apply_theta_object_ref, _make_safe_candidate_square),
    "nut_assembly": TaskAdapter(_metrics_nut, _make_candidate_nut, _apply_theta_object_ref, _make_safe_candidate_nut),
    "coffee": TaskAdapter(_metrics_coffee, _make_candidate_coffee, _apply_theta_object_ref, _make_safe_candidate_coffee),
    "coffee_preparation": TaskAdapter(_metrics_generic, _make_candidate_generic, _apply_theta_generic, _make_safe_candidate_generic),
    "hammer_cleanup": TaskAdapter(_metrics_generic, _make_candidate_generic, _apply_theta_generic, _make_safe_candidate_generic),
    "kitchen": TaskAdapter(_metrics_generic, _make_candidate_generic, _apply_theta_generic, _make_safe_candidate_generic),
    "mug_cleanup": TaskAdapter(_metrics_generic, _make_candidate_generic, _apply_theta_generic, _make_safe_candidate_generic),
    "pick_place": TaskAdapter(_metrics_pick, _make_candidate_generic, _apply_theta_pick, _make_safe_candidate_generic),
    "stack": TaskAdapter(_metrics_stack, _make_candidate_generic, _apply_theta_object_ref, _make_safe_candidate_generic),
    "stack_three": TaskAdapter(_metrics_stack_three, _make_candidate_generic, _apply_theta_object_ref, _make_safe_candidate_generic),
    "threading": TaskAdapter(_metrics_generic, _make_candidate_generic, _apply_theta_generic, _make_safe_candidate_generic),
    "three_piece_assembly": TaskAdapter(_metrics_generic, _make_candidate_generic, _apply_theta_generic, _make_safe_candidate_generic),
}



def dataset_paths(dataset: str | Path) -> tuple[Path, Path, Path]:
    root = Path(dataset).expanduser().resolve()
    demo = root / "demo" if (root / "demo").is_dir() else root
    config = demo / "mg_config.json"
    success = demo / "demo.hdf5"
    failed = demo / "demo_failed.hdf5"
    for path in (config, success, failed):
        if not path.exists():
            raise FileNotFoundError(path)
    return config, success, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal 12-task RP-RF PINN inference")
    parser.add_argument("--task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--dataset", required=True, help="Dataset root containing demo/mg_config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rollout-seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=900000)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--candidate-mode", choices=["default", "safe"], default="safe")
    parser.add_argument("--max-failed-demos", type=int, default=0)
    parser.add_argument("--source-hdf5", default=None)
    parser.add_argument("--export-success-hdf5", default=None)
    parser.add_argument("--export-merged-hdf5", default=None)
    parser.add_argument("--no-stop-after-success", action="store_false", dest="stop_after_success")
    parser.set_defaults(stop_after_success=True)
    args = parser.parse_args()

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    adapter = TASK_REGISTRY[args.task]
    config_path, success_hdf5, failed_hdf5 = dataset_paths(args.dataset)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path, args.source_hdf5)
    raw_success = count_demos(success_hdf5)
    raw_failed = count_demos(failed_hdf5)
    raw_total = raw_success + raw_failed
    num_success_anchors = set_success_anchors(success_hdf5)
    contexts = load_failed_contexts(
        failed_hdf5,
        args.max_failed_demos if args.max_failed_demos > 0 else None,
        adapter.metrics,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, task_to_id, checkpoint_meta = pinn.load_checkpoint(Path(args.checkpoint), device)
    if args.task not in task_to_id:
        raise RuntimeError(f"Task {args.task} is not in checkpoint: {sorted(task_to_id)}")

    plan_path = output / "candidate_plan.jsonl"
    plan_records = [
        {
            "task": args.task,
            "demo_key": context["demo_key"],
            "context_metrics": context["context_metrics"],
            "success": False,
        }
        for context in contexts
    ]
    plan_info = pinn.build_candidate_plan(
        task=args.task,
        task_to_id=task_to_id,
        records=plan_records,
        model=model,
        out_path=plan_path,
        pool_size=args.pool_size,
        budget=args.budget,
        seed=args.seed,
        start_index=args.start_index,
        boundary_weight=args.boundary_weight,
        include_repaired=False,
        candidate_mode=args.candidate_mode,
        demo_sort_key=demo_sort_key,
        make_candidate=adapter.make_candidate,
        make_safe_candidate=adapter.make_safe_candidate,
    )
    candidate_plan = load_candidate_plan(plan_path)

    env, env_interface, data_generator = make_env_and_generator(cfg)
    exceptions_to_except = tuple(env.rollout_exceptions)
    feedback_path = output / "feedback_candidates.jsonl"
    repaired: dict[str, dict[str, Any]] = {}
    repaired_trajs: list[dict[str, Any]] = []
    per_demo: dict[str, dict[str, Any]] = {}
    num_candidates = 0
    num_candidate_success = 0
    num_problematic = 0
    rollout_seed_base = args.rollout_seed if args.rollout_seed is not None else args.seed + 10000

    with feedback_path.open("w", encoding="utf-8") as feedback:
        for demo_idx, context in enumerate(contexts):
            demo_key = context["demo_key"]
            candidates = candidate_plan.get(demo_key, [])
            per_demo[demo_key] = {
                "context_energy": context["context_metrics"]["energy"],
                "num_candidates": 0,
                "num_success": 0,
                "best_energy": None,
                "best_success_candidate": None,
            }
            for local_idx, candidate in enumerate(candidates):
                candidate_index = int(candidate.get("candidate_index", local_idx))
                theta = candidate.get("theta", candidate)
                adapter.apply_theta(data_generator.task_spec, theta)
                rollout_seed = rollout_seed_base + demo_idx * 10000 + candidate_index
                random.seed(rollout_seed)
                np.random.seed(rollout_seed)
                row: dict[str, Any] = {
                    "task": args.task,
                    "demo_key": demo_key,
                    "candidate_index": candidate_index,
                    "candidate_local_index": local_idx,
                    "rollout_seed": rollout_seed,
                    "theta": theta,
                    "planner_score": candidate.get("planner_score"),
                    "planner_rank": candidate.get("planner_rank"),
                    "context_metrics": context["context_metrics"],
                    "success": False,
                    "problematic": False,
                }
                rollout_started = time.time()
                try:
                    trajectory = generate_from_initial_state(
                        data_generator=data_generator,
                        env=env,
                        env_interface=env_interface,
                        initial_state=context["initial_state"],
                        theta=theta,
                    )
                    if trajectory["datagen_infos"]:
                        final_pos = object_positions_from_datagen_info(trajectory["datagen_infos"][-1])
                    else:
                        final_pos = object_positions_from_datagen_info(env_interface.get_datagen_info())
                    metrics = adapter.metrics(final_pos)
                    success = bool(trajectory["success"])
                    row.update(
                        {
                            "success": success,
                            "metrics": metrics,
                            "src_demo_inds": trajectory["src_demo_inds"],
                            "num_steps": int(trajectory["actions"].shape[0]),
                            "elapsed_sec": float(time.time() - rollout_started),
                        }
                    )
                except exceptions_to_except as exc:
                    num_problematic += 1
                    row.update(
                        {
                            "problematic": True,
                            "exception": repr(exc),
                            "elapsed_sec": float(time.time() - rollout_started),
                        }
                    )
                except Exception as exc:
                    num_problematic += 1
                    row.update(
                        {
                            "problematic": True,
                            "exception": repr(exc),
                            "elapsed_sec": float(time.time() - rollout_started),
                        }
                    )

                feedback.write(json.dumps(row, ensure_ascii=True) + "\n")
                feedback.flush()
                num_candidates += 1
                per_demo[demo_key]["num_candidates"] += 1
                if row.get("metrics") is not None:
                    energy = row["metrics"]["energy"]
                    best_energy = per_demo[demo_key]["best_energy"]
                    if best_energy is None or energy < best_energy:
                        per_demo[demo_key]["best_energy"] = energy
                        per_demo[demo_key]["best_theta"] = theta
                if row["success"]:
                    num_candidate_success += 1
                    per_demo[demo_key]["num_success"] += 1
                    if demo_key not in repaired:
                        per_demo[demo_key]["best_success_candidate"] = candidate_index
                        repaired[demo_key] = row
                        repaired_trajs.append(
                            {
                                "task": args.task,
                                "demo_key": demo_key,
                                "candidate_index": candidate_index,
                                "theta": theta,
                                "traj": trajectory,
                            }
                        )
                    if args.stop_after_success:
                        break
            print(
                f"[{args.task}:{demo_key}] success={per_demo[demo_key]['num_success']} "
                f"candidates={per_demo[demo_key]['num_candidates']}"
            )

    repaired_keys = sorted(repaired, key=demo_sort_key)
    evaluated_keys = {context["demo_key"] for context in contexts}
    repaired_count = len(repaired_keys)
    repaired_success_hdf5 = args.export_success_hdf5
    merged_hdf5 = args.export_merged_hdf5
    if repaired_success_hdf5:
        write_repaired_success_hdf5(repaired_success_hdf5, repaired_trajs, success_hdf5)
    if merged_hdf5:
        if not repaired_success_hdf5:
            repaired_success_hdf5 = str(output / "demo_repaired_success.hdf5")
            write_repaired_success_hdf5(repaired_success_hdf5, repaired_trajs, success_hdf5)
        merge_success_and_repaired_hdf5(success_hdf5, repaired_success_hdf5, merged_hdf5)
    summary = {
        "method": "universal_12task_rp_rf_pinn_cold_start",
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_trained_device": checkpoint_meta.get("trained_device", "unknown"),
        "runtime_device": str(device),
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "config": str(config_path),
        "source_hdf5": cfg["experiment"]["source"]["dataset_path"],
        "success_hdf5": str(success_hdf5),
        "failed_hdf5": str(failed_hdf5),
        "raw_success": raw_success,
        "raw_failed": raw_failed,
        "raw_total": raw_total,
        "raw_success_rate": raw_success / raw_total if raw_total else 0.0,
        "num_success_anchors": num_success_anchors,
        "evaluated_failed_demos": len(contexts),
        "pool_size": args.pool_size,
        "budget_per_demo": args.budget,
        "candidate_mode": args.candidate_mode,
        "stop_after_success": args.stop_after_success,
        "num_candidates": num_candidates,
        "num_candidate_success": num_candidate_success,
        "candidate_success_rate": num_candidate_success / num_candidates if num_candidates else 0.0,
        "num_problematic": num_problematic,
        "repaired_demo_keys": repaired_keys,
        "repaired_count": repaired_count,
        "unrepaired_demo_keys": sorted(evaluated_keys - set(repaired_keys), key=demo_sort_key),
        "final_success": raw_success + repaired_count,
        "final_success_rate": (raw_success + repaired_count) / raw_total if raw_total else 0.0,
        "repaired_success_hdf5": repaired_success_hdf5,
        "merged_hdf5": merged_hdf5,
        "plan_info": plan_info,
        "feedback_path": str(feedback_path),
        "per_demo": per_demo,
        "elapsed_sec": float(time.time() - started),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_success": f"{raw_success}/{raw_total}",
                "repaired": f"{repaired_count}/{len(contexts)}",
                "final_success": f"{raw_success + repaired_count}/{raw_total}",
                "num_candidates": num_candidates,
                "num_problematic": num_problematic,
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

