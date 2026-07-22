#!/usr/bin/env python3
"""Replay HDF5 demo and validate stack-cube behavior (Isaac subprocess)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Replay Stack Cube HDF5 and audit behavior.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0")
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--demo_index", type=int, default=0)
parser.add_argument("--report_out", type=str, required=True)

try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    print(f"ERROR: isaaclab unavailable: {exc}", file=sys.stderr)
    sys.exit(1)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from integrations.isaac_lab.scripts.stack_cube_expert_policy_behavior import (
    GRASP_LIFT_MIN_Z,
    HEIGHT_DIFF,
    PLACE_XY_TOLERANCE,
    PLACE_Z_TOLERANCE,
    compute_stack_error,
)


def _object_pos(env, name: str) -> torch.Tensor:
    return env.scene[name].data.root_pos_w[0] - env.scene.env_origins[0]


def main() -> int:
    dataset_path = Path(args_cli.dataset_file)
    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_path))
    episode_names = list(handler.get_episode_names())
    if not episode_names:
        print("No episodes")
        return 1

    demo_index = int(args_cli.demo_index)
    if demo_index < 0 or demo_index >= len(episode_names):
        print(f"Invalid demo_index {demo_index}")
        return 1

    env_cfg = parse_env_cfg(args_cli.task.split(":")[-1], device=args_cli.device, num_envs=1)
    success_term = env_cfg.terminations.success if hasattr(env_cfg.terminations, "success") else None
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    ep = handler.load_episode(episode_names[demo_index], env.device)
    env.reset()
    actions = ep.data["actions"]

    grasp_z_before: dict[str, float] = {}
    cube_lifted_flags = [False, False]
    cube_placed_flags = [False, False]
    max_cube_z_during_grasp = [0.0, 0.0]
    pick_objects = ["cube_2", "cube_3"]
    pick_idx = 0
    prev_gripper = 1.0

    with torch.inference_mode():
        for step_i in range(actions.shape[0]):
            action = actions[step_i : step_i + 1]
            gripper = float(action[0, -1].item())
            if prev_gripper > 0.25 and gripper < -0.25 and pick_idx < len(pick_objects):
                obj = pick_objects[pick_idx]
                grasp_z_before[obj] = float(_object_pos(env, obj)[2].item())
            env.step(action)
            if pick_idx < len(pick_objects):
                obj = pick_objects[pick_idx]
                if obj in grasp_z_before:
                    dz = float(_object_pos(env, obj)[2].item()) - grasp_z_before[obj]
                    max_cube_z_during_grasp[pick_idx] = max(max_cube_z_during_grasp[pick_idx], dz)
                    if dz >= GRASP_LIFT_MIN_Z:
                        cube_lifted_flags[pick_idx] = True
            if prev_gripper < -0.25 and gripper > 0.25 and pick_idx < len(pick_objects):
                pick_idx += 1
            prev_gripper = gripper

    final_positions = []
    for name in ("cube_1", "cube_2", "cube_3"):
        pos = _object_pos(env, name)
        final_positions.append([float(pos[0].item()), float(pos[1].item()), float(pos[2].item())])

    stack_error = compute_stack_error(final_positions)
    final_success = False
    if success_term is not None:
        env_cfg.terminations.success = success_term
        final_success = bool(success_term.func(env, **success_term.params)[0])

    c2 = _object_pos(env, "cube_2")
    c1 = _object_pos(env, "cube_1")
    c3 = _object_pos(env, "cube_3")
    place2_xy = float(torch.norm(c2[:2] - c1[:2]).item())
    place3_xy = float(torch.norm(c3[:2] - c2[:2]).item())
    place2_z = abs(float(c2[2].item() - c1[2].item() - HEIGHT_DIFF))
    place3_z = abs(float(c3[2].item() - c2[2].item() - HEIGHT_DIFF))
    cube_placed_flags[0] = place2_xy <= PLACE_XY_TOLERANCE and place2_z <= PLACE_Z_TOLERANCE
    cube_placed_flags[1] = place3_xy <= PLACE_XY_TOLERANCE and place3_z <= PLACE_Z_TOLERANCE

    replay_success = final_success and all(cube_lifted_flags) and all(cube_placed_flags)
    failure_reason = None
    if not all(cube_lifted_flags):
        failure_reason = "grasp_not_lifted"
    elif not all(cube_placed_flags):
        failure_reason = "place_error"
    elif not final_success:
        failure_reason = "success_term_false"

    report = {
        "demoIndex": demo_index,
        "replaySuccess": replay_success,
        "finalSuccessTerm": final_success,
        "cubeLiftedFlags": cube_lifted_flags,
        "cubePlacedFlags": cube_placed_flags,
        "maxCubeZDuringGrasp": max_cube_z_during_grasp,
        "finalCubePositions": final_positions,
        "finalStackError": stack_error,
        "graspVerified": all(cube_lifted_flags),
        "placeVerified": all(cube_placed_flags),
        "failureReason": failure_reason,
    }

    out = Path(args_cli.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

    env.close()
    handler.close()
    simulation_app.close()
    return 0 if replay_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
