import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.cable_threading.utils import (
    DEFAULT_OBS_KEYS,
    SUPPORTED_CABLE_MODELS,
    SUPPORTED_DIFFICULTIES,
    aggregate_rows,
    clip_action,
    make_env,
    obs_to_vector,
    save_dataset,
    summarize_episode,
    write_results_csv,
)
from examples.dlo.common import add_grasp_mode_arg, apply_scene_randomization, resolve_scene_randomization
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.utils.dlo.episode_schema import formal_metadata


def make_device(env, device_name, pos_sensitivity, rot_sensitivity):
    if device_name == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
        return device
    if device_name == "spacemouse":
        from robosuite.devices import SpaceMouse

        return SpaceMouse(env=env, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
    if device_name == "dualsense":
        from robosuite.devices import DualSense

        return DualSense(env=env, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
    if device_name == "mjgui":
        from robosuite.devices.mjgui import MJGUI

        return MJGUI(env=env)
    raise ValueError(f"Unsupported teleop device: {device_name}")


def teleop_action_to_env_action(env, device, input_action_dict, previous_gripper_actions):
    active_robot = env.robots[device.active_robot]
    action_dict = deepcopy(input_action_dict)

    for arm in active_robot.arms:
        if isinstance(active_robot.composite_controller, WholeBody):
            controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
        else:
            controller_input_type = active_robot.part_controllers[arm].input_type

        if controller_input_type == "delta":
            action_dict[arm] = input_action_dict[f"{arm}_delta"]
        elif controller_input_type == "absolute":
            action_dict[arm] = input_action_dict[f"{arm}_abs"]
        else:
            raise ValueError(f"Unsupported controller input_type: {controller_input_type}")

    env_action = [robot.create_action_vector(previous_gripper_actions[i]) for i, robot in enumerate(env.robots)]
    env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
    env_action = np.concatenate(env_action)

    for gripper_action_name in previous_gripper_actions[device.active_robot]:
        previous_gripper_actions[device.active_robot][gripper_action_name] = action_dict[gripper_action_name]

    return clip_action(env, env_action)


def update_task_attachment(env, action, attach_distance):
    should_close = bool(np.asarray(action)[-1] > 0.0)
    if not should_close:
        env.set_attachment_enabled(False)
        return

    if env.attachment_enabled:
        return

    eef_to_cable = float(np.linalg.norm(env._get_gripper_site_position() - env._get_cable_end_pos()))
    if eef_to_cable <= attach_distance:
        env.set_attachment_enabled(True)


def update_viewer(env):
    if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
        env.viewer.update()
    else:
        env.render()


def rollout_human_episode(
    env,
    device,
    *,
    episode_index,
    seed,
    obs_keys,
    attach_distance,
    max_fr,
    render_sleep,
    end_on_reset_key,
):
    if seed is not None:
        env.seed = int(seed)
        env.rng = np.random.default_rng(seed)
    obs = env.reset()
    env.set_attachment_enabled(False)
    update_viewer(env)

    device.start_control()
    previous_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    total_reward = 0.0
    rows = []
    transitions = []
    reset_requested = False

    while len(rows) < env.horizon:
        start = time.time()
        input_action_dict = device.input2action()
        if input_action_dict is None:
            if end_on_reset_key:
                reset_requested = True
                break
            device.start_control()
            continue

        action = teleop_action_to_env_action(env, device, input_action_dict, previous_gripper_actions)
        update_task_attachment(env, action, attach_distance=attach_distance)

        prev_obs = obs
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        rows.append(dict(info))
        transitions.append(
            {
                "obs": obs_to_vector(prev_obs, obs_keys),
                "next_obs": obs_to_vector(obs, obs_keys),
                "action": action.copy(),
                "reward": float(reward),
                "done": bool(done),
                "info": dict(info),
                "phase": "human_teleop",
            }
        )

        update_viewer(env)
        if render_sleep > 0:
            time.sleep(render_sleep)
        if max_fr is not None:
            elapsed = time.time() - start
            sleep_time = 1.0 / max_fr - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if done:
            break
    ended_by_horizon = len(rows) >= env.horizon

    summary = summarize_episode(
        rows,
        env,
        total_reward,
        policy_name="human_teleop",
        episode_index=episode_index,
        seed=seed,
        phase_log=["human_teleop"] * len(rows),
    )
    summary["reset_requested"] = bool(reset_requested)
    summary["ended_by_horizon"] = bool(ended_by_horizon)
    return summary, transitions


def reset_summary_jsonable(env):
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in env.last_reset_summary.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CableThreading")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot", type=str, default="UR5e")
    parser.add_argument("--cable-model", choices=SUPPORTED_CABLE_MODELS, default="rmb")
    add_grasp_mode_arg(parser)
    parser.add_argument("--rmb-robot-preset", default=None)
    parser.add_argument("--rmb-world-idx", type=int, default=None)
    parser.add_argument("--rmb-world-random-scale", type=float, default=None)
    parser.add_argument("--difficulty", choices=SUPPORTED_DIFFICULTIES, default="easy")
    parser.add_argument("--device", choices=["keyboard", "spacemouse", "dualsense", "mjgui"], default="keyboard")
    parser.add_argument("--camera", type=str, default="topview")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--attach-distance", type=float, default=0.04)
    parser.add_argument("--max-fr", type=int, default=20)
    parser.add_argument("--render-sleep", type=float, default=0.0)
    parser.add_argument("--scene-randomization", choices=["random", "fixed"], default="random")
    parser.add_argument("--fixed-initial-state", action="store_true")
    parser.add_argument("--random-initial-state", action="store_true")
    parser.add_argument(
        "--end-on-reset-key",
        action="store_true",
        help="End the current episode when the device reset key is pressed. Keyboard reset is q in this robosuite version.",
    )
    parser.add_argument("--out", type=str, default="datasets/cable_threading/ur5e_rmb_human_train.npz")
    parser.add_argument("--results-out", type=str, default="results/cable_threading_human_collect.csv")
    parser.add_argument("--failures-out", type=str, default="results/cable_threading_human_collect_failures.json")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not raise an error if no final-success human episodes were collected.",
    )
    parser.add_argument("--save-failures", action="store_true", help="Save all episodes, not only final-success episodes.")
    args = parser.parse_args()

    env_kwargs = {}
    if args.rmb_robot_preset:
        env_kwargs["rmb_robot_preset"] = args.rmb_robot_preset
    if args.rmb_world_idx is not None:
        env_kwargs["rmb_world_idx"] = args.rmb_world_idx
    if args.rmb_world_random_scale is not None:
        env_kwargs["rmb_world_random_scale"] = args.rmb_world_random_scale

    env = make_env(
        env_name=args.env,
        robot=args.robot,
        horizon=args.horizon,
        seed=args.seed,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        cable_model=args.cable_model,
        grasp_mode=args.grasp_mode,
        difficulty=args.difficulty,
        **env_kwargs,
    )
    scene_randomization = resolve_scene_randomization(args, default="random")
    apply_scene_randomization(env, scene_randomization)

    trajectories = []
    episode_metadata = []
    rows = []
    failures = []

    try:
        device = make_device(env, args.device, args.pos_sensitivity, args.rot_sensitivity)
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            print(
                f"episode {episode}: teleoperate the task; "
                "episode ends at horizon by default; use Ctrl+C to stop the script"
            )
            summary, transitions = rollout_human_episode(
                env,
                device,
                episode_index=episode,
                seed=episode_seed,
                obs_keys=DEFAULT_OBS_KEYS,
                attach_distance=args.attach_distance,
                max_fr=args.max_fr,
                render_sleep=args.render_sleep,
                end_on_reset_key=args.end_on_reset_key,
            )
            rows.append(summary)
            print(episode, summary)

            metadata = {
                "episode": episode,
                "seed": episode_seed,
                "summary": summary,
                "reset_summary": reset_summary_jsonable(env),
                "scene_randomization": scene_randomization,
            }
            if summary["final_success"] or args.save_failures:
                trajectories.append(transitions)
                episode_metadata.append(metadata)
            else:
                failures.append(metadata)
    finally:
        env.close()

    write_results_csv(args.results_out, rows)
    stats = aggregate_rows(rows)
    print("saved_csv:", Path(args.results_out).expanduser())
    print("final_success_rate:", stats["final_success_rate"])
    print("ever_success_rate:", stats["ever_success_rate"])
    print("successful_episodes:", len(trajectories))

    failures_path = Path(args.failures_out).expanduser()
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(json.dumps(failures, indent=2))
    print("saved_failures:", failures_path)

    if not trajectories:
        if args.allow_empty:
            print("saved_dataset: skipped because no final-success human teleop episodes were collected")
            return
        raise RuntimeError("No final-success human teleop episodes were collected.")

    save_dataset(
        args.out,
        trajectories=trajectories,
        obs_keys=DEFAULT_OBS_KEYS,
        metadata=formal_metadata(
            env_name=args.env,
            robot=args.robot,
            controller="default",
            seed=args.seed,
            horizon=args.horizon,
            scene_randomization=scene_randomization,
            policy="human_teleop",
            cable_model=args.cable_model,
            grasp_mode=args.grasp_mode,
            difficulty=args.difficulty,
            success_semantics="final_only",
            obs_keys=DEFAULT_OBS_KEYS,
            env_params={
                "pole_spacing": 0.05,
                "pole_radius": 0.01,
                "pole_height": 0.06,
                "attach_distance": args.attach_distance,
                "rmb_robot_preset": args.rmb_robot_preset,
                "rmb_world_idx": args.rmb_world_idx,
            },
            data_version="cable_threading_human_v1",
        ),
        episode_metadata=episode_metadata,
    )
    print("saved_dataset:", Path(args.out).expanduser())


if __name__ == "__main__":
    main()
