import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from examples.dlo.task_registry import (
    ENV_OPTION_CABLE_MODEL,
    ENV_OPTION_GOAL_CHARACTER,
    ENV_OPTION_GOAL_FILE,
    ENV_OPTION_GRASP_MODE,
    SOURCE_OPTION_RMB_ROBOT_PRESET,
    SOURCE_OPTION_RMB_WORLD_IDX,
    SOURCE_OPTION_RMB_WORLD_RANDOM_SCALE,
    TELEOP_DLO_HUMAN,
    TELEOP_THREADING_HUMAN,
    TASKS,
    TASK_SPECS,
    expert_tasks,
    get_task_spec,
    tasks_with_expert_entry,
    tasks_with_teleop_entry,
)


SCENE_RANDOMIZATION_CHOICES = ("random", "fixed")
GRASP_MODE_CHOICES = ("attachment", "physical")

DEFAULT_ROBOTS = {name: spec.default_robot for name, spec in TASK_SPECS.items()}
DEFAULT_CAMERAS = {name: spec.default_camera for name, spec in TASK_SPECS.items()}


def spec_supports(task, option):
    return get_task_spec(task).supports_option(option)


BASE_ENDPOINT_TASKS = {name for name, spec in TASK_SPECS.items() if spec.task_family == "endpoint_manipulation"}
DEFORMABLE_RAVENS_TASKS = {name for name, spec in TASK_SPECS.items() if spec.source == "deformable-ravens"}
DEFORMABLE_RAVENS_GOAL_FILE_TASKS = {name for name in DEFORMABLE_RAVENS_TASKS if spec_supports(name, ENV_OPTION_GOAL_FILE)}
SOFTGYM_ROPE_TASKS = {name for name, spec in TASK_SPECS.items() if spec.source == "SoftGym"}
RMB_TASKS = {name for name, spec in TASK_SPECS.items() if spec.source == "RoboManipBaselines"}
DLO_HUMAN_COLLECTOR_TASKS = tasks_with_teleop_entry(TELEOP_DLO_HUMAN)
THREADING_HUMAN_COLLECTOR_TASKS = tasks_with_teleop_entry(TELEOP_THREADING_HUMAN)
EXPERT_TASKS = expert_tasks()


def _capability_from_spec(spec):
    return {
        "name": spec.name,
        "source": spec.source,
        "task_family": spec.task_family,
        "default_robot": spec.default_robot,
        "default_camera": spec.default_camera,
        "default_controller": spec.default_controller,
        "default_cable_model": spec.default_cable_model,
        "supported_cable_models": list(spec.supported_cable_models),
        "env_options": list(spec.env_options),
        "source_options": list(spec.source_options),
        "source_defaults": dict(spec.source_defaults),
        "supports_cable_model": spec.supports_option(ENV_OPTION_CABLE_MODEL),
        "supports_grasp_mode": spec.supports_option(ENV_OPTION_GRASP_MODE),
        "supports_goal_file": spec.supports_option(ENV_OPTION_GOAL_FILE),
        "supports_goal_character": spec.supports_option(ENV_OPTION_GOAL_CHARACTER),
        "supports_rmb_robot_preset": spec.supports_option(SOURCE_OPTION_RMB_ROBOT_PRESET),
        "supports_rmb_world_idx": spec.supports_option(SOURCE_OPTION_RMB_WORLD_IDX),
        "supports_rmb_world_random_scale": spec.supports_option(SOURCE_OPTION_RMB_WORLD_RANDOM_SCALE),
        "default_rmb_robot_preset": spec.default_for(SOURCE_OPTION_RMB_ROBOT_PRESET),
        "teleop_collector": spec.teleop_entry,
        "expert_entry": spec.expert_entry,
        "supports_recording_schema": spec.supports_recording_schema,
        "default_recording_schema": spec.default_recording_schema,
        "default_obs_keys": list(spec.default_obs_keys),
    }


TASK_CAPABILITIES = {name: _capability_from_spec(spec) for name, spec in TASK_SPECS.items()}


def task_capability(task):
    try:
        return TASK_CAPABILITIES[str(task)]
    except KeyError:
        raise ValueError(f"Unsupported task: {task}")


def control_panel_capabilities():
    return {task: {**capability} for task, capability in TASK_CAPABILITIES.items()}


def supported_env_kwargs(task, values):
    cap = task_capability(task)
    kwargs = {}
    if cap["supports_cable_model"]:
        kwargs["cable_model"] = values.get("cable_model") or cap["default_cable_model"]
    if cap["supports_rmb_robot_preset"] and values.get("rmb_robot_preset"):
        kwargs["rmb_robot_preset"] = values["rmb_robot_preset"]
    if cap["supports_rmb_world_idx"] and values.get("rmb_world_idx") is not None:
        kwargs["rmb_world_idx"] = values["rmb_world_idx"]
    if cap["supports_rmb_world_idx"] and values.get("rmb_world_random_scale") is not None:
        kwargs["rmb_world_random_scale"] = values["rmb_world_random_scale"]
    if cap["supports_goal_file"] and values.get("goal_file"):
        kwargs["goal_file"] = values["goal_file"]
    if cap["supports_goal_character"] and values.get("goal_character"):
        kwargs["goal_character"] = values["goal_character"]
    if cap["supports_grasp_mode"] and values.get("grasp_mode"):
        kwargs["grasp_mode"] = values["grasp_mode"]
    if values.get("anchor_enabled") is not None:
        kwargs["anchor_enabled"] = values["anchor_enabled"]
    return kwargs


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def print_json(label, value):
    print(f"{label}: {json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)}")


def controller_for(robot, controller=None):
    if controller:
        try:
            return load_composite_controller_config(robot=robot, controller=controller)
        except TypeError:
            return load_composite_controller_config(controller=controller)
    try:
        return load_composite_controller_config(robot=robot)
    except TypeError:
        return load_composite_controller_config(controller="BASIC")


_ROBOT_CAMERA_NAMES = {"eye_in_hand", "robotview"}


def _resolve_robot_camera(name, robot_prefix="robot0"):
    """Robot-local camera names get prefixed when merged into the env (e.g. eye_in_hand -> robot0_eye_in_hand)."""
    if name in _ROBOT_CAMERA_NAMES:
        return f"{robot_prefix}_{name}"
    return name


def make_dlo_env(
    env_name,
    robot=None,
    horizon=None,
    seed=None,
    render=False,
    offscreen=False,
    camera="frontview",
    controller=None,
    record_video=False,
    camera_names=None,
    camera_height=None,
    image_width=None,
    image_height=None,
    control_freq=20,
    **kwargs,
):
    robot = robot or DEFAULT_ROBOTS.get(env_name, "Panda")
    camera = camera or DEFAULT_CAMERAS.get(env_name, "frontview")
    camera = _resolve_robot_camera(camera)
    if record_video and camera_names is None:
        camera_names = [camera]
    if camera_names is not None:
        camera_names = [_resolve_robot_camera(c) for c in camera_names]
    controller_config = controller_for(robot, controller=controller)
    needs_offscreen = offscreen or record_video or camera_names is not None
    camera_widths = None
    camera_heights = None
    if image_width is not None:
        width = int(image_width)
        camera_widths = [width] * len(camera_names) if camera_names is not None else width
    elif record_video:
        camera_widths = [1280] * len(camera_names) if camera_names is not None else 1280
    if image_height is not None:
        height = int(image_height)
        camera_heights = [height] * len(camera_names) if camera_names is not None else height
    elif record_video:
        camera_heights = [720] * len(camera_names) if camera_names is not None else 720
    make_kwargs = dict(
        env_name=env_name,
        robots=robot,
        controller_configs=controller_config,
        has_renderer=render,
        has_offscreen_renderer=needs_offscreen,
        render_camera=camera,
        use_camera_obs=record_video or camera_names is not None,
        use_object_obs=True,
        seed=seed,
        control_freq=control_freq,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
    )
    if horizon is not None:
        make_kwargs["horizon"] = horizon
    make_kwargs.update(kwargs)
    env = suite.make(**make_kwargs)
    if camera_height is not None and hasattr(env, "arena"):
        _apply_camera_height(env.arena, "agentview", float(camera_height))
    return env


def _apply_camera_height(arena, camera_name, height):
    from robosuite.utils.mjcf_utils import find_elements, string_to_array, array_to_string
    camera = find_elements(root=arena.worldbody, tags="camera", attribs={"name": camera_name}, return_first=True)
    if camera is None:
        return
    pos = string_to_array(camera.get("pos"))
    pos[2] = height
    quat = string_to_array(camera.get("quat"))
    arena.set_camera(camera_name=camera_name, pos=pos.tolist(), quat=quat.tolist())


def add_scene_randomization_arg(parser, default=None):
    parser.add_argument(
        "--scene-randomization",
        choices=SCENE_RANDOMIZATION_CHOICES,
        default=default,
        help="Initial scene reset mode: random samples task reset state; fixed uses deterministic task reset where supported.",
    )
    parser.add_argument(
        "--fixed-initial-state",
        action="store_true",
        help="Alias for --scene-randomization fixed.",
    )
    parser.add_argument(
        "--random-initial-state",
        action="store_true",
        help="Alias for --scene-randomization random.",
    )


def add_grasp_mode_arg(parser, default="attachment"):
    parser.add_argument(
        "--grasp-mode",
        choices=GRASP_MODE_CHOICES,
        default=default,
        help="Cable grasping mode: attachment keeps legacy mocap weld behavior; physical uses only contacts and gripper force.",
    )


def resolve_scene_randomization(args, default="random"):
    if getattr(args, "fixed_initial_state", False) and getattr(args, "random_initial_state", False):
        raise ValueError("Use only one of --fixed-initial-state or --random-initial-state")
    if getattr(args, "fixed_initial_state", False):
        return "fixed"
    if getattr(args, "random_initial_state", False):
        return "random"
    value = getattr(args, "scene_randomization", None)
    return value or default


def apply_scene_randomization(env, mode):
    mode = str(mode)
    if mode not in SCENE_RANDOMIZATION_CHOICES:
        raise ValueError(f"Unsupported scene randomization mode: {mode}")
    if hasattr(env, "deterministic_reset"):
        env.deterministic_reset = mode == "fixed"
    return mode


def configure_env_scene_randomization(env, args, default="random"):
    mode = resolve_scene_randomization(args, default=default)
    return apply_scene_randomization(env, mode)


def add_common_env_args(parser):
    parser.add_argument(
        "--env",
        default="CableStraighten",
        choices=TASKS,
    )
    parser.add_argument("--robot", default=None)
    parser.add_argument("--horizon", type=int, default=None, help="每 episode 最大步数（默认按 20Hz/500 步自动缩放）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--camera-height", type=float, default=None, help="覆盖 agentview 相机高度 (z 坐标)")
    parser.add_argument("--controller", default=None)
    parser.add_argument("--cable-model", default=None)
    parser.add_argument("--rmb-robot-preset", default=None)
    parser.add_argument("--rmb-world-idx", type=int, default=None)
    parser.add_argument("--rmb-world-random-scale", type=float, default=None)
    parser.add_argument("--goal-file", default=None)
    parser.add_argument("--goal-character", default=None)
    add_scene_randomization_arg(parser)
    add_grasp_mode_arg(parser)
    parser.add_argument("--no-anchor", action="store_true", help="不锚定线缆起点（两端均可自由移动）")
    parser.add_argument("--enable-wrist-cam", action="store_true", help="启用腕部相机 (eye_in_hand) 作为额外观察源")
    parser.add_argument("--record-video", action="store_true", help="录制后台渲染视频")
    parser.add_argument("--video-output", default=None, help="视频输出路径，默认 ~/Videos/rms/<task>_<timestamp>.mp4")
    parser.add_argument("--video-fps", type=int, default=20, help="视频帧率，默认 20")
    parser.add_argument("--video-width", type=int, default=1280, help="离屏录制宽度，默认 1280")
    parser.add_argument("--video-height", type=int, default=720, help="离屏录制高度，默认 720")
    parser.add_argument(
        "--video-quality",
        choices=("standard", "high", "ultra"),
        default="high",
        help="视频编码质量预设：standard / high / ultra",
    )


def env_kwargs_from_args(args):
    kwargs = {
        "cable_model": getattr(args, "cable_model", None),
        "rmb_robot_preset": getattr(args, "rmb_robot_preset", None),
        "rmb_world_idx": getattr(args, "rmb_world_idx", None),
        "rmb_world_random_scale": getattr(args, "rmb_world_random_scale", None),
        "goal_file": getattr(args, "goal_file", None),
        "goal_character": getattr(args, "goal_character", None),
        "grasp_mode": getattr(args, "grasp_mode", None),
    }
    if getattr(args, "no_anchor", False):
        kwargs["anchor_enabled"] = False
    return supported_env_kwargs(getattr(args, "env", "CableStraighten"), kwargs)
