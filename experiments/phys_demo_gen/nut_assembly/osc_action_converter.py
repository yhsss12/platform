"""V2-B2 / V2-B2.5：eef waypoint → OSC_POSE delta actions（7-dim）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from trajectory_parameterization import TrajectoryPhases, TrajectoryProxy, _shift_1d_signal

DEFAULT_OSC_LIMITS = {
    "output_max": np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=float),
    "output_min": np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], dtype=float),
}

NEAR_PEG_XY_THRESH = 0.08


@dataclass
class SimLoopParams:
    """V2-B2.5 仿真闭环局部搜索参数（作用于 insertion 窗口 action，不修改 object_poses）。"""

    insert_z_offset: float = 0.0
    z_gain: float = 0.55
    insertion_steps: int = 30
    hold_steps: int = 10
    insertion_speed_scale: float = 1.0
    release_shift: float = 0.0
    pre_insert_pause: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEARCH_SPACE: dict[str, list[float | int]] = {
    "insert_z_offset": [-0.04, -0.06, -0.08, -0.10, -0.12],
    "z_gain": [0.45, 0.55, 0.70, 0.85, 1.00],
    "insertion_steps": [10, 20, 30, 40],
    "hold_steps": [5, 10, 20],
    "insertion_speed_scale": [0.5, 0.75, 1.0],
    "release_shift": [-5, 0, 5, 10],
    "pre_insert_pause": [0, 5, 10],
}


def osc_limits_from_env_args(env_args: dict[str, Any]) -> dict[str, np.ndarray]:
    cfg = env_args.get("env_kwargs", env_args).get("controller_configs", {})
    if "body_parts" in cfg:
        cfg = cfg["body_parts"].get("right", cfg)
    out_max = np.array(cfg.get("output_max", DEFAULT_OSC_LIMITS["output_max"]), dtype=float)
    out_min = np.array(cfg.get("output_min", DEFAULT_OSC_LIMITS["output_min"]), dtype=float)
    return {"output_max": out_max[:6], "output_min": out_min[:6]}


def rotmat_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=float).reshape(3, 3)
    trace = np.trace(rot)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return np.zeros(3, dtype=float)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=float,
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.zeros(3, dtype=float)
    return axis / norm * angle


def encode_osc_delta(
    delta_pos: np.ndarray,
    delta_ori: np.ndarray,
    output_max: np.ndarray,
    output_min: np.ndarray,
) -> np.ndarray:
    pos_scale = np.where(delta_pos >= 0, output_max[:3], -output_min[:3])
    ori_scale = np.where(delta_ori >= 0, output_max[3:6], -output_min[3:6])
    pos_scale = np.maximum(pos_scale, 1e-6)
    ori_scale = np.maximum(ori_scale, 1e-6)
    action = np.zeros(6, dtype=float)
    action[:3] = delta_pos / pos_scale
    action[3:6] = delta_ori / ori_scale
    return np.clip(action, -1.0, 1.0)


def _insertion_gain(step: int, phases: TrajectoryPhases) -> float:
    t_min = phases.t_min_xy
    final = phases.final_index
    if step < t_min:
        return 0.08
    if step > final:
        return 0.0
    return 0.55 + 0.35 * (step - t_min) / max(1, final - t_min)


def waypoints_to_osc_actions(
    original_eef: np.ndarray,
    refined_eef: np.ndarray,
    original_actions: np.ndarray,
    shifted_gripper: np.ndarray,
    phases: TrajectoryPhases,
    osc_limits: dict[str, np.ndarray],
    speed_scale: float,
) -> np.ndarray:
    """
    以原始 action 为基底，按 refined eef waypoint 偏差注入 OSC 修正。

    真实 rollout 中 nut 不会自动跟随 proxy 规则，因此 insertion 窗口增益更高、
    transport 窗口增益更低，避免破坏已 grasp 的 transport 几何。
    """
    speed_scale = float(np.clip(speed_scale, 0.5, 1.2))
    output_max = osc_limits["output_max"]
    output_min = osc_limits["output_min"]
    length = len(original_actions)
    actions = original_actions.copy().astype(float)

    for step in range(length):
        gain = _insertion_gain(step, phases)
        delta_pos = (refined_eef[step, :3, 3] - original_eef[step, :3, 3]) * gain
        delta_rot = refined_eef[step, :3, :3] @ original_eef[step, :3, :3].T
        delta_ori = rotmat_to_axis_angle(delta_rot) * gain
        correction = encode_osc_delta(delta_pos, delta_ori, output_max, output_min)
        actions[step, :6] = np.clip(actions[step, :6] + correction, -1.0, 1.0)

    if abs(speed_scale - 1.0) > 1e-6 and length > 1:
        deltas = np.diff(actions[:, :6], axis=0) / speed_scale
        actions[0, :6] = original_actions[0, :6]
        for step in range(1, length):
            actions[step, :6] = actions[step - 1, :6] + deltas[step - 1]
            actions[step, :6] = np.clip(actions[step, :6], -1.0, 1.0)

    grip = np.clip(shifted_gripper, -1.0, 1.0)
    actions[:, 6] = grip
    return actions


def build_insertion_focused_actions(
    proxy: TrajectoryProxy,
    original_actions: np.ndarray,
    theta: dict[str, Any],
    env_args: dict[str, Any],
    *,
    z_gain: float = 0.55,
    apply_gripper_shifts: bool = False,
) -> np.ndarray:
    """insertion_failed 专用：保留原始 transport，仅修正 insertion z（可选 gripper）。"""
    limits = osc_limits_from_env_args(env_args)
    z_scale = limits["output_max"][2]
    actions = original_actions.copy().astype(float)
    phases = proxy.phases
    t_min = phases.t_min_xy
    length = proxy.length
    insert_z = float(theta["insert_z_offset"])

    for step in range(t_min, length):
        gamma = (step - t_min) / max(1, length - 1 - t_min)
        z_bias = (insert_z * gamma) / max(z_scale, 1e-6)
        actions[step, 2] = np.clip(actions[step, 2] + z_bias * z_gain, -1.0, 1.0)

    if apply_gripper_shifts:
        grip = _shift_1d_signal(proxy.gripper_action, float(theta["gripper_close_shift"]))
        grip = _shift_1d_signal(grip, float(theta["release_shift"]) * 0.5)
        actions[:, 6] = np.clip(grip, -1.0, 1.0)

    return actions


def build_refined_actions(
    proxy: TrajectoryProxy,
    original_eef: np.ndarray,
    refined_eef: np.ndarray,
    original_actions: np.ndarray,
    shifted_gripper: np.ndarray,
    phases: TrajectoryPhases,
    env_args: dict[str, Any],
    theta: dict[str, Any],
) -> np.ndarray:
    min_xy = float(proxy.xy_distance_baseline.min())
    speed_scale = float(theta.get("speed_scale", 1.0))

    if min_xy <= NEAR_PEG_XY_THRESH:
        return build_insertion_focused_actions(proxy, original_actions, theta, env_args)

    limits = osc_limits_from_env_args(env_args)
    return waypoints_to_osc_actions(
        original_eef,
        refined_eef,
        original_actions,
        shifted_gripper,
        phases,
        limits,
        speed_scale,
    )


def eef_waypoints_to_osc_actions(
    waypoints: np.ndarray,
    gripper: np.ndarray,
    env_args: dict[str, Any],
    *,
    speed_scale: float = 1.0,
) -> np.ndarray:
    """
    纯 eef waypoint → OSC open-loop actions（不依赖 HDF5 原始 actions）。

    V2-B2.5 current-controller baseline 使用此路径，保证 original / refined 公平对比。
    """
    speed_scale = float(max(speed_scale, 0.25))
    limits = osc_limits_from_env_args(env_args)
    output_max = limits["output_max"]
    output_min = limits["output_min"]
    length = len(waypoints)
    grip = np.clip(np.asarray(gripper, dtype=float).reshape(-1), -1.0, 1.0)
    actions = np.zeros((length, 7), dtype=float)

    for step in range(length):
        if step < length - 1:
            dp = (waypoints[step + 1, :3, 3] - waypoints[step, :3, 3]) / speed_scale
            drot = waypoints[step + 1, :3, :3] @ waypoints[step, :3, :3].T
            dori = rotmat_to_axis_angle(drot) / speed_scale
            actions[step, :6] = encode_osc_delta(dp, dori, output_max, output_min)
        else:
            actions[step, :6] = 0.0
        actions[step, 6] = grip[step]
    return actions


def apply_sim_loop_params(
    actions: np.ndarray,
    proxy: TrajectoryProxy,
    gripper: np.ndarray,
    env_args: dict[str, Any],
    sim_params: SimLoopParams,
) -> np.ndarray:
    """在 insertion 窗口叠加 sim-in-loop 局部搜索参数（仅改 action，不改 object_poses）。"""
    out = actions.copy().astype(float)
    phases = proxy.phases
    t_min = phases.t_min_xy
    length = proxy.length
    limits = osc_limits_from_env_args(env_args)
    z_scale = max(float(limits["output_max"][2]), 1e-6)

    pause_start = max(0, t_min - int(sim_params.pre_insert_pause))
    for step in range(pause_start, t_min):
        out[step, :6] *= 0.05

    insert_steps = max(1, int(sim_params.insertion_steps))
    insert_end = min(length, t_min + insert_steps)
    for step in range(t_min, insert_end):
        gamma = (step - t_min) / max(1, insert_end - t_min - 1) if insert_end > t_min else 1.0
        z_bias = (float(sim_params.insert_z_offset) * gamma) / z_scale * float(sim_params.z_gain)
        out[step, 2] = np.clip(out[step, 2] + z_bias, -1.0, 1.0)

    if float(sim_params.insertion_speed_scale) != 1.0 and insert_end > t_min + 1:
        scaled = out[t_min:insert_end, :6].copy()
        for step in range(t_min + 1, insert_end):
            delta = scaled[step - t_min, :6] - scaled[step - t_min - 1, :6]
            scaled[step - t_min, :6] = scaled[step - t_min - 1, :6] + delta / float(
                sim_params.insertion_speed_scale
            )
        out[t_min:insert_end, :6] = np.clip(scaled, -1.0, 1.0)

    hold_steps = int(sim_params.hold_steps)
    hold_end = min(length, insert_end + hold_steps)
    if insert_end > t_min and hold_end > insert_end:
        hold_action = out[insert_end - 1, :6].copy() * 0.15
        for step in range(insert_end, hold_end):
            out[step, :6] = hold_action

    if int(sim_params.release_shift) != 0:
        grip = _shift_1d_signal(gripper, float(sim_params.release_shift))
        out[:, 6] = np.clip(grip, -1.0, 1.0)

    return out


def build_waypoint_rollout_actions(
    proxy: TrajectoryProxy,
    target_eef: np.ndarray,
    gripper: np.ndarray,
    env_args: dict[str, Any],
    sim_params: SimLoopParams | None = None,
    *,
    base_speed_scale: float = 1.0,
) -> np.ndarray:
    """V2-B2.5：从 eef waypoint 构建 current-controller 可执行 actions（open-loop 预计算）。"""
    params = sim_params or SimLoopParams()
    actions = eef_waypoints_to_osc_actions(
        target_eef,
        gripper,
        env_args,
        speed_scale=base_speed_scale,
    )
    return apply_sim_loop_params(actions, proxy, gripper, env_args, params)


def compute_closed_loop_waypoint_action(
    env: Any,
    target_waypoint: np.ndarray,
    gripper_val: float,
    env_args: dict[str, Any],
    *,
    speed_scale: float = 1.0,
) -> np.ndarray:
    """当前 sim eef → target waypoint 的闭环 OSC action（单步）。"""
    from robosuite_env_loader import get_sim_eef_pose4

    speed_scale = float(max(speed_scale, 0.25))
    limits = osc_limits_from_env_args(env_args)
    current = get_sim_eef_pose4(env)
    dp = (target_waypoint[:3, 3] - current[:3, 3]) / speed_scale
    drot = target_waypoint[:3, :3] @ current[:3, :3].T
    dori = rotmat_to_axis_angle(drot) / speed_scale
    arm = encode_osc_delta(dp, dori, limits["output_max"], limits["output_min"])
    action = np.zeros(7, dtype=float)
    action[:6] = arm
    action[6] = float(np.clip(gripper_val, -1.0, 1.0))
    return action


def apply_sim_loop_step_overlay(
    action: np.ndarray,
    step: int,
    proxy: TrajectoryProxy,
    gripper: np.ndarray,
    env_args: dict[str, Any],
    sim_params: SimLoopParams,
) -> np.ndarray:
    """单步 sim-in-loop 参数叠加。"""
    out = action.copy()
    t_min = proxy.phases.t_min_xy
    length = proxy.length
    limits = osc_limits_from_env_args(env_args)
    z_scale = max(float(limits["output_max"][2]), 1e-6)

    pause_start = max(0, t_min - int(sim_params.pre_insert_pause))
    if pause_start <= step < t_min:
        out[:6] *= 0.05

    insert_steps = max(1, int(sim_params.insertion_steps))
    insert_end = min(length, t_min + insert_steps)
    if t_min <= step < insert_end:
        gamma = (step - t_min) / max(1, insert_end - t_min - 1) if insert_end > t_min else 1.0
        z_bias = (float(sim_params.insert_z_offset) * gamma) / z_scale * float(sim_params.z_gain)
        out[2] = np.clip(out[2] + z_bias, -1.0, 1.0)
        if float(sim_params.insertion_speed_scale) != 1.0:
            out[:6] /= float(sim_params.insertion_speed_scale)

    hold_end = min(length, insert_end + int(sim_params.hold_steps))
    if insert_end <= step < hold_end and insert_end > t_min:
        out[:6] *= 0.15

    if int(sim_params.release_shift) != 0:
        grip = _shift_1d_signal(gripper, float(sim_params.release_shift))
        out[6] = float(np.clip(grip[step], -1.0, 1.0))

    return out


def run_closed_loop_waypoint_rollout_steps(
    env: Any,
    proxy: TrajectoryProxy,
    target_eef: np.ndarray,
    gripper: np.ndarray,
    env_args: dict[str, Any],
    sim_params: SimLoopParams,
    *,
    base_speed_scale: float = 1.0,
) -> np.ndarray:
    """逐步闭环跟踪 eef waypoint 并执行 sim-in-loop 参数叠加。"""
    length = len(target_eef)
    actions = np.zeros((length, 7), dtype=float)
    grip = np.asarray(gripper, dtype=float).reshape(-1)
    for step in range(length):
        target_idx = min(step + 1, length - 1)
        action = compute_closed_loop_waypoint_action(
            env,
            target_eef[target_idx],
            grip[step],
            env_args,
            speed_scale=base_speed_scale,
        )
        actions[step] = apply_sim_loop_step_overlay(
            action, step, proxy, grip, env_args, sim_params
        )
        env.step(actions[step])
    return actions
