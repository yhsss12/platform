from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

FAILURE_TYPES = (
    "success",
    "grasp_failed",
    "lift_failed",
    "alignment_failed",
    "insertion_failed",
    "timeout",
    "out_of_bounds",
    "unknown_failure",
)


class RolloutPhase(IntEnum):
    APPROACH_NUT = 0
    DESCEND_TO_GRASP = 1
    CLOSE_GRIPPER = 2
    SETTLE_AFTER_GRASP = 3
    LIFT_NUT = 4
    MOVE_TO_PEG = 5
    ALIGN_OVER_PEG = 6
    DESCEND_INSERT = 7
    RELEASE = 8
    VERIFY_SUCCESS = 9

    # Legacy aliases (P2 names)
    REACH_NUT = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 4
    MOVE_PEG = 5
    ALIGN = 6
    INSERT = 7
    HOLD = 9


STAGE_MAX_STEPS: dict[RolloutPhase, int] = {
    RolloutPhase.APPROACH_NUT: 100,
    RolloutPhase.DESCEND_TO_GRASP: 80,
    RolloutPhase.CLOSE_GRIPPER: 35,
    RolloutPhase.SETTLE_AFTER_GRASP: 30,
    RolloutPhase.LIFT_NUT: 90,
    RolloutPhase.MOVE_TO_PEG: 140,
    RolloutPhase.ALIGN_OVER_PEG: 130,
    RolloutPhase.DESCEND_INSERT: 130,
    RolloutPhase.RELEASE: 35,
    RolloutPhase.VERIFY_SUCCESS: 40,
}


def _nut_body_pos(env, nut_name: str = "SquareNut") -> np.ndarray | None:
    body_id = getattr(env, "obj_body_id", {}).get(nut_name)
    if body_id is None:
        return None
    return np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)


def _nut_grasp_pos(env, nut_name: str = "SquareNut") -> np.ndarray | None:
    """Prefer handle site — body center is offset from grasp point on square nuts."""
    for nut in getattr(env, "nuts", []):
        if nut.name != nut_name:
            continue
        handle_key = getattr(nut, "important_sites", {}).get("handle")
        if handle_key:
            try:
                site_id = env.sim.model.site_name2id(handle_key)
                return np.array(env.sim.data.site_xpos[site_id], dtype=np.float64)
            except Exception:
                pass
    return _nut_body_pos(env, nut_name)


def _peg_body_pos(env, peg_id: int = 0) -> np.ndarray | None:
    try:
        peg_body_id = env.peg1_body_id if peg_id == 0 else env.peg2_body_id
        return np.array(env.sim.data.body_xpos[peg_body_id], dtype=np.float64)
    except Exception:
        return None


def _eef_pos(env) -> np.ndarray:
    robot = env.robots[0]
    site_id: int | None = None
    eef_site = getattr(robot, "eef_site_id", None)
    if isinstance(eef_site, dict):
        if hasattr(robot, "arms") and robot.arms:
            site_id = eef_site.get(robot.arms[0])
        if site_id is None and eef_site:
            site_id = next(iter(eef_site.values()))
    elif eef_site is not None:
        site_id = int(eef_site)
    if site_id is None:
        raise RuntimeError("unable to resolve end-effector site id")
    return np.array(env.sim.data.site_xpos[site_id], dtype=np.float64)


def _is_grasping(env, nut_name: str = "SquareNut") -> bool:
    if not hasattr(env, "_check_grasp"):
        return False
    nut = next((n for n in env.nuts if n.name == nut_name), None)
    if nut is None:
        return False
    try:
        return bool(
            env._check_grasp(
                gripper=env.robots[0].gripper,
                object_geoms=list(nut.contact_geoms),
            )
        )
    except Exception:
        return False


def _nut_tilt_error(env, nut_name: str = "SquareNut") -> float | None:
    body_id = getattr(env, "obj_body_id", {}).get(nut_name)
    if body_id is None:
        return None
    z_axis = np.array(env.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)[:, 2]
    return float(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))


def compute_pose_errors(env, *, nut_name: str = "SquareNut") -> dict[str, float | None]:
    nut = _nut_grasp_pos(env, nut_name)
    nut_body = _nut_body_pos(env, nut_name)
    eef = _eef_pos(env)
    peg = _peg_body_pos(env, 0)
    if nut is None or peg is None:
        return {
            "xy_error": None,
            "z_error": None,
            "height_error": None,
            "tilt_error": None,
            "final_xy_error": None,
            "final_height_error": None,
        }
    ref = nut_body if nut_body is not None else nut
    xy_error = float(np.linalg.norm(eef[:2] - nut[:2]))
    z_error = float(eef[2] - nut[2])
    height_error = float(ref[2] - peg[2])
    final_xy_error = float(np.linalg.norm(ref[:2] - peg[:2]))
    final_height_error = float(ref[2] - peg[2])
    return {
        "xy_error": xy_error,
        "z_error": z_error,
        "height_error": height_error,
        "tilt_error": _nut_tilt_error(env, nut_name),
        "final_xy_error": final_xy_error,
        "final_height_error": final_height_error,
    }


def check_env_success(env) -> bool:
    if hasattr(env, "_check_success"):
        try:
            return bool(env._check_success())
        except Exception:
            pass
    return False


def check_success_pose_fallback(env, *, nut_name: str = "SquareNut", peg_id: int = 0) -> bool:
    nut_pos = _nut_body_pos(env, nut_name)
    peg_pos = _peg_body_pos(env, peg_id)
    if nut_pos is None or peg_pos is None:
        return False
    if hasattr(env, "on_peg"):
        try:
            return bool(env.on_peg(nut_pos, peg_id))
        except Exception:
            pass
    xy_close = np.linalg.norm(nut_pos[:2] - peg_pos[:2]) < 0.04
    z_ok = nut_pos[2] < peg_pos[2] + 0.08
    in_bounds = nut_pos[2] > 0.75
    return bool(xy_close and z_ok and in_bounds)


def check_episode_success(env, *, nut_name: str = "SquareNut", peg_id: int = 0) -> tuple[bool, str]:
    if check_env_success(env):
        return True, "env_check_success"
    nut_body = _nut_body_pos(env, nut_name)
    if nut_body is not None and hasattr(env, "on_peg"):
        try:
            if env.on_peg(nut_body, peg_id):
                return True, "on_peg"
        except Exception:
            pass
    if check_success_pose_fallback(env, nut_name=nut_name, peg_id=peg_id):
        return True, "pose_fallback"
    return False, "none"


def verify_success_stable(env, *, steps: int = 5) -> bool:
    if not check_env_success(env) and not check_success_pose_fallback(env):
        return False
    for _ in range(max(steps - 1, 0)):
        low, high = env.action_spec
        action = np.zeros_like((low + high) / 2.0)
        env.step(action)
        if not check_env_success(env) and not check_success_pose_fallback(env):
            return False
    return True


def is_out_of_bounds(env, *, nut_name: str = "SquareNut", table_z: float = 0.775) -> bool:
    nut_pos = _nut_body_pos(env, nut_name)
    eef = _eef_pos(env)
    if nut_pos is not None:
        if nut_pos[2] < table_z - 0.05 or nut_pos[2] > 1.6:
            return True
        if np.linalg.norm(nut_pos[:2]) > 0.8:
            return True
    if eef[2] < table_z - 0.08 or eef[2] > 1.8:
        return True
    if np.linalg.norm(eef[:2]) > 1.0:
        return True
    return False


def classify_failure_type(
    *,
    success: bool,
    max_phase: RolloutPhase,
    grasp_success: bool,
    lift_success: bool,
    alignment_success: bool,
    insertion_attempted: bool,
    timed_out: bool,
    out_of_bounds: bool,
) -> str:
    if success:
        return "success"
    if out_of_bounds:
        return "out_of_bounds"
    if timed_out:
        if not grasp_success:
            return "grasp_failed"
        if not lift_success:
            return "lift_failed"
        if not alignment_success:
            return "alignment_failed"
        if insertion_attempted:
            return "insertion_failed"
        return "timeout"
    if not grasp_success:
        return "grasp_failed"
    if not lift_success:
        return "lift_failed"
    if not alignment_success:
        return "alignment_failed"
    if max_phase >= RolloutPhase.DESCEND_INSERT:
        return "insertion_failed"
    return "unknown_failure"


def build_episode_metadata(
    *,
    success_flag: bool,
    failure_type: str,
    episode_steps: int,
    env_name: str,
    generation_mode: str,
    policy_mode: str,
    seed: int,
    episode_index: int,
    grasp_attempts: int = 1,
    valid_for_training: bool | None = None,
    final_xy_error: float | None = None,
    final_height_error: float | None = None,
    grasp_success: bool = False,
    lift_success: bool = False,
    alignment_success: bool = False,
    insertion_success: bool = False,
    max_phase: str | None = None,
) -> dict[str, Any]:
    if valid_for_training is None:
        valid_for_training = bool(success_flag)
    return {
        "success_flag": bool(success_flag),
        "failure_type": failure_type,
        "valid_for_training": bool(valid_for_training),
        "episode_steps": int(episode_steps),
        "env_name": env_name,
        "generation_mode": generation_mode,
        "policy_mode": policy_mode,
        "policyMode": policy_mode,
        "generationMode": generation_mode,
        "seed": int(seed),
        "episode_index": int(episode_index),
        "grasp_attempts": int(grasp_attempts),
        "grasp_success": bool(grasp_success),
        "lift_success": bool(lift_success),
        "alignment_success": bool(alignment_success),
        "insertion_success": bool(insertion_success),
        "final_xy_error": final_xy_error,
        "final_height_error": final_height_error,
        "max_phase": max_phase,
    }


def build_stage_debug_record(
    *,
    episode: int,
    stage: str,
    nut_pos: list[float] | None,
    peg_pos: list[float] | None,
    eef_pos: list[float] | None,
    xy_error: float | None,
    height_error: float | None,
    tilt_error: float | None,
    gripper_state: str,
    stage_success: bool,
    grasp_attempts: int,
    failure_type: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "episode": episode,
        "stage": stage,
        "nut_pos": nut_pos,
        "peg_pos": peg_pos,
        "eef_pos": eef_pos,
        "xy_error": xy_error,
        "height_error": height_error,
        "tilt_error": tilt_error,
        "gripper_state": gripper_state,
        "stage_success": stage_success,
        "grasp_attempts": grasp_attempts,
    }
    if failure_type is not None:
        record["failure_type"] = failure_type
    if extra:
        record.update(extra)
    return record
