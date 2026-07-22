from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.episode_eval import (
    STAGE_MAX_STEPS,
    RolloutPhase,
    _eef_pos,
    _is_grasping,
    _nut_body_pos,
    _nut_grasp_pos,
    _peg_body_pos,
)

POS_SCALE_APPROACH = 5.0
POS_SCALE_DESCEND = 4.0
POS_SCALE_FINE = 3.0
POS_SCALE_INSERT = 3.5
Z_ABOVE = 0.10
Z_GRASP_OFFSET = 0.006
Z_LIFT_TARGET = 0.12
Z_ABOVE_PEG = 0.12
Z_INSERT = 0.004
XY_APPROACH_TOL = 0.015
XY_DESCEND_TOL = 0.012
XY_PEG_TOL = 0.022
GRASP_SETTLE_STEPS = 8
LIFT_DELTA_MIN = 0.035
MAX_REGRASP = 2


@dataclass
class ScriptedPolicyState:
    phase: RolloutPhase = RolloutPhase.APPROACH_NUT
    max_phase: RolloutPhase = RolloutPhase.APPROACH_NUT
    phase_step: int = 0
    grasp_attempts: int = 1
    regrasp_used: bool = False

    grasp_success: bool = False
    lift_success: bool = False
    alignment_success: bool = False
    insertion_success: bool = False

    grasp_stable_count: int = 0
    nut_z_at_grasp: float | None = None
    nut_z_at_lift_start: float | None = None
    stage_failure: str | None = None
    done: bool = False

    nut_name: str = "SquareNut"
    peg_id: int = 0
    extra_xy_bias: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.extra_xy_bias is None:
            self.extra_xy_bias = np.zeros(2, dtype=np.float64)

    def advance(self, new_phase: RolloutPhase) -> None:
        if new_phase > self.max_phase:
            self.max_phase = new_phase
        if new_phase != self.phase:
            self.phase = new_phase
            self.phase_step = 0

    def tick(self) -> None:
        self.phase_step += 1

    def stage_timed_out(self) -> bool:
        limit = STAGE_MAX_STEPS.get(self.phase, 100)
        return self.phase_step >= limit

    def trigger_regrasp(self) -> bool:
        if self.regrasp_used or self.grasp_attempts > MAX_REGRASP:
            return False
        self.regrasp_used = True
        self.grasp_attempts += 1
        self.grasp_success = False
        self.lift_success = False
        self.alignment_success = False
        self.insertion_success = False
        self.grasp_stable_count = 0
        self.nut_z_at_grasp = None
        self.nut_z_at_lift_start = None
        self.stage_failure = None
        self.done = False
        self.advance(RolloutPhase.APPROACH_NUT)
        return True


def _delta_action(env, target_pos: np.ndarray, gripper: float, *, scale: float = POS_SCALE_APPROACH) -> np.ndarray:
    from robosuite.utils.control_utils import orientation_error

    low, high = env.action_spec
    eef = _eef_pos(env)
    delta = target_pos - eef
    dim = int(getattr(env, "action_dim", len(low)))
    action = np.zeros(dim, dtype=np.float32)
    pos_dim = min(3, dim)
    action[:pos_dim] = np.clip(delta[:pos_dim] * scale, low[:pos_dim], high[:pos_dim])
    # Keep the Panda fingers parallel to the nut handle. Nut placement samples
    # an arbitrary yaw on every reset, while a zero rotational action preserves
    # the robot's initial yaw and only happens to grasp a small subset of poses.
    # The desired frame keeps the tool pointing down and aligns its x axis with
    # the handle's long axis (the fingers close across the local y axis).
    # Once closed, hold the achieved wrist orientation. Continuously following
    # the carried nut's measured yaw creates a feedback loop through contact
    # forces and can make the EEF climb while insertion commands descend.
    if dim >= 7 and gripper < 0:
        body_id = getattr(env, "obj_body_id", {}).get("SquareNut")
        if body_id is not None:
            nut_ori = np.asarray(env.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
            handle_axis = nut_ori[:2, 0]
            handle_norm = float(np.linalg.norm(handle_axis))
            if handle_norm > 1e-8:
                handle_axis = handle_axis / handle_norm
            else:
                handle_axis = np.array([1.0, 0.0])
            desired_x = np.array([handle_axis[0], handle_axis[1], 0.0])
            desired_y = np.array([handle_axis[1], -handle_axis[0], 0.0])
            desired_ori = np.column_stack((desired_x, desired_y, np.array([0.0, 0.0, -1.0])))
            robot = env.robots[0]
            site_ids = getattr(robot, "eef_site_id", None)
            if isinstance(site_ids, dict):
                site_id = next(iter(site_ids.values()))
            else:
                site_id = int(site_ids)
            current_ori = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
            ori_delta = orientation_error(desired_ori, current_ori)
            action[3:6] = np.clip(ori_delta * 2.0, low[3:6], high[3:6])
    action[-1] = float(gripper)
    return action


def _nut_body(env, state: ScriptedPolicyState, fallback: np.ndarray) -> np.ndarray:
    raw = _nut_body_pos(env, state.nut_name)
    return raw if raw is not None else fallback


def _on_peg(env, nut_body: np.ndarray, peg_id: int) -> bool:
    if not hasattr(env, "on_peg"):
        return False
    try:
        return bool(env.on_peg(nut_body, peg_id))
    except Exception:
        return False


def _update_grasp_lift_flags(env, state: ScriptedPolicyState, nut: np.ndarray, peg: np.ndarray) -> None:
    grasping = _is_grasping(env, state.nut_name)
    nut_body = _nut_body(env, state, nut)
    if grasping:
        state.grasp_stable_count += 1
        if state.nut_z_at_grasp is None:
            state.nut_z_at_grasp = float(nut_body[2])
    elif state.phase < RolloutPhase.LIFT_NUT:
        state.grasp_stable_count = max(0, state.grasp_stable_count - 1)

    if state.grasp_stable_count >= GRASP_SETTLE_STEPS:
        state.grasp_success = True

    if state.phase >= RolloutPhase.LIFT_NUT and state.nut_z_at_lift_start is not None:
        lifted = nut_body[2] - state.nut_z_at_lift_start
        if lifted >= LIFT_DELTA_MIN and (grasping or lifted >= LIFT_DELTA_MIN * 0.8):
            state.lift_success = True

    xy_to_peg = float(np.linalg.norm(nut_body[:2] - peg[:2]))
    if state.lift_success and xy_to_peg < XY_PEG_TOL:
        state.alignment_success = True
    if state.alignment_success and xy_to_peg < 0.03 and nut_body[2] < peg[2] + 0.05:
        state.insertion_success = True
    if _on_peg(env, nut_body, state.peg_id):
        state.insertion_success = True


def compute_scripted_action(env, state: ScriptedPolicyState) -> np.ndarray:
    if state.done:
        low, high = env.action_spec
        return ((low + high) / 2.0).astype(np.float32)

    state.tick()

    nut = _nut_grasp_pos(env, state.nut_name)
    peg = _peg_body_pos(env, state.peg_id)
    eef = _eef_pos(env)

    if nut is None or peg is None:
        low, high = env.action_spec
        return ((low + high) / 2.0).astype(np.float32)

    nut_body = _nut_body(env, state, nut)
    _update_grasp_lift_flags(env, state, nut, peg)

    gripper_open = -1.0
    gripper_close = 1.0

    if state.phase == RolloutPhase.APPROACH_NUT:
        target = nut.copy()
        target[2] += Z_ABOVE
        action = _delta_action(env, target, gripper_open, scale=POS_SCALE_APPROACH)
        if np.linalg.norm(eef[:2] - target[:2]) < XY_APPROACH_TOL and abs(eef[2] - target[2]) < 0.03:
            state.advance(RolloutPhase.DESCEND_TO_GRASP)
        elif state.stage_timed_out():
            state.advance(RolloutPhase.DESCEND_TO_GRASP)
        return action

    if state.phase == RolloutPhase.DESCEND_TO_GRASP:
        target = nut.copy()
        target[2] += Z_GRASP_OFFSET
        action = _delta_action(env, target, gripper_open, scale=POS_SCALE_DESCEND)
        if np.linalg.norm(eef[:2] - target[:2]) < XY_DESCEND_TOL and abs(eef[2] - target[2]) < 0.008:
            state.advance(RolloutPhase.CLOSE_GRIPPER)
        elif state.stage_timed_out():
            if state.trigger_regrasp():
                return _delta_action(env, nut + np.array([0, 0, Z_ABOVE]), gripper_open)
            state.advance(RolloutPhase.CLOSE_GRIPPER)
        return action

    if state.phase == RolloutPhase.CLOSE_GRIPPER:
        target = nut.copy()
        target[2] = nut[2] + Z_GRASP_OFFSET
        action = _delta_action(env, target, gripper_close, scale=POS_SCALE_FINE)
        if _is_grasping(env, state.nut_name):
            # Nut handles are shallow (20 mm tall). Holding the end effector
            # stationary after first contact lets the handle slip back onto the
            # table before the old multi-step stability gate can complete.
            # Start lifting immediately while keeping the gripper closed; the
            # lift phase still verifies that the nut actually follows the EEF.
            state.grasp_success = True
            state.grasp_stable_count = 1
            state.nut_z_at_grasp = float(nut_body[2])
            state.nut_z_at_lift_start = float(nut_body[2])
            state.advance(RolloutPhase.LIFT_NUT)
        elif state.stage_timed_out():
            if state.trigger_regrasp():
                return _delta_action(env, nut + np.array([0, 0, Z_ABOVE]), gripper_open)
            state.advance(RolloutPhase.SETTLE_AFTER_GRASP)
        return action

    if state.phase == RolloutPhase.SETTLE_AFTER_GRASP:
        target = eef.copy()
        action = _delta_action(env, target, gripper_close, scale=POS_SCALE_FINE)
        if state.grasp_success:
            state.nut_z_at_lift_start = float(nut_body[2])
            state.advance(RolloutPhase.LIFT_NUT)
        elif state.stage_timed_out():
            if state.trigger_regrasp():
                return _delta_action(env, nut + np.array([0, 0, Z_ABOVE]), gripper_open)
            if _is_grasping(env, state.nut_name):
                state.grasp_success = True
                state.nut_z_at_lift_start = float(nut_body[2])
                state.advance(RolloutPhase.LIFT_NUT)
            else:
                state.stage_failure = "grasp_not_stable"
                state.done = True
        return action

    if state.phase == RolloutPhase.LIFT_NUT:
        target = eef.copy()
        target[2] = max(eef[2], (state.nut_z_at_lift_start or nut_body[2]) + Z_LIFT_TARGET)
        action = _delta_action(env, target, gripper_close, scale=POS_SCALE_APPROACH)
        if state.lift_success:
            state.advance(RolloutPhase.MOVE_TO_PEG)
        elif state.stage_timed_out():
            if state.trigger_regrasp():
                return _delta_action(env, nut + np.array([0, 0, Z_ABOVE]), gripper_open)
            state.stage_failure = "lift_timeout"
            state.done = True
        return action

    if state.phase == RolloutPhase.MOVE_TO_PEG:
        target = peg.copy()
        target[2] = max(eef[2], peg[2] + Z_ABOVE_PEG)
        action = _delta_action(env, target, gripper_close, scale=POS_SCALE_APPROACH)
        if np.linalg.norm(eef[:2] - peg[:2]) < 0.028 or state.phase_step > 95:
            state.advance(RolloutPhase.ALIGN_OVER_PEG)
        return action

    if state.phase == RolloutPhase.ALIGN_OVER_PEG:
        bias = state.extra_xy_bias if state.extra_xy_bias is not None else np.zeros(2)
        xy_err = float(np.linalg.norm(nut_body[:2] - peg[:2]))
        target = eef.copy()
        target[:2] += peg[:2] - nut_body[:2] + bias
        target[2] = max(eef[2], peg[2] + Z_ABOVE_PEG * 0.85)
        action = _delta_action(env, target, gripper_close, scale=POS_SCALE_DESCEND)
        if xy_err < XY_PEG_TOL:
            state.alignment_success = True
            state.advance(RolloutPhase.DESCEND_INSERT)
        elif state.stage_timed_out():
            state.advance(RolloutPhase.DESCEND_INSERT)
        return action

    if state.phase == RolloutPhase.DESCEND_INSERT:
        bias = state.extra_xy_bias if state.extra_xy_bias is not None else np.zeros(2)
        target = eef.copy()
        target[:2] += peg[:2] - nut_body[:2] + bias
        target[2] = max(peg[2] + Z_INSERT, eef[2] - 0.004)
        action = _delta_action(env, target, gripper_close, scale=POS_SCALE_INSERT)
        xy_err = float(np.linalg.norm(nut_body[:2] - peg[:2]))
        if _on_peg(env, nut_body, state.peg_id) or (xy_err < 0.028 and nut_body[2] < peg[2] + 0.038):
            state.insertion_success = True
            state.advance(RolloutPhase.RELEASE)
        elif state.stage_timed_out():
            state.advance(RolloutPhase.RELEASE)
        return action

    if state.phase == RolloutPhase.RELEASE:
        nut_body = _nut_body(env, state, nut)
        target = eef.copy()
        action = _delta_action(env, target, gripper_open, scale=POS_SCALE_FINE)
        xy_err = float(np.linalg.norm(nut_body[:2] - peg[:2]))
        if state.phase_step > 8 and (
            _on_peg(env, nut_body, state.peg_id) or (xy_err < 0.032 and nut_body[2] < peg[2] + 0.045)
        ):
            state.insertion_success = True
            state.advance(RolloutPhase.VERIFY_SUCCESS)
        elif state.phase_step > 28:
            state.advance(RolloutPhase.VERIFY_SUCCESS)
        return action

    target = eef.copy()
    if state.phase_step > 18:
        state.done = True
    return _delta_action(env, target, gripper_open, scale=POS_SCALE_FINE)


def policy_mode_label(state: ScriptedPolicyState, *, used_scripted: bool) -> str:
    if not used_scripted:
        return "random_rollout"
    return "partial_scripted"
