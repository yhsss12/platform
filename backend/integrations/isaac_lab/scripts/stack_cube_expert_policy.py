#!/usr/bin/env python3
# Copyright (c) 2026 EAI Platform
"""Stack Cube expert policy — 自动物块堆叠并录制 HDF5（Isaac subprocess 内运行）。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI（AppLauncher 之前解析）
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Stack Cube expert policy demo recorder.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Stack-Cube-Franka-IK-Rel-v0",
    help="Isaac Lab task id.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    required=True,
    help="Output HDF5 path (e.g. artifacts/dataset.hdf5).",
)
parser.add_argument("--num_demos", type=int, default=1, help="Number of successful demos to record.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for env reset.")
parser.add_argument(
    "--max_attempts",
    type=int,
    default=0,
    help="Max episode attempts (0 = num_demos * 5).",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive success steps before exporting a demo.",
)
parser.add_argument(
    "--enable_action_smoothing",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Low-pass filter arm commands and limit per-step delta.",
)
parser.add_argument(
    "--action_smoothing_alpha",
    type=float,
    default=0.6,
    help="EMA weight for raw arm command (higher = snappier).",
)
parser.add_argument(
    "--max_action_delta",
    type=float,
    default=0.08,
    help="Max per-step change for smoothed arm command dims.",
)
parser.add_argument(
    "--enable_quality_checks",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable grasp / place sanity checks (does not override env success_term).",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=10,
    help="Steps to wait after place before advancing to next cube.",
)
parser.add_argument(
    "--max_export_arm_delta",
    type=float,
    default=0.55,
    help="Reject demo export when max per-step arm action delta exceeds this threshold.",
)
parser.add_argument(
    "--record_camera_obs",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record RGB camera observations into HDF5 obs/ (requires --enable_cameras).",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default=128,
    help="Square RGB frame resolution (H=W) written to HDF5 image obs.",
)
parser.add_argument(
    "--include_wrist_camera",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Also record robot0_eye_in_hand_image when a wrist camera is available.",
)
parser.add_argument(
    "--live_frame_dir",
    type=str,
    default="",
    help="Directory for live/latest.jpg preview frames during generation.",
)
parser.add_argument(
    "--live_status_out",
    type=str,
    default="",
    help="JSON path for live preview status (frame_count, phase, progress).",
)
parser.add_argument(
    "--live_frame_every",
    type=int,
    default=5,
    help="Write a live preview frame every N simulation steps.",
)

try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    print(
        "ERROR: isaaclab is not available. Install Isaac Sim and run ./isaaclab.sh --install.\n"
        f"Detail: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_demos < 1:
    parser.error("--num_demos must be >= 1")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Isaac imports（仅 subprocess 内）
# ---------------------------------------------------------------------------

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, SceneEntityCfg

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

from integrations.isaac_lab.scripts.stack_cube_expert_policy_quality import (
    ExpertQualityTracker,
    interpolate_pose,
    min_jerk_progress,
    smooth_ik_rel_action,
)
from integrations.isaac_lab.scripts.stack_cube_expert_policy_behavior import (
    ExpertBehaviorTracker,
    GRASP_LIFT_MIN_Z,
    HEIGHT_DIFF as BEHAVIOR_HEIGHT_DIFF,
    PLACE_XY_TOLERANCE,
    PLACE_Z_TOLERANCE,
    compute_stack_error,
    summarize_behavior_status,
)
from integrations.isaac_lab.camera_capture import (
    capture_viewport_rgb,
    configure_single_env_live_viewer,
    warmup_viewport_capture,
)
from integrations.isaac_lab.hdf5_image_obs import (
    default_camera_keys,
    inject_camera_observations,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
HEIGHT_DIFF = BEHAVIOR_HEIGHT_DIFF
CUBE_HEIGHT = 0.0203
APPROACH_Z_OFFSET = 0.10
ALIGN_Z_OFFSET = 0.06
LIFT_Z_OFFSET = 0.10
GRASP_HEIGHT_OFFSET = 0.012
PLACE_HEIGHT_OFFSET = 0.012
POS_THRESHOLD = 0.015
GRASP_XY_THRESHOLD = 0.012
MAX_POS_DELTA = 0.05
MAX_ROT_DELTA = 0.06
ORIENTATION_GAIN = 0.30
NEUTRAL_Z_OFFSET = 0.18
ACTION_SCALE = 0.5
NUM_ENVS = 1

NEAR_STREAK_APPROACH = 10
NEAR_STREAK_DESCEND = 8
NEAR_STREAK_GRIPPER = 22
NEAR_STREAK_LIFT = 12
NEAR_STREAK_RETREAT_TO_SAFE = 30

GRASP_LIFT_CHECK_START = 8
GRASP_LIFT_CHECK_END = 40
GRASP_XY_MAX = GRASP_XY_THRESHOLD
PLACE_SETTLE_STEPS = 15


class ExpertPolicyState(IntEnum):
    INIT_NEUTRAL = 0
    MOVE_TO_PRE_GRASP = 1
    ALIGN_GRASP = 2
    DESCEND_TO_GRASP = 3
    CLOSE_GRIPPER = 4
    VERIFY_GRASP = 5
    LIFT_OBJECT = 6
    MOVE_TO_PRE_PLACE = 7
    DESCEND_TO_PLACE = 8
    OPEN_GRIPPER = 9
    VERIFY_PLACE = 10
    RETREAT_TO_SAFE = 11
    NEUTRAL_TRANSITION = 12
    CHECK_TASK_SUCCESS = 13
    FAIL = 14
    DONE = 15


STATE_DURATION_STEPS: dict[ExpertPolicyState, int] = {
    ExpertPolicyState.INIT_NEUTRAL: 35,
    ExpertPolicyState.MOVE_TO_PRE_GRASP: 55,
    ExpertPolicyState.ALIGN_GRASP: 25,
    ExpertPolicyState.DESCEND_TO_GRASP: 45,
    ExpertPolicyState.CLOSE_GRIPPER: 25,
    ExpertPolicyState.VERIFY_GRASP: 10,
    ExpertPolicyState.LIFT_OBJECT: 45,
    ExpertPolicyState.MOVE_TO_PRE_PLACE: 65,
    ExpertPolicyState.DESCEND_TO_PLACE: 45,
    ExpertPolicyState.OPEN_GRIPPER: 22,
    ExpertPolicyState.VERIFY_PLACE: 20,
    ExpertPolicyState.RETREAT_TO_SAFE: 40,
    ExpertPolicyState.NEUTRAL_TRANSITION: 50,
    ExpertPolicyState.CHECK_TASK_SUCCESS: 600,
}


STATE_LABELS = {s.value: s.name for s in ExpertPolicyState}

LIVE_TRACKING_STATES = frozenset({
    ExpertPolicyState.MOVE_TO_PRE_GRASP,
    ExpertPolicyState.DESCEND_TO_GRASP,
    ExpertPolicyState.MOVE_TO_PRE_PLACE,
    ExpertPolicyState.DESCEND_TO_PLACE,
})

INTERPOLATED_MOTION_STATES = frozenset({
    ExpertPolicyState.INIT_NEUTRAL,
    ExpertPolicyState.LIFT_OBJECT,
    ExpertPolicyState.RETREAT_TO_SAFE,
    ExpertPolicyState.NEUTRAL_TRANSITION,
})


@dataclass
class CubeTarget:
    object_name: str
    place_pos: torch.Tensor
    place_quat: torch.Tensor


@dataclass
class StateContext:
    state: ExpertPolicyState = ExpertPolicyState.INIT_NEUTRAL
    cube_index: int = 0
    step_in_state: int = 0
    state_duration_steps: int = 40
    near_target_streak: int = 0
    gripper_command: float = GRIPPER_OPEN
    target_pos: Optional[torch.Tensor] = None
    target_quat: Optional[torch.Tensor] = None
    state_start_pos: Optional[torch.Tensor] = None
    state_target_pos: Optional[torch.Tensor] = None
    state_target_quat: Optional[torch.Tensor] = None
    transition_progress: float = 0.0
    stack_base_xy: Optional[torch.Tensor] = None
    targets: list[CubeTarget] = field(default_factory=list)
    failure_reason: Optional[str] = None
    failed_from_state: Optional[ExpertPolicyState] = None
    grasp_object_z: Optional[float] = None
    grasp_object_xy: Optional[torch.Tensor] = None
    grasp_ee_z: Optional[float] = None
    grasp_ee_xy: Optional[torch.Tensor] = None
    lift_reference_z: Optional[float] = None
    place_reference_z: Optional[float] = None
    place_check_warning: Optional[str] = None
    cube_lifted_flags: list[bool] = field(default_factory=list)
    cube_placed_flags: list[bool] = field(default_factory=list)
    micro_lift: bool = False
    total_episode_steps: int = 0


class StackCubeExpertPolicy:
    """物块堆叠专家策略：state-machine expert policy with IK-Rel control."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        pos_threshold: float = POS_THRESHOLD,
        state_timeout_steps: int = 240,
        enable_action_smoothing: bool = True,
        action_smoothing_alpha: float = 0.6,
        max_action_delta: float = 0.08,
        enable_quality_checks: bool = True,
        settle_steps: int = 10,
    ) -> None:
        self.device = device
        self.pos_threshold = pos_threshold
        self.state_timeout_steps = state_timeout_steps
        self.enable_action_smoothing = enable_action_smoothing
        self.action_smoothing_alpha = action_smoothing_alpha
        self.max_action_delta = max_action_delta
        self.enable_quality_checks = enable_quality_checks
        self.settle_steps = max(1, int(settle_steps))
        self.ctx = StateContext()
        self.down_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.previous_action: Optional[torch.Tensor] = None

    def reset_episode(self, env) -> None:
        self.ctx = StateContext()
        self.previous_action = torch.zeros(7, device=self.device)
        self._init_targets(env)
        _, ee_quat = self._ee_pose(env)
        self.down_quat = ee_quat.clone()
        self._enter_state(ExpertPolicyState.MOVE_TO_PRE_GRASP, env)

    def _grasp_target_pos(self, obj_pos: torch.Tensor, *, phase: str) -> torch.Tensor:
        target = obj_pos.clone()
        if phase == "pre_grasp":
            target[2] = obj_pos[2] + APPROACH_Z_OFFSET
        elif phase == "align":
            target[2] = obj_pos[2] + ALIGN_Z_OFFSET
        else:
            target[2] = obj_pos[2] + GRASP_HEIGHT_OFFSET
        return target

    def _refresh_live_tracking_targets(self, env) -> None:
        """Approach/place phases track live geometry; other states keep frozen waypoints."""
        state = self.ctx.state
        if state == ExpertPolicyState.MOVE_TO_PRE_GRASP:
            obj_pos = self._object_pos(env, self._current_object_name())
            self._sync_motion_target(self._grasp_target_pos(obj_pos, phase="pre_grasp"))
        elif state == ExpertPolicyState.DESCEND_TO_GRASP:
            obj_pos = self._object_pos(env, self._current_object_name())
            self._sync_motion_target(self._grasp_target_pos(obj_pos, phase="grasp"))
        elif state == ExpertPolicyState.MOVE_TO_PRE_PLACE:
            place = self._stack_place_pos(env, self.ctx.cube_index)
            target = place.clone()
            target[2] = place[2] + APPROACH_Z_OFFSET
            self._sync_motion_target(target)
        elif state == ExpertPolicyState.DESCEND_TO_PLACE:
            place = self._stack_place_pos(env, self.ctx.cube_index)
            self._sync_motion_target(self._place_target_pos(place))

    def _place_target_pos(self, place: torch.Tensor) -> torch.Tensor:
        target = place.clone()
        target[2] = place[2] + PLACE_HEIGHT_OFFSET
        return target

    def _neutral_pose(self) -> torch.Tensor:
        if self.ctx.stack_base_xy is None:
            base_xy = torch.zeros(2, device=self.device)
        else:
            base_xy = self.ctx.stack_base_xy.clone()
        table_z = float(self.ctx.targets[0].place_pos[2].item()) if self.ctx.targets else CUBE_HEIGHT
        return torch.tensor([base_xy[0], base_xy[1], table_z + NEUTRAL_Z_OFFSET], device=self.device)

    def _waypoint_for_state(
        self,
        state: ExpertPolicyState,
        env,
        ee_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        gripper = GRIPPER_OPEN

        if state == ExpertPolicyState.INIT_NEUTRAL:
            target = self._neutral_pose()
        elif state == ExpertPolicyState.ALIGN_GRASP:
            if self.ctx.state_target_pos is not None:
                target = self.ctx.state_target_pos.clone()
            else:
                obj_pos = self._object_pos(env, self._current_object_name())
                target = self._grasp_target_pos(obj_pos, phase="align")
        elif state == ExpertPolicyState.VERIFY_GRASP:
            target = ee_pos.clone()
            gripper = GRIPPER_CLOSE
        elif state == ExpertPolicyState.MOVE_TO_PRE_GRASP:
            obj_pos = self._object_pos(env, self._current_object_name())
            target = self._grasp_target_pos(obj_pos, phase="pre_grasp")
        elif state == ExpertPolicyState.DESCEND_TO_GRASP:
            obj_pos = self._object_pos(env, self._current_object_name())
            target = self._grasp_target_pos(obj_pos, phase="grasp")
        elif state == ExpertPolicyState.CLOSE_GRIPPER:
            target = ee_pos.clone()
            gripper = GRIPPER_CLOSE
        elif state == ExpertPolicyState.LIFT_OBJECT:
            ref_z = self.ctx.lift_reference_z
            if ref_z is None:
                ref_z = float(ee_pos[2].item())
            target = ee_pos.clone()
            target[2] = ref_z + LIFT_Z_OFFSET
            gripper = GRIPPER_CLOSE
        elif state == ExpertPolicyState.MOVE_TO_PRE_PLACE:
            place = self._stack_place_pos(env, self.ctx.cube_index)
            target = place.clone()
            target[2] = place[2] + APPROACH_Z_OFFSET
            gripper = GRIPPER_CLOSE
        elif state == ExpertPolicyState.DESCEND_TO_PLACE:
            place = self._stack_place_pos(env, self.ctx.cube_index)
            target = self._place_target_pos(place)
            gripper = GRIPPER_CLOSE
        elif state == ExpertPolicyState.OPEN_GRIPPER:
            target = ee_pos.clone()
            gripper = GRIPPER_OPEN
        elif state == ExpertPolicyState.RETREAT_TO_SAFE:
            if self.ctx.place_reference_z is None:
                place = self._stack_place_pos(env, self.ctx.cube_index)
                self.ctx.place_reference_z = float(place[2].item() + PLACE_HEIGHT_OFFSET)
            target = ee_pos.clone()
            target[2] = self.ctx.place_reference_z + LIFT_Z_OFFSET
            gripper = GRIPPER_OPEN
        elif state == ExpertPolicyState.VERIFY_PLACE:
            target = ee_pos.clone()
            gripper = GRIPPER_OPEN
        elif state == ExpertPolicyState.NEUTRAL_TRANSITION:
            target = self._neutral_pose()
            gripper = GRIPPER_OPEN
        elif state == ExpertPolicyState.CHECK_TASK_SUCCESS:
            target = ee_pos.clone()
            gripper = GRIPPER_OPEN
        else:
            target = ee_pos.clone()
        return target, gripper

    def _enter_state(self, new_state: ExpertPolicyState, env) -> None:
        ee_pos, _ = self._ee_pose(env)
        self.ctx.state = new_state
        self.ctx.step_in_state = 0
        self.ctx.near_target_streak = 0
        self.ctx.state_start_pos = ee_pos.clone()
        self.ctx.transition_progress = 0.0
        self.ctx.state_duration_steps = STATE_DURATION_STEPS.get(new_state, 60)

        if new_state == ExpertPolicyState.LIFT_OBJECT:
            self.ctx.lift_reference_z = (
                float(self.ctx.state_target_pos[2].item())
                if self.ctx.state_target_pos is not None
                else float(ee_pos[2].item())
            )
            lift = ee_pos.clone()
            lift[2] = self.ctx.lift_reference_z + LIFT_Z_OFFSET
            self._sync_motion_target(lift)
        elif new_state == ExpertPolicyState.MOVE_TO_PRE_PLACE:
            self.ctx.lift_reference_z = float(ee_pos[2].item())
            place = self._stack_place_pos(env, self.ctx.cube_index)
            target = place.clone()
            target[2] = place[2] + APPROACH_Z_OFFSET
            self._sync_motion_target(target)
        elif new_state == ExpertPolicyState.DESCEND_TO_PLACE:
            place = self._stack_place_pos(env, self.ctx.cube_index)
            self._sync_motion_target(self._place_target_pos(place))
        elif new_state == ExpertPolicyState.RETREAT_TO_SAFE:
            place = self._stack_place_pos(env, self.ctx.cube_index)
            self.ctx.place_reference_z = float(place[2].item() + PLACE_HEIGHT_OFFSET)
        elif new_state == ExpertPolicyState.MOVE_TO_PRE_GRASP:
            self.ctx.lift_reference_z = None
            self.ctx.place_reference_z = None
            obj_pos = self._object_pos(env, self._current_object_name())
            self._sync_motion_target(self._grasp_target_pos(obj_pos, phase="pre_grasp"))
        elif new_state == ExpertPolicyState.DESCEND_TO_GRASP:
            obj_pos = self._object_pos(env, self._current_object_name())
            self._sync_motion_target(self._grasp_target_pos(obj_pos, phase="grasp"))

        target, gripper = self._waypoint_for_state(new_state, env, ee_pos)
        if new_state not in LIVE_TRACKING_STATES:
            self._sync_motion_target(target)
        self.ctx.gripper_command = gripper

    def _init_targets(self, env) -> None:
        c1 = self._object_pos(env, "cube_1")
        self.ctx.stack_base_xy = c1[:2].clone()
        down = self.down_quat.clone()
        # Pick cube_2 onto cube_1, then cube_3 onto cube_2 (Isaac stack subtask order).
        self.ctx.targets = [
            CubeTarget("cube_2", torch.zeros(3, device=self.device), down.clone()),
            CubeTarget("cube_3", torch.zeros(3, device=self.device), down.clone()),
        ]

    def _object_pos(self, env, name: str) -> torch.Tensor:
        return env.scene[name].data.root_pos_w[0] - env.scene.env_origins[0]

    def _ee_pose(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        ee = env.scene["ee_frame"]
        pos = ee.data.target_pos_w[0, 0] - env.scene.env_origins[0]
        quat = ee.data.target_quat_w[0, 0]
        return pos, quat

    def _current_object_name(self) -> str:
        return self.ctx.targets[self.ctx.cube_index].object_name

    def _stack_place_pos(self, env, cube_index: int) -> torch.Tensor:
        if cube_index == 0:
            c1 = self._object_pos(env, "cube_1")
            place = c1.clone()
            place[2] = c1[2] + HEIGHT_DIFF
            return place
        c2 = self._object_pos(env, "cube_2")
        place = c2.clone()
        place[2] = c2[2] + HEIGHT_DIFF
        return place

    def _fail(self, reason: str) -> None:
        self.ctx.failed_from_state = self.ctx.state
        self.ctx.failure_reason = reason
        self.ctx.state = ExpertPolicyState.FAIL

    def _near_target(self, ee_pos: torch.Tensor, *, threshold: Optional[float] = None) -> bool:
        if self.ctx.state_target_pos is None:
            return False
        limit = self.pos_threshold if threshold is None else threshold
        return torch.norm(ee_pos - self.ctx.state_target_pos).item() < limit

    def _timeout(self) -> bool:
        if self.ctx.state == ExpertPolicyState.CHECK_TASK_SUCCESS:
            return self.ctx.step_in_state >= 600
        return self.ctx.step_in_state >= self.state_timeout_steps

    def _update_transition_progress(self, ee_pos: torch.Tensor) -> None:
        if self.ctx.state_start_pos is None or self.ctx.state_target_pos is None:
            return
        total = torch.norm(self.ctx.state_target_pos - self.ctx.state_start_pos).item()
        if total < 1e-6:
            self.ctx.transition_progress = 1.0
            return
        moved = torch.norm(ee_pos - self.ctx.state_start_pos).item()
        self.ctx.transition_progress = min(1.0, moved / total)

    def _isaac_object_grasped(self, env) -> bool:
        obj_name = self._current_object_name()
        return bool(
            stack_mdp.object_grasped(
                env,
                robot_cfg=SceneEntityCfg("robot"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                object_cfg=SceneEntityCfg(obj_name),
            )[0]
        )

    def _check_grasp_closed(self, env) -> tuple[bool, str]:
        if not self.enable_quality_checks or self.ctx.grasp_object_z is None:
            return True, ""
        obj_pos = self._object_pos(env, self._current_object_name())
        ee_pos, _ = self._ee_pose(env)
        xy_dist = float(torch.norm(obj_pos[:2] - ee_pos[:2]).item())
        if xy_dist > GRASP_XY_MAX:
            return False, "grasp_xy_misaligned"
        if float(obj_pos[2].item()) < self.ctx.grasp_object_z - 0.004:
            return False, "cube_dropped_at_grasp"
        if not self._isaac_object_grasped(env):
            return False, "grasp_not_closed"
        return True, ""

    def _check_grasp_lift_strong(self, env) -> tuple[bool, str]:
        if not self.enable_quality_checks or self.ctx.grasp_object_z is None:
            return True, ""
        obj_pos = self._object_pos(env, self._current_object_name())
        ee_pos, _ = self._ee_pose(env)
        dz = float(obj_pos[2].item()) - self.ctx.grasp_object_z
        if dz < GRASP_LIFT_MIN_Z:
            return False, "grasp_not_lifted"
        xy_dist = float(torch.norm(obj_pos[:2] - ee_pos[:2]).item())
        if xy_dist > GRASP_XY_MAX:
            return False, "cube_drift_xy"
        if self.ctx.grasp_object_xy is not None:
            ee_xy_delta = float(torch.norm(ee_pos[:2] - self.ctx.grasp_ee_xy).item()) if self.ctx.grasp_ee_xy is not None else 0.0
            obj_xy_delta = float(torch.norm(obj_pos[:2] - self.ctx.grasp_object_xy).item())
            if ee_xy_delta > 0.02 and obj_xy_delta < ee_xy_delta * 0.35:
                return False, "cube_not_following_ee"
        return True, ""

    def _check_place_verified(self, env) -> tuple[bool, dict[str, float]]:
        obj_pos = self._object_pos(env, self._current_object_name())
        place = self._stack_place_pos(env, self.ctx.cube_index)
        xy_err = float(torch.norm(obj_pos[:2] - place[:2]).item())
        z_err = abs(float(obj_pos[2].item() - place[2].item()))
        metrics = {"place_xy_error": xy_err, "place_z_error": z_err}
        verified = xy_err <= PLACE_XY_TOLERANCE and z_err <= PLACE_Z_TOLERANCE
        return verified, metrics

    def _check_place_settle(self, env) -> Optional[str]:
        verified, metrics = self._check_place_verified(env)
        if not verified:
            return (
                f"place_error: xy={metrics['place_xy_error']:.4f} "
                f"z={metrics['place_z_error']:.4f} "
                f"tolerance_xy={PLACE_XY_TOLERANCE} tolerance_z={PLACE_Z_TOLERANCE}"
            )
        return None

    def _cube_positions_world(self, env) -> list[list[float]]:
        positions = []
        for name in ("cube_1", "cube_2", "cube_3"):
            pos = self._object_pos(env, name)
            positions.append([float(pos[0].item()), float(pos[1].item()), float(pos[2].item())])
        return positions

    def step(self, env, _success_fn) -> tuple[torch.Tensor, bool]:
        ee_pos, ee_quat = self._ee_pose(env)
        if self.ctx.state in LIVE_TRACKING_STATES:
            self._refresh_live_tracking_targets(env)
        self.ctx.step_in_state += 1
        self.ctx.total_episode_steps += 1
        state = self.ctx.state

        if state in (
            ExpertPolicyState.MOVE_TO_PRE_GRASP,
            ExpertPolicyState.DESCEND_TO_GRASP,
            ExpertPolicyState.LIFT_OBJECT,
            ExpertPolicyState.MOVE_TO_PRE_PLACE,
            ExpertPolicyState.DESCEND_TO_PLACE,
            ExpertPolicyState.RETREAT_TO_SAFE,
            ExpertPolicyState.NEUTRAL_TRANSITION,
        ):
            threshold = 0.018 if state in (ExpertPolicyState.DESCEND_TO_GRASP, ExpertPolicyState.DESCEND_TO_PLACE) else None
            streak = NEAR_STREAK_DESCEND if state in (ExpertPolicyState.DESCEND_TO_GRASP, ExpertPolicyState.DESCEND_TO_PLACE) else (
                NEAR_STREAK_LIFT if state == ExpertPolicyState.LIFT_OBJECT else NEAR_STREAK_APPROACH
            )
            if state == ExpertPolicyState.RETREAT_TO_SAFE:
                streak = NEAR_STREAK_RETREAT_TO_SAFE
            if (
                state == ExpertPolicyState.LIFT_OBJECT
                and self.enable_quality_checks
                and self.ctx.step_in_state >= GRASP_LIFT_CHECK_START
                and self.ctx.step_in_state <= GRASP_LIFT_CHECK_END
            ):
                ok, reason = self._check_grasp_lift_strong(env)
                if not ok:
                    self._fail(reason or "grasp_not_lifted")
            elif state == ExpertPolicyState.RETREAT_TO_SAFE and self.ctx.step_in_state >= NEAR_STREAK_RETREAT_TO_SAFE:
                self._enter_state(ExpertPolicyState.VERIFY_PLACE, env)
            elif self._near_target(ee_pos, threshold=threshold):
                self.ctx.near_target_streak += 1
                if self.ctx.near_target_streak >= streak:
                    if state == ExpertPolicyState.MOVE_TO_PRE_GRASP:
                        self._enter_state(ExpertPolicyState.DESCEND_TO_GRASP, env)
                    elif state == ExpertPolicyState.DESCEND_TO_GRASP:
                        self._enter_state(ExpertPolicyState.CLOSE_GRIPPER, env)
                    elif state == ExpertPolicyState.LIFT_OBJECT:
                        ok, _ = self._check_grasp_lift_strong(env)
                        if ok:
                            self.ctx.cube_lifted_flags.append(True)
                            self._enter_state(ExpertPolicyState.MOVE_TO_PRE_PLACE, env)
                        else:
                            self._fail("grasp_not_lifted")
                    elif state == ExpertPolicyState.MOVE_TO_PRE_PLACE:
                        self._enter_state(ExpertPolicyState.DESCEND_TO_PLACE, env)
                    elif state == ExpertPolicyState.DESCEND_TO_PLACE:
                        self._enter_state(ExpertPolicyState.OPEN_GRIPPER, env)
                    elif state == ExpertPolicyState.RETREAT_TO_SAFE:
                        self._enter_state(ExpertPolicyState.VERIFY_PLACE, env)
                    elif state == ExpertPolicyState.NEUTRAL_TRANSITION:
                        self._enter_state(ExpertPolicyState.MOVE_TO_PRE_GRASP, env)
            elif self._timeout():
                self._fail("state_timeout")

        elif state == ExpertPolicyState.CLOSE_GRIPPER:
            if self.ctx.step_in_state == 1:
                obj_pos = self._object_pos(env, self._current_object_name())
                self.ctx.grasp_object_z = float(obj_pos[2].item())
                self.ctx.grasp_object_xy = obj_pos[:2].clone()
                self.ctx.grasp_ee_z = float(ee_pos[2].item())
                self.ctx.grasp_ee_xy = ee_pos[:2].clone()
            if self.ctx.step_in_state >= NEAR_STREAK_GRIPPER:
                self._enter_state(ExpertPolicyState.VERIFY_GRASP, env)

        elif state == ExpertPolicyState.VERIFY_GRASP:
            if self.ctx.step_in_state >= 8:
                ok, reason = self._check_grasp_closed(env)
                if not ok:
                    self._fail(reason or "grasp_failed")
                else:
                    self._enter_state(ExpertPolicyState.LIFT_OBJECT, env)

        elif state == ExpertPolicyState.OPEN_GRIPPER:
            if self.ctx.step_in_state >= NEAR_STREAK_GRIPPER:
                self._enter_state(ExpertPolicyState.RETREAT_TO_SAFE, env)

        elif state == ExpertPolicyState.VERIFY_PLACE:
            if self.ctx.step_in_state >= PLACE_SETTLE_STEPS:
                warning = self._check_place_settle(env)
                verified, _ = self._check_place_verified(env)
                if warning and self.enable_quality_checks:
                    self._fail("place_error")
                else:
                    self.ctx.cube_placed_flags.append(verified)
                    if warning:
                        self.ctx.place_check_warning = warning
                    self._advance_after_place(env)

        elif state == ExpertPolicyState.CHECK_TASK_SUCCESS:
            self._sync_motion_target(ee_pos.clone())
            if self._timeout():
                self._fail("success_term_timeout")

        elif state == ExpertPolicyState.FAIL:
            pass

        action = self._compute_ik_rel_action(ee_pos, ee_quat)
        done = self.ctx.state == ExpertPolicyState.FAIL
        return action, done

    def _sync_motion_target(self, target: torch.Tensor) -> None:
        self.ctx.state_target_pos = target.clone()
        self.ctx.target_pos = target.clone()
        self.ctx.state_target_quat = self.down_quat.clone()
        self.ctx.target_quat = self.down_quat.clone()

    def _advance_after_place(self, env) -> None:
        self.ctx.cube_index += 1
        if self.ctx.cube_index >= len(self.ctx.targets):
            self._enter_state(ExpertPolicyState.CHECK_TASK_SUCCESS, env)
        else:
            self._enter_state(ExpertPolicyState.NEUTRAL_TRANSITION, env)

    def _orientation_error(self, ee_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        q_c = ee_quat / (torch.norm(ee_quat) + 1e-8)
        q_t = target_quat / (torch.norm(target_quat) + 1e-8)
        w1, x1, y1, z1 = q_c[0], q_c[1], q_c[2], q_c[3]
        w2, x2, y2, z2 = q_t[0], q_t[1], q_t[2], q_t[3]
        qx = w2 * (-x1) + x2 * w1 + y2 * (-z1) + z2 * y1
        qy = w2 * (-y1) + x2 * z1 + y2 * w1 + z2 * (-x1)
        qz = w2 * (-z1) + x2 * (-y1) + y2 * x1 + z2 * w1
        qw = w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1
        sign = 1.0 if float(qw) >= 0 else -1.0
        rot_vec = 2.0 * sign * torch.stack([qx, qy, qz])
        return torch.clamp(rot_vec, -MAX_ROT_DELTA, MAX_ROT_DELTA)

    def _motion_interpolated_target(self) -> Optional[torch.Tensor]:
        if self.ctx.state_start_pos is None or self.ctx.state_target_pos is None:
            return self.ctx.state_target_pos
        return interpolate_pose(
            self.ctx.state_start_pos,
            self.ctx.state_target_pos,
            step=self.ctx.step_in_state,
            duration_steps=self.ctx.state_duration_steps,
        )

    def _compute_ik_rel_action(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
        waypoint = self.ctx.state_target_pos

        if waypoint is None:
            delta_pos = torch.zeros(3, device=self.device)
        else:
            delta_pos = waypoint - ee_pos
            delta_pos = torch.clamp(delta_pos, -MAX_POS_DELTA, MAX_POS_DELTA)

        target_quat = self.ctx.state_target_quat if self.ctx.state_target_quat is not None else self.down_quat
        hold_orientation = self.ctx.state in (
            ExpertPolicyState.CLOSE_GRIPPER,
            ExpertPolicyState.VERIFY_GRASP,
            ExpertPolicyState.OPEN_GRIPPER,
            ExpertPolicyState.VERIFY_PLACE,
            ExpertPolicyState.CHECK_TASK_SUCCESS,
            ExpertPolicyState.MOVE_TO_PRE_GRASP,
            ExpertPolicyState.DESCEND_TO_GRASP,
            ExpertPolicyState.DESCEND_TO_PLACE,
        )
        apply_orientation = self.ctx.state in (
            ExpertPolicyState.LIFT_OBJECT,
            ExpertPolicyState.MOVE_TO_PRE_PLACE,
            ExpertPolicyState.RETREAT_TO_SAFE,
            ExpertPolicyState.NEUTRAL_TRANSITION,
        )
        if hold_orientation or not apply_orientation:
            delta_rot = torch.zeros(3, device=self.device)
        else:
            delta_rot = self._orientation_error(ee_quat, target_quat) * ORIENTATION_GAIN

        arm = delta_pos / ACTION_SCALE
        rot_cmd = delta_rot / ACTION_SCALE
        arm = torch.clamp(arm, -1.0, 1.0)
        rot_cmd = torch.clamp(rot_cmd, -1.0, 1.0)
        gripper = torch.tensor([self.ctx.gripper_command], device=self.device, dtype=ee_pos.dtype)
        raw_action = torch.cat([arm, rot_cmd, gripper])
        raw_action = torch.clamp(raw_action.reshape(-1), -1.0, 1.0)

        if self.previous_action is None:
            self.previous_action = torch.zeros(7, device=self.device)

        action = smooth_ik_rel_action(
            raw_action,
            previous_action=self.previous_action,
            enable_smoothing=self.enable_action_smoothing,
            alpha=self.action_smoothing_alpha,
            max_action_delta=self.max_action_delta,
        )
        self.previous_action = action.clone()
        return action


def setup_output_directories(dataset_file: str) -> tuple[str, str]:
    output_dir = os.path.dirname(dataset_file)
    output_file_name = os.path.splitext(os.path.basename(dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def resolve_sidecar_paths(dataset_file: str) -> tuple[Path, Path, Path, Path]:
    artifacts_dir = Path(dataset_file).expanduser().resolve().parent
    metrics_path = artifacts_dir / "expert_policy_metrics.json"
    failures_path = artifacts_dir / "expert_policy_failures.json"
    behavior_path = artifacts_dir / "expert_policy_behavior_report.json"
    replay_behavior_path = artifacts_dir / "replay_behavior_report.json"
    return metrics_path, failures_path, behavior_path, replay_behavior_path


def _camera_obs_enabled() -> bool:
    return bool(getattr(args_cli, "enable_cameras", False) and getattr(args_cli, "record_camera_obs", False))


def _active_camera_keys() -> tuple[str, ...]:
    return default_camera_keys(include_wrist=bool(getattr(args_cli, "include_wrist_camera", False)))


class ExpertPolicyLivePreview:
    """Writes live/latest.jpg and live/status.json during expert policy rollout."""

    def __init__(
        self,
        live_dir: Path,
        *,
        status_path: Path | None,
        every: int,
        target_demos: int,
    ) -> None:
        self.live_dir = live_dir
        self.frames_dir = live_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = status_path
        self.every = max(1, every)
        self.target_demos = max(1, target_demos)
        self.step_count = 0
        self.frame_count = 0
        self.recorded_demos = 0
        self.phase = "scripted_expert"
        self.message = "仿真环境启动中…"

    def _write_status(self) -> None:
        if self.status_path is None:
            return
        latest = self.live_dir / "latest.jpg"
        payload = {
            "phase": self.phase,
            "visualPhase": self.phase,
            "message": self.message,
            "frameCount": self.frame_count,
            "stepCount": self.step_count,
            "recordedDemos": self.recorded_demos,
            "targetDemos": self.target_demos,
            "liveFrameAvailable": latest.is_file(),
            "liveFrameBlack": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def maybe_write_rgb(self, rgb) -> None:
        self.step_count += 1
        if self.step_count == 1:
            self.phase = "scripted_expert"
            self.message = "仿真环境已启动，正在采集数据…"
            self._write_status()
        if self.step_count % self.every != 0:
            return
        try:
            import cv2
        except ImportError:
            return
        if rgb is None:
            return
        bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        self.frame_count += 1
        cv2.imwrite(str(self.frames_dir / f"frame_{self.frame_count:06d}.jpg"), bgr)
        cv2.imwrite(str(self.live_dir / "latest.jpg"), bgr)
        self.message = f"正在采集 episode {self.recorded_demos + 1}/{self.target_demos}"
        self._write_status()

    def on_demo_recorded(self, recorded: int) -> None:
        self.recorded_demos = recorded
        self.message = f"已完成 episode {recorded}/{self.target_demos}"
        self._write_status()


class EpisodeCameraRecorder:
    """Buffers per-episode RGB frames for HDF5 obs injection."""

    def __init__(self, camera_keys: tuple[str, ...]) -> None:
        self.camera_keys = camera_keys
        self._frames: dict[str, list] = {key: [] for key in camera_keys}
        self.capture_attempts = 0
        self.capture_successes = 0

    def reset(self) -> None:
        self._frames = {key: [] for key in self.camera_keys}

    def record_step(self, env, simulation_app, live_preview: ExpertPolicyLivePreview | None = None) -> None:
        if not self.camera_keys:
            return
        self.capture_attempts += 1
        agentview = capture_viewport_rgb(env, simulation_app, max_attempts=16)
        if agentview is None:
            return
        self.capture_successes += 1
        primary_key = self.camera_keys[0]
        self._frames[primary_key].append(agentview)
        for extra_key in self.camera_keys[1:]:
            self._frames[extra_key].append(agentview.copy())
        if live_preview is not None:
            live_preview.maybe_write_rgb(agentview)

    def export_to_hdf5(self, dataset_path: str, demo_index: int) -> None:
        if not any(self._frames.values()):
            print(
                f"WARNING: camera export skipped for demo_{demo_index}: "
                f"no frames captured (attempts={self.capture_attempts}, "
                f"successes={self.capture_successes})",
                file=sys.stderr,
            )
            return
        resolution = max(32, int(getattr(args_cli, "image_resolution", 128) or 128))
        inject_camera_observations(
            dataset_path,
            f"demo_{demo_index}",
            self._frames,
            height=resolution,
            width=resolution,
        )
        frame_count = len(self._frames[self.camera_keys[0]])
        print(
            f"Injected camera obs for demo_{demo_index}: "
            f"frames={frame_count} keys={list(self._frames.keys())} "
            f"resolution={resolution}"
        )


def create_environment_config(output_dir: str, output_file_name: str):
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=NUM_ENVS)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as exc:
        logger.error("Failed to parse environment configuration: %s", exc)
        raise

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    if _camera_obs_enabled():
        configure_single_env_live_viewer(env_cfg, env_index=0)

    if not hasattr(env_cfg, "recorders") or ActionStateRecorderManagerCfg is None:
        raise RuntimeError("expert policy recorder is not implemented")

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def check_success(env, success_term, success_step_count: int) -> tuple[int, bool]:
    if success_term is None:
        return success_step_count, False
    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            return success_step_count, True
    else:
        success_step_count = 0
    return success_step_count, False


def export_success_episode(env, *, camera_recorder: EpisodeCameraRecorder | None = None, dataset_path: str = "", demo_index: int = 0) -> None:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes(
        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
    )
    env.recorder_manager.export_episodes([0])
    if camera_recorder is not None and dataset_path:
        camera_recorder.export_to_hdf5(dataset_path, demo_index)


def run_recording_loop(
    env,
    success_term,
    expert: StackCubeExpertPolicy,
    tracker: ExpertQualityTracker,
    behavior: ExpertBehaviorTracker,
    *,
    camera_recorder: EpisodeCameraRecorder | None = None,
    live_preview: ExpertPolicyLivePreview | None = None,
    dataset_path: str = "",
) -> int:
    if not hasattr(env, "recorder_manager") or env.recorder_manager is None:
        raise RuntimeError("expert policy recorder is not implemented")

    target_demos = int(args_cli.num_demos)
    max_attempts = int(args_cli.max_attempts) if args_cli.max_attempts > 0 else target_demos * 5
    recorded = 0
    attempts = 0
    success_step_count = 0
    camera_warmed_up = False

    def success_fn():
        if success_term is None:
            return False
        return bool(success_term.func(env, **success_term.params)[0])

    actions = torch.zeros(env.action_space.shape, device=env.device)

    while simulation_app.is_running() and recorded < target_demos and attempts < max_attempts:
        with torch.inference_mode():
            if attempts == 0 and recorded == 0:
                env.reset(seed=args_cli.seed)
                if camera_recorder is not None and not camera_warmed_up:
                    warmup_ok = warmup_viewport_capture(env, simulation_app, max_attempts=60)
                    print(f"Camera viewport warmup ok={warmup_ok}")
                    if not warmup_ok:
                        raise RuntimeError("viewport camera warmup failed; cannot record agentview_image")
                    camera_warmed_up = True
                expert.reset_episode(env)
                if camera_recorder is not None:
                    camera_recorder.reset()
                tracker.begin_attempt()
                behavior.begin_attempt(attempt_index=1, seed=args_cli.seed)
                attempts += 1

            prev_action = expert.previous_action.clone() if expert.previous_action is not None else None
            action, sm_done = expert.step(env, success_fn)
            tracker.record_step_action(action, prev_action)
            actions[0] = action
            env.step(actions)
            if camera_recorder is not None:
                camera_recorder.record_step(env, simulation_app, live_preview=live_preview)

            if expert.ctx.place_check_warning:
                tracker.record_place_warning(expert.ctx.place_check_warning)
                expert.ctx.place_check_warning = None

            if expert.ctx.state == ExpertPolicyState.CHECK_TASK_SUCCESS:
                success_step_count, ready_to_export = check_success(env, success_term, success_step_count)
                if ready_to_export:
                    max_arm_delta = float(args_cli.max_export_arm_delta)
                    expected_picks = len(expert.ctx.targets)
                    behavior_ok = (
                        len(expert.ctx.cube_lifted_flags) >= expected_picks
                        and len(expert.ctx.cube_placed_flags) >= expected_picks
                        and all(expert.ctx.cube_lifted_flags)
                        and all(expert.ctx.cube_placed_flags)
                    )
                    final_success = success_fn()
                    cube_positions = expert._cube_positions_world(env)
                    stack_error = compute_stack_error(cube_positions)
                    if not final_success:
                        behavior.finish_attempt_failure(
                            failed_state=STATE_LABELS.get(ExpertPolicyState.CHECK_TASK_SUCCESS.value),
                            failure_reason="success_term_false",
                            cube_index=expert.ctx.cube_index,
                            final_success_term=False,
                            final_stack_error=stack_error,
                        )
                        tracker.finish_attempt_failure(
                            seed=args_cli.seed + attempts - 1,
                            failed_state_name=STATE_LABELS.get(ExpertPolicyState.CHECK_TASK_SUCCESS.value),
                            failure_reason="success_term_false",
                            cube_index=expert.ctx.cube_index,
                        )
                        print("Attempt rejected: success_term false at export gate", file=sys.stderr)
                    elif not behavior_ok:
                        behavior.finish_attempt_failure(
                            failed_state=STATE_LABELS.get(ExpertPolicyState.CHECK_TASK_SUCCESS.value),
                            failure_reason="behavior_check_failed",
                            cube_index=expert.ctx.cube_index,
                            final_success_term=final_success,
                            final_stack_error=stack_error,
                        )
                        tracker.finish_attempt_failure(
                            seed=args_cli.seed + attempts - 1,
                            failed_state_name=STATE_LABELS.get(ExpertPolicyState.CHECK_TASK_SUCCESS.value),
                            failure_reason="behavior_check_failed",
                            cube_index=expert.ctx.cube_index,
                        )
                        print(
                            f"Attempt rejected: behavior check failed "
                            f"lifted={expert.ctx.cube_lifted_flags} placed={expert.ctx.cube_placed_flags}",
                            file=sys.stderr,
                        )
                    elif tracker.episode_passes_export_gate(max_arm_delta):
                        export_success_episode(
                            env,
                            camera_recorder=camera_recorder,
                            dataset_path=dataset_path,
                            demo_index=recorded,
                        )
                        if camera_recorder is not None:
                            camera_recorder.reset()
                        recorded += 1
                        if live_preview is not None:
                            live_preview.on_demo_recorded(recorded)
                        tracker.record_demo_exported()
                        tracker.finish_attempt_success()
                        behavior.finish_attempt_success(
                            demo_index=recorded - 1,
                            cube_lifted_flags=list(expert.ctx.cube_lifted_flags),
                            cube_placed_flags=list(expert.ctx.cube_placed_flags),
                            final_cube_positions=cube_positions,
                            final_stack_error=stack_error,
                            final_success_term=final_success,
                        )
                        print(f"Recorded demo {recorded}/{target_demos}")
                        print(
                            "EXPERT_POLICY_METRICS "
                            + json.dumps(tracker.to_metrics_dict(), ensure_ascii=False)
                        )
                    else:
                        behavior.finish_attempt_failure(
                            failed_state=STATE_LABELS.get(ExpertPolicyState.CHECK_TASK_SUCCESS.value),
                            failure_reason="quality_rejected_arm_delta",
                            cube_index=expert.ctx.cube_index,
                            final_success_term=final_success,
                            final_stack_error=stack_error,
                        )
                        tracker.record_quality_rejection("quality_rejected_arm_delta")
                        tracker.finish_attempt_failure(
                            seed=args_cli.seed + attempts - 1,
                            failed_state_name=STATE_LABELS.get(ExpertPolicyState.CHECK_TASK_SUCCESS.value),
                            failure_reason="quality_rejected_arm_delta",
                            cube_index=expert.ctx.cube_index,
                        )
                        print(
                            f"Attempt {attempts} quality rejected: max_arm_delta="
                            f"{tracker._current_episode_max_arm_delta:.3f} threshold={max_arm_delta}",
                            file=sys.stderr,
                        )
                    success_step_count = 0
                    if recorded >= target_demos:
                        break
                    env.sim.reset()
                    env.recorder_manager.reset()
                    env.reset(seed=args_cli.seed + recorded)
                    expert.reset_episode(env)
                    if camera_recorder is not None:
                        camera_recorder.reset()
                    tracker.begin_attempt()
                    behavior.begin_attempt(attempt_index=attempts + 1, seed=args_cli.seed + attempts)
                    attempts += 1
                    continue

            if sm_done:
                if expert.ctx.state == ExpertPolicyState.FAIL:
                    reason = expert.ctx.failure_reason or "state_machine_fail"
                    if reason == "state_timeout":
                        tracker.record_timeout()
                    failed_state = expert.ctx.failed_from_state or expert.ctx.state
                    cube_positions = expert._cube_positions_world(env)
                    stack_error = compute_stack_error(cube_positions)
                    behavior.finish_attempt_failure(
                        failed_state=STATE_LABELS.get(failed_state.value, str(failed_state)),
                        failure_reason=reason,
                        cube_index=expert.ctx.cube_index,
                        final_success_term=success_fn(),
                        final_stack_error=stack_error,
                    )
                    tracker.finish_attempt_failure(
                        seed=args_cli.seed + attempts - 1,
                        failed_state_name=STATE_LABELS.get(failed_state.value, str(failed_state)),
                        failure_reason=reason,
                        cube_index=expert.ctx.cube_index,
                    )
                    print(
                        f"Attempt {attempts} failed: state={STATE_LABELS.get(failed_state.value)} "
                        f"reason={reason} cube={expert.ctx.cube_index} "
                        f"step={expert.ctx.total_episode_steps}",
                        file=sys.stderr,
                    )
                    success_step_count = 0
                    env.sim.reset()
                    env.recorder_manager.reset()
                    env.reset(seed=args_cli.seed + attempts)
                    expert.reset_episode(env)
                    if camera_recorder is not None:
                        camera_recorder.reset()
                    tracker.begin_attempt()
                    behavior.begin_attempt(attempt_index=attempts + 1, seed=args_cli.seed + attempts)
                    attempts += 1

    return recorded


def write_sidecar_artifacts(
    tracker: ExpertQualityTracker,
    behavior: ExpertBehaviorTracker,
    dataset_file: str,
) -> None:
    metrics_path, failures_path, behavior_path, _ = resolve_sidecar_paths(dataset_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(tracker.to_metrics_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failures_path.write_text(
        json.dumps(tracker.failure_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    behavior_report = behavior.to_report_dict()
    behavior_status, behavior_warnings = summarize_behavior_status(behavior_report)
    behavior_report["behaviorStatus"] = behavior_status
    behavior_report["behaviorWarnings"] = behavior_warnings
    behavior_path.write_text(
        json.dumps(behavior_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote failure log to {failures_path}")
    print(f"Wrote behavior report to {behavior_path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    output_dir, output_file_name = setup_output_directories(args_cli.dataset_file)
    dataset_path = os.path.join(output_dir, f"{output_file_name}.hdf5")

    try:
        env_cfg, success_term = create_environment_config(output_dir, output_file_name)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Failed to configure environment: {exc}", file=sys.stderr)
        return 1

    render_mode = "rgb_array" if _camera_obs_enabled() else None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode).unwrapped
    except Exception as exc:
        print(f"ERROR: Failed to create environment: {exc}", file=sys.stderr)
        return 1

    if not hasattr(env, "recorder_manager") or env.recorder_manager is None:
        print("ERROR: expert policy recorder is not implemented", file=sys.stderr)
        env.close()
        return 1

    expert = StackCubeExpertPolicy(
        device=env.device,
        enable_action_smoothing=bool(args_cli.enable_action_smoothing),
        action_smoothing_alpha=float(args_cli.action_smoothing_alpha),
        max_action_delta=float(args_cli.max_action_delta),
        enable_quality_checks=bool(args_cli.enable_quality_checks),
        settle_steps=int(args_cli.settle_steps),
    )
    tracker = ExpertQualityTracker(requested_demos=int(args_cli.num_demos))
    behavior = ExpertBehaviorTracker(requested_demos=int(args_cli.num_demos))
    camera_recorder: EpisodeCameraRecorder | None = None
    live_preview: ExpertPolicyLivePreview | None = None
    if _camera_obs_enabled():
        camera_recorder = EpisodeCameraRecorder(_active_camera_keys())
        print(
            f"Camera obs recording enabled: keys={list(_active_camera_keys())} "
            f"resolution={int(args_cli.image_resolution)}"
        )
    live_dir_raw = str(getattr(args_cli, "live_frame_dir", "") or "").strip()
    if live_dir_raw:
        live_dir = Path(live_dir_raw).expanduser().resolve()
        status_raw = str(getattr(args_cli, "live_status_out", "") or "").strip()
        status_path = Path(status_raw).expanduser().resolve() if status_raw else live_dir / "status.json"
        live_preview = ExpertPolicyLivePreview(
            live_dir,
            status_path=status_path,
            every=max(1, int(getattr(args_cli, "live_frame_every", 5) or 5)),
            target_demos=int(args_cli.num_demos),
        )
        live_preview._write_status()

    try:
        recorded = run_recording_loop(
            env,
            success_term,
            expert,
            tracker,
            behavior,
            camera_recorder=camera_recorder,
            live_preview=live_preview,
            dataset_path=dataset_path,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"ERROR: Recording loop failed: {exc}", file=sys.stderr)
        env.close()
        simulation_app.close()
        return 1

    env.close()
    write_sidecar_artifacts(tracker, behavior, args_cli.dataset_file)

    if os.path.isfile(dataset_path):
        try:
            from integrations.isaac_lab.trajectory_quality import write_trajectory_quality_report

            report_path = Path(output_dir) / "trajectory_quality_report.json"
            behavior_path = Path(output_dir) / "expert_policy_behavior_report.json"
            behavior_data = {}
            if behavior_path.is_file():
                behavior_data = json.loads(behavior_path.read_text(encoding="utf-8"))
            write_trajectory_quality_report(
                dataset_path,
                report_path,
                generation_mode="expert_policy",
                behavior_report=behavior_data,
            )
            print(f"Wrote trajectory quality report to {report_path}")
        except Exception as exc:
            logger.warning("trajectory quality report skipped: %s", exc)

    if recorded < args_cli.num_demos:
        print(
            f"ERROR: Only recorded {recorded}/{args_cli.num_demos} demos "
            f"(dataset exists={os.path.isfile(dataset_path)})",
            file=sys.stderr,
        )
        simulation_app.close()
        return 1

    if camera_recorder is not None and camera_recorder.capture_successes == 0:
        print(
            "ERROR: camera obs recording enabled but zero frames were captured",
            file=sys.stderr,
        )
        simulation_app.close()
        return 1

    if not os.path.isfile(dataset_path) or os.path.getsize(dataset_path) == 0:
        print("ERROR: dataset.hdf5 missing or empty after recording", file=sys.stderr)
        simulation_app.close()
        return 1

    print(f"SUCCESS: exported {recorded} demos to {dataset_path}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
