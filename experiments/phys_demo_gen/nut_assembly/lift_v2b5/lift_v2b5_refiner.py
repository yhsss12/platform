"""V2-B5：更强 lift-aware refiner（re-grasp / two-stage lift / contact settle）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from grasp_waypoint_builder import GraspSearchParams, apply_grasp_params_to_eef_waypoints
from refined_waypoint_builder import load_eef_pose_sequence
from trajectory_parameterization import TrajectoryProxy, _shift_1d_signal, load_trajectory_proxy


@dataclass
class LiftV2B5Params:
  """V2-B5 lift_failed 物理精修参数（仅 eef waypoint / gripper 时序）。"""

  grasp_xy_offset_x: float = 0.0
  grasp_xy_offset_y: float = 0.0
  lateral_correction_x: float = 0.0
  lateral_correction_y: float = 0.0
  pre_grasp_height: float = 0.05
  approach_height: float = 0.02
  gripper_close_shift: float = 0.0
  regrasp_shift: float = 0.0
  gripper_extra_close: float = 0.0
  contact_settle_steps: int = 25
  post_grasp_settle_steps: int = 10
  micro_lift_height_stage1: float = 0.03
  micro_lift_height_stage2: float = 0.06
  micro_lift_steps_stage1: int = 15
  micro_lift_steps_stage2: int = 25
  lift_pause_between_stages: int = 5
  lift_speed_scale: float = 0.4
  lift_direction_bias_z: float = 0.0
  enable_two_stage_lift: float = 1.0
  micro_lift_check_threshold: float = 0.008
  nut_follow_threshold: float = 0.05
  gripper_hold_steps: int = 30
  post_extension_steps: int = 0
  extension_lift_height: float = 0.05

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


LIFT_V2B5_SEARCH_SPACE: dict[str, list[float | int]] = {
  "grasp_xy_offset_x": [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06],
  "grasp_xy_offset_y": [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06],
  "lateral_correction_x": [-0.03, -0.015, 0.0, 0.015, 0.03],
  "lateral_correction_y": [-0.03, -0.015, 0.0, 0.015, 0.03],
  "pre_grasp_height": [0.02, 0.04, 0.06, 0.08, 0.10],
  "approach_height": [0.01, 0.02, 0.03, 0.04],
  "gripper_close_shift": [-20, -15, -10, -5, 0, 5],
  "regrasp_shift": [-15, -10, -5, 0, 5, 10],
  "gripper_extra_close": [-0.35, -0.25, -0.15, -0.05, 0.0],
  "contact_settle_steps": [15, 25, 35, 45],
  "post_grasp_settle_steps": [5, 10, 15, 20],
  "micro_lift_height_stage1": [0.02, 0.03, 0.04, 0.05],
  "micro_lift_height_stage2": [0.04, 0.06, 0.08, 0.10, 0.12],
  "micro_lift_steps_stage1": [10, 15, 20, 25],
  "micro_lift_steps_stage2": [15, 20, 30, 40],
  "lift_pause_between_stages": [0, 3, 5, 8, 12],
  "lift_speed_scale": [0.15, 0.25, 0.35, 0.5, 0.7],
  "lift_direction_bias_z": [-0.02, -0.01, 0.0, 0.01, 0.02],
  "enable_two_stage_lift": [0.0, 1.0],
  "micro_lift_check_threshold": [0.005, 0.008, 0.01, 0.015],
  "nut_follow_threshold": [0.03, 0.04, 0.05, 0.06],
  "gripper_hold_steps": [20, 30, 40, 50],
  "post_extension_steps": [0, 15, 30, 45, 60],
  "extension_lift_height": [0.03, 0.05, 0.08, 0.10, 0.12],
}

# demo_3 首轮搜索中唯一 grasp_success_proxy 参数（作为局部精修种子）
GRASP_SUCCESS_SEED_PARAMS = LiftV2B5Params(
  grasp_xy_offset_x=0.04,
  grasp_xy_offset_y=-0.06,
  lateral_correction_x=0.015,
  lateral_correction_y=-0.015,
  pre_grasp_height=0.02,
  approach_height=0.01,
  gripper_close_shift=0,
  regrasp_shift=0,
  gripper_extra_close=-0.25,
  contact_settle_steps=15,
  post_grasp_settle_steps=15,
  micro_lift_height_stage1=0.02,
  micro_lift_height_stage2=0.06,
  micro_lift_steps_stage1=10,
  micro_lift_steps_stage2=30,
  lift_pause_between_stages=5,
  lift_speed_scale=0.25,
  lift_direction_bias_z=0.01,
  enable_two_stage_lift=1.0,
  micro_lift_check_threshold=0.008,
  nut_follow_threshold=0.03,
  gripper_hold_steps=30,
  post_extension_steps=30,
  extension_lift_height=0.08,
)


def apply_lift_v2b5_params_to_eef_waypoints(
  proxy: TrajectoryProxy,
  eef_pose: np.ndarray,
  params: LiftV2B5Params,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
  """返回 refined eef、gripper 时序、阶段索引。"""
  grasp_params = GraspSearchParams(
    grasp_xy_offset_x=params.grasp_xy_offset_x,
    grasp_xy_offset_y=params.grasp_xy_offset_y,
    pre_grasp_height=params.pre_grasp_height,
    approach_height=params.approach_height,
    gripper_close_shift=params.gripper_close_shift + params.regrasp_shift,
    gripper_hold_steps=int(params.contact_settle_steps),
    lift_height=float(params.micro_lift_height_stage1 + params.micro_lift_height_stage2),
    lift_steps=int(params.micro_lift_steps_stage1 + params.micro_lift_steps_stage2),
    speed_scale=float(params.lift_speed_scale),
  )
  refined, shifted_gripper = apply_grasp_params_to_eef_waypoints(proxy, eef_pose, grasp_params)

  grasp_idx = proxy.phases.grasp_index
  length = proxy.length

  # lateral correction during approach
  approach_start = max(0, grasp_idx - 12)
  for step in range(approach_start, grasp_idx + 1):
    w = (step - approach_start) / max(1, grasp_idx - approach_start)
    refined[step, 0, 3] += float(params.lateral_correction_x) * w
    refined[step, 1, 3] += float(params.lateral_correction_y) * w

  if params.gripper_extra_close != 0.0:
    shifted_gripper = np.clip(shifted_gripper - float(params.gripper_extra_close), -1.0, 0.0)

  # re-grasp pulse: brief reopen then harder close
  regrasp_start = min(length - 1, grasp_idx + int(params.contact_settle_steps) // 2)
  regrasp_end = min(length - 1, regrasp_start + 3)
  for step in range(regrasp_start, regrasp_end + 1):
    shifted_gripper[step] = min(0.2, shifted_gripper[step] + 0.35)
  for step in range(regrasp_end + 1, min(length, regrasp_end + 6)):
    shifted_gripper[step] = -1.0

  settle_end = min(length - 1, grasp_idx + int(params.post_grasp_settle_steps))
  if settle_end > grasp_idx:
    hold = refined[grasp_idx, :3, 3].copy()
    rot = refined[grasp_idx, :3, :3].copy()
    for step in range(grasp_idx + 1, settle_end + 1):
      refined[step, :3, 3] = hold
      refined[step, :3, :3] = rot

  lift_begin = settle_end + 1
  stage1_end = min(length - 1, lift_begin + int(params.micro_lift_steps_stage1))
  pause_end = min(length - 1, stage1_end + int(params.lift_pause_between_stages))
  stage2_end = min(length - 1, pause_end + int(params.micro_lift_steps_stage2))

  if stage1_end > lift_begin:
    denom = max(1, stage1_end - lift_begin)
    for step in range(lift_begin, stage1_end + 1):
      g = (step - lift_begin) / denom
      refined[step, 2, 3] += float(params.micro_lift_height_stage1) * g
      refined[step, 2, 3] += float(params.lift_direction_bias_z) * g

  if float(params.enable_two_stage_lift) > 0.5 and stage2_end > pause_end:
    # pause hold
    if pause_end > stage1_end:
      ppos = refined[stage1_end, :3, 3].copy()
      prot = refined[stage1_end, :3, :3].copy()
      for step in range(stage1_end + 1, pause_end + 1):
        refined[step, :3, 3] = ppos
        refined[step, :3, :3] = prot
    denom2 = max(1, stage2_end - pause_end)
    for step in range(pause_end + 1, stage2_end + 1):
      g = (step - pause_end) / denom2
      refined[step, 2, 3] += float(params.micro_lift_height_stage2) * g
      refined[step, 2, 3] += float(params.lift_direction_bias_z) * g * 0.5

  phases = {
    "grasp_index": grasp_idx,
    "lift_begin": lift_begin,
    "stage1_end": stage1_end,
    "pause_end": pause_end,
    "stage2_end": stage2_end,
  }
  refined, shifted_gripper = _extend_trajectory_for_post_lift(
    refined, shifted_gripper.reshape(-1), phases, params
  )
  return refined, shifted_gripper, phases


def _extend_trajectory_for_post_lift(
  refined: np.ndarray,
  gripper: np.ndarray,
  phases: dict[str, int],
  params: LiftV2B5Params,
) -> tuple[np.ndarray, np.ndarray]:
  """在原始 demo 末尾追加纯 Z 抬升 waypoint（不修改 object_poses）。"""
  ext_steps = int(params.post_extension_steps)
  if ext_steps <= 0:
    return refined, gripper

  last_pose = refined[-1].copy()
  z_step = float(params.extension_lift_height) / max(1, ext_steps)
  extra_poses: list[np.ndarray] = []
  extra_grip: list[float] = []
  for i in range(ext_steps):
    pose = last_pose.copy()
    pose[2, 3] += z_step * (i + 1)
    extra_poses.append(pose)
    extra_grip.append(-1.0)

  refined_ext = np.concatenate([refined, np.stack(extra_poses, axis=0)], axis=0)
  grip_ext = np.concatenate([gripper, np.asarray(extra_grip, dtype=float)])
  new_end = len(refined_ext) - 1
  phases["stage2_end"] = new_end
  phases["extension_end"] = new_end
  return refined_ext, grip_ext


def build_lift_v2b5_waypoints_from_hdf5(
  hdf5_path: str,
  demo_key: str,
  label: str,
  params: LiftV2B5Params,
) -> tuple[TrajectoryProxy, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
  proxy = load_trajectory_proxy(hdf5_path, demo_key, label)
  original_eef = load_eef_pose_sequence(hdf5_path, demo_key)
  refined, gripper, phases = apply_lift_v2b5_params_to_eef_waypoints(proxy, original_eef, params)
  return proxy, original_eef, refined, gripper, phases


def lift_v2b5_params_from_dict(raw: dict[str, Any]) -> LiftV2B5Params:
  valid = {f.name for f in fields(LiftV2B5Params)}
  return LiftV2B5Params(**{k: v for k, v in raw.items() if k in valid})
