"""HDF5 validation and helpers for Franka Stack Cube state-based replay video."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

TABLE_Z_APPROX = 0.02
MIN_LIFTED_CUBE_Z = 0.05
GRIPPER_MIN_STD = 1e-4
GRIPPER_ACTION_MIN_TRANSITIONS = 1

STATE_REPLAY_VIDEO_SUFFIX = "_state_replay.mp4"
STATE_REPLAY_SCRIPT = "scripts/platform/stack_cube_state_replay_video.py"


@dataclass
class StateReplayValidation:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    demo_count: int = 0
    demo_key: str = ""
    num_steps: int = 0
    cube_positions_present: bool = False
    gripper_variation_ok: bool = False
    cube_lift_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "demo_count": self.demo_count,
            "demo_key": self.demo_key,
            "num_steps": self.num_steps,
            "cube_positions_present": self.cube_positions_present,
            "gripper_variation_ok": self.gripper_variation_ok,
            "cube_lift_ok": self.cube_lift_ok,
        }


def state_replay_video_name(episode_id: str) -> str:
    return f"{episode_id}{STATE_REPLAY_VIDEO_SUFFIX}"


def state_replay_video_rel(episode_id: str) -> str:
    return f"videos/{state_replay_video_name(episode_id)}"


def validate_hdf5_for_state_replay(
    dataset_path: Path | str,
    *,
    demo_index: int = 0,
) -> StateReplayValidation:
    path = Path(dataset_path)
    result = StateReplayValidation(ok=False)
    if not path.is_file():
        result.reasons.append("dataset_hdf5_missing")
        return result

    try:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                result.reasons.append("missing_data_group")
                return result
            demos = sorted(k for k in handle["data"].keys() if k.startswith("demo_"))
            result.demo_count = len(demos)
            if result.demo_count <= 0:
                result.reasons.append("zero_demos")
                return result
            if demo_index < 0 or demo_index >= len(demos):
                result.reasons.append("invalid_demo_index")
                return result

            demo_key = demos[demo_index]
            result.demo_key = demo_key
            demo = handle["data"][demo_key]

            if "obs" not in demo or "cube_positions" not in demo["obs"]:
                result.reasons.append("missing_cube_positions")
                return result
            result.cube_positions_present = True

            cube_positions = np.asarray(demo["obs"]["cube_positions"])
            result.num_steps = int(cube_positions.shape[0])
            if result.num_steps <= 0:
                result.reasons.append("empty_episode")
                return result

            cp3 = cube_positions.reshape(result.num_steps, 3, 3)
            final_z = cp3[-1, :, 2]
            max_z = float(np.max(final_z))
            result.cube_lift_ok = max_z >= MIN_LIFTED_CUBE_Z
            if not result.cube_lift_ok:
                result.reasons.append("cube_not_lifted_at_end")

            gripper_ok = False
            if "actions" in demo:
                actions = np.asarray(demo["actions"])
                if actions.ndim == 2 and actions.shape[0] > 1 and actions.shape[1] >= 1:
                    grip = actions[:, -1]
                    transitions = int(np.sum(np.abs(np.diff(grip)) > 0.5))
                    gripper_ok = transitions >= GRIPPER_ACTION_MIN_TRANSITIONS
            if not gripper_ok and "obs" in demo and "gripper_pos" in demo["obs"]:
                gp = np.asarray(demo["obs"]["gripper_pos"])
                gripper_ok = float(np.std(gp)) > GRIPPER_MIN_STD
            result.gripper_variation_ok = gripper_ok
            if not gripper_ok:
                result.reasons.append("gripper_no_variation")

            has_states = "states" in demo and "articulation" in demo["states"]
            has_obs_joints = "obs" in demo and "joint_pos" in demo["obs"]
            if not has_states and not has_obs_joints:
                result.reasons.append("missing_states_and_joint_obs")
                return result

            result.ok = result.cube_lift_ok and result.gripper_variation_ok
            if not result.ok and not result.reasons:
                result.reasons.append("validation_failed")
    except (OSError, KeyError, ValueError) as exc:
        result.reasons.append(f"hdf5_read_error:{exc}")

    return result


def build_state_replay_cli_args(
    *,
    task_id: str,
    dataset_file: Path,
    live_frame_dir: Path,
    headless: bool,
    enable_cameras: bool,
    device: str = "cpu",
    live_status_out: Path | None = None,
    preview_video_out: Path | None = None,
    live_frame_every: int = 1,
    select_episodes: list[int] | None = None,
) -> list[str]:
    args = [
        "--task",
        task_id.strip(),
        "--dataset_file",
        str(dataset_file),
        "--live_frame_dir",
        str(live_frame_dir),
        "--live_frame_every",
        str(max(1, live_frame_every)),
        "--device",
        device,
        "--replay_mode",
        "state",
    ]
    eps = select_episodes if select_episodes else [0]
    args.append("--select_episodes")
    args.extend(str(int(ep)) for ep in eps)
    if live_status_out is not None:
        args.extend(["--live_status_out", str(live_status_out)])
    if preview_video_out is not None:
        args.extend(["--preview_video_out", str(preview_video_out)])
    if headless:
        args.append("--headless")
    if enable_cameras:
        args.append("--enable_cameras")
    return args
