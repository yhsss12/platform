"""Platform LeRobot export adapter for cable_threading data generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robosuite.utils.dlo.hdf5_dataset import derive_gripper_actions, derive_joint_delta_actions
from robosuite.utils.dlo.lerobot_dataset import LEROBOT_IMAGE_KEYS, LEROBOT_STATE_KEYS, save_dataset_lerobot

PI0_JOINT_STATE_KEYS = ("robot0_joint_pos", "robot0_gripper_qpos")
PI0_IMAGE_RAW_KEYS = dict(LEROBOT_IMAGE_KEYS)
DEFAULT_TASK_INSTRUCTION = "thread the cable through the pole"


def _concat_state(raw_obs: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    for key in keys:
        val = raw_obs.get(key)
        if val is None:
            return None
        parts.append(np.asarray(val, dtype=np.float32).ravel())
    return np.concatenate(parts) if parts else None


def _trajectories_have_images(trajectories: list[list[dict[str, Any]]]) -> bool:
    if not trajectories or not trajectories[0]:
        return False
    for step in trajectories[0]:
        raw_obs = step.get("raw_obs", {})
        if any(raw_obs.get(raw_key) is not None for raw_key in PI0_IMAGE_RAW_KEYS.values()):
            return True
    return False


def _prepare_pi0_joint_trajectories(
    trajectories: list[list[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], int, int, bool, str]:
    """Build trajectories with 9D state labels and 8D joint_delta_derived actions when possible."""
    prepared: list[list[dict[str, Any]]] = []
    for traj in trajectories:
        raw_obs_list = [step.get("raw_obs", {}) for step in traj]
        if not raw_obs_list or not any(raw_obs_list):
            return trajectories, 0, 0, False, "raw_obs missing; enable camera/offscreen collection for LeRobot"

        joint_deltas = derive_joint_delta_actions(raw_obs_list)
        if joint_deltas is None:
            return trajectories, 0, 0, False, "robot0_joint_pos missing from raw_obs"

        actions_stack = np.stack(
            [np.asarray(step["action"], dtype=np.float32).ravel() for step in traj],
            axis=0,
        )
        gripper = derive_gripper_actions(actions_stack)
        joint_actions = np.concatenate([joint_deltas, gripper], axis=1)
        if joint_actions.shape[1] != 8:
            return (
                trajectories,
                0,
                0,
                False,
                f"derived action_dim={joint_actions.shape[1]}, expected 8",
            )

        new_traj: list[dict[str, Any]] = []
        for idx, step in enumerate(traj):
            raw_obs = step.get("raw_obs", {})
            state = _concat_state(raw_obs, PI0_JOINT_STATE_KEYS)
            if state is None:
                return trajectories, 0, 0, False, "missing robot0_joint_pos / robot0_gripper_qpos for pi0 export"
            new_step = dict(step)
            new_step["action"] = joint_actions[idx]
            new_traj.append(new_step)
        prepared.append(new_traj)

    state_dim = len(_concat_state(prepared[0][0].get("raw_obs", {}), PI0_JOINT_STATE_KEYS))
    action_dim = 8
    if state_dim != 9:
        return prepared, state_dim, action_dim, False, f"state_dim={state_dim}, expected 9"
    if not _trajectories_have_images(prepared):
        return prepared, state_dim, action_dim, False, "missing RGB observations (agentview_image / robot0_eye_in_hand_image)"
    return prepared, state_dim, action_dim, True, ""


def assess_pi0_readiness(
    *,
    robot: str,
    controller_type: str,
    action_mode: str,
    action_representation: str,
    state_dim: int,
    action_dim: int,
    task_instruction: str,
    has_images: bool,
) -> tuple[bool, str]:
    reasons: list[str] = []
    if robot != "Panda":
        reasons.append(f"robot={robot}, expected Panda")
    if controller_type != "JOINT_POSITION":
        reasons.append(f"controller_type={controller_type}, expected JOINT_POSITION")
    if state_dim != 9:
        reasons.append(f"state_dim={state_dim}, expected 9")
    if action_dim != 8:
        reasons.append(f"action_dim={action_dim}, expected 8")
    if action_mode != "joint_delta_derived":
        reasons.append(f"action_mode={action_mode}, expected joint_delta_derived")
    if action_representation != "normalized_joint_delta":
        reasons.append(f"action_representation={action_representation}, expected normalized_joint_delta")
    if not (task_instruction or "").strip():
        reasons.append("task instruction missing")
    if not has_images:
        reasons.append("missing required image keys")
    if reasons:
        return False, "; ".join(reasons)
    return True, ""


def save_cable_threading_lerobot_dataset(
    path: str | Path,
    trajectories: list[list[dict[str, Any]]],
    *,
    robot: str = "Panda",
    task_instruction: str = DEFAULT_TASK_INSTRUCTION,
    fps: int = 20,
    success_flags: list[bool] | None = None,
    episode_metadata: list[dict[str, Any]] | None = None,
    source_controller_type: str = "OSC_POSE",
) -> dict[str, Any]:
    """Save cable_threading trajectories as LeRobot v3 dataset plus platform sidecar files."""
    output_dir = Path(path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    instruction = (task_instruction or DEFAULT_TASK_INSTRUCTION).strip()
    export_traj, state_dim, action_dim, pi0_joint_ok, pi0_reason = _prepare_pi0_joint_trajectories(trajectories)

    if pi0_joint_ok:
        state_keys = PI0_JOINT_STATE_KEYS
        controller_type = "JOINT_POSITION"
        action_mode = "joint_delta_derived"
        action_representation = "normalized_joint_delta"
        warnings: list[str] = []
    else:
        state_keys = LEROBOT_STATE_KEYS
        export_traj = trajectories
        first_obs = trajectories[0][0].get("raw_obs", {}) if trajectories and trajectories[0] else {}
        state_vec = _concat_state(first_obs, state_keys)
        state_dim = len(state_vec) if state_vec is not None else 0
        action_dim = int(np.asarray(trajectories[0][0]["action"]).ravel().shape[0]) if trajectories else 0
        controller_type = source_controller_type
        action_mode = "osc_pose_delta_eef"
        action_representation = "osc_pose_delta_eef"
        if not pi0_reason:
            pi0_reason = (
                f"action_dim is {action_dim} / controller is {source_controller_type}, "
                "not Panda JOINT_POSITION 8D"
            )
        warnings = [pi0_reason]

    if success_flags is None and episode_metadata:
        success_flags = [bool(m.get("summary", {}).get("final_success")) for m in episode_metadata]

    save_dataset_lerobot(
        output_dir,
        export_traj,
        state_keys=state_keys,
        image_keys=PI0_IMAGE_RAW_KEYS,
        fps=fps,
        task_description=instruction,
        success_flags=success_flags,
    )

    has_images = _trajectories_have_images(export_traj)
    pi0_ready, pi0_ready_reason = assess_pi0_readiness(
        robot=robot,
        controller_type=controller_type,
        action_mode=action_mode,
        action_representation=action_representation,
        state_dim=state_dim,
        action_dim=action_dim,
        task_instruction=instruction,
        has_images=has_images,
    )
    if not pi0_ready and not pi0_ready_reason:
        pi0_ready_reason = pi0_reason

    episode_count = len(trajectories)
    frame_count = sum(len(traj) for traj in trajectories)

    meta_stats_path = output_dir / "meta" / "stats.json"
    stats_summary: dict[str, Any] = {}
    if meta_stats_path.is_file():
        stats_summary = json.loads(meta_stats_path.read_text(encoding="utf-8"))

    metadata = {
        "format": "lerobot",
        "source": "cable_threading_generation",
        "task_instruction": instruction,
        "robot": robot,
        "controller_type": controller_type,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_mode": action_mode,
        "action_representation": action_representation,
        "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim_keys": list(state_keys),
        "pi0Ready": pi0_ready,
        "pi0ReadyReason": pi0_ready_reason if not pi0_ready else "",
        "fps": fps,
        "episode_count": episode_count,
        "frame_count": frame_count,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    episode_lengths = [len(traj) for traj in trajectories]
    root_stats: dict[str, Any] = {
        "observation.state": stats_summary.get("observation.state", {}),
        "action": stats_summary.get("action", {}),
        "episode_length": {
            "min": min(episode_lengths) if episode_lengths else 0,
            "max": max(episode_lengths) if episode_lengths else 0,
            "mean": float(frame_count / episode_count) if episode_count else 0.0,
        },
    }
    if has_images:
        sample_img = None
        for step in export_traj[0]:
            raw = step.get("raw_obs", {})
            for raw_key in PI0_IMAGE_RAW_KEYS.values():
                if raw.get(raw_key) is not None:
                    sample_img = np.asarray(raw[raw_key])
                    break
            if sample_img is not None:
                break
        if sample_img is not None:
            root_stats["images"] = {"shape": list(sample_img.shape), "dtype": str(sample_img.dtype)}

    (output_dir / "stats.json").write_text(
        json.dumps(root_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "episode_count": episode_count,
        "frame_count": frame_count,
        "output_format": "lerobot",
        "output_dir": str(output_dir),
        "success": True,
        "warnings": warnings,
        "errors": [],
    }
    (output_dir / "generation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        **metadata,
        "lerobotPath": str(output_dir),
        "reportPath": str(output_dir / "generation_report.json"),
        "statsPath": str(output_dir / "stats.json"),
        "metadataPath": str(output_dir / "metadata.json"),
        "warnings": warnings,
    }
