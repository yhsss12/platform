"""Isaac Lab Franka stack cube task assets and video validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.services.isaacsim_franka_pick_place_assets import (
    VALID_VIDEO_STATUSES,
    VIDEO_STATUS_AVAILABLE,
    VIDEO_STATUS_FAILED,
    VIDEO_STATUS_PARTIAL,
    VIDEO_STATUS_PENDING,
    aggregate_dataset_video_status,
    contains_forbidden_video_path_hint,
    normalize_episode_video_status,
    sync_video_status_fields,
)

TASK_ID = "isaaclab_franka_stack_cube"
TASK_NAME = "Isaac Lab Franka Stack Cube"
SIMULATOR = "Isaac Lab"
TASK_PACKAGE_REL = Path("integrations/IsaacLabBlockStacking")
PLATFORM_RUN_REL = TASK_PACKAGE_REL / "run" / "platform_run.py"
ROBOT = "Franka Panda"
TASK_TYPE = "stacking"
REGISTRY_TASK_CONFIG_ID = "task_isaaclab_franka_stack_cube_v1"
DEFAULT_ISAAC_LAB_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
DEFAULT_MIMIC_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"
VIDEO_REPLAY_MODE_STATE_BASED = "state_based"
VIDEO_REPLAY_MODE_OPEN_LOOP = "open_loop_preview"
VIDEO_CONSISTENCY_HDF5_STATE = "hdf5_state_replay"
VIDEO_CONSISTENCY_OPEN_LOOP = "action_replay_approximate"

EXPERT_SOURCE_EXPERT_POLICY = "Isaac Lab Mimic seed demonstration"
EXPERT_SOURCE_MIMIC = "Isaac Lab Mimic seed demonstration"
EXPERT_SOURCE_TELEOP = "Isaac Lab 官方 teleoperation demonstration pipeline"
EXPERT_SOURCE_POLICY = "Isaac Lab 官方 policy checkpoint"

EXPERT_SOURCE_SCRIPTED = "Isaac Lab Stack Cube scripted expert"

FORBIDDEN_VIDEO_PATH_KEYWORDS = (
    "cable",
    "thread",
    "threading",
    "dual_arm",
    "dac_gen",
    "ct_gen",
    "cable_threading",
    "dual_arm_cable",
    "panda_composite_cable",
    "isaacsim_franka_pick_place",
    "frankapickplace",
    "pick_place_official",
)


def resolve_task_package_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent.parent
    return (root / TASK_PACKAGE_REL).resolve()


def resolve_platform_run_script(project_root: Path | None = None) -> Path:
    return resolve_task_package_path(project_root) / "run" / "platform_run.py"


def expert_source_for_mode(generation_mode: str) -> str:
    mode = (generation_mode or "expert_policy").strip()
    if mode == "mimic_auto":
        return EXPERT_SOURCE_MIMIC
    if mode == "teleop_record":
        return EXPERT_SOURCE_TELEOP
    if mode in {"policy", "policy_checkpoint"}:
        return EXPERT_SOURCE_POLICY
    if mode in {"expert_policy", "scripted_expert"}:
        return EXPERT_SOURCE_SCRIPTED
    return EXPERT_SOURCE_EXPERT_POLICY


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def validate_manifest_task_ids(
    *,
    dataset_manifest: dict[str, Any],
    episode_manifest: dict[str, Any],
    expected_task_id: str = TASK_ID,
) -> tuple[bool, Optional[str]]:
    dataset_task = str(dataset_manifest.get("task_id") or dataset_manifest.get("taskType") or "")
    episode_task = str(episode_manifest.get("task_id") or "")
    if dataset_task and dataset_task != expected_task_id:
        return False, f"dataset task_id mismatch: {dataset_task}"
    if episode_task and episode_task != expected_task_id:
        return False, f"episode task_id mismatch: {episode_task}"
    return True, None


def resolve_job_episode_video_path(
    job_root: Path,
    episode_id: str = "ep_000001",
) -> tuple[Optional[Path], dict[str, Any]]:
    from app.services.isaaclab_franka_stack_cube_state_replay import state_replay_video_name

    meta: dict[str, Any] = {"episodeId": episode_id}
    episode_manifest = _read_json(job_root / "episodes" / episode_id / "episode_manifest.json")
    dataset_manifest = _read_json(job_root / "dataset_manifest.json")
    ok, err = validate_manifest_task_ids(
        dataset_manifest=dataset_manifest,
        episode_manifest=episode_manifest,
    )
    meta["taskIdValidated"] = ok
    if err:
        meta["validationError"] = err

    replay_mode = str(
        episode_manifest.get("video_replay_mode")
        or episode_manifest.get("videoReplayMode")
        or ""
    )
    meta["video_replay_mode"] = replay_mode or None
    meta["videoReplayMode"] = replay_mode or None

    manifest_video_rel = episode_manifest.get("video_path") or episode_manifest.get("videoPath")
    state_video = job_root / "videos" / state_replay_video_name(episode_id)
    legacy_video = job_root / "videos" / f"{episode_id}.mp4"

    candidates: list[Path] = []
    if isinstance(manifest_video_rel, str) and manifest_video_rel.strip():
        candidates.append(job_root / manifest_video_rel.strip())
    if replay_mode == VIDEO_REPLAY_MODE_STATE_BASED:
        candidates.insert(0, state_video)
    elif state_video.is_file():
        candidates.insert(0, state_video)
    candidates.append(legacy_video)

    video_path: Optional[Path] = None
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if contains_forbidden_video_path_hint(candidate):
            meta["validationError"] = "forbidden video path hint"
            meta["taskIdValidated"] = False
            return None, meta
        if candidate.is_file() and candidate.stat().st_size > 0:
            video_path = candidate
            break

    video_status = str(
        episode_manifest.get("video_status")
        or episode_manifest.get("videoStatus")
        or dataset_manifest.get("video_status")
        or dataset_manifest.get("videoStatus")
        or (VIDEO_STATUS_AVAILABLE if video_path is not None else VIDEO_STATUS_PENDING)
    )
    if video_path is not None and video_status not in {VIDEO_STATUS_AVAILABLE, VIDEO_STATUS_PARTIAL}:
        video_status = VIDEO_STATUS_AVAILABLE
    meta["video_status"] = video_status
    meta["videoStatus"] = video_status
    if video_path is None:
        return None, meta
    return video_path, meta
