from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from integrations.dual_arm_cable.dataset_export_schema import (
    DEFAULT_ACTION_DIM,
    DEFAULT_CONTROL_FREQUENCY,
    DEFAULT_OBS_KEYS,
    DATASET_FORMAT,
    build_dataset_manifest,
)

MIN_TRAJECTORY_STEPS = 2


class IlExportError(Exception):
    def __init__(self, message: str, *, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class StepInspection:
    step_id: str
    success: bool
    observation_timesteps: int = 0
    action_timesteps: int = 0
    action_available: bool = False
    observation_available: bool = False
    missing_fields: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _results_dir(job_dir: Path) -> Path:
    if (job_dir / "results").is_dir():
        return job_dir / "results"
    return job_dir / "episode"


def _list_step_dirs(results_dir: Path) -> list[Path]:
    steps_root = results_dir / "steps"
    if steps_root.is_dir():
        return sorted(p for p in steps_root.iterdir() if p.is_dir() and p.name.startswith("step_"))
    return sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("step_"))


def _trajectory_dir(step_dir: Path) -> Path:
    return step_dir / "trajectory"


def _read_trajectory_manifest(step_dir: Path) -> dict[str, Any]:
    path = _trajectory_dir(step_dir) / "trajectory_manifest.json"
    return _read_json(path) if path.is_file() else {}


def _load_obs_npz(traj_dir: Path, name: str) -> Optional[dict[str, np.ndarray]]:
    path = traj_dir / name
    if not path.is_file():
        return None
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _resolve_rgb_for_index(traj_dir: Path, frame_idx: int) -> np.ndarray:
    if frame_idx < 0:
        return np.zeros(0, dtype=np.uint8)
    frame_dir = traj_dir / "frames" / f"frame_{frame_idx:06d}"
    rgb_path = frame_dir / "rgb.png"
    if rgb_path.is_file():
        return np.frombuffer(rgb_path.read_bytes(), dtype=np.uint8)
    rgb_npy = frame_dir / "rgb.npy"
    if rgb_npy.is_file():
        arr = np.load(rgb_npy)
        return np.asarray(arr, dtype=np.uint8).reshape(-1)
    return np.zeros(0, dtype=np.uint8)


def _load_actions_array(step_dir: Path) -> Optional[np.ndarray]:
    traj_dir = _trajectory_dir(step_dir)
    actions_npy = traj_dir / "actions.npy"
    if actions_npy.is_file():
        arr = np.load(actions_npy)
        if arr.ndim >= 1 and arr.shape[0] > 0:
            return np.asarray(arr, dtype=np.float32)

    trajectory_json = traj_dir / "trajectory.json"
    if trajectory_json.is_file():
        data = _read_json(trajectory_json)
        raw = data.get("actions")
        if isinstance(raw, list) and len(raw) > 0:
            return np.asarray(raw, dtype=np.float32)

    result_json = step_dir / "result.json"
    if result_json.is_file():
        data = _read_json(result_json)
        raw = data.get("actions")
        if isinstance(raw, list) and len(raw) >= MIN_TRAJECTORY_STEPS - 1:
            return np.asarray(raw, dtype=np.float32)

    return None


def _count_observation_timesteps(step_dir: Path) -> int:
    traj_dir = _trajectory_dir(step_dir)
    obs_npz = _load_obs_npz(traj_dir, "obs.npz")
    if obs_npz and "left_arm_joint_pos" in obs_npz:
        t = int(obs_npz["left_arm_joint_pos"].shape[0])
        if t > 0:
            return t

    next_obs_npz = _load_obs_npz(traj_dir, "next_obs.npz")
    if next_obs_npz and "left_arm_joint_pos" in next_obs_npz:
        t = int(next_obs_npz["left_arm_joint_pos"].shape[0])
        if t > 0:
            return t

    traj_frames = traj_dir / "frames"
    if traj_frames.is_dir():
        manifests = sorted(traj_frames.rglob("render_manifest.json"))
        if len(manifests) >= MIN_TRAJECTORY_STEPS:
            return len(manifests)

    traj_obs = traj_dir / "observations.npy"
    if traj_obs.is_file():
        arr = np.load(traj_obs)
        if arr.ndim >= 1 and arr.shape[0] >= MIN_TRAJECTORY_STEPS:
            return int(arr.shape[0])

    frame_dir = step_dir / "frame"
    if frame_dir.is_dir() and (frame_dir / "render_manifest.json").is_file():
        return 1

    return 0


def _trajectory_pairing_valid(step_dir: Path) -> bool:
    traj_dir = _trajectory_dir(step_dir)
    actions = _load_actions_array(step_dir)
    obs = _load_obs_npz(traj_dir, "obs.npz")
    next_obs = _load_obs_npz(traj_dir, "next_obs.npz")
    if actions is None or obs is None or next_obs is None:
        return False
    t = int(actions.shape[0])
    if t <= 0:
        return False
    obs_t = int(obs["left_arm_joint_pos"].shape[0])
    next_t = int(next_obs["left_arm_joint_pos"].shape[0])
    return t == obs_t == next_t and t >= MIN_TRAJECTORY_STEPS


def _inspect_step(step_dir: Path) -> StepInspection:
    step_id = step_dir.name
    result = _read_json(step_dir / "result.json")
    success = bool(result.get("task_success") or result.get("grasp_success"))
    actions = _load_actions_array(step_dir)
    obs_steps = _count_observation_timesteps(step_dir)
    pairing_ok = _trajectory_pairing_valid(step_dir)

    inspection = StepInspection(
        step_id=step_id,
        success=success,
        observation_timesteps=obs_steps,
        action_timesteps=int(actions.shape[0]) if actions is not None else 0,
        action_available=actions is not None and int(actions.shape[0]) > 0,
        observation_available=pairing_ok,
    )

    if actions is None or int(actions.shape[0]) <= 0:
        inspection.missing_fields.append("step_level_actions")
    if not pairing_ok:
        inspection.missing_fields.append("continuous_observation_sequence")
    if actions is not None and not pairing_ok:
        inspection.missing_fields.append("continuous_observation_action_pairs")

    return inspection


def inspect_job(job_dir: Path) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    results_dir = _results_dir(job_dir)
    episode_result = _read_json(results_dir / "episode_result.json")
    step_dirs = _list_step_dirs(results_dir)

    step_reports: list[dict[str, Any]] = []
    missing_fields: set[str] = set()
    action_available = False
    observation_available = False
    exportable_steps = 0

    for step_dir in step_dirs:
        ins = _inspect_step(step_dir)
        step_reports.append(
            {
                "stepId": ins.step_id,
                "success": ins.success,
                "observationTimesteps": ins.observation_timesteps,
                "actionTimesteps": ins.action_timesteps,
                "actionAvailable": ins.action_available,
                "observationAvailable": ins.observation_available,
                "missingFields": ins.missing_fields,
            }
        )
        missing_fields.update(ins.missing_fields)
        action_available = action_available or ins.action_available
        observation_available = observation_available or ins.observation_available
        if ins.success and ins.action_available and ins.observation_available:
            exportable_steps += 1

    episode_success = bool(episode_result.get("episode_success"))
    inspected_episodes = max(len(step_dirs), 1 if episode_result else 0)

    export_ready = exportable_steps > 0
    failure_reason: Optional[str] = None
    if not step_dirs:
        failure_reason = "no step directories found under results/steps"
        missing_fields.add("results/steps")
    elif not action_available:
        failure_reason = "missing step-level actions; cannot export IL dataset"
    elif not observation_available:
        failure_reason = "missing continuous observation/action sequence; cannot export IL dataset"
    elif exportable_steps == 0:
        failure_reason = "no successful steps with paired actions and observations"

    report = {
        "jobDir": str(job_dir),
        "inspectedAt": _utc_now_iso(),
        "inspectedEpisodes": inspected_episodes,
        "exportedEpisodes": 0,
        "skippedEpisodes": inspected_episodes,
        "exportableSteps": exportable_steps,
        "episodeSuccess": episode_success,
        "missingFields": sorted(missing_fields),
        "actionAvailable": action_available,
        "observationAvailable": observation_available,
        "exportReady": export_ready,
        "failureReason": failure_reason,
        "hdf5Created": False,
        "steps": step_reports,
    }
    return report


def _write_export_report(job_dir: Path, report: dict[str, Any]) -> Path:
    out = job_dir / "datasets" / "export_report.json"
    _write_json(out, report)
    return out


def _split_dual_arm_state(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 14:
        raise ValueError(f"qpos too short for dual-arm split: {qpos.shape[0]}")
    left = qpos[:7]
    right = qpos[7:14]
    cable = qpos[14:] if qpos.shape[0] > 14 else np.zeros(0, dtype=np.float32)
    return left, right, cable


def _load_trajectory_demo(step_dir: Path) -> dict[str, Any]:
    traj_dir = _trajectory_dir(step_dir)
    actions = _load_actions_array(step_dir)
    if actions is None or actions.shape[0] <= 0:
        raise IlExportError(
            "missing step-level actions; cannot export IL dataset",
            report=inspect_job(step_dir.parent.parent if step_dir.name.startswith("step_") else step_dir),
        )

    obs_npz = _load_obs_npz(traj_dir, "obs.npz")
    next_obs_npz = _load_obs_npz(traj_dir, "next_obs.npz")
    if obs_npz is None or next_obs_npz is None:
        raise IlExportError(
            "missing continuous observation/action sequence; cannot export IL dataset",
            report=inspect_job(step_dir.parent.parent if step_dir.name.startswith("step_") else step_dir),
        )

    t = int(actions.shape[0])
    if int(obs_npz["left_arm_joint_pos"].shape[0]) != t or int(next_obs_npz["left_arm_joint_pos"].shape[0]) != t:
        raise IlExportError(
            "action/observation length mismatch; cannot export IL dataset",
            report=inspect_job(step_dir.parent.parent if step_dir.name.startswith("step_") else step_dir),
        )

    rewards_path = traj_dir / "rewards.npy"
    dones_path = traj_dir / "dones.npy"
    rewards = np.load(rewards_path).astype(np.float32) if rewards_path.is_file() else np.zeros((t,), dtype=np.float32)
    dones = np.load(dones_path).astype(np.uint8) if dones_path.is_file() else np.zeros((t,), dtype=np.uint8)

    obs_steps: list[dict[str, np.ndarray]] = []
    for idx in range(t):
        frame_idx = int(obs_npz.get("overhead_rgb_frame_idx", np.full((t, 1), -1))[idx, 0])
        obs_steps.append(
            {
                "left_arm_joint_pos": np.asarray(obs_npz["left_arm_joint_pos"][idx], dtype=np.float32),
                "right_arm_joint_pos": np.asarray(obs_npz["right_arm_joint_pos"][idx], dtype=np.float32),
                "left_arm_joint_vel": np.asarray(obs_npz["left_arm_joint_vel"][idx], dtype=np.float32),
                "right_arm_joint_vel": np.asarray(obs_npz["right_arm_joint_vel"][idx], dtype=np.float32),
                "cable_state": np.asarray(obs_npz["cable_state"][idx], dtype=np.float32),
                "overhead_rgb": _resolve_rgb_for_index(traj_dir, frame_idx),
            }
        )

    next_obs_steps: list[dict[str, np.ndarray]] = []
    for idx in range(t):
        frame_idx = int(next_obs_npz.get("overhead_rgb_frame_idx", np.full((t, 1), -1))[idx, 0])
        next_obs_steps.append(
            {
                "left_arm_joint_pos": np.asarray(next_obs_npz["left_arm_joint_pos"][idx], dtype=np.float32),
                "right_arm_joint_pos": np.asarray(next_obs_npz["right_arm_joint_pos"][idx], dtype=np.float32),
                "left_arm_joint_vel": np.asarray(next_obs_npz["left_arm_joint_vel"][idx], dtype=np.float32),
                "right_arm_joint_vel": np.asarray(next_obs_npz["right_arm_joint_vel"][idx], dtype=np.float32),
                "cable_state": np.asarray(next_obs_npz["cable_state"][idx], dtype=np.float32),
                "overhead_rgb": _resolve_rgb_for_index(traj_dir, frame_idx),
            }
        )

    traj_manifest = _read_trajectory_manifest(step_dir)
    return {
        "obs_steps": obs_steps,
        "next_obs_steps": next_obs_steps,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "action_semantics": traj_manifest.get("actionSemantics", "recorded_joint_position_targets"),
        "control_frequency": int(traj_manifest.get("controlFrequency", DEFAULT_CONTROL_FREQUENCY)),
    }


def _write_hdf5_demo(h5_group, demo: dict[str, Any]) -> None:
    obs_steps: list[dict[str, np.ndarray]] = demo["obs_steps"]
    next_obs_steps: list[dict[str, np.ndarray]] = demo.get("next_obs_steps") or obs_steps
    actions = np.asarray(demo["actions"], dtype=np.float32)
    rewards_raw = demo.get("rewards")
    dones_raw = demo.get("dones")
    rewards = (
        np.asarray(rewards_raw, dtype=np.float32)
        if rewards_raw is not None
        else np.zeros((actions.shape[0],), dtype=np.float32)
    )
    dones = (
        np.asarray(dones_raw, dtype=np.uint8)
        if dones_raw is not None
        else np.zeros((actions.shape[0],), dtype=np.uint8)
    )
    t_obs = len(obs_steps)

    def stack_obs_key(steps: list[dict[str, np.ndarray]], key: str, *, allow_rgb: bool = False) -> np.ndarray:
        if allow_rgb:
            return np.stack([step[key] for step in steps], axis=0)
        return np.stack([step[key] for step in steps], axis=0)

    obs_grp = h5_group.create_group("obs")
    obs_grp.create_dataset("left_arm_joint_pos", data=stack_obs_key(obs_steps, "left_arm_joint_pos"))
    obs_grp.create_dataset("right_arm_joint_pos", data=stack_obs_key(obs_steps, "right_arm_joint_pos"))
    obs_grp.create_dataset("left_arm_joint_vel", data=stack_obs_key(obs_steps, "left_arm_joint_vel"))
    obs_grp.create_dataset("right_arm_joint_vel", data=stack_obs_key(obs_steps, "right_arm_joint_vel"))
    obs_grp.create_dataset("cable_state", data=stack_obs_key(obs_steps, "cable_state"))
    rgb_arrays = [step["overhead_rgb"] for step in obs_steps if step["overhead_rgb"].size > 0]
    if rgb_arrays and all(arr.shape == rgb_arrays[0].shape for arr in rgb_arrays):
        obs_grp.create_dataset("overhead_rgb", data=np.stack(rgb_arrays, axis=0))

    next_obs_grp = h5_group.create_group("next_obs")
    next_obs_grp.create_dataset("left_arm_joint_pos", data=stack_obs_key(next_obs_steps, "left_arm_joint_pos"))
    next_obs_grp.create_dataset("right_arm_joint_pos", data=stack_obs_key(next_obs_steps, "right_arm_joint_pos"))
    next_obs_grp.create_dataset("left_arm_joint_vel", data=stack_obs_key(next_obs_steps, "left_arm_joint_vel"))
    next_obs_grp.create_dataset("right_arm_joint_vel", data=stack_obs_key(next_obs_steps, "right_arm_joint_vel"))
    next_obs_grp.create_dataset("cable_state", data=stack_obs_key(next_obs_steps, "cable_state"))
    next_rgb_arrays = [step["overhead_rgb"] for step in next_obs_steps if step["overhead_rgb"].size > 0]
    if next_rgb_arrays and all(arr.shape == next_rgb_arrays[0].shape for arr in next_rgb_arrays):
        next_obs_grp.create_dataset("overhead_rgb", data=np.stack(next_rgb_arrays, axis=0))

    h5_group.create_dataset("actions", data=actions)
    h5_group.create_dataset("dones", data=dones[: actions.shape[0]])
    h5_group.create_dataset("rewards", data=rewards[: actions.shape[0]])


def _episode_generation_stats(job_dir: Path) -> dict[str, int]:
    results_dir = _results_dir(job_dir)
    episode_result = _read_json(results_dir / "episode_result.json")
    step_dirs = _list_step_dirs(results_dir)
    total = int(episode_result.get("max_cables") or len(step_dirs) or 0)
    if total <= 0 and step_dirs:
        total = len(step_dirs)
    attempted = int(episode_result.get("num_steps_attempted") or len(step_dirs) or total)
    succeeded = int(episode_result.get("num_cables_succeeded") or 0)
    return {
        "totalEpisodes": max(total, attempted, len(step_dirs)),
        "completedEpisodes": attempted,
        "successfulEpisodesRaw": succeeded,
    }


def export_job(job_dir: Path, *, job_id: Optional[str] = None) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    job_id = job_id or job_dir.name
    report = inspect_job(job_dir)
    if not report.get("exportReady"):
        report["skippedEpisodes"] = report.get("inspectedEpisodes", 0)
        report["exportedEpisodes"] = 0
        report["hdf5Created"] = False
        _write_export_report(job_dir, report)
        raise IlExportError(report.get("failureReason") or "export not ready", report=report)

    try:
        import h5py
    except ImportError as exc:
        report["failureReason"] = "h5py not available"
        report["hdf5Created"] = False
        _write_export_report(job_dir, report)
        raise IlExportError("h5py not available", report=report) from exc

    results_dir = _results_dir(job_dir)
    datasets_dir = job_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = datasets_dir / "dataset.hdf5"
    manifest_path = datasets_dir / "dataset.manifest.json"

    demos: list[dict[str, Any]] = []
    skipped = 0
    for step_dir in _list_step_dirs(results_dir):
        ins = _inspect_step(step_dir)
        if not (ins.success and ins.action_available and ins.observation_available):
            skipped += 1
            continue
        demos.append(_load_trajectory_demo(step_dir))

    if not demos:
        report["failureReason"] = "no exportable successful demos"
        report["skippedEpisodes"] = skipped
        report["exportedEpisodes"] = 0
        report["hdf5Created"] = False
        _write_export_report(job_dir, report)
        raise IlExportError(report["failureReason"], report=report)

    action_dim = int(demos[0]["actions"].shape[-1]) if demos[0]["actions"].ndim > 1 else DEFAULT_ACTION_DIM
    action_semantics = str(demos[0].get("action_semantics") or "recorded_joint_position_targets")
    control_frequency = int(demos[0].get("control_frequency") or DEFAULT_CONTROL_FREQUENCY)

    with h5py.File(hdf5_path, "w") as h5:
        data_grp = h5.create_group("data")
        for idx, demo in enumerate(demos):
            demo_grp = data_grp.create_group(f"demo_{idx}")
            _write_hdf5_demo(demo_grp, demo)

    gen_stats = _episode_generation_stats(job_dir)
    total_episodes = max(gen_stats["totalEpisodes"], len(demos) + skipped)
    successful_episodes = len(demos)
    failed_episodes = max(0, total_episodes - successful_episodes)
    success_rate = float(successful_episodes / total_episodes) if total_episodes > 0 else 0.0

    manifest = build_dataset_manifest(
        source_job_id=job_id,
        control_frequency=control_frequency,
        num_episodes=successful_episodes,
        successful_episodes=successful_episodes,
        obs_keys=list(DEFAULT_OBS_KEYS),
        action_dim=action_dim,
        action_semantics=action_semantics,
        artifacts={
            "hdf5": str(hdf5_path),
            "manifest": str(manifest_path),
            "exportReport": str(datasets_dir / "export_report.json"),
        },
        extra={
            "totalEpisodes": total_episodes,
            "completedEpisodes": gen_stats["completedEpisodes"],
            "failedEpisodes": failed_episodes,
            "successRate": success_rate,
            "sourceType": "simulation_generated",
            "format": DATASET_FORMAT,
        },
    )
    manifest["createdAt"] = _utc_now_iso()
    _write_json(manifest_path, manifest)

    report.update(
        {
            "exportedEpisodes": len(demos),
            "skippedEpisodes": skipped,
            "missingFields": [],
            "actionAvailable": True,
            "observationAvailable": True,
            "hdf5Created": True,
            "failureReason": None,
            "datasetManifestPath": str(manifest_path),
            "hdf5Path": str(hdf5_path),
        }
    )
    _write_export_report(job_dir, report)
    return {
        "jobId": job_id,
        "manifest": manifest,
        "manifestPath": str(manifest_path),
        "hdf5Path": str(hdf5_path),
        "exportReport": report,
    }
