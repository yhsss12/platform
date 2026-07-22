"""Best-effort runtime metrics for dual_arm_cable_manipulation eval_* jobs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DUAL_ARM_TASK_TYPE = "dual_arm_cable_manipulation"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _probe_video_metadata(video_path: Path) -> dict[str, float | None]:
    if not video_path.is_file():
        return {"videoDurationSec": None, "videoFpsMeasured": None, "videoFrameCount": None}
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"videoDurationSec": None, "videoFpsMeasured": None, "videoFrameCount": None}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        duration = round(frames / fps, 4) if fps > 0 and frames > 0 else None
        return {
            "videoDurationSec": duration,
            "videoFpsMeasured": round(fps, 4) if fps > 0 else None,
            "videoFrameCount": int(frames) if frames > 0 else None,
        }
    except Exception as exc:
        logger.debug("dual_arm video metadata probe failed for %s: %s", video_path, exc)
        return {"videoDurationSec": None, "videoFpsMeasured": None, "videoFrameCount": None}


def _discover_trajectory_dirs(job_root: Path) -> list[Path]:
    patterns = (
        "episodes/episode_*/episode/step_*/trajectory",
        "episodes/episode_*/episode/*/trajectory",
    )
    found: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for traj_dir in sorted(job_root.glob(pattern)):
            key = str(traj_dir.resolve())
            if key in seen or not traj_dir.is_dir():
                continue
            seen.add(key)
            found.append(traj_dir)
    return found


def _episode_index_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("episode_"):
            try:
                return int(part.split("_", 1)[1])
            except (IndexError, ValueError):
                return None
    return None


def _load_actions(traj_dir: Path) -> np.ndarray | None:
    actions_path = traj_dir / "actions.npy"
    if not actions_path.is_file():
        return None
    try:
        arr = np.asarray(np.load(actions_path))
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        return arr.astype(np.float64, copy=False)
    except Exception:
        return None


def _load_obs_qvel(traj_dir: Path) -> tuple[np.ndarray | None, float | None]:
    obs_path = traj_dir / "obs.npz"
    if not obs_path.is_file():
        return None, None
    try:
        obs = np.load(obs_path)
        parts: list[np.ndarray] = []
        for key in ("left_arm_joint_vel", "right_arm_joint_vel", "qvel"):
            if key not in obs:
                continue
            arr = np.asarray(obs[key], dtype=np.float64)
            if arr.ndim == 1:
                parts.append(arr.reshape(1, -1))
            elif arr.ndim == 2:
                parts.append(arr)
        if not parts:
            return None, None
        qvel = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
        manifest = _read_json(traj_dir / "trajectory_manifest.json")
        control_hz = manifest.get("controlFrequency")
        dt = 1.0 / float(control_hz) if isinstance(control_hz, (int, float)) and control_hz > 0 else None
        return qvel, dt
    except Exception:
        return None, None


def _action_metrics(actions: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(actions, axis=1)
    max_norm = float(np.max(norms)) if norms.size else 0.0
    if actions.shape[0] < 2:
        smoothness = 1.0
        mean_delta = 0.0
    else:
        deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        mean_delta = float(np.mean(deltas))
        smoothness = float(1.0 / (1.0 + mean_delta))
    return {
        "maxActionNorm": max_norm,
        "meanActionNorm": float(np.mean(norms)) if norms.size else 0.0,
        "meanActionDelta": mean_delta,
        "maxActionDelta": float(np.max(np.linalg.norm(np.diff(actions, axis=0), axis=1))) if actions.shape[0] > 1 else 0.0,
        "smoothnessScore": smoothness,
    }


def _joint_speed_metrics(qvel: np.ndarray, dt: float | None) -> dict[str, float | None]:
    if qvel.ndim == 1:
        qvel = qvel.reshape(1, -1)
    speeds = np.linalg.norm(qvel, axis=1)
    out: dict[str, float | None] = {
        "meanJointSpeed": float(np.mean(speeds)) if speeds.size else None,
        "maxJointSpeed": float(np.max(speeds)) if speeds.size else None,
        "meanJointAcceleration": None,
        "maxJointAcceleration": None,
    }
    if dt is not None and dt > 0 and qvel.shape[0] > 1:
        acc = np.linalg.norm(np.diff(qvel, axis=0) / dt, axis=1)
        out["meanJointAcceleration"] = float(np.mean(acc))
        out["maxJointAcceleration"] = float(np.max(acc))
    return out


def _resolve_episode_step_count(job_root: Path, traj_dir: Path, manifest: dict[str, Any]) -> tuple[int | None, str]:
    summary_glob = list(
        (job_root / "results" / "step_metrics").glob(f"**/summary.json")
    )
    ep_idx = _episode_index_from_path(traj_dir)
    if ep_idx is not None:
        for summary_path in summary_glob:
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if int(summary.get("episodeIndex", -1)) == ep_idx and summary.get("stepCount") is not None:
                return int(summary["stepCount"]), "step_metrics.summary.stepCount"

    num_transitions = manifest.get("numTransitions")
    if isinstance(num_transitions, (int, float)) and num_transitions > 0:
        return int(num_transitions), "trajectory_manifest.numTransitions"

    per_episode = _read_json(job_root / "results" / "per_episode_results.json")
    episodes = per_episode.get("episodes") if isinstance(per_episode.get("episodes"), list) else []
    if ep_idx is not None and ep_idx < len(episodes):
        ep = episodes[ep_idx] if isinstance(episodes[ep_idx], dict) else {}
        for key in ("stepCount", "steps", "numTransitions"):
            value = ep.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value), f"per_episode_results.{key}"
            if key == "steps" and isinstance(value, list) and value:
                return len(value), "per_episode_results.steps.length"

    actions = _load_actions(traj_dir)
    if actions is not None:
        return int(actions.shape[0]), "actions.npy.length"

    return None, ""


def _find_episode_video(job_root: Path, traj_dir: Path) -> Path | None:
    ep_idx = _episode_index_from_path(traj_dir)
    if ep_idx is not None:
        for candidate in (
            job_root / "videos" / f"episode_{ep_idx:02d}.mp4",
            job_root / f"episodes/episode_{ep_idx:02d}/episode/episode_video.mp4",
        ):
            if candidate.is_file():
                return candidate
    videos = sorted((job_root / "videos").glob("*.mp4")) if (job_root / "videos").is_dir() else []
    return videos[0] if videos else None


def compute_dual_arm_run_metrics(
    job_root: Path | str,
    aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate runMetrics from dual-arm eval artifacts (best-effort, no fabrication)."""
    root = Path(job_root)
    traj_dirs = _discover_trajectory_dirs(root)
    if not traj_dirs:
        return {}

    step_counts: list[int] = []
    step_sources: list[str] = []
    control_freqs: list[float] = []
    max_action_norms: list[float] = []
    smoothness_scores: list[float] = []
    mean_joint_speeds: list[float] = []
    max_joint_speeds: list[float] = []
    mean_joint_accels: list[float] = []
    max_joint_accels: list[float] = []
    runtime_secs: list[float] = []
    video_fps_values: list[float] = []

    per_episode = _read_json(root / "results" / "per_episode_results.json")
    episodes = per_episode.get("episodes") if isinstance(per_episode.get("episodes"), list) else []

    for traj_dir in traj_dirs:
        manifest = _read_json(traj_dir / "trajectory_manifest.json")
        step_count, step_source = _resolve_episode_step_count(root, traj_dir, manifest)
        if step_count is not None:
            step_counts.append(step_count)
            if step_source:
                step_sources.append(step_source)

        control_hz = manifest.get("controlFrequency")
        if isinstance(control_hz, (int, float)) and control_hz > 0:
            control_freqs.append(float(control_hz))

        actions = _load_actions(traj_dir)
        if actions is not None:
            action_metrics = _action_metrics(actions)
            max_action_norms.append(action_metrics["maxActionNorm"])
            smoothness_scores.append(action_metrics["smoothnessScore"])

        qvel, dt = _load_obs_qvel(traj_dir)
        if qvel is not None:
            joint_metrics = _joint_speed_metrics(qvel, dt)
            if joint_metrics["meanJointSpeed"] is not None:
                mean_joint_speeds.append(joint_metrics["meanJointSpeed"])
            if joint_metrics["maxJointSpeed"] is not None:
                max_joint_speeds.append(joint_metrics["maxJointSpeed"])
            if joint_metrics["meanJointAcceleration"] is not None:
                mean_joint_accels.append(joint_metrics["meanJointAcceleration"])
            if joint_metrics["maxJointAcceleration"] is not None:
                max_joint_accels.append(joint_metrics["maxJointAcceleration"])

        ep_idx = _episode_index_from_path(traj_dir)
        if ep_idx is not None and ep_idx < len(episodes):
            ep = episodes[ep_idx] if isinstance(episodes[ep_idx], dict) else {}
            runtime_sec = ep.get("runtimeSec")
            if isinstance(runtime_sec, (int, float)) and runtime_sec >= 0:
                runtime_secs.append(float(runtime_sec))

        video_path = _find_episode_video(root, traj_dir)
        if video_path is not None:
            meta = _probe_video_metadata(video_path)
            if meta.get("videoFpsMeasured") is not None:
                video_fps_values.append(float(meta["videoFpsMeasured"]))

    if not step_counts:
        return {}

    mean_steps = float(np.mean(step_counts))
    max_steps = float(np.max(step_counts))
    control_frequency_hz = float(np.mean(control_freqs)) if control_freqs else None

    sim_times: list[float] = []
    if control_frequency_hz and control_frequency_hz > 0:
        sim_times = [count / control_frequency_hz for count in step_counts]

    run_metrics: dict[str, Any] = {
        "meanSteps": round(mean_steps, 4),
        "maxSteps": round(max_steps, 4),
        "stepCountSource": step_sources[0] if step_sources else "unknown",
    }

    if control_frequency_hz is not None:
        run_metrics["controlFrequencyHz"] = round(control_frequency_hz, 4)
    if sim_times:
        run_metrics["meanSimTimeSec"] = round(float(np.mean(sim_times)), 4)
        run_metrics["maxSimTimeSec"] = round(float(np.max(sim_times)), 4)
    if max_action_norms:
        run_metrics["maxActionNorm"] = round(float(np.max(max_action_norms)), 4)
    if smoothness_scores:
        run_metrics["smoothnessScore"] = round(float(np.mean(smoothness_scores)), 4)
    if mean_joint_speeds:
        run_metrics["meanJointSpeed"] = round(float(np.mean(mean_joint_speeds)), 4)
    if max_joint_speeds:
        run_metrics["maxJointSpeed"] = round(float(np.max(max_joint_speeds)), 4)
    if mean_joint_accels:
        run_metrics["meanJointAcceleration"] = round(float(np.mean(mean_joint_accels)), 4)
    if max_joint_accels:
        run_metrics["maxJointAcceleration"] = round(float(np.max(max_joint_accels)), 4)
    if video_fps_values:
        run_metrics["videoFps"] = round(float(np.mean(video_fps_values)), 4)

    if step_sources and all("trajectory_manifest.numTransitions" in s for s in step_sources):
        run_metrics["stepCountNote"] = (
            "stepCount 来自 trajectory_manifest.numTransitions（recorded_joint_position_targets），"
            "为轨迹 transition 数，不等同于 MuJoCo env.step 次数"
        )

    return run_metrics


def build_dual_arm_episode_metric_rows(job_root: Path | str) -> dict[int, dict[str, Any]]:
    """Per-episode stepCount / simTimeSec / action metrics from trajectory artifacts."""
    from app.services.evaluation.sim_time_metrics import compute_episode_sim_time_sec

    root = Path(job_root)
    rows: dict[int, dict[str, Any]] = {}
    for traj_dir in _discover_trajectory_dirs(root):
        ep_idx = _episode_index_from_path(traj_dir)
        if ep_idx is None:
            continue
        manifest = _read_json(traj_dir / "trajectory_manifest.json")
        step_count, step_source = _resolve_episode_step_count(root, traj_dir, manifest)
        control_hz = manifest.get("controlFrequency")

        actions = _load_actions(traj_dir)
        action_metrics = _action_metrics(actions) if actions is not None else {}

        sim_time = compute_episode_sim_time_sec(
            step_count=step_count,
            control_frequency_hz=control_hz,
        )
        rows[ep_idx] = {
            "stepCount": step_count,
            "controlFrequencyHz": control_hz,
            "simTimeSec": sim_time,
            "maxActionNorm": action_metrics.get("maxActionNorm"),
            "smoothnessScore": action_metrics.get("smoothnessScore"),
            "stepCountSource": step_source,
        }
    return rows
