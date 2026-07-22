"""Step-level evaluation metrics recorder (summary-first)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MISSING_KINEMATICS = ["eePosition", "qpos", "qvel"]


def _to_numpy_action(action: Any) -> Optional[np.ndarray]:
    if action is None:
        return None
    try:
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return arr
    except (TypeError, ValueError):
        return None


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
        logger.debug("video metadata probe failed for %s: %s", video_path, exc)
        return {"videoDurationSec": None, "videoFpsMeasured": None, "videoFrameCount": None}


class StepMetricRecorder:
    """Records per-step summary statistics for an evaluation episode."""

    def __init__(
        self,
        job_id: str,
        episode_index: int,
        output_dir: Path,
        dt: float | None = None,
        control_frequency_hz: float | None = None,
        record_full_arrays: bool = False,
        downsample: int = 1,
        video_fps: float | None = None,
    ) -> None:
        self.job_id = job_id
        self.episode_index = int(episode_index)
        self.output_dir = Path(output_dir)
        self.dt = float(dt) if dt is not None else None
        self.control_frequency_hz = float(control_frequency_hz) if control_frequency_hz is not None else None
        self.record_full_arrays = bool(record_full_arrays)
        self.downsample = max(1, int(downsample))
        self.video_fps = float(video_fps) if video_fps is not None else None

        self._episode_started_at: float | None = None
        self._step_count = 0
        self._recorded_steps = 0
        self._step_wall_time_sec = 0.0
        self._action_norms: list[float] = []
        self._action_deltas: list[float] = []
        self._prev_action: Optional[np.ndarray] = None
        self._available_fields: set[str] = set()
        self._full_actions: list[np.ndarray] = []
        self._last_error: str | None = None

    def start_episode(self) -> None:
        self._episode_started_at = time.perf_counter()
        self._step_count = 0
        self._recorded_steps = 0
        self._step_wall_time_sec = 0.0
        self._action_norms.clear()
        self._action_deltas.clear()
        self._prev_action = None
        self._available_fields.clear()
        self._full_actions.clear()
        self._last_error = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_step(
        self,
        step: int,
        action: Any | None = None,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
        wall_time_sec: float | None = None,
        step_wall_sec: float | None = None,
    ) -> None:
        del info  # reserved for phase-2 kinematics
        self._step_count = max(self._step_count, int(step) + 1)
        if self._step_count % self.downsample != 0 and int(step) > 0:
            return

        self._recorded_steps += 1
        if reward is not None:
            self._available_fields.add("reward")
        if done is not None:
            self._available_fields.add("done")

        step_delta = step_wall_sec if step_wall_sec is not None else wall_time_sec
        if step_delta is not None and step_delta >= 0:
            self._step_wall_time_sec += float(step_delta)
            self._available_fields.add("stepWallTimeSec")

        action_arr = _to_numpy_action(action)
        if action_arr is not None:
            self._available_fields.add("action")
            norm = float(np.linalg.norm(action_arr))
            self._action_norms.append(norm)
            if self._prev_action is not None and self._prev_action.shape == action_arr.shape:
                delta = float(np.linalg.norm(action_arr - self._prev_action))
                self._action_deltas.append(delta)
            self._prev_action = action_arr.copy()
            if self.record_full_arrays:
                self._full_actions.append(action_arr.copy())

    def finish_episode(
        self,
        success: bool | None = None,
        timeout: bool | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        episode_wall_time = None
        if self._episode_started_at is not None:
            episode_wall_time = round(time.perf_counter() - self._episode_started_at, 4)

        step_wall_time = round(self._step_wall_time_sec, 4) if self._step_wall_time_sec > 0 else None
        sim_time = None
        if self.dt is not None and self.dt > 0 and self._step_count > 0:
            sim_time = round(float(self._step_count) * self.dt, 4)
        elif self.control_frequency_hz and self.control_frequency_hz > 0 and self._step_count > 0:
            sim_time = round(float(self._step_count) / self.control_frequency_hz, 4)

        runtime_sec = step_wall_time if step_wall_time is not None else sim_time
        if runtime_sec is None:
            runtime_sec = episode_wall_time

        mean_action_norm = (
            round(float(np.mean(self._action_norms)), 6) if self._action_norms else None
        )
        max_action_norm = (
            round(float(np.max(self._action_norms)), 6) if self._action_norms else None
        )
        mean_action_delta = (
            round(float(np.mean(self._action_deltas)), 6) if self._action_deltas else None
        )
        max_action_delta = (
            round(float(np.max(self._action_deltas)), 6) if self._action_deltas else None
        )
        smoothness_score = None
        if mean_action_delta is not None:
            smoothness_score = round(1.0 / (1.0 + mean_action_delta), 6)

        delta_count = max(self._step_count - 1, 0)
        has_action = "action" in self._available_fields
        summary: dict[str, Any] = {
            "jobId": self.job_id,
            "episodeIndex": self.episode_index,
            "stepCount": self._step_count,
            "recordedSteps": self._recorded_steps,
            "deltaCount": delta_count,
            "stepWallTimeSec": step_wall_time,
            "simTimeSec": sim_time,
            "episodeWallTimeSec": episode_wall_time,
            "wallTimeSec": runtime_sec,
            "dt": self.dt,
            "controlFrequencyHz": self.control_frequency_hz,
            "videoFps": self.video_fps,
            "meanActionNorm": mean_action_norm,
            "maxActionNorm": max_action_norm,
            "meanActionDelta": mean_action_delta,
            "maxActionDelta": max_action_delta,
            "smoothnessScore": smoothness_score,
            "success": success,
            "timeout": timeout,
            "error": error,
            "availableFields": sorted(self._available_fields),
            "missingFields": list(_MISSING_KINEMATICS),
            "available": has_action,
        }
        if not has_action:
            summary["reason"] = "缺少 action 序列"

        try:
            summary_path = self.output_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if self.record_full_arrays and self._full_actions:
                np.savez_compressed(
                    self.output_dir / "step_arrays.npz",
                    actions=np.stack(self._full_actions, axis=0),
                )
        except OSError as exc:
            self._last_error = str(exc)
            logger.warning("step metrics write failed: %s", exc)
            summary["writeWarning"] = str(exc)

        return summary


    @staticmethod
    def safe_finish(recorder: Optional["StepMetricRecorder"], **kwargs: Any) -> None:
        if recorder is None:
            return
        try:
            recorder.finish_episode(**kwargs)
        except Exception as exc:
            logger.warning("step metrics finish failed: %s", exc)


def _episode_runtime_sec(summary: dict[str, Any]) -> float | None:
    step_wall = summary.get("stepWallTimeSec")
    if step_wall is not None:
        try:
            parsed = float(step_wall)
            if parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            pass

    sim_time = summary.get("simTimeSec")
    if sim_time is None:
        step_count = summary.get("stepCount")
        dt = summary.get("dt")
        if step_count is not None and dt is not None:
            try:
                sim_time = float(step_count) * float(dt)
            except (TypeError, ValueError):
                sim_time = None
        elif step_count is not None and summary.get("controlFrequencyHz"):
            try:
                hz = float(summary["controlFrequencyHz"])
                if hz > 0:
                    sim_time = float(step_count) / hz
            except (TypeError, ValueError):
                sim_time = None
    if sim_time is not None:
        try:
            parsed = float(sim_time)
            if parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            pass

    for key in ("wallTimeSec", "episodeWallTimeSec"):
        value = summary.get(key)
        if value is not None:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                return parsed
    return None


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return None
    return float(sum(v * w for v, w in zip(values, weights)) / total_weight)


def aggregate_run_metrics_from_summaries(
    summaries: list[dict[str, Any]],
    *,
    default_video_fps: float | None = None,
    default_control_hz: float | None = None,
    video_metadata_by_episode: dict[int, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    """Aggregate episode summaries into job-level runMetrics."""
    if not summaries:
        return {}

    step_counts = [int(s.get("stepCount") or 0) for s in summaries if s.get("stepCount") is not None]
    runtime_values = [_episode_runtime_sec(s) for s in summaries]
    runtime_values = [v for v in runtime_values if v is not None]
    sim_times = [float(s["simTimeSec"]) for s in summaries if s.get("simTimeSec") is not None]
    if not sim_times:
        sim_times = [
            float(s["stepCount"]) * float(s["dt"])
            for s in summaries
            if s.get("stepCount") is not None and s.get("dt") is not None and float(s["dt"]) > 0
        ]
    episode_wall_times = [
        float(s["episodeWallTimeSec"]) for s in summaries if s.get("episodeWallTimeSec") is not None
    ]
    action_norms = [float(s["meanActionNorm"]) for s in summaries if s.get("meanActionNorm") is not None]
    max_action_norms = [float(s["maxActionNorm"]) for s in summaries if s.get("maxActionNorm") is not None]
    mean_deltas = [float(s["meanActionDelta"]) for s in summaries if s.get("meanActionDelta") is not None]
    delta_weights = [
        float(s.get("deltaCount") if s.get("deltaCount") is not None else max(int(s.get("stepCount") or 0) - 1, 0))
        for s in summaries
        if s.get("meanActionDelta") is not None
    ]
    max_deltas = [float(s["maxActionDelta"]) for s in summaries if s.get("maxActionDelta") is not None]

    video_fps = default_video_fps
    control_hz = default_control_hz
    measured_video_fps: list[float] = []
    video_durations: list[float] = []
    for item in summaries:
        if video_fps is None and item.get("videoFps") is not None:
            video_fps = float(item["videoFps"])
        if control_hz is None and item.get("controlFrequencyHz") is not None:
            control_hz = float(item["controlFrequencyHz"])
        if item.get("dt") is not None and control_hz is None:
            dt = float(item["dt"])
            if dt > 0:
                control_hz = round(1.0 / dt, 4)
        episode_idx = int(item.get("episodeIndex") or 0)
        meta = (video_metadata_by_episode or {}).get(episode_idx) or {}
        if meta.get("videoFpsMeasured") is not None:
            measured_video_fps.append(float(meta["videoFpsMeasured"]))
        if meta.get("videoDurationSec") is not None:
            video_durations.append(float(meta["videoDurationSec"]))
        if item.get("videoDurationSec") is not None:
            video_durations.append(float(item["videoDurationSec"]))
        if item.get("videoFpsMeasured") is not None:
            measured_video_fps.append(float(item["videoFpsMeasured"]))

    weighted_mean_delta = _weighted_mean(mean_deltas, delta_weights)
    smoothness_score = None
    if weighted_mean_delta is not None:
        smoothness_score = round(1.0 / (1.0 + weighted_mean_delta), 6)
    else:
        smoothness_values = [float(s["smoothnessScore"]) for s in summaries if s.get("smoothnessScore") is not None]
        if smoothness_values:
            smoothness_score = round(float(np.mean(smoothness_values)), 6)

    if measured_video_fps:
        video_fps = round(float(np.mean(measured_video_fps)), 4)

    run_metrics: dict[str, Any] = {
        "episodeCount": len(summaries),
        "meanSteps": round(float(np.mean(step_counts)), 4) if step_counts else None,
        "maxSteps": int(max(step_counts)) if step_counts else None,
        "meanRuntimeSec": round(float(np.mean(runtime_values)), 4) if runtime_values else None,
        "maxRuntimeSec": round(float(max(runtime_values)), 4) if runtime_values else None,
        "meanSimTimeSec": round(float(np.mean(sim_times)), 4) if sim_times else None,
        "maxSimTimeSec": round(float(max(sim_times)), 4) if sim_times else None,
        "meanEpisodeWallTimeSec": round(float(np.mean(episode_wall_times)), 4) if episode_wall_times else None,
        "maxEpisodeWallTimeSec": round(float(max(episode_wall_times)), 4) if episode_wall_times else None,
        "meanVideoDurationSec": round(float(np.mean(video_durations)), 4) if video_durations else None,
        "videoFps": video_fps,
        "videoFpsMeasured": round(float(np.mean(measured_video_fps)), 4) if measured_video_fps else None,
        "controlFrequencyHz": control_hz,
        "meanActionNorm": round(float(np.mean(action_norms)), 6) if action_norms else None,
        "maxActionNorm": round(float(max(max_action_norms)), 6) if max_action_norms else None,
        "meanActionDelta": round(float(np.mean(mean_deltas)), 6) if mean_deltas else None,
        "maxActionDelta": round(float(max(max_deltas)), 6) if max_deltas else None,
        "smoothnessScore": smoothness_score,
        "runtimeMetricDefinition": {
            "meanSimTimeSec": "mean(stepCount / controlFrequencyHz)",
            "meanRuntimeSec": "legacy wall/step timing; not used for 平均仿真时长",
            "videoFps": "mp4 metadata when available else recording config",
            "controlFrequencyHz": "env.control_freq or 1/dt",
            "smoothnessScore": "1/(1+weighted_mean_action_delta)",
        },
    }
    return {k: v for k, v in run_metrics.items() if v is not None}


def load_episode_summaries(step_metrics_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not step_metrics_root.is_dir():
        return summaries
    for episode_dir in sorted(step_metrics_root.glob("episode_*")):
        summary_path = episode_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                summaries.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return summaries


def _collect_video_metadata(results_dir: Path, summaries: list[dict[str, Any]]) -> dict[int, dict[str, float | None]]:
    videos_dir = results_dir.parent / "videos"
    if not videos_dir.is_dir():
        return {}
    metadata_by_episode: dict[int, dict[str, float | None]] = {}
    for summary in summaries:
        episode_idx = int(summary.get("episodeIndex") or 0)
        if episode_idx <= 0:
            continue
        for name in (f"episode_{episode_idx:03d}.mp4", f"episode_{episode_idx:03d}.browser.mp4"):
            meta = _probe_video_metadata(videos_dir / name)
            if meta.get("videoDurationSec") is not None:
                metadata_by_episode[episode_idx] = meta
                break
    return metadata_by_episode


def attach_run_metrics_to_aggregate(aggregate: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    """Merge runMetrics from step_metrics summaries into aggregate dict."""
    merged = dict(aggregate)
    summaries = load_episode_summaries(results_dir / "step_metrics")
    if not summaries:
        merged.setdefault("runMetricsWarnings", []).append("no step_metrics summaries found")
        return merged

    video_metadata = _collect_video_metadata(results_dir, summaries)
    video_fps = merged.get("videoFps")
    run_metrics = aggregate_run_metrics_from_summaries(
        summaries,
        default_video_fps=float(video_fps) if video_fps is not None else None,
        video_metadata_by_episode=video_metadata,
    )
    if run_metrics:
        merged["runMetrics"] = run_metrics
        metrics_block = merged.get("metrics")
        if not isinstance(metrics_block, dict):
            metrics_block = {}
            merged["metrics"] = metrics_block
        metrics_block.update(run_metrics)
    return merged
