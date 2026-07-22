from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

ReplayContentKind = Literal[
    "dataset_trajectory_replay",
    "generation_process_preview",
    "evaluation_replay",
]


def _pick_int(*values: Any) -> Optional[int]:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _list_hdf5_demo_names(hdf5_path: Path) -> list[str]:
    if not hdf5_path.is_file():
        return []
    try:
        import h5py

        with h5py.File(hdf5_path, "r") as handle:
            data_group = handle.get("data")
            if data_group is None:
                return []
            return sorted(
                str(key)
                for key in data_group.keys()
                if str(key).startswith("demo_")
            )
    except Exception as exc:
        logger.debug("Failed to inspect HDF5 demos at %s: %s", hdf5_path, exc)
        return []


def _load_manifest(job_root: Path) -> dict[str, Any]:
    manifest = _read_json(job_root / "datasets" / "dataset.manifest.json")
    return manifest if isinstance(manifest, dict) else {}


def _load_failure_records(job_root: Path) -> list[dict[str, Any]]:
    failures_path = job_root / "results" / "failures.json"
    data = _read_json(failures_path)
    if not isinstance(data, list):
        return []

    records: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        source_episode_index = _pick_int(item.get("episode"), summary.get("episode"))
        seed = item.get("seed", summary.get("seed"))
        reason = (
            str(
                item.get("failureReason")
                or item.get("failure_reason")
                or summary.get("failure_reason")
                or summary.get("failureReason")
                or ""
            ).strip()
            or "未满足最终成功条件"
        )
        if reason == "未满足最终成功条件" and summary:
            try:
                from examples.cable_threading.failure_reason import build_cable_threading_failure_reason

                derived = build_cable_threading_failure_reason(summary)
                if derived:
                    reason = derived
            except Exception:
                pass
        failure_code = str(
            item.get("failureCode") or item.get("failure_code") or summary.get("failure_code") or ""
        ).strip() or None
        display_episode_number = (
            source_episode_index + 1 if source_episode_index is not None else None
        )
        records.append(
            {
                "sourceEpisodeIndex": source_episode_index,
                "displayEpisodeNumber": display_episode_number,
                "episodeIndex": display_episode_number,
                "seed": seed,
                "failureReason": reason,
                "failureCode": failure_code,
                "writtenToDataset": False,
            }
        )
    return records


def _build_trajectory_records(hdf5_path: Path, demo_names: list[str]) -> list[dict[str, Any]]:
    if not demo_names:
        return []

    records: list[dict[str, Any]] = []
    for demo_idx, demo_name in enumerate(demo_names):
        source_episode_index: Optional[int] = None
        seed: Any = None
        if hdf5_path.is_file():
            try:
                import h5py

                with h5py.File(hdf5_path, "r") as handle:
                    demo = handle.get("data", {}).get(demo_name)
                    if demo is not None:
                        source_episode_index = _pick_int(demo.attrs.get("episode_index"))
                        if "seed" in demo.attrs:
                            seed = demo.attrs["seed"]
                        if source_episode_index is None:
                            meta_raw = demo.attrs.get("benchmark_episode_metadata")
                            if meta_raw:
                                meta = json.loads(str(meta_raw))
                                source_episode_index = _pick_int(
                                    meta.get("episode"),
                                    (meta.get("summary") or {}).get("episode"),
                                )
                                if seed is None:
                                    seed = meta.get("seed")
            except Exception as exc:
                logger.debug("Failed to read trajectory metadata for %s: %s", demo_name, exc)

        successful_trajectory_index = demo_idx
        display_episode_number = (
            source_episode_index + 1
            if source_episode_index is not None
            else successful_trajectory_index + 1
        )
        records.append(
            {
                "demoName": demo_name,
                "sourceEpisodeIndex": source_episode_index,
                "displayEpisodeNumber": display_episode_number,
                "successfulTrajectoryIndex": successful_trajectory_index,
                "seed": seed,
                "writtenToDataset": True,
            }
        )
    return records


def _has_evaluation_result(job_root: Path) -> bool:
    candidates = (
        job_root / "results" / "aggregate_result.json",
        job_root / "results" / "results.json",
        job_root / "results" / "per_episode_results.json",
        job_root / "videos" / "eval.mp4",
    )
    return any(path.is_file() for path in candidates)


def detect_cable_threading_replay_content(
    job_root: Path,
    *,
    job_id: str,
    metrics: Optional[dict[str, Any]] = None,
    live: Optional[dict[str, Any]] = None,
    is_eval_job: bool = False,
) -> dict[str, Any]:
    metrics_data = dict(metrics or {})
    live_data = dict(live or {})
    manifest = _load_manifest(job_root)
    debug: dict[str, Any] = {}

    hdf5_path = job_root / "datasets" / "dataset.hdf5"
    has_hdf5_file = hdf5_path.is_file()
    trajectories = _list_hdf5_demo_names(hdf5_path) if has_hdf5_file else []
    has_hdf5_trajectories = len(trajectories) > 0

    generate_video_path = job_root / "videos" / "generate.mp4"
    has_generation_preview = generate_video_path.is_file()

    failure_records = _load_failure_records(job_root)
    has_failures = bool(failure_records) or (job_root / "results" / "failures.json").is_file()
    has_evaluation_result = _has_evaluation_result(job_root)

    trajectory_count = len(trajectories) if trajectories else None
    successful_episodes = _pick_int(
        metrics_data.get("successfulEpisodes"),
        live_data.get("successfulEpisodes"),
        manifest.get("successfulEpisodes"),
        manifest.get("num_successful"),
        trajectory_count,
    )
    total_episodes = _pick_int(
        metrics_data.get("episodes"),
        live_data.get("episodes"),
        manifest.get("totalEpisodes"),
        manifest.get("generationRounds"),
    )
    if total_episodes is None and successful_episodes is not None:
        failed_from_metrics = _pick_int(metrics_data.get("failedEpisodes"), manifest.get("failedEpisodes"))
        if failed_from_metrics is not None:
            total_episodes = successful_episodes + failed_from_metrics
        elif manifest.get("num_failed") is not None:
            total_episodes = successful_episodes + int(manifest["num_failed"])

    failed_episodes = _pick_int(
        metrics_data.get("failedEpisodes"),
        manifest.get("failedEpisodes"),
        manifest.get("num_failed"),
        len(failure_records) if failure_records else None,
    )
    if failed_episodes is None and total_episodes is not None and successful_episodes is not None:
        failed_episodes = max(total_episodes - successful_episodes, 0)

    if trajectory_count is None and successful_episodes is not None:
        trajectory_count = successful_episodes

    if is_eval_job or (job_id.startswith("ct_eval_") and has_evaluation_result):
        replay_content_kind: ReplayContentKind = "evaluation_replay"
        primary_source = "evaluation_result"
    elif has_hdf5_trajectories:
        replay_content_kind = "dataset_trajectory_replay"
        primary_source = "dataset.hdf5"
    elif has_generation_preview:
        replay_content_kind = "generation_process_preview"
        primary_source = "videos/generate.mp4"
        debug["hdf5_trajectory_replay_unavailable"] = True
    else:
        replay_content_kind = "generation_process_preview"
        primary_source = None
        debug["hdf5_trajectory_replay_unavailable"] = True

    tabs: list[dict[str, str]] = []
    if has_hdf5_trajectories:
        tabs.append({"id": "dataset_trajectory_replay", "label": "数据集轨迹回放"})
    if has_generation_preview:
        tabs.append({"id": "generation_process_preview", "label": "生成过程预览"})
    if replay_content_kind == "evaluation_replay":
        tabs = [{"id": "evaluation_replay", "label": "评测回放"}]

    if not tabs:
        tabs = [{"id": replay_content_kind, "label": "回放"}]

    has_rgb_observation = False
    rgb_cameras: list[str] = []
    trajectory_display_mode = "state_trajectory"
    if has_hdf5_file and trajectories:
        try:
            from app.services.cable_threading_hdf5_trajectory import inspect_hdf5_trajectory_capabilities

            capabilities = inspect_hdf5_trajectory_capabilities(hdf5_path)
            has_rgb_observation = bool(capabilities.get("hasRgbObservation"))
            rgb_cameras = list(capabilities.get("rgbCameras") or [])
            trajectory_display_mode = str(
                capabilities.get("trajectoryDisplayMode") or "state_trajectory"
            )
        except Exception as exc:
            logger.debug("Failed to inspect HDF5 trajectory capabilities: %s", exc)

    return {
        "replayContentKind": replay_content_kind,
        "hasHdf5Trajectories": has_hdf5_trajectories,
        "trajectoryCount": trajectory_count,
        "totalEpisodes": total_episodes,
        "failedEpisodes": failed_episodes,
        "hasGenerationPreview": has_generation_preview,
        "hasFailures": has_failures,
        "hasEvaluationResult": has_evaluation_result,
        "hasRgbObservation": has_rgb_observation,
        "rgbCameras": rgb_cameras,
        "trajectoryDisplayMode": trajectory_display_mode,
        "primarySource": primary_source,
        "tabs": tabs,
        "trajectories": trajectories,
        "trajectoryRecords": _build_trajectory_records(hdf5_path, trajectories),
        "failureRecords": failure_records,
        "debug": debug,
    }
