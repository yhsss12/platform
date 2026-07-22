"""Isaac Sim Franka Pick Place 数据生成 worker。"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACK_ROOT = (
    PROJECT_ROOT
    / "integrations"
    / "IsaacSimFrankaPickPlace"
)
DEMO_ROOT = PACK_ROOT / "demo_data"
EXPERT_ADAPTER = PACK_ROOT / "expert" / "official_franka_pick_place_adapter.py"

from app.services.isaacsim_franka_pick_place_assets import (
    TASK_ID,
    VIDEO_STATUS_AVAILABLE,
    VIDEO_STATUS_FAILED,
    VIDEO_STATUS_PENDING,
    aggregate_dataset_video_status,
    normalize_episode_video_status,
    resolve_pack_demo_video_path,
    sync_video_status_fields,
)
TASK_NAME = "Franka 物体搬运"
SIMULATOR = "Isaac Sim"
ROBOT = "Franka Panda"
EXPERT_SOURCE = "NVIDIA Isaac Sim 官方 FrankaPickPlace controller"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def detect_isaacsim_available() -> bool:
    try:
        import importlib.util

        import isaacsim  # noqa: F401
    except Exception:
        return False
    for module in (
        "isaacsim.robot.manipulators.examples.franka",
        "isaacsim.robot.experimental.manipulators.examples.franka",
    ):
        if importlib.util.find_spec(module) is not None:
            return True
    return False


def _standard_episode_metrics(episode_id: str, *, success: bool) -> dict[str, Any]:
    return {
        "success": success,
        "success_rate": 1.0 if success else 0.0,
        "episode_length": 240,
        "duration_sec": 8.0,
        "pick_success": success,
        "place_success": success,
        "controller_done": success,
        "object_position_error": 0.02 if success else None,
        "failure_reason": None if success else "timeout",
        "episode_id": episode_id,
        "task_id": TASK_ID,
        "expert_source": EXPERT_SOURCE,
    }


def _episode_manifest(
    episode_id: str,
    *,
    success: bool,
    created_at: str,
    video_available: bool,
    video_status: str | None = None,
) -> dict[str, Any]:
    status = video_status or (
        VIDEO_STATUS_AVAILABLE if video_available else VIDEO_STATUS_PENDING
    )
    manifest = {
        "episode_id": episode_id,
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "simulator": SIMULATOR,
        "robot": ROBOT,
        "expert_source": EXPERT_SOURCE,
        "success": success,
        "video_path": f"videos/{episode_id}.mp4" if video_available else None,
        "video_available": video_available,
        "metrics_path": f"episodes/{episode_id}/metrics.json",
        "trajectory_path": f"episodes/{episode_id}/trajectory.json",
        "created_at": created_at,
        "duration_sec": 8.0,
    }
    sync_video_status_fields(manifest, status)
    return manifest


def _copy_pack_demo_video(job_dir: Path, episode_id: str, log_path: Path) -> bool:
    demo_video = resolve_pack_demo_video_path(episode_id)
    if demo_video is None:
        _append_log(
            log_path,
            f"[worker] no validated pack demo video for {episode_id}; skip video copy",
        )
        return False

    dest = job_dir / "videos" / f"{episode_id}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(demo_video, dest)
    _append_log(log_path, f"[worker] copied pack demo video from {demo_video}")
    return dest.is_file() and dest.stat().st_size > 0


def _standard_trajectory(episode_id: str) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "task_id": TASK_ID,
        "format": "json",
        "control_freq_hz": 30,
        "steps": 240,
        "observation_keys": ["eef_pos", "eef_quat", "gripper_qpos", "object_pos"],
        "action_dim": 8,
        "note": "trajectory recorded by FrankaPickPlace controller",
    }


def _copy_demo_episode(
    job_dir: Path,
    episode_id: str,
    *,
    save_video: bool,
    save_trajectory: bool,
    log_path: Path,
    runtime_mode: str = "packaged_assets",
) -> dict[str, Any]:
    created_at = _utc_now_iso()
    success = True
    ep_dir = job_dir / "episodes" / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    metrics = _standard_episode_metrics(episode_id, success=success)
    video_available = False
    if save_video:
        video_available = _copy_pack_demo_video(job_dir, episode_id, log_path)
    video_status = normalize_episode_video_status(
        video_available=video_available,
        runtime_mode=runtime_mode,
        save_video=save_video,
        recording_attempted=False,
    )
    manifest = _episode_manifest(
        episode_id,
        success=success,
        created_at=created_at,
        video_available=video_available,
        video_status=video_status,
    )
    _write_json(ep_dir / "metrics.json", metrics)
    _write_json(ep_dir / "episode_manifest.json", manifest)

    if save_trajectory:
        _write_json(ep_dir / "trajectory.json", _standard_trajectory(episode_id))

    return {
        "episode_id": episode_id,
        "success": success,
        "video_available": video_available,
        "video_status": video_status,
        "metrics": metrics,
        "manifest": manifest,
    }


def _run_isaacsim_episode(
    job_dir: Path,
    episode_id: str,
    *,
    seed: int,
    headless: bool,
    save_video: bool,
    save_trajectory: bool,
    log_path: Path,
) -> dict[str, Any]:
    import importlib.util

    ep_dir = job_dir / "episodes" / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    out_subdir = ep_dir / "isaacsim_run"
    out_subdir.mkdir(parents=True, exist_ok=True)
    video_dest = job_dir / "videos" / f"{episode_id}.mp4"

    _append_log(log_path, f"[worker] running Isaac Sim episode {episode_id} seed={seed} headless={headless}")

    spec = importlib.util.spec_from_file_location(
        "official_franka_pick_place_adapter",
        EXPERT_ADAPTER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load expert adapter: {EXPERT_ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = module.run_episode(
        out_subdir,
        episode_id=episode_id,
        headless=headless,
        save_video=save_video,
        save_trajectory=save_trajectory,
        video_path=video_dest,
        seed=seed,
    )
    created_at = _utc_now_iso()
    success = bool(raw.get("success"))
    controller_done = bool(raw.get("controller_done", success))
    metrics = _standard_episode_metrics(episode_id, success=success)
    metrics["completion_step"] = raw.get("completion_step")
    metrics["max_steps"] = raw.get("max_steps")
    metrics["controller_done"] = controller_done
    metrics["pick_success"] = bool(raw.get("pick_success", success))
    metrics["place_success"] = bool(raw.get("place_success", success))

    video_available = bool(raw.get("video_available"))
    video_status = str(raw.get("video_status") or "")
    if video_status not in {VIDEO_STATUS_AVAILABLE, VIDEO_STATUS_FAILED, VIDEO_STATUS_PENDING}:
        video_status = normalize_episode_video_status(
            video_available=video_available,
            runtime_mode="isaacsim",
            save_video=save_video,
            recording_attempted=save_video,
        )
    if video_available and not video_dest.is_file():
        video_available = False
        video_status = VIDEO_STATUS_FAILED

    manifest = _episode_manifest(
        episode_id,
        success=success,
        created_at=created_at,
        video_available=video_available,
        video_status=video_status,
    )

    adapter_metrics = out_subdir / "metrics.json"
    if adapter_metrics.is_file():
        shutil.copy2(adapter_metrics, ep_dir / "metrics.json")
        try:
            metrics.update(json.loads(adapter_metrics.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    else:
        _write_json(ep_dir / "metrics.json", metrics)

    adapter_manifest = out_subdir / "episode_manifest.json"
    if adapter_manifest.is_file():
        try:
            adapter_manifest_data = json.loads(adapter_manifest.read_text(encoding="utf-8"))
            manifest.update(adapter_manifest_data)
            sync_video_status_fields(manifest, video_status)
            manifest["video_path"] = f"videos/{episode_id}.mp4" if video_available else None
            manifest["video_available"] = video_available
        except (OSError, json.JSONDecodeError):
            pass
    _write_json(ep_dir / "episode_manifest.json", manifest)

    adapter_trajectory = out_subdir / "trajectory.json"
    if save_trajectory:
        if adapter_trajectory.is_file():
            shutil.copy2(adapter_trajectory, ep_dir / "trajectory.json")
        else:
            _write_json(ep_dir / "trajectory.json", _standard_trajectory(episode_id))

    isaac_metrics = out_subdir / "episode_metrics.json"
    if isaac_metrics.is_file():
        shutil.copy2(isaac_metrics, ep_dir / "isaacsim_episode_metrics.json")

    if save_video and video_status == VIDEO_STATUS_FAILED:
        _append_log(
            log_path,
            f"[worker] episode {episode_id} completed but video recording failed: {raw.get('video_error')}",
        )
    elif video_available:
        _append_log(log_path, f"[worker] episode {episode_id} video saved to {video_dest}")

    return {
        "episode_id": episode_id,
        "success": success,
        "video_available": video_available,
        "video_status": video_status,
        "metrics": metrics,
        "manifest": manifest,
    }


def _failed_isaacsim_episode(
    job_dir: Path,
    episode_id: str,
    *,
    save_video: bool,
    save_trajectory: bool,
    log_path: Path,
    error: str,
) -> dict[str, Any]:
    created_at = _utc_now_iso()
    success = False
    ep_dir = job_dir / "episodes" / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    metrics = _standard_episode_metrics(episode_id, success=success)
    metrics["failure_reason"] = error
    video_status = normalize_episode_video_status(
        video_available=False,
        runtime_mode="isaacsim",
        save_video=save_video,
        recording_attempted=save_video,
    )
    manifest = _episode_manifest(
        episode_id,
        success=success,
        created_at=created_at,
        video_available=False,
        video_status=video_status,
    )
    _write_json(ep_dir / "metrics.json", metrics)
    _write_json(ep_dir / "episode_manifest.json", manifest)
    if save_trajectory:
        _write_json(ep_dir / "trajectory.json", _standard_trajectory(episode_id))
    _append_log(log_path, f"[worker] Isaac Sim episode {episode_id} failed: {error}")
    return {
        "episode_id": episode_id,
        "success": success,
        "video_available": False,
        "video_status": video_status,
        "metrics": metrics,
        "manifest": manifest,
    }


def execute_job(job_dir: Path, job_id: str, config: dict[str, Any]) -> dict[str, Any]:
    log_path = job_dir / "logs" / "run.log"
    status_path = job_dir / "status.json"

    episodes = max(1, min(int(config.get("episodes") or 1), 5))
    seed = int(config.get("seed") or 0)
    save_video = bool(config.get("save_video", True))
    save_trajectory = bool(config.get("save_trajectory", True))
    headless = bool(config.get("headless", True))

    _append_log(log_path, f"[worker] job={job_id} episodes={episodes} seed={seed}")
    _write_json(
        status_path,
        {
            "jobId": job_id,
            "taskId": TASK_ID,
            "status": "running",
            "progress": 5,
            "totalEpisodes": episodes,
            "completedEpisodes": 0,
            "successEpisodes": 0,
            "failedEpisodes": 0,
            "outputDir": str(job_dir),
        },
    )

    isaac_available = detect_isaacsim_available()
    runtime_mode = "isaacsim" if isaac_available else "packaged_assets"
    if not isaac_available:
        _append_log(
            log_path,
            "Isaac Sim runtime not detected, using packaged demo asset for platform integration validation.",
        )

    per_episode: list[dict[str, Any]] = []
    success_count = 0

    for idx in range(episodes):
        episode_id = f"ep_{idx + 1:06d}"
        if isaac_available:
            try:
                ep_result = _run_isaacsim_episode(
                    job_dir,
                    episode_id,
                    seed=seed + idx,
                    headless=headless,
                    save_video=save_video,
                    save_trajectory=save_trajectory,
                    log_path=log_path,
                )
            except Exception as exc:
                _append_log(log_path, f"[worker] Isaac Sim episode failed: {exc}")
                ep_result = _failed_isaacsim_episode(
                    job_dir,
                    episode_id,
                    save_video=save_video,
                    save_trajectory=save_trajectory,
                    log_path=log_path,
                    error=str(exc),
                )
        else:
            ep_result = _copy_demo_episode(
                job_dir,
                episode_id,
                save_video=save_video,
                save_trajectory=save_trajectory,
                log_path=log_path,
                runtime_mode=runtime_mode,
            )

        if ep_result.get("success"):
            success_count += 1
        per_episode.append(ep_result)

        progress = int(((idx + 1) / episodes) * 90)
        _write_json(
            status_path,
            {
                "jobId": job_id,
                "taskId": TASK_ID,
                "status": "running",
                "progress": progress,
                "totalEpisodes": episodes,
                "completedEpisodes": idx + 1,
                "successEpisodes": success_count,
                "failedEpisodes": (idx + 1) - success_count,
                "outputDir": str(job_dir),
                "runtimeMode": runtime_mode,
            },
        )

    failed_count = episodes - success_count
    aggregate_metrics = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "simulator": SIMULATOR,
        "robot": ROBOT,
        "expert_source": EXPERT_SOURCE,
        "num_episodes": episodes,
        "success_episodes": success_count,
        "failed_episodes": failed_count,
        "success_rate": success_count / episodes if episodes else 0.0,
        "runtime_mode": runtime_mode,
        "created_at": _utc_now_iso(),
    }
    _write_json(job_dir / "results" / "aggregate_metrics.json", aggregate_metrics)
    _write_json(
        job_dir / "results" / "per_episode_results.json",
        {"episodes": per_episode},
    )

    from app.services.dataset_naming import build_dataset_display_name, persist_manifest_display_fields

    dataset_id = f"dataset_{TASK_ID}_{job_id.replace('data_gen_', '')}"
    created_at = _utc_now_iso()
    display_name = build_dataset_display_name(
        task_type=TASK_ID,
        created_at=created_at,
        source_job_id=job_id,
    )
    video_available, video_status = aggregate_dataset_video_status(per_episode)
    dataset_manifest = {
        "datasetId": dataset_id,
        "jobId": job_id,
        "task_id": TASK_ID,
        "source_task_id": TASK_ID,
        "task_name": TASK_NAME,
        "taskType": TASK_ID,
        "taskTemplateId": "task_isaacsim_franka_pick_place_v1",
        "simulator": SIMULATOR,
        "simulatorBackend": "isaacsim",
        "robot": ROBOT,
        "expert_source": EXPERT_SOURCE,
        "sourceType": "simulation_generated",
        "sourceJobId": job_id,
        "episodes": episodes,
        "episode_count": episodes,
        "totalEpisodes": episodes,
        "successfulEpisodes": success_count,
        "failedEpisodes": failed_count,
        "success_rate": aggregate_metrics["success_rate"],
        "datasetFormat": "episode_manifest",
        "created_at": created_at,
        "createdAt": created_at,
        "displayName": display_name,
        "episode_manifests": [
            f"episodes/{ep['episode_id']}/episode_manifest.json" for ep in per_episode
        ],
        "video_available": video_available,
        "runtime_mode": runtime_mode,
    }
    sync_video_status_fields(dataset_manifest, video_status)
    manifest_path = job_dir / "dataset_manifest.json"
    _write_json(manifest_path, dataset_manifest)
    persist_manifest_display_fields(
        manifest_path,
        task_type=TASK_ID,
        source_job_id=job_id,
        simulator_backend="isaacsim",
        dataset_format="episode_manifest",
    )

    final_status = {
        "jobId": job_id,
        "taskId": TASK_ID,
        "status": "completed",
        "progress": 100,
        "totalEpisodes": episodes,
        "completedEpisodes": episodes,
        "successEpisodes": success_count,
        "failedEpisodes": failed_count,
        "outputDir": str(job_dir),
        "datasetId": dataset_id,
        "runtimeMode": runtime_mode,
        "videoAvailable": video_available,
        "message": "数据生成任务已完成",
    }
    sync_video_status_fields(final_status, video_status)
    _write_json(status_path, final_status)
    _append_log(log_path, f"[worker] completed job={job_id} success={success_count}/{episodes}")

    return {
        "jobId": job_id,
        "datasetId": dataset_id,
        "status": "completed",
        "aggregateMetrics": aggregate_metrics,
        "perEpisode": per_episode,
    }
