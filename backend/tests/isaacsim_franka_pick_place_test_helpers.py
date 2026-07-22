from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.services import isaacsim_franka_pick_place_data_worker as worker
from app.services import isaacsim_franka_pick_place_service as svc
from app.services import workspace_dataset_service as ws_dataset_svc
from app.services.isaac_lab import isaac_dataset_service as isaac_dataset_svc
from app.services.isaacsim_franka_pick_place_assets import contains_forbidden_video_path_hint

FORBIDDEN_UI_STRINGS = (
    "占位",
    "示例数据",
    "mock",
    "fake",
    "未接入真实后端",
)

FORBIDDEN_VIDEO_HINTS = ("cable", "thread", "dual_arm", "dac_gen", "ct_gen")

REQUIRED_JOB_REL_PATHS = (
    "status.json",
    "dataset_manifest.json",
    "metadata/job_config.json",
    "results/aggregate_metrics.json",
    "results/per_episode_results.json",
    "episodes/ep_000001/episode_manifest.json",
    "episodes/ep_000001/metrics.json",
    "episodes/ep_000001/trajectory.json",
)


def assert_no_forbidden_ui_strings(payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False).lower()
    for forbidden in FORBIDDEN_UI_STRINGS:
        assert forbidden.lower() not in text, f"forbidden UI label found: {forbidden}"


def configure_isaacsim_job_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(worker, "detect_isaacsim_available", lambda: False)
    monkeypatch.setattr(svc, "record_workspace_job_start", lambda **_: None)
    monkeypatch.setattr(svc, "sync_workspace_job_from_runtime", lambda _: None)
    return tmp_path


def configure_dataset_scan_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    data_gen_jobs_root: Path | None = None,
    cable_jobs_root: Path | None = None,
) -> None:
    monkeypatch.setattr(
        ws_dataset_svc,
        "DATA_GENERATION_ROOT",
        data_gen_jobs_root if data_gen_jobs_root is not None else tmp_path / "data_gen" / "jobs",
    )
    monkeypatch.setattr(
        ws_dataset_svc,
        "CABLE_THREADING_ROOT",
        cable_jobs_root if cable_jobs_root is not None else tmp_path / "cable" / "jobs",
    )
    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path / "dual" / "jobs")
    monkeypatch.setattr(isaac_dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", tmp_path / "registry.json")


def run_completed_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    root = configure_isaacsim_job_root(tmp_path, monkeypatch)

    def _sync_worker(job_id: str, job_dir: Path, config: dict[str, Any]) -> None:
        worker.execute_job(job_dir, job_id, config)

    monkeypatch.setattr(svc, "_spawn_worker", _sync_worker)

    started = svc.start_generate_async(
        episodes=1,
        seed=0,
        save_video=True,
        save_trajectory=True,
        headless=True,
    )
    job_id = started["jobId"]
    job_dir = root / "jobs" / job_id
    return job_id, job_dir


def resolve_replay_assets(job_id: str, job_dir: Path) -> dict[str, Any]:
    status = svc.get_job_status(job_id)
    video_path = svc.resolve_job_video_path(job_id, episode_id="ep_000001")
    episode_manifest_path = job_dir / "episodes" / "ep_000001" / "episode_manifest.json"
    metrics_path = job_dir / "episodes" / "ep_000001" / "metrics.json"
    episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "status": status,
        "video_path": video_path,
        "episode_manifest_path": episode_manifest_path,
        "metrics_path": metrics_path,
        "episode_manifest": episode_manifest,
        "metrics": metrics,
    }


def assert_video_path_task_scoped(video_path: str | Path | None) -> None:
    if video_path is None:
        return
    lowered = str(video_path).replace("\\", "/").lower()
    for hint in FORBIDDEN_VIDEO_HINTS:
        assert hint not in lowered, f"cross-task video hint found: {hint} in {video_path}"
    assert contains_forbidden_video_path_hint(video_path) is False
