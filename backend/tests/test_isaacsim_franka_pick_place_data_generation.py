from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from app.api.routes_workspace_isaacsim_franka_pick_place import (
    generate_isaacsim_franka_pick_place_async,
    get_isaacsim_franka_pick_place_job_status,
)
from app.schemas.isaacsim_franka_pick_place import IsaacSimFrankaPickPlaceGenerateRequest
from app.services import workspace_dataset_service as ws_dataset_svc
from app.services import workspace_task_template_service as template_svc
from tests.isaacsim_franka_pick_place_test_helpers import (
    REQUIRED_JOB_REL_PATHS,
    assert_no_forbidden_ui_strings,
    assert_video_path_task_scoped,
    configure_dataset_scan_roots,
    configure_isaacsim_job_root,
    resolve_replay_assets,
    run_completed_job,
)


def test_task_template_loaded_by_workspace_service():
    templates = template_svc.list_task_templates()
    row = next(item for item in templates if item["id"] == "isaacsim_franka_pick_place")

    assert row["name"] == "Franka 物体搬运"
    assert row["simulatorBackend"] == "isaacsim"
    assert row["status"] == "integration_pending"
    assert row["supportsDatasetGeneration"] is False
    assert row["registryTaskConfigId"] == "task_isaacsim_franka_pick_place_v1"
    assert "FrankaPickPlace" in str(row.get("expertSource", ""))


def test_generate_async_api_creates_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    configure_isaacsim_job_root(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.services.isaacsim_franka_pick_place_service._spawn_worker",
        lambda *_args, **_kwargs: None,
    )

    response = asyncio.run(
        generate_isaacsim_franka_pick_place_async(
            IsaacSimFrankaPickPlaceGenerateRequest(
                taskId="isaacsim_franka_pick_place",
                episodes=1,
                seed=0,
                saveVideo=True,
                saveTrajectory=True,
                headless=True,
            ),
            object(),
        )
    )

    assert re.match(r"^data_gen_\d{8}_\d{6}_[a-f0-9]{4}$", response.jobId)
    assert response.taskId == "isaacsim_franka_pick_place"
    assert response.status == "running"
    assert response.message == "数据生成任务已启动"
    assert response.statusUrl is not None
    assert "generate-async" not in (response.statusUrl or "")

    job_dir = tmp_path / "jobs" / response.jobId
    assert (job_dir / "status.json").is_file()
    assert (job_dir / "metadata" / "job_config.json").is_file()
    assert_no_forbidden_ui_strings(response.model_dump())


def test_worker_demo_fallback_completes_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)

    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["taskId"] == "isaacsim_franka_pick_place"
    assert status["completedEpisodes"] == 1
    assert status["successEpisodes"] == 1

    log_text = (job_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "Isaac Sim runtime not detected, using packaged demo asset" in log_text

    aggregate = json.loads((job_dir / "results" / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregate["runtime_mode"] == "packaged_assets"
    assert aggregate["success_rate"] == 1.0


def test_job_output_directory_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _job_id, job_dir = run_completed_job(tmp_path, monkeypatch)

    for rel_path in REQUIRED_JOB_REL_PATHS:
        target = job_dir / rel_path
        assert target.is_file(), f"missing required artifact: {rel_path}"

    dataset_manifest = json.loads((job_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert dataset_manifest["source_task_id"] == "isaacsim_franka_pick_place"
    assert dataset_manifest["task_id"] == "isaacsim_franka_pick_place"
    assert dataset_manifest.get("video_available") is False
    assert dataset_manifest.get("video_status") == "pending"
    assert dataset_manifest.get("videoStatus") == "pending"

    episode_manifest = json.loads(
        (job_dir / "episodes" / "ep_000001" / "episode_manifest.json").read_text(encoding="utf-8")
    )
    assert episode_manifest["task_id"] == "isaacsim_franka_pick_place"
    assert episode_manifest["task_name"] == "Franka 物体搬运"
    assert episode_manifest["simulator"] == "Isaac Sim"
    assert episode_manifest["robot"] == "Franka Panda"
    assert episode_manifest.get("video_available") is False
    assert episode_manifest.get("video_path") is None
    assert episode_manifest.get("video_status") == "pending"
    assert episode_manifest.get("videoStatus") == "pending"


def test_workspace_dataset_service_lists_isaacsim_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)
    configure_dataset_scan_roots(tmp_path, monkeypatch, data_gen_jobs_root=job_dir.parent)

    rows = ws_dataset_svc.list_datasets()
    row = next(item for item in rows if item["sourceJobId"] == job_id)

    assert row["simulatorBackend"] == "isaacsim"
    assert row["taskType"] == "isaacsim_franka_pick_place"
    assert row["replayAvailable"] is False
    assert row["status"] == "available"
    assert row["videoAvailable"] is False
    assert row["video_status"] == "pending"
    assert row["videoStatus"] == "pending"
    assert row["episodeCount"] == 1
    assert "Franka 物体搬运" in row["name"]
    assert_no_forbidden_ui_strings(row)


def test_replay_resolver_finds_manifest_and_metrics_without_cross_task_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)
    assets = resolve_replay_assets(job_id, job_dir)

    assert assets["video_path"] is None
    assert assets["status"]["videoExists"] is False
    assert assets["status"]["videoStatus"] == "pending"
    assert assets["status"]["video_status"] == "pending"
    assert assets["status"]["taskIdValidated"] is True
    assert assets["episode_manifest"]["episode_id"] == "ep_000001"
    assert assets["episode_manifest"]["video_path"] is None
    assert assets["metrics"]["pick_success"] is True
    assert assets["metrics"]["place_success"] is True
    assert assets["metrics"]["controller_done"] is True
    assert assets["status"]["manifestPath"] is not None
    assert_video_path_task_scoped(assets["status"].get("videoPath"))


def test_status_api_response_has_no_forbidden_ui_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    job_id, _job_dir = run_completed_job(tmp_path, monkeypatch)

    response = asyncio.run(get_isaacsim_franka_pick_place_job_status(job_id, object()))
    assert response.status == "completed"
    assert response.videoExists is False
    assert response.videoStatus == "pending"
    assert response.video_status == "pending"
    assert response.taskIdValidated is True
    assert response.metrics.get("pick_success") is True
    assert_no_forbidden_ui_strings(response.model_dump())


def _mock_isaacsim_episode_with_video(
    job_dir: Path,
    episode_id: str,
    *,
    seed: int,
    headless: bool,
    save_video: bool,
    save_trajectory: bool,
    log_path: Path,
) -> dict:
    from app.services.isaacsim_franka_pick_place_data_worker import (
        _episode_manifest,
        _standard_episode_metrics,
        _standard_trajectory,
        _write_json,
    )
    from app.services.isaacsim_franka_pick_place_assets import VIDEO_STATUS_AVAILABLE

    ep_dir = job_dir / "episodes" / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    video_dest = job_dir / "videos" / f"{episode_id}.mp4"
    video_dest.parent.mkdir(parents=True, exist_ok=True)
    video_dest.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    metrics = _standard_episode_metrics(episode_id, success=True)
    manifest = _episode_manifest(
        episode_id,
        success=True,
        created_at="2026-06-17T13:00:00+00:00",
        video_available=True,
        video_status=VIDEO_STATUS_AVAILABLE,
    )
    _write_json(ep_dir / "metrics.json", metrics)
    _write_json(ep_dir / "episode_manifest.json", manifest)
    if save_trajectory:
        _write_json(ep_dir / "trajectory.json", _standard_trajectory(episode_id))
    return {
        "episode_id": episode_id,
        "success": True,
        "video_available": True,
        "video_status": VIDEO_STATUS_AVAILABLE,
        "metrics": metrics,
        "manifest": manifest,
    }


def test_isaacsim_available_path_records_video_and_video_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.services import isaacsim_franka_pick_place_service as isaac_svc
    from app.services import isaacsim_franka_pick_place_data_worker as worker

    configure_isaacsim_job_root(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "detect_isaacsim_available", lambda: True)
    monkeypatch.setattr(worker, "_run_isaacsim_episode", _mock_isaacsim_episode_with_video)

    def _sync_worker(job_id: str, job_dir: Path, config: dict) -> None:
        worker.execute_job(job_dir, job_id, config)

    monkeypatch.setattr(isaac_svc, "_spawn_worker", _sync_worker)
    started = isaac_svc.start_generate_async(
        episodes=1,
        seed=0,
        save_video=True,
        save_trajectory=True,
        headless=True,
    )
    job_id = started["jobId"]
    job_dir = tmp_path / "jobs" / job_id

    status = asyncio.run(get_isaacsim_franka_pick_place_job_status(job_id, object()))
    assert status.status == "completed"
    assert status.videoExists is True
    assert status.video_status == "available"
    assert status.videoUrl is not None
    assert (job_dir / "videos" / "ep_000001.mp4").is_file()
    assert_video_path_task_scoped(status.videoPath)

    dataset_manifest = json.loads((job_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert dataset_manifest["video_status"] == "available"
    assert dataset_manifest["video_available"] is True


def test_video_status_failed_keeps_dataset_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.services import isaacsim_franka_pick_place_data_worker as worker
    from app.services.isaacsim_franka_pick_place_assets import VIDEO_STATUS_FAILED

    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)

    episode_manifest_path = job_dir / "episodes" / "ep_000001" / "episode_manifest.json"
    episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    episode_manifest.update(
        {
            "video_available": False,
            "video_status": VIDEO_STATUS_FAILED,
            "videoStatus": VIDEO_STATUS_FAILED,
            "video_path": None,
        }
    )
    episode_manifest_path.write_text(json.dumps(episode_manifest, ensure_ascii=False), encoding="utf-8")

    dataset_manifest_path = job_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest.update(
        {
            "video_available": False,
            "video_status": VIDEO_STATUS_FAILED,
            "videoStatus": VIDEO_STATUS_FAILED,
        }
    )
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, ensure_ascii=False), encoding="utf-8")

    configure_dataset_scan_roots(tmp_path, monkeypatch, data_gen_jobs_root=job_dir.parent)
    rows = ws_dataset_svc.list_datasets()
    row = next(item for item in rows if item["sourceJobId"] == job_id)
    assert row["status"] == "available"
    assert row["video_status"] == VIDEO_STATUS_FAILED

    status = asyncio.run(get_isaacsim_franka_pick_place_job_status(job_id, object()))
    assert status.videoExists is False
    assert status.video_status == VIDEO_STATUS_FAILED
    assert status.videoUrl is None
    assert status.metrics.get("pick_success") is True
