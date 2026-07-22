from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import isaacsim_franka_pick_place_data_worker as worker
from app.services import isaacsim_franka_pick_place_service as isaac_svc
from app.services import workspace_dataset_service as ws_dataset_svc
from app.services import workspace_task_template_service as template_svc
from tests.isaacsim_franka_pick_place_test_helpers import (
    FORBIDDEN_UI_STRINGS,
    assert_no_forbidden_ui_strings,
    assert_video_path_task_scoped,
    configure_dataset_scan_roots,
    configure_isaacsim_job_root,
    resolve_replay_assets,
    run_completed_job,
)
from app.services import isaacsim_franka_pick_place_service as svc
from app.services.isaacsim_franka_pick_place_assets import contains_forbidden_video_path_hint


def test_cable_threading_template_still_loaded():
    templates = template_svc.list_task_templates()
    row = next(item for item in templates if item["id"] == "cable_threading_single_arm")

    assert row["name"] == "线缆穿杆"
    assert row["simulatorBackend"] == "mujoco"
    assert row["supportsDatasetGeneration"] is True
    assert row["registryTaskConfigId"] == "task_cable_threading_v1"


def test_cable_threading_dataset_scan_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cable_jobs_root = tmp_path / "cable" / "jobs"
    job_id = "ct_gen_20260617_120000_abcd"
    job_dir = cable_jobs_root / job_id
    datasets_dir = job_dir / "datasets"
    datasets_dir.mkdir(parents=True)
    manifest = {
        "num_successful": 7,
        "num_failed": 3,
        "created_at": "2026-06-17T12:00:00",
        "simulatorBackend": "mujoco",
    }
    (datasets_dir / "dataset.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    configure_dataset_scan_roots(
        tmp_path,
        monkeypatch,
        cable_jobs_root=cable_jobs_root,
        data_gen_jobs_root=tmp_path / "data_gen" / "jobs",
    )

    rows = ws_dataset_svc.list_datasets()
    row = next(item for item in rows if item["sourceJobId"] == job_id)

    assert row["simulatorBackend"] == "mujoco"
    assert row["successfulEpisodes"] == 7
    assert row["totalEpisodes"] == 10
    assert row["validTrajectories"] == 7
    assert row["generationRounds"] == 10
    assert row["taskDisplayName"] == "线缆穿杆"
    assert row["name"] == "线缆穿杆数据_20260617_1200"
    assert_no_forbidden_ui_strings(row)


def test_cable_threading_and_isaacsim_datasets_coexist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cable_jobs_root = tmp_path / "cable" / "jobs"
    cable_job_id = "ct_gen_20260617_130000_abcd"
    cable_job_dir = cable_jobs_root / cable_job_id
    cable_datasets_dir = cable_job_dir / "datasets"
    cable_datasets_dir.mkdir(parents=True)
    (cable_datasets_dir / "dataset.manifest.json").write_text(
        json.dumps(
            {
                "num_successful": 2,
                "num_failed": 0,
                "created_at": "2026-06-17T13:00:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    isaac_root = tmp_path / "data_generation"
    configure_isaacsim_job_root(isaac_root, monkeypatch)

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
    isaac_job_id = started["jobId"]

    configure_dataset_scan_roots(
        tmp_path,
        monkeypatch,
        cable_jobs_root=cable_jobs_root,
        data_gen_jobs_root=isaac_root / "jobs",
    )

    rows = ws_dataset_svc.list_datasets()
    cable_row = next(item for item in rows if item["sourceJobId"] == cable_job_id)
    isaac_row = next(item for item in rows if item["sourceJobId"] == isaac_job_id)

    assert cable_row["taskDisplayName"] == "线缆穿杆"
    assert isaac_row["taskDisplayName"] == "Franka 物体搬运"
    assert cable_row["simulatorBackend"] == "mujoco"
    assert isaac_row["simulatorBackend"] == "isaacsim"


def test_isaacsim_replay_resolver_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)
    assets = resolve_replay_assets(job_id, job_dir)

    assert assets["video_path"] is None
    assert assets["episode_manifest_path"].is_file()
    assert assets["metrics_path"].is_file()
    assert assets["episode_manifest"]["metrics_path"] == "episodes/ep_000001/metrics.json"
    assert assets["episode_manifest"]["trajectory_path"] == "episodes/ep_000001/trajectory.json"
    assert assets["metrics"]["success"] is True
    assert assets["metrics"]["duration_sec"] == 8.0
    assert_video_path_task_scoped(assets["status"].get("videoPath"))


def test_isaacsim_dataset_scan_keeps_available_status_without_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)
    configure_dataset_scan_roots(tmp_path, monkeypatch, data_gen_jobs_root=job_dir.parent)

    rows = ws_dataset_svc.list_datasets()
    row = next(item for item in rows if item["sourceJobId"] == job_id)

    assert row["status"] == "available"
    assert row["videoAvailable"] is False
    assert row["video_status"] == "pending"
    assert row["replayAvailable"] is False
    assert not (job_dir / "videos" / "ep_000001.mp4").is_file()


def test_isaacsim_status_api_does_not_expose_video_url_when_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    job_id, _job_dir = run_completed_job(tmp_path, monkeypatch)

    status = svc.get_job_status(job_id)
    assert status["videoExists"] is False
    assert status["videoUrl"] is None
    assert status["video_status"] == "pending"
    assert_video_path_task_scoped(status.get("videoPath"))


def test_isaacsim_resolver_rejects_cable_video_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)
    cable_video = job_dir / "videos" / "ep_000001.mp4"
    cable_video.parent.mkdir(parents=True, exist_ok=True)
    cable_video.write_bytes(b"fake")

    episode_manifest_path = job_dir / "episodes" / "ep_000001" / "episode_manifest.json"
    episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    episode_manifest["video_path"] = "runs/cable_threading/jobs/ct_gen_x/videos/demo.mp4"
    episode_manifest_path.write_text(json.dumps(episode_manifest, ensure_ascii=False), encoding="utf-8")

    video_path = svc.resolve_job_video_path(job_id, episode_id="ep_000001")
    status = svc.get_job_status(job_id)
    assert video_path is None
    assert status["videoExists"] is False
    assert status["videoStatus"] in {"invalid", "pending"}
    assert_video_path_task_scoped(status.get("videoPath"))


def test_dual_arm_video_path_not_returned_for_isaacsim_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dual_video = tmp_path / "dac_gen_demo.mp4"
    dual_video.write_bytes(b"fake")
    assert contains_forbidden_video_path_hint(str(dual_video)) is True

    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)
    episode_manifest_path = job_dir / "episodes" / "ep_000001" / "episode_manifest.json"
    episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    episode_manifest["video_path"] = str(dual_video)
    episode_manifest_path.write_text(json.dumps(episode_manifest, ensure_ascii=False), encoding="utf-8")

    assert svc.resolve_job_video_path(job_id) is None


def test_dataset_scan_responses_exclude_forbidden_ui_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    job_id, job_dir = run_completed_job(tmp_path, monkeypatch)
    configure_dataset_scan_roots(tmp_path, monkeypatch, data_gen_jobs_root=job_dir.parent)

    rows = ws_dataset_svc.list_datasets()
    isaac_row = next(item for item in rows if item["sourceJobId"] == job_id)

    cable_jobs_root = tmp_path / "cable" / "jobs"
    cable_job_id = "ct_gen_20260617_140000_abcd"
    cable_job_dir = cable_jobs_root / cable_job_id / "datasets"
    cable_job_dir.mkdir(parents=True)
    (cable_job_dir / "dataset.manifest.json").write_text(
        json.dumps({"num_successful": 1, "num_failed": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(ws_dataset_svc, "CABLE_THREADING_ROOT", cable_jobs_root)

    rows = ws_dataset_svc.list_datasets()
    cable_row = next(item for item in rows if item["sourceJobId"] == cable_job_id)

    for row in (isaac_row, cable_row):
        assert_no_forbidden_ui_strings(row)

    combined_text = json.dumps(rows, ensure_ascii=False).lower()
    for forbidden in FORBIDDEN_UI_STRINGS:
        assert forbidden.lower() not in combined_text
