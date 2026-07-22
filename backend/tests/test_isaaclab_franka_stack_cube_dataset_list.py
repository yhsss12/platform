from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import workspace_dataset_service as ws_dataset_svc
from app.services.isaac_lab import isaac_dataset_service as isaac_dataset_svc


def _write_stack_cube_job(
    jobs_root: Path,
    job_id: str,
    *,
    successful_episodes: int = 2,
    total_episodes: int = 2,
    video_status: str = "pending",
    with_video: bool = False,
) -> Path:
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "datasets").mkdir()
    (job_dir / "episodes" / "ep_000001").mkdir(parents=True)
    (job_dir / "videos").mkdir(parents=True)

    manifest = {
        "datasetId": f"dataset_isaaclab_franka_stack_cube_{job_id[-17:]}",
        "jobId": job_id,
        "task_id": "isaaclab_franka_stack_cube",
        "taskType": "isaaclab_franka_stack_cube",
        "taskTemplateId": "task_isaaclab_franka_stack_cube_v1",
        "simulatorBackend": "isaac_lab",
        "sourceJobId": job_id,
        "episodes": total_episodes,
        "totalEpisodes": total_episodes,
        "successfulEpisodes": successful_episodes,
        "episode_manifests": [
            "episodes/ep_000001/episode_manifest.json",
            "episodes/ep_000002/episode_manifest.json",
        ][:total_episodes],
        "video_status": video_status,
        "videoStatus": video_status,
        "created_at": "2026-06-22T10:45:48+00:00",
    }
    (job_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (job_dir / "datasets" / "dataset.hdf5").write_bytes(b"hdf5")
    (job_dir / "episodes" / "ep_000001" / "episode_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "isaaclab_franka_stack_cube",
                "video_status": video_status,
            }
        ),
        encoding="utf-8",
    )
    if with_video:
        video_path = job_dir / "videos" / "ep_000001_state_replay.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    return job_dir


@pytest.fixture
def dataset_scan_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_gen_root = tmp_path / "data_generation" / "jobs"
    monkeypatch.setattr(ws_dataset_svc, "CABLE_THREADING_ROOT", tmp_path / "cable" / "jobs")
    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path / "dual" / "jobs")
    monkeypatch.setattr(ws_dataset_svc, "DATA_GENERATION_ROOT", data_gen_root)
    monkeypatch.setattr(isaac_dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", tmp_path / "registry.json")
    return data_gen_root


def test_stack_cube_trajectory_clamped_not_exceed_requested(dataset_scan_roots: Path):
    job_id = "data_gen_20260622_104548_a331"
    _write_stack_cube_job(
        dataset_scan_roots,
        job_id,
        successful_episodes=4,
        total_episodes=2,
    )

    row = next(item for item in ws_dataset_svc.list_datasets() if item["sourceJobId"] == job_id)

    assert row["successfulEpisodes"] == 2
    assert row["totalEpisodes"] == 2
    assert row["taskType"] == "isaaclab_franka_stack_cube"


def test_stack_cube_replay_available_when_video_file_exists(dataset_scan_roots: Path):
    job_id = "data_gen_20260622_124456_56e9"
    _write_stack_cube_job(
        dataset_scan_roots,
        job_id,
        video_status="pending",
        with_video=True,
    )

    row = next(item for item in ws_dataset_svc.list_datasets() if item["sourceJobId"] == job_id)

    assert row["replayAvailable"] is True
    assert row["videoPath"]
    assert row["hdf5Path"]
    assert row["runtimePath"]
    assert row["manifestPath"]


def test_stack_cube_replay_unavailable_without_video(dataset_scan_roots: Path):
    job_id = "data_gen_20260622_104548_b332"
    _write_stack_cube_job(
        dataset_scan_roots,
        job_id,
        video_status="pending",
        with_video=False,
    )

    row = next(item for item in ws_dataset_svc.list_datasets() if item["sourceJobId"] == job_id)

    assert row["replayAvailable"] is False


def test_delete_stack_cube_dataset_removes_from_scan(dataset_scan_roots: Path):
    job_id = "data_gen_20260622_104548_c333"
    _write_stack_cube_job(dataset_scan_roots, job_id)

    assert any(item["sourceJobId"] == job_id for item in ws_dataset_svc.list_datasets())

    result = ws_dataset_svc.delete_data_generation_dataset(job_id)

    assert result["ok"] is True
    assert result["runtimeDeleted"] is True
    assert not (dataset_scan_roots / job_id).exists()
    assert not any(item["sourceJobId"] == job_id for item in ws_dataset_svc.list_datasets())


def test_deleted_marker_filters_tombstoned_job(dataset_scan_roots: Path):
    job_id = "data_gen_20260622_104548_d444"
    job_dir = _write_stack_cube_job(dataset_scan_roots, job_id)
    (job_dir / ws_dataset_svc.DELETED_MARKER_NAME).write_text(
        json.dumps({"jobId": job_id}),
        encoding="utf-8",
    )

    assert not any(item["sourceJobId"] == job_id for item in ws_dataset_svc.list_datasets())
