from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.isaacsim_franka_pick_place_assets import (
    TASK_ID,
    VIDEO_STATUS_AVAILABLE,
    VIDEO_STATUS_FAILED,
    VIDEO_STATUS_PARTIAL,
    VIDEO_STATUS_PENDING,
    aggregate_dataset_video_status,
    contains_forbidden_video_path_hint,
    resolve_job_episode_video_path,
    resolve_pack_demo_video_path,
    validate_manifest_task_ids,
)


def test_forbidden_video_path_detects_cable_assets():
    assert contains_forbidden_video_path_hint("runs/cable_threading/jobs/ct_gen_x/videos/demo.mp4")
    assert contains_forbidden_video_path_hint("runs/dual_arm_cable/jobs/dac_gen_x/videos/generate.mp4")
    assert contains_forbidden_video_path_hint("panda_composite_cable_demo.mp4")
    assert not contains_forbidden_video_path_hint("videos/ep_000001.mp4")


def test_aggregate_dataset_video_status():
    available, status = aggregate_dataset_video_status(
        [{"video_available": True, "video_status": VIDEO_STATUS_AVAILABLE}]
    )
    assert available is True
    assert status == VIDEO_STATUS_AVAILABLE

    available, status = aggregate_dataset_video_status(
        [
            {"video_available": True, "video_status": VIDEO_STATUS_AVAILABLE},
            {"video_available": False, "video_status": VIDEO_STATUS_PENDING},
        ]
    )
    assert available is True
    assert status == VIDEO_STATUS_PARTIAL

    available, status = aggregate_dataset_video_status(
        [{"video_available": False, "video_status": VIDEO_STATUS_FAILED}]
    )
    assert available is False
    assert status == VIDEO_STATUS_FAILED

    available, status = aggregate_dataset_video_status(
        [{"video_available": False, "video_status": VIDEO_STATUS_PENDING}]
    )
    assert available is False
    assert status == VIDEO_STATUS_PENDING


def test_resolve_job_episode_video_respects_failed_status(tmp_path: Path):
    job_dir = tmp_path / "job"
    episode_id = "ep_000001"
    (job_dir / "episodes" / episode_id).mkdir(parents=True)
    (job_dir / "dataset_manifest.json").write_text(
        json.dumps({"task_id": TASK_ID, "source_task_id": TASK_ID}, ensure_ascii=False),
        encoding="utf-8",
    )
    (job_dir / "episodes" / episode_id / "episode_manifest.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "video_available": False,
                "video_status": VIDEO_STATUS_FAILED,
                "video_path": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    video_path, meta = resolve_job_episode_video_path(job_dir, episode_id)
    assert video_path is None
    assert meta["video_status"] == VIDEO_STATUS_FAILED


def test_pack_demo_video_not_available_without_manifest_entry():
    assert resolve_pack_demo_video_path("ep_000001") is None


def test_validate_manifest_task_ids_requires_source_task_id():
    ok, reason = validate_manifest_task_ids(
        dataset_manifest={"task_id": "isaacsim_franka_pick_place", "source_task_id": TASK_ID},
        episode_manifest={"task_id": TASK_ID},
    )
    assert ok is True
    assert reason is None

    bad, bad_reason = validate_manifest_task_ids(
        dataset_manifest={"task_id": "cable_threading"},
        episode_manifest={"task_id": "isaacsim_franka_pick_place"},
    )
    assert bad is False
    assert bad_reason is not None


def test_resolve_job_episode_video_rejects_cross_task_manifest(tmp_path: Path):
    job_dir = tmp_path / "job"
    episode_id = "ep_000001"
    (job_dir / "episodes" / episode_id).mkdir(parents=True)
    (job_dir / "videos").mkdir(parents=True)
    cable_video = job_dir / "videos" / "ep_000001.mp4"
    cable_video.write_bytes(b"fake")

    (job_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "source_task_id": TASK_ID,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (job_dir / "episodes" / episode_id / "episode_manifest.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "video_path": "runs/cable_threading/jobs/ct_gen_x/videos/demo.mp4",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    video_path, meta = resolve_job_episode_video_path(job_dir, episode_id)
    assert video_path is None
    assert meta["videoStatus"] == "invalid"
    assert meta["taskIdValidated"] is True


def test_resolve_job_episode_video_rejects_task_id_mismatch(tmp_path: Path):
    job_dir = tmp_path / "job"
    episode_id = "ep_000001"
    (job_dir / "episodes" / episode_id).mkdir(parents=True)
    (job_dir / "dataset_manifest.json").write_text(
        json.dumps({"task_id": TASK_ID, "source_task_id": TASK_ID}, ensure_ascii=False),
        encoding="utf-8",
    )
    (job_dir / "episodes" / episode_id / "episode_manifest.json").write_text(
        json.dumps({"task_id": "dual_arm_cable_manipulation", "video_path": None}, ensure_ascii=False),
        encoding="utf-8",
    )

    video_path, meta = resolve_job_episode_video_path(job_dir, episode_id)
    assert video_path is None
    assert meta["taskIdValidated"] is False


def test_resolve_job_episode_video_uses_manifest_relative_path(tmp_path: Path):
    job_dir = tmp_path / "job"
    episode_id = "ep_000001"
    (job_dir / "episodes" / episode_id).mkdir(parents=True)
    (job_dir / "videos").mkdir(parents=True)
    video = job_dir / "videos" / f"{episode_id}.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    (job_dir / "dataset_manifest.json").write_text(
        json.dumps({"task_id": TASK_ID, "source_task_id": TASK_ID}, ensure_ascii=False),
        encoding="utf-8",
    )
    (job_dir / "episodes" / episode_id / "episode_manifest.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "task_name": "Franka 物体搬运",
                "simulator": "Isaac Sim",
                "robot": "Franka Panda",
                "video_path": f"videos/{episode_id}.mp4",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    resolved, meta = resolve_job_episode_video_path(job_dir, episode_id)
    assert resolved == video.resolve()
    assert meta["videoStatus"] == "available"
    assert meta["taskIdValidated"] is True
