from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.replay_content_detection import detect_cable_threading_replay_content


def _write_hdf5(path: Path, demo_names: list[str]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for demo_name in demo_names:
            data.create_group(demo_name)


def test_detect_dataset_trajectory_replay_with_generation_preview(tmp_path: Path) -> None:
    job_root = tmp_path / "ct_gen_test"
    datasets = job_root / "datasets"
    videos = job_root / "videos"
    results = job_root / "results"
    datasets.mkdir(parents=True)
    videos.mkdir(parents=True)
    results.mkdir(parents=True)

    _write_hdf5(datasets / "dataset.hdf5", [f"demo_{idx}" for idx in range(9)])
    (videos / "generate.mp4").write_bytes(b"fake")
    (datasets / "dataset.manifest.json").write_text(
        json.dumps(
            {
                "successfulEpisodes": 9,
                "failedEpisodes": 1,
                "totalEpisodes": 10,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (results / "failures.json").write_text(
        json.dumps([{"episode": 2, "seed": 2, "failure_reason": "timeout"}]),
        encoding="utf-8",
    )

    content = detect_cable_threading_replay_content(
        job_root,
        job_id="ct_gen_test",
        metrics={"successfulEpisodes": 9, "episodes": 10, "failedEpisodes": 1},
    )

    assert content["replayContentKind"] == "dataset_trajectory_replay"
    assert content["hasHdf5Trajectories"] is True
    assert content["trajectoryCount"] == 9
    assert content["totalEpisodes"] == 10
    assert content["failedEpisodes"] == 1
    assert content["hasGenerationPreview"] is True
    assert content["primarySource"] == "dataset.hdf5"
    assert [tab["id"] for tab in content["tabs"]] == [
        "dataset_trajectory_replay",
        "generation_process_preview",
    ]
    assert content["trajectories"] == [f"demo_{idx}" for idx in range(9)]
    assert len(content["failureRecords"]) == 1
    failure = content["failureRecords"][0]
    assert failure["writtenToDataset"] is False
    assert failure["sourceEpisodeIndex"] == 2
    assert failure["displayEpisodeNumber"] == 3
    assert len(content["trajectoryRecords"]) == 9
    assert content["trajectoryRecords"][0]["demoName"] == "demo_0"
    assert content["trajectoryRecords"][0]["writtenToDataset"] is True


def test_detect_generation_process_preview_only(tmp_path: Path) -> None:
    job_root = tmp_path / "ct_gen_preview_only"
    videos = job_root / "videos"
    videos.mkdir(parents=True)
    (videos / "generate.mp4").write_bytes(b"fake")

    content = detect_cable_threading_replay_content(
        job_root,
        job_id="ct_gen_preview_only",
    )

    assert content["replayContentKind"] == "generation_process_preview"
    assert content["hasHdf5Trajectories"] is False
    assert content["hasGenerationPreview"] is True
    assert content["debug"]["hdf5_trajectory_replay_unavailable"] is True
    assert [tab["id"] for tab in content["tabs"]] == ["generation_process_preview"]


def test_detect_evaluation_replay(tmp_path: Path) -> None:
    job_root = tmp_path / "ct_eval_test"
    results = job_root / "results"
    videos = job_root / "videos"
    results.mkdir(parents=True)
    videos.mkdir(parents=True)
    (results / "aggregate_result.json").write_text("{}", encoding="utf-8")
    (videos / "eval.mp4").write_bytes(b"fake")

    content = detect_cable_threading_replay_content(
        job_root,
        job_id="ct_eval_test",
        is_eval_job=True,
    )

    assert content["replayContentKind"] == "evaluation_replay"
    assert content["hasEvaluationResult"] is True
