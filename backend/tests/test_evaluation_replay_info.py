from __future__ import annotations

import json
from pathlib import Path

from app.services.evaluation_replay_info import (
    build_cable_threading_replay_info,
    build_evaluation_replay_info,
    resolve_episode_video_path,
)


def test_representative_video_with_two_completed_episodes(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    videos = job_root / "videos"
    videos.mkdir(parents=True)
    (videos / "eval.mp4").write_bytes(b"fake")
    (videos / "eval.browser.mp4").write_bytes(b"fake")

    results_dir = job_root / "results"
    results_dir.mkdir(parents=True)
    per_episode = [
        {"episode": 0, "success": True, "final_success": True},
        {"episode": 1, "success": True, "final_success": True},
    ]
    (results_dir / "per_episode_results.json").write_text(json.dumps(per_episode), encoding="utf-8")

    meta = job_root / "metadata"
    meta.mkdir(parents=True)
    (meta / "evaluation_context.json").write_text(
        json.dumps({"config": {"episodes": 2, "horizon": 300}}),
        encoding="utf-8",
    )

    live = {"episodes": 2, "status": "completed", "evalVideoExists": True}
    results_data = {"num_episodes": 2, "episodes": per_episode}
    aggregate = {"total_episodes": 2, "success_episodes": 2, "failure_count": 0}

    info = build_cable_threading_replay_info(
        "ct_eval_test_001",
        job_root,
        live=live,
        results_data=results_data,
        aggregate_file=aggregate,
        status_value="completed",
    )

    assert info["requestedEpisodes"] == 2
    assert info["completedEpisodes"] == 2
    assert info["successfulEpisodes"] == 2
    assert info["failedEpisodes"] == 0
    assert info["recordedVideoCount"] == 1
    assert info["isRepresentativeVideo"] is True
    assert len(info["replayUris"]) == 1
    assert info["replayUris"][0]["label"] == "代表性回放"
    assert "代表性回放" in (info["warning"] or "")


def test_multiple_episode_videos(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    videos = job_root / "videos"
    videos.mkdir(parents=True)
    (videos / "episode_001.mp4").write_bytes(b"fake")
    (videos / "episode_002.mp4").write_bytes(b"fake")

    info = build_cable_threading_replay_info(
        "ct_eval_test_002",
        job_root,
        live={"episodes": 2, "status": "completed"},
        results_data={"num_episodes": 2},
        aggregate_file={"total_episodes": 2, "success_episodes": 1, "failure_count": 1},
        status_value="completed",
    )

    assert info["recordedVideoCount"] == 2
    assert len(info["replayUris"]) == 2
    assert info["replayUris"][0]["label"] == "第 1 轮轨迹"
    assert info["replayUris"][1]["label"] == "第 2 轮轨迹"
    assert info["isRepresentativeVideo"] is False


def test_dual_arm_zero_based_episode_videos(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    videos = job_root / "videos"
    videos.mkdir(parents=True)
    (videos / "episode_00.mp4").write_bytes(b"fake")
    (videos / "episode_01.mp4").write_bytes(b"fake")
    (videos / "episode_02.mp4").write_bytes(b"fake")

    info = build_evaluation_replay_info(
        "eval_test_001",
        job_root,
        live={"totalEpisodes": 3, "status": "failed"},
        aggregate_file={"totalEpisodes": 3, "successEpisodes": 2, "failureCount": 1},
        status_value="failed",
    )

    assert info["recordedVideoCount"] == 3
    assert len(info["replayUris"]) == 3
    assert info["replayUris"][0]["episodeIndex"] == 1
    assert info["replayUris"][0]["uri"].endswith("?episode=0")
    assert info["replayUris"][2]["label"] == "第 3 轮轨迹"
    assert info["isRepresentativeVideo"] is False


def test_resolve_episode_video_path_zero_based(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    videos = job_root / "videos"
    videos.mkdir(parents=True)
    (videos / "episode_01.mp4").write_bytes(b"fake")

    assert resolve_episode_video_path(job_root, episode=1) == videos / "episode_01.mp4"
