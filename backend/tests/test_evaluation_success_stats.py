from __future__ import annotations

import json
from pathlib import Path

from app.services.evaluation.success_stats import resolve_success_stats


def test_resolve_success_stats_from_per_episode_ct_eval(tmp_path: Path) -> None:
    job_id = "ct_eval_20260625_124018_9ce1"
    job_root = tmp_path / "runs" / "cable_threading" / "jobs" / job_id
    results = job_root / "results"
    results.mkdir(parents=True)
    per_episode = [
        {"episodeIndex": 0, "success": True, "final_success": True},
        {"episodeIndex": 1, "success": True, "final_success": True},
        {"episodeIndex": 2, "success": False, "final_success": False},
        {"episodeIndex": 3, "success": True, "final_success": True},
    ]
    (results / "per_episode_results.json").write_text(json.dumps(per_episode), encoding="utf-8")
    (results / "aggregate_result.json").write_text(
        json.dumps({"total_episodes": 4, "success_episodes": 3}),
        encoding="utf-8",
    )

    stats = resolve_success_stats(
        job_id,
        runtime_path=str(job_root),
    )
    assert stats["available"] is True
    assert stats["display"] == "3/4"
    assert stats["successEpisodes"] == 3
    assert stats["totalEpisodes"] == 4
    assert stats["source"] == "per_episode_results.json"


def test_resolve_success_stats_dual_arm_summary() -> None:
    stats = resolve_success_stats(
        "eval_20260626_103509_1b68",
        aggregate_result={
            "summary": {"totalEpisodes": 3, "successEpisodes": 3, "successRate": 1.0},
        },
        per_episode_results={
            "episodes": [
                {"episodeSuccess": True},
                {"episodeSuccess": True},
                {"episodeSuccess": True},
            ]
        },
    )
    assert stats["display"] == "3/3"
    assert stats["source"] == "per_episode_results.json"


def test_running_without_success_shows_unavailable() -> None:
    stats = resolve_success_stats(
        "ct_eval_running_sample",
        status_json={"status": "running", "totalEpisodes": 10, "completedEpisodes": 2},
        metrics={"requestedEpisodes": 10, "completedEpisodes": 2},
    )
    assert stats["available"] is False
    assert stats["display"] == "-/-"


def test_only_success_rate_without_episode_count_unavailable() -> None:
    stats = resolve_success_stats(
        "isaac_eval_sample",
        aggregate_result={"successRate": 0.75, "meanReward": 1.0},
    )
    assert stats["available"] is False
    assert stats["display"] == "-/-"


def test_isaac_per_episode_zero_of_one(tmp_path: Path) -> None:
    job_id = "isaac_eval_20260617_193339_c9cf"
    job_root = tmp_path / "runs" / "evaluations" / "jobs" / job_id
    results = job_root / "results"
    results.mkdir(parents=True)
    (results / "aggregate_result.json").write_text(
        json.dumps({"episodeCount": 1, "successRate": 0.0}),
        encoding="utf-8",
    )
    (results / "per_episode_results.json").write_text(
        json.dumps({"episodes": [{"episodeIndex": 0, "success": False}]}),
        encoding="utf-8",
    )

    stats = resolve_success_stats(job_id, runtime_path=str(job_root))
    assert stats["display"] == "0/1"
    assert stats["available"] is True


def test_cached_summary_success_stats() -> None:
    stats = resolve_success_stats(
        "eval_cached",
        summary_json={
            "successStats": {
                "successEpisodes": 7,
                "totalEpisodes": 10,
                "display": "7/10",
                "available": True,
                "source": "aggregate_result",
            }
        },
    )
    assert stats["display"] == "7/10"
    assert stats["source"] == "aggregate_result"
