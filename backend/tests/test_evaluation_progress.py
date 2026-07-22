from __future__ import annotations

from app.services.evaluation.evaluation_progress import resolve_evaluation_progress


def test_resolve_evaluation_progress_from_completed_episodes() -> None:
    info = resolve_evaluation_progress(
        status="running",
        metrics={"requestedEpisodes": 5, "completedEpisodes": 2},
    )
    assert info["requestedEpisodes"] == 5
    assert info["completedEpisodes"] == 2
    assert info["progressPercent"] == 40
    assert info["progressLabel"] == "2/5"


def test_resolve_evaluation_progress_from_current_episode() -> None:
    info = resolve_evaluation_progress(
        status="running",
        runtime_status={
            "status": "running",
            "phase": "episode_running",
            "currentEpisode": 3,
            "totalEpisodes": 5,
        },
    )
    assert info["completedEpisodes"] == 2
    assert info["progressLabel"] == "2/5"
    assert info["progressPercent"] == 40


def test_resolve_evaluation_progress_completed_job() -> None:
    info = resolve_evaluation_progress(
        status="completed",
        metrics={"totalEpisodes": 3, "completedEpisodes": 3},
    )
    assert info["progressPercent"] == 100
    assert info["progressLabel"] == "3/3"


def test_running_without_progress_defaults_to_zero_over_total() -> None:
    info = resolve_evaluation_progress(
        status="running",
        metrics={"requestedEpisodes": 5},
    )
    assert info["completedEpisodes"] == 0
    assert info["progressPercent"] == 0
    assert info["progressLabel"] == "0/5"


def test_extract_user_task_name_with_name_field() -> None:
    from app.services.evaluation.display_name import extract_user_evaluation_task_name

    assert extract_user_evaluation_task_name({"name": "线缆穿杆评测demo1"}) == "线缆穿杆评测demo1"
    assert extract_user_evaluation_task_name({"taskName": "线缆整理评测demo1"}) == "线缆整理评测demo1"
