import pytest


def test_progress_guard_blocks_regression_without_reset():
    from app.api import routes_jobs

    next_current, next_total, percent, blocked_c, blocked_t = routes_jobs._apply_progress_guard(
        existing_current=2,
        existing_total=50,
        existing_percent=4,
        desired_current=0,
        desired_total=50,
        allow_reset=False,
    )
    assert blocked_c or blocked_t
    assert next_current == 2
    assert next_total == 50
    assert percent == 4


def test_progress_guard_allows_explicit_reset():
    from app.api import routes_jobs

    next_current, next_total, percent, blocked_c, blocked_t = routes_jobs._apply_progress_guard(
        existing_current=2,
        existing_total=50,
        existing_percent=4,
        desired_current=0,
        desired_total=0,
        allow_reset=True,
    )
    assert not (blocked_c or blocked_t)
    assert next_current == 0
    assert next_total == 0
    assert percent == 0


def test_progress_guard_preserves_total_when_missing_or_zero():
    from app.api import routes_jobs

    next_current, next_total, percent, blocked_c, blocked_t = routes_jobs._apply_progress_guard(
        existing_current=2,
        existing_total=50,
        existing_percent=4,
        desired_current=3,
        desired_total=0,
        allow_reset=False,
    )
    assert not (blocked_c or blocked_t)
    assert next_current == 3
    assert next_total == 50
    assert percent == 6


def test_reset_payload_detection():
    from app.api import routes_jobs

    assert routes_jobs._is_explicit_progress_reset_payload(
        {"status": "PENDING", "completed_count": 0, "progress": 0, "mcap_path": ""}
    )
    assert not routes_jobs._is_explicit_progress_reset_payload(
        {"status": "RUNNING", "completed_count": 0, "progress": 0, "mcap_path": ""}
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
