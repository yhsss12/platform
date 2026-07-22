from __future__ import annotations

from typing import Optional


def apply_progress_guard(
    *,
    existing_current: int,
    existing_total: int,
    existing_percent: int,
    desired_current: Optional[int],
    desired_total: Optional[int],
    allow_reset: bool,
    protect_total_regression: bool = True,
) -> tuple[int, int, int, bool, bool]:
    attempted_current = existing_current if desired_current is None else int(desired_current)
    attempted_total = existing_total if desired_total is None else int(desired_total)

    blocked_current = (not allow_reset) and attempted_current < existing_current
    next_current = existing_current if blocked_current else max(0, attempted_current)

    blocked_total = False
    next_total = attempted_total
    if not allow_reset and next_total <= 0 and existing_total > 0:
        next_total = existing_total
    if protect_total_regression and (not allow_reset) and existing_total > 0 and next_total > 0 and next_total < existing_total:
        next_total = existing_total
        blocked_total = True
    next_total = max(0, int(next_total))

    if next_total > 0:
        next_percent = min(100, int((next_current / next_total) * 100))
    else:
        next_percent = 0 if allow_reset else int(existing_percent or 0)
    return next_current, next_total, next_percent, blocked_current, blocked_total

