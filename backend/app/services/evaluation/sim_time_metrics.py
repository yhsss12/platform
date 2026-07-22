"""Simulation-time metric helpers (stepCount / controlFrequencyHz)."""

from __future__ import annotations

from typing import Any


def enrich_run_metrics_sim_time(run_metrics: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(run_metrics, dict):
        return {}
    enriched = dict(run_metrics)
    if enriched.get("meanSimTimeSec") is not None:
        return enriched
    mean_steps = enriched.get("meanSteps")
    control_hz = enriched.get("controlFrequencyHz")
    if isinstance(mean_steps, (int, float)) and isinstance(control_hz, (int, float)) and control_hz > 0:
        enriched["meanSimTimeSec"] = round(float(mean_steps) / float(control_hz), 4)
    return enriched


def compute_episode_sim_time_sec(
    *,
    step_count: Any = None,
    control_frequency_hz: Any = None,
    existing_sim_time: Any = None,
) -> float | None:
    if isinstance(existing_sim_time, (int, float)) and not isinstance(existing_sim_time, bool):
        return round(float(existing_sim_time), 4)
    steps = _as_positive_number(step_count)
    freq = _as_positive_number(control_frequency_hz)
    if steps is None or freq is None:
        return None
    return round(steps / freq, 4)


def _as_positive_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number >= 0 else None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        number = float(text)
        return number if number >= 0 else None
    except ValueError:
        return None
