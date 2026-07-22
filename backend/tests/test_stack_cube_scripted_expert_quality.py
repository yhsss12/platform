"""Unit tests for scripted expert quality helpers (no Isaac Sim required)."""

from __future__ import annotations

from collections import Counter

import torch

from integrations.isaac_lab.scripts.stack_cube_scripted_expert_quality import (
    ExpertQualityTracker,
    smooth_ik_rel_action,
)


def test_smooth_action_limits_delta():
    prev = torch.zeros(7)
    raw = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    out = smooth_ik_rel_action(
        raw,
        previous_action=prev,
        enable_smoothing=True,
        alpha=0.6,
        max_action_delta=0.08,
    )
    delta = float(torch.norm(out[:6] - prev[:6]))
    assert delta <= 0.08 + 1e-6
    assert float(out[6]) == -1.0


def test_smooth_action_disabled_passes_raw():
    prev = torch.zeros(7)
    raw = torch.tensor([0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0])
    out = smooth_ik_rel_action(
        raw,
        previous_action=prev,
        enable_smoothing=False,
        alpha=0.6,
        max_action_delta=0.08,
    )
    assert torch.allclose(out, raw)


def test_quality_tracker_metrics():
    tracker = ExpertQualityTracker(requested_demos=1)
    tracker.begin_attempt()
    a0 = torch.zeros(7)
    a1 = torch.tensor([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    tracker.record_step_action(a0, None)
    tracker.record_step_action(a1, a0)
    assert tracker.episode_passes_export_gate(0.55) is False
    assert tracker.episode_passes_export_gate(0.65) is True
    tracker.record_quality_rejection("quality_rejected_arm_delta")
    metrics = tracker.to_metrics_dict()
    assert metrics["qualityRejectedEpisodes"] == 1
    assert metrics["maxArmActionDelta"] is not None
    tracker = ExpertQualityTracker(requested_demos=3)
    tracker.begin_attempt()
    a0 = torch.zeros(7)
    a1 = torch.tensor([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    tracker.record_step_action(a0, None)
    tracker.record_step_action(a1, a0)
    tracker.finish_attempt_failure(
        seed=0,
        failed_state_name="LIFT_OBJECT",
        failure_reason="grasp_failed",
        cube_index=0,
    )
    metrics = tracker.to_metrics_dict()
    assert metrics["attempts"] == 1
    assert metrics["failedAttempts"] == 1
    assert metrics["failureReasons"]["grasp_failed"] == 1
    assert metrics["meanActionDelta"] is not None
    assert metrics["actionSmoothnessScore"] is not None
