"""Pure helpers extracted from stack_cube_scripted_expert for testing (no Isaac import)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


def smooth_ik_rel_action(
    raw_action: torch.Tensor,
    *,
    previous_action: torch.Tensor,
    enable_smoothing: bool,
    alpha: float,
    max_action_delta: float,
) -> torch.Tensor:
    raw_action = raw_action.reshape(-1)
    previous_action = previous_action.reshape(-1)
    raw_action = torch.clamp(raw_action, -1.0, 1.0)
    if not enable_smoothing:
        return raw_action.clone()
    arm_smoothed = alpha * raw_action[:6] + (1.0 - alpha) * previous_action[:6]
    arm_delta = torch.clamp(
        arm_smoothed - previous_action[:6],
        -max_action_delta,
        max_action_delta,
    )
    smoothed_arm = previous_action[:6] + arm_delta
    grip = raw_action[6:7].clone()
    if abs(float(raw_action[6] - previous_action[6])) < 0.25:
        grip = previous_action[6:7].clone()
    action = torch.cat([smoothed_arm, grip])
    return torch.clamp(action.reshape(-1), -1.0, 1.0)


@dataclass
class ExpertQualityTracker:
    requested_demos: int
    recorded_demos: int = 0
    attempts: int = 0
    success_attempts: int = 0
    failed_attempts: int = 0
    failure_reasons: Counter = field(default_factory=Counter)
    failure_log: list[dict[str, Any]] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    action_deltas: list[float] = field(default_factory=list)
    arm_action_deltas: list[float] = field(default_factory=list)
    action_norms: list[float] = field(default_factory=list)
    gripper_switch_count: int = 0
    state_timeout_count: int = 0
    quality_rejected_episodes: int = 0
    place_warnings: list[str] = field(default_factory=list)
    _last_gripper: Optional[float] = None
    _current_episode_steps: int = 0
    _current_episode_max_arm_delta: float = 0.0

    def begin_attempt(self) -> None:
        self.attempts += 1
        self._current_episode_steps = 0
        self._current_episode_max_arm_delta = 0.0

    def record_step_action(self, action: torch.Tensor, previous_action: Optional[torch.Tensor]) -> None:
        self._current_episode_steps += 1
        self.action_norms.append(float(torch.norm(action).item()))
        if previous_action is not None:
            delta = action - previous_action
            self.action_deltas.append(float(torch.norm(delta).item()))
            arm_delta = float(torch.norm(delta[:6]).item())
            self.arm_action_deltas.append(arm_delta)
            self._current_episode_max_arm_delta = max(self._current_episode_max_arm_delta, arm_delta)
        gripper = float(action[-1].item())
        if self._last_gripper is not None and abs(gripper - self._last_gripper) > 0.5:
            self.gripper_switch_count += 1
        self._last_gripper = gripper

    def finish_attempt_success(self) -> None:
        self.success_attempts += 1
        self.episode_lengths.append(self._current_episode_steps)

    def finish_attempt_failure(
        self,
        *,
        seed: int,
        failed_state_name: str,
        failure_reason: str,
        cube_index: int,
    ) -> None:
        self.failed_attempts += 1
        self.failure_reasons[failure_reason] += 1
        self.failure_log.append(
            {
                "attemptIndex": self.attempts,
                "seed": seed,
                "failedState": failed_state_name,
                "failureReason": failure_reason,
                "step": self._current_episode_steps,
                "cubeIndex": cube_index,
            }
        )

    def record_timeout(self) -> None:
        self.state_timeout_count += 1

    def record_demo_exported(self) -> None:
        self.recorded_demos += 1

    def record_place_warning(self, message: str) -> None:
        self.place_warnings.append(message)

    def record_quality_rejection(self, reason: str) -> None:
        self.quality_rejected_episodes += 1
        self.failure_reasons[reason] += 1

    def episode_passes_export_gate(self, max_arm_delta: float) -> bool:
        return self._current_episode_max_arm_delta <= max_arm_delta + 1e-6

    @property
    def max_arm_action_delta(self) -> Optional[float]:
        if not self.arm_action_deltas:
            return None
        return max(self.arm_action_deltas)

    @property
    def mean_arm_action_delta(self) -> Optional[float]:
        if not self.arm_action_deltas:
            return None
        return sum(self.arm_action_deltas) / len(self.arm_action_deltas)

    @property
    def mean_episode_length(self) -> Optional[float]:
        if not self.episode_lengths:
            return None
        return sum(self.episode_lengths) / len(self.episode_lengths)

    @property
    def mean_action_delta(self) -> Optional[float]:
        if not self.action_deltas:
            return None
        return sum(self.action_deltas) / len(self.action_deltas)

    @property
    def max_action_delta(self) -> Optional[float]:
        if not self.action_deltas:
            return None
        return max(self.action_deltas)

    @property
    def mean_action_norm(self) -> Optional[float]:
        if not self.action_norms:
            return None
        return sum(self.action_norms) / len(self.action_norms)

    @property
    def action_smoothness_score(self) -> Optional[float]:
        mean_delta = self.mean_action_delta
        if mean_delta is None:
            return None
        return 1.0 / (1.0 + mean_delta)

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "requestedDemos": self.requested_demos,
            "recordedDemos": self.recorded_demos,
            "attempts": self.attempts,
            "successAttempts": self.success_attempts,
            "failedAttempts": self.failed_attempts,
            "failureReasons": dict(self.failure_reasons),
            "meanEpisodeLength": self.mean_episode_length,
            "meanActionDelta": self.mean_action_delta,
            "maxActionDelta": self.max_action_delta,
            "meanArmActionDelta": self.mean_arm_action_delta,
            "maxArmActionDelta": self.max_arm_action_delta,
            "meanActionNorm": self.mean_action_norm,
            "gripperSwitchCount": self.gripper_switch_count,
            "stateTimeoutCount": self.state_timeout_count,
            "qualityRejectedEpisodes": self.quality_rejected_episodes,
            "actionSmoothnessScore": self.action_smoothness_score,
            "placeWarnings": list(self.place_warnings),
        }
