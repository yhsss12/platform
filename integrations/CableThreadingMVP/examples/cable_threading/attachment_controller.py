"""Attachment side-channel control for cable threading policy rollout."""

from __future__ import annotations

from typing import Any

import numpy as np


def _metric_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    return bool(value)


def apply_attachment_flag(env, want_attach: bool, prev_attach: bool) -> None:
    """Apply attachment_enabled transition (shared by replay and debug modes)."""
    if want_attach and not prev_attach:
        env.set_attachment_enabled(True)
        if getattr(env, "_attach_pending", False) and hasattr(env, "_activate_flex_attachment"):
            env._activate_flex_attachment()
    elif not want_attach and prev_attach:
        env.set_attachment_enabled(False)


def gripper_close_to_cable(env, max_distance: float | None = None) -> bool:
    max_dist = max_distance
    if max_dist is None:
        max_dist = float(getattr(env, "_attach_max_distance", 0.05))
    grip = np.asarray(env._get_gripper_site_position(), dtype=np.float32)
    cable_end = np.asarray(env._get_cable_end_pos(), dtype=np.float32)
    return float(np.linalg.norm(grip - cable_end)) < max_dist


class _AttachmentStatsMixin:
    """Track attach/detach transitions for eval diagnostics."""

    def _reset_stats(self) -> None:
        self.attach_count = 0
        self.detach_count = 0
        self.first_attach_step: int | None = None
        self._step_index = 0
        self._attached_steps = 0

    def _record_attachment_transition(self, was_attached: bool, now_attached: bool) -> None:
        if now_attached and not was_attached:
            self.attach_count += 1
            if self.first_attach_step is None:
                self.first_attach_step = self._step_index
        elif was_attached and not now_attached:
            self.detach_count += 1
        self._step_index += 1

    def _track_attachment_ratio(self, was_attached: bool, now_attached: bool) -> None:
        self._attached_steps += int(now_attached)

    def attachment_stats(self) -> dict[str, Any]:
        total = max(int(self._step_index), 1)
        ratio = float(self._attached_steps) / float(total)
        return {
            "attachment_mode": getattr(self, "mode", "unknown"),
            "attach_count": int(self.attach_count),
            "detach_count": int(self.detach_count),
            "attach_transitions": int(self.attach_count),
            "detach_transitions": int(self.detach_count),
            "first_attach_step": self.first_attach_step,
            "attachment_enabled_ratio": ratio,
        }


class PolicyAttachmentController(_AttachmentStatsMixin):
    """Infer attach/detach during policy rollout (no expert phase_log)."""

    def __init__(
        self,
        env,
        *,
        gripper_close_threshold: float = 0.0,
        thread_completion_detach: float = 0.95,
    ) -> None:
        self.env = env
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.thread_completion_detach = float(thread_completion_detach)
        self._attached = False
        self._prev_grip_closed = False
        self.mode = "policy"

    @property
    def grasp_mode(self) -> str:
        return str(getattr(self.env, "grasp_mode", "attachment"))

    def reset(self) -> None:
        self._attached = False
        self._prev_grip_closed = False
        self._reset_stats()
        if self.grasp_mode == "attachment":
            self.env.set_attachment_enabled(False)

    def pre_step(self, action: np.ndarray, *, info: dict[str, Any] | None = None) -> None:
        if self.grasp_mode != "attachment":
            return
        was_attached = self._attached
        action = np.asarray(action, dtype=np.float32)
        grip_closed = bool(self.env._is_gripper_closed(action))
        if not self._attached:
            if getattr(self.env, "_is_flex_cable", False):
                close_enough = gripper_close_to_cable(self.env) or bool(
                    getattr(self.env, "_attach_pending", False) and self.env._is_gripper_close_enough()
                )
                if grip_closed and close_enough:
                    apply_attachment_flag(self.env, True, False)
                    self._attached = True
            elif grip_closed and not self._prev_grip_closed:
                # composite / RMB attachment: weld snaps endpoint; match expert side-channel timing
                apply_attachment_flag(self.env, True, False)
                self._attached = True
        elif self._should_detach(action, info=info):
            apply_attachment_flag(self.env, False, True)
            self._attached = False
        self._prev_grip_closed = grip_closed
        self._record_attachment_transition(was_attached, self._attached)
        self._track_attachment_ratio(was_attached, self._attached)

    def _should_detach(self, action: np.ndarray, *, info: dict[str, Any] | None) -> bool:
        if self.env._is_gripper_closed(action):
            return False
        metrics = info or {}
        if not metrics and hasattr(self.env, "_compute_metrics"):
            try:
                metrics = self.env._compute_metrics()
            except Exception:
                metrics = {}
        tc = float(metrics.get("thread_completion", metrics.get("thread_completion_final", 0.0)))
        past_gap = _metric_bool(
            metrics.get("endpoint_past_gap_final", metrics.get("endpoint_past_gap", False))
        )
        return tc >= self.thread_completion_detach or past_gap


class RecordedAttachmentController(_AttachmentStatsMixin):
    """Debug-only: replay HDF5/NPZ recorded attachment_enabled schedule."""

    def __init__(self, env) -> None:
        self.env = env
        self.mode = "recorded"
        self._schedule: list[bool] = []
        self._index = 0
        self._prev = False

    def reset(self, schedule: list[bool] | None = None) -> None:
        self._schedule = [bool(x) for x in (schedule or [])]
        self._index = 0
        self._prev = False
        self._reset_stats()
        if str(getattr(self.env, "grasp_mode", "attachment")) == "attachment":
            self.env.set_attachment_enabled(False)

    def pre_step(self, action: np.ndarray, *, info: dict[str, Any] | None = None) -> None:
        if self.grasp_mode != "attachment":
            return
        was = self._prev
        want = self._schedule[self._index] if self._index < len(self._schedule) else self._prev
        apply_attachment_flag(self.env, want, self._prev)
        self._prev = want
        self._index += 1
        self._record_attachment_transition(was, self._prev)
        self._track_attachment_ratio(was, self._prev)

    @property
    def grasp_mode(self) -> str:
        return str(getattr(self.env, "grasp_mode", "attachment"))


def build_attachment_controller(env, *, replay_mode: str = "policy", attachment_schedule: list[bool] | None = None):
    if replay_mode == "none":
        return None
    if replay_mode == "recorded":
        ctrl = RecordedAttachmentController(env)
        ctrl.reset(attachment_schedule)
        return ctrl
    ctrl = PolicyAttachmentController(env)
    ctrl.reset()
    return ctrl
