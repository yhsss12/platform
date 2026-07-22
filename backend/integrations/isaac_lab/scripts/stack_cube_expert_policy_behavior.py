"""Behavior-level QA helpers for stack_cube_expert_policy (no Isaac import in core helpers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

HEIGHT_DIFF = 0.0468
GRASP_LIFT_MIN_Z = 0.015
PLACE_XY_TOLERANCE = 0.035
PLACE_Z_TOLERANCE = 0.012
STACK_XY_TOLERANCE = 0.045
STACK_Z_TOLERANCE = 0.008


@dataclass
class AttemptBehaviorRecord:
    attempt_index: int
    seed: int
    success: bool = False
    failed_state: Optional[str] = None
    failure_reason: Optional[str] = None
    cube_index: int = 0
    grasp_verified: bool = False
    lift_verified: bool = False
    place_verified: bool = False
    final_success_term: bool = False
    final_stack_error: Optional[float] = None
    cube_lifted_flags: list[bool] = field(default_factory=list)
    cube_placed_flags: list[bool] = field(default_factory=list)


@dataclass
class DemoBehaviorRecord:
    demo_index: int
    cube_lifted_flags: list[bool] = field(default_factory=list)
    cube_placed_flags: list[bool] = field(default_factory=list)
    final_cube_positions: Optional[list[list[float]]] = None
    final_stack_error: Optional[float] = None
    replay_success: bool = False
    final_success_term: bool = False
    grasp_verified: bool = False
    place_verified: bool = False
    failure_reason: Optional[str] = None


@dataclass
class ExpertBehaviorTracker:
    requested_demos: int
    recorded_demos: int = 0
    attempts: int = 0
    per_attempt: list[AttemptBehaviorRecord] = field(default_factory=list)
    per_demo: list[DemoBehaviorRecord] = field(default_factory=list)
    _current: Optional[AttemptBehaviorRecord] = None

    def begin_attempt(self, *, attempt_index: int, seed: int) -> None:
        self.attempts += 1
        self._current = AttemptBehaviorRecord(attempt_index=attempt_index, seed=seed)

    def record_grasp_verified(self) -> None:
        if self._current:
            self._current.grasp_verified = True

    def record_lift_verified(self) -> None:
        if self._current:
            self._current.lift_verified = True

    def record_place_verified(self) -> None:
        if self._current:
            self._current.place_verified = True

    def record_cube_lifted(self, ok: bool) -> None:
        if self._current:
            self._current.cube_lifted_flags.append(ok)

    def record_cube_placed(self, ok: bool) -> None:
        if self._current:
            self._current.cube_placed_flags.append(ok)

    def finish_attempt_failure(
        self,
        *,
        failed_state: str,
        failure_reason: str,
        cube_index: int,
        final_success_term: bool = False,
        final_stack_error: Optional[float] = None,
    ) -> None:
        if not self._current:
            return
        self._current.success = False
        self._current.failed_state = failed_state
        self._current.failure_reason = failure_reason
        self._current.cube_index = cube_index
        self._current.final_success_term = final_success_term
        self._current.final_stack_error = final_stack_error
        self.per_attempt.append(self._current)
        self._current = None

    def finish_attempt_success(
        self,
        *,
        demo_index: int,
        cube_lifted_flags: list[bool],
        cube_placed_flags: list[bool],
        final_cube_positions: list[list[float]],
        final_stack_error: float,
        final_success_term: bool,
    ) -> None:
        if self._current:
            self._current.success = True
            self._current.cube_lifted_flags = list(cube_lifted_flags)
            self._current.cube_placed_flags = list(cube_placed_flags)
            self._current.final_success_term = final_success_term
            self._current.final_stack_error = final_stack_error
            self._current.grasp_verified = all(cube_lifted_flags)
            self._current.place_verified = all(cube_placed_flags)
            self._current.lift_verified = all(cube_lifted_flags)
            self.per_attempt.append(self._current)
            self._current = None
        self.recorded_demos += 1
        self.per_demo.append(
            DemoBehaviorRecord(
                demo_index=demo_index,
                cube_lifted_flags=list(cube_lifted_flags),
                cube_placed_flags=list(cube_placed_flags),
                final_cube_positions=final_cube_positions,
                final_stack_error=final_stack_error,
                replay_success=True,
                final_success_term=final_success_term,
                grasp_verified=all(cube_lifted_flags),
                place_verified=all(cube_placed_flags),
            )
        )

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "requestedDemos": self.requested_demos,
            "recordedDemos": self.recorded_demos,
            "attempts": self.attempts,
            "perAttempt": [
                {
                    "attemptIndex": a.attempt_index,
                    "seed": a.seed,
                    "success": a.success,
                    "failedState": a.failed_state,
                    "failureReason": a.failure_reason,
                    "cubeIndex": a.cube_index,
                    "graspVerified": a.grasp_verified,
                    "liftVerified": a.lift_verified,
                    "placeVerified": a.place_verified,
                    "finalSuccessTerm": a.final_success_term,
                    "finalStackError": a.final_stack_error,
                    "cubeLiftedFlags": a.cube_lifted_flags,
                    "cubePlacedFlags": a.cube_placed_flags,
                }
                for a in self.per_attempt
            ],
            "perDemo": [
                {
                    "demoIndex": d.demo_index,
                    "cubeLiftedFlags": d.cube_lifted_flags,
                    "cubePlacedFlags": d.cube_placed_flags,
                    "finalCubePositions": d.final_cube_positions,
                    "finalStackError": d.final_stack_error,
                    "replaySuccess": d.replay_success,
                    "finalSuccessTerm": d.final_success_term,
                    "graspVerified": d.grasp_verified,
                    "placeVerified": d.place_verified,
                    "failureReason": d.failure_reason,
                }
                for d in self.per_demo
            ],
        }


def compute_stack_error(cube_positions: list[list[float]]) -> float:
    """Lower is better. Expect cube_2 on cube_1 and cube_3 on cube_2."""
    if len(cube_positions) < 3:
        return float("inf")
    c1, c2, c3 = cube_positions[0], cube_positions[1], cube_positions[2]
    err12_xy = ((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2) ** 0.5
    err23_xy = ((c3[0] - c2[0]) ** 2 + (c3[1] - c2[1]) ** 2) ** 0.5
    err12_z = abs((c2[2] - c1[2]) - HEIGHT_DIFF)
    err23_z = abs((c3[2] - c2[2]) - HEIGHT_DIFF)
    return err12_xy + err23_xy + err12_z + err23_z


def summarize_behavior_status(report: dict[str, Any]) -> tuple[str, list[str]]:
    """Return behaviorStatus and behaviorWarnings."""
    warnings: list[str] = []
    per_demo = report.get("perDemo") or []
    if not per_demo:
        return "failed", ["no_demo_behavior_records"]

    for demo in per_demo:
        idx = demo.get("demoIndex", 0)
        if not demo.get("replaySuccess", False):
            warnings.append(f"demo_{idx}:replay_failed")
        if not demo.get("finalSuccessTerm", False):
            warnings.append(f"demo_{idx}:success_term_false")
        lifted = demo.get("cubeLiftedFlags") or []
        placed = demo.get("cubePlacedFlags") or []
        if lifted and not all(lifted):
            warnings.append(f"demo_{idx}:grasp_not_lifted")
        if placed and not all(placed):
            warnings.append(f"demo_{idx}:place_error")
        stack_err = demo.get("finalStackError")
        if stack_err is not None and stack_err > 0.08:
            warnings.append(f"demo_{idx}:stack_error:{stack_err:.4f}")

    per_attempt = report.get("perAttempt") or []
    for att in per_attempt:
        reason = att.get("failureReason")
        if reason in {"grasp_not_lifted", "place_error", "grasp_failed"}:
            warnings.append(f"attempt_{att.get('attemptIndex')}: {reason}")

    if any("grasp_not_lifted" in w or "replay_failed" in w or "success_term_false" in w for w in warnings):
        return "failed", warnings
    if warnings:
        return "warning", warnings
    return "passed", warnings
