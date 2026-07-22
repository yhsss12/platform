#!/usr/bin/env python3
"""Compare HDF5 demo replay under recorded / policy / none attachment modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.cable_threading.attachment_controller import (
    PolicyAttachmentController,
    RecordedAttachmentController,
    apply_attachment_flag,
)
from examples.cable_threading.hdf5_replay import reset_env_for_replay
from examples.cable_threading.utils import clip_action, make_env


def _metric_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    return bool(value)


def _summarize_rows(rows: list[dict], env) -> dict:
    if not rows:
        final = env._compute_metrics()
    else:
        final = rows[-1]
    thread_completion_max = (
        max(float(r.get("thread_completion", r.get("thread_completion_final", 0.0))) for r in rows)
        if rows
        else float(final.get("thread_completion", 0.0))
    )
    endpoint_errors = [
        float(r.get("endpoint_goal_error_final", r.get("endpoint_goal_error", np.nan))) for r in rows
    ]
    endpoint_errors = [x for x in endpoint_errors if np.isfinite(x)]
    endpoint_goal_error_min = min(endpoint_errors) if endpoint_errors else float(
        final.get("endpoint_goal_error_final", final.get("endpoint_goal_error", np.nan))
    )
    return {
        "final_success": bool(final.get("final_success", False)),
        "thread_completion_max": thread_completion_max,
        "endpoint_goal_error_min": endpoint_goal_error_min,
        "endpoint_goal_error_final": float(
            final.get("endpoint_goal_error_final", final.get("endpoint_goal_error", np.nan))
        ),
        "ever_threaded": any(_metric_bool(r.get("threaded_final", r.get("threaded", False))) for r in rows),
        "endpoint_region_final": bool(final.get("endpoint_region_final", False)),
        "straightened_final": bool(final.get("straightened_final", False)),
        "steps": len(rows),
    }


class PolicyAttachTracker(PolicyAttachmentController):
    def __init__(self, env):
        super().__init__(env)
        self.attach_count = 0
        self.detach_count = 0
        self.first_attach_step: int | None = None
        self._step = 0
        self.attach_steps: list[int] = []
        self.detach_steps: list[int] = []

    def pre_step(self, action, *, info=None):
        was = self._attached
        super().pre_step(action, info=info)
        if self._attached and not was:
            self.attach_count += 1
            self.attach_steps.append(self._step)
            if self.first_attach_step is None:
                self.first_attach_step = self._step
        elif was and not self._attached:
            self.detach_count += 1
            self.detach_steps.append(self._step)
        self._step += 1


class RecordedAttachTracker(RecordedAttachmentController):
    def __init__(self, env):
        super().__init__(env)
        self.attach_count = 0
        self.detach_count = 0
        self.first_attach_step: int | None = None
        self._step = 0
        self.attach_steps: list[int] = []
        self.detach_steps: list[int] = []

    def pre_step(self, action, *, info=None):
        was = self._prev
        super().pre_step(action, info=info)
        if self._prev and not was:
            self.attach_count += 1
            self.attach_steps.append(self._step)
            if self.first_attach_step is None:
                self.first_attach_step = self._step
        elif was and not self._prev:
            self.detach_count += 1
            self.detach_steps.append(self._step)
        self._step += 1


def replay_demo(hdf5_path: Path, demo: str, mode: str) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        grp = f["data"][demo]
        actions = np.asarray(grp["actions"], dtype=np.float32)
        attach = (
            np.asarray(grp["attachment_enabled"], dtype=bool)
            if "attachment_enabled" in grp
            else np.zeros(len(actions), dtype=bool)
        )
        meta_raw = grp.attrs.get("benchmark_episode_metadata", "{}")
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        meta = json.loads(meta_raw) if meta_raw else {}
        env_args_raw = f["data"].attrs.get("env_args", "{}")
        if isinstance(env_args_raw, bytes):
            env_args_raw = env_args_raw.decode("utf-8")
        env_args = json.loads(env_args_raw) if env_args_raw else {}

    seed = int(meta.get("seed", env_args.get("seed", 0)))
    env = make_env(
        robot=str(env_args.get("robot", "Panda")),
        cable_model=str(env_args.get("cable_model", "composite_cable")),
        grasp_mode=str(env_args.get("grasp_mode", "attachment")),
        difficulty=str(env_args.get("difficulty", "easy")),
        horizon=int(env_args.get("horizon", 600)),
        seed=seed,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    reset_env_for_replay(env)

    recorded_first_attach = int(np.argmax(attach)) if attach.any() else None
    recorded_attach_count = int(np.sum(np.diff(np.concatenate([[False], attach])) == 1))
    recorded_detach_count = int(np.sum(np.diff(np.concatenate([[False], attach])) == -1))

    rows = []
    ctrl = None
    if mode == "recorded":
        ctrl = RecordedAttachTracker(env)
        ctrl.reset(list(attach))
    elif mode == "policy":
        ctrl = PolicyAttachTracker(env)
        ctrl.reset()
    # mode == "none": no controller

    for t, action in enumerate(actions):
        if mode == "none":
            pass
        elif ctrl is not None:
            ctrl.pre_step(action, info=rows[-1] if rows else None)
        _, _, _, info = env.step(clip_action(env, action))
        rows.append(dict(info))

    summary = _summarize_rows(rows, env)
    result = {
        "demo": demo,
        "mode": mode,
        "seed": seed,
        **summary,
        "recorded_first_attach_step": recorded_first_attach,
        "recorded_attach_transitions": recorded_attach_count,
        "recorded_detach_transitions": recorded_detach_count,
    }
    if ctrl is not None:
        result.update(
            {
                "attach_transitions": ctrl.attach_count,
                "detach_transitions": ctrl.detach_count,
                "first_attach_step": ctrl.first_attach_step,
                "attach_steps": ctrl.attach_steps,
                "detach_steps": ctrl.detach_steps,
            }
        )
        if recorded_first_attach is not None and ctrl.first_attach_step is not None:
            result["first_attach_step_delta_vs_recorded"] = ctrl.first_attach_step - recorded_first_attach
        else:
            result["first_attach_step_delta_vs_recorded"] = None
    env.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--demos", nargs="+", default=["demo_0", "demo_1", "demo_2"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    all_results = []
    for demo in args.demos:
        for mode in ("recorded", "policy", "none"):
            row = replay_demo(hdf5_path, demo, mode)
            all_results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
