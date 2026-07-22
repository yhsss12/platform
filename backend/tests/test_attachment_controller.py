"""Unit tests for cable threading attachment side-channel controllers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from examples.cable_threading.attachment_controller import (
    PolicyAttachmentController,
    RecordedAttachmentController,
)


def _mock_env(*, grasp_mode: str = "attachment", flex: bool = False):
    env = MagicMock()
    env.grasp_mode = grasp_mode
    env._is_flex_cable = flex
    env._attach_pending = False
    env._is_gripper_closed = MagicMock(return_value=False)
    env._is_gripper_close_enough = MagicMock(return_value=False)
    env._compute_metrics = MagicMock(
        return_value={
            "thread_completion": 0.0,
            "endpoint_past_gap_final": False,
        }
    )
    return env


def test_policy_controller_tracks_attach_stats_on_gripper_close_edge():
    env = _mock_env()
    ctrl = PolicyAttachmentController(env)
    ctrl.reset()
    assert ctrl.attachment_stats()["attach_transitions"] == 0

    env._is_gripper_closed.return_value = True
    ctrl.pre_step(np.zeros(7, dtype=np.float32))
    stats = ctrl.attachment_stats()
    assert stats["attachment_mode"] == "policy"
    assert stats["attach_transitions"] == 1
    assert stats["first_attach_step"] == 0
    env.set_attachment_enabled.assert_called_with(True)


def test_recorded_controller_replays_schedule():
    env = _mock_env()
    ctrl = RecordedAttachmentController(env)
    schedule = [False, False, True, True, False]
    ctrl.reset(schedule)
    for _ in schedule:
        ctrl.pre_step(np.zeros(7, dtype=np.float32))
    stats = ctrl.attachment_stats()
    assert stats["attachment_mode"] == "recorded"
    assert stats["attach_transitions"] == 1
    assert stats["detach_transitions"] == 1
    assert stats["first_attach_step"] == 2


def test_policy_controller_skips_when_not_attachment_mode():
    env = _mock_env(grasp_mode="physical")
    ctrl = PolicyAttachmentController(env)
    ctrl.reset()
    env._is_gripper_closed.return_value = True
    ctrl.pre_step(np.zeros(7, dtype=np.float32))
    assert ctrl.attachment_stats()["attach_transitions"] == 0
    env.set_attachment_enabled.assert_not_called()


@pytest.mark.integration
def test_rollout_policy_episode_includes_attachment_stats():
    pytest.importorskip("robosuite")
    from examples.cable_threading.hdf5_replay import load_npz_trajectories, reset_env_for_replay
    from examples.cable_threading.utils import clip_action, make_env

    npz = (
        Path(__file__).resolve().parents[2]
        / "runs"
        / "cable_threading"
        / "jobs"
        / "ct_gen_20260618_095819_8aa6"
        / "datasets"
        / "dataset.npz"
    )
    if not npz.is_file():
        pytest.skip("cable threading NPZ not present")

    trajectories, episode_metadata, metadata = load_npz_trajectories(npz)
    traj = trajectories[0]
    seed = int(episode_metadata[0].get("seed", 0))
    env = make_env(
        robot=str(metadata.get("robot", "Panda")),
        cable_model=str(metadata.get("cable_model", "composite_cable")),
        grasp_mode="attachment",
        difficulty=str(metadata.get("difficulty", "easy")),
        horizon=int(metadata.get("horizon", 600)),
        seed=seed,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    reset_env_for_replay(env)

    class DemoReplayPolicy:
        def __init__(self, actions):
            self.actions = actions
            self.i = 0

        def reset(self):
            self.i = 0

        def act(self, obs):
            idx = min(self.i, len(self.actions) - 1)
            action = self.actions[idx]
            self.i += 1
            return action

    from examples.cable_threading.utils import rollout_policy_episode

    policy = DemoReplayPolicy([step["action"] for step in traj])
    summary, _ = rollout_policy_episode(
        env,
        policy,
        episode_index=0,
        seed=seed,
        policy_name="demo_replay",
        attachment_mode="policy",
    )
    env.close()
    assert "attach_transitions" in summary
    assert summary["attachment_mode"] == "policy"
    assert summary["attach_transitions"] >= 1
    assert summary["first_attach_step"] is not None
