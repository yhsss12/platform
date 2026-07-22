"""Unit tests for joint-space DP pipeline (no long simulation in CI)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_CABLE_MVP = _BACKEND.parent / "integrations" / "CableThreadingMVP"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
_VERIFICATION_TOOLS = _BACKEND / "tools" / "verification"
if str(_VERIFICATION_TOOLS) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_TOOLS))
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from examples.cable_threading.dp_lab.config import DpLabConfig  # noqa: E402
from examples.cable_threading.dp_lab.dataset import CableThreadingDpDataset  # noqa: E402
from examples.cable_threading.dp_lab.normalizer import DatasetStats, LinearNormalizer  # noqa: E402


@pytest.mark.integration
def test_joint_position_controller_action_dim():
    from joint_space_dp_utils import inspect_joint_position_controller

    info = inspect_joint_position_controller()
    assert info["controller_type"] == "JOINT_POSITION"
    assert info["action_dim"] == 8
    assert info["arm_action_dim"] == 7
    assert info["input_type"] == "delta"


def test_dp_config_joint_yaml_fields():
    cfg = DpLabConfig.from_yaml(
        _CABLE_MVP
        / "examples"
        / "cable_threading"
        / "dp_configs"
        / "cable_threading_joint_obs_joint_action.yaml"
    )
    assert cfg.action_dim == 8
    assert cfg.action_key == "actions"
    assert cfg.controller_type == "JOINT_POSITION"
    assert "robot0_joint_pos" in cfg.low_dim_keys
    assert "robot0_eef_pos" not in cfg.low_dim_keys


def test_dataset_loader_action_key_and_dim(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "joint.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((20, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eye_in_hand_image", data=np.zeros((20, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("robot0_joint_pos", data=np.zeros((20, 7), dtype=np.float32))
        obs.create_dataset("robot0_gripper_qpos", data=np.zeros((20, 2), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((20, 8), dtype=np.float32))
        demo.create_dataset("joint_actions", data=np.zeros((20, 7), dtype=np.float32))
        demo.create_dataset("gripper_actions", data=np.zeros((20, 1), dtype=np.float32))
        mask = handle.create_group("mask")
        mask.create_dataset("train", data=np.asarray([b"demo_0"]))

    cfg = DpLabConfig(
        action_dim=8,
        action_key="actions",
        low_dim_keys=["robot0_joint_pos", "robot0_gripper_qpos"],
        low_dim_dim=9,
        image_size=8,
        horizon=8,
        n_obs_steps=2,
    )
    stats = DatasetStats(
        action=LinearNormalizer.fit(np.zeros((20, 8), dtype=np.float32)),
        low_dim=LinearNormalizer.fit(np.zeros((20, 9), dtype=np.float32)),
    )
    ds = CableThreadingDpDataset(path, cfg, stats, split="train")
    sample = ds[0]
    assert sample["actions"].shape[-1] == 8


def test_dataset_loader_missing_action_key_raises(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "missing.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_joint_pos", data=np.zeros((20, 7), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((20, 7), dtype=np.float32))
        mask = handle.create_group("mask")
        mask.create_dataset("train", data=np.asarray([b"demo_0"]))

    cfg = DpLabConfig(action_dim=8, action_key="joint_actions")
    with pytest.raises(KeyError):
        from examples.cable_threading.dp_lab.dataset import compute_dataset_stats

        compute_dataset_stats(path, cfg)


def test_dataset_loader_action_dim_mismatch_raises(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bad_dim.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((20, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eye_in_hand_image", data=np.zeros((20, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("robot0_joint_pos", data=np.zeros((20, 7), dtype=np.float32))
        obs.create_dataset("robot0_gripper_qpos", data=np.zeros((20, 2), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((20, 7), dtype=np.float32))
        mask = handle.create_group("mask")
        mask.create_dataset("train", data=np.asarray([b"demo_0"]))

    cfg = DpLabConfig(action_dim=8, low_dim_dim=9, image_size=8, horizon=8, n_obs_steps=2)
    stats = DatasetStats(
        action=LinearNormalizer.fit(np.zeros((20, 7), dtype=np.float32)),
        low_dim=LinearNormalizer.fit(np.zeros((20, 9), dtype=np.float32)),
    )
    with pytest.raises(ValueError, match="expected"):
        CableThreadingDpDataset(path, cfg, stats, split="train")


def test_eef_baseline_default_action_key():
    cfg = DpLabConfig()
    assert cfg.action_key == "actions"
    assert cfg.action_dim == 7
    assert cfg.controller_type == "OSC_POSE"


def test_eval_controller_mismatch_raises():
    from joint_space_dp_utils import make_joint_position_env, validate_eval_controller_match

    import tempfile
    import torch

    cfg = DpLabConfig(action_dim=7, controller_type="OSC_POSE")
    payload = {"train_config": cfg.to_checkpoint_dict(), "state_dict": {}, "normalizer": {}}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(payload, tmp.name)
        ckpt_path = Path(tmp.name)
    env = make_joint_position_env(use_camera_obs=False, has_offscreen_renderer=False)
    try:
        with pytest.raises(ValueError, match="JOINT_POSITION"):
            validate_eval_controller_match(ckpt_path, env)
    finally:
        env.close()
        ckpt_path.unlink(missing_ok=True)
