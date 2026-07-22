"""Unit tests for cable_threading HDF5 writer expanded observation export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from robosuite.utils.dlo.hdf5_dataset import (  # noqa: E402
    HDF5_IMAGE_KEYS,
    HDF5_LOW_DIM_KEYS,
    HDF5_TASK_OBS_KEYS,
    POLICY_SCHEMAS,
    PREFERRED_POLICY_SCHEMAS,
    build_hdf5_manifest_fields,
    derive_joint_delta_actions,
    save_dataset_hdf5,
)


def _synthetic_raw_obs(t: int) -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((64, 64, 3), dtype=np.uint8) + t,
        "robot0_eye_in_hand_image": np.ones((64, 64, 3), dtype=np.uint8),
        "robot0_joint_pos": np.linspace(0, 1, 7, dtype=np.float64) + t * 0.01,
        "robot0_joint_vel": np.zeros(7, dtype=np.float64),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float64),
        "robot0_gripper_qvel": np.zeros(2, dtype=np.float64),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "attachment_state": np.float64(1.0 if t else 0.0),
        "cable_end_pos": np.zeros(3, dtype=np.float64),
        "pole_points": np.zeros(6, dtype=np.float64),
        "endpoint_goal_pos": np.zeros(3, dtype=np.float64),
        "cable_points": np.zeros(21, dtype=np.float64),
        "physical_grasp_state": np.zeros(3, dtype=np.float64),
        "object-state": np.zeros(10, dtype=np.float64),
    }


def _synthetic_trajectory(steps: int = 4) -> list[dict]:
    traj = []
    for t in range(steps):
        traj.append(
            {
                "raw_obs": _synthetic_raw_obs(t),
                "action": np.zeros(7, dtype=np.float32),
                "reward": 0.0,
                "done": t == steps - 1,
                "attachment_enabled": False,
            }
        )
    return traj


def test_save_dataset_hdf5_includes_joint_pos_and_task_fields(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    save_info = save_dataset_hdf5(
        hdf5_path,
        [_synthetic_trajectory()],
        image_keys=list(HDF5_IMAGE_KEYS),
        low_dim_keys=list(HDF5_LOW_DIM_KEYS),
        task_obs_keys=list(HDF5_TASK_OBS_KEYS),
        metadata={"env_name": "CableThreading", "grasp_mode": "attachment"},
    )

    assert "robot0_joint_pos" in save_info["available_obs_keys"]
    assert "robot0_joint_vel" in save_info["available_obs_keys"]
    assert "robot0_eef_pos" in save_info["available_obs_keys"]
    assert "attachment_state" in save_info["available_obs_keys"]
    assert "object-state" in save_info["available_obs_keys"]
    assert save_info["robot_state_available"] is True
    assert save_info["task_state_available"] is True

    with h5py.File(hdf5_path, "r") as handle:
        obs = handle["data"]["demo_0"]["obs"]
        assert "robot0_joint_pos" in obs
        assert obs["robot0_joint_pos"].shape == (4, 7)
        assert obs["robot0_gripper_qpos"].shape == (4, 2)
        assert obs["attachment_state"].shape == (4, 1)
        assert obs["agentview_image"].shape == (4, 64, 64, 3)

        attrs_obs_keys = json.loads(handle["data"].attrs["obs_keys"])
        assert "robot0_joint_pos" in attrs_obs_keys
        available = json.loads(handle["data"].attrs["available_obs_keys"])
        assert "robot0_joint_pos" in available


def test_build_hdf5_manifest_fields(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    save_info = save_dataset_hdf5(
        hdf5_path,
        [_synthetic_trajectory(3)],
        image_keys=list(HDF5_IMAGE_KEYS),
        low_dim_keys=list(HDF5_LOW_DIM_KEYS),
        task_obs_keys=list(HDF5_TASK_OBS_KEYS),
    )
    manifest_fields = build_hdf5_manifest_fields(save_info)

    assert "robot0_joint_pos" in manifest_fields["availableObservationKeys"]
    assert "actions" in manifest_fields["availableActionKeys"]
    assert "joint_actions" in manifest_fields["availableActionKeys"]
    assert "gripper_actions" in manifest_fields["availableActionKeys"]
    assert manifest_fields["joint_action_available"] is True
    assert manifest_fields["gripper_action_available"] is True
    assert manifest_fields["current_action_mode"] == "osc_pose_delta_eef"
    assert "joint_state_obs_eef_action" in manifest_fields["policySchemas"]
    assert "joint_state_obs_joint_action" in manifest_fields["policySchemas"]
    assert manifest_fields["policySchemas"]["joint_state_obs_joint_action"]["output"]["action_mode"] == "joint_delta_derived"
    assert manifest_fields["observationSchema"] == "cable_threading_joint_obs_v1"
    assert manifest_fields["actionSchema"] == "cable_threading_joint_delta_v1"
    assert manifest_fields["evalExecutor"] == "joint_position"
    assert manifest_fields["trainedActionMode"] == "joint_delta"


def test_derived_joint_actions_not_equal_to_raw_actions(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    traj = _synthetic_trajectory(5)
    # non-trivial actions distinct from joint deltas
    for i, step in enumerate(traj):
        step["action"] = np.array([0.5, -0.3, 0.1, 0.0, 0.0, 0.0, float(i % 2)], dtype=np.float32)
    save_dataset_hdf5(hdf5_path, [traj], image_keys=list(HDF5_IMAGE_KEYS), low_dim_keys=list(HDF5_LOW_DIM_KEYS))
    raw_obs = [step["raw_obs"] for step in traj]
    joint_deltas = derive_joint_delta_actions(raw_obs)
    with h5py.File(hdf5_path, "r") as handle:
        stored_joint = np.asarray(handle["data"]["demo_0"]["joint_actions"])
        stored_actions = np.asarray(handle["data"]["demo_0"]["actions"])
    assert stored_joint.shape == (5, 7)
    assert not np.allclose(stored_joint, stored_actions[:, :7])
    assert np.allclose(stored_joint, joint_deltas)


def test_standalone_joint_pos_schema_keys_present_in_export(tmp_path: Path):
    """Keys required by standalone joint_state_dp validation must exist in exported HDF5."""
    hdf5_path = tmp_path / "dataset.hdf5"
    save_dataset_hdf5(
        hdf5_path,
        [_synthetic_trajectory(2)],
        image_keys=list(HDF5_IMAGE_KEYS),
        low_dim_keys=list(HDF5_LOW_DIM_KEYS),
        task_obs_keys=list(HDF5_TASK_OBS_KEYS),
    )
    required = {"agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_gripper_qpos"}
    with h5py.File(hdf5_path, "r") as handle:
        obs_keys = set(handle["data"]["demo_0"]["obs"].keys())
    assert required.issubset(obs_keys)


def test_missing_keys_warn_without_crashing(tmp_path: Path, caplog):
    caplog.set_level("WARNING")
    partial_obs = {
        "robot0_joint_pos": np.zeros(7),
        "robot0_gripper_qpos": np.zeros(2),
    }
    traj = [
        {
            "raw_obs": partial_obs,
            "action": np.zeros(7, dtype=np.float32),
            "reward": 0.0,
            "done": True,
            "attachment_enabled": False,
        }
    ]
    hdf5_path = tmp_path / "partial.hdf5"
    save_info = save_dataset_hdf5(
        hdf5_path,
        [traj],
        image_keys=list(HDF5_IMAGE_KEYS),
        low_dim_keys=list(HDF5_LOW_DIM_KEYS),
        task_obs_keys=list(HDF5_TASK_OBS_KEYS),
    )
    assert "robot0_joint_pos" in save_info["available_obs_keys"]
    assert "robot0_eef_pos" in save_info["missing_obs_keys"]
    assert any("missing obs key" in rec.message for rec in caplog.records)
