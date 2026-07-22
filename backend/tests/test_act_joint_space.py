"""Unit tests for ACT joint-space schema integration (no training/eval runs)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_CABLE_MVP = _BACKEND.parent / "integrations" / "CableThreadingMVP"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from examples.cable_threading.act_lab.config import ActLabConfig  # noqa: E402
from examples.cable_threading.act_lab.dataset import ActDataset  # noqa: E402
from examples.cable_threading.act_eval_runtime import resolve_act_eval_runtime  # noqa: E402

from app.services.adapter_layer.training_adaptation_integration import build_act_config_dict  # noqa: E402
from app.services.policy_schema_resolver import (  # noqa: E402
    resolve_act_eval_executor,
    resolve_act_training_schema,
)
from robosuite.utils.dlo.hdf5_platform_schema import ACTION_SCHEMA_JOINT, OBS_SCHEMA_JOINT  # noqa: E402


def _joint_manifest_profile():
    return {
        "datasetId": "ds_joint_act",
        "taskType": "cable_threading",
        "stateDim": 516,
        "actionDim": 8,
        "cameraKeys": ["agentview_image", "robot0_eye_in_hand_image"],
        "observationKeys": [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_joint_pos",
            "robot0_gripper_qpos",
        ],
        "actionSchema": ACTION_SCHEMA_JOINT,
        "observationSchema": OBS_SCHEMA_JOINT,
        "joint_action_available": True,
        "availableActionKeys": ["actions", "joint_actions", "gripper_actions"],
        "policySchemas": {
            "joint_state_obs_joint_action": {
                "input": {
                    "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
                    "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
                },
                "output": {
                    "action_key": "actions",
                    "action_mode": "joint_delta_derived",
                    "controller_type": "JOINT_POSITION",
                    "action_dim": 8,
                },
            }
        },
    }


def _make_joint_hdf5(path: Path, action_dim: int = 8) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((20, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eye_in_hand_image", data=np.zeros((20, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("robot0_joint_pos", data=np.zeros((20, 7), dtype=np.float32))
        obs.create_dataset("robot0_gripper_qpos", data=np.zeros((20, 2), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((20, action_dim), dtype=np.float32))
        mask = handle.create_group("mask")
        mask.create_dataset("train", data=np.asarray([b"demo_0"]))


def test_act_joint_yaml_config_fields():
    cfg = ActLabConfig.from_yaml(
        _CABLE_MVP
        / "examples"
        / "cable_threading"
        / "act_configs"
        / "cable_threading_joint_obs_joint_action.yaml"
    )
    assert cfg.action_dim == 8
    assert cfg.action_key == "actions"
    assert cfg.controller_type == "JOINT_POSITION"
    assert cfg.eval_executor == "joint_position"
    assert cfg.low_dim_dim == 9
    assert "robot0_joint_pos" in cfg.low_dim_keys
    assert "robot0_eef_pos" not in cfg.low_dim_keys


def test_act_training_schema_from_manifest():
    schema = resolve_act_training_schema(_joint_manifest_profile())
    assert schema.policy_schema_id == "joint_state_obs_joint_action"
    assert schema.action_key == "actions"
    assert schema.action_dim == 8
    assert schema.controller_type == "JOINT_POSITION"
    assert schema.eval_executor == "joint_position"
    assert schema.low_dim_keys == ["robot0_joint_pos", "robot0_gripper_qpos"]


def test_build_act_config_dict_joint_space(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    _make_joint_hdf5(hdf5_path)
    profile = _joint_manifest_profile()
    profile["storageUri"] = str(hdf5_path)
    schema = resolve_act_training_schema(profile, hdf5_path=hdf5_path)
    adaptation = {
        "inputConfig": {
            "low_dim_keys": schema.low_dim_keys,
            "camera_keys": schema.image_keys,
        },
        "outputConfig": {
            "action_dim": schema.action_dim,
            "action_key": schema.action_key,
            "action_mode": schema.action_mode,
            "controller_type": schema.controller_type,
            "eval_executor": schema.eval_executor,
        },
        "architectureConfig": {"chunk_size": 20},
        "trainingConfig": {"epochs": 2, "batchSize": 8, "learningRate": 1e-4, "seed": 1},
        "advancedConfig": {},
    }
    act_cfg = build_act_config_dict(profile, adaptation, act_schema=schema)
    assert act_cfg["action_dim"] == 8
    assert act_cfg["action_key"] == "actions"
    assert act_cfg["low_dim_dim"] == 9
    assert act_cfg["eval_executor"] == "joint_position"
    assert act_cfg["controller_type"] == "JOINT_POSITION"


def test_act_dataset_loader_uses_schema_keys(tmp_path: Path):
    hdf5_path = tmp_path / "joint.hdf5"
    _make_joint_hdf5(hdf5_path)
    cfg = ActLabConfig(
        action_dim=8,
        action_key="actions",
        image_keys=["agentview_image", "robot0_eye_in_hand_image"],
        low_dim_keys=["robot0_joint_pos", "robot0_gripper_qpos"],
        image_size=8,
        chunk_size=4,
    )
    ds = ActDataset(hdf5_path, cfg, split="train", max_samples=4)
    sample = ds[0]
    assert sample["actions"].shape[-1] == 8
    assert sample["proprio"].numel() == 9


def test_act_dataset_action_dim_mismatch_raises(tmp_path: Path):
    hdf5_path = tmp_path / "bad.hdf5"
    _make_joint_hdf5(hdf5_path, action_dim=7)
    cfg = ActLabConfig(
        action_dim=8,
        action_key="actions",
        image_keys=["agentview_image", "robot0_eye_in_hand_image"],
        low_dim_keys=["robot0_joint_pos", "robot0_gripper_qpos"],
        image_size=8,
        chunk_size=4,
    )
    ds = ActDataset(hdf5_path, cfg, split="train", max_samples=1)
    with pytest.raises(ValueError, match="expected"):
        ds[0]


def test_act_checkpoint_metadata_joint_schema(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from examples.cable_threading.act_lab.model import ActPolicy

    cfg = ActLabConfig(
        action_dim=8,
        action_key="actions",
        controller_type="JOINT_POSITION",
        eval_executor="joint_position",
        action_mode="joint_delta_derived",
        low_dim_keys=["robot0_joint_pos", "robot0_gripper_qpos"],
        low_dim_dim=9,
        image_keys=["agentview_image", "robot0_eye_in_hand_image"],
        chunk_size=4,
    )
    model = ActPolicy(
        action_dim=8,
        chunk_size=4,
        state_dim=9,
        num_cameras=2,
        hidden_dim=64,
        latent_dim=8,
        enc_layers=2,
        nheads=4,
        dim_feedforward=128,
    )
    ckpt_path = tmp_path / "model_final.pt"
    train_config = cfg.to_dict()
    payload = {
        "state_dict": model.state_dict(),
        "backend": "act",
        "shape_meta": {
            "action_dim": 8,
            "action_key": "actions",
            "action_mode": "joint_delta_derived",
            "controller_type": "JOINT_POSITION",
            "eval_executor": "joint_position",
            "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
            "low_dim_dim": 9,
            "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "chunk_size": 4,
            "state_dim": 9,
        },
        "config": train_config,
        "train_config": train_config,
    }
    torch.save(payload, ckpt_path)

    runtime = resolve_act_eval_runtime(policy="act", checkpoint_path=ckpt_path)
    assert runtime["evalExecutor"] == "joint_position"
    assert runtime["controllerType"] == "JOINT_POSITION"

    spec = resolve_act_eval_executor(policy="act", checkpoint_path=ckpt_path)
    assert spec.eval_executor == "joint_position"
    assert spec.controller_type == "JOINT_POSITION"


def test_act_eval_resolver_legacy_eef_still_osc_pose(tmp_path: Path):
    torch = pytest.importorskip("torch")
    ckpt_path = tmp_path / "eef.pt"
    torch.save(
        {
            "state_dict": {},
            "backend": "act",
            "shape_meta": {
                "action_dim": 7,
                "low_dim_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                "controller_type": "OSC_POSE",
                "eval_executor": "osc_pose",
            },
            "config": {"action_dim": 7, "controller_type": "OSC_POSE"},
        },
        ckpt_path,
    )
    spec = resolve_act_eval_executor(policy="act", checkpoint_path=ckpt_path)
    assert spec.eval_executor == "osc_pose"
    assert spec.controller_type == "OSC_POSE"


def test_act_cannot_map_7d_osc_to_joint_executor(tmp_path: Path):
    torch = pytest.importorskip("torch")
    ckpt_path = tmp_path / "bad.pt"
    torch.save(
        {
            "state_dict": {},
            "backend": "act",
            "shape_meta": {
                "action_dim": 7,
                "action_mode": "joint_delta_derived",
                "controller_type": "JOINT_POSITION",
                "eval_executor": "joint_position",
            },
            "config": {},
        },
        ckpt_path,
    )
    with pytest.raises(ValueError, match="7D OSC"):
        resolve_act_eval_executor(policy="act", checkpoint_path=ckpt_path)
