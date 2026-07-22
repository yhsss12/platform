"""Regression tests for joint-space DP platform E2E blockers."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from examples.cable_threading.utils import build_expert_env_make_kwargs  # noqa: E402
from robosuite.utils.dlo.hdf5_dataset import save_dataset_hdf5  # noqa: E402

from app.services.adapter_layer.hdf5_inspector import sum_low_dim_key_dims  # noqa: E402
from app.services.adapter_layer.training_adaptation_integration import build_dp_config_dict  # noqa: E402
from app.services.dp_schema_resolver import resolve_dp_eval_executor  # noqa: E402
from robosuite.utils.dlo.hdf5_platform_schema import ACTION_SCHEMA_JOINT, OBS_SCHEMA_JOINT  # noqa: E402


def _traj(steps: int = 4):
    rows = []
    for t in range(steps):
        rows.append(
            {
                "raw_obs": {
                    "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
                    "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
                    "robot0_joint_pos": np.linspace(0, 1, 7, dtype=np.float64) + t * 0.01,
                    "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float64),
                },
                "action": np.zeros(7, dtype=np.float32),
                "reward": 0.0,
                "done": t == steps - 1,
                "attachment_enabled": False,
            }
        )
    return rows


def test_build_expert_env_make_kwargs_avoids_duplicate_camera_names():
    live_config = {"camera": "agentview", "frame_height": 720, "frame_width": 1280}
    kwargs = build_expert_env_make_kwargs(live_enabled=True, live_config=live_config, hdf5_out=True)
    assert kwargs["camera_names"] == ["agentview", "robot0_eye_in_hand"]
    assert kwargs["camera_heights"] == 720
    assert kwargs["camera_widths"] == 1280

    hdf5_only = build_expert_env_make_kwargs(live_enabled=False, live_config=None, hdf5_out=True)
    assert hdf5_only["camera_names"] == ["agentview", "robot0_eye_in_hand"]
    assert "camera_heights" not in hdf5_only


def test_sum_low_dim_key_dims_joint_space_is_nine(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    save_dataset_hdf5(hdf5_path, [_traj()], metadata={"taskType": "cable_threading"})
    dim = sum_low_dim_key_dims(hdf5_path, ["robot0_joint_pos", "robot0_gripper_qpos"])
    assert dim == 9


def test_build_dp_config_dict_uses_low_dim_keys_not_state_dim(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    save_dataset_hdf5(hdf5_path, [_traj()], metadata={"taskType": "cable_threading"})
    profile = {
        "datasetId": "ds_joint",
        "taskType": "cable_threading",
        "stateDim": 516,
        "storageUri": str(hdf5_path),
        "cameraKeys": ["agentview_image", "robot0_eye_in_hand_image"],
        "observationKeys": [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_joint_pos",
            "robot0_gripper_qpos",
            "attachment_enabled",
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
                    "action_key": "joint_actions",
                    "gripper_action_key": "gripper_actions",
                    "action_mode": "joint_delta_derived",
                    "action_dim": 7,
                    "gripper_action_dim": 1,
                },
            }
        },
    }
    adaptation = {
        "inputConfig": {
            "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
            "camera_keys": ["agentview_image", "robot0_eye_in_hand_image"],
        },
        "architectureConfig": {"obs_horizon": 2, "action_horizon": 8, "pred_horizon": 16},
        "trainingConfig": {"epochs": 2, "batchSize": 8, "learningRate": 1e-4, "seed": 1},
        "advancedConfig": {},
    }
    dp_cfg = build_dp_config_dict(profile, adaptation)
    assert dp_cfg["low_dim_dim"] == 9
    assert dp_cfg["action_key"] == "joint_actions"
    assert dp_cfg["gripper_action_key"] == "gripper_actions"
    assert dp_cfg["action_dim"] == 8
    assert dp_cfg["controller_type"] == "JOINT_POSITION"
    assert dp_cfg["trained_action_mode"] == "joint_delta"
    assert dp_cfg["eval_executor"] == "joint_position"
    assert "attachment_enabled" not in dp_cfg["low_dim_keys"]


def test_build_dataset_profile_reads_top_level_hdf5_storage_uri(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    save_dataset_hdf5(hdf5_path, [_traj()], metadata={"taskType": "cable_threading"})
    from app.services.adapter_layer.dataset_profiler import build_dataset_profile

    profile = build_dataset_profile(
        {
            "datasetId": "ds_test",
            "taskType": "cable_threading",
            "hdf5": str(hdf5_path),
            "cameraKeys": ["agentview_image", "robot0_eye_in_hand_image"],
            "joint_action_available": True,
            "availableActionKeys": ["joint_actions", "gripper_actions", "actions"],
        }
    )
    assert profile.storageUri == str(hdf5_path)


def test_joint_dp_eval_executor_and_expert_legacy():
    joint = resolve_dp_eval_executor(
        policy="diffusion_policy",
        model_asset={
            "actionSchema": ACTION_SCHEMA_JOINT,
            "trainedActionMode": "joint_delta",
            "evalExecutor": "joint_position",
            "controllerType": "JOINT_POSITION",
        },
    )
    assert joint.uses_joint_executor()

    legacy = resolve_dp_eval_executor(
        policy="diffusion_policy",
        model_asset={"modelType": "diffusion_policy", "actionDim": 7},
    )
    assert legacy.eval_executor == "osc_pose"

    expert = resolve_dp_eval_executor(policy="scripted")
    assert expert.eval_executor == "osc_pose"


def test_cmd_expert_make_env_does_not_duplicate_camera_names():
    import examples.cable_threading.utils as ct_utils

    captured: dict = {}

    def _fake_make_env(**kwargs):
        captured.update(kwargs)
        return object()

    live_config = {"camera": "agentview", "frame_height": 720, "frame_width": 1280}
    with patch.object(ct_utils, "make_env", side_effect=_fake_make_env):
        ct_utils.make_env(
            robot="Panda",
            cable_model="composite_cable",
            difficulty="easy",
            horizon=600,
            seed=0,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            **build_expert_env_make_kwargs(
                live_enabled=True,
                live_config=live_config,
                hdf5_out=True,
            ),
        )
    assert captured["camera_names"] == ["agentview", "robot0_eye_in_hand"]
    assert "camera_heights" in captured
