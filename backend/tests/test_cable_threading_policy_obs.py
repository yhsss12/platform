from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from examples.cable_threading.obs_schema import (  # noqa: E402
    POLICY_EVAL_IMAGE_CAMERA_NAMES,
    policy_eval_camera_kwargs,
    validate_policy_obs_schema,
)


class _FakePolicy:
    obs_keys = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "agentview_image",
        "robot0_eye_in_hand_image",
    ]
    ckpt_dict = {
        "shape_metadata": {
            "all_shapes": {
                "robot0_eef_pos": [3],
                "robot0_eef_quat": [4],
                "robot0_gripper_qpos": [2],
                "agentview_image": [256, 256, 3],
                "robot0_eye_in_hand_image": [256, 256, 3],
            }
        }
    }


def test_policy_eval_camera_kwargs_matches_training_cameras():
    kwargs = policy_eval_camera_kwargs()
    assert kwargs["camera_names"] == POLICY_EVAL_IMAGE_CAMERA_NAMES
    assert kwargs["camera_heights"] == 256
    assert kwargs["camera_widths"] == 256


def test_validate_policy_obs_schema_detects_missing_wrist_image():
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }
    result = validate_policy_obs_schema(_FakePolicy(), obs)
    assert result["valid"] is False
    assert "robot0_eye_in_hand_image" in result["missingKeys"]


def test_validate_policy_obs_schema_accepts_full_obs():
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }
    result = validate_policy_obs_schema(_FakePolicy(), obs)
    assert result["valid"] is True
    assert result["missingKeys"] == []
