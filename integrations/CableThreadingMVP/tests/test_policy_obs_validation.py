import numpy as np

from examples.cable_threading.utils import (
    POLICY_EVAL_IMAGE_CAMERA_NAMES,
    POLICY_EVAL_IMAGE_OBS_KEYS,
    policy_eval_camera_kwargs,
    RobomimicPolicyAdapter,
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
    assert "robot0_eye_in_hand_image" in result["errorMessage"]


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
    for key in POLICY_EVAL_IMAGE_OBS_KEYS:
        assert key in result["expectedObsKeys"]


def test_validate_policy_obs_schema_accepts_resizable_chw_checkpoint_images():
    policy = _FakePolicy()
    policy.ckpt_dict = {
        "shape_metadata": {
            "all_shapes": {
                **_FakePolicy.ckpt_dict["shape_metadata"]["all_shapes"],
                "agentview_image": [3, 84, 84],
                "robot0_eye_in_hand_image": [3, 84, 84],
            }
        }
    }
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }

    result = validate_policy_obs_schema(policy, obs)

    assert result["valid"] is True
    assert {item["key"] for item in result["shapeWarnings"]} == set(
        POLICY_EVAL_IMAGE_OBS_KEYS
    )


def test_robomimic_adapter_resizes_images_before_policy_call():
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"rgb": ["agentview_image"], "low_dim": ["robot0_eef_pos", "attachment_state"]}}
    )
    received = {}

    class _CapturePolicy:
        def __call__(self, ob):
            received.update(ob)
            return np.zeros(7, dtype=np.float32)

    adapter = RobomimicPolicyAdapter.__new__(RobomimicPolicyAdapter)
    adapter.obs_keys = ["agentview_image", "robot0_eef_pos", "attachment_state"]
    adapter.obs_shapes = {
        "agentview_image": [3, 84, 84],
        "robot0_eef_pos": [3],
        "attachment_state": [1],
    }
    adapter.policy = _CapturePolicy()

    action = adapter.act(
        {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float64),
            "attachment_state": np.asarray(0.0),
        }
    )

    assert received["agentview_image"].shape == (3, 84, 84)
    assert received["agentview_image"].dtype == np.float32
    assert received["robot0_eef_pos"].dtype == np.float32
    assert received["attachment_state"].shape == (1,)
    assert action.shape == (7,)


def test_validate_policy_obs_schema_accepts_reshapeable_scalar():
    policy = _FakePolicy()
    policy.obs_keys = [*_FakePolicy.obs_keys, "attachment_state"]
    policy.ckpt_dict = {
        "shape_metadata": {
            "all_shapes": {
                **_FakePolicy.ckpt_dict["shape_metadata"]["all_shapes"],
                "attachment_state": [1],
            }
        }
    }
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "attachment_state": np.asarray(0.0),
    }

    result = validate_policy_obs_schema(policy, obs)

    assert result["valid"] is True
    assert result["shapeWarnings"] == [
        {
            "key": "attachment_state",
            "expected": (1,),
            "actual": (),
            "note": "will_reshape_at_inference",
        }
    ]
