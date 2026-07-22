from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from examples.cable_threading.dp_lab.config import DpLabConfig  # noqa: E402
from examples.cable_threading.dp_lab.normalizer import LinearNormalizer  # noqa: E402
from examples.cable_threading.obs_schema import validate_diffusion_policy_obs_schema  # noqa: E402


def test_checkpoint_dict_writes_resolved_low_dim_dim():
    cfg = DpLabConfig()
    assert cfg.low_dim_dim is None
    payload = cfg.to_checkpoint_dict()
    assert payload["low_dim_dim"] == 9


def test_linear_normalizer_zero_variance_dims_use_identity():
    data = np.zeros((10, 7), dtype=np.float32)
    data[:, :3] = np.linspace(-1, 1, 10)[:, None]
    data[:, 6] = np.linspace(-1, 1, 10)
    norm = LinearNormalizer.fit(data)
    assert np.allclose(norm.scale[3:6], 1.0)
    assert np.allclose(norm.offset[3:6], 0.0)
    restored = norm.unnormalize(norm.normalize(data))
    assert np.allclose(restored[:, 3:6], 0.0)


def test_linear_normalizer_loads_legacy_extreme_scale():
    legacy = LinearNormalizer(
        scale=np.array([1.0, 1.0, 1.0, 2e6, 2e6, 2e6, 1.0], dtype=np.float32),
        offset=np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
    )
    sample = legacy.normalize(np.zeros((1, 7), dtype=np.float32))
    assert sample.shape == (1, 7)


class _FakeDpCfg:
    image_keys = ["agentview_image", "robot0_eye_in_hand_image"]
    low_dim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    action_dim = 7
    n_obs_steps = 2
    n_action_steps = 8
    image_size = 128
    low_dim_dim = None

    @property
    def resolved_low_dim_dim(self) -> int:
        return 9


def test_validate_diffusion_policy_obs_schema_detects_low_dim_mismatch():
    policy = MagicMock()
    policy.cfg = _FakeDpCfg()
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }
    result = validate_diffusion_policy_obs_schema(policy, obs)
    assert result["valid"] is True
    assert result["shapeWarnings"]
    assert result["expectedConfig"]["low_dim_dim"] == 9
    assert result["expectedConfig"]["n_action_steps"] == 8


def test_validate_diffusion_policy_obs_schema_rejects_bad_channels():
    policy = MagicMock()
    policy.cfg = _FakeDpCfg()
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 1), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }
    result = validate_diffusion_policy_obs_schema(policy, obs)
    assert result["valid"] is False
    assert result["shapeMismatchKeys"]


def test_adapt_diffusion_policy_decouples_horizon_and_action_steps(tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "dataset_dp.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((140, 64, 64, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eye_in_hand_image", data=np.zeros((140, 64, 64, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eef_pos", data=np.zeros((140, 3), dtype=np.float32))
        obs.create_dataset("robot0_eef_quat", data=np.zeros((140, 4), dtype=np.float32))
        obs.create_dataset("robot0_gripper_qpos", data=np.zeros((140, 2), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((140, 7), dtype=np.float32))

    from app.services.adapter_layer.training_adaptation_service import build_training_adaptation_plan

    plan = build_training_adaptation_plan(
        raw_manifest={"datasetId": "ds", "artifacts": {"hdf5": str(hdf5)}},
        model_type="diffusion_policy",
    )
    advanced = plan["modelAdaptation"]["advancedConfig"]
    assert advanced["horizon"] == 16
    assert advanced["n_action_steps"] == 8
    assert advanced["horizon"] != advanced["n_action_steps"]


def test_append_dp_advanced_args_includes_vision_and_diffusion_steps():
    from app.services import training_service as svc

    cmd = svc._build_train_command(
        backend="diffusion_policy",
        hdf5_path=Path("/tmp/dataset.hdf5"),
        out_dir=Path("/tmp/out"),
        train_config={
            "epochs": 2,
            "advancedEnabled": True,
            "modelParams": {
                "num_diffusion_steps": 12,
                "vision_encoder": "tiny_cnn",
                "image_size": 96,
            },
        },
    )
    assert "--num-diffusion-steps" in cmd
    assert cmd[cmd.index("--num-diffusion-steps") + 1] == "12"
    assert "--vision-encoder" in cmd
    assert cmd[cmd.index("--vision-encoder") + 1] == "tiny_cnn"
    assert "--image-size" in cmd
    assert cmd[cmd.index("--image-size") + 1] == "96"


def test_compute_eval_timeout_scales_with_episodes_and_horizon():
    from app.services.cable_threading_service import _compute_eval_timeout

    short = _compute_eval_timeout(episodes=1, horizon=50)
    long = _compute_eval_timeout(episodes=10, horizon=500)
    assert long > short
    assert short >= 600
