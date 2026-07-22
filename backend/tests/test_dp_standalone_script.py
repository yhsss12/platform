from __future__ import annotations

import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "tools" / "verification"
MODULE_PATH = SCRIPTS_DIR / "test_dp_cable_threading_standalone.py"
UTILS_PATH = SCRIPTS_DIR / "standalone_dp_eval_utils.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module(MODULE_PATH, "dp_standalone_runner")
utils = _load_module(UTILS_PATH, "standalone_dp_eval_utils")


def _train_config() -> dict:
    return dict(utils.STANDALONE_DP_OBS_SCHEMA) | {
        "backend": "diffusion_policy",
        "image_size": 128,
        "vision_encoder": "resnet18",
        "n_obs_steps": 2,
        "n_action_steps": 8,
    }


def _save_checkpoint(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "model.pt"
    buffer = BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    return path


def _valid_payload(**overrides) -> dict:
    train_config = _train_config()
    train_config.update(overrides.get("train_config") or {})
    payload = {
        "state_dict": {"layer.weight": torch.zeros(2, 2)},
        "normalizer": {
            "action": {"scale": [1.0] * 7, "offset": [0.0] * 7},
            "low_dim": {"scale": [1.0] * 9, "offset": [0.0] * 9},
        },
        "train_config": train_config,
    }
    payload.update({k: v for k, v in overrides.items() if k != "train_config"})
    return payload


def test_parse_args_eval_only_defaults():
    args = mod.parse_args(
        [
            "--mode",
            "eval-only",
            "--checkpoint",
            "/tmp/model_final.pt",
        ]
    )
    assert args.mode == "eval-only"
    assert args.checkpoint == "/tmp/model_final.pt"


def test_inspect_checkpoint_expects_joint_pos_schema(tmp_path: Path):
    ckpt = _save_checkpoint(tmp_path, _valid_payload())
    result = mod.inspect_checkpoint(ckpt)
    assert result["ok"] is True
    assert result["train_config"]["low_dim_keys"] == ["robot0_joint_pos", "robot0_gripper_qpos"]


def test_inspect_checkpoint_rejects_eef_schema(tmp_path: Path):
    ckpt = _save_checkpoint(
        tmp_path,
        _valid_payload(
            train_config={
                "low_dim_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            }
        ),
    )
    result = mod.inspect_checkpoint(ckpt)
    assert result["ok"] is False
    assert any("low_dim_keys" in err for err in result["errors"])


def test_adapt_raw_obs_joint_pos_schema_ignores_extra_fields():
    raw_obs = {
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "object-state": np.zeros(10, dtype=np.float32),
        "agentview_image": np.zeros((64, 64, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((64, 64, 3), dtype=np.uint8),
    }
    adapted, warnings, diag = utils.build_standalone_dp_observation(raw_obs, _train_config())
    concat = np.concatenate(
        [adapted["robot0_joint_pos"].reshape(-1), adapted["robot0_gripper_qpos"].reshape(-1)]
    )
    assert concat.shape == (9,)
    assert diag["low_dim_concat_shape"] == [9]
    assert "robot0_eef_pos" not in adapted
    assert warnings == []


def test_adapt_raw_obs_crops_gripper_with_warning():
    raw_obs = {
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(6, dtype=np.float32),
        "agentview_image": np.zeros((64, 64, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((64, 64, 3), dtype=np.uint8),
    }
    adapted, warnings, diag = utils.adapt_raw_obs_for_checkpoint(raw_obs, _train_config())
    assert adapted["robot0_gripper_qpos"].shape == (2,)
    assert diag["low_dim_concat_shape"] == [9]
    assert any("robot0_gripper_qpos" in item for item in warnings)


def test_adapt_raw_obs_missing_joint_pos_errors():
    raw_obs = {
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((64, 64, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((64, 64, 3), dtype=np.uint8),
    }
    with pytest.raises(utils.StandaloneObsAdaptError, match="robot0_joint_pos"):
        utils.build_standalone_dp_observation(raw_obs, _train_config())


def test_validate_dataset_joint_pos_schema_detects_missing_joint_pos(tmp_path: Path):
    import h5py

    path = tmp_path / "no_joint.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eye_in_hand_image", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("robot0_gripper_qpos", data=np.zeros((4, 2), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))

    result = utils.validate_dataset_joint_pos_schema([path])
    assert result["ok"] is False
    assert utils.DATASET_MISSING_JOINT_POS_MESSAGE in result["errors"][0]


def test_assess_train_and_eval_smoke_requires_rollout():
    passed = mod.assess_train_and_eval_smoke(
        train_exit_code=0,
        eval_exit_code=0,
        checkpoint_path=MODULE_PATH,
        episodes_requested=3,
        eval_payload={"episodes": [{}, {}, {}]},
    )
    assert passed["smoke_passed"] is True

    no_rollout = mod.assess_train_and_eval_smoke(
        train_exit_code=0,
        eval_exit_code=0,
        checkpoint_path=MODULE_PATH,
        episodes_requested=3,
        eval_payload={"episodes": []},
    )
    assert no_rollout["smoke_passed"] is False
    assert no_rollout["failure_step"] == "eval_no_rollout"


def test_write_standalone_train_config_yaml(tmp_path: Path):
    out = utils.write_standalone_train_config_yaml(tmp_path)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "robot0_joint_pos" in text
    assert "robot0_eef_pos" not in text
