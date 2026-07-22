"""Tests for pretrained DP init: backend canonicalization, schema compat, naming."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi import HTTPException

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from robosuite.utils.dlo.hdf5_dataset import save_dataset_hdf5  # noqa: E402

from app.services import training_dataset_compat as compat
from app.services import training_service as svc
from app.services.model_asset_naming import build_checkpoint_asset_display_name
from app.services.training_backend_canonical import (
    canonicalize_training_backend,
    training_backends_compatible,
)


JOINT_DP_TRAIN_CONFIG = {
    "action_dim": 8,
    "action_key": "joint_actions",
    "eval_executor": "joint_position",
    "controller_type": "JOINT_POSITION",
    "trained_action_mode": "joint_delta",
    "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
    "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
    "low_dim_dim": 9,
    "image_size": 128,
    "vision_encoder": "resnet18",
}

EEF_DP_TRAIN_CONFIG = {
    "action_dim": 7,
    "action_key": "actions",
    "eval_executor": "osc_pose",
    "controller_type": "OSC_POSE",
    "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
    "low_dim_keys": ["robot0_eef_pos", "robot0_gripper_qpos"],
    "low_dim_dim": 9,
    "image_size": 128,
    "vision_encoder": "resnet18",
}


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


def _save_dp_checkpoint(path: Path, train_cfg: dict) -> None:
    action_dim = int(train_cfg.get("action_dim") or 7)
    low_dim_dim = int(train_cfg.get("low_dim_dim") or 9)
    torch.save(
        {
            "state_dict": {"layer": torch.zeros(1)},
            "normalizer": {
                "action": {"scale": [1.0] * action_dim, "offset": [0.0] * action_dim},
                "low_dim": {"scale": [1.0] * low_dim_dim, "offset": [0.0] * low_dim_dim},
            },
            "train_config": train_cfg,
        },
        path,
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Diffusion Policy", "diffusion_policy"),
        ("diffusion-policy", "dp"),
        ("diffusion_policy", "Diffusion Policy"),
    ],
)
def test_diffusion_policy_backend_aliases_are_compatible(left: str, right: str):
    assert canonicalize_training_backend(left) == "diffusion_policy"
    assert canonicalize_training_backend(right) == "diffusion_policy"
    assert training_backends_compatible(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Robomimic BC", "robomimic_bc"),
        ("robomimic", "robomimic_bc"),
        ("ACT", "act"),
        ("pi0", "openpi"),
    ],
)
def test_other_backend_aliases_are_compatible(left: str, right: str):
    assert training_backends_compatible(left, right)


def test_validate_pretrained_model_accepts_diffusion_policy_display_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])
    checkpoint = tmp_path / "joint_final.pt"
    _save_dp_checkpoint(checkpoint, JOINT_DP_TRAIN_CONFIG)

    asset = {
        "id": "model_joint_final",
        "name": "线缆穿杆数据_20260625_0929 · Final",
        "framework": "Diffusion Policy",
        "modelType": "Diffusion Policy",
        "taskTemplateId": "task_cable_threading_v1",
        "checkpointPath": str(checkpoint),
        "sourceTrainingJobId": "train_prev",
    }

    with patch("app.services.workspace_model_asset_service.get_model_asset_by_id", return_value=asset):
        normalized = svc._validate_pretrained_model(
            pretrained={"modelAssetId": "model_joint_final"},
            resolved_backend="diffusion_policy",
            manifest={"taskType": "cable_threading"},
            train_config={"dpConfig": dict(JOINT_DP_TRAIN_CONFIG)},
        )

    assert normalized["initializationMode"] == "from_checkpoint"
    assert normalized["trainingBackend"] == "diffusion_policy"
    assert normalized["checkpointPath"] == str(checkpoint.resolve())


def test_joint_space_dp_checkpoint_can_initialize_joint_space_training(tmp_path: Path):
    ckpt_path = tmp_path / "joint.pt"
    _save_dp_checkpoint(ckpt_path, JOINT_DP_TRAIN_CONFIG)
    compat.validate_dp_pretrained_checkpoint(
        checkpoint_path=ckpt_path,
        train_config={"dpConfig": dict(JOINT_DP_TRAIN_CONFIG)},
    )


def test_eef_dp_checkpoint_rejected_for_joint_space_training(tmp_path: Path):
    ckpt_path = tmp_path / "eef.pt"
    _save_dp_checkpoint(ckpt_path, EEF_DP_TRAIN_CONFIG)
    with pytest.raises(HTTPException) as exc:
        compat.validate_dp_pretrained_checkpoint(
            checkpoint_path=ckpt_path,
            train_config={"dpConfig": dict(JOINT_DP_TRAIN_CONFIG)},
        )
    assert "EEF/OSC Diffusion Policy" in str(exc.value.detail)
    assert "Joint-Space Diffusion Policy" in str(exc.value.detail)


def test_checkpoint_asset_display_name_does_not_duplicate_final():
    context = "线缆穿杆数据_20260625_0929 · Final"
    assert build_checkpoint_asset_display_name(context_label=context, kind="final") == (
        "线缆穿杆数据_20260625_0929 · Final"
    )


def test_create_training_job_with_joint_dp_pretrained_writes_init_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "CABLE_WORKING_DIR", tmp_path / "CableThreadingMVP")
    monkeypatch.setattr(svc, "TRAIN_DP_SCRIPT", tmp_path / "CableThreadingMVP/examples/cable_threading/train_dp.py")
    (tmp_path / "CableThreadingMVP/examples/cable_threading").mkdir(parents=True)
    (svc.TRAIN_DP_SCRIPT).write_text("# stub", encoding="utf-8")
    (tmp_path / "CableThreadingMVP/examples/cable_threading/train_bc.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(svc, "TRAIN_BC_SCRIPT", tmp_path / "CableThreadingMVP/examples/cable_threading/train_bc.py")
    monkeypatch.setattr(svc, "PYTHON_BIN", Path("/usr/bin/python3"))
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    hdf5 = tmp_path / "joint_dataset.hdf5"
    save_dataset_hdf5(hdf5, [_traj()], metadata={"taskType": "cable_threading"})

    checkpoint = tmp_path / "init_joint.pt"
    _save_dp_checkpoint(checkpoint, JOINT_DP_TRAIN_CONFIG)

    manifest = {
        "datasetId": "ds_joint_init",
        "datasetName": "线缆穿杆数据",
        "taskType": "cable_threading",
        "taskName": "单臂穿线",
        "successfulEpisodes": 1,
        "cameraKeys": ["agentview_image", "robot0_eye_in_hand_image"],
        "artifacts": {"hdf5": str(hdf5)},
        "joint_action_available": True,
        "availableActionKeys": ["joint_actions", "gripper_actions", "actions"],
    }

    asset = {
        "id": "model_joint_final",
        "displayName": "线缆穿杆数据_20260625_0929 · Final",
        "framework": "Diffusion Policy",
        "taskTemplateId": "task_cable_threading_v1",
        "checkpointPath": str(checkpoint),
        "sourceTrainingJobId": "train_prev",
    }

    with patch.object(svc, "_execute_training_job", side_effect=lambda job_id: None):
        with patch("app.services.workspace_model_asset_service.get_model_asset_by_id", return_value=asset):
            result = svc.create_training_job(
                {
                    "datasetId": "ds_joint_init",
                    "datasetManifest": manifest,
                    "downstreamModelType": "Diffusion Policy",
                    "trainingBackend": "diffusion_policy",
                    "epochs": 2,
                    "batchSize": 8,
                    "pretrained": {"modelAssetId": "model_joint_final"},
                }
            )

    assert result["status"] == "queued"
    job_dir = tmp_path / "training" / "jobs" / result["trainJobId"]
    train_config = svc._read_json(job_dir / "config" / "train_config.json")
    assert train_config["pretrained"]["initializationMode"] == "from_checkpoint"
    assert train_config["initializationWeight"]["mode"] == "from_checkpoint"
    assert train_config["pretrainedModel"]["modelAssetId"] == "model_joint_final"
    assert train_config["pretrained"]["checkpointPath"] == str(checkpoint.resolve())

    cmd = svc._build_train_command(
        backend="diffusion_policy",
        hdf5_path=hdf5,
        out_dir=job_dir / "checkpoints",
        train_config=train_config,
    )
    assert cmd[cmd.index("--init-checkpoint") + 1] == str(checkpoint.resolve())
