"""Tests for DP initialization-weight schema sync and compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from fastapi import HTTPException

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from app.services.dp_init_weight_compat import (
    dp_init_weights_compatible,
    enrich_asset_dp_init_schema,
    extract_dp_schema_fields_from_checkpoint,
    extract_dp_init_schema_from_cfg,
)
from app.services import training_dataset_compat as compat

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
}

LEGACY_JOINT_ACTIONS_KEY_CONFIG = {
    "action_dim": 8,
    "action_key": "actions",
    "controller_type": "JOINT_POSITION",
    "action_mode": "joint_delta_derived",
    "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
    "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
    "low_dim_dim": 9,
    "image_size": 128,
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
}


def _save_ckpt(path: Path, cfg: dict) -> None:
    action_dim = int(cfg.get("action_dim") or 7)
    low_dim_dim = int(cfg.get("low_dim_dim") or 9)
    torch.save(
        {
            "state_dict": {"layer": torch.zeros(1)},
            "normalizer": {
                "action": {"scale": [1.0] * action_dim, "offset": [0.0] * action_dim},
                "low_dim": {"scale": [1.0] * low_dim_dim, "offset": [0.0] * low_dim_dim},
            },
            "train_config": cfg,
        },
        path,
    )


def test_extract_dp_schema_fields_from_joint_actions_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "joint.pt"
    _save_ckpt(ckpt, JOINT_DP_TRAIN_CONFIG)
    fields = extract_dp_schema_fields_from_checkpoint(ckpt)
    assert fields["actionKey"] == "joint_actions"
    assert fields["evalExecutor"] == "joint_position"
    assert fields["controllerType"] == "JOINT_POSITION"
    assert fields["actionDim"] == 8


def test_legacy_joint_checkpoint_keeps_actions_action_key(tmp_path: Path):
    ckpt = tmp_path / "legacy_joint.pt"
    _save_ckpt(ckpt, LEGACY_JOINT_ACTIONS_KEY_CONFIG)
    fields = extract_dp_schema_fields_from_checkpoint(ckpt)
    assert fields["actionKey"] == "actions"
    assert fields["controllerType"] == "JOINT_POSITION"


def test_real_ull_pipeline_checkpoint_is_legacy_joint_actions_key():
    ckpt = Path(
        "/home/ubuntu/project/eai-idev2.1/runs/standalone_dp_joint_space_tests/"
        "20260624_full_pipeline/train_200ep/checkpoints/model_final.pt"
    )
    if not ckpt.is_file():
        pytest.skip("standalone joint pipeline checkpoint missing")
    fields = extract_dp_schema_fields_from_checkpoint(ckpt)
    assert fields["actionKey"] == "actions"
    assert fields["controllerType"] == "JOINT_POSITION"


def test_joint_actions_checkpoint_compatible_with_joint_training(tmp_path: Path):
    ckpt = tmp_path / "joint.pt"
    _save_ckpt(ckpt, JOINT_DP_TRAIN_CONFIG)
    compat.validate_dp_pretrained_checkpoint(
        checkpoint_path=ckpt,
        train_config={"dpConfig": dict(JOINT_DP_TRAIN_CONFIG)},
    )


def test_legacy_joint_actions_key_compatible_with_joint_actions_training(tmp_path: Path):
    ckpt = tmp_path / "legacy.pt"
    _save_ckpt(ckpt, LEGACY_JOINT_ACTIONS_KEY_CONFIG)
    source = extract_dp_init_schema_from_cfg(LEGACY_JOINT_ACTIONS_KEY_CONFIG)
    target = extract_dp_init_schema_from_cfg(JOINT_DP_TRAIN_CONFIG)
    ok, reason = dp_init_weights_compatible(source, target)
    assert ok, reason
    compat.validate_dp_pretrained_checkpoint(
        checkpoint_path=ckpt,
        train_config={"dpConfig": dict(JOINT_DP_TRAIN_CONFIG)},
    )


def test_eef_checkpoint_rejected_for_joint_training(tmp_path: Path):
    ckpt = tmp_path / "eef.pt"
    _save_ckpt(ckpt, EEF_DP_TRAIN_CONFIG)
    with pytest.raises(HTTPException) as exc:
        compat.validate_dp_pretrained_checkpoint(
            checkpoint_path=ckpt,
            train_config={"dpConfig": dict(JOINT_DP_TRAIN_CONFIG)},
        )
    detail = str(exc.value.detail)
    assert "EEF/OSC" in detail or "actions" in detail


def test_enrich_asset_dp_init_schema_reads_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ckpt = tmp_path / "joint.pt"
    _save_ckpt(ckpt, JOINT_DP_TRAIN_CONFIG)
    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.resolve_local_checkpoint_path",
        lambda **kwargs: str(ckpt),
    )
    enriched = enrich_asset_dp_init_schema(
        {
            "id": "model_test",
            "framework": "Diffusion Policy",
            "checkpointPath": str(ckpt),
            "actionKey": "actions",
        }
    )
    assert enriched["actionKey"] == "joint_actions"
    assert enriched["dpInitSchema"]["actionKey"] == "joint_actions"


def test_enrich_registry_entry_from_checkpoint_overrides_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.services.training_job_sync_service import _enrich_registry_entry_from_checkpoint

    ckpt = tmp_path / "joint.pt"
    _save_ckpt(ckpt, JOINT_DP_TRAIN_CONFIG)
    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.resolve_local_checkpoint_path",
        lambda **kwargs: str(ckpt),
    )
    entry = {
        "modelAssetId": "model_test",
        "checkpointPath": str(ckpt),
        "actionKey": "actions",
        "evalExecutor": "osc_pose",
    }
    enriched = _enrich_registry_entry_from_checkpoint(entry)
    assert enriched["actionKey"] == "joint_actions"
    assert enriched["evalExecutor"] == "joint_position"
