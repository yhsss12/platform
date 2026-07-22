from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation.evaluation_request_resolver import (
    _resolve_trained_policy_for_model_asset,
    normalize_evaluate_request,
)


def test_resolve_act_model_asset_policy_type():
    asset = {
        "id": "asset-act-1",
        "modelType": "act",
        "framework": "ACT",
        "trainingBackend": "act",
        "checkpointPath": "/tmp/model_final.pt",
    }
    with patch("app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id", return_value=asset):
        assert _resolve_trained_policy_for_model_asset("asset-act-1") == "act"


def test_resolve_pi0_model_asset_policy_type():
    asset = {
        "id": "asset-pi0-1",
        "modelTypeId": "pi0",
        "modelType": "pi0",
        "framework": "pi0",
        "trainingBackend": "pi0",
        "checkpointPath": "/tmp/pi0_final.pt",
    }
    with patch("app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id", return_value=asset):
        assert _resolve_trained_policy_for_model_asset("asset-pi0-1") == "pi0"


def test_normalize_cable_threading_act_evaluation():
    asset = {
        "id": "asset-act-1",
        "modelType": "act",
        "framework": "ACT",
        "checkpointPath": "/tmp/act/checkpoints/model_final.pt",
    }
    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        modelAssetId="asset-act-1",
    )
    from app.services.model_asset_validation import ModelAssetValidationResult

    validation = ModelAssetValidationResult(
        ok=True,
        reason="",
        model_asset_id="asset-act-1",
        artifact_path="/tmp/act/checkpoints/model_final.pt",
        backend_type="act",
        source_task_type="cable_threading",
        file_exists=True,
        file_size_bytes=1024,
        status="available",
    )
    with patch(
        "app.services.model_asset_validation.validate_model_asset",
        return_value=validation,
    ), patch(
        "app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id",
        return_value=asset,
    ):
        normalized = normalize_evaluate_request(request)
    assert normalized.policy == "act"
    assert normalized.checkpoint_path == "/tmp/act/checkpoints/model_final.pt"


def test_act_policy_adapter_loads_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from examples.cable_threading.act_lab.config import ActLabConfig
    from examples.cable_threading.act_lab.model import ActPolicy

    cfg = ActLabConfig(
        action_dim=7,
        chunk_size=4,
        image_keys=["agentview_image"],
        low_dim_keys=["robot0_eef_pos"],
        hidden_dim=64,
        latent_dim=8,
        enc_layers=2,
        nheads=4,
        dim_feedforward=128,
    )
    model = ActPolicy(
        action_dim=7,
        chunk_size=4,
        state_dim=3,
        num_cameras=1,
        hidden_dim=64,
        latent_dim=8,
        enc_layers=2,
        nheads=4,
        dim_feedforward=128,
    )
    ckpt_path = tmp_path / "model_final.pt"
    payload = {
        "state_dict": model.state_dict(),
        "backend": "act",
        "shape_meta": {
            "action_dim": 7,
            "chunk_size": 4,
            "state_dim": 3,
            "image_keys": ["agentview_image"],
            "low_dim_keys": ["robot0_eef_pos"],
        },
        "config": cfg.to_dict(),
    }
    torch.save(payload, ckpt_path)

    from examples.cable_threading.act_lab.policy_runtime import ACTPolicyAdapter

    adapter = ACTPolicyAdapter(ckpt_path, device="cpu")
    assert adapter.cfg.chunk_size == 4
    assert adapter.cfg.image_keys == ["agentview_image"]


def test_validate_act_obs_schema_missing_camera():
    from examples.cable_threading.obs_schema import validate_act_obs_schema

    class _FakePolicy:
        cfg = type("Cfg", (), {"image_keys": ["agentview_image"], "low_dim_keys": ["robot0_eef_pos"]})()

    validation = validate_act_obs_schema(_FakePolicy(), {"robot0_eef_pos": [0.0, 0.0, 0.0]})
    assert validation["valid"] is False
    assert "camera obs" in validation["errorMessage"]


def test_normalize_cable_threading_act_joint_space_selects_panda():
    asset = {
        "id": "asset-act-joint",
        "modelType": "act",
        "framework": "ACT",
        "checkpointPath": "/tmp/act/checkpoints/model_final.pt",
        "evalExecutor": "joint_position",
        "controllerType": "JOINT_POSITION",
        "actionDim": 8,
        "lowDimKeys": ["robot0_joint_pos", "robot0_gripper_qpos"],
        "preferredPolicySchemaId": "joint_state_obs_joint_action",
    }
    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        modelAssetId="asset-act-joint",
        numEpisodes=1,
        horizon=200,
    )
    from app.services.model_asset_validation import ModelAssetValidationResult

    validation = ModelAssetValidationResult(
        ok=True,
        reason="",
        model_asset_id="asset-act-joint",
        artifact_path="/tmp/act/checkpoints/model_final.pt",
        backend_type="act",
        source_task_type="cable_threading",
        file_exists=True,
        file_size_bytes=1024,
        status="available",
    )
    with patch(
        "app.services.model_asset_validation.validate_model_asset",
        return_value=validation,
    ), patch(
        "app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id",
        return_value=asset,
    ), patch(
        "app.services.policy_schema_resolver.resolve_act_eval_executor",
    ) as mock_act_exec:
        from app.services.dp_schema_resolver import DpEvalExecutorSpec

        mock_act_exec.return_value = DpEvalExecutorSpec(
            eval_executor="joint_position",
            controller_type="JOINT_POSITION",
            action_mode="joint_delta_derived",
            policy_type="act",
            side_channel_mode="policy",
            source="test",
        )
        normalized = normalize_evaluate_request(request)
    assert normalized.policy == "act"
    assert normalized.eval_executor == "joint_position"
    assert normalized.controller_type == "JOINT_POSITION"
    assert normalized.robot == "Panda"


def test_resolve_eval_robot_for_joint_act_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from app.services.policy_schema_resolver import resolve_eval_robot_for_policy

    ckpt = tmp_path / "model_final.pt"
    torch.save(
        {
            "shape_meta": {
                "action_dim": 8,
                "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
                "eval_executor": "joint_position",
                "controller_type": "JOINT_POSITION",
                "preferred_policy_schema_id": "joint_state_obs_joint_action",
            },
            "train_config": {},
        },
        ckpt,
    )
    robot, warnings = resolve_eval_robot_for_policy(
        policy="act",
        model_asset={},
        checkpoint_path=str(ckpt),
        eval_executor="joint_position",
        controller_type="JOINT_POSITION",
    )
    assert robot == "Panda"
    assert warnings
