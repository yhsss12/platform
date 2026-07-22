"""Tests for DP platform schema resolution and eval executor selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from robosuite.utils.dlo.hdf5_dataset import (  # noqa: E402
    HDF5_IMAGE_KEYS,
    HDF5_LOW_DIM_KEYS,
    build_hdf5_manifest_fields,
    save_dataset_hdf5,
)
from robosuite.utils.dlo.hdf5_platform_schema import (  # noqa: E402
    ACTION_SCHEMA_JOINT,
    OBS_SCHEMA_JOINT,
)

from app.services.dp_schema_resolver import (  # noqa: E402
    resolve_dp_eval_executor,
    resolve_dp_training_schema,
)


def _traj(steps: int = 3):
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


def test_hdf5_manifest_includes_platform_schemas(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    save_info = save_dataset_hdf5(
        hdf5_path,
        [_traj()],
        image_keys=list(HDF5_IMAGE_KEYS),
        low_dim_keys=list(HDF5_LOW_DIM_KEYS),
        metadata={"taskTemplateId": "cable_threading_single_arm", "taskType": "cable_threading", "robot": "Panda"},
    )
    manifest_fields = build_hdf5_manifest_fields(save_info)
    assert manifest_fields["observationSchema"] == OBS_SCHEMA_JOINT
    assert manifest_fields["actionSchema"] == ACTION_SCHEMA_JOINT
    assert manifest_fields["trainedActionMode"] == "joint_delta"
    assert manifest_fields["evalExecutor"] == "joint_position"
    assert manifest_fields["controllerSchemaDetail"]["controllerType"] == "JOINT_POSITION"


def test_resolve_dp_training_schema_joint_from_manifest():
    manifest = {
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
    spec = resolve_dp_training_schema(manifest)
    assert spec.action_key == "joint_actions"
    assert spec.action_dim == 8
    assert spec.controller_type == "JOINT_POSITION"
    assert spec.eval_executor == "joint_position"
    assert spec.trained_action_mode == "joint_delta"


def test_resolve_dp_eval_executor_joint_from_model_asset():
    asset = {
        "modelType": "diffusion_policy",
        "actionSchema": ACTION_SCHEMA_JOINT,
        "trainedActionMode": "joint_delta",
        "evalExecutor": "joint_position",
        "controllerType": "JOINT_POSITION",
    }
    spec = resolve_dp_eval_executor(policy="diffusion_policy", model_asset=asset)
    assert spec.uses_joint_executor()
    assert spec.controller_type == "JOINT_POSITION"


def test_resolve_dp_eval_executor_legacy_without_schema():
    asset = {"modelType": "diffusion_policy", "actionDim": 7}
    spec = resolve_dp_eval_executor(policy="diffusion_policy", model_asset=asset)
    assert spec.eval_executor == "osc_pose"
    assert not spec.uses_joint_executor()


def test_expert_eval_stays_on_legacy_executor():
    spec = resolve_dp_eval_executor(policy="scripted")
    assert spec.eval_executor == "osc_pose"
    assert spec.policy_type == "expert"


def test_dp_eval_resolver_wires_joint_executor_for_joint_asset():
    from app.schemas.evaluation import EvaluateAsyncRequest
    from app.services.evaluation.evaluation_request_resolver import normalize_evaluate_request
    from app.services.model_asset_validation import ModelAssetValidationResult

    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        numEpisodes=10,
        modelAssetId="model__joint_final",
    )
    asset = {
        "id": "model__joint_final",
        "modelType": "diffusion_policy",
        "framework": "Diffusion Policy",
        "checkpointPath": "/tmp/model_final.pt",
        "actionSchema": ACTION_SCHEMA_JOINT,
        "trainedActionMode": "joint_delta",
        "evalExecutor": "joint_position",
        "controllerType": "JOINT_POSITION",
    }
    validation = ModelAssetValidationResult(
        ok=True,
        reason="",
        model_asset_id="model__joint_final",
        artifact_path="/tmp/model_final.pt",
        backend_type="diffusion_policy",
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
    assert normalized.policy == "diffusion_policy"
    assert normalized.eval_executor == "joint_position"
    assert normalized.action_mode == "joint_delta"


def test_robomimic_asset_unaffected():
    from app.schemas.evaluation import EvaluateAsyncRequest
    from app.services.evaluation.evaluation_request_resolver import normalize_evaluate_request
    from app.services.model_asset_validation import ModelAssetValidationResult

    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        numEpisodes=10,
        modelAssetId="model__bc_final",
    )
    asset = {
        "id": "model__bc_final",
        "modelType": "robomimic_bc",
        "framework": "Robomimic BC",
        "checkpointPath": "/tmp/model.pth",
    }
    validation = ModelAssetValidationResult(
        ok=True,
        reason="",
        model_asset_id="model__bc_final",
        artifact_path="/tmp/model.pth",
        backend_type="robomimic_bc",
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
    assert normalized.policy == "robomimic"
    assert normalized.eval_executor is None
