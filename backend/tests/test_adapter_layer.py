from __future__ import annotations

import pytest

from app.services.adapter_layer.adapter_service import (
    analyze_dataset_compatibility,
    build_evaluation_plan,
    build_training_plan,
    normalize_dataset_manifest,
    recommend_training_models,
)
from app.services.adapter_layer.compatibility_checker import CompatibilityAnalysis


def _mujoco_panda_low_dim_manifest(**overrides) -> dict:
    base = {
        "datasetId": "ds_test_mujoco_panda",
        "datasetName": "测试 MuJoCo Panda 数据集",
        "taskName": "线缆穿杆",
        "taskType": "cable_threading",
        "backend": "mujoco",
        "robotType": "Panda",
        "dataFormat": "HDF5",
        "observationSpace": {
            "type": "low_dim",
            "keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        },
        "actionSpace": {"type": "continuous", "dim": 7, "supportsSequence": False},
        "episodes": 12,
        "successfulEpisodes": 10,
        "artifacts": {"hdf5": "/tmp/dataset.hdf5"},
        "manifestVersion": "1.0",
    }
    base.update(overrides)
    return base


def _sequence_image_manifest(**overrides) -> dict:
    base = {
        "datasetId": "ds_test_dp",
        "datasetName": "DP 序列动作数据集",
        "taskName": "线缆穿杆",
        "taskType": "cable_threading",
        "backend": "mujoco",
        "robotType": "Panda",
        "dataFormat": "HDF5",
        "observationSpace": {
            "type": "image",
            "keys": ["agentview_image", "robot0_eye_in_hand_image"],
        },
        "actionSpace": {
            "type": "continuous",
            "dim": 7,
            "supportsSequence": True,
            "horizon": 8,
        },
        "horizon": 8,
        "episodes": 20,
        "successfulEpisodes": 18,
        "artifacts": {"hdf5": "/tmp/dataset_dp.hdf5"},
        "manifestVersion": "1.0",
    }
    base.update(overrides)
    return base


def test_normalize_dataset_manifest_maps_legacy_fields():
    raw = {
        "datasetId": "ds_legacy",
        "backend": "mujoco",
        "episodes": 5,
        "successfulEpisodes": 4,
        "dataFormat": "HDF5",
        "artifacts": {"hdf5": "/data/dataset.hdf5"},
    }
    manifest = normalize_dataset_manifest(raw)
    assert manifest.datasetId == "ds_legacy"
    assert manifest.simulator == "mujoco"
    assert manifest.episodeCount == 5
    assert manifest.successCount == 4
    assert manifest.storageUri == "/data/dataset.hdf5"
    assert manifest.manifestVersion == "1.0"


def test_mujoco_panda_hdf5_low_dim_recommends_robomimic_bc():
    raw = _mujoco_panda_low_dim_manifest()
    manifest = normalize_dataset_manifest(raw)
    recommended = recommend_training_models(manifest)
    assert "robomimic_bc" in recommended
    assert recommended[0] == "robomimic_bc"

    analysis = analyze_dataset_compatibility(manifest)
    assert isinstance(analysis, CompatibilityAnalysis)
    robomimic = next(item for item in analysis.results if item.modelType == "robomimic_bc")
    assert robomimic.compatible is True
    assert not robomimic.reasons


def test_sequence_action_dataset_recommends_diffusion_policy():
    raw = _sequence_image_manifest()
    manifest = normalize_dataset_manifest(raw)
    recommended = recommend_training_models(manifest)
    assert "diffusion_policy" in recommended
    assert recommended[0] == "diffusion_policy"

    dp = next(item for item in analyze_dataset_compatibility(manifest).results if item.modelType == "diffusion_policy")
    assert dp.compatible is True


def test_missing_observation_keys_dp_low_dim_still_adaptable():
    raw = _sequence_image_manifest(
        observationSpace={
            "type": "low_dim",
            "keys": ["robot0_eef_pos"],
        },
        actionSpace={"type": "continuous", "dim": 7},
    )
    raw.pop("horizon", None)
    analysis = analyze_dataset_compatibility(raw)
    dp = next(item for item in analysis.results if item.modelType == "diffusion_policy")
    assert dp.compatible is True


def test_missing_storage_uri_blocks_compatibility():
    raw = _mujoco_panda_low_dim_manifest(artifacts={})
    analysis = analyze_dataset_compatibility(raw)
    assert analysis.compatible is False
    assert any("storageUri" in reason or "存储" in reason for reason in analysis.blockingReasons)


def test_build_training_plan_default_hyperparameters():
    raw = _mujoco_panda_low_dim_manifest()
    plan = build_training_plan(raw, "robomimic_bc")
    assert plan["epochs"] == 5
    assert plan["batchSize"] == 16
    assert plan["learningRate"] == 0.0001
    assert plan["advancedConfig"]["actor_hidden_dims"] == "512,512"
    assert plan["trainingBackend"] == "robomimic_bc"
    assert plan["datasetId"] == "ds_test_mujoco_panda"


def test_build_training_plan_rejects_incompatible_model():
    raw = _mujoco_panda_low_dim_manifest()
    with pytest.raises(ValueError, match="不兼容"):
        build_training_plan(raw, "torch_bc")


def test_build_evaluation_plan_from_training_plan():
    training_plan = build_training_plan(_mujoco_panda_low_dim_manifest(), "robomimic_bc")
    eval_plan = build_evaluation_plan(training_plan)
    assert eval_plan["evaluationMode"] == "trained_model_evaluation"
    assert eval_plan["taskTemplateId"] == "cable_threading_single_arm"
    assert eval_plan["simulator"] == "mujoco"
    assert eval_plan["robotType"] == "Panda"
    assert eval_plan["numEpisodes"] == 10
    assert "metric_cable_success_rate_v1" in eval_plan["metrics"]
    assert eval_plan["policyType"] == "robomimic_bc"


def test_build_evaluation_plan_from_model_asset():
    eval_plan = build_evaluation_plan(
        {
            "id": "model_test_final",
            "modelType": "diffusion_policy",
            "taskTemplateId": "cable_threading_single_arm",
            "sourceDatasetId": "ds_test_dp",
            "checkpointPath": "/tmp/model_final.pt",
            "framework": "Diffusion Policy",
            "trainingBackend": "diffusion_policy",
        }
    )
    assert eval_plan["modelType"] == "diffusion_policy"
    assert eval_plan["modelAssetId"] == "model_test_final"
    assert eval_plan["checkpointPath"] == "/tmp/model_final.pt"
    assert eval_plan["datasetId"] == "ds_test_dp"
