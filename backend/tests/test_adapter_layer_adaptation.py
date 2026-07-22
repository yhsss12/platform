from __future__ import annotations

from pathlib import Path

import pytest

from app.services.adapter_layer.adaptation_validator import validate_adaptation
from app.services.adapter_layer.dataset_profiler import build_dataset_profile
from app.services.adapter_layer.hdf5_inspector import inspect_hdf5
from app.services.adapter_layer.model_adaptation_builder import build_model_adaptation_plan
from app.services.adapter_layer.training_adaptation_service import build_training_adaptation_plan


def _write_low_dim_hdf5(path, *, action_dim=7, obs_keys=None, horizon=50):
    h5py = pytest.importorskip("h5py")
    obs_keys = obs_keys or ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        for key in obs_keys:
            obs.create_dataset(key, data=[[0.0] * 3])
        demo.create_dataset("actions", data=[[0.0] * action_dim] * horizon)
        demo.create_dataset("rewards", data=[0.0] * horizon)
        demo.create_dataset("dones", data=[0] * horizon)


def _write_image_hdf5(path, *, action_dim=7, horizon=32, h=64, w=64):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((horizon, h, w, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eye_in_hand_image", data=np.zeros((horizon, h, w, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eef_pos", data=np.zeros((horizon, 3), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((horizon, action_dim), dtype=np.float32))
        demo.create_dataset("rewards", data=np.zeros((horizon,), dtype=np.float32))
        demo.create_dataset("dones", data=np.zeros((horizon,), dtype=np.int32))


def test_robomimic_bc_adaptation_from_low_dim_hdf5(tmp_path):
    hdf5 = tmp_path / "dataset.hdf5"
    _write_low_dim_hdf5(hdf5, action_dim=7)

    manifest = {
        "datasetId": "ds_robo",
        "backend": "mujoco",
        "robotType": "Panda",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(
        dataset_id="ds_robo",
        raw_manifest=manifest,
        model_type="robomimic_bc",
    )

    profile = plan["datasetProfile"]
    assert profile["simulator"] == "MuJoCo"
    assert profile["robotType"] == "Panda"
    assert profile["actionDim"] == 7
    assert profile["observationType"] == "low_dim"
    assert "robot0_eef_pos" in profile["observationKeys"]

    adaptation = plan["modelAdaptation"]
    assert adaptation["modelType"] == "robomimic_bc"
    assert adaptation["outputConfig"]["action_dim"] == 7
    assert adaptation["architectureConfig"]["actor_hidden_dims"] == [512, 512]
    assert adaptation["advancedConfig"]["actor_hidden_dims"] == "512,512"
    assert plan["validation"]["adaptable"] is True


def test_diffusion_policy_adaptation_from_image_hdf5(tmp_path):
    hdf5 = tmp_path / "dataset_dp.hdf5"
    _write_image_hdf5(hdf5, horizon=12)

    manifest = {
        "datasetId": "ds_dp",
        "backend": "mujoco",
        "robotType": "Panda",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(
        raw_manifest=manifest,
        model_type="diffusion_policy",
    )

    profile = plan["datasetProfile"]
    assert profile["observationType"] in {"image", "mixed"}
    assert "agentview_image" in profile["cameraKeys"]
    assert "robot0_eye_in_hand_image" in profile["cameraKeys"]

    adaptation = plan["modelAdaptation"]
    assert adaptation["architectureConfig"]["obs_encoder"] == "multi_image"
    assert adaptation["architectureConfig"]["image_encoder"]["camera_keys"] == profile["cameraKeys"]
    assert adaptation["advancedConfig"]["n_obs_steps"] >= 2
    assert adaptation["advancedConfig"]["horizon"] == 16
    assert adaptation["advancedConfig"]["n_action_steps"] == 8
    assert adaptation["normalizationConfig"]["mode"] == "min_max"
    assert plan["validation"]["adaptable"] is True


def test_diffusion_policy_adaptation_low_dim_without_cameras(tmp_path):
    hdf5 = tmp_path / "dataset_dp_lowdim.hdf5"
    _write_low_dim_hdf5(hdf5, action_dim=7)

    manifest = {
        "datasetId": "ds_dp_lowdim",
        "backend": "isaac_lab",
        "taskName": "物块堆叠",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(raw_manifest=manifest, model_type="diffusion_policy")
    assert plan["validation"]["adaptable"] is True
    assert plan["modelAdaptation"]["architectureConfig"]["obs_encoder"] == "low_dim"
    assert plan["modelAdaptation"]["inputConfig"]["camera_keys"] == []


def test_act_not_adaptable_without_images(tmp_path):
    hdf5 = tmp_path / "dataset_act.hdf5"
    _write_low_dim_hdf5(hdf5)

    manifest = {
        "datasetId": "ds_act",
        "backend": "mujoco",
        "robotType": "Panda",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(
        raw_manifest=manifest,
        model_type="act",
    )
    assert plan["validation"]["adaptable"] is False
    assert any("image observations" in err for err in plan["validation"]["errors"])


def test_act_adaptable_with_image_hdf5(tmp_path):
    hdf5 = tmp_path / "act_image.hdf5"
    _write_image_hdf5(hdf5, horizon=40)

    manifest = {
        "datasetId": "ds_act",
        "backend": "mujoco",
        "robotType": "Panda",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(raw_manifest=manifest, model_type="act")
    assert plan["validation"]["adaptable"] is True
    assert plan["modelAdaptation"]["modelType"] == "act"
    assert plan["modelAdaptation"]["inputConfig"]["camera_names"]

    from app.services.adapter_layer.training_adaptation_integration import build_act_config_dict

    act_cfg = build_act_config_dict(plan["datasetProfile"], plan["modelAdaptation"])
    assert act_cfg["image_keys"]
    assert act_cfg["action_dim"] == 7
    assert act_cfg["chunk_size"] >= 10


def test_act_config_written_on_adaptation(tmp_path):
    hdf5 = tmp_path / "act_image.hdf5"
    _write_image_hdf5(hdf5)
    manifest = {
        "datasetId": "ds_act_cfg",
        "successfulEpisodes": 1,
        "artifacts": {"hdf5": str(hdf5)},
    }
    from app.services.adapter_layer.training_adaptation_integration import apply_training_adaptation

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    merged, snapshot = apply_training_adaptation(
        manifest=manifest,
        payload={"downstreamModelType": "ACT", "trainingBackend": "act"},
        train_job_dir=job_dir,
    )
    assert (job_dir / "config" / "act_adapted.yaml").is_file()
    assert merged.get("actConfigPath")
    assert snapshot["modelType"] == "act"


def test_manifest_missing_fields_inferred_from_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "bare.hdf5"
    _write_low_dim_hdf5(hdf5, action_dim=8, horizon=20)

    profile = build_dataset_profile({"artifacts": {"hdf5": str(hdf5)}})
    assert profile.actionDim == 8
    assert profile.horizon == 20
    assert profile.episodeCount == 1
    assert "hdf5" in profile.inferenceSources
    assert any("推断" in w for w in profile.warnings)


def test_overrides_merge_into_adaptation_plan(tmp_path):
    hdf5 = tmp_path / "dataset.hdf5"
    _write_low_dim_hdf5(hdf5)

    manifest = {
        "datasetId": "ds_override",
        "backend": "mujoco",
        "robotType": "Panda",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(
        raw_manifest=manifest,
        model_type="robomimic_bc",
        overrides={
            "trainingConfig": {"epochs": 20, "batchSize": 32},
            "advancedConfig": {"actor_hidden_dims": "256,256"},
        },
    )
    assert plan["modelAdaptation"]["trainingConfig"]["epochs"] == 20
    assert plan["modelAdaptation"]["trainingConfig"]["batchSize"] == 32
    assert plan["modelAdaptation"]["advancedConfig"]["actor_hidden_dims"] == "256,256"
    assert plan["configPatch"]["epochs"] == 20


def test_hdf5_inspector_detects_reward_done(tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "meta.hdf5"
    _write_low_dim_hdf5(hdf5)
    result = inspect_hdf5(hdf5)
    assert result.has_reward is True
    assert result.has_done is True
    assert result.action_dim == 7


def test_config_patch_structure_for_training_job(tmp_path):
    hdf5 = tmp_path / "dataset.hdf5"
    _write_low_dim_hdf5(hdf5)
    manifest = {
        "datasetId": "ds_patch",
        "backend": "mujoco",
        "robotType": "Panda",
        "successfulEpisodes": 5,
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(raw_manifest=manifest, model_type="robomimic_bc")
    patch = plan["configPatch"]
    assert patch["datasetId"] == "ds_patch"
    assert patch["trainingBackend"] == "robomimic_bc"
    assert "architectureConfig" in patch
    assert "dataLoaderConfig" in patch
    assert "normalizationConfig" in patch
    assert patch["modelParams"]


def test_torch_bc_adaptation_low_dim(tmp_path):
    hdf5 = tmp_path / "dual.hdf5"
    _write_low_dim_hdf5(hdf5, action_dim=14, obs_keys=["left_arm_qpos", "right_arm_qpos"])

    manifest = {
        "datasetId": "ds_torch",
        "taskType": "dual_arm_cable_manipulation",
        "backend": "mujoco",
        "artifacts": {"hdf5": str(hdf5)},
    }
    profile = build_dataset_profile(manifest)
    plan = build_model_adaptation_plan(profile, "torch_bc")
    validation = validate_adaptation(profile, plan)
    assert plan.architectureConfig["output_dim"] == 14
    assert plan.inputConfig["input_dim"] >= 1
    assert validation.adaptable is True


def test_manifest_partial_obs_keys_overridden_by_hdf5():
    """Isaac 数据集 manifest 可能只声明部分 obsKeys，训练适配应以 HDF5 结构为准。"""
    hdf5 = Path(
        "/home/ubuntu/project/eai-idev2.1/runs/isaac_lab/jobs/"
        "isaac_gen_20260617_223736_1fab/artifacts/dataset.hdf5"
    )
    if not hdf5.is_file():
        pytest.skip("Isaac block stacking HDF5 fixture not available")

    manifest = {
        "datasetId": "isaac_ds_test",
        "taskType": "isaac_block_stacking",
        "obsKeys": ["eef_pos", "eef_quat", "gripper_pos", "object"],
        "actionDim": 7,
        "artifacts": {"hdf5": str(hdf5)},
    }
    profile = build_dataset_profile(manifest)
    assert "cube_positions" in profile.observationKeys
    assert "actions" not in profile.observationKeys
    assert profile.stateDim > 4
    assert any("HDF5" in w for w in profile.warnings)

    plan = build_training_adaptation_plan(raw_manifest=manifest, model_type="diffusion_policy")
    dp_cfg = plan["configPatch"].get("dpConfig") or {}
    if not dp_cfg:
        from app.services.adapter_layer.training_adaptation_integration import build_dp_config_dict

        dp_cfg = build_dp_config_dict(plan["datasetProfile"], plan["modelAdaptation"])
    assert int(dp_cfg["low_dim_dim"]) == profile.stateDim
    assert int(dp_cfg["low_dim_dim"]) >= 48
