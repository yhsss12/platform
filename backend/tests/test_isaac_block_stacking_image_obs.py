from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.adapter_layer.dataset_profiler import build_dataset_profile
from app.services.adapter_layer.training_adaptation_service import build_training_adaptation_plan
from integrations.isaac_lab.hdf5_image_obs import (
    build_observation_manifest_fields,
    inject_camera_observations,
    inspect_hdf5_observation_metadata,
)


def _write_block_stacking_low_dim_hdf5(path: Path, *, horizon: int = 40) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("eef_pos", data=np.zeros((horizon, 3), dtype=np.float32))
        obs.create_dataset("eef_quat", data=np.zeros((horizon, 4), dtype=np.float32))
        obs.create_dataset("gripper_pos", data=np.zeros((horizon, 2), dtype=np.float32))
        obs.create_dataset("object", data=np.zeros((horizon, 39), dtype=np.float32))
        obs.create_dataset("cube_positions", data=np.zeros((horizon, 9), dtype=np.float32))
        obs.create_dataset("cube_orientations", data=np.zeros((horizon, 12), dtype=np.float32))
        obs.create_dataset("joint_pos", data=np.zeros((horizon, 9), dtype=np.float32))
        obs.create_dataset("joint_vel", data=np.zeros((horizon, 9), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((horizon, 7), dtype=np.float32))


def _write_block_stacking_mixed_hdf5(
    path: Path,
    *,
    horizon: int = 32,
    resolution: int = 128,
    include_wrist: bool = False,
) -> None:
    _write_block_stacking_low_dim_hdf5(path, horizon=horizon)
    frames = [np.full((resolution, resolution, 3), fill_value=idx % 255, dtype=np.uint8) for idx in range(horizon)]
    camera_frames = {"agentview_image": frames}
    if include_wrist:
        camera_frames["robot0_eye_in_hand_image"] = [frame.copy() for frame in frames]
    inject_camera_observations(
        path,
        "demo_0",
        camera_frames,
        height=resolution,
        width=resolution,
    )


def test_inject_camera_observations_aligns_with_actions(tmp_path: Path):
    hdf5 = tmp_path / "mixed.hdf5"
    horizon = 25
    resolution = 128
    _write_block_stacking_low_dim_hdf5(hdf5, horizon=horizon)
    inject_camera_observations(
        hdf5,
        "demo_0",
        {"agentview_image": [np.zeros((64, 64, 3), dtype=np.uint8)] * (horizon + 3)},
        height=resolution,
        width=resolution,
    )

    h5py = pytest.importorskip("h5py")
    with h5py.File(hdf5, "r") as handle:
        demo = handle["data/demo_0"]
        assert demo["actions"].shape[0] == horizon
        image = demo["obs/agentview_image"]
        assert image.shape == (horizon, resolution, resolution, 3)
        assert image.dtype == np.uint8


def test_inspect_hdf5_observation_metadata_mixed(tmp_path: Path):
    hdf5 = tmp_path / "mixed.hdf5"
    _write_block_stacking_mixed_hdf5(hdf5, horizon=18, resolution=128)

    meta = inspect_hdf5_observation_metadata(hdf5)
    assert meta["observationType"] == "mixed"
    assert "agentview_image" in meta["cameraKeys"]
    assert "agentview_image" in meta["imageKeys"]
    assert "eef_pos" in meta["obsKeys"]
    assert meta["imageShape"] == {"height": 128, "width": 128, "channels": 3}
    assert meta["actionDim"] == 7


def test_build_observation_manifest_fields_for_block_stacking(tmp_path: Path):
    hdf5 = tmp_path / "mixed.hdf5"
    _write_block_stacking_mixed_hdf5(hdf5)

    fields = build_observation_manifest_fields(hdf5)
    assert fields["observationType"] == "mixed"
    assert fields["cameraKeys"] == ["agentview_image"]
    assert fields["imageKeys"] == ["agentview_image"]
    assert fields["simulator"] == "Isaac"
    assert fields["robotType"] == "Panda"
    assert fields["format"] == "HDF5"
    assert fields["quality"]["hasImage"] is True


def test_dataset_profile_recognizes_block_stacking_mixed(tmp_path: Path):
    hdf5 = tmp_path / "mixed.hdf5"
    _write_block_stacking_mixed_hdf5(hdf5, horizon=30)

    manifest = {
        "datasetId": "isaac_ds_image_test",
        "taskType": "isaac_block_stacking",
        "taskName": "物块堆叠",
        "simulatorBackend": "isaac_lab",
        "robotType": "Panda",
        "obsKeys": ["eef_pos", "eef_quat", "gripper_pos", "object"],
        "actionDim": 7,
        "artifacts": {"hdf5": str(hdf5)},
    }
    profile = build_dataset_profile(manifest)
    assert profile.observationType == "mixed"
    assert "agentview_image" in profile.cameraKeys
    assert "agentview_image" in profile.imageKeys
    assert "eef_pos" in profile.observationKeys
    assert "cube_positions" in profile.observationKeys
    assert profile.actionDim == 7
    assert profile.stateDim > 0
    assert profile.episodeCount == 1


def test_act_adaptable_for_block_stacking_image_dataset(tmp_path: Path):
    hdf5 = tmp_path / "mixed.hdf5"
    _write_block_stacking_mixed_hdf5(hdf5, horizon=40)

    manifest = {
        "datasetId": "isaac_ds_act_image",
        "taskType": "isaac_block_stacking",
        "taskName": "物块堆叠",
        "simulatorBackend": "isaac_lab",
        "robotType": "Panda",
        "successfulEpisodes": 1,
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(raw_manifest=manifest, model_type="act")
    assert plan["validation"]["adaptable"] is True
    assert plan["modelAdaptation"]["modelType"] == "act"
    assert plan["modelAdaptation"]["inputConfig"]["camera_names"] == ["agentview_image"]
    assert plan["modelAdaptation"]["inputConfig"]["image_keys"] == ["agentview_image"]
    assert plan["modelAdaptation"]["inputConfig"]["act_variant"] == "image_proprio"
    assert plan["modelAdaptation"]["outputConfig"]["action_dim"] == 7
    assert "eef_pos" in plan["modelAdaptation"]["inputConfig"]["low_dim_keys"]


def test_act_rejects_old_block_stacking_low_dim_dataset(tmp_path: Path):
    hdf5 = tmp_path / "lowdim.hdf5"
    _write_block_stacking_low_dim_hdf5(hdf5)

    manifest = {
        "datasetId": "isaac_ds_lowdim",
        "taskType": "isaac_block_stacking",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(raw_manifest=manifest, model_type="act")
    assert plan["validation"]["adaptable"] is False
    assert any("image observations" in err for err in plan["validation"]["errors"])


def test_act_supports_single_camera_block_stacking(tmp_path: Path):
    hdf5 = tmp_path / "single_cam.hdf5"
    _write_block_stacking_mixed_hdf5(hdf5, include_wrist=False)

    from app.services.adapter_layer.training_adaptation_integration import build_act_config_dict

    manifest = {
        "datasetId": "isaac_ds_single_cam",
        "taskType": "isaac_block_stacking",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(raw_manifest=manifest, model_type="act")
    act_cfg = build_act_config_dict(plan["datasetProfile"], plan["modelAdaptation"])
    assert act_cfg["image_keys"] == ["agentview_image"]
    assert len(act_cfg["image_keys"]) == 1


def test_act_adaptation_with_dual_cameras_when_present(tmp_path: Path):
    hdf5 = tmp_path / "dual_cam.hdf5"
    _write_block_stacking_mixed_hdf5(hdf5, include_wrist=True, horizon=24)

    manifest = {
        "datasetId": "isaac_ds_dual_cam",
        "taskType": "isaac_block_stacking",
        "artifacts": {"hdf5": str(hdf5)},
    }
    plan = build_training_adaptation_plan(raw_manifest=manifest, model_type="act")
    assert plan["validation"]["adaptable"] is True
    camera_names = plan["modelAdaptation"]["inputConfig"]["camera_names"]
    assert "agentview_image" in camera_names
    assert "robot0_eye_in_hand_image" in camera_names
