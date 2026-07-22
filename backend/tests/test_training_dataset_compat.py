from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi import HTTPException

from app.services import training_dataset_compat as compat


def _cable_manifest(dataset_id: str, hdf5_path: Path, episodes: int = 9) -> dict:
    return {
        "datasetId": dataset_id,
        "datasetName": f"dataset-{dataset_id}",
        "taskType": "cable_threading",
        "taskName": "单臂穿线",
        "robotType": "Panda",
        "simulatorBackend": "mujoco",
        "cameraKeys": ["agentview_image", "robot0_eye_in_hand_image"],
        "imageKeys": ["agentview_image", "robot0_eye_in_hand_image"],
        "observationKeys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "actionDim": 7,
        "imageSize": 84,
        "sampleCount": episodes,
        "artifacts": {"hdf5": str(hdf5_path)},
    }


def test_merge_training_manifests_rejects_mismatched_action_dim(tmp_path: Path):
    left = _cable_manifest("ds_a", tmp_path / "a.hdf5")
    right = dict(_cable_manifest("ds_b", tmp_path / "b.hdf5"))
    right["actionDim"] = 14
    for path in (tmp_path / "a.hdf5", tmp_path / "b.hdf5"):
        path.write_bytes(b"\x00" * 8)

    with patch.object(compat, "_training_paths") as mock_paths:
        ts = mock_paths.return_value
        ts._resolve_hdf5_path.side_effect = lambda manifest: Path(manifest["artifacts"]["hdf5"])
        ts._is_valid_hdf5_file.return_value = True
        with pytest.raises(HTTPException) as exc:
            compat.merge_training_manifests([left, right])
    assert "结构不一致" in str(exc.value.detail)


def test_merge_training_manifests_merges_compatible_datasets(tmp_path: Path):
    left = _cable_manifest("ds_a", tmp_path / "a.hdf5", episodes=9)
    right = _cable_manifest("ds_b", tmp_path / "b.hdf5", episodes=81)
    for path in (tmp_path / "a.hdf5", tmp_path / "b.hdf5"):
        path.write_bytes(b"\x00" * 8)

    with patch.object(compat, "_training_paths") as mock_paths:
        ts = mock_paths.return_value
        ts._resolve_hdf5_path.side_effect = lambda manifest: Path(manifest["artifacts"]["hdf5"])
        ts._is_valid_hdf5_file.return_value = True
        merged, paths, _ = compat.merge_training_manifests([left, right])

    assert merged["mergedDatasetCount"] == 2
    assert merged["sampleCount"] == 90
    assert len(paths) == 2
    assert merged["artifacts"]["hdf5Paths"] == [str(tmp_path / "a.hdf5"), str(tmp_path / "b.hdf5")]


def test_validate_dp_pretrained_checkpoint_rejects_missing_normalizer(tmp_path: Path):
    ckpt_path = tmp_path / "bad.pt"
    torch.save({"state_dict": {}, "train_config": {"action_dim": 7}}, ckpt_path)
    with pytest.raises(HTTPException) as exc:
        compat.validate_dp_pretrained_checkpoint(
            checkpoint_path=ckpt_path,
            train_config={"dpConfig": {"action_dim": 7}},
        )
    assert "normalizer" in str(exc.value.detail)


def test_validate_dp_pretrained_checkpoint_rejects_action_dim_mismatch(tmp_path: Path):
    ckpt_path = tmp_path / "bad.pt"
    torch.save(
        {
            "state_dict": {"layer": torch.zeros(1)},
            "normalizer": {
                "action": {"scale": [1.0], "offset": [0.0]},
                "low_dim": {"scale": [1.0], "offset": [0.0]},
            },
            "train_config": {
                "action_dim": 14,
                "image_keys": ["agentview_image"],
                "low_dim_keys": ["robot0_eef_pos"],
                "image_size": 84,
                "low_dim_dim": 9,
                "vision_encoder": "resnet18",
            },
        },
        ckpt_path,
    )
    with pytest.raises(HTTPException) as exc:
        compat.validate_dp_pretrained_checkpoint(
            checkpoint_path=ckpt_path,
            train_config={
                "dpConfig": {
                    "action_dim": 7,
                    "image_keys": ["agentview_image"],
                    "low_dim_keys": ["robot0_eef_pos"],
                    "image_size": 84,
                    "low_dim_dim": 9,
                    "vision_encoder": "resnet18",
                }
            },
        )
    assert "action_dim" in str(exc.value.detail)
