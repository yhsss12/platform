import json

import h5py
import numpy as np

from examples.cable_threading.train_bc import (
    dataset_obs_keys,
    dataset_obs_modalities,
    ensure_dataset_training_metadata,
    prepare_resized_image_dataset,
)


def _write_dataset(path):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["obs_keys"] = json.dumps(["camera_image", "joint_pos"])
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("camera_image", data=np.zeros((2, 12, 16, 3), dtype=np.uint8))
        obs.create_dataset("joint_pos", data=np.zeros((2, 7), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))


def test_infers_image_and_low_dim_modalities(tmp_path):
    dataset = tmp_path / "dataset.hdf5"
    _write_dataset(dataset)

    keys = dataset_obs_keys(dataset)
    assert dataset_obs_modalities(dataset, keys) == {
        "rgb": ["camera_image"],
        "low_dim": ["joint_pos"],
    }


def test_prepares_resized_training_view_without_mutating_source(tmp_path):
    dataset = tmp_path / "dataset.hdf5"
    _write_dataset(dataset)

    resized = prepare_resized_image_dataset(dataset, tmp_path / "output", ["camera_image"], 8)

    with h5py.File(dataset, "r") as source:
        assert source["data/demo_0/obs/camera_image"].shape == (2, 12, 16, 3)
    with h5py.File(resized, "r") as prepared:
        assert prepared["data/demo_0/obs/camera_image"].shape == (2, 8, 8, 3)
        assert prepared["data/demo_0/obs/joint_pos"].shape == (2, 7)
        assert prepared["data/demo_0/actions"].shape == (2, 7)


def test_backfills_train_mask_for_standard_mimicgen_output(tmp_path):
    dataset = tmp_path / "dataset.hdf5"
    _write_dataset(dataset)

    ensure_dataset_training_metadata(dataset)

    with h5py.File(dataset, "r") as prepared:
        assert prepared["mask/train"][...].tolist() == [b"demo_0"]


def test_rejects_dataset_with_explicitly_invalid_demos(tmp_path):
    dataset = tmp_path / "dataset.hdf5"
    _write_dataset(dataset)
    with h5py.File(dataset, "r+") as prepared:
        prepared["data/demo_0"].attrs["valid_for_training"] = False

    try:
        ensure_dataset_training_metadata(dataset)
    except ValueError as exc:
        assert "valid_for_training" in str(exc)
    else:
        raise AssertionError("explicitly invalid demonstrations must not be trained")
