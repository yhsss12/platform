from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.cable_threading_hdf5_trajectory import (
    get_demo_frame_jpeg,
    get_demo_step_detail,
    get_demo_trajectory_meta,
    inspect_hdf5_trajectory_capabilities,
    normalize_hdf5_rgb_frame_for_display,
)


def _write_rgb_demo_hdf5(path: Path, demo_name: str, steps: int = 5) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    gradient = np.stack(
        [
            np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1)),
            np.zeros((32, 32), dtype=np.uint8),
            np.zeros((32, 32), dtype=np.uint8),
        ],
        axis=-1,
    )
    frames = np.stack([gradient + step for step in range(steps)], axis=0)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group(demo_name)
        demo.create_dataset("actions", data=np.zeros((steps, 7), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=frames)
        obs.create_dataset("robot0_eef_pos", data=np.zeros((steps, 3), dtype=np.float32))


def test_get_demo_trajectory_meta_with_rgb(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_rgb_demo_hdf5(hdf5_path, "demo_0", steps=8)

    meta = get_demo_trajectory_meta(hdf5_path, "demo_0")
    assert meta["stepCount"] == 8
    assert meta["hasRgbObservation"] is True
    assert meta["trajectoryDisplayMode"] == "rgb_frame_replay"
    assert "agentview_image" in meta["rgbCameras"]
    assert meta["displayTransformApplied"] == "vertical_flip"
    assert meta["cameraDisplayInfo"]["agentview_image"]["displayOnlyTransform"] is True


def test_normalize_hdf5_rgb_frame_for_display_skips_when_manifest_normalized(tmp_path: Path) -> None:
    raw = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    manifest = {"displayOrientationNormalized": True}
    display, info = normalize_hdf5_rgb_frame_for_display(
        raw,
        camera_name="agentview_image",
        dataset_metadata=manifest,
    )
    assert np.array_equal(display, raw[..., :3])
    assert info["displayTransformApplied"] == "none"


def test_get_demo_frame_jpeg_applies_vertical_flip_for_display(tmp_path: Path) -> None:
    import cv2
    import h5py

    hdf5_path = tmp_path / "dataset.hdf5"
    _write_rgb_demo_hdf5(hdf5_path, "demo_0", steps=2)

    with h5py.File(hdf5_path, "r") as handle:
        raw = np.asarray(handle["data/demo_0/obs/agentview_image"][0])

    frame = get_demo_frame_jpeg(hdf5_path, "demo_0", camera="agentview_image", frame_index=0)
    decoded = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    expected = cv2.cvtColor(np.flipud(raw), cv2.COLOR_RGB2BGR)
    assert decoded.shape == expected.shape
    assert float(np.mean(np.abs(decoded.astype(float) - expected.astype(float)))) < 3.0


def test_get_demo_step_detail(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_rgb_demo_hdf5(hdf5_path, "demo_1", steps=3)

    detail = get_demo_step_detail(hdf5_path, "demo_1", step_index=1)
    assert detail["demoName"] == "demo_1"
    assert detail["stepIndex"] == 1
    assert len(detail["action"]) == 7
    assert "robot0_eef_pos" in detail["obs"]


def test_get_demo_frame_jpeg(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_rgb_demo_hdf5(hdf5_path, "demo_0", steps=2)

    frame = get_demo_frame_jpeg(hdf5_path, "demo_0", camera="agentview_image", frame_index=0)
    assert frame[:2] == b"\xff\xd8"


def test_inspect_capabilities(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_rgb_demo_hdf5(hdf5_path, "demo_0")

    caps = inspect_hdf5_trajectory_capabilities(hdf5_path)
    assert caps["hasRgbObservation"] is True
    assert caps["trajectoryDisplayMode"] == "rgb_frame_replay"
