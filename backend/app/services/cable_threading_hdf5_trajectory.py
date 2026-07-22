from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_RGB_SUFFIX_HINTS = ("image", "rgb", "camera", "cam")

_DISPLAY_ORIENTATION_DEFAULTS = {
    "displayOrientation": "top_left",
    "rawStorageOrientation": "opengl_bottom_left",
    "displayTransformApplied": "vertical_flip",
    "displayOnlyTransform": True,
}

_DISPLAY_ORIENTATION_IDENTITY = {
    "displayOrientation": "top_left",
    "rawStorageOrientation": "top_left",
    "displayTransformApplied": "none",
    "displayOnlyTransform": False,
}


def resolve_job_hdf5_path(job_root: Path) -> Optional[Path]:
    candidate = job_root / "datasets" / "dataset.hdf5"
    return candidate if candidate.is_file() else None


def load_hdf5_dataset_display_metadata(hdf5_path: Path) -> dict[str, Any]:
    manifest_path = hdf5_path.parent / "dataset.manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read dataset manifest for display metadata: %s", exc)
        return {}


def _coerce_rgb_u8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr[..., :3].copy()


def normalize_hdf5_rgb_frame_for_display(
    frame: np.ndarray,
    *,
    task_type: str = "cable_threading",
    camera_name: str,
    dataset_metadata: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Display-only RGB orientation normalization; must not be used for training reads."""
    del camera_name  # reserved for future per-camera overrides
    meta = dict(dataset_metadata or {})
    arr = _coerce_rgb_u8(frame)

    if meta.get("displayOrientationNormalized") is True:
        return arr, dict(_DISPLAY_ORIENTATION_IDENTITY)

    orientation = str(meta.get("orientation") or meta.get("displayOrientation") or "").strip().lower()
    if orientation in {"top_left", "display_normalized"}:
        return arr, dict(_DISPLAY_ORIENTATION_IDENTITY)

    if task_type == "cable_threading":
        return arr[::-1].copy(), dict(_DISPLAY_ORIENTATION_DEFAULTS)

    return arr, {
        "displayOrientation": "unknown",
        "rawStorageOrientation": "unknown",
        "displayTransformApplied": "none",
        "displayOnlyTransform": False,
    }


def build_camera_display_orientation_info(
    *,
    camera_name: str,
    dataset_metadata: Optional[dict[str, Any]] = None,
    task_type: str = "cable_threading",
) -> dict[str, Any]:
    _, orientation = normalize_hdf5_rgb_frame_for_display(
        np.zeros((2, 2, 3), dtype=np.uint8),
        task_type=task_type,
        camera_name=camera_name,
        dataset_metadata=dataset_metadata,
    )
    return {
        "camera": camera_name,
        **orientation,
    }


def _list_demo_names(handle: h5py.File) -> list[str]:
    data_group = handle.get("data")
    if data_group is None:
        return []
    return sorted(str(key) for key in data_group.keys() if str(key).startswith("demo_"))


def _is_rgb_dataset(dataset: h5py.Dataset) -> bool:
    if dataset.ndim < 3:
        return False
    return int(dataset.shape[-1]) in {1, 3, 4}


def _list_rgb_cameras(obs_group: h5py.Group) -> list[str]:
    cameras: list[str] = []
    for key in obs_group.keys():
        node = obs_group[key]
        if isinstance(node, h5py.Dataset) and _is_rgb_dataset(node):
            cameras.append(str(key))
            continue
        key_lower = str(key).lower()
        if any(hint in key_lower for hint in _RGB_SUFFIX_HINTS):
            if isinstance(node, h5py.Dataset) and node.ndim >= 3:
                cameras.append(str(key))
    return sorted(set(cameras))


def _list_low_dim_obs_keys(obs_group: h5py.Group, rgb_cameras: set[str]) -> list[str]:
    keys: list[str] = []
    for key in obs_group.keys():
        if str(key) in rgb_cameras:
            continue
        node = obs_group[key]
        if isinstance(node, h5py.Dataset) and node.ndim >= 1 and node.shape[0] > 0:
            keys.append(str(key))
    return sorted(keys)


def _encode_frame_jpeg(frame: np.ndarray, *, quality: int = 85) -> bytes:
    import cv2

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("failed to encode frame as JPEG")
    return encoded.tobytes()


def get_demo_trajectory_meta(
    hdf5_path: Path,
    demo_name: str,
    *,
    dataset_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata = dataset_metadata if dataset_metadata is not None else load_hdf5_dataset_display_metadata(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        demos = _list_demo_names(handle)
        if demo_name not in demos:
            raise KeyError(f"demo not found: {demo_name}")

        demo = handle["data"][demo_name]
        actions = demo.get("actions")
        step_count = int(actions.shape[0]) if isinstance(actions, h5py.Dataset) else 0
        obs_group = demo.get("obs")
        rgb_cameras: list[str] = []
        low_dim_keys: list[str] = []
        if isinstance(obs_group, h5py.Group):
            rgb_cameras = _list_rgb_cameras(obs_group)
            low_dim_keys = _list_low_dim_obs_keys(obs_group, set(rgb_cameras))

        has_rgb = len(rgb_cameras) > 0
        display_mode = "rgb_frame_replay" if has_rgb else "state_trajectory"
        default_camera = rgb_cameras[0] if rgb_cameras else None
        default_orientation = (
            build_camera_display_orientation_info(
                camera_name=default_camera,
                dataset_metadata=metadata,
            )
            if default_camera
            else None
        )
        camera_display_info = {
            camera: build_camera_display_orientation_info(
                camera_name=camera,
                dataset_metadata=metadata,
            )
            for camera in rgb_cameras
        }
        payload: dict[str, Any] = {
            "demoName": demo_name,
            "stepCount": step_count,
            "actionDim": int(actions.shape[1]) if isinstance(actions, h5py.Dataset) and actions.ndim == 2 else None,
            "hasRgbObservation": has_rgb,
            "rgbCameras": rgb_cameras,
            "defaultCamera": default_camera,
            "lowDimObsKeys": low_dim_keys,
            "trajectoryDisplayMode": display_mode,
            "hasActions": isinstance(actions, h5py.Dataset),
            "hasStates": isinstance(demo.get("states"), h5py.Dataset),
            "cameraDisplayInfo": camera_display_info,
        }
        if default_orientation:
            payload.update(default_orientation)
        return payload


def get_demo_frame_jpeg(
    hdf5_path: Path,
    demo_name: str,
    *,
    camera: str,
    frame_index: int,
    quality: int = 85,
    dataset_metadata: Optional[dict[str, Any]] = None,
) -> bytes:
    metadata = dataset_metadata if dataset_metadata is not None else load_hdf5_dataset_display_metadata(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data"][demo_name]
        obs_group = demo.get("obs")
        if not isinstance(obs_group, h5py.Group) or camera not in obs_group:
            raise KeyError(f"camera not found: {camera}")
        dataset = obs_group[camera]
        if not isinstance(dataset, h5py.Dataset):
            raise KeyError(f"camera not found: {camera}")
        if frame_index < 0 or frame_index >= dataset.shape[0]:
            raise IndexError("frame index out of range")
        frame = np.asarray(dataset[frame_index])
        display_frame, _orientation = normalize_hdf5_rgb_frame_for_display(
            frame,
            task_type="cable_threading",
            camera_name=camera,
            dataset_metadata=metadata,
        )
        return _encode_frame_jpeg(display_frame, quality=quality)


def get_demo_step_detail(
    hdf5_path: Path,
    demo_name: str,
    *,
    step_index: int,
    low_dim_keys: Optional[list[str]] = None,
) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data"][demo_name]
        actions = demo.get("actions")
        if not isinstance(actions, h5py.Dataset):
            raise KeyError("actions missing")
        if step_index < 0 or step_index >= actions.shape[0]:
            raise IndexError("step index out of range")

        action = np.asarray(actions[step_index]).astype(float).tolist()
        obs_group = demo.get("obs")
        obs_values: dict[str, Any] = {}
        keys = low_dim_keys
        if keys is None and isinstance(obs_group, h5py.Group):
            rgb_cameras = set(_list_rgb_cameras(obs_group))
            keys = _list_low_dim_obs_keys(obs_group, rgb_cameras)
        if isinstance(obs_group, h5py.Group) and keys:
            for key in keys:
                node = obs_group.get(key)
                if isinstance(node, h5py.Dataset) and step_index < node.shape[0]:
                    obs_values[key] = np.asarray(node[step_index]).astype(float).tolist()

        reward = None
        rewards = demo.get("rewards")
        if isinstance(rewards, h5py.Dataset) and step_index < rewards.shape[0]:
            reward = float(np.asarray(rewards[step_index]))

        done = None
        dones = demo.get("dones")
        if isinstance(dones, h5py.Dataset) and step_index < dones.shape[0]:
            done = bool(int(np.asarray(dones[step_index])))

        return {
            "demoName": demo_name,
            "stepIndex": step_index,
            "action": action,
            "obs": obs_values,
            "reward": reward,
            "done": done,
        }


def inspect_hdf5_trajectory_capabilities(hdf5_path: Path) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as handle:
        demos = _list_demo_names(handle)
        if not demos:
            return {
                "hasRgbObservation": False,
                "rgbCameras": [],
                "trajectoryDisplayMode": "state_trajectory",
            }
        meta = get_demo_trajectory_meta(hdf5_path, demos[0])
        return {
            "hasRgbObservation": meta["hasRgbObservation"],
            "rgbCameras": meta["rgbCameras"],
            "trajectoryDisplayMode": meta["trajectoryDisplayMode"],
            "sampleStepCount": meta["stepCount"],
            "displayOrientation": meta.get("displayOrientation"),
            "rawStorageOrientation": meta.get("rawStorageOrientation"),
            "displayTransformApplied": meta.get("displayTransformApplied"),
            "displayOnlyTransform": meta.get("displayOnlyTransform"),
        }
