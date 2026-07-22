"""HDF5 camera observation helpers for Isaac block stacking datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from integrations.isaac_lab.camera_capture import (
    DEFAULT_AGENTVIEW_KEY,
    DEFAULT_WRIST_CAMERA_KEY,
    resize_rgb_frame,
)

IMAGE_KEY_HINTS = ("image", "rgb", "camera", "agentview", "eye_in_hand")


def is_image_obs_key(key: str) -> bool:
    lower = str(key).lower()
    return any(hint in lower for hint in IMAGE_KEY_HINTS)


def align_frame_sequence(
    frames: list[np.ndarray],
    target_length: int,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Align captured frames to action horizon; returns (T, H, W, 3) uint8."""
    target_length = max(0, int(target_length))
    if target_length == 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)

    resized = [resize_rgb_frame(frame, height, width) for frame in frames]
    if not resized:
        return np.zeros((target_length, height, width, 3), dtype=np.uint8)

    if len(resized) > target_length:
        resized = resized[-target_length:]
    elif len(resized) < target_length:
        pad_count = target_length - len(resized)
        resized = [resized[0]] * pad_count + resized

    stacked = np.stack(resized, axis=0)
    if stacked.dtype != np.uint8:
        stacked = stacked.astype(np.uint8)
    return stacked


def inject_camera_observations(
    hdf5_path: Path | str,
    demo_key: str,
    camera_frames: dict[str, list[np.ndarray]],
    *,
    height: int,
    width: int,
) -> None:
    """Write camera obs arrays under data/{demo_key}/obs/{camera_key}."""
    import h5py

    path = Path(hdf5_path).expanduser()
    demo_path = f"data/{demo_key}"
    with h5py.File(path, "a") as handle:
        if demo_path not in handle:
            raise KeyError(f"HDF5 demo not found: {demo_path}")
        demo = handle[demo_path]
        actions = demo.get("actions")
        if actions is None:
            raise KeyError(f"HDF5 demo missing actions: {demo_path}")
        target_length = int(actions.shape[0])
        obs_grp = demo.require_group("obs")
        for camera_key, frames in camera_frames.items():
            aligned = align_frame_sequence(
                frames,
                target_length,
                height=height,
                width=width,
            )
            if camera_key in obs_grp:
                del obs_grp[camera_key]
            obs_grp.create_dataset(
                camera_key,
                data=aligned,
                compression="gzip",
                compression_opts=4,
                chunks=(1, height, width, 3),
            )


def inspect_hdf5_observation_metadata(hdf5_path: Path | str) -> dict[str, Any]:
    """Inspect HDF5 obs keys and image metadata for manifest generation."""
    import h5py

    path = Path(hdf5_path).expanduser()
    result: dict[str, Any] = {
        "obsKeys": [],
        "cameraKeys": [],
        "imageKeys": [],
        "observationType": "low_dim",
        "imageShape": None,
        "actionDim": None,
        "episodeCount": 0,
        "horizon": None,
    }
    if not path.is_file():
        return result

    try:
        with h5py.File(path, "r") as handle:
            data_group = handle.get("data")
            if data_group is None:
                return result
            demo_keys = sorted(k for k in data_group.keys() if str(k).startswith("demo_"))
            result["episodeCount"] = len(demo_keys)
            if not demo_keys:
                return result

            first_demo = data_group[demo_keys[0]]
            obs_group = first_demo.get("obs")
            if obs_group is None:
                return result

            obs_keys: list[str] = []
            camera_keys: list[str] = []
            for key in obs_group.keys():
                key_str = str(key)
                if key_str in {"actions", "action"}:
                    continue
                obs_keys.append(key_str)
                ds = obs_group[key]
                shape = getattr(ds, "shape", ())
                if is_image_obs_key(key_str) or (shape and len(shape) >= 4 and int(shape[-1]) == 3):
                    camera_keys.append(key_str)
                    if result["imageShape"] is None and shape and len(shape) >= 4:
                        result["imageShape"] = {
                            "height": int(shape[-3]),
                            "width": int(shape[-2]),
                            "channels": int(shape[-1]),
                        }

            result["obsKeys"] = obs_keys
            result["cameraKeys"] = camera_keys
            result["imageKeys"] = list(camera_keys)
            if camera_keys and len(camera_keys) == len(obs_keys):
                result["observationType"] = "image"
            elif camera_keys:
                result["observationType"] = "mixed"
            else:
                result["observationType"] = "low_dim"

            actions = first_demo.get("actions")
            if actions is not None and getattr(actions, "shape", None):
                shape = actions.shape
                if len(shape) >= 2:
                    result["actionDim"] = int(shape[-1])
                    result["horizon"] = int(shape[0])
    except OSError:
        return result

    return result


def build_observation_manifest_fields(
    hdf5_path: Path | str,
    *,
    simulator: str = "Isaac",
    robot_type: str = "Panda",
    data_format: str = "HDF5",
) -> dict[str, Any]:
    """Build manifest-compatible observation metadata from an HDF5 dataset."""
    meta = inspect_hdf5_observation_metadata(hdf5_path)
    has_image = bool(meta.get("cameraKeys"))
    fields: dict[str, Any] = {
        "obsKeys": meta.get("obsKeys") or [],
        "observationType": meta.get("observationType") or "low_dim",
        "cameraKeys": meta.get("cameraKeys") or [],
        "imageKeys": meta.get("imageKeys") or [],
        "simulator": simulator,
        "robotType": robot_type,
        "format": data_format,
        "observationSpace": {
            "type": meta.get("observationType") or "low_dim",
            "keys": meta.get("obsKeys") or [],
        },
        "quality": {
            "hasImage": has_image,
        },
    }
    if meta.get("imageShape"):
        fields["imageShape"] = meta["imageShape"]
    if meta.get("actionDim") is not None:
        fields["actionDim"] = meta["actionDim"]
    if meta.get("horizon") is not None:
        fields["horizon"] = meta["horizon"]
    return fields


def default_camera_keys(*, include_wrist: bool = False) -> tuple[str, ...]:
    keys = [DEFAULT_AGENTVIEW_KEY]
    if include_wrist:
        keys.append(DEFAULT_WRIST_CAMERA_KEY)
    return tuple(keys)
