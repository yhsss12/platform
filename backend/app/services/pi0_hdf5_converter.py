"""Convert platform HDF5 demonstrations to LeRobot-style on-disk dataset for openpi/pi0."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _write_parquet_or_npz(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write episode frame data as parquet when pyarrow is available, else .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({key: pa.array(value) for key, value in columns.items()})
        pq.write_table(table, path)
    except ImportError:
        np.savez_compressed(path.with_suffix(".npz"), **columns)
        if path.suffix == ".parquet":
            path.unlink(missing_ok=True)


def convert_hdf5_to_lerobot_dataset(
    *,
    hdf5_path: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    camera_keys: list[str],
    low_dim_keys: list[str],
    task_prompt: str,
) -> Path:
    """
    Export HDF5 demos to ``output_dir/lerobot_dataset`` with meta + per-episode data files.

    Produces a LeRobot-like layout consumable by openpi configs that expect on-disk datasets.
    """
    import h5py

    hdf5_path = hdf5_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    lerobot_root = output_dir / "lerobot_dataset"
    meta_dir = lerobot_root / "meta"
    data_dir = lerobot_root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta: list[dict[str, Any]] = []
    total_frames = 0
    action_dim = 0

    with h5py.File(hdf5_path, "r") as handle:
        data_group = handle["data"]
        demo_keys = sorted(k for k in data_group.keys() if str(k).startswith("demo_"))
        if not demo_keys:
            raise ValueError("HDF5 data 分组内无 demo_* 轨迹")

        for episode_index, demo_key in enumerate(demo_keys):
            demo = data_group[demo_key]
            actions = np.asarray(demo["actions"], dtype=np.float32)
            obs = demo["obs"]
            horizon = int(actions.shape[0])
            action_dim = int(actions.shape[-1]) if actions.ndim > 1 else 1
            total_frames += horizon

            columns: dict[str, np.ndarray] = {
                "action": actions.reshape(horizon, -1),
                "episode_index": np.full((horizon,), episode_index, dtype=np.int64),
                "frame_index": np.arange(horizon, dtype=np.int64),
                "timestamp": np.arange(horizon, dtype=np.float32) / 30.0,
                "task_index": np.zeros((horizon,), dtype=np.int64),
            }

            if low_dim_keys:
                state_parts = []
                for key in low_dim_keys:
                    if key not in obs:
                        raise ValueError(f"HDF5 obs 缺少 low_dim 键: {key}")
                    state_parts.append(np.asarray(obs[key], dtype=np.float32).reshape(horizon, -1))
                columns["observation.state"] = np.concatenate(state_parts, axis=-1)

            for cam_key in camera_keys:
                if cam_key not in obs:
                    raise ValueError(f"HDF5 obs 缺少图像键: {cam_key}")
                images = np.asarray(obs[cam_key])
                if images.ndim == 4 and images.shape[-1] in {1, 3, 4}:
                    columns[f"observation.images.{cam_key}"] = images[..., :3]
                else:
                    raise ValueError(f"图像键 {cam_key} 维度异常: {images.shape}")

            episode_path = data_dir / f"episode_{episode_index:06d}.parquet"
            _write_parquet_or_npz(episode_path, columns)

            episodes_meta.append(
                {
                    "episode_index": episode_index,
                    "episode_id": str(demo_key),
                    "length": horizon,
                    "task": task_prompt,
                }
            )

    info = {
        "codebase_version": "platform_lerobot_export_v1",
        "robot_type": manifest.get("robotType") or "unknown",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "fps": 30,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "action": {"dtype": "float32", "shape": [action_dim]},
            "observation.state": {
                "dtype": "float32",
                "shape": [int(columns.get("observation.state", np.zeros((1, 0))).shape[-1])],
            }
            if low_dim_keys
            else None,
            **{
                f"observation.images.{key}": {
                    "dtype": "image",
                    "shape": [3, 128, 128],
                    "names": ["channels", "height", "width"],
                }
                for key in camera_keys
            },
        },
        "task_prompt": task_prompt,
        "source_hdf5": str(hdf5_path),
        "camera_keys": camera_keys,
        "low_dim_keys": low_dim_keys,
    }
    info = {k: v for k, v in info.items() if v is not None}
    (meta_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_index": 0, "task": task_prompt}, ensure_ascii=False) + "\n")

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in episodes_meta:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    index = {
        "format": "platform_lerobot_export_v1",
        "repo_id": f"platform/{manifest.get('datasetId') or 'dataset'}",
        "root": str(lerobot_root),
        "sourceHdf5": str(hdf5_path),
        "taskPrompt": task_prompt,
        "cameraKeys": camera_keys,
        "lowDimKeys": low_dim_keys,
        "episodes": episodes_meta,
        "episodeCount": len(episodes_meta),
        "totalFrames": total_frames,
    }
    (lerobot_root / "dataset_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "exported lerobot dataset: %s (%s episodes, %s frames)",
        lerobot_root,
        len(episodes_meta),
        total_frames,
    )
    return lerobot_root


# Backward-compatible alias used by older imports
convert_hdf5_to_lerobot_index = convert_hdf5_to_lerobot_dataset
