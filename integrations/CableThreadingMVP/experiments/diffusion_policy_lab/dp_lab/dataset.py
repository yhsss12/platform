from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dp_lab.config import DpLabConfig
from dp_lab.normalizer import DatasetStats, LinearNormalizer


def _load_demo_names(hdf5_path: Path, split: str = "train") -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        if f"mask/{split}" in f:
            raw = f[f"mask/{split}"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(raw, np.ndarray):
                return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in raw]
            return list(raw)
        return sorted(k for k in f["data"].keys() if k.startswith("demo_"))


def _resize_image_hwc(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    import torch.nn.functional as F

    tensor = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)


def compute_dataset_stats(hdf5_path: Path, cfg: DpLabConfig) -> DatasetStats:
    actions: list[np.ndarray] = []
    low_dims: list[np.ndarray] = []
    demo_names = _load_demo_names(hdf5_path, split="train")

    with h5py.File(hdf5_path, "r") as f:
        for demo in demo_names:
            grp = f["data"][demo]
            actions.append(np.asarray(grp["actions"], dtype=np.float32))
            obs = grp["obs"]
            parts = [np.asarray(obs[key], dtype=np.float32).reshape(len(actions[-1]), -1) for key in cfg.low_dim_keys]
            low_dims.append(np.concatenate(parts, axis=-1))

    action_arr = np.concatenate(actions, axis=0)
    low_dim_arr = np.concatenate(low_dims, axis=0)
    return DatasetStats(
        action=LinearNormalizer.fit(action_arr),
        low_dim=LinearNormalizer.fit(low_dim_arr),
    )


def inspect_hdf5(hdf5_path: Path) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(k for k in f["data"].keys() if k.startswith("demo_"))
        obs_keys_attr = f["data"].attrs.get("obs_keys")
        if isinstance(obs_keys_attr, bytes):
            obs_keys_attr = obs_keys_attr.decode("utf-8")
        obs_keys = json.loads(obs_keys_attr) if obs_keys_attr else []
        sample = f["data"][demos[0]]
        obs_shapes = {k: tuple(sample["obs"][k].shape) for k in sample["obs"].keys()}
        masks = list(f["mask"].keys()) if "mask" in f else []
        return {
            "path": str(hdf5_path),
            "num_demos": len(demos),
            "obs_keys": obs_keys,
            "obs_shapes": obs_shapes,
            "action_shape": tuple(sample["actions"].shape),
            "masks": masks,
        }


class CableThreadingDpDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str | Path,
        cfg: DpLabConfig,
        stats: DatasetStats,
        *,
        split: str = "train",
        image_size: int | None = None,
        max_windows: int | None = None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser().resolve()
        self.cfg = cfg
        self.stats = stats
        self.split = split
        self.image_size = image_size or cfg.image_size
        self.demo_names = _load_demo_names(self.hdf5_path, split=split)
        if not self.demo_names:
            raise ValueError(f"no demos found for split={split!r} in {self.hdf5_path}")

        self._index: list[tuple[str, int]] = []
        with h5py.File(self.hdf5_path, "r") as f:
            for demo in self.demo_names:
                length = int(f["data"][demo]["actions"].shape[0])
                max_start = length - cfg.horizon
                min_start = cfg.n_obs_steps - 1
                if max_start < min_start:
                    continue
                for start in range(min_start, max_start + 1):
                    self._index.append((demo, start))

        if not self._index:
            raise ValueError(
                f"no valid training windows (need length >= horizon={cfg.horizon}, "
                f"n_obs_steps={cfg.n_obs_steps})"
            )
        if max_windows is not None and max_windows > 0:
            self._index = self._index[:max_windows]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo, start = self._index[idx]
        cfg = self.cfg
        obs_start = start - cfg.n_obs_steps + 1
        obs_end = start + 1
        act_end = start + cfg.horizon

        with h5py.File(self.hdf5_path, "r") as f:
            grp = f["data"][demo]
            obs_grp = grp["obs"]
            actions = np.asarray(grp["actions"][start:act_end], dtype=np.float32)

            low_parts = []
            for t in range(obs_start, obs_end):
                parts = [np.asarray(obs_grp[key][t], dtype=np.float32).reshape(-1) for key in cfg.low_dim_keys]
                low_parts.append(np.concatenate(parts, axis=0))
            low_dim = np.stack(low_parts, axis=0)

            images = []
            for key in cfg.image_keys:
                cam = np.asarray(obs_grp[key][obs_start:obs_end], dtype=np.uint8)
                if self.image_size != cam.shape[1]:
                    cam = _resize_image_hwc(cam, self.image_size)
                images.append(cam)

        images_arr = np.stack(images, axis=1)  # (T, N_cam, H, W, C)
        images_arr = images_arr.astype(np.float32) / 255.0
        images_arr = np.transpose(images_arr, (0, 1, 4, 2, 3))  # (T, N_cam, C, H, W)

        action_norm = self.stats.action.normalize(actions)
        low_dim_norm = self.stats.low_dim.normalize(low_dim)

        return {
            "images": torch.from_numpy(images_arr),
            "low_dim": torch.from_numpy(low_dim_norm),
            "actions": torch.from_numpy(action_norm),
        }
