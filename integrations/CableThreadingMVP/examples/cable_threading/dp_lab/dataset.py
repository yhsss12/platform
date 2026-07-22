from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DpLabConfig
from .normalizer import DatasetStats, LinearNormalizer


def _load_action_window(grp, cfg: DpLabConfig, start: int, end: int) -> np.ndarray:
    action_key = cfg.action_key
    if action_key not in grp:
        raise KeyError(f"HDF5 demo missing action key {action_key!r}")
    actions = np.asarray(grp[action_key][start:end], dtype=np.float32)
    if cfg.gripper_action_key and cfg.gripper_action_key in grp:
        gripper = np.asarray(grp[cfg.gripper_action_key][start:end], dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        actions = np.concatenate([actions, gripper], axis=-1)
    if actions.ndim != 2 or actions.shape[-1] != cfg.action_dim:
        raise ValueError(
            f"HDF5 action window shape {actions.shape} != expected (T, {cfg.action_dim})"
        )
    return actions


def _load_action_array(grp, cfg: DpLabConfig) -> np.ndarray:
    return _load_action_window(grp, cfg, 0, int(grp[cfg.action_key].shape[0]))


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


def compute_dataset_stats(hdf5_path: Path | str | list[Path | str], cfg: DpLabConfig) -> DatasetStats:
    paths = [Path(p).expanduser().resolve() for p in (hdf5_path if isinstance(hdf5_path, list) else [hdf5_path])]
    actions: list[np.ndarray] = []
    low_dims: list[np.ndarray] = []
    for path in paths:
        demo_names = _load_demo_names(path, split="train")
        with h5py.File(path, "r") as f:
            for demo in demo_names:
                grp = f["data"][demo]
                action_arr = _load_action_array(grp, cfg)
                actions.append(action_arr)
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
        hdf5_path: str | Path | list[str | Path],
        cfg: DpLabConfig,
        stats: DatasetStats,
        *,
        split: str = "train",
        image_size: int | None = None,
        max_windows: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.stats = stats
        self.split = split
        self.image_size = image_size or cfg.image_size
        self.hdf5_paths = [Path(p).expanduser().resolve() for p in (hdf5_path if isinstance(hdf5_path, list) else [hdf5_path])]
        if not self.hdf5_paths:
            raise ValueError("at least one HDF5 path is required")

        self.demo_names: list[str] = []
        for path in self.hdf5_paths:
            self.demo_names.extend(_load_demo_names(path, split=split))
        self.demo_names = sorted(set(self.demo_names))
        if not self.demo_names:
            raise ValueError(f"no demos found for split={split!r} in {self.hdf5_paths}")

        self._index: list[tuple[Path, str, int]] = []
        for path in self.hdf5_paths:
            with h5py.File(path, "r") as f:
                for demo in _load_demo_names(path, split=split):
                    grp = f["data"][demo]
                    action_key = cfg.action_key
                    if action_key not in grp:
                        raise KeyError(f"HDF5 demo {demo!r} missing action key {action_key!r}")
                    action_arr = _load_action_array(grp, cfg)
                    length = int(grp[action_key].shape[0])
                    max_start = length - cfg.horizon
                    min_start = cfg.n_obs_steps - 1
                    if max_start < min_start:
                        continue
                    for start in range(min_start, max_start + 1):
                        self._index.append((path, demo, start))

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
        path, demo, start = self._index[idx]
        cfg = self.cfg
        obs_start = start - cfg.n_obs_steps + 1
        obs_end = start + 1
        act_end = start + cfg.horizon

        with h5py.File(path, "r") as f:
            grp = f["data"][demo]
            obs_grp = grp["obs"]
            actions = _load_action_window(grp, cfg, start, act_end)

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

        if images:
            images_arr = np.stack(images, axis=1)  # (T, N_cam, H, W, C)
            images_arr = images_arr.astype(np.float32) / 255.0
            images_arr = np.transpose(images_arr, (0, 1, 4, 2, 3))  # (T, N_cam, C, H, W)
        else:
            images_arr = np.zeros((cfg.n_obs_steps, 0, 3, 1, 1), dtype=np.float32)

        action_norm = self.stats.action.normalize(actions)
        low_dim_norm = self.stats.low_dim.normalize(low_dim)

        return {
            "images": torch.from_numpy(images_arr),
            "low_dim": torch.from_numpy(low_dim_norm),
            "actions": torch.from_numpy(action_norm),
        }
