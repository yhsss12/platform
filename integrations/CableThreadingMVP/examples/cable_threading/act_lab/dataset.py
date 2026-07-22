from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ActLabConfig


def _load_action_window(grp, cfg: ActLabConfig, start: int, end: int) -> np.ndarray:
    action_key = cfg.action_key
    if action_key not in grp:
        raise KeyError(f"HDF5 demo missing action key {action_key!r}")
    actions = np.asarray(grp[action_key][start:end], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    if (
        cfg.gripper_action_key
        and cfg.gripper_action_key in grp
        and actions.shape[-1] < cfg.action_dim
    ):
        gripper = np.asarray(grp[cfg.gripper_action_key][start:end], dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        actions = np.concatenate([actions, gripper], axis=-1)
    if actions.ndim != 2 or actions.shape[-1] != cfg.action_dim:
        raise ValueError(
            f"HDF5 action window shape {actions.shape} != expected (T, {cfg.action_dim})"
        )
    return actions


def _load_action_array(grp, cfg: ActLabConfig) -> np.ndarray:
    action_key = cfg.action_key
    return _load_action_window(grp, cfg, 0, int(grp[action_key].shape[0]))


def _load_demo_names(hdf5_path: Path, split: str = "train") -> list[str]:
    with h5py.File(hdf5_path, "r") as handle:
        if f"mask/{split}" in handle:
            raw = handle[f"mask/{split}"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(raw, np.ndarray):
                return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in raw]
            return list(raw)
        return sorted(k for k in handle["data"].keys() if k.startswith("demo_"))


def _validate_obs_keys(hdf5_path: Path, cfg: ActLabConfig) -> dict[str, tuple[int, ...]]:
    with h5py.File(hdf5_path, "r") as handle:
        demos = _load_demo_names(hdf5_path, split="train")
        if not demos:
            raise ValueError("HDF5 中无 demo_* 轨迹")
        sample = handle["data"][demos[0]]
        obs = sample.get("obs")
        if obs is None:
            raise ValueError("HDF5 demo 缺少 obs 分组")
        shapes: dict[str, tuple[int, ...]] = {}
        missing: list[str] = []
        for key in cfg.image_keys + cfg.low_dim_keys:
            if key not in obs:
                missing.append(key)
            else:
                shapes[key] = tuple(obs[key].shape)
        if missing:
            raise ValueError(f"HDF5 obs 缺少键: {', '.join(missing)}")
        action_key = cfg.action_key
        if action_key not in sample:
            raise ValueError(f"HDF5 demo 缺少 action key {action_key!r}")
        shapes[action_key] = tuple(sample[action_key].shape)
        if cfg.gripper_action_key and cfg.gripper_action_key in sample:
            shapes[cfg.gripper_action_key] = tuple(sample[cfg.gripper_action_key].shape)
        return shapes


def _split_demos(demos: list[str], *, val_ratio: float, split: str) -> list[str]:
    if not demos or val_ratio <= 0:
        return demos
    val_count = max(1, int(round(len(demos) * val_ratio)))
    train_demos = demos[:-val_count] if len(demos) > val_count else demos
    val_demos = demos[-val_count:]
    if split == "val":
        return val_demos
    return train_demos


def _read_image(obs, key: str, t: int, image_size: int) -> np.ndarray:
    raw = np.asarray(obs[key][t], dtype=np.float32)
    if raw.ndim == 1 and raw.size <= 4:
        raise ValueError(f"观测键 {key!r} 不是有效图像数组，shape={raw.shape}")
    if raw.ndim == 3:
        img = raw
    elif raw.ndim == 4:
        img = raw[0]
    else:
        raise ValueError(f"观测键 {key!r} 图像维度不支持: {raw.shape}")
    if img.shape[-1] not in {1, 3, 4}:
        if img.shape[0] in {1, 3, 4}:
            img = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError(f"观测键 {key!r} 图像通道布局未知: {img.shape}")
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    if image_size and (img.shape[0] != image_size or img.shape[1] != image_size):
        import torch.nn.functional as F

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    if img.max() > 1.5:
        img = img / 255.0
    return img.astype(np.float32)


class ActDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str | Path,
        cfg: ActLabConfig,
        *,
        split: str = "train",
        max_samples: int | None = None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser().resolve()
        self.cfg = cfg
        self.split = split
        _validate_obs_keys(self.hdf5_path, cfg)

        all_demos = _load_demo_names(self.hdf5_path, split="train")
        self.demo_names = _split_demos(all_demos, val_ratio=cfg.val_ratio, split=split)
        if not self.demo_names:
            raise ValueError(f"split={split!r} 无可用 demo")

        self._index: list[tuple[str, int]] = []
        with h5py.File(self.hdf5_path, "r") as handle:
            for demo in self.demo_names:
                horizon = int(handle["data"][demo]["actions"].shape[0])
                for t in range(horizon):
                    self._index.append((demo, t))

        if max_samples is not None and len(self._index) > max_samples:
            self._index = self._index[:max_samples]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        demo, t = self._index[idx]
        chunk = self.cfg.chunk_size
        with h5py.File(self.hdf5_path, "r") as handle:
            grp = handle["data"][demo]
            obs = grp["obs"]
            horizon = int(grp[self.cfg.action_key].shape[0])
            end = min(t + chunk, horizon)
            action_chunk = _load_action_window(grp, self.cfg, t, end)
            pad_len = chunk - action_chunk.shape[0]
            is_pad = np.zeros((chunk,), dtype=np.float32)
            if pad_len > 0:
                pad = np.zeros((pad_len, self.cfg.action_dim), dtype=np.float32)
                action_chunk = np.concatenate([action_chunk, pad], axis=0)
                is_pad[end - t :] = 1.0

            images = []
            for key in self.cfg.image_keys:
                images.append(_read_image(obs, key, t, self.cfg.image_size))
            image_arr = np.stack(images, axis=0) if images else np.zeros((0, 3, 3, 3), dtype=np.float32)
            if image_arr.ndim == 4 and image_arr.shape[-1] in {1, 3, 4}:
                image_arr = np.transpose(image_arr, (0, 3, 1, 2))
                if image_arr.shape[1] == 1:
                    image_arr = np.repeat(image_arr, 3, axis=1)
                elif image_arr.shape[1] == 4:
                    image_arr = image_arr[:, :3]

            proprio_parts = []
            for key in self.cfg.low_dim_keys:
                value = np.asarray(obs[key][t], dtype=np.float32).reshape(-1)
                proprio_parts.append(value)
            proprio = np.concatenate(proprio_parts, axis=0) if proprio_parts else np.zeros((0,), dtype=np.float32)

        return {
            "images": torch.from_numpy(image_arr),
            "proprio": torch.from_numpy(proprio),
            "actions": torch.from_numpy(action_chunk),
            "is_pad": torch.from_numpy(is_pad),
        }


def inspect_hdf5(hdf5_path: Path) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as handle:
        demos = sorted(k for k in handle["data"].keys() if k.startswith("demo_"))
        sample = handle["data"][demos[0]]
        obs_shapes = {k: tuple(sample["obs"][k].shape) for k in sample["obs"].keys()}
        return {
            "path": str(hdf5_path),
            "num_demos": len(demos),
            "obs_shapes": obs_shapes,
            "action_shape": tuple(sample["actions"].shape),
        }
