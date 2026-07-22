"""NutAssembly-PINN v1: feature construction, MLP model, and trajectory delta helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

SEGMENT_LEN = 48
ACTION_DIM = 7
FEATURE_DIM = 343
DELTA_DIM = SEGMENT_LEN * 3
XY_OFFSETS_M = (0.01, 0.02, 0.03, 0.05)


def _demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(k for k in data_group.keys() if k.startswith("demo_"))


def _pose_xy(mat: np.ndarray) -> np.ndarray:
    return np.asarray(mat[:2, 3], dtype=np.float64)


def _pose_z(mat: np.ndarray) -> float:
    return float(mat[2, 3])


def _tilt_from_mat(mat: np.ndarray) -> float:
    z_axis = np.asarray(mat[:3, :3], dtype=np.float64)[:, 2]
    return float(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))


def _pad_sequence(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) >= length:
        return arr[:length]
    if len(arr) == 0:
        return np.zeros((length, *arr.shape[1:]), dtype=np.float32)
    pad = np.repeat(arr[-1:], length - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)


def _find_insert_start(dg: h5py.Group, total: int) -> int:
    sts = dg.get("subtask_term_signals")
    if sts is None:
        return max(total // 2, 1)
    for sig_name in ("insert_square_nut", "insert_round_nut", "grasp_square_nut"):
        if sig_name in sts:
            sig = np.asarray(sts[sig_name], dtype=np.float64).reshape(-1)
            hits = np.where(sig > 0.5)[0]
            if len(hits) > 0:
                return max(int(hits[0]) - 8, 0)
    return max(total // 2, 1)


def extract_align_insert_segment(
    demo_grp: h5py.Group,
    *,
    segment_len: int = SEGMENT_LEN,
) -> dict[str, np.ndarray]:
    obs = demo_grp.get("obs")
    if obs is not None and "robot0_eef_pos" in obs:
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    elif "robot0_eef_pos" in demo_grp:
        eef = np.asarray(demo_grp["robot0_eef_pos"], dtype=np.float64)
    else:
        raise KeyError("robot0_eef_pos not found in demo group")
    total = len(eef)
    dg = demo_grp.get("datagen_info")
    start = 0
    nut_mats = peg_mats = None
    if dg is not None and "object_poses" in dg:
        op = dg["object_poses"]
        for nk in ("square_nut", "round_nut"):
            if nk in op:
                nut_mats = np.asarray(op[nk], dtype=np.float64)
                break
        for pk in ("square_peg", "round_peg"):
            if pk in op:
                peg_mats = np.asarray(op[pk], dtype=np.float64)
                break
        start = _find_insert_start(dg, total)
    end = min(start + segment_len, total)
    sl = slice(start, end)
    eef_seg = _pad_sequence(eef[sl], segment_len)
    if nut_mats is not None and peg_mats is not None:
        nut_seg = _pad_sequence(nut_mats[sl], segment_len)
        peg_seg = _pad_sequence(peg_mats[sl], segment_len)
        rel_xy = nut_seg[:, :2, 3] - peg_seg[:, :2, 3]
        rel_z = (nut_seg[:, 2, 3] - peg_seg[:, 2, 3]).reshape(-1, 1)
        tilt = np.array([_tilt_from_mat(m) for m in nut_seg], dtype=np.float64).reshape(-1, 1)
    else:
        rel_xy = np.zeros((segment_len, 2), dtype=np.float64)
        rel_z = np.zeros((segment_len, 1), dtype=np.float64)
        tilt = np.zeros((segment_len, 1), dtype=np.float64)
    actions = np.asarray(demo_grp["actions"], dtype=np.float64)
    act_seg = _pad_sequence(actions[sl], segment_len)
    return {
        "eef_seg": eef_seg.astype(np.float32),
        "rel_xy": rel_xy.astype(np.float32),
        "rel_z": rel_z.astype(np.float32),
        "tilt": tilt.astype(np.float32),
        "actions_seg": act_seg.astype(np.float32),
        "start_idx": np.array([start], dtype=np.int32),
    }


def apply_eef_perturbation(
    eef_seg: np.ndarray,
    *,
    xy_offset: tuple[float, float] = (0.0, 0.0),
    z_offset: float = 0.0,
    action_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    out = eef_seg.copy()
    start = int(len(out) * 0.55)
    out[start:, 0] += xy_offset[0]
    out[start:, 1] += xy_offset[1]
    out[start:, 2] += z_offset
    if action_noise > 0:
        out[start:] += rng.normal(0, action_noise, out[start:].shape)
    return out


def build_feature_vector(
    segment: dict[str, np.ndarray],
    *,
    perturbed_eef: np.ndarray,
    xy_offset_m: float = 0.0,
    extra_scalars: list[float] | None = None,
) -> np.ndarray:
    eef_flat = perturbed_eef.reshape(-1)
    rel_xy_flat = segment["rel_xy"].reshape(-1)
    rel_z_flat = segment["rel_z"].reshape(-1)
    tilt_flat = segment["tilt"].reshape(-1)
    final_xy = float(np.linalg.norm(segment["rel_xy"][-1]))
    final_z = float(segment["rel_z"][-1, 0])
    tilt_last = float(segment["tilt"][-1, 0])
    act = segment["actions_seg"]
    smooth = float(1.0 / (1.0 + np.mean(np.abs(np.diff(act, axis=0))))) if len(act) > 1 else 1.0
    scalars = [
        xy_offset_m,
        final_xy,
        final_z,
        tilt_last,
        smooth,
        float(np.max(np.linalg.norm(segment["rel_xy"], axis=1))),
        float(np.std(segment["rel_z"])),
    ]
    if extra_scalars:
        scalars.extend(extra_scalars)
    while len(scalars) < 55:
        scalars.append(0.0)
    scalars = scalars[:55]
    feat = np.concatenate([eef_flat, rel_xy_flat, rel_z_flat, tilt_flat, np.asarray(scalars, dtype=np.float32)])
    if feat.shape[0] < FEATURE_DIM:
        feat = np.pad(feat, (0, FEATURE_DIM - feat.shape[0]))
    return feat[:FEATURE_DIM].astype(np.float32)


def build_delta_vector(clean_eef: np.ndarray, perturbed_eef: np.ndarray) -> np.ndarray:
    delta = (clean_eef - perturbed_eef).reshape(-1)
    if delta.shape[0] < DELTA_DIM:
        delta = np.pad(delta, (0, DELTA_DIM - delta.shape[0]))
    return delta[:DELTA_DIM].astype(np.float32)


def delta_to_xy_bias(delta: np.ndarray) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64).reshape(SEGMENT_LEN, 3)
    tail = arr[max(SEGMENT_LEN // 2, 0) :]
    if len(tail) == 0:
        return np.zeros(2, dtype=np.float64)
    return np.mean(tail[:, :2], axis=0)


def compute_align_error_vector(demo_grp: h5py.Group) -> np.ndarray | None:
    dg = demo_grp.get("datagen_info")
    if dg is None or "object_poses" not in dg:
        return None
    op = dg["object_poses"]
    nut_key = "square_nut" if "square_nut" in op else ("round_nut" if "round_nut" in op else None)
    peg_key = "square_peg" if "square_peg" in op else ("round_peg" if "round_peg" in op else None)
    if not nut_key or not peg_key:
        return None
    nut = np.asarray(op[nut_key], dtype=np.float64)
    peg = np.asarray(op[peg_key], dtype=np.float64)
    err = _pose_xy(nut[-1]) - _pose_xy(peg[-1])
    return err.astype(np.float64)


if nn is not None:

    class TrajectoryDeltaMLP(nn.Module):
        def __init__(self, input_dim: int = FEATURE_DIM, output_dim: int = DELTA_DIM, hidden_dim: int = 256, num_layers: int = 3):
            super().__init__()
            layers: list[nn.Module] = []
            d = input_dim
            for _ in range(num_layers):
                layers.extend([nn.Linear(d, hidden_dim), nn.ReLU()])
                d = hidden_dim
            layers.append(nn.Linear(d, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def load_torch_model(model_path: Path) -> tuple[Any, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("torch not available")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    meta = {k: ckpt[k] for k in ("input_dim", "output_dim", "hidden_dim", "num_layers", "model_id", "pipeline_version") if k in ckpt}
    model = TrajectoryDeltaMLP(
        input_dim=int(meta.get("input_dim", FEATURE_DIM)),
        output_dim=int(meta.get("output_dim", DELTA_DIM)),
        hidden_dim=int(meta.get("hidden_dim", 256)),
        num_layers=int(meta.get("num_layers", 3)),
    )
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state)
    model.eval()
    return model, meta


def predict_trajectory_delta(model: Any, features: np.ndarray) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch not available")
    x = torch.from_numpy(np.asarray(features, dtype=np.float32).reshape(1, -1))
    with torch.no_grad():
        out = model(x).cpu().numpy().reshape(-1)
    return out[:DELTA_DIM].astype(np.float32)


def build_features_from_demo_group(
    demo_grp: h5py.Group,
    *,
    xy_offset_m: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = rng or np.random.default_rng(0)
    segment = extract_align_insert_segment(demo_grp)
    clean_eef = segment["eef_seg"]
    angle = rng.uniform(0, 2 * np.pi)
    xy_off = (xy_offset_m * np.cos(angle), xy_offset_m * np.sin(angle))
    perturbed = apply_eef_perturbation(clean_eef, xy_offset=xy_off, z_offset=rng.uniform(-0.005, 0.005), action_noise=0.002, rng=rng)
    feat = build_feature_vector(segment, perturbed_eef=perturbed, xy_offset_m=xy_offset_m)
    return feat, {"segment": segment, "clean_eef": clean_eef, "perturbed_eef": perturbed, "xy_offset": xy_off}


def load_demo_group(hdf5_path: Path, demo_key: str) -> h5py.Group:
    f = h5py.File(hdf5_path, "r")
    return f["data"][demo_key]
