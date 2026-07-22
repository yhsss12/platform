"""Regression tests: HDF5 export must match NPZ trajectories and replay to success."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from examples.cable_threading.hdf5_replay import load_npz_trajectories, replay_hdf5_demo

_DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "cable_threading"
    / "jobs"
    / "ct_gen_20260618_095819_8aa6"
    / "datasets"
)


def _dataset_paths() -> tuple[Path, Path]:
    return _DEFAULT_DATASET_DIR / "dataset.npz", _DEFAULT_DATASET_DIR / "dataset.hdf5"


@pytest.mark.integration
def test_hdf5_actions_match_npz_step_counts():
    npz_path, hdf5_path = _dataset_paths()
    if not npz_path.is_file() or not hdf5_path.is_file():
        pytest.skip("cable threading dataset artifacts not present")

    trajectories, episode_metadata, _ = load_npz_trajectories(npz_path)
    with h5py.File(hdf5_path, "r") as f:
        for ep_idx, traj in enumerate(trajectories):
            demo = f"demo_{ep_idx}"
            h5_actions = np.asarray(f["data"][demo]["actions"], dtype=np.float32)
            npz_actions = np.stack([step["action"] for step in traj], axis=0)
            assert h5_actions.shape[0] == len(traj), (
                f"demo_{ep_idx}: HDF5 {h5_actions.shape[0]} steps != NPZ {len(traj)}"
            )
            assert h5_actions.shape == npz_actions.shape
            max_diff = float(np.max(np.abs(h5_actions - npz_actions)))
            assert max_diff < 1e-5, f"demo_{ep_idx}: action mismatch max_diff={max_diff}"


@pytest.mark.integration
def test_hdf5_has_attachment_side_channel():
    _, hdf5_path = _dataset_paths()
    if not hdf5_path.is_file():
        pytest.skip("cable threading HDF5 not present")

    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        assert data.attrs.get("attachment_side_channel") in (True, np.bool_(True))
        assert str(data.attrs.get("grasp_mode", "")) == "attachment"
        side_keys_raw = data.attrs.get("side_channel_keys", "[]")
        if isinstance(side_keys_raw, bytes):
            side_keys_raw = side_keys_raw.decode("utf-8")
        side_keys = json.loads(side_keys_raw) if isinstance(side_keys_raw, str) else list(side_keys_raw)
        assert "attachment_enabled" in side_keys
        demo0 = data["demo_0"]
        assert "attachment_enabled" in demo0
        attach = np.asarray(demo0["attachment_enabled"], dtype=bool)
        actions = np.asarray(demo0["actions"])
        assert attach.shape[0] == actions.shape[0]


@pytest.mark.integration
def test_hdf5_replay_reproduces_recorded_success():
    npz_path, hdf5_path = _dataset_paths()
    if not npz_path.is_file() or not hdf5_path.is_file():
        pytest.skip("cable threading dataset artifacts not present")

    trajectories, episode_metadata, _ = load_npz_trajectories(npz_path)
    success_idx = None
    for idx, meta in enumerate(episode_metadata):
        summary = meta.get("summary", meta)
        if bool(summary.get("final_success")):
            success_idx = idx
            break
    if success_idx is None:
        # fallback: first demo in HDF5 attrs
        with h5py.File(hdf5_path, "r") as f:
            for idx in range(len(f["data"].keys()) - 0):
                demo = f"demo_{idx}"
                if demo not in f["data"]:
                    break
                meta_raw = f["data"][demo].attrs.get("benchmark_episode_metadata", "{}")
                if isinstance(meta_raw, bytes):
                    meta_raw = meta_raw.decode("utf-8")
                h5_meta = json.loads(meta_raw) if meta_raw else {}
                if bool(h5_meta.get("summary", {}).get("final_success")):
                    success_idx = idx
                    episode_metadata[idx] = h5_meta
                    break
    assert success_idx is not None, "no successful demo found"

    recorded = episode_metadata[success_idx]
    summary = recorded.get("summary", recorded)
    recorded_err = float(
        summary.get("endpoint_goal_error_final", summary.get("endpoint_goal_error", 1.0))
    )
    assert bool(summary.get("final_success")) is True

    final_info, _ = replay_hdf5_demo(
        hdf5_path,
        f"demo_{success_idx}",
        seed=int(recorded.get("seed", 0)),
    )
    assert bool(final_info.get("final_success")) is True
    replay_err = float(
        final_info.get("endpoint_goal_error_final", final_info.get("endpoint_goal_error", 1.0))
    )
    assert abs(replay_err - recorded_err) < 0.05, (
        f"endpoint_goal_error drift: recorded={recorded_err:.4f} replay={replay_err:.4f}"
    )

    with h5py.File(hdf5_path, "r") as f:
        meta_raw = f["data"][f"demo_{success_idx}"].attrs.get("benchmark_episode_metadata", "{}")
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        h5_meta = json.loads(meta_raw) if meta_raw else {}
    h5_summary = h5_meta.get("summary", h5_meta)
    assert bool(h5_summary.get("final_success")) is True
