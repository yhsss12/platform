from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from integrations.dual_arm_cable.export_il_dataset import export_job, inspect_job

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _write_synthetic_step(job_dir: Path, *, t: int = 4, success: bool = True) -> None:
    step_dir = job_dir / "results" / "steps" / "step_00"
    traj_dir = step_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)

    actions = np.random.randn(t, 14).astype(np.float32)
    cable_dim = 20
    obs = {
        "left_arm_joint_pos": np.random.randn(t, 7).astype(np.float32),
        "right_arm_joint_pos": np.random.randn(t, 7).astype(np.float32),
        "left_arm_joint_vel": np.random.randn(t, 7).astype(np.float32),
        "right_arm_joint_vel": np.random.randn(t, 7).astype(np.float32),
        "cable_state": np.random.randn(t, cable_dim).astype(np.float32),
        "overhead_rgb_frame_idx": np.full((t, 1), -1, dtype=np.int32),
    }
    next_obs = {key: arr.copy() for key, arr in obs.items()}
    np.save(traj_dir / "actions.npy", actions)
    np.save(traj_dir / "rewards.npy", np.array([0, 0, 0, 1], dtype=np.float32))
    np.save(traj_dir / "dones.npy", np.array([0, 0, 0, 1], dtype=np.uint8))
    np.save(traj_dir / "success.npy", np.array([0, 0, 0, 1], dtype=np.uint8))
    np.savez(traj_dir / "obs.npz", **obs)
    np.savez(traj_dir / "next_obs.npz", **next_obs)
    (traj_dir / "trajectory_manifest.json").write_text(
        json.dumps(
            {
                "actionSemantics": "recorded_joint_position_targets",
                "actionDim": 14,
                "controlFrequency": 500,
                "numTransitions": t,
            }
        ),
        encoding="utf-8",
    )
    (step_dir / "result.json").write_text(
        json.dumps({"task_success": success, "grasp_success": success}),
        encoding="utf-8",
    )
    (job_dir / "results" / "episode_result.json").write_text(
        json.dumps({"episode_success": success, "steps": []}),
        encoding="utf-8",
    )


def test_inspect_and_export_synthetic_trajectory(tmp_path: Path):
    job_dir = tmp_path / "dac_gen_test_synthetic"
    job_dir.mkdir()
    _write_synthetic_step(job_dir)

    report = inspect_job(job_dir)
    assert report["exportReady"] is True
    assert report["actionAvailable"] is True
    assert report["observationAvailable"] is True

    result = export_job(job_dir, job_id=job_dir.name)
    assert (job_dir / "datasets" / "dataset.hdf5").is_file()
    assert (job_dir / "datasets" / "dataset.manifest.json").is_file()
    manifest = result["manifest"]
    assert manifest["trainable"] is True
    assert manifest["actionDim"] == 14
    assert manifest["actionSemantics"] == "recorded_joint_position_targets"
    assert manifest.get("totalEpisodes") == 1
    assert manifest.get("successfulEpisodes") == 1
