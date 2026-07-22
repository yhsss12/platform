from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services import dual_arm_cable_dataset_service as dac_dataset_svc
from app.services import workspace_dataset_service as ws_dataset_svc
from integrations.dual_arm_cable.export_il_dataset import export_job


def _write_synthetic_step_at(job_dir: Path, step_name: str, *, t: int = 4) -> None:
    step_dir = job_dir / "results" / "steps" / step_name
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
    np.save(traj_dir / "rewards.npy", np.zeros((t,), dtype=np.float32))
    np.save(traj_dir / "dones.npy", np.zeros((t,), dtype=np.uint8))
    np.save(traj_dir / "success.npy", np.zeros((t,), dtype=np.uint8))
    np.savez(traj_dir / "obs.npz", **obs)
    np.savez(traj_dir / "next_obs.npz", **next_obs)
    (traj_dir / "trajectory_manifest.json").write_text(
        json.dumps({"actionSemantics": "recorded_joint_position_targets", "actionDim": 14, "controlFrequency": 20}),
        encoding="utf-8",
    )
    (step_dir / "result.json").write_text(
        json.dumps({"task_success": True, "grasp_success": True}),
        encoding="utf-8",
    )


def _write_multi_step_job(job_dir: Path, *, steps: int = 3) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(steps):
        _write_synthetic_step_at(job_dir, f"step_{idx:02d}")
    (job_dir / "results" / "episode_result.json").write_text(
        json.dumps(
            {
                "max_cables": steps,
                "num_steps_attempted": steps,
                "num_cables_succeeded": steps,
                "episode_success": True,
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(
        json.dumps({"status": "completed", "maxCables": steps}),
        encoding="utf-8",
    )


def test_dual_arm_hdf5_registry_lists_trainable_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id = "dac_gen_20260617_160000_ab12"
    job_dir = tmp_path / job_id
    _write_multi_step_job(job_dir, steps=3)

    monkeypatch.setattr(dac_dataset_svc, "DUAL_ARM_ROOT", tmp_path)
    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path)

    export_job(job_dir, job_id=job_id)
    auto = dac_dataset_svc.auto_build_il_dataset_after_generate(job_id)
    assert auto["status"] in {"built", "already_built"}

    rows = ws_dataset_svc.list_datasets()
    row = next((r for r in rows if r.get("sourceJobId") == job_id), None)
    assert row is not None
    assert row.get("format") == "hdf5"
    assert row.get("datasetFormat") == "hdf5"
    assert row.get("trainable") is True
    assert row.get("trainingBackends") == ["torch_bc"]
    assert row.get("datasetFile")
    assert row.get("successfulEpisodes") == 3
    assert row.get("totalEpisodes") == 3
    assert (job_dir / "generation_manifest.json").is_file()


def test_manifest_only_dual_arm_shows_failure_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id = "dac_gen_20260617_160100_cd34"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "results").mkdir()
    (job_dir / "results" / "episode_manifest.json").write_text(
        json.dumps({"max_cables": 1, "episode_success": True}),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path)

    rows = ws_dataset_svc.list_datasets()
    row = next((r for r in rows if r.get("sourceJobId") == job_id), None)
    assert row is not None
    assert row.get("format") == "manifest"
    assert row.get("trainable") is False
    assert row.get("ilExportFailureReason")
