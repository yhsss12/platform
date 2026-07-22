from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.services import cable_threading_service as svc


def _fake_trajectory(steps: int = 3) -> list[dict]:
    traj = []
    for idx in range(steps):
        joint = np.linspace(0.0, 0.1, 7, dtype=np.float32) + idx * 0.01
        gripper = np.array([0.02, 0.02], dtype=np.float32)
        raw_obs = {
            "robot0_joint_pos": joint,
            "robot0_gripper_qpos": gripper,
            "agentview_image": np.zeros((64, 64, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((64, 64, 3), dtype=np.uint8),
        }
        traj.append(
            {
                "action": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
                "raw_obs": raw_obs,
                "reward": 0.0,
                "done": idx == steps - 1,
            }
        )
    return traj


def test_save_cable_threading_lerobot_dataset_writes_sidecars(tmp_path: Path):
    from robosuite.utils.dlo.lerobot_platform_export import save_cable_threading_lerobot_dataset

    out_dir = tmp_path / "lerobot_dataset"
    info = save_cable_threading_lerobot_dataset(
        out_dir,
        [_fake_trajectory()],
        robot="Panda",
        task_instruction="thread the cable through the pole",
        fps=20,
    )
    assert (out_dir / "metadata.json").is_file()
    assert (out_dir / "stats.json").is_file()
    assert (out_dir / "generation_report.json").is_file()
    assert (out_dir / "meta" / "info.json").is_file()
    assert (out_dir / "data" / "chunk-000" / "file-000.parquet").is_file()
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "lerobot"
    assert metadata["state_dim"] == 9
    assert metadata["action_dim"] == 8
    assert metadata["pi0Ready"] is True
    assert info["pi0Ready"] is True


def test_save_cable_threading_lerobot_dataset_marks_pi0_not_ready_for_osc(tmp_path: Path):
    from robosuite.utils.dlo.lerobot_platform_export import save_cable_threading_lerobot_dataset

    traj = []
    for idx in range(2):
        traj.append(
            {
                "action": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
                "raw_obs": {
                    "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                    "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
                    "robot0_gripper_qpos": np.array([0.02], dtype=np.float32),
                },
                "reward": 0.0,
                "done": idx == 1,
            }
        )
    out_dir = tmp_path / "lerobot_osc"
    info = save_cable_threading_lerobot_dataset(
        out_dir,
        [traj],
        robot="Panda",
        task_instruction="thread the cable through the pole",
        fps=20,
        source_controller_type="OSC_POSE",
    )
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pi0Ready"] is False
    assert metadata["action_dim"] == 7
    assert "OSC_POSE" in metadata["pi0ReadyReason"] or metadata["controller_type"] == "OSC_POSE"
    assert info["pi0Ready"] is False


def test_maybe_persist_manifest_records_lerobot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    job_id = "ct_gen_20260626_120000_ab12"
    job_root = tmp_path / "jobs" / job_id
    datasets = job_root / "datasets"
    lerobot_dir = datasets / "lerobot_dataset"
    lerobot_dir.mkdir(parents=True)
    (lerobot_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format": "lerobot",
                "task_instruction": "thread the cable through the pole",
                "robot": "Panda",
                "state_dim": 9,
                "action_dim": 8,
                "pi0Ready": True,
                "pi0ReadyReason": "",
            }
        ),
        encoding="utf-8",
    )
    (datasets / "dataset.manifest.json").write_text(
        json.dumps({"num_successful": 1, "num_failed": 0, "primaryFormat": "lerobot"}),
        encoding="utf-8",
    )

    manifest = svc._maybe_persist_cable_dataset_manifest(job_root, job_id)
    assert manifest.get("primaryFormat") == "lerobot"
    assert manifest.get("availableFormats") == ["lerobot"]
    assert manifest.get("lerobot", {}).get("status") == "ready"


def test_run_py_expert_accepts_lerobot_out_flag():
    run_py = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP" / "run.py"
    text = run_py.read_text(encoding="utf-8")
    assert "--lerobot-out" in text
    assert "save_cable_threading_lerobot_dataset" in text
