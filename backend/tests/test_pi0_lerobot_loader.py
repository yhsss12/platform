from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.pi0_lerobot_loader import (
    inspect_lerobot_dataset,
    is_platform_lerobot_v3_dataset,
    iter_lerobot_training_batches,
    validate_lerobot_for_pi0,
)
from app.services.pi0_lerobot_smoke_runner import assess_pi0_lerobot_training_capability
from app.services.pi0_training_runner import manifest_has_lerobot_dataset, prepare_pi0_job_artifacts
from app.services.policy_schema_resolver import PI0_JOINT_SPACE_ENABLED

SMOKE_DATASET = (
    Path(__file__).resolve().parents[2]
    / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/lerobot_dataset"
)


@pytest.fixture()
def lerobot_fixture_dir() -> Path:
    if not SMOKE_DATASET.is_dir():
        pytest.skip("smoke LeRobot dataset not present")
    return SMOKE_DATASET


def test_is_platform_lerobot_v3_dataset(lerobot_fixture_dir: Path):
    assert is_platform_lerobot_v3_dataset(lerobot_fixture_dir) is True


def test_validate_smoke_dataset_pi0_ready(lerobot_fixture_dir: Path):
    ok, reason = validate_lerobot_for_pi0(lerobot_fixture_dir)
    assert ok is True, reason
    spec = inspect_lerobot_dataset(lerobot_fixture_dir)
    assert spec.state_dim == 9
    assert spec.action_dim == 8
    assert spec.task_instruction == "thread the cable through the pole"


def test_iter_batches_state_action_dims(lerobot_fixture_dir: Path):
    batch = next(iter_lerobot_training_batches(lerobot_fixture_dir, batch_size=2, max_batches=1))
    assert batch["observation.state"].shape[-1] == 9
    assert batch["action"].shape[-1] == 8
    assert batch["task_instruction"]


def test_manifest_has_lerobot_v3_parquet(lerobot_fixture_dir: Path):
    manifest = {
        "sourceJobId": "ct_gen_20260630_120927_1153",
        "primaryFormat": "lerobot",
        "lerobot": {
            "status": "ready",
            "path": str(lerobot_fixture_dir),
        },
        "artifacts": {"lerobotPath": str(lerobot_fixture_dir)},
    }
    assert manifest_has_lerobot_dataset(manifest) is True


def test_prepare_pi0_skips_hdf5_converter(tmp_path: Path, lerobot_fixture_dir: Path, monkeypatch: pytest.MonkeyPatch):
    called = {"convert": False}

    def _fail_convert(**kwargs):
        called["convert"] = True
        raise AssertionError("HDF5 converter should not run for native LeRobot")

    monkeypatch.setattr("app.services.pi0_hdf5_converter.convert_hdf5_to_lerobot_dataset", _fail_convert)

    manifest = {
        "sourceJobId": "ct_gen_20260630_120927_1153",
        "primaryFormat": "lerobot",
        "availableFormats": ["lerobot"],
        "lerobot": {"status": "ready", "path": str(lerobot_fixture_dir)},
        "artifacts": {"lerobotPath": str(lerobot_fixture_dir)},
        "taskDescription": "thread the cable through the pole",
    }
    train_job_dir = tmp_path / "train_job"
    train_job_dir.mkdir(parents=True)
    train_config = {
        "epochs": 1,
        "batchSize": 2,
        "pi0Config": {
            "camera_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
        },
        "adaptationSnapshot": {
            "modelAdaptation": {
                "inputConfig": {
                    "camera_keys": ["agentview_image", "robot0_eye_in_hand_image"],
                    "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
                },
                "outputConfig": {"action_dim": 8, "action_horizon": 8},
            }
        },
    }
    config_path = prepare_pi0_job_artifacts(
        train_job_dir=train_job_dir,
        manifest=manifest,
        train_config=train_config,
        hdf5_path=None,
    )
    assert config_path.is_file()
    assert called["convert"] is False
    saved_manifest = json.loads((train_job_dir / "artifacts" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest.get("dataFormat") == "lerobot"


def test_capability_after_smoke(lerobot_fixture_dir: Path):
    cap = assess_pi0_lerobot_training_capability(dataset_path=lerobot_fixture_dir, smoke_success=True)
    assert cap["data_format_ready"] is True
    assert cap["lerobot_loader_ready"] is True
    assert cap["training_smoke_ready"] is True
    assert cap["pi0_joint_space_enabled"] is False
    assert PI0_JOINT_SPACE_ENABLED is False


def test_missing_prompt_fails(tmp_path: Path):
    root = tmp_path / "lerobot_dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    metadata = {
        "format": "lerobot",
        "task_instruction": "",
        "robot": "Panda",
        "controller_type": "JOINT_POSITION",
        "state_dim": 9,
        "action_dim": 8,
        "action_mode": "joint_delta_derived",
        "action_representation": "normalized_joint_delta",
        "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
        "pi0Ready": True,
        "pi0ReadyReason": "",
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_frames": 2,
                "total_episodes": 1,
                "features": {
                    "observation.state": {"shape": [9]},
                    "action": {"shape": [8]},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "stats.json").write_text("{}", encoding="utf-8")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "observation.state": [[0.0] * 9, [1.0] * 9],
                "action": [[0.0] * 8, [1.0] * 8],
                "frame_index": [0, 1],
            }
        )
        pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    except ImportError:
        pytest.skip("pyarrow required")

    for key in ("observation.images.agentview", "observation.images.eye_in_hand"):
        vdir = root / "videos" / key / "chunk-000"
        vdir.mkdir(parents=True)
        (vdir / "file-000.mp4").write_bytes(b"\x00")

    ok, reason = validate_lerobot_for_pi0(root)
    assert ok is False
    assert "task_instruction" in reason
