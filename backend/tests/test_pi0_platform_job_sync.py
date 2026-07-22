from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.training_job_status import is_training_process_active

def test_is_training_process_active_ignores_current_pgrep_match(tmp_path):
    marker = tmp_path / "train_marker_job"
    marker.mkdir()
    assert is_training_process_active("train_marker_job", train_job_dir=marker) is False


def test_pi0_completed_status_not_downgraded_by_pgrep(tmp_path):
    from app.services.training_job_status import enrich_and_persist_training_job_status
    from app.services.training_service import _write_json

    train_job_dir = tmp_path / "train_pi0_done"
    (train_job_dir / "config").mkdir(parents=True)
    (train_job_dir / "artifacts").mkdir(parents=True)
    (train_job_dir / "logs").mkdir(parents=True)
    (train_job_dir / "checkpoints/pi0/checkpoints").mkdir(parents=True)
    ckpt = train_job_dir / "checkpoints/pi0/checkpoints/model_final.pt"
    ckpt.write_text("{}", encoding="utf-8")
    (train_job_dir / "config/train_config.json").write_text(
        json.dumps({"trainingBackend": "pi0", "epochs": 1}),
        encoding="utf-8",
    )
    (train_job_dir / "artifacts/dataset_manifest.json").write_text("{}", encoding="utf-8")
    (train_job_dir / "logs/train.log").write_text(
        "pi0 LeRobot smoke completed\ntraining completed\n",
        encoding="utf-8",
    )
    status = {
        "trainJobId": "train_pi0_done",
        "status": "completed",
        "progress": 1.0,
        "epoch": 1,
        "totalEpochs": 1,
        "checkpointExists": True,
        "checkpointPath": str(ckpt),
        "modelType": "pi0",
        "datasetFormat": "lerobot",
        "finalLoss": 0.1,
    }
    _write_json(train_job_dir / "status.json", status)

    enriched = enrich_and_persist_training_job_status("train_pi0_done", train_job_dir, status)
    assert enriched.get("status") == "completed"
    assert float(enriched.get("progress") or 0) >= 1.0


def test_pi0_json_checkpoint_schema_extraction(tmp_path):
    from app.services.policy_schema_resolver import extract_pi0_schema_fields_from_checkpoint

    ckpt = tmp_path / "model_final.pt"
    ckpt.write_text(
        json.dumps(
            {
                "format": "pi0_lerobot_smoke_v1",
                "backend": "pi0",
                "state_dim": 9,
                "action_dim": 8,
                "robot": "Panda",
                "controller_type": "JOINT_POSITION",
                "action_mode": "joint_delta_derived",
                "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
                "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
            }
        ),
        encoding="utf-8",
    )
    fields = extract_pi0_schema_fields_from_checkpoint(ckpt)
    assert fields.get("modelType") == "pi0"
    assert fields.get("stateDim") == 9
    assert fields.get("actionDim") == 8
    assert fields.get("canEvaluate") is False
    assert fields.get("evalDisabledReason") == "pi0 eval adapter not ready"
