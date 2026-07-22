from __future__ import annotations

import json
from pathlib import Path

from app.services.training_job_status import (
    enrich_and_persist_training_job_status,
    enrich_training_job_status,
    infer_training_job_completed,
    log_indicates_training_completed,
)


def test_log_completion_ignores_init_checkpoint_in_command_line():
    log = (
        "command: train_dp.py --init-checkpoint /data/pretrain/model_final.pt\n"
        "2026-06-25 18:32:55 INFO Epoch 1 Loss: 0.23\n"
    )
    assert not log_indicates_training_completed(log)


def test_log_completion_matches_saved_checkpoint_marker():
    log = "2026-06-25 19:00:00 INFO saved checkpoint: /job/checkpoints/model_final.pt\n"
    assert log_indicates_training_completed(log)


def test_infer_not_completed_with_early_final_and_init_in_command(tmp_path: Path):
    train_job_dir = tmp_path / "job"
    train_job_dir.mkdir(parents=True)
    (train_job_dir / "logs").mkdir(parents=True)
    (train_job_dir / "checkpoints" / "diffusion_policy" / "checkpoints").mkdir(parents=True)
    final = train_job_dir / "checkpoints" / "diffusion_policy" / "checkpoints" / "model_final.pt"
    final.write_bytes(b"x")
    (train_job_dir / "logs" / "train.log").write_text(
        "command: --init-checkpoint /external/model_final.pt\nEpoch 1 Loss: 0.2\n",
        encoding="utf-8",
    )
    status = {"status": "running", "totalEpochs": 200, "epoch": 1}
    assert not infer_training_job_completed(status, train_job_dir=train_job_dir)


def test_enrich_downgrades_false_completed_when_epoch_behind(tmp_path: Path):
    train_job_dir = tmp_path / "job"
    (train_job_dir / "artifacts").mkdir(parents=True)
    (train_job_dir / "artifacts" / "metrics.jsonl").write_text(
        '{"epoch": 3, "trainLoss": 0.2}\n',
        encoding="utf-8",
    )
    status = {"status": "completed", "totalEpochs": 200, "epoch": 200, "progress": 1.0}
    enriched = enrich_training_job_status(train_job_dir, status)
    assert enriched["status"] == "running"
    assert enriched["epoch"] == 3
    assert enriched["progress"] < 1.0


def test_persist_downgrades_false_completed_status(tmp_path: Path):
    train_job_dir = tmp_path / "job"
    (train_job_dir / "artifacts").mkdir(parents=True)
    (train_job_dir / "artifacts" / "metrics.jsonl").write_text(
        '{"epoch": 3, "trainLoss": 0.2}\n',
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps({"status": "completed", "totalEpochs": 200, "epoch": 200, "progress": 1.0}),
        encoding="utf-8",
    )

    resolved = enrich_and_persist_training_job_status(
        "train_downgrade_test",
        train_job_dir,
        json.loads((train_job_dir / "status.json").read_text(encoding="utf-8")),
    )

    assert resolved.get("status") == "running"
    assert resolved.get("epoch") == 3
    on_disk = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
    assert on_disk.get("status") == "running"
    assert on_disk.get("epoch") == 3
