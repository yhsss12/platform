from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.training_job_status import (
    compute_training_progress_fraction,
    enrich_and_persist_training_job_status,
    is_training_process_active,
    resolve_canonical_training_job_status,
)


def _write_running_job(train_job_dir: Path, *, status: str = "completed", epoch: int = 5, total: int = 200) -> None:
    train_job_dir.mkdir(parents=True, exist_ok=True)
    (train_job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "artifacts" / "metrics.jsonl").write_text(
        "\n".join(
            f'{{"epoch": {i}, "trainLoss": 0.2 + i * 0.01}}'
            for i in range(1, epoch + 1)
        ),
        encoding="utf-8",
    )
    log_lines = ["command: train_dp.py --init-checkpoint /external/model_final.pt\n"]
    log_lines += [f"Epoch {i} Loss: 0.23\n" for i in range(1, epoch + 1)]
    (train_job_dir / "logs" / "train.log").write_text("".join(log_lines), encoding="utf-8")
    (train_job_dir / "checkpoints" / "diffusion_policy" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "checkpoints" / "diffusion_policy" / "checkpoints" / "model_final.pt").write_bytes(b"x")
    (train_job_dir / "status.json").write_text(
        json.dumps(
            {
                "status": status,
                "epoch": 200 if status == "completed" else epoch,
                "totalEpochs": total,
                "progress": 1.0 if status == "completed" else epoch / total,
                "loss": 0.231848,
            }
        ),
        encoding="utf-8",
    )


def test_canonical_running_when_process_active_and_epoch_behind(tmp_path: Path):
    train_job_id = "train_20260625_proc_test_abcd"
    train_job_dir = tmp_path / train_job_id
    _write_running_job(train_job_dir, status="completed", epoch=5, total=200)

    with patch(
        "app.services.training_job_status.is_training_process_active",
        return_value=True,
    ):
        resolved = resolve_canonical_training_job_status(train_job_id, train_job_dir, json.loads(
            (train_job_dir / "status.json").read_text(encoding="utf-8")
        ))

    assert resolved["status"] == "running"
    assert resolved["epoch"] == 5
    assert resolved["totalEpochs"] == 200
    assert resolved["progress"] == compute_training_progress_fraction(5, 200, "running")
    assert resolved["progress"] < 0.03


def test_progress_not_one_when_epoch_five_of_two_hundred(tmp_path: Path):
    train_job_dir = tmp_path / "job"
    _write_running_job(train_job_dir, status="running", epoch=5, total=200)

    with patch("app.services.training_job_status.is_training_process_active", return_value=False):
        resolved = resolve_canonical_training_job_status("train_job", train_job_dir, json.loads(
            (train_job_dir / "status.json").read_text(encoding="utf-8")
        ))

    assert resolved["progress"] == compute_training_progress_fraction(5, 200, "running")
    assert resolved["progress"] == 0.025


def test_running_status_final_loss_null_in_metrics(tmp_path: Path):
    from app.services.training_metrics import normalized_training_metrics

    train_job_dir = tmp_path / "job"
    _write_running_job(train_job_dir, status="running", epoch=5, total=200)
    status = resolve_canonical_training_job_status(
        "train_job",
        train_job_dir,
        json.loads((train_job_dir / "status.json").read_text(encoding="utf-8")),
    )
    metrics = normalized_training_metrics(train_job_dir, status)
    assert metrics.get("finalLoss") is None
    assert metrics.get("bestLoss") is not None
    assert metrics.get("loss") is not None


def test_init_checkpoint_in_command_does_not_complete(tmp_path: Path):
    train_job_dir = tmp_path / "job"
    _write_running_job(train_job_dir, status="running", epoch=2, total=200)

    with patch("app.services.training_job_status.is_training_process_active", return_value=False):
        resolved = resolve_canonical_training_job_status(
            "train_job",
            train_job_dir,
            json.loads((train_job_dir / "status.json").read_text(encoding="utf-8")),
        )

    assert resolved["status"] == "running"


def test_persist_corrects_false_completed_on_disk(tmp_path: Path):
    train_job_id = "train_persist_fix_abcd"
    train_job_dir = tmp_path / train_job_id
    _write_running_job(train_job_dir, status="completed", epoch=5, total=200)

    with patch("app.services.training_job_status.is_training_process_active", return_value=True):
        enrich_and_persist_training_job_status(
            train_job_id,
            train_job_dir,
            json.loads((train_job_dir / "status.json").read_text(encoding="utf-8")),
        )

    on_disk = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "running"
    assert on_disk["epoch"] == 5
    assert on_disk["progress"] == 0.025
