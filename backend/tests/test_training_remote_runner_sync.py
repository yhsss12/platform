from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.training_job_status import enrich_and_persist_training_job_status, enrich_training_job_status
from app.services.training_metrics import sanitize_training_log_for_display, sync_metrics_from_logs
from app.services.training_service import read_training_job_log, _reconcile_stale_running_training_job


def _write_remote_ssh_job(
    train_job_dir: Path,
    *,
    status: str = "running",
    epoch: int = 1,
    total_epochs: int = 1,
) -> None:
    train_job_dir.mkdir(parents=True, exist_ok=True)
    (train_job_dir / "config").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "checkpoints" / "diffusion_policy" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"epochs": total_epochs, "trainingBackend": "diffusion_policy", "saveFinal": True}),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps(
            {
                "trainJobId": train_job_dir.name,
                "status": status,
                "executionMode": "remote_ssh",
                "trainingNodeId": "l20-172-18-0-73",
                "epoch": epoch,
                "totalEpochs": total_epochs,
                "progress": 0.99,
            }
        ),
        encoding="utf-8",
    )
    (train_job_dir / "logs" / "train.log").write_text(
        "remote command: nohup python train_dp.py\n"
        "2026-06-26 12:57:52,273 INFO Epoch 1 Loss: 0.999762\n"
        "2026-06-26 12:57:57,623 INFO saved checkpoint: model_final.pt\n",
        encoding="utf-8",
    )
    ckpt = train_job_dir / "checkpoints" / "diffusion_policy" / "checkpoints" / "model_final.pt"
    ckpt.write_bytes(b"checkpoint")


def test_sanitize_training_log_strips_runner_noise():
    raw = "\n".join(
        [
            "remote command: nohup python train_dp.py",
            "  12%|█▏| 5.00M/44.7M [00:00<00:02, 19.4MB/s]",
            "2026-06-26 12:57:52,273 INFO Epoch 1 Loss: 0.999762",
        ]
    )
    cleaned = sanitize_training_log_for_display(raw)
    assert "remote command" not in cleaned
    assert "12%|" not in cleaned
    assert "Epoch 1 Loss" in cleaned


def test_remote_ssh_completed_not_downgraded_on_sync(tmp_path: Path):
    train_job_dir = tmp_path / "train_20260626_125605_11be"
    _write_remote_ssh_job(train_job_dir, status="completed", epoch=1, total_epochs=1)
    status_data = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))

    with patch("app.services.training_job_status.is_training_process_active", return_value=False):
        enriched = enrich_and_persist_training_job_status(train_job_dir.name, train_job_dir, status_data)

    assert enriched.get("status") == "completed"
    assert float(enriched.get("progress") or 0) == 1.0


def test_remote_ssh_running_with_checkpoint_infers_completed(tmp_path: Path):
    train_job_dir = tmp_path / "train_remote_done"
    _write_remote_ssh_job(train_job_dir, status="running", epoch=1, total_epochs=1)
    status_data = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))

    with patch("app.services.training_job_status.is_training_process_active", return_value=False):
        enriched = enrich_training_job_status(train_job_dir, status_data)

    assert enriched.get("status") == "completed"
    assert float(enriched.get("progress") or 0) == 1.0


def test_reconcile_stale_remote_ssh_without_logs_demotes_to_starting(tmp_path: Path):
    train_job_dir = tmp_path / "train_remote_stuck"
    train_job_dir.mkdir(parents=True)
    (train_job_dir / "config").mkdir(parents=True)
    (train_job_dir / "logs").mkdir(parents=True)
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"epochs": 100, "trainingBackend": "diffusion_policy"}),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps(
            {
                "trainJobId": train_job_dir.name,
                "status": "running",
                "executionMode": "remote_ssh",
                "trainingNodeId": "l20-172-18-0-73",
                "epoch": 0,
                "totalEpochs": 100,
                "progress": 0,
                "message": "远程训练进行中",
            }
        ),
        encoding="utf-8",
    )
    (train_job_dir / "logs" / "train.log").write_text("", encoding="utf-8")
    status_data = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))

    with patch("app.services.training_service._RUNNING_PROCS", {}):
        with patch(
            "app.services.training_service._reconcile_remote_training_on_read",
            side_effect=lambda _job_id, _dir, data: data,
        ):
            with patch("app.services.training_job_status.is_training_process_active", return_value=False):
                result = _reconcile_stale_running_training_job(train_job_dir.name, train_job_dir, status_data)

    assert result.get("status") == "starting"
    assert float(result.get("progress") or 0) == 0.0


def test_canonical_remote_running_without_activity_is_starting(tmp_path: Path):
    train_job_dir = tmp_path / "train_remote_starting"
    train_job_dir.mkdir(parents=True)
    (train_job_dir / "config").mkdir(parents=True)
    (train_job_dir / "logs").mkdir(parents=True)
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"epochs": 50, "executionMode": "remote_ssh", "trainingBackend": "diffusion_policy"}),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps(
            {
                "trainJobId": train_job_dir.name,
                "status": "running",
                "executionMode": "remote_ssh",
                "epoch": 0,
                "totalEpochs": 50,
            }
        ),
        encoding="utf-8",
    )
    status_data = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))

    with patch("app.services.training_job_status.is_training_process_active", return_value=False):
        enriched = enrich_training_job_status(train_job_dir, status_data)

    assert enriched.get("status") == "starting"
    assert int(enriched.get("epoch") or 0) == 0


def test_reconcile_stale_remote_ssh_does_not_mark_failed(tmp_path: Path):
    train_job_dir = tmp_path / "train_remote_done"
    _write_remote_ssh_job(train_job_dir, status="running", epoch=1, total_epochs=1)
    status_data = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))

    with patch("app.services.training_service._RUNNING_PROCS", {}):
        with patch("app.services.training_job_status.is_training_process_active", return_value=False):
            result = _reconcile_stale_running_training_job(train_job_dir.name, train_job_dir, status_data)

    assert result.get("status") == "completed"


def test_sync_metrics_from_logs_builds_loss_series(tmp_path: Path):
    train_job_dir = tmp_path / "train_metrics"
    _write_remote_ssh_job(train_job_dir, status="running", epoch=1, total_epochs=3)
    status_data = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
    status_data["totalEpochs"] = 3
    (train_job_dir / "logs" / "train.log").write_text(
        "Epoch 1 Loss: 0.9\nEpoch 2 Loss: 0.8\n",
        encoding="utf-8",
    )

    series = sync_metrics_from_logs(train_job_dir, status_data)
    assert len(series) >= 2
    assert int(series[0]["epoch"]) == 1


def test_read_training_job_log_prefers_sanitized_train_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_dir = tmp_path / "train_20260626_125605_11be"
    _write_remote_ssh_job(train_job_dir)

    monkeypatch.setattr(
        "app.services.workspace_runtime_paths.resolve_training_job_root",
        lambda _job_id: train_job_dir,
    )
    log = read_training_job_log(train_job_dir.name)
    assert "Epoch 1 Loss" in log
    assert "remote command" not in log
