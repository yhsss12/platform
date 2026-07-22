from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import workspace_model_asset_service as svc
from app.services.training_job_status import (
    enrich_and_persist_training_job_status,
    enrich_training_job_status,
    infer_training_job_completed,
    normalize_api_training_status,
)


def test_normalize_api_training_status_maps_pending_to_queued():
    assert normalize_api_training_status("pending") == "queued"
    assert normalize_api_training_status("running") == "running"
    assert normalize_api_training_status("completed") == "completed"
    assert normalize_api_training_status("canceled") == "failed"
    assert normalize_api_training_status("deleted") == "backend_unavailable"


def _write_running_job_at_epoch_100(train_job_dir: Path) -> None:
    train_job_dir.mkdir(parents=True, exist_ok=True)
    (train_job_dir / "config").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"epochs": 100, "saveFinal": True, "taskName": "demo"}),
        encoding="utf-8",
    )
    (train_job_dir / "artifacts" / "dataset_manifest.json").write_text(
        json.dumps({"datasetId": "ds1"}),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps({"status": "running", "totalEpochs": 100, "epoch": 18, "datasetName": "ds"}),
        encoding="utf-8",
    )


def test_infer_completed_from_metrics_epoch(tmp_path: Path):
    train_job_dir = tmp_path / "job"
    _write_running_job_at_epoch_100(train_job_dir)
    log_path = train_job_dir / "logs" / "train.log"
    lines = [f"Train Epoch {i}\nEpoch {i} Loss: 0.1\n" for i in range(1, 101)]
    log_path.write_text("".join(lines), encoding="utf-8")

    status = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
    with patch("app.services.training_job_status.is_training_process_active", return_value=False):
        enriched = enrich_training_job_status(train_job_dir, status)
        assert infer_training_job_completed(enriched, train_job_dir=train_job_dir)

    assert enriched["status"] == "completed"
    assert enriched["epoch"] == 100
    assert enriched["progress"] == 1.0


def test_persist_completion_registers_final_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260620_140000_abcd"
    train_job_dir = tmp_path / "jobs" / train_job_id
    monkeypatch.setattr(svc, "TRAINING_JOBS_ROOT", tmp_path / "jobs")

    _write_running_job_at_epoch_100(train_job_dir)
    (train_job_dir / "checkpoints" / "model_final.pth").write_bytes(b"final")
    lines = [f"Train Epoch {i}\nEpoch {i} Loss: 0.1\n" for i in range(1, 101)]
    (train_job_dir / "logs" / "train.log").write_text("".join(lines), encoding="utf-8")

    status = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
    resolved = enrich_and_persist_training_job_status(train_job_id, train_job_dir, status)

    assert resolved.get("status") == "completed"
    detail = list(svc.list_training_job_model_assets_detail(train_job_id).get("modelAssets") or [])
    final_row = next(item for item in detail if item.get("checkpointKind") == "final")
    assert final_row.get("isPlaceholder") is False
    assert final_row.get("canEvaluate") is True
    assert final_row.get("displayStatus") == "ready"


def test_failed_status_not_promoted_to_completed(tmp_path: Path):
    train_job_dir = tmp_path / "job"
    _write_running_job_at_epoch_100(train_job_dir)
    (train_job_dir / "status.json").write_text(
        json.dumps({"status": "failed", "totalEpochs": 100, "epoch": 100}),
        encoding="utf-8",
    )

    status = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
    enriched = enrich_training_job_status(train_job_dir, status)

    assert enriched["status"] == "failed"
    assert not infer_training_job_completed(enriched, train_job_dir=train_job_dir)
