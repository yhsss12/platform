from __future__ import annotations

import json
from pathlib import Path

from app.services.training_metrics import (
    collect_training_log_paths,
    normalized_training_metrics,
    resolve_training_metrics_log_path,
    sync_metrics_from_logs,
)


def test_resolve_training_metrics_log_path_prefers_more_epochs(tmp_path: Path):
    job_dir = tmp_path / "train_job"
    logs_dir = job_dir / "logs"
    nested_dir = job_dir / "checkpoints" / "robomimic_bc" / "run" / "logs"
    logs_dir.mkdir(parents=True)
    nested_dir.mkdir(parents=True)

    (logs_dir / "train.log").write_text(
        "\n".join(
            [
                "command: python train.py",
                "Train Epoch 1",
                '"Loss": 1.0',
                "Train Epoch 2",
                '"Loss": 0.8',
            ]
        ),
        encoding="utf-8",
    )
    nested_lines = [
        line
        for epoch in range(1, 11)
        for line in (f"Train Epoch {epoch}", f'"Loss": {1.0 - epoch * 0.1}')
    ]
    (nested_dir / "log.txt").write_text("\n".join(nested_lines), encoding="utf-8")

    resolved = resolve_training_metrics_log_path(job_dir)
    assert resolved.name == "log.txt"
    assert len(collect_training_log_paths(job_dir)) == 2


def test_normalized_training_metrics_includes_all_epochs(tmp_path: Path):
    job_dir = tmp_path / "train_job"
    logs_dir = job_dir / "logs"
    nested_dir = job_dir / "checkpoints" / "robomimic_bc" / "run" / "logs"
    logs_dir.mkdir(parents=True)
    nested_dir.mkdir(parents=True)

    (logs_dir / "train.log").write_text(
        "\n".join(
            [
                "noise " * 100,
                "Train Epoch 7",
                '"Loss": 0.3',
            ]
        ),
        encoding="utf-8",
    )
    nested_lines = [
        line
        for epoch in range(1, 11)
        for line in (f"Train Epoch {epoch}", f'"Loss": {1.0 - epoch * 0.05}')
    ]
    (nested_dir / "log.txt").write_text("\n".join(nested_lines), encoding="utf-8")

    status = {"status": "completed", "epoch": 10, "totalEpochs": 10, "loss": 0.55}
    normalized = normalized_training_metrics(job_dir, status)
    epochs = [int(row["epoch"]) for row in normalized["lossSeries"]]
    assert epochs == list(range(1, 11))
    assert normalized["epoch"] == 10
    assert normalized["loss"] is not None


def test_sync_metrics_from_logs_rewrites_jsonl(tmp_path: Path):
    job_dir = tmp_path / "train_job"
    nested_dir = job_dir / "checkpoints" / "robomimic_bc" / "run" / "logs"
    nested_dir.mkdir(parents=True)
    (nested_dir / "log.txt").write_text(
        "Train Epoch 1\n\"Loss\": 1.0\nTrain Epoch 2\n\"Loss\": 0.5\n",
        encoding="utf-8",
    )
    status = {"status": "completed", "epoch": 2, "totalEpochs": 2}

    series = sync_metrics_from_logs(job_dir, status)
    assert len(series) == 2

    metrics_path = job_dir / "artifacts" / "metrics.jsonl"
    assert metrics_path.is_file()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["epoch"] for row in rows] == [1, 2]


def test_act_log_series_train_and_valid_loss(tmp_path: Path):
    job_dir = tmp_path / "train_job"
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True)
    lines = []
    for epoch in range(1, 13):
        lines.append(f"2026-06-19 INFO Epoch {epoch} Loss: {1.0 - epoch * 0.01:.6f}")
        lines.append(f"2026-06-19 INFO Validation Epoch {epoch} Loss: {0.95 - epoch * 0.01:.6f}")
    (logs_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = {"status": "running", "epoch": 12, "totalEpochs": 100}

    normalized = normalized_training_metrics(job_dir, status)
    series = normalized["lossSeries"]
    assert len(series) == 12
    for epoch in range(1, 13):
        row = next(item for item in series if int(item["epoch"]) == epoch)
        assert row.get("trainLoss") is not None
        assert row.get("validLoss") is not None


def test_pollution_metrics_jsonl_does_not_erase_train_loss(tmp_path: Path):
    job_dir = tmp_path / "train_job"
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True)
    metrics_path = artifacts / "metrics.jsonl"
    rows = []
    for epoch in range(1, 4):
        rows.append(
            json.dumps(
                {
                    "epoch": epoch,
                    "totalEpochs": 100,
                    "trainLoss": 1.0 - epoch * 0.1,
                    "validLoss": 0.9 - epoch * 0.1,
                    "currentLoss": 1.0 - epoch * 0.1,
                    "progress": epoch / 100,
                }
            )
        )
        for _ in range(5):
            rows.append(json.dumps({"epoch": epoch, "loss": 0.9 - epoch * 0.1}))
    metrics_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    status = {"status": "running", "epoch": 3, "totalEpochs": 100}
    normalized = normalized_training_metrics(job_dir, status)
    series = normalized["lossSeries"]
    assert len(series) == 3
    assert all(row.get("trainLoss") is not None for row in series)
    assert all(row.get("validLoss") is not None for row in series)


def test_running_status_does_not_set_final_loss_in_summary(tmp_path: Path):
    from app.services.training_job_sync_service import _summarize_loss_series

    series = [
        {"epoch": 1, "trainLoss": 1.0, "validLoss": 0.9},
        {"epoch": 2, "trainLoss": 0.8, "validLoss": 0.7},
    ]
    best, final = _summarize_loss_series(series, job_status="running")
    assert best == 0.7
    assert final is None
    _, final_completed = _summarize_loss_series(series, job_status="completed")
    assert final_completed == 0.8
    _, final_failed = _summarize_loss_series(series, job_status="failed")
    assert final_failed is None


def test_dp_epoch_loss_is_train_only_not_valid(tmp_path: Path):
    job_dir = tmp_path / "train_job"
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True)
    lines = [f"2026-06-19 INFO Epoch {epoch} Loss: {1.0 - epoch * 0.01:.6f}" for epoch in range(1, 31)]
    (logs_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True)
    metrics_path = artifacts / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(json.dumps({"epoch": epoch, "loss": 1.0 - epoch * 0.01}) for epoch in range(1, 6))
        + "\n",
        encoding="utf-8",
    )
    status = {"status": "completed", "epoch": 30, "totalEpochs": 30}

    normalized = normalized_training_metrics(job_dir, status)
    series = normalized["lossSeries"]
    assert len(series) == 30
    assert all(row.get("trainLoss") is not None for row in series)
    assert all(row.get("validLoss") is None for row in series)
