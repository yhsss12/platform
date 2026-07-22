from __future__ import annotations

from pathlib import Path

from app.services.checkpoint_registry import (
    CheckpointRecord,
    dedupe_registry_assets,
    list_displayable_registry_assets,
    normalize_discovered_checkpoints,
    normalize_registry_assets,
    register_checkpoint_assets,
)


def test_normalize_discovered_checkpoints_merges_final_epoch_when_complete(tmp_path: Path):
    final_path = tmp_path / "model_final.pth"
    epoch_path = tmp_path / "model_epoch_10.pth"
    mid_path = tmp_path / "model_epoch_5.pth"
    final_path.write_bytes(b"x")
    epoch_path.write_bytes(b"y")
    mid_path.write_bytes(b"z")

    records = [
        CheckpointRecord(path=final_path, kind="final"),
        CheckpointRecord(path=epoch_path, kind="epoch", epoch=10),
        CheckpointRecord(path=mid_path, kind="epoch", epoch=5),
    ]

    normalized = normalize_discovered_checkpoints(records, total_epochs=10, training_complete=True)
    kinds = sorted((item.kind, item.epoch) for item in normalized)
    assert kinds == [("epoch", 5), ("final", None)]
    assert len([item for item in normalized if item.kind == "final"]) == 1


def test_normalize_discovered_checkpoints_skips_final_while_training(tmp_path: Path):
    final_path = tmp_path / "model_final.pth"
    epoch_path = tmp_path / "model_epoch_10.pth"
    final_path.write_bytes(b"x")
    epoch_path.write_bytes(b"y")

    records = [
        CheckpointRecord(path=final_path, kind="final"),
        CheckpointRecord(path=epoch_path, kind="epoch", epoch=10),
    ]

    normalized = normalize_discovered_checkpoints(records, total_epochs=10, training_complete=False)
    assert all(item.kind != "final" for item in normalized)
    assert any(item.kind == "epoch" and item.epoch == 10 for item in normalized)


def test_dedupe_registry_assets_collapses_final_and_last_epoch_when_complete():
    assets = [
        {
            "modelAssetId": "model_a",
            "checkpointKind": "final",
            "checkpointPath": "/tmp/model_final.pth",
            "status": "ready",
        },
        {
            "modelAssetId": "model_b",
            "checkpointKind": "epoch",
            "checkpointEpoch": 10,
            "checkpointPath": "/tmp/model_epoch_10.pth",
            "status": "ready",
        },
        {
            "modelAssetId": "model_c",
            "checkpointKind": "epoch",
            "checkpointEpoch": 5,
            "checkpointPath": "/tmp/model_epoch_5.pth",
            "status": "ready",
        },
    ]

    deduped = dedupe_registry_assets(assets, total_epochs=10, training_complete=True)
    assert len(deduped) == 2
    kinds = sorted(str(item.get("checkpointKind")) for item in deduped)
    assert kinds == ["epoch", "final"]


def test_normalize_registry_assets_keeps_single_best(tmp_path: Path):
    best_old = tmp_path / "best_old.pth"
    best_new = tmp_path / "best_new.pth"
    best_old.write_bytes(b"1")
    best_new.write_bytes(b"2")

    assets = [
        {
            "modelAssetId": "model_job_best_loss",
            "checkpointKind": "best",
            "checkpointMetricName": "Loss",
            "checkpointMetricValue": 1.2,
            "checkpointEpoch": 2,
            "checkpointPath": str(best_old),
            "status": "ready",
        },
        {
            "modelAssetId": "model_job_old_path",
            "checkpointKind": "best",
            "checkpointMetricName": "Loss",
            "checkpointMetricValue": 0.8,
            "checkpointEpoch": 4,
            "checkpointPath": str(best_new),
            "status": "ready",
        },
    ]

    normalized = normalize_registry_assets(assets, status={"status": "running"}, total_epochs=100)
    active_best = [
        item
        for item in normalized
        if item.get("checkpointKind") == "best" and str(item.get("status")).lower() != "superseded"
    ]
    assert len(active_best) == 1
    assert active_best[0]["checkpointMetricValue"] == 0.8


def test_register_checkpoint_assets_skips_final_during_training(tmp_path: Path):
    job_dir = tmp_path / "train_job"
    ckpt_dir = job_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (job_dir / "config").mkdir(parents=True)
    (job_dir / "artifacts").mkdir(parents=True)

    final_path = ckpt_dir / "model_final.pth"
    best_path = ckpt_dir / "model_epoch_3_best_validation_0.5.pth"
    epoch_path = ckpt_dir / "model_epoch_2.pth"
    final_path.write_bytes(b"x")
    best_path.write_bytes(b"y")
    epoch_path.write_bytes(b"z")

    status = {"status": "running", "totalEpochs": 100, "epoch": 18}
    train_config = {"epochs": 100, "taskName": "demo"}
    manifest = {"datasetId": "ds1"}

    displayable = register_checkpoint_assets(
        train_job_dir=job_dir,
        train_job_id="train_demo",
        manifest=manifest,
        train_config=train_config,
        status=status,
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=False,
    )

    kinds = {item.get("checkpointKind") for item in displayable}
    assert "final" not in kinds
    assert "best" in kinds


def test_register_checkpoint_assets_registers_final_on_complete(tmp_path: Path):
    job_dir = tmp_path / "train_job"
    ckpt_dir = job_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (job_dir / "config").mkdir(parents=True)
    (job_dir / "artifacts").mkdir(parents=True)

    final_path = ckpt_dir / "model_final.pth"
    final_path.write_bytes(b"x")

    status = {"status": "completed", "totalEpochs": 10, "epoch": 10}
    train_config = {"epochs": 10, "taskName": "demo"}
    manifest = {"datasetId": "ds1"}

    displayable = register_checkpoint_assets(
        train_job_dir=job_dir,
        train_job_id="train_demo",
        manifest=manifest,
        train_config=train_config,
        status=status,
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=True,
    )

    assert any(item.get("checkpointKind") == "final" for item in displayable)


def test_list_displayable_registry_assets_hides_superseded_and_pending(tmp_path: Path):
    ready_path = tmp_path / "a.pth"
    ready_path.write_bytes(b"1")
    assets = [
        {
            "modelAssetId": "a",
            "checkpointKind": "best",
            "status": "ready",
            "checkpointPath": str(ready_path),
        },
        {
            "modelAssetId": "b",
            "checkpointKind": "best",
            "status": "superseded",
            "checkpointPath": str(tmp_path / "b.pth"),
        },
        {
            "modelAssetId": "c",
            "checkpointKind": "final",
            "status": "pending",
            "checkpointPath": str(tmp_path / "c.pth"),
        },
    ]
    visible = list_displayable_registry_assets(
        assets,
        status={"status": "running"},
        total_epochs=100,
    )
    assert len(visible) == 1
    assert visible[0]["modelAssetId"] == "a"
