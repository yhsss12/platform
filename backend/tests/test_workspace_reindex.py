"""Tests for workspace reindex / dataset backfill."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.hdf5_platform_metadata import build_dataset_row_from_hdf5
from app.services.runtime_job_lifecycle import mark_job_deleted
from app.services.workspace_reindex_service import (
    _looks_like_eval_job,
    _runtime_job_deleted,
    normalize_reindex_job_type,
)


def test_normalize_reindex_job_type_maps_data_generation():
    assert normalize_reindex_job_type("data_generation") == "generate"
    assert normalize_reindex_job_type("all") is None


def test_runtime_job_deleted(tmp_path: Path):
    job_root = tmp_path / "train_deleted"
    job_root.mkdir()
    mark_job_deleted(job_root / "status.json")
    assert _runtime_job_deleted(job_root) is True


def test_looks_like_eval_job(tmp_path: Path):
    job_root = tmp_path / "ct_eval_demo"
    job_root.mkdir()
    (job_root / "results").mkdir()
    (job_root / "results" / "aggregate_result.json").write_text("{}", encoding="utf-8")
    assert _looks_like_eval_job(job_root) is True


def test_build_dataset_row_joint_from_hdf5(tmp_path: Path):
    hdf5_path = tmp_path / "dataset.hdf5"
    try:
        import h5py
        import numpy as np
    except ImportError:
        pytest.skip("h5py unavailable")

    with h5py.File(hdf5_path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["joint_action_available"] = True
        data.attrs["trained_action_mode"] = "joint_delta"
        data.attrs["eval_executor"] = "joint_position"
        data.attrs["action_schema"] = json.dumps(
            {"id": "cable_threading_joint_delta_v1", "actionMode": "joint_delta"}
        )
        demo = data.create_group("demo_0")
        demo.create_dataset("joint_actions", data=np.zeros((2, 7), dtype=np.float32))
        demo.create_dataset("gripper_actions", data=np.zeros((2, 1), dtype=np.float32))

    row = build_dataset_row_from_hdf5(hdf5_path, manifest={"sourceJobId": "ct_gen_test"})
    assert row["trainedActionMode"] == "joint_delta"
    assert row["evalExecutor"] == "joint_position"
    assert row["jointActionAvailable"] is True


def test_reindex_dry_run_counts(monkeypatch):
    from app.services import workspace_reindex_service as reindex_svc

    monkeypatch.setattr(
        reindex_svc,
        "_iter_train_job_ids_for_sync",
        lambda: ["train_20260623_hdf5fix"],
    )
    monkeypatch.setattr(
        reindex_svc,
        "_iter_eval_job_ids_for_sync",
        lambda: ["ct_eval_20260624_dp_smoke"],
    )
    monkeypatch.setattr(
        "app.services.workspace_job_service._iter_runtime_job_dirs",
        lambda **kwargs: [("ct_gen_x", "generate", "cable_threading", "run.py", Path("/tmp/x"))],
    )
    monkeypatch.setattr(
        "app.services.workspace_dataset_backfill_service.discover_dataset_hdf5_paths",
        lambda: [Path("/tmp/dataset.hdf5")],
    )

    result = reindex_svc.reindex_workspace_all(dry_run=True, job_type="all")
    assert result["scanned"] == 1
    assert result["syncedTrainingJobs"] == 1
    assert result["syncedEvalJobs"] == 1
    assert result["scannedDatasets"] == 1


def test_reindex_skips_deleted_train_job(tmp_path: Path, monkeypatch):
    from app.services import workspace_reindex_service as reindex_svc

    train_dir = tmp_path / "train_20260623_smoke200"
    train_dir.mkdir()
    (train_dir / "config").mkdir()
    (train_dir / "config" / "train_config.json").write_text("{}", encoding="utf-8")
    mark_job_deleted(train_dir / "status.json")

    monkeypatch.setattr(reindex_svc, "TRAINING_JOBS_ROOT", tmp_path)
    monkeypatch.setattr(
        reindex_svc,
        "_reindex_workspace_skip_deleted",
        lambda **kwargs: {"scanned": 0, "insertedJobs": 0, "updatedJobs": 0, "insertedArtifacts": 0, "skipped": 0, "skippedDeleted": 0, "errors": []},
    )
    monkeypatch.setattr(
        reindex_svc,
        "backfill_hdf5_dataset_records",
        lambda **kwargs: {"scannedDatasets": 0},
    )

    synced: list[str] = []

    def _fake_sync(job_id: str, **kwargs):
        synced.append(job_id)

    monkeypatch.setattr(reindex_svc, "sync_training_job_from_runtime", _fake_sync)
    result = reindex_svc.reindex_workspace_all(job_type="training", restore_deleted=False)
    assert "train_20260623_smoke200" not in synced
    assert result["skippedDeleted"] >= 1
