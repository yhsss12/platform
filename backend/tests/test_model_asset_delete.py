from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.services import workspace_model_asset_service as svc
from app.services.checkpoint_registry import register_checkpoint_assets


def test_delete_model_asset_removes_registry_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260620_120000_abcd"
    train_job_dir = tmp_path / "jobs" / train_job_id
    checkpoints_dir = train_job_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    ckpt = checkpoints_dir / "model_final.pth"
    ckpt.write_bytes(b"fake")

    monkeypatch.setattr(svc, "TRAINING_JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(
        "app.services.training_service._resolve_safe_path",
        lambda raw: Path(raw).resolve(),
    )
    monkeypatch.setattr(
        "app.services.training_service.ALLOWED_PATH_ROOTS",
        (tmp_path.resolve(),),
    )

    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest={"datasetId": "ds_test", "taskType": "cable_threading"},
        train_config={"taskName": "测试任务", "epochs": 10},
        status={"status": "completed", "datasetName": "线缆穿杆数据_20260620_1200", "totalEpochs": 10},
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=True,
    )
    (train_job_dir / "status.json").write_text(
        '{"status":"completed","totalEpochs":10}',
        encoding="utf-8",
    )

    assets = svc.list_model_assets()
    assert len(assets) == 1
    asset_id = assets[0]["id"]

    result = svc.delete_model_asset(asset_id)
    assert result["deleted"] is True
    assert svc.get_model_asset_by_id(asset_id) is None
    assert not ckpt.is_file()


def test_get_model_asset_by_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260620_130000_abcd"
    train_job_dir = tmp_path / "jobs" / train_job_id
    checkpoints_dir = train_job_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    ckpt = checkpoints_dir / "model_final.pth"
    ckpt.write_bytes(b"fake")

    monkeypatch.setattr(svc, "TRAINING_JOBS_ROOT", tmp_path / "jobs")

    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest={"datasetId": "ds_test", "taskType": "cable_threading"},
        train_config={"taskName": "测试任务", "epochs": 10},
        status={"status": "completed", "datasetName": "线缆穿杆数据_20260620_1200", "totalEpochs": 10},
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=True,
    )
    (train_job_dir / "status.json").write_text(
        '{"status":"completed","totalEpochs":10}',
        encoding="utf-8",
    )

    assets = svc.list_model_assets()
    assert len(assets) == 1
    asset_id = assets[0]["id"]

    found = svc.get_model_asset_by_id(asset_id)
    assert found is not None
    assert found["id"] == asset_id
    assert found["checkpointPath"]

    assert svc.get_model_asset_by_id("model_missing") is None


def test_delete_model_asset_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "TRAINING_JOBS_ROOT", tmp_path / "jobs")
    with pytest.raises(HTTPException) as exc:
        svc.delete_model_asset("model_missing")
    assert exc.value.status_code == 404
