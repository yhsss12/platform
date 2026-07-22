"""model_asset_validation 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.model_asset_validation import (
    check_checkpoint_file,
    enrich_model_asset,
    filter_evaluable_model_assets,
    is_model_asset_compatible_with_evaluation,
    validate_model_asset,
)


def test_check_checkpoint_file_missing() -> None:
    exists, size = check_checkpoint_file("/tmp/nonexistent_model_asset_test.pt")
    assert exists is False
    assert size == 0


def test_enrich_model_asset_marks_missing() -> None:
    enriched = enrich_model_asset(
        {
            "id": "model__test_missing",
            "checkpointPath": "/tmp/nonexistent_model_asset_test.pt",
            "status": "ready",
            "sourceTrainingJobId": "train_fake",
        }
    )
    assert enriched["fileExists"] is False
    assert enriched["status"] in {"missing", "invalid"}


def test_filter_evaluable_only_available_with_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "model_final.pth"
    ckpt.write_bytes(b"x" * 16)

    def fake_train_exists(_job_id: str) -> bool:
        return True

    monkeypatch.setattr(
        "app.services.model_asset_validation._training_job_exists",
        fake_train_exists,
    )

    assets = [
        enrich_model_asset(
            {
                "id": "model__ok",
                "checkpointPath": str(ckpt),
                "status": "ready",
                "sourceTrainingJobId": "train_ok",
                "backendType": "act",
                "taskType": "cable_threading",
            }
        ),
        enrich_model_asset(
            {
                "id": "model__bad",
                "checkpointPath": str(tmp_path / "missing.pth"),
                "status": "ready",
                "sourceTrainingJobId": "train_ok",
            }
        ),
    ]
    filtered = filter_evaluable_model_assets(assets, evaluation_task_type="cable_threading")
    assert len(filtered) == 1
    assert filtered[0]["id"] == "model__ok"


def test_block_stacking_rejects_act_backend() -> None:
    compatible, reason = is_model_asset_compatible_with_evaluation(
        {"backendType": "act", "taskType": "cable_threading"},
        evaluation_task_type="block_stacking",
    )
    assert compatible is False
    assert reason


def test_validate_model_asset_not_found() -> None:
    result = validate_model_asset("model__does_not_exist_9999")
    assert result.ok is False
    assert result.status == "missing"
