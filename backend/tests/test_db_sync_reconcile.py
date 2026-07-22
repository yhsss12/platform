from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest_plugins = ["tests.test_training_job_sync"]

from app.core.db_health import check_db_health, parse_database_target
from app.core.db_session import db_session_scope


def test_parse_database_target_masks_password():
    target = parse_database_target("postgresql://admin:secret@127.0.0.1:5432/eai_ide")
    assert target["host"] == "127.0.0.1"
    assert target["port"] == 5432
    assert target["database"] == "eai_ide"
    assert target["user"] == "admin"


def test_status_payload_prefers_runtime_without_db_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.services import training_service as svc

    job_id = "train_20260626_125605_11be"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "config").mkdir()
    (job_dir / "logs").mkdir()
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "trainJobId": job_id,
                "status": "completed",
                "progress": 1.0,
                "epoch": 1,
                "totalEpochs": 1,
                "executionMode": "remote_ssh",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "logs" / "train.log").write_text("Epoch 1 Loss: 0.999762\n", encoding="utf-8")
    (job_dir / "config" / "train_config.json").write_text(json.dumps({"epochs": 1}), encoding="utf-8")

    monkeypatch.setattr(svc, "_train_job_dir", lambda _job: job_dir)
    monkeypatch.setattr(
        "app.services.training_job_sync_service.sync_training_job_from_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not sync")),
    )

    payload = svc._status_payload(job_id, sync_db=False)
    assert payload["status"] == "completed"
    assert payload["progress"] == 1.0


def test_list_model_assets_detail_without_sync_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.services import workspace_model_asset_service as asset_svc

    job_id = "train_db_sync_test"
    called = {"sync": 0}

    def _sync(*_args, **_kwargs):
        called["sync"] += 1
        return {"ok": True}

    monkeypatch.setattr(
        "app.services.training_job_sync_service.sync_training_job_from_runtime",
        _sync,
    )
    monkeypatch.setattr(asset_svc, "TRAINING_JOBS_ROOT", tmp_path)
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "config").mkdir()
    (job_dir / "artifacts").mkdir()
    (job_dir / "status.json").write_text(json.dumps({"status": "completed", "totalEpochs": 1}), encoding="utf-8")
    (job_dir / "config" / "train_config.json").write_text(json.dumps({"epochs": 1}), encoding="utf-8")

    monkeypatch.setattr(
        "app.services.model_asset_db_service.list_training_job_model_assets_detail_from_db",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        asset_svc,
        "_resolve_job_backend_context",
        lambda *_args, **_kwargs: ({}, {}, {}, "diffusion_policy", "Diffusion Policy", "diffusion_policy"),
    )
    monkeypatch.setattr(asset_svc, "list_training_job_detail_registry_entries", lambda **_kwargs: [])
    monkeypatch.setattr(asset_svc, "resolve_training_job_model_assets_list_message", lambda **_kwargs: None)

    payload = asset_svc.list_training_job_model_assets_detail(job_id, sync_db=False)
    assert called["sync"] == 0
    assert payload["modelAssets"] == []


def test_sync_training_job_returns_step_result(sync_env):
    from tests.test_training_job_sync import _seed_workspace_job, _write_completed_job
    from app.services import training_job_sync_service as sync_svc

    jobs_root, Session = sync_env
    job_id = "train_20260626_125605_11be"
    train_job_dir = _write_completed_job(jobs_root, job_id)
    _seed_workspace_job(Session, job_id, train_job_dir)

    result = sync_svc.sync_training_job_from_runtime(job_id)
    assert result["jobId"] == job_id
    assert result["steps"].get("sync_status_metrics", {}).get("ok") is True


def test_reconcile_local_only_skips_ssh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys):
    import sys

    import tools.maintenance.reconcile_remote_training_job as reconcile_mod

    job_id = "train_20260626_125605_11be"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "config").mkdir()
    (job_dir / "artifacts").mkdir()
    (job_dir / "logs").mkdir()
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "executionMode": "remote_ssh",
                "epoch": 1,
                "totalEpochs": 1,
                "progress": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "config" / "train_config.json").write_text(json.dumps({"epochs": 1}), encoding="utf-8")
    (job_dir / "logs" / "train.log").write_text("Epoch 1 Loss: 0.999762\n", encoding="utf-8")

    monkeypatch.setattr(
        "app.core.db_health.check_db_health",
        lambda **_kwargs: type(
            "H",
            (),
            {
                "level": "DB_HEALTH_OK",
                "ok": True,
                "connect_ms": 1.0,
                "select1_ms": 1.0,
                "idle_in_transaction": 0,
                "blocking_locks": 0,
                "errors": [],
                "warnings": [],
            },
        )(),
    )
    monkeypatch.setattr(
        "app.services.workspace_runtime_paths.resolve_training_job_root",
        lambda _job: job_dir,
    )
    monkeypatch.setattr(
        "app.services.training_remote_runner.reconcile_remote_training_job_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ssh should not run")),
    )
    monkeypatch.setattr(
        "app.services.training_job_sync_service.sync_training_job_from_runtime",
        lambda *_args, **_kwargs: {"ok": True, "steps": {"sync_status_metrics": {"ok": True}}},
    )
    monkeypatch.setattr(
        "app.services.training_job_sync_service.get_training_job_summary_from_db",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "progress": 1.0,
            "epoch": 1,
            "totalEpochs": 1,
        },
    )
    monkeypatch.setattr(
        "app.services.workspace_model_asset_service.list_training_job_model_assets_detail",
        lambda *_args, **_kwargs: {
            "modelAssets": [{"checkpointKind": "final", "displayStatus": "ready", "status": "ready"}],
            "warning": None,
        },
    )
    monkeypatch.setattr(
        "app.core.database.engine",
        type("E", (), {"dispose": lambda self: None})(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_remote_training_job.py",
            job_id,
            "--local-only",
        ],
    )

    rc = reconcile_mod.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "DB_UNAVAILABLE" not in out
    assert '"ok": true' in out.lower()
