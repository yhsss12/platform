from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import training_service as training_svc
from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/dataset.manifest.json"
)


@pytest.fixture()
def pi0_platform_env(tmp_path, monkeypatch):
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.base import Base
    from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
    from app.services import model_type_service as model_svc

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, _compiler, **_kw):
        return "JSON"

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(model_svc, "SessionLocal", TestSession)
    model_svc.ensure_default_model_types(TestSession())

    jobs_root = tmp_path / "runs" / "training" / "jobs"
    jobs_root.mkdir(parents=True)
    monkeypatch.setattr(training_svc, "PROJECT_ROOT", Path(__file__).resolve().parents[2])
    monkeypatch.setattr(training_svc, "TRAINING_ROOT", tmp_path / "runs" / "training")
    return jobs_root


def _load_smoke_manifest() -> dict:
    assert SMOKE_MANIFEST_PATH.is_file(), f"missing smoke manifest: {SMOKE_MANIFEST_PATH}"
    return json.loads(SMOKE_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_pi0_platform_training_uses_lerobot_path_without_hdf5_converter(pi0_platform_env):
    manifest = _load_smoke_manifest()
    manifest["datasetId"] = "ds_pi0_lerobot_smoke"
    manifest["datasetName"] = "pi0 lerobot smoke"
    manifest.pop("artifacts", None)

    payload = {
        "datasetId": manifest["datasetId"],
        "modelTypeId": "pi0",
        "datasetManifest": manifest,
        "epochs": 1,
        "batchSize": 2,
        "maxSteps": 10,
        "seed": 1,
        "datasetFormat": "lerobot",
        "taskInstruction": "thread the cable through the pole",
    }

    with patch.object(
        training_svc,
        "probe_training_capabilities",
        return_value={
            "supportedTrainingBackends": ["pi0"],
            "pi0Capability": {"ready": True, "platformTrainingReady": True},
        },
    ), patch(
        "app.services.pi0_hdf5_converter.convert_hdf5_to_lerobot_index"
    ) as convert_mock, patch.object(
        training_svc,
        "_execute_training_job",
        return_value=None,
    ):
        created = training_svc.create_training_job(payload)

    convert_mock.assert_not_called()
    train_job_dir = training_svc._train_job_dir(created["trainJobId"])
    train_config = json.loads((train_job_dir / "config" / "train_config.json").read_text(encoding="utf-8"))
    assert train_config["datasetFormat"] == "lerobot"
    assert train_config["maxSteps"] == 10
    assert train_config["taskInstruction"] == "thread the cable through the pole"
    lerobot_path = resolve_lerobot_path_from_manifest(
        json.loads((train_job_dir / "artifacts" / "dataset_manifest.json").read_text(encoding="utf-8"))
    )
    assert lerobot_path is not None
    assert lerobot_path.name == "lerobot_dataset"


def test_pi0_platform_smoke_generates_status_metrics_and_checkpoint(pi0_platform_env):
    manifest = _load_smoke_manifest()
    manifest["datasetId"] = "ds_pi0_platform_smoke"
    manifest["datasetName"] = "pi0 platform smoke"

    payload = {
        "datasetId": manifest["datasetId"],
        "modelTypeId": "pi0",
        "datasetManifest": manifest,
        "epochs": 1,
        "batchSize": 2,
        "maxSteps": 10,
        "seed": 1,
    }

    with patch.object(
        training_svc,
        "probe_training_capabilities",
        return_value={"supportedTrainingBackends": ["pi0"], "pi0Capability": {"ready": True}},
    ), patch.object(training_svc, "sync_workspace_job_from_runtime"), patch.object(
        training_svc, "finalize_training_job_sync"
    ), patch.object(
        training_svc, "record_workspace_job_start"
    ), patch.object(
        training_svc.threading,
        "Thread",
        lambda *args, **kwargs: type("NoopThread", (), {"start": lambda self: None})(),
    ):
        created = training_svc.create_training_job(payload)
        training_svc._execute_training_job(created["trainJobId"])

    train_job_dir = training_svc._train_job_dir(created["trainJobId"])
    status = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status.get("datasetFormat") == "lerobot" or status.get("modelType") == "pi0"
    assert (train_job_dir / "logs" / "train.log").is_file()
    assert (train_job_dir / "artifacts" / "metrics.jsonl").is_file()
    assert (train_job_dir / "checkpoints" / "pi0" / "checkpoints" / "model_final.pt").is_file()
    train_config = json.loads((train_job_dir / "config" / "train_config.json").read_text(encoding="utf-8"))
    assert train_config["datasetFormat"] == "lerobot"
