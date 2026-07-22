from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
from app.services import model_type_service as svc
from app.services import training_service as training_svc
from app.services.pi0_training_runner import PI0_RUNNER_DISABLED_REASON


@pytest.fixture()
def pi0_env(tmp_path, monkeypatch):
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

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
    monkeypatch.setattr(svc, "SessionLocal", TestSession)
    svc.ensure_default_model_types(TestSession())

    jobs_root = tmp_path / "runs" / "training" / "jobs"
    jobs_root.mkdir(parents=True)
    monkeypatch.setattr(training_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(training_svc, "TRAINING_ROOT", tmp_path / "runs" / "training")

    return jobs_root


MANIFEST = {
    "datasetId": "ds_pi0",
    "datasetName": "pi0 test",
    "taskType": "cable_threading",
    "observationSpace": {"type": "image", "keys": ["agentview_image"]},
    "actionSpace": {"type": "continuous", "dim": 7},
    "episodes": 10,
    "successfulEpisodes": 10,
    "artifacts": {"hdf5": "/tmp/dataset.hdf5"},
}


def test_create_training_job_pi0_runner_unavailable(pi0_env):
    payload = {
        "datasetId": "ds_pi0",
        "modelTypeId": "pi0",
        "datasetManifest": MANIFEST,
        "epochs": 5,
        "batchSize": 8,
        "learningRate": 0.0001,
        "seed": 1,
    }
    with patch.object(
        training_svc,
        "probe_training_capabilities",
        return_value={
            "supportedTrainingBackends": ["act", "robomimic_bc"],
            "pi0Capability": {"ready": False, "reason": PI0_RUNNER_DISABLED_REASON},
        },
    ):
        with pytest.raises(HTTPException) as exc:
            training_svc.create_training_job(payload)
    assert exc.value.status_code == 400
    assert "openpi" in str(exc.value.detail).lower() or "OPENPI" in str(exc.value.detail)
