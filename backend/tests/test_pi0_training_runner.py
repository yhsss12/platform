from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
from app.services import model_type_service as model_svc
from app.services import training_service as training_svc
from app.services.pi0_training_runner import (
    PI0_HDF5_NOT_SUPPORTED_MESSAGE,
    PI0_RUNNER_DISABLED_REASON,
    PI0_RUNNER_SCRIPT,
    probe_pi0_training_capability,
    validate_pi0_dataset,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MOCK_OPENPI_ROOT = FIXTURES / "mock_openpi"


@pytest.fixture()
def pi0_db_env(tmp_path, monkeypatch):
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
    monkeypatch.setattr(model_svc, "SessionLocal", TestSession)
    model_svc.ensure_default_model_types(TestSession())

    jobs_root = tmp_path / "runs" / "training" / "jobs"
    jobs_root.mkdir(parents=True)
    monkeypatch.setattr(training_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(training_svc, "TRAINING_ROOT", tmp_path / "runs" / "training")
    monkeypatch.setattr(
        training_svc,
        "ALLOWED_PATH_ROOTS",
        [tmp_path.resolve(), MOCK_OPENPI_ROOT.resolve()],
    )
    return tmp_path


def _configure_mock_openpi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI0_RUNNER_ENABLED", "true")
    monkeypatch.setenv("OPENPI_ROOT", str(MOCK_OPENPI_ROOT))
    monkeypatch.setenv("OPENPI_PYTHON", os.environ.get("OPENPI_PYTHON") or "python3")
    monkeypatch.delenv("OPENPI_TRAIN_SCRIPT", raising=False)
    monkeypatch.delenv("PI0_USE_PLATFORM_SHIM", raising=False)
    monkeypatch.delenv("PI0_TRAIN_MODE", raising=False)


def test_probe_pi0_disabled_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PI0_RUNNER_ENABLED", raising=False)
    cap = probe_pi0_training_capability()
    assert cap["ready"] is False
    assert "PI0_RUNNER_ENABLED" in cap["reason"]


def test_probe_pi0_ready_with_mock_openpi(monkeypatch: pytest.MonkeyPatch):
    _configure_mock_openpi(monkeypatch)
    cap = probe_pi0_training_capability()
    assert cap["ready"] is True, cap


def test_probe_training_capabilities_includes_pi0_when_ready(monkeypatch: pytest.MonkeyPatch):
    _configure_mock_openpi(monkeypatch)
    caps = training_svc.probe_training_capabilities()
    assert "pi0" in caps["supportedTrainingBackends"]
    assert caps["pi0Capability"]["ready"] is True


def test_model_type_pi0_training_ready_follows_probe(monkeypatch: pytest.MonkeyPatch, pi0_db_env):
    _configure_mock_openpi(monkeypatch)
    monkeypatch.setattr(
        training_svc,
        "_probe_training_capabilities_uncached",
        lambda: {
            "supportedTrainingBackends": ["pi0"],
            "pi0Capability": {"ready": True, "reason": None, "pending": False, "evidence": []},
            "evidence": [],
        },
    )
    model_svc.refresh_model_type_readiness(force=True)
    rows = model_svc.list_model_types(status="available")
    pi0 = next(row for row in rows if row["modelTypeId"] == "pi0")
    assert pi0["trainingReady"] is True
    assert pi0.get("disabledReason") in (None, "")


def test_model_type_pi0_not_ready_without_openpi(monkeypatch: pytest.MonkeyPatch, pi0_db_env):
    monkeypatch.delenv("PI0_RUNNER_ENABLED", raising=False)
    monkeypatch.delenv("OPENPI_ROOT", raising=False)
    monkeypatch.setattr(
        training_svc,
        "_probe_training_capabilities_uncached",
        lambda: {
            "supportedTrainingBackends": ["act", "robomimic_bc"],
            "pi0Capability": {"ready": False, "reason": PI0_RUNNER_DISABLED_REASON, "pending": False},
            "evidence": [],
        },
    )
    model_svc.refresh_model_type_readiness(force=True)
    rows = model_svc.list_model_types(status="available")
    pi0 = next(row for row in rows if row["modelTypeId"] == "pi0")
    assert pi0["trainingReady"] is False
    assert pi0.get("disabledReason") in {PI0_RUNNER_DISABLED_REASON, "PI0_RUNNER_ENABLED 未启用"}


def _make_image_hdf5(path: Path, *, with_images: bool = True) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        if with_images:
            obs.create_dataset("agentview_image", data=np.zeros((8, 32, 32, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eef_pos", data=np.zeros((8, 3), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32))


def test_validate_pi0_dataset_requires_images(tmp_path: Path):
    hdf5 = tmp_path / "lowdim.hdf5"
    _make_image_hdf5(hdf5, with_images=False)
    train_config = {
        "adaptationSnapshot": {
            "modelAdaptation": {
                "inputConfig": {"camera_keys": ["agentview_image"]},
                "architectureConfig": {"language_conditioning": True},
            }
        }
    }
    ok, reason = validate_pi0_dataset(
        hdf5,
        train_config,
        manifest={"taskType": "pick_place", "datasetName": "test"},
    )
    assert ok is False
    assert "agentview_image" in reason


MANIFEST = {
    "datasetId": "ds_pi0",
    "datasetName": "pi0 test",
    "taskType": "cable_threading",
    "observationSpace": {"type": "image", "keys": ["agentview_image"]},
    "actionSpace": {"type": "continuous", "dim": 7},
    "episodes": 10,
    "successfulEpisodes": 10,
}


def test_create_training_job_pi0_rejected_without_openpi(pi0_db_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PI0_RUNNER_ENABLED", raising=False)
    monkeypatch.delenv("OPENPI_ROOT", raising=False)
    monkeypatch.delenv("OPENPI_BASE_CONFIG", raising=False)
    monkeypatch.setattr(
        training_svc,
        "probe_training_capabilities",
        lambda: {
            "supportedTrainingBackends": ["act", "robomimic_bc"],
            "pi0Capability": {"ready": False, "reason": PI0_RUNNER_DISABLED_REASON, "pending": False},
        },
    )
    hdf5 = pi0_db_env / "dataset.hdf5"
    _make_image_hdf5(hdf5)
    payload = {
        "datasetId": "ds_pi0",
        "modelTypeId": "pi0",
        "datasetManifest": {**MANIFEST, "artifacts": {"hdf5": str(hdf5)}},
        "epochs": 1,
        "batchSize": 4,
        "learningRate": 0.0001,
        "seed": 1,
    }
    with pytest.raises(HTTPException) as exc:
        training_svc.create_training_job(payload)
    assert exc.value.status_code == 400
    assert "openpi" in str(exc.value.detail).lower() or "PI0_RUNNER" in str(exc.value.detail)


def test_validate_pi0_dataset_rejects_hdf5_for_lerobot_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENPI_BASE_CONFIG", "pi05_libero")
    hdf5 = tmp_path / "dataset.hdf5"
    _make_image_hdf5(hdf5)
    train_config = {
        "adaptationSnapshot": {
            "modelAdaptation": {
                "inputConfig": {"camera_keys": ["agentview_image"]},
                "architectureConfig": {"language_conditioning": True},
            }
        }
    }
    ok, reason = validate_pi0_dataset(
        hdf5,
        train_config,
        manifest={"taskType": "pick_place", "artifacts": {"hdf5": str(hdf5)}},
    )
    assert ok is False
    assert PI0_HDF5_NOT_SUPPORTED_MESSAGE in reason


def test_pi0_training_with_mock_openpi_cli(monkeypatch: pytest.MonkeyPatch, pi0_db_env):
    monkeypatch.setenv("PI0_RUNNER_ENABLED", "true")
    monkeypatch.setenv("OPENPI_ROOT", str(MOCK_OPENPI_ROOT))
    monkeypatch.setenv("OPENPI_PYTHON", os.environ.get("OPENPI_PYTHON") or "python3")
    monkeypatch.setenv("OPENPI_BASE_CONFIG", "pi0_mock")
    monkeypatch.delenv("OPENPI_TRAIN_SCRIPT", raising=False)
    monkeypatch.delenv("PI0_USE_PLATFORM_SHIM", raising=False)

    hdf5 = pi0_db_env / "pi0_dataset.hdf5"
    _make_image_hdf5(hdf5)

    supported = {"robomimic_bc", "act", "diffusion_policy", "pi0"}
    monkeypatch.setattr(
        training_svc,
        "probe_training_capabilities",
        lambda: {
            "supportedTrainingBackends": list(supported),
            "pi0Capability": {"ready": True, "reason": None, "evidence": []},
        },
    )

    result = training_svc.create_training_job(
        {
            "datasetId": "ds_pi0",
            "modelTypeId": "pi0",
            "datasetManifest": {**MANIFEST, "artifacts": {"hdf5": str(hdf5)}},
            "trainingBackend": "pi0",
            "downstreamModelType": "pi0",
            "epochs": 1,
            "batchSize": 4,
            "learningRate": 0.0001,
            "device": "cpu",
            "seed": 1,
        }
    )

    train_job_id = result["trainJobId"]
    train_job_dir = training_svc._train_job_dir(train_job_id)

    deadline = time.time() + 120
    status: dict = {}
    while time.time() < deadline:
        status_path = train_job_dir / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") in {"completed", "failed", "backend_unavailable"}:
                break
        time.sleep(1)

    assert status.get("status") == "completed", status.get("message")
    log_text = (train_job_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "--exp-name" in log_text
    assert "--config-path" not in log_text
    assert "--output-dir" not in log_text

    final_path = train_job_dir / "checkpoints" / "pi0" / "checkpoints" / "model_final.pt"
    assert final_path.is_file()

    metrics_path = train_job_dir / "artifacts" / "metrics.jsonl"
    assert metrics_path.is_file()
    assert metrics_path.read_text(encoding="utf-8").strip()

    registry = train_job_dir / "artifacts" / "model_assets_registry.json"
    assets = json.loads(registry.read_text(encoding="utf-8")).get("assets") or []
    final_assets = [a for a in assets if (a.get("checkpointKind") or "").lower() == "final"]
    assert final_assets
    assert final_assets[0].get("modelType") == "pi0"


def test_create_training_job_pi0_converts_hdf5_for_pi05_libero(monkeypatch: pytest.MonkeyPatch, pi0_db_env):
    _configure_mock_openpi(monkeypatch)
    monkeypatch.setenv("OPENPI_BASE_CONFIG", "pi05_libero")
    hdf5 = pi0_db_env / "dataset.hdf5"
    _make_image_hdf5(hdf5)
    payload = {
        "datasetId": "ds_pi0",
        "modelTypeId": "pi0",
        "datasetManifest": {**MANIFEST, "artifacts": {"hdf5": str(hdf5)}},
        "epochs": 1,
        "batchSize": 4,
        "learningRate": 0.0001,
        "seed": 1,
    }
    supported = {"robomimic_bc", "act", "diffusion_policy", "pi0"}
    monkeypatch.setattr(
        training_svc,
        "probe_training_capabilities",
        lambda: {
            "supportedTrainingBackends": list(supported),
            "pi0Capability": {"ready": True, "reason": None, "evidence": []},
        },
    )
    result = training_svc.create_training_job(payload)
    train_job_dir = training_svc._train_job_dir(result["trainJobId"])
    lerobot_root = train_job_dir / "artifacts" / "lerobot_dataset"
    assert (lerobot_root / "meta" / "info.json").is_file()
    assert any((lerobot_root / "data" / "chunk-000").glob("episode_*"))
