from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
from app.services import model_type_service as mt_svc
from app.services import training_service as training_svc

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MOCK_OPENPI_ROOT = FIXTURES / "mock_openpi"


@pytest.fixture()
def pi0_structure_env(tmp_path, monkeypatch):
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
    monkeypatch.setattr(mt_svc, "SessionLocal", TestSession)
    mt_svc.ensure_default_model_types(TestSession())
    mt_svc.create_model_type(
        {
            "name": "Custom pi0",
            "modelTypeId": "custom-pi0-vit",
            "baseAlgorithm": "pi0",
            "structureConfig": {
                "action_horizon": 24,
                "context_window": 128,
                "vision_encoder": "vit",
                "language_conditioning": True,
            },
            "trainingDefaults": {
                "default_epochs": 2,
                "default_batch_size": 6,
                "default_learning_rate": 0.0002,
            },
            "status": "available",
        }
    )

    monkeypatch.setattr(training_svc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(training_svc, "TRAINING_ROOT", tmp_path / "runs" / "training")
    monkeypatch.setattr(
        training_svc,
        "ALLOWED_PATH_ROOTS",
        [tmp_path.resolve(), MOCK_OPENPI_ROOT.resolve()],
    )
    monkeypatch.setenv("PI0_RUNNER_ENABLED", "true")
    monkeypatch.setenv("OPENPI_ROOT", str(MOCK_OPENPI_ROOT))
    monkeypatch.setenv("OPENPI_PYTHON", os.environ.get("OPENPI_PYTHON") or "python3")
    monkeypatch.setenv("OPENPI_BASE_CONFIG", "pi0_mock")
    return tmp_path


def _make_image_hdf5(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((8, 32, 32, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eef_pos", data=np.zeros((8, 3), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32))


def test_custom_pi0_structure_config_written_to_openpi_yaml(pi0_structure_env, monkeypatch):
    hdf5 = pi0_structure_env / "dataset.hdf5"
    _make_image_hdf5(hdf5)
    monkeypatch.setattr(
        training_svc,
        "probe_training_capabilities",
        lambda: {
            "supportedTrainingBackends": ["pi0"],
            "pi0Capability": {"ready": True, "reason": None, "evidence": []},
        },
    )

    with patch.object(training_svc, "_execute_training_job", side_effect=lambda job_id: None):
        result = training_svc.create_training_job(
            {
                "datasetId": "ds_pi0_custom",
                "modelTypeId": "custom-pi0-vit",
                "datasetManifest": {
                    "datasetId": "ds_pi0_custom",
                    "datasetName": "pi0 custom",
                    "taskType": "cable_threading",
                    "observationSpace": {"type": "image", "keys": ["agentview_image"]},
                    "actionSpace": {"type": "continuous", "dim": 7},
                    "artifacts": {"hdf5": str(hdf5)},
                },
            }
        )

    train_job_dir = training_svc._train_job_dir(result["trainJobId"])
    train_config = training_svc._read_json(train_job_dir / "config" / "train_config.json")
    assert train_config["modelTypeId"] == "custom-pi0-vit"
    assert train_config["structureConfig"]["action_horizon"] == 24
    assert train_config["structureConfig"]["vision_encoder"] == "vit"
    assert train_config["epochs"] == 2
    assert train_config["batchSize"] == 6
    assert train_config["learningRate"] == 0.0002

    config_path = train_job_dir / "config" / "openpi_platform_config.yaml"
    if not config_path.is_file():
        config_path = train_job_dir / "config" / "openpi_platform_config.json"
    assert config_path.is_file()
    if config_path.suffix == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["training"]["batch_size"] == 6
    assert payload["training"]["learning_rate"] == 0.0002
    assert payload["structure"]["action_horizon"] == 24
    assert payload["structure"]["context_window"] == 128
    assert payload["structure"]["vision_encoder"] == "vit"
