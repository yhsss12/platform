from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
from app.services import model_type_service as svc
from app.services.adapter_layer.model_type_training_config import (
    build_training_config_from_model_type,
    resolve_training_payload_from_model_type,
)


@pytest.fixture()
def model_type_db(monkeypatch):
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
    return TestSession


MUJOCO_MANIFEST = {
    "datasetId": "ds_test",
    "datasetName": "测试数据集",
    "taskType": "cable_threading",
    "observationSpace": {"type": "low_dim", "keys": ["robot0_eef_pos"]},
    "actionSpace": {"type": "continuous", "dim": 7},
    "artifacts": {"hdf5": "/tmp/dataset.hdf5"},
}


def test_build_training_config_merges_structure_and_params(model_type_db):
    svc.ensure_default_model_types(model_type_db())
    defn = svc.get_model_type("act")
    assert defn is not None

    merged = build_training_config_from_model_type(
        model_type_definition=defn,
        dataset_manifest=MUJOCO_MANIFEST,
        training_params={"epochs": 10, "batchSize": 8, "learningRate": 0.001, "seed": 42},
        save_policy={"saveFinal": True, "saveBest": True, "checkpointIntervalEpochs": 2},
    )

    assert merged["modelTypeId"] == "act"
    assert merged["trainingBackend"] == "act"
    assert merged["epochs"] == 10
    assert merged["batchSize"] == 8
    assert merged["modelParams"]["hidden_dim"] == 512
    assert merged["modelParams"]["chunk_size"] == 100
    assert merged["saveBest"] is True


def test_resolve_training_payload_from_model_type_id(model_type_db):
    svc.ensure_default_model_types(model_type_db())
    payload = resolve_training_payload_from_model_type(
        {
            "modelTypeId": "robomimic-bc",
            "datasetManifest": MUJOCO_MANIFEST,
            "epochs": 7,
            "batchSize": 32,
            "learningRate": 0.0002,
            "seed": 3,
            "saveFinal": True,
        }
    )
    assert payload["modelTypeId"] == "robomimic-bc"
    assert payload["trainingBackend"] == "robomimic_bc"
    assert payload["modelParams"]["actor_hidden_dims"] == "512,512"
    assert payload["epochs"] == 7


def test_resolve_training_payload_preserves_isaac_backend(model_type_db):
    svc.ensure_default_model_types(model_type_db())
    payload = resolve_training_payload_from_model_type(
        {
            "modelTypeId": "robomimic-bc",
            "trainingBackend": "isaac_robomimic_bc",
            "datasetManifest": {
                **MUJOCO_MANIFEST,
                "taskType": "isaac_block_stacking",
                "taskTemplateId": "isaac_block_stacking",
            },
        }
    )
    assert payload["trainingBackend"] == "isaac_robomimic_bc"


def test_resolve_training_payload_rejects_disabled(model_type_db):
    from fastapi import HTTPException

    svc.ensure_default_model_types(model_type_db())
    svc.update_model_type("diffusion-policy", {"status": "disabled"})
    with pytest.raises(HTTPException):
        resolve_training_payload_from_model_type(
            {
                "modelTypeId": "diffusion-policy",
                "datasetManifest": MUJOCO_MANIFEST,
            }
        )


def test_resolve_training_payload_legacy_act(model_type_db):
    svc.ensure_default_model_types(model_type_db())
    payload = resolve_training_payload_from_model_type(
        {
            "downstreamModelType": "ACT",
            "trainingBackend": "act",
            "datasetManifest": MUJOCO_MANIFEST,
            "epochs": 5,
        }
    )
    assert payload["modelTypeId"] == "act"
    assert payload["trainingBackend"] == "act"


def test_build_training_config_supports_pi0(model_type_db):
    svc.ensure_default_model_types(model_type_db())
    defn = svc.get_model_type("pi0")
    assert defn is not None

    merged = build_training_config_from_model_type(
        model_type_definition=defn,
        dataset_manifest=MUJOCO_MANIFEST,
        training_params={"epochs": 6, "batchSize": 4, "learningRate": 0.0002, "seed": 7},
        save_policy={"saveFinal": True},
    )
    assert merged["modelTypeId"] == "pi0"
    assert merged["trainingBackend"] == "pi0"
    assert merged["modelParams"]["context_window"] == 256
    assert merged["epochs"] == 6
    assert merged["batchSize"] == 4


def test_pi0_runner_unavailable_message_constant():
    from app.services.model_type_service import PI0_RUNNER_UNAVAILABLE_MESSAGE

    assert "openpi" in PI0_RUNNER_UNAVAILABLE_MESSAGE.lower()
