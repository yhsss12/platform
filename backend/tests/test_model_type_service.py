from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
from app.services import model_type_service as svc


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


def test_default_model_types_seeded(model_type_db):
    rows = svc.list_model_types()
    ids = {row["modelTypeId"] for row in rows}
    assert "robomimic-bc" in ids
    assert "act" in ids
    assert "diffusion-policy" in ids
    assert "pi0" in ids
    available = [row for row in rows if row["status"] == "available"]
    assert len(available) >= 4


def test_list_model_types_seeds_empty_table(model_type_db):
    rows = svc.list_model_types()
    assert len(rows) >= 4


def test_list_model_types_raises_when_table_missing(model_type_db, monkeypatch):
    from fastapi import HTTPException

    def _fake_inspect(_bind):
        class _Inspector:
            def get_table_names(self):
                return []

        return _Inspector()

    monkeypatch.setattr(svc, "inspect", _fake_inspect)
    with pytest.raises(HTTPException) as exc:
        svc.list_model_types()
    assert exc.value.status_code == 503
    assert "022_model_type_definitions" in str(exc.value.detail)


def test_list_model_types_not_blocked_by_slow_pi0_probe(model_type_db, monkeypatch):
    import time
    from app.services import training_service as training_svc

    def _slow_full_probe():
        time.sleep(2.0)
        return training_svc._probe_training_capabilities_uncached()

    monkeypatch.setattr(training_svc, "_probe_training_capabilities_uncached", _slow_full_probe)

    started = time.time()
    rows = svc.list_model_types()
    elapsed = time.time() - started

    assert elapsed < 1.0
    assert len(rows) >= 4
    pi0 = next(row for row in rows if row["modelTypeId"] == "pi0")
    assert pi0["trainingReadinessStatus"] == "pending"
    assert pi0["trainingReady"] is False
    assert pi0["disabledReason"] == svc.PI0_PROBE_PENDING_MESSAGE
    act = next(row for row in rows if row["modelTypeId"] == "act")
    assert act["trainingReady"] is True


def test_background_probe_persists_readiness_to_db(model_type_db, monkeypatch):
    from app.services import training_service as training_svc

    def _mock_capabilities():
        return {
            "supportedTrainingBackends": ["robomimic_bc", "act", "diffusion_policy", "pi0"],
            "pi0Capability": {
                "ready": True,
                "reason": None,
                "pending": False,
                "status": "ready",
                "evidence": ["openpi ok"],
            },
            "evidence": ["scripts ok"],
        }

    monkeypatch.setattr(training_svc, "_probe_training_capabilities_uncached", _mock_capabilities)
    svc.refresh_model_type_readiness(force=True)

    db = model_type_db()
    pi0 = (
        db.query(ModelTypeDefinition)
        .filter(ModelTypeDefinition.model_type_id == "pi0")
        .one()
    )
    assert pi0.training_ready is True
    assert pi0.training_readiness_status == "ready"
    assert pi0.capability_checked_at is not None
    assert pi0.capability_evidence == ["openpi ok"]


def test_readiness_survives_service_recreate(model_type_db, monkeypatch):
    from app.services import training_service as training_svc

    def _mock_capabilities():
        return {
            "supportedTrainingBackends": ["robomimic_bc", "act", "diffusion_policy"],
            "pi0Capability": {
                "ready": False,
                "reason": svc.PI0_RUNNER_UNAVAILABLE_MESSAGE,
                "pending": False,
                "status": "disabled",
                "evidence": [],
            },
            "evidence": [],
        }

    monkeypatch.setattr(training_svc, "_probe_training_capabilities_uncached", _mock_capabilities)
    svc.refresh_model_type_readiness(force=True)

    rows = svc.list_model_types(status="available")
    by_id = {row["modelTypeId"]: row for row in rows}
    assert by_id["pi0"]["trainingReadinessStatus"] == "unavailable"
    assert by_id["pi0"]["disabledReason"] == svc.PI0_RUNNER_UNAVAILABLE_MESSAGE
    assert by_id["act"]["trainingReady"] is True


def test_create_custom_bc_model_type(model_type_db):
    created = svc.create_model_type(
        {
            "name": "自定义 BC",
            "modelTypeId": "custom-bc-test",
            "baseAlgorithm": "robomimic_bc",
            "structureConfig": {"actor_hidden_dims": "256,256"},
            "status": "available",
        }
    )
    assert created["modelTypeId"] == "custom-bc-test"
    assert created["adapterKey"] == "robomimic_bc_adapter"


def test_create_model_type_duplicate_id_raises(model_type_db):
    svc.create_model_type(
        {
            "name": "自定义 BC 1",
            "modelTypeId": "dup-bc",
            "baseAlgorithm": "robomimic_bc",
            "structureConfig": {"actor_hidden_dims": "512,512"},
        }
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        svc.create_model_type(
            {
                "name": "自定义 BC 2",
                "modelTypeId": "dup-bc",
                "baseAlgorithm": "robomimic_bc",
                "structureConfig": {"actor_hidden_dims": "512,512"},
            }
        )
    assert exc.value.status_code == 409


def test_create_model_type_invalid_structure_raises(model_type_db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        svc.create_model_type(
            {
                "name": "无效 BC",
                "modelTypeId": "invalid-bc",
                "baseAlgorithm": "robomimic_bc",
                "structureConfig": {"actor_hidden_dims": "not-valid-dims"},
                "status": "available",
            }
        )
    assert exc.value.status_code == 400


def test_validate_structure_config_act():
    errors = svc.validate_structure_config("act", {"hidden_dim": 512, "chunk_size": 20, "n_action_steps": 20})
    assert errors == []


def test_resolve_legacy_model_type_id():
    assert svc.resolve_legacy_model_type_id(downstream_model_type="ACT") == "act"
    assert svc.resolve_legacy_model_type_id(training_backend="robomimic_bc") == "robomimic-bc"


def test_get_available_model_type_rejects_disabled(model_type_db):
    from fastapi import HTTPException

    svc.ensure_default_model_types(model_type_db())
    svc.update_model_type("act", {"status": "disabled"})
    with pytest.raises(HTTPException) as exc:
        svc.get_available_model_type("act")
    assert exc.value.status_code == 400


def test_get_available_model_type_not_found(model_type_db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        svc.get_available_model_type("missing-model")
    assert exc.value.status_code == 404


def test_resolve_training_readiness_for_adapter():
    supported = {"robomimic_bc", "act", "diffusion_policy"}

    ready, reason = svc.resolve_training_readiness_for_adapter(
        "robomimic_bc_adapter", supported_backends=supported
    )
    assert ready is True
    assert reason is None

    ready, reason = svc.resolve_training_readiness_for_adapter("act_adapter", supported_backends=supported)
    assert ready is True
    assert reason is None

    ready, reason = svc.resolve_training_readiness_for_adapter(
        "diffusion_policy_adapter", supported_backends=supported
    )
    assert ready is True
    assert reason is None

    ready, reason = svc.resolve_training_readiness_for_adapter(
        "pi0_adapter",
        supported_backends=supported,
        pi0_capability={"ready": False, "reason": svc.PI0_RUNNER_UNAVAILABLE_MESSAGE},
    )
    assert ready is False
    assert reason == svc.PI0_RUNNER_UNAVAILABLE_MESSAGE


def test_list_model_types_includes_training_readiness_from_db(model_type_db, monkeypatch):
    from app.services import training_service as training_svc

    def _mock_capabilities():
        return {
            "supportedTrainingBackends": ["robomimic_bc", "act", "diffusion_policy"],
            "pi0Capability": {
                "ready": False,
                "reason": svc.PI0_RUNNER_UNAVAILABLE_MESSAGE,
                "pending": False,
                "status": "disabled",
            },
            "evidence": [],
        }

    monkeypatch.setattr(training_svc, "_probe_training_capabilities_uncached", _mock_capabilities)
    svc.refresh_model_type_readiness(force=True)
    rows = svc.list_model_types(status="available")
    by_id = {row["modelTypeId"]: row for row in rows}

    assert by_id["pi0"]["trainingReady"] is False
    assert by_id["pi0"]["disabledReason"] == svc.PI0_RUNNER_UNAVAILABLE_MESSAGE
    assert by_id["pi0"]["trainingReadinessStatus"] == "unavailable"
    assert by_id["act"]["trainingReady"] is True
    assert by_id["diffusion-policy"]["trainingReady"] is True
    assert by_id["robomimic-bc"]["trainingReady"] is True
