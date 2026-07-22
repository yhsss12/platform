from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.workspace import model_types as model_types_routes
from app.core.deps import get_current_user
from app.db.base import Base
from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
from app.services import model_type_service as svc


@pytest.fixture()
def client(monkeypatch) -> TestClient:
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

    async def _run_sync(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _run_sync)

    app = FastAPI()
    app.include_router(model_types_routes.router, prefix="/api/workspace")

    async def _fake_user():
        return SimpleNamespace(id="test-user", role="SUPER_ADMIN", is_active=True)

    app.dependency_overrides[get_current_user] = _fake_user
    return TestClient(app)


def test_api_list_model_types_seeds_empty_db(client: TestClient):
    response = client.get("/api/workspace/model-types")
    assert response.status_code == 200
    body = response.json()
    assert "modelTypes" in body
    assert body["total"] == len(body["modelTypes"])
    ids = {item["modelTypeId"] for item in body["modelTypes"]}
    assert ids >= {"robomimic-bc", "act", "diffusion-policy", "pi0"}


def test_api_list_model_types_response_shape(client: TestClient):
    response = client.get("/api/workspace/model-types")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("modelTypes"), list)
    assert body.get("total") == len(body["modelTypes"])


def test_api_model_types_training_readiness_from_db(client: TestClient, monkeypatch):
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

    response = client.get("/api/workspace/model-types?status=available")
    assert response.status_code == 200
    by_id = {item["modelTypeId"]: item for item in response.json()["modelTypes"]}

    assert by_id["robomimic-bc"]["trainingReady"] is True
    assert by_id["robomimic-bc"].get("disabledReason") in (None, "")

    assert by_id["act"]["trainingReady"] is True
    assert by_id["diffusion-policy"]["trainingReady"] is True

    assert by_id["pi0"]["trainingReady"] is False
    assert by_id["pi0"]["disabledReason"] == svc.PI0_RUNNER_UNAVAILABLE_MESSAGE
    assert by_id["pi0"]["trainingReadinessStatus"] == "unavailable"


def test_api_list_not_blocked_by_slow_pi0_probe(client: TestClient, monkeypatch):
    import time
    from app.services import training_service as training_svc

    def _slow_full_probe():
        time.sleep(2.0)
        return training_svc._probe_training_capabilities_uncached()

    monkeypatch.setattr(training_svc, "_probe_training_capabilities_uncached", _slow_full_probe)

    started = time.time()
    response = client.get("/api/workspace/model-types")
    elapsed = time.time() - started

    assert response.status_code == 200
    assert elapsed < 1.0
    body = response.json()
    assert len(body["modelTypes"]) >= 4
    by_id = {item["modelTypeId"]: item for item in body["modelTypes"]}
    assert by_id["act"]["trainingReady"] is True
    assert by_id["pi0"]["trainingReady"] is False
    assert by_id["pi0"]["trainingReadinessStatus"] == "pending"


def test_api_probe_refresh_accepts_and_persists(client: TestClient, monkeypatch):
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
            "evidence": [],
        }

    monkeypatch.setattr(training_svc, "_probe_training_capabilities_uncached", _mock_capabilities)

    refresh_resp = client.post("/api/workspace/model-types/probe/refresh")
    assert refresh_resp.status_code == 200
    assert refresh_resp.json()["accepted"] is True

    svc.refresh_model_type_readiness(force=True)
    list_resp = client.get("/api/workspace/model-types")
    by_id = {item["modelTypeId"]: item for item in list_resp.json()["modelTypes"]}
    assert by_id["pi0"]["trainingReady"] is True
    assert by_id["pi0"]["trainingReadinessStatus"] == "ready"


def test_api_create_and_validate_model_type(client: TestClient):
    create_resp = client.post(
        "/api/workspace/model-types",
        json={
            "name": "API 自定义 BC",
            "modelTypeId": "api-custom-bc",
            "baseAlgorithm": "robomimic_bc",
            "structureConfig": {"actor_hidden_dims": "256,256"},
            "status": "available",
        },
    )
    assert create_resp.status_code == 200
    assert create_resp.json()["modelTypeId"] == "api-custom-bc"

    validate_resp = client.post("/api/workspace/model-types/api-custom-bc/validate")
    assert validate_resp.status_code == 200
    assert validate_resp.json()["valid"] is True
