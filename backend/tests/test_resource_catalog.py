from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes_workspace_resources
from app.api.workspace import model_assets as model_assets_routes
from app.api.workspace import model_types as model_types_routes
from app.api.workspace import task_templates as task_templates_routes
from app.core.deps import get_current_user
from app.db.base import Base
from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
from app.models.resource_definition import ResourceDefinition  # noqa: F401
from app.models.task_template_catalog import TaskTemplateCatalog  # noqa: F401
from app.models.workspace_index import ModelAsset  # noqa: F401
from app.models.workspace_job import WorkspaceJob  # noqa: F401
from app.services import model_type_service as model_type_svc
from app.services import resource_definition_service as resource_svc
from app.services import task_template_catalog_service as template_svc


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

    for module in (resource_svc, template_svc, model_type_svc):
        monkeypatch.setattr(module, "SessionLocal", TestSession)

    async def _run_sync(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _run_sync)

    app = FastAPI()
    app.include_router(routes_workspace_resources.router, prefix="/api/workspace")
    app.include_router(task_templates_routes.router, prefix="/api/workspace")
    app.include_router(model_types_routes.router, prefix="/api/workspace")

    async def _fake_user():
        return SimpleNamespace(id="test-user", role="SUPER_ADMIN", is_active=True)

    app.dependency_overrides[get_current_user] = _fake_user
    return TestClient(app)


def test_migration_tables_exist():
  """resource_definitions 与 task_template_catalog 可由 ORM 创建。"""
  from sqlalchemy.dialects.postgresql import JSONB
  from sqlalchemy.ext.compiler import compiles

  @compiles(JSONB, "sqlite")
  def _compile_jsonb_sqlite(_type, _compiler, **_kw):
      return "JSON"

  from app.models.workspace_job import WorkspaceJob  # noqa: F401

  engine = create_engine("sqlite:///:memory:")
  Base.metadata.create_all(engine)
  from sqlalchemy import inspect

  tables = set(inspect(engine).get_table_names())
  assert "resource_definitions" in tables
  assert "task_template_catalog" in tables


def test_reindex_idempotent(monkeypatch):
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
    monkeypatch.setattr(resource_svc, "SessionLocal", TestSession)

    from app.services import resource_registry_service as registry

    registry.scan_resource_registry(force=True)
    first = resource_svc.reindex_resource_registry_to_db()
    second = resource_svc.reindex_resource_registry_to_db()

    assert first["synced"] > 0
    assert first["created"] > 0
    assert second["created"] == 0
    assert second["synced"] == first["synced"]


def test_physics_proxy_seed_and_overview(client: TestClient):
    resource_svc.reindex_resource_registry_to_db()
    resource_svc.seed_physics_proxy_models()
    template_svc.seed_default_task_templates()

    overview = client.get("/api/workspace/resources/overview")
    assert overview.status_code == 200
    body = overview.json()
    assert body["physicsProxies"] == 3
    assert body["metrics"] >= 1
    assert body["taskTemplates"] == 5

    proxies = client.get("/api/workspace/resources?resourceType=physics_proxy")
    assert proxies.status_code == 200
    assert proxies.json()["total"] == body["physicsProxies"]


def test_metrics_from_resource_definitions_not_eval_summary(client: TestClient):
    resource_svc.reindex_resource_registry_to_db()

    metrics = client.get("/api/workspace/resources?assetType=metric")
    assert metrics.status_code == 200
    total = metrics.json()["total"]
    overview = client.get("/api/workspace/resources/overview").json()
    assert overview["metrics"] == total
    for item in metrics.json()["resources"]:
        assert item["assetType"] == "metric"


def test_task_templates_from_catalog(client: TestClient):
    response = client.get("/api/workspace/task-templates")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == len(body["taskTemplates"])
    assert body["total"] >= 5
    ids = {item["id"] for item in body["taskTemplates"]}
    assert "cable_threading_single_arm" in ids


def test_overview_matches_detail_counts(client: TestClient):
    resource_svc.reindex_resource_registry_to_db()
    resource_svc.seed_physics_proxy_models()
    template_svc.seed_default_task_templates()

    overview = client.get("/api/workspace/resources/overview").json()

    scenes = client.get("/api/workspace/resources?assetType=scene").json()["total"]
    robots = client.get("/api/workspace/resources?assetType=robot").json()["total"]
    objects = client.get("/api/workspace/resources?assetType=object").json()["total"]
    end_effectors = client.get("/api/workspace/resources?assetType=end_effector").json()["total"]
    policies = client.get("/api/workspace/resources?assetType=policy").json()["total"]
    craft = client.get("/api/workspace/resources?assetType=task").json()["total"]
    metrics = client.get("/api/workspace/resources?assetType=metric").json()["total"]
    proxies = client.get("/api/workspace/resources?resourceType=physics_proxy").json()["total"]
    templates = client.get("/api/workspace/task-templates").json()["total"]
    model_types = client.get("/api/workspace/model-types").json()["total"]

    assert overview["scenes"] == scenes
    assert overview["robots"] == robots
    assert overview["objects"] == objects + end_effectors
    assert overview["policyAssets"] == policies
    assert overview["craftConfig"] == craft
    assert overview["metrics"] == metrics
    assert overview["physicsProxies"] == proxies
    assert overview["taskTemplates"] == templates
    assert overview["modelTypes"] == model_types


def test_model_types_unaffected(client: TestClient):
    response = client.get("/api/workspace/model-types")
    assert response.status_code == 200
    ids = {item["modelTypeId"] for item in response.json()["modelTypes"]}
    assert "robomimic-bc" in ids
    assert "act" in ids
