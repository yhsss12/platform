"""Workspace 基础 API 契约：无数据时也必须 200，不能 404。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.router import api_router
from app.core.deps import get_current_user


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router, prefix="/api")

    async def _fake_user():
        return SimpleNamespace(id="contract-test-user", role="SUPER_ADMIN", is_active=True)

    app.dependency_overrides[get_current_user] = _fake_user
    return TestClient(app)


def test_dataset_index_returns_200(client: TestClient) -> None:
    resp = client.get("/api/workspace/datasets?limit=1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "datasets" in body
    assert isinstance(body["datasets"], list)


def test_workspace_jobs_list_returns_200(client: TestClient) -> None:
    resp = client.get("/api/workspace/jobs?limit=1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "jobs" in body
    assert isinstance(body["jobs"], list)


def test_evaluation_jobs_list_returns_200(client: TestClient) -> None:
    resp = client.get("/api/workspace/evaluation/jobs?limit=1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "jobs" in body
    assert isinstance(body["jobs"], list)


def test_resources_overview_returns_200(client: TestClient) -> None:
    resp = client.get("/api/workspace/resources/overview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "resourcesByType" in body or "taskTemplates" in body


def test_model_assets_list_returns_200(client: TestClient) -> None:
    resp = client.get("/api/workspace/model-assets?limit=1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "assets" in body
    assert isinstance(body["assets"], list)
    assert "total" in body
