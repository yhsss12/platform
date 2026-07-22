"""GET /api/workspace/evaluation/jobs 列表回归：workspace_jobs DB-first，不依赖 evaluation_jobs 表。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_workspace_evaluation
from app.core.deps import get_current_user
from app.schemas.evaluation import EvaluationJobListItem, EvaluationJobListResponse


@pytest.fixture()
def eval_list_client():
    app = FastAPI()
    app.include_router(routes_workspace_evaluation.router, prefix="/api/workspace/evaluation")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, username="test")
    return TestClient(app)


def test_list_evaluation_jobs_api_uses_workspace_jobs_shape(eval_list_client: TestClient) -> None:
    sample_rows = [
        {
            "workspaceJobId": 42,
            "evalJobId": "ct_eval_20260630_133008_3c5f",
            "jobId": "ct_eval_20260630_133008_3c5f",
            "taskType": "cable_threading",
            "evaluationMode": "trained_model_evaluation",
            "status": "completed",
            "taskName": "pi0 Platform Eval Smoke",
            "metrics": {
                "modelType": "pi0",
                "evalExecutor": "joint_position",
                "successRate": 0.0,
                "success_rate": 0.0,
                "modelAssetId": "model__123947_ebd2_final",
            },
            "successStats": {
                "successEpisodes": 0,
                "totalEpisodes": 1,
                "display": "0/1",
                "available": True,
                "source": "per_episode_results.json",
            },
        }
    ]
    with patch(
        "app.api.routes_workspace_evaluation.svc.list_evaluation_jobs",
        return_value=sample_rows,
    ) as list_mock:
        response = eval_list_client.get("/api/workspace/evaluation/jobs?limit=10&offset=0")

    list_mock.assert_called_once_with(sync_stale=False)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    jobs = payload.get("jobs") or []
    assert len(jobs) == 1
    job = jobs[0]
    assert job["evalJobId"] == "ct_eval_20260630_133008_3c5f"
    assert job["status"] == "completed"
    assert job["metrics"]["successRate"] == 0.0
    assert job["successStats"]["display"] == "0/1"


def test_list_evaluation_jobs_pydantic_accepts_zero_success_rate() -> None:
    item = EvaluationJobListItem(
        evalJobId="ct_eval_20260630_133008_3c5f",
        status="completed",
        metrics={"successRate": 0.0, "modelType": "pi0"},
        successStats={
            "successEpisodes": 0,
            "totalEpisodes": 1,
            "display": "0/1",
            "available": True,
        },
    )
    response = EvaluationJobListResponse(jobs=[item], total=1)
    dumped = response.model_dump()
    assert dumped["jobs"][0]["metrics"]["successRate"] == 0.0


def test_list_evaluation_jobs_service_db_first(monkeypatch) -> None:
    """list_evaluation_jobs 应优先读 workspace_jobs，sync_stale 可关闭。"""
    from app.services.evaluation import evaluation_service as eval_svc
    from app.services import eval_job_db_service

    captured: dict[str, bool] = {}

    def _fake_list(*, sync_stale: bool = True):
        captured["sync_stale"] = sync_stale
        return [{"evalJobId": "eval_sample", "status": "completed"}]

    monkeypatch.setattr(eval_job_db_service, "list_evaluation_jobs_from_db", _fake_list)
    rows = eval_svc.list_evaluation_jobs(sync_stale=False)
    assert captured["sync_stale"] is False
    assert rows and rows[0]["evalJobId"] == "eval_sample"
