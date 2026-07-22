from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.deps import get_current_user
from app.main import app


def _fake_user() -> SimpleNamespace:
    return SimpleNamespace(id=1, username="test")


def test_list_evaluation_jobs_api_includes_success_stats() -> None:
    app.dependency_overrides[get_current_user] = _fake_user
    sample_rows = [
        {
            "evalJobId": "eval_20260626_103509_1b68",
            "status": "completed",
            "taskName": "线缆整理评测",
            "successStats": {
                "successEpisodes": 3,
                "totalEpisodes": 3,
                "display": "3/3",
                "available": True,
                "source": "per_episode_results.json",
            },
        }
    ]
    try:
        with patch(
            "app.services.evaluation.evaluation_service.list_evaluation_jobs",
            return_value=sample_rows,
        ):
            client = TestClient(app)
            response = client.get("/api/workspace/evaluation/jobs?limit=10&offset=0")
        assert response.status_code == 200, response.text
        payload = response.json()
        jobs = payload.get("jobs") or []
        assert jobs, payload
        assert jobs[0].get("successStats") is not None
        assert jobs[0]["successStats"]["display"] == "3/3"
        assert jobs[0]["successStats"]["available"] is True
    finally:
        app.dependency_overrides.pop(get_current_user, None)
