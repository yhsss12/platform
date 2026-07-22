from __future__ import annotations

from app.schemas.evaluation import EvaluationJobListItem, EvaluationJobListResponse


def test_evaluation_job_list_item_includes_success_stats() -> None:
    row = {
        "evalJobId": "eval_20260626_103509_1b68",
        "status": "completed",
        "successStats": {
            "successEpisodes": 3,
            "totalEpisodes": 3,
            "display": "3/3",
            "available": True,
            "source": "per_episode_results.json",
        },
    }
    item = EvaluationJobListItem(**row)
    payload = item.model_dump()
    assert payload["successStats"] is not None
    assert payload["successStats"]["display"] == "3/3"
    assert payload["successStats"]["available"] is True


def test_evaluation_job_list_response_serializes_success_stats() -> None:
    response = EvaluationJobListResponse(
        jobs=[
            EvaluationJobListItem(
                evalJobId="ct_eval_sample",
                status="completed",
                successStats={
                    "successEpisodes": 8,
                    "totalEpisodes": 10,
                    "display": "8/10",
                    "available": True,
                    "source": "aggregate_result.json",
                },
            )
        ],
        total=1,
    )
    job = response.model_dump()["jobs"][0]
    assert job["successStats"]["display"] == "8/10"
