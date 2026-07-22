from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.evaluation_metric_service import (
    ISAAC_STACK_DEFAULT_METRIC_IDS,
    UNIVERSAL_SUCCESS_RATE_METRIC_ID,
    attach_isaac_eval_metric_metadata,
    compute_timeout_rate,
    resolve_metric_value,
)


def test_compute_timeout_rate_from_per_episode():
    per_episode = {
        "episodes": [
            {"failureReason": "horizon_reached"},
            {"failureReason": "terminated"},
            {"failureReason": "horizon_reached"},
        ]
    }
    assert compute_timeout_rate(per_episode) == pytest.approx(2 / 3)


def test_attach_isaac_eval_metric_metadata():
    aggregate = {
        "successRate": 0.0,
        "meanReward": 0.0,
        "meanEpisodeLength": 400,
        "failureCount": 1,
    }
    per_episode = {
        "episodes": [
            {"failureReason": "horizon_reached", "success": False, "episodeLength": 400},
        ]
    }
    enriched = attach_isaac_eval_metric_metadata(aggregate, per_episode)
    assert enriched["computedMetricIds"] == list(ISAAC_STACK_DEFAULT_METRIC_IDS)
    assert enriched["metricsSource"] == "task_evaluation_script"
    assert enriched["metrics"]["timeoutRate"] == pytest.approx(1.0)


def test_resolve_universal_success_rate_metric():
    meta = {
        "implemented": True,
        "calculationMode": "aggregate_field",
        "sourceField": "successRate",
    }
    aggregate = {"summary": {"successRate": 0.8}, "successRate": 0.75}
    assert resolve_metric_value(meta, aggregate=aggregate, per_episode=None) == 0.75
    assert UNIVERSAL_SUCCESS_RATE_METRIC_ID == "metric_success_rate_v1"


def test_resolve_isaac_success_rate_metric():
    meta = {
        "implemented": True,
        "calculationMode": "aggregate_field",
        "sourceField": "successRate",
    }
    value = resolve_metric_value(meta, aggregate={"successRate": 0.25}, per_episode=None)
    assert value == 0.25


def test_resolve_timeout_rate_metric():
    meta = {
        "implemented": True,
        "calculationMode": "per_episode_failure_reason",
        "sourceField": "failureReason",
        "failureReasonValue": "horizon_reached",
    }
    per_episode = {
        "episodes": [
            {"failureReason": "horizon_reached"},
            {"failureReason": "terminated"},
        ]
    }
    value = resolve_metric_value(meta, aggregate={}, per_episode=per_episode)
    assert value == pytest.approx(0.5)


def test_existing_isaac_eval_job_enrichment(tmp_path: Path):
    from app.services.isaac_lab import eval_service as isaac_eval

    job_root = tmp_path / "isaac_eval_test"
    results = job_root / "results"
    results.mkdir(parents=True)
    aggregate = {
        "successRate": 0.0,
        "meanReward": 0.0,
        "meanEpisodeLength": 400,
        "failureCount": 1,
    }
    per_episode = {
        "episodes": [
            {
                "episodeIndex": 0,
                "success": False,
                "reward": 0.0,
                "episodeLength": 400,
                "failureReason": "horizon_reached",
            }
        ]
    }
    (results / "aggregate_result.json").write_text(json.dumps(aggregate), encoding="utf-8")
    (results / "per_episode_results.json").write_text(json.dumps(per_episode), encoding="utf-8")

    enriched = isaac_eval._enrich_aggregate_result(
        job_root,
        model_asset_id="model_test",
        model_manifest={"taskEnv": "Isaac-Stack-Cube-Franka-IK-Rel-v0"},
        asset={"sourceDatasetId": "ds_test"},
    )
    assert enriched["metrics"]["timeoutRate"] == pytest.approx(1.0)
    assert "isaac_stack_success_rate_v1" in enriched["computedMetricIds"]
