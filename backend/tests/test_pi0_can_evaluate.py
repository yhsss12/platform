"""Tests for pi0 canEvaluate gating (Phase G)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.services.policy_schema_resolver import (  # noqa: E402
    PI0_EVAL_ADAPTER_READY_MARKER,
    PI0_PLATFORM_EVAL_READY_MARKER,
    explain_pi0_model_asset_eval_blocker,
    mark_pi0_platform_eval_ready,
    pi0_platform_eval_ready,
    resolve_pi0_model_asset_eval_fields,
)


def _joint_asset() -> dict:
    return {
        "modelType": "pi0",
        "policyType": "pi0",
        "datasetFormat": "lerobot",
        "stateDim": 9,
        "actionDim": 8,
        "robot": "Panda",
        "controllerType": "JOINT_POSITION",
        "actionMode": "joint_delta_derived",
        "taskInstruction": "thread the cable through the pole",
    }


@pytest.fixture()
def adapter_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adapter_path = tmp_path / "adapter_ready.json"
    adapter_path.write_text(json.dumps({"eval_adapter_ready": True}), encoding="utf-8")
    platform_path = tmp_path / "platform_eval_ready.json"
    import app.services.policy_schema_resolver as psr

    monkeypatch.setattr(psr, "PI0_EVAL_ADAPTER_READY_MARKER", adapter_path)
    monkeypatch.setattr(psr, "PI0_PLATFORM_EVAL_READY_MARKER", platform_path)
    return adapter_path, platform_path


def test_pi0_can_evaluate_false_when_platform_not_ready(adapter_marker):
    adapter_path, platform_path = adapter_marker
    assert adapter_path.is_file()
    assert not platform_path.is_file()
    fields = resolve_pi0_model_asset_eval_fields(_joint_asset())
    assert fields["canEvaluate"] is False
    assert fields["evalDisabledReason"] == "pi0 platform evaluation not enabled"


def test_pi0_can_evaluate_true_when_platform_ready(adapter_marker):
    adapter_path, platform_path = adapter_marker
    assert adapter_path.is_file()
    mark_pi0_platform_eval_ready(eval_job_id="ct_eval_test", model_asset_id="model__test")
    assert pi0_platform_eval_ready()
    fields = resolve_pi0_model_asset_eval_fields(_joint_asset())
    assert fields["canEvaluate"] is True
    assert fields["evalDisabledReason"] is None
    assert explain_pi0_model_asset_eval_blocker(_joint_asset()) is None
