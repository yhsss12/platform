from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.benchmark_adapters.registry import (
    get_benchmark_adapter,
    list_benchmark_adapters,
    resolve_benchmark_adapter_for_eval_job,
)
from app.services.evaluation.cable_threading_adapter import CableThreadingEvaluationAdapter
from app.services.evaluation.dual_arm_cable_adapter import DualArmCableEvaluationAdapter
from app.services.evaluation import evaluation_service as eval_svc
from app.services.evaluation.registry import get_evaluation_adapter, get_supported_modes, list_registered_task_types
from app.services import cable_threading_service as ct_svc
from app.services import dual_arm_cable_service as dac_svc


class _FakeProc:
    def poll(self):
        return None


def test_benchmark_registry_lists_adapters():
    adapters = list_benchmark_adapters()
    template_ids = {a.task_template_id for a in adapters}
    assert template_ids == {
        "cable_threading_single_arm",
        "dual_arm_cable_manipulation",
        "isaac_block_stacking",
        "nut_assembly_single_arm",
    }


def test_benchmark_registry_resolves_cable_threading_single_arm():
    adapter = get_benchmark_adapter("cable_threading_single_arm")
    assert adapter.task_type == "cable_threading"
    caps = adapter.get_capabilities()
    assert caps.supported_evaluation_modes == [
        "expert_policy_evaluation",
        "trained_model_evaluation",
    ]


def test_benchmark_registry_resolves_dual_arm_cable_manipulation():
    adapter = get_benchmark_adapter("dual_arm_cable_manipulation")
    assert adapter.task_type == "dual_arm_cable_manipulation"
    caps = adapter.get_capabilities()
    assert caps.supported_evaluation_modes == ["episode_stability", "trained_model_evaluation"]
    assert caps.supports_train_model_evaluation is True


def test_benchmark_registry_unknown_adapter():
    with pytest.raises(HTTPException) as exc:
        get_benchmark_adapter("unknown_task")
    assert exc.value.status_code == 400


def test_resolve_benchmark_adapter_for_ct_eval_job():
    adapter = resolve_benchmark_adapter_for_eval_job("ct_eval_20260611_120000_abcd")
    assert adapter is not None
    assert adapter.task_template_id == "cable_threading_single_arm"


def test_registry_lists_both_task_types():
    types = list_registered_task_types()
    assert "cable_threading" in types
    assert "dual_arm_cable_manipulation" in types


def test_cable_threading_supported_modes():
    assert get_supported_modes("cable_threading") == ["policy_evaluation"]


def test_dual_arm_supported_modes():
    assert get_supported_modes("dual_arm_cable_manipulation") == [
        "episode_stability",
        "trained_model_evaluation",
    ]


def test_dual_arm_rejects_policy_evaluation_mode():
    adapter = get_evaluation_adapter("dual_arm_cable_manipulation")
    req = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="policy_evaluation",
        numEpisodes=1,
        policyType="robomimic",
        checkpointId="/tmp/fake.pt",
    )
    with pytest.raises(HTTPException) as exc:
        adapter.validate_request(req)
    assert exc.value.status_code == 400


def test_dual_arm_rejects_checkpoint_on_episode_stability():
    adapter = get_evaluation_adapter("dual_arm_cable_manipulation")
    req = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="episode_stability",
        numEpisodes=1,
        seeds=[42],
        checkpointId="/tmp/fake.pt",
    )
    with pytest.raises(HTTPException) as exc:
        adapter.validate_request(req)
    assert exc.value.status_code == 400
    assert "checkpoint" in str(exc.value.detail).lower() or "policy" in str(exc.value.detail).lower()


def test_dual_arm_episode_stability_requires_seeds():
    adapter = get_evaluation_adapter("dual_arm_cable_manipulation")
    req = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="episode_stability",
        numEpisodes=3,
        seed=None,
        seeds=None,
    )
    with pytest.raises(HTTPException) as exc:
        adapter.validate_request(req)
    assert exc.value.status_code == 400


def test_dual_arm_start_async_via_unified_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.services.evaluation.job_paths.EVAL_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(dac_svc, "PYTHON_BIN", tmp_path / "python")
    monkeypatch.setattr(dac_svc, "PLATFORM_RUNNER", tmp_path / "platform_runner.py")
    monkeypatch.setattr(dac_svc, "SCENE_XML", tmp_path / "scene.xml")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "platform_runner.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "scene.xml").write_text("<mujoco/>", encoding="utf-8")

    with patch(
        "app.services.evaluation.dual_arm_cable_adapter.spawn_evaluation_worker",
        return_value=None,
    ):
        result = eval_svc.start_evaluate_async(
            EvaluateAsyncRequest(
                taskTemplateId="dual_arm_cable_manipulation",
                evaluationMode="episode_stability",
                numEpisodes=2,
                seeds=[42, 43],
            )
        )
    assert result["evalJobId"].startswith("eval_")
    assert result["taskTemplateId"] == "dual_arm_cable_manipulation"
    assert result["evaluationMode"] == "episode_stability"
    assert (tmp_path / "jobs" / result["evalJobId"]).is_dir()


def test_unknown_task_type():
    with pytest.raises(HTTPException) as exc:
        get_evaluation_adapter("unknown_task")
    assert exc.value.status_code == 400


def test_cable_threading_unified_start_returns_ct_eval_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.services.evaluation.job_paths.EVAL_OUTPUT_ROOT", tmp_path / "evaluations")
    monkeypatch.setattr(ct_svc, "OUTPUT_ROOT", tmp_path / "cable_threading")
    monkeypatch.setattr(ct_svc, "WORKING_DIR", tmp_path / "CableThreadingMVP")
    monkeypatch.setattr(ct_svc, "RUN_PY", tmp_path / "CableThreadingMVP" / "run.py")
    monkeypatch.setattr(ct_svc, "PYTHON_BIN", tmp_path / "python")
    (tmp_path / "CableThreadingMVP").mkdir(parents=True)
    (tmp_path / "CableThreadingMVP" / "run.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "python").chmod(0o755)

    with patch.object(ct_svc.subprocess, "Popen", lambda *a, **k: _FakeProc()):
        result = eval_svc.start_evaluate_async(
            EvaluateAsyncRequest(
                taskTemplateId="cable_threading_single_arm",
                evaluationMode="expert_policy_evaluation",
                numEpisodes=1,
                seed=0,
            )
        )

    eval_job_id = result["evalJobId"]
    assert eval_job_id.startswith("ct_eval_")
    assert result["taskType"] == "cable_threading"
    assert result["taskTemplateId"] == "cable_threading_single_arm"
    assert result["evaluationMode"] == "expert_policy_evaluation"
    assert result["runtimePath"]
    job_root = tmp_path / "cable_threading" / "jobs" / eval_job_id
    assert job_root.is_dir()
    assert (job_root / "metadata" / "evaluation_context.json").is_file()


def test_legacy_cable_wrapper_adapter_get_result_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """历史 eval_* 包装 job（evaluation.cable_threading_adapter）结果结构。"""
    monkeypatch.setattr(ct_svc, "OUTPUT_ROOT", tmp_path)
    adapter = CableThreadingEvaluationAdapter()
    eval_job_id = "eval_20260611_120000_abcd"
    job_root = tmp_path / "evaluations" / "jobs" / eval_job_id
    ct_job_id = "ct_eval_20260611_120000_abcd"
    ct_root = tmp_path / "jobs" / ct_job_id
    (job_root / "metadata").mkdir(parents=True)
    (job_root / "metadata" / "source_jobs.json").write_text(
        json.dumps({"cable_threading": {"evalJobId": ct_job_id, "jobRoot": str(ct_root)}}),
        encoding="utf-8",
    )
    (job_root / "metadata" / "evaluation_request.json").write_text(
        json.dumps({"evaluationMode": "policy_evaluation"}),
        encoding="utf-8",
    )
    (ct_root / "results").mkdir(parents=True)
    (ct_root / "results" / "eval.results.json").write_text(
        json.dumps(
            {
                "num_episodes": 1,
                "success_rate": 1.0,
                "ever_success_rate": 1.0,
                "episodes": [],
                "aggregate": {"total_episodes": 1, "success_episodes": 1},
            }
        ),
        encoding="utf-8",
    )
    (ct_root / "live").mkdir(parents=True)
    (ct_root / "live" / "status.json").write_text(json.dumps({"status": "completed", "episodes": 1}), encoding="utf-8")
    (ct_root / "logs").mkdir(parents=True)
    (ct_root / "logs" / "run.log").write_text("", encoding="utf-8")

    result = adapter.get_result(eval_job_id, job_root)
    assert result["evalJobId"] == eval_job_id
    assert result["taskType"] == "cable_threading"
    assert result["successRate"] == 1.0
    assert "aggregate" in result
    assert "artifacts" in result
