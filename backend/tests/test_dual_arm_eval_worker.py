from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation import evaluation_service as eval_svc
from app.services.evaluation import job_paths as eval_job_paths
from app.services.evaluation.dual_arm_cable_adapter import DualArmCableEvaluationAdapter
from app.services.evaluation.dual_arm_cable_eval_worker import (
    aggregate_episode_records,
    episode_record_from_result,
    resolve_eval_params,
    run_evaluation_worker,
)
from app.services.evaluation.registry import get_evaluation_adapter
from app.services import dual_arm_cable_service as dac_svc


def _dual_arm_request(**overrides) -> EvaluateAsyncRequest:
    base = {
        "taskType": "dual_arm_cable_manipulation",
        "evaluationMode": "episode_stability",
        "numEpisodes": 1,
        "seeds": [42],
        "maxCables": 1,
        "record": True,
        "headless": True,
        "dualArmCable": {
            "stretchMode": "fixed_distance",
            "releaseMode": "three_phase",
        },
    }
    base.update(overrides)
    return EvaluateAsyncRequest(**base)


def _mock_episode_success(episode_dir: Path) -> None:
    result = {
        "episode_success": True,
        "num_cables_succeeded": 1,
        "max_cables": 1,
        "steps": [
            {
                "result": {
                    "left_contact": True,
                    "right_contact": True,
                    "stretch_reached": True,
                    "sag_m": 0.0316,
                    "span_m": 0.4496,
                    "final_sag_m": 0.0013,
                    "final_span_m": 0.3802,
                }
            }
        ],
    }
    results_dir = episode_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "episode_result.json").write_text(json.dumps(result), encoding="utf-8")
    videos_dir = episode_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    (videos_dir / "generate.mp4").write_bytes(b"fake-mp4")


def test_dual_arm_episode_stability_validate_success():
    adapter = get_evaluation_adapter("dual_arm_cable_manipulation")
    adapter.validate_request(_dual_arm_request())


def test_dual_arm_validate_auto_seeds_from_seed():
    adapter = DualArmCableEvaluationAdapter()
    req = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="episode_stability",
        numEpisodes=3,
        seed=42,
    )
    adapter.validate_request(req)
    assert resolve_eval_params(req)["seeds"] == [42, 43, 44]


def test_dual_arm_rejects_checkpoint_on_episode_stability():
    adapter = get_evaluation_adapter("dual_arm_cable_manipulation")
    req = _dual_arm_request(checkpointId="/tmp/fake.pt")
    with pytest.raises(HTTPException) as exc:
        adapter.validate_request(req)
    assert exc.value.status_code == 400


def test_dual_arm_trained_model_validate_success():
    adapter = get_evaluation_adapter("dual_arm_cable_manipulation")
    ckpt = "/home/ubuntu/project/eai-idev2.1/runs/training/jobs/train_20260614_221420_e773/checkpoints/model_final.pt"
    if not Path(ckpt).is_file():
        pytest.skip("acceptance checkpoint not available")
    req = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="trained_model_evaluation",
        numEpisodes=1,
        seed=0,
        modelAssetId="model_20260614_221500_82da",
        checkpointPath=ckpt,
        dualArmCable={"checkpointPath": ckpt, "modelAssetId": "model_20260614_221500_82da"},
    )
    adapter.validate_request(req)


def test_dual_arm_trained_model_requires_checkpoint():
    adapter = get_evaluation_adapter("dual_arm_cable_manipulation")
    req = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="trained_model_evaluation",
        numEpisodes=1,
        seed=0,
    )
    with pytest.raises(HTTPException) as exc:
        adapter.validate_request(req)
    assert exc.value.status_code == 400


def test_dual_arm_accepts_torch_backend_with_display_framework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "model_final.pt"
    checkpoint.write_bytes(b"checkpoint")
    asset = {
        "id": "model__150020_d08d_final",
        "backendType": "torch_bc",
        "trainingBackend": "torch_bc",
        "framework": "BC (PyTorch)",
        "taskTemplateId": "dual_arm_cable_manipulation",
        "checkpointPath": str(checkpoint),
    }
    monkeypatch.setattr(
        "app.services.evaluation.dual_arm_cable_adapter.get_model_asset_by_id",
        lambda _model_asset_id: asset,
    )
    request = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="trained_model_evaluation",
        numEpisodes=1,
        seed=42,
        modelAssetId=asset["id"],
        checkpointPath=str(checkpoint),
        dualArmCable={"modelAssetId": asset["id"], "checkpointPath": str(checkpoint)},
    )

    DualArmCableEvaluationAdapter().validate_request(request)


def test_worker_policy_rollout_mock_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dac_svc, "PYTHON_BIN", tmp_path / "python")
    monkeypatch.setattr(dac_svc, "EVAL_POLICY_ROLLOUT", tmp_path / "eval_policy_rollout.py")
    monkeypatch.setattr(dac_svc, "SCENE_XML", tmp_path / "scene.xml")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "eval_policy_rollout.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "scene.xml").write_text("<mujoco/>", encoding="utf-8")

    eval_job_id = "eval_20260614_230000_abcd"
    job_root = tmp_path / eval_job_id
    for sub in ("logs", "results", "videos", "metadata", "episodes"):
        (job_root / sub).mkdir(parents=True)

    ckpt = tmp_path / "model_final.pt"
    ckpt.write_bytes(b"stub")
    request = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="trained_model_evaluation",
        numEpisodes=1,
        seed=0,
        modelAssetId="model_test",
        checkpointPath=str(ckpt),
        dualArmCable={"checkpointPath": str(ckpt), "modelAssetId": "model_test"},
    )

    def mock_runner(**kwargs):
        episode_dir = kwargs["episode_dir"]
        result = {
            "episode_success": False,
            "steps_executed": 120,
            "mean_reward": 0.0,
            "total_reward": 0.0,
            "backend_type": "torch_bc",
            "policyMode": "torch_bc_policy",
        }
        results_dir = episode_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "episode_result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"returnCode": 0, "runtimeSec": 30.0, "episodeResult": result}

    run_evaluation_worker(eval_job_id, job_root, request, episode_runner=mock_runner)

    aggregate = json.loads((job_root / "results" / "aggregate_result.json").read_text())
    assert aggregate["evaluationMode"] == "trained_model_evaluation"
    assert aggregate["backendType"] == "torch_bc"
    assert aggregate["episodeCount"] == 1
    assert aggregate["successRate"] == 0.0
    assert aggregate["failureCount"] == 1
    assert aggregate["meanEpisodeLength"] == 120.0

    status = json.loads((job_root / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["evaluationMode"] == "trained_model_evaluation"


def test_dual_arm_start_async_creates_eval_job_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(eval_svc, "EVAL_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(eval_job_paths, "EVAL_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(dac_svc, "PYTHON_BIN", tmp_path / "python")
    monkeypatch.setattr(dac_svc, "PLATFORM_RUNNER", tmp_path / "platform_runner.py")
    monkeypatch.setattr(dac_svc, "SCENE_XML", tmp_path / "scene.xml")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "platform_runner.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "scene.xml").write_text("<mujoco/>", encoding="utf-8")

    def _noop_spawn(eval_job_id: str, job_root: Path, request: EvaluateAsyncRequest) -> None:
        return None

    with patch(
        "app.services.evaluation.dual_arm_cable_adapter.spawn_evaluation_worker",
        side_effect=_noop_spawn,
    ):
        result = eval_svc.start_evaluate_async(_dual_arm_request())

    eval_job_id = result["evalJobId"]
    assert eval_job_id.startswith("eval_")
    job_root = tmp_path / "jobs" / eval_job_id
    assert job_root.is_dir()
    assert (job_root / "metadata" / "evaluation_request.json").is_file()
    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "queued"
    assert status["evaluationMode"] == "episode_stability"


def test_worker_mock_episode_generates_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dac_svc, "PYTHON_BIN", tmp_path / "python")
    monkeypatch.setattr(dac_svc, "PLATFORM_RUNNER", tmp_path / "platform_runner.py")
    monkeypatch.setattr(dac_svc, "SCENE_XML", tmp_path / "scene.xml")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "platform_runner.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "scene.xml").write_text("<mujoco/>", encoding="utf-8")

    eval_job_id = "eval_20260611_120000_abcd"
    job_root = tmp_path / eval_job_id
    for sub in ("logs", "results", "videos", "metadata", "episodes"):
        (job_root / sub).mkdir(parents=True)
    request = _dual_arm_request()

    def mock_runner(**kwargs):
        episode_dir = kwargs["episode_dir"]
        _mock_episode_success(episode_dir)
        return {
            "returnCode": 0,
            "failureReason": None,
            "runtimeSec": 600,
            "episodeResult": json.loads((episode_dir / "results" / "episode_result.json").read_text()),
        }

    run_evaluation_worker(eval_job_id, job_root, request, episode_runner=mock_runner)

    per_path = job_root / "results" / "per_episode_results.json"
    agg_path = job_root / "results" / "aggregate_result.json"
    assert per_path.is_file()
    assert agg_path.is_file()

    per_data = json.loads(per_path.read_text(encoding="utf-8"))
    assert len(per_data["episodes"]) == 1
    assert per_data["episodes"][0]["episodeSuccess"] is True
    assert per_data["episodes"][0]["seed"] == 42

    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))
    assert aggregate["summary"]["successRate"] == 1.0
    assert aggregate["taskMetrics"]["meanFinalSag"] == 0.0013
    assert aggregate["taskMetrics"]["meanFinalSpan"] == 0.3802
    assert (job_root / "videos" / "episode_00.mp4").is_file()

    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["metrics"]["successRate"] == 1.0


def test_worker_single_episode_failure_still_aggregates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dac_svc, "PYTHON_BIN", tmp_path / "python")
    monkeypatch.setattr(dac_svc, "PLATFORM_RUNNER", tmp_path / "platform_runner.py")
    monkeypatch.setattr(dac_svc, "SCENE_XML", tmp_path / "scene.xml")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "platform_runner.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "scene.xml").write_text("<mujoco/>", encoding="utf-8")

    eval_job_id = "eval_20260611_120001_abcd"
    job_root = tmp_path / eval_job_id
    for sub in ("logs", "results", "videos", "metadata", "episodes"):
        (job_root / sub).mkdir(parents=True)
    request = _dual_arm_request(numEpisodes=2, seeds=[42, 43])

    call_count = {"n": 0}

    def mock_runner(**kwargs):
        call_count["n"] += 1
        if kwargs["seed"] == 42:
            _mock_episode_success(kwargs["episode_dir"])
            return {"returnCode": 0, "runtimeSec": 100}
        return {"returnCode": 1, "failureReason": "sim crash", "runtimeSec": 50}

    run_evaluation_worker(eval_job_id, job_root, request, episode_runner=mock_runner)
    assert call_count["n"] == 2

    aggregate = json.loads((job_root / "results" / "aggregate_result.json").read_text(encoding="utf-8"))
    assert aggregate["summary"]["totalEpisodes"] == 2
    assert aggregate["summary"]["successEpisodes"] == 1
    assert aggregate["summary"]["successRate"] == 0.5
    assert 43 in aggregate["taskMetrics"]["failureSeeds"]

    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"


def test_parse_episode_result_fallback_episode_dir(tmp_path: Path):
    episode_dir = tmp_path / "episode_00"
    payload = {"episode_success": True, "num_cables_succeeded": 1}
    (episode_dir / "episode").mkdir(parents=True)
    (episode_dir / "episode" / "episode_result.json").write_text(json.dumps(payload), encoding="utf-8")
    parsed = __import__(
        "app.services.evaluation.dual_arm_cable_eval_worker",
        fromlist=["parse_episode_result"],
    ).parse_episode_result(episode_dir)
    assert parsed.get("episode_success") is True


def test_finalize_partial_episode_stability_marks_failed(tmp_path: Path):
    from app.services.evaluation.dual_arm_cable_eval_worker import (
        _finalize_episode_stability_job,
        episode_record_from_result,
    )

    eval_job_id = "eval_20260624_partial_abcd"
    job_root = tmp_path / eval_job_id
    (job_root / "logs").mkdir(parents=True)
    per_episode = [
        episode_record_from_result(
            eval_job_id=eval_job_id,
            episode_index=0,
            seed=42,
            episode_dir=job_root / "episodes" / "episode_00",
            episode_status="completed",
            episode_result={"episode_success": True, "num_cables_succeeded": 1, "max_cables": 1, "steps": []},
            return_code=0,
            runtime_sec=100.0,
            failure_reason=None,
        )
    ]
    _finalize_episode_stability_job(
        job_root,
        eval_job_id,
        _dual_arm_request(numEpisodes=5, seeds=[42, 43, 44, 45, 46]),
        per_episode=per_episode,
        total=5,
        started_at="2026-06-24T00:00:00+00:00",
        aborted=True,
        abort_message="worker interrupted",
    )
    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["completedEpisodes"] == 1
    assert (job_root / "results" / "aggregate_result.json").is_file()


def test_worker_environment_error_marks_job_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dac_svc, "PYTHON_BIN", tmp_path / "missing-python")
    monkeypatch.setattr(dac_svc, "PLATFORM_RUNNER", tmp_path / "platform_runner.py")
    monkeypatch.setattr(dac_svc, "SCENE_XML", tmp_path / "scene.xml")

    eval_job_id = "eval_20260611_120002_abcd"
    job_root = tmp_path / eval_job_id
    (job_root / "logs").mkdir(parents=True)

    run_evaluation_worker(eval_job_id, job_root, _dual_arm_request())
    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "Python interpreter not found" in status["message"]


def test_adapter_get_video_episode_zero(tmp_path: Path):
    adapter = DualArmCableEvaluationAdapter()
    eval_job_id = "eval_20260611_120003_abcd"
    job_root = tmp_path / eval_job_id
    (job_root / "videos").mkdir(parents=True)
    video = job_root / "videos" / "episode_00.mp4"
    video.write_bytes(b"mp4")

    path = adapter.get_video_path(eval_job_id, job_root, episode=0)
    assert path == video

    path_default = adapter.get_video_path(eval_job_id, job_root, episode=None)
    assert path_default == video


def test_aggregate_episode_records_structure():
    episodes = [
        episode_record_from_result(
            eval_job_id="eval_x",
            episode_index=0,
            seed=42,
            episode_dir=Path("/tmp/ep"),
            episode_status="completed",
            episode_result={
                "episode_success": True,
                "num_cables_succeeded": 1,
                "max_cables": 1,
                "steps": [{"result": {"left_contact": True, "right_contact": True, "stretch_reached": True, "final_sag_m": 0.01, "final_span_m": 0.02}}],
            },
            return_code=0,
            runtime_sec=600,
            failure_reason=None,
        )
    ]
    agg = aggregate_episode_records("eval_x", episodes)
    assert agg["summary"]["successEpisodes"] == 1
    assert agg["taskMetrics"]["contactSuccessRate"] == 1.0


def test_unified_status_and_result_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(eval_svc, "EVAL_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(eval_job_paths, "EVAL_OUTPUT_ROOT", tmp_path)
    eval_job_id = "eval_20260611_120004_abcd"
    job_root = tmp_path / "jobs" / eval_job_id
    (job_root / "metadata").mkdir(parents=True)
    (job_root / "metadata" / "evaluation_request.json").write_text(
        json.dumps({"taskType": "dual_arm_cable_manipulation"}),
        encoding="utf-8",
    )
    (job_root / "status.json").write_text(
        json.dumps(
            {
                "evalJobId": eval_job_id,
                "taskType": "dual_arm_cable_manipulation",
                "evaluationMode": "episode_stability",
                "status": "completed",
                "message": "done",
                "metrics": {"successRate": 1.0},
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    (job_root / "results").mkdir(parents=True)
    (job_root / "results" / "aggregate_result.json").write_text(
        json.dumps({"summary": {"successRate": 1.0}}),
        encoding="utf-8",
    )
    (job_root / "logs").mkdir(parents=True)
    (job_root / "logs" / "eval.log").write_text("line1\nline2\n", encoding="utf-8")

    status = eval_svc.get_evaluation_status(eval_job_id)
    assert status["status"] == "completed"
    assert status["metrics"]["successRate"] == 1.0

    result = eval_svc.get_evaluation_result(eval_job_id)
    assert result["summary"]["successRate"] == 1.0

    log = eval_svc.read_evaluation_log_tail(eval_job_id)
    assert "line2" in log
