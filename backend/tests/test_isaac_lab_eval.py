from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services.evaluation.evaluation_request_resolver import normalize_evaluate_request
from app.services.isaac_lab import eval_service as isaac_eval
from app.schemas.evaluation import EvaluateAsyncRequest


@pytest.fixture
def isaac_model_manifest(tmp_path: Path) -> tuple[dict, dict, Path]:
    ckpt = tmp_path / "model_final.pth"
    ckpt.write_bytes(b"checkpoint")
    manifest = {
        "modelAssetId": "model_isaac_test",
        "taskType": "isaac_block_stacking",
        "taskTemplateId": "isaac_block_stacking",
        "backendType": "isaac_robomimic_bc",
        "framework": "Isaac Robomimic BC",
        "actionDim": 7,
        "taskEnv": "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "checkpointPath": str(ckpt),
    }
    asset = {
        "id": "model_isaac_test",
        "taskTemplateId": "isaac_block_stacking",
        "framework": "Isaac Robomimic BC",
        "checkpointPath": str(ckpt),
        "manifestPath": str(tmp_path / "model_manifest.json"),
    }
    (tmp_path / "model_manifest.json").write_text('{"backendType":"isaac_robomimic_bc"}', encoding="utf-8")
    return asset, manifest, ckpt


def test_block_stacking_rejects_expert_policy():
    request = EvaluateAsyncRequest(
        taskTemplateId="isaac_block_stacking",
        evaluationMode="expert_policy_evaluation",
        numEpisodes=1,
    )
    with pytest.raises(HTTPException) as exc:
        normalize_evaluate_request(request)
    assert exc.value.status_code == 400
    assert "trained_model_evaluation" in str(exc.value.detail)


def test_block_stacking_accepts_trained_model_evaluation(isaac_model_manifest):
    asset, manifest, ckpt = isaac_model_manifest
    with patch.object(isaac_eval, "get_model_asset_by_id", return_value=asset):
        with patch.object(isaac_eval, "_load_model_manifest", return_value=manifest):
            validated_asset, validated_manifest, path = isaac_eval.validate_isaac_robomimic_model_asset(
                "model_isaac_test"
            )
    assert path == ckpt.resolve()
    assert validated_asset["id"] == "model_isaac_test"


def test_validate_rejects_act_model_for_block_stacking(tmp_path: Path):
    ckpt = tmp_path / "act.pt"
    ckpt.write_bytes(b"x")
    asset = {
        "id": "model_act",
        "taskTemplateId": "isaac_block_stacking",
        "framework": "ACT",
        "checkpointPath": str(ckpt),
        "manifestPath": "",
    }
    manifest = {
        "backendType": "act",
        "framework": "ACT",
        "taskType": "isaac_block_stacking",
        "taskEnv": "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "actionDim": 7,
        "checkpointPath": str(ckpt),
    }
    with patch.object(isaac_eval, "get_model_asset_by_id", return_value=asset):
        with patch.object(isaac_eval, "_load_model_manifest", return_value=manifest):
            with pytest.raises(HTTPException) as exc:
                isaac_eval.validate_isaac_robomimic_model_asset("model_act")
    assert exc.value.status_code == 400
    detail = exc.value.detail
    assert isinstance(detail, dict)
    assert detail.get("code") == "MODEL_ASSET_BACKEND_TYPE_INCOMPATIBLE"
    assert "act" in str(detail.get("message")).lower()
    assert detail.get("expectedBackendTypes") == ["isaac_robomimic_bc"]


def test_validate_rejects_mujoco_robomimic_model(tmp_path: Path):
    ckpt = tmp_path / "mujoco.pt"
    ckpt.write_bytes(b"x")
    asset = {
        "id": "model_mujoco",
        "taskTemplateId": "task_cable_threading_v1",
        "framework": "robomimic_bc",
        "checkpointPath": str(ckpt),
        "manifestPath": "",
    }
    manifest = {
        "backendType": "robomimic_bc",
        "taskType": "cable_threading",
        "taskEnv": "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "actionDim": 7,
        "checkpointPath": str(ckpt),
    }
    with patch.object(isaac_eval, "get_model_asset_by_id", return_value=asset):
        with patch.object(isaac_eval, "_load_model_manifest", return_value=manifest):
            with pytest.raises(HTTPException) as exc:
                isaac_eval.validate_isaac_robomimic_model_asset("model_mujoco")
    assert exc.value.status_code == 400


def test_isaaclab_franka_stack_cube_routes_to_block_stacking_adapter(isaac_model_manifest):
    asset, _manifest, _ckpt = isaac_model_manifest
    request = EvaluateAsyncRequest(
        taskTemplateId="isaaclab_franka_stack_cube",
        evaluationMode="trained_model_evaluation",
        numEpisodes=1,
        modelAssetId="model_isaac_test",
    )
    with patch("app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id", return_value=asset):
        normalized = normalize_evaluate_request(request)
    assert normalized.task_type == "block_stacking"
    assert normalized.task_template_id == "isaac_block_stacking"


def test_execute_marks_failed_without_results(tmp_path: Path, isaac_model_manifest, monkeypatch):
    asset, manifest, ckpt = isaac_model_manifest
    job_root = tmp_path / "isaac_eval_test"
    for sub in ("logs", "results", "videos", "metadata", "artifacts"):
        (job_root / sub).mkdir(parents=True)

    runner = MagicMock()
    runner.root = tmp_path / "IsaacLab"
    runner.root.mkdir()
    runner.is_ready.return_value = True
    runner.build_command.return_value = ["isaaclab.sh"]
    runner.run_to_files.return_value = MagicMock(returncode=0, timed_out=False)

    monkeypatch.setattr(isaac_eval, "IsaacLabCliRunner", MagicMock(from_settings=lambda: runner))
    monkeypatch.setattr(isaac_eval, "probe_isaac_eval_capability", lambda: {"ready": True, "issues": []})
    monkeypatch.setattr(isaac_eval, "_sync_platform_script", lambda _runner: runner.root / "script.py")
    monkeypatch.setattr(isaac_eval, "sync_workspace_job_from_runtime", lambda _job_id: None)

    request = EvaluateAsyncRequest(
        taskTemplateId="isaac_block_stacking",
        evaluationMode="trained_model_evaluation",
        numEpisodes=1,
        seed=0,
        headless=True,
        modelAssetId="model_isaac_test",
    )
    with patch("app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id", return_value=asset):
        with patch.object(isaac_eval, "get_model_asset_by_id", return_value=asset):
            normalized = normalize_evaluate_request(request)

            isaac_eval.execute_isaac_evaluation(
                "isaac_eval_test_001",
                job_root,
                request,
                normalized,
                asset=asset,
                model_manifest=manifest,
                checkpoint=ckpt,
            )

    status = isaac_eval._read_json(job_root / "status.json")
    assert status["status"] == "failed"
    assert "aggregate_result" in status["message"]
