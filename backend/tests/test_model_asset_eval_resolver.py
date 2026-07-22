"""Tests for model asset local checkpoint resolution and eval policy routing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation.evaluation_request_resolver import normalize_evaluate_request
from app.services.model_asset_checkpoint_resolver import (
    infer_trained_policy_type,
    resolve_local_checkpoint_path,
)
from app.services.model_asset_validation import enrich_model_asset, validate_model_asset


def _write_dp_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {},
            "backend": "diffusion_policy",
            "action_key": "joint_actions",
            "train_config": {
                "backend": "diffusion_policy",
                "eval_executor": "joint_position",
                "trained_action_mode": "joint_delta",
                "controller_type": "JOINT_POSITION",
                "low_dim_dim": 9,
            },
        },
        path,
    )


def test_resolve_local_checkpoint_from_minio_db_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260625_testresolver"
    job_dir = tmp_path / "runs" / "training" / "jobs" / train_job_id
    ckpt = job_dir / "checkpoints" / "model_final.pt"
    _write_dp_checkpoint(ckpt)
    manifest = job_dir / "artifacts" / "model_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '{"modelAssetId":"model__testresolver_final","checkpointPath":"'
        + str(ckpt)
        + '"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.TRAINING_JOBS_ROOT",
        tmp_path / "runs" / "training" / "jobs",
    )
    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.PROJECT_ROOT",
        tmp_path,
    )

    asset = {
        "id": "model__testresolver_final",
        "checkpointPath": "minio://eai-checkpoints/checkpoints/train_20260625_testresolver/model_final.pt",
        "sourceTrainingJobId": train_job_id,
        "modelType": "diffusion_policy",
        "trainingBackend": "diffusion_policy",
    }
    resolved = resolve_local_checkpoint_path(asset=asset)
    assert resolved == str(ckpt.resolve())


def test_enrich_model_asset_uses_local_fallback_for_minio_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260625_enrich"
    job_dir = tmp_path / "runs" / "training" / "jobs" / train_job_id
    ckpt = job_dir / "checkpoints" / "diffusion_policy" / "checkpoints" / "model_final.pt"
    _write_dp_checkpoint(ckpt)

    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.TRAINING_JOBS_ROOT",
        tmp_path / "runs" / "training" / "jobs",
    )
    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.services.model_asset_validation._training_job_exists",
        lambda _job_id: True,
    )

    enriched = enrich_model_asset(
        {
            "id": "model__enrich_final",
            "checkpointPath": "minio://bucket/model_final.pt",
            "sourceTrainingJobId": train_job_id,
            "status": "ready",
            "backendType": "diffusion_policy",
            "taskType": "cable_threading",
        }
    )
    assert enriched["fileExists"] is True
    assert enriched["checkpointPath"] == str(ckpt.resolve())
    assert enriched["status"] == "available"


def test_infer_trained_policy_type_from_dp_checkpoint_only(tmp_path: Path):
    ckpt = tmp_path / "model_final.pt"
    _write_dp_checkpoint(ckpt)
    assert infer_trained_policy_type(checkpoint_path=str(ckpt)) == "diffusion_policy"


def test_unified_eval_short_checkpoint_path_routes_to_diffusion_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260625_shortpath"
    job_dir = tmp_path / "runs" / "training" / "jobs" / train_job_id
    ckpt = job_dir / "checkpoints" / "model_final.pt"
    _write_dp_checkpoint(ckpt)

    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.TRAINING_JOBS_ROOT",
        tmp_path / "runs" / "training" / "jobs",
    )
    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.PROJECT_ROOT",
        tmp_path,
    )

    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        numEpisodes=2,
        checkpointPath=str(job_dir / "checkpoints" / "model_final.pt"),
    )
    normalized = normalize_evaluate_request(request)
    assert normalized.policy == "diffusion_policy"
    assert normalized.eval_executor == "joint_position"
    assert normalized.controller_type == "JOINT_POSITION"
    assert normalized.action_mode == "joint_delta"


def test_unified_eval_model_asset_id_resolves_local_checkpoint():
    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        numEpisodes=2,
        modelAssetId="model__165923_aa62_final",
    )
    normalized = normalize_evaluate_request(request)
    assert normalized.policy == "diffusion_policy"
    assert normalized.eval_executor == "joint_position"
    assert normalized.checkpoint_path
    assert Path(normalized.checkpoint_path).is_file()


def test_legacy_eef_dp_still_resolves_osc_pose():
    from app.services.dp_schema_resolver import resolve_dp_eval_executor

    spec = resolve_dp_eval_executor(
        policy="diffusion_policy",
        model_asset={"modelType": "diffusion_policy", "actionDim": 7},
    )
    assert spec.eval_executor == "osc_pose"


def test_expert_policy_still_scripted():
    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="expert_policy_evaluation",
        numEpisodes=2,
    )
    normalized = normalize_evaluate_request(request)
    assert normalized.policy == "scripted"
    assert normalized.eval_executor is None
