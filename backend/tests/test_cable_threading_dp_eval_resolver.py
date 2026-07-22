from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation.evaluation_request_resolver import normalize_evaluate_request


def test_cable_threading_dp_asset_uses_diffusion_policy():
    from app.services.model_asset_validation import ModelAssetValidationResult

    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        numEpisodes=10,
        modelAssetId="model__171526_8025_final",
    )
    asset = {
        "id": "model__171526_8025_final",
        "modelType": "diffusion_policy",
        "framework": "Diffusion Policy",
        "checkpointPath": "/tmp/model_final.pt",
    }
    validation = ModelAssetValidationResult(
        ok=True,
        reason="",
        model_asset_id="model__171526_8025_final",
        artifact_path="/tmp/model_final.pt",
        backend_type="diffusion_policy",
        source_task_type="cable_threading",
        file_exists=True,
        file_size_bytes=1024,
        status="available",
    )
    with patch(
        "app.services.model_asset_validation.validate_model_asset",
        return_value=validation,
    ), patch(
        "app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id",
        return_value=asset,
    ):
        normalized = normalize_evaluate_request(request)
    assert normalized.policy == "diffusion_policy"
    assert normalized.model_asset_id == "model__171526_8025_final"


def test_cable_threading_bc_asset_uses_robomimic():
    from app.services.model_asset_validation import ModelAssetValidationResult

    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        numEpisodes=10,
        modelAssetId="model__bc_final",
    )
    asset = {
        "id": "model__bc_final",
        "modelType": "robomimic_bc",
        "framework": "Robomimic BC",
        "checkpointPath": "/tmp/model.pth",
    }
    validation = ModelAssetValidationResult(
        ok=True,
        reason="",
        model_asset_id="model__bc_final",
        artifact_path="/tmp/model.pth",
        backend_type="robomimic_bc",
        source_task_type="cable_threading",
        file_exists=True,
        file_size_bytes=1024,
        status="available",
    )
    with patch(
        "app.services.model_asset_validation.validate_model_asset",
        return_value=validation,
    ), patch(
        "app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id",
        return_value=asset,
    ):
        normalized = normalize_evaluate_request(request)
    assert normalized.policy == "robomimic"
