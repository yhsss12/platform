"""Tests for model asset resource indexing without reading binary checkpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.services.model_asset_validation import enrich_model_asset  # noqa: E402


def test_enrich_model_asset_dp_does_not_read_binary_checkpoint_as_utf8(tmp_path: Path):
    binary_ckpt = tmp_path / "model_final.pt"
    binary_ckpt.write_bytes(b"\x80" * 128)
    asset = {
        "id": "model__binary_dp",
        "modelAssetId": "model__binary_dp",
        "backendType": "diffusion_policy",
        "modelType": "diffusion_policy",
        "checkpointPath": str(binary_ckpt),
        "status": "available",
        "fileExists": True,
        "actionDim": 8,
        "structureConfig": {
            "input": {
                "image_keys": ["agentview_image"],
                "low_dim_keys": ["robot0_joint_pos"],
            },
            "output": {"action_dim": 8, "action_key": "joint_actions"},
        },
    }
    with patch(
        "app.services.model_asset_checkpoint_resolver.resolve_local_checkpoint_path",
        return_value=str(binary_ckpt),
    ):
        enriched = enrich_model_asset(asset)

    assert enriched.get("id") == "model__binary_dp"
    assert enriched.get("backendType") == "diffusion_policy"


def test_enrich_model_asset_pi0_uses_json_smoke_checkpoint(tmp_path: Path):
    smoke_ckpt = tmp_path / "model_final.pt"
    smoke_ckpt.write_text(
        json.dumps(
            {
                "format": "pi0_lerobot_smoke_v1",
                "backend": "pi0",
                "state_dim": 9,
                "action_dim": 8,
                "robot": "Panda",
                "controller_type": "JOINT_POSITION",
                "action_mode": "joint_delta_derived",
                "task_instruction": "thread the cable through the pole",
            }
        ),
        encoding="utf-8",
    )
    asset = {
        "id": "model__pi0_smoke",
        "backendType": "pi0",
        "modelType": "pi0",
        "policyType": "pi0",
        "datasetFormat": "lerobot",
        "checkpointPath": str(smoke_ckpt),
        "actionDim": 8,
        "controllerType": "JOINT_POSITION",
        "actionMode": "joint_delta_derived",
        "status": "available",
        "fileExists": True,
    }
    with patch(
        "app.services.model_asset_checkpoint_resolver.resolve_local_checkpoint_path",
        return_value=str(smoke_ckpt),
    ), patch(
        "app.services.policy_schema_resolver.pi0_platform_eval_ready",
        return_value=True,
    ), patch(
        "app.services.policy_schema_resolver.pi0_eval_adapter_ready",
        return_value=True,
    ):
        enriched = enrich_model_asset(asset)

    assert enriched.get("modelType") == "pi0"
    assert enriched.get("evalExecutor") == "joint_position"
