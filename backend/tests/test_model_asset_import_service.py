from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
import torch
from fastapi import HTTPException

from app.services import model_asset_import_service as svc


def _dp_checkpoint(action_dim: int = 7) -> bytes:
    payload = {
        "state_dict": {"layer.weight": torch.zeros(2, 2)},
        "normalizer": {
            "action": {"scale": [1.0] * action_dim, "offset": [0.0] * action_dim},
            "low_dim": {"scale": [1.0] * 9, "offset": [0.0] * 9},
        },
        "train_config": {
            "backend": "diffusion_policy",
            "action_dim": action_dim,
            "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "low_dim_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            "image_size": 84,
            "low_dim_dim": 9,
            "vision_encoder": "resnet18",
        },
    }
    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def test_import_rejects_mismatched_action_dim():
    checkpoint = _dp_checkpoint(action_dim=14)
    upload = AsyncMock()
    upload.filename = "model.pt"
    upload.read = AsyncMock(return_value=checkpoint)

    manifest = {
        "datasetId": "ds_test",
        "datasetName": "test",
        "taskType": "cable_threading",
        "actionDim": 7,
        "cameraKeys": ["agentview_image", "robot0_eye_in_hand_image"],
        "observationKeys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "imageSize": 84,
        "artifacts": {"hdf5": "/tmp/fake.hdf5"},
    }

    async def _run() -> None:
        with patch("app.services.model_asset_import_service.resolve_manifest_by_dataset_id", return_value=manifest):
            with patch("app.services.training_service._resolve_hdf5_path", return_value=None):
                with pytest.raises(HTTPException) as exc:
                    await svc.import_pretrained_model_asset(
                        model_name="bad import",
                        model_type="diffusion_policy",
                        task_type="cable_threading",
                        dataset_id="ds_test",
                        checkpoint_file=upload,
                    )
        assert "action_dim" in str(exc.value.detail).lower() or "不一致" in str(exc.value.detail)

    asyncio.run(_run())
