from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import workspace_model_asset_service as svc
from app.services.checkpoint_registry import register_checkpoint_assets
from app.services.training_job_generated_assets import (
    collect_init_checkpoint_paths,
    filter_training_job_detail_model_assets,
    is_init_checkpoint_asset,
)


def _setup_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_job_id: str) -> Path:
    train_job_dir = tmp_path / "jobs" / train_job_id
    ckpt_dir = train_job_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (train_job_dir / "config").mkdir(parents=True)
    (train_job_dir / "artifacts").mkdir(parents=True)
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"epochs": 100, "saveFinal": True, "taskName": "demo"}),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps({"status": "running", "totalEpochs": 100, "epoch": 18, "datasetName": "ds"}),
        encoding="utf-8",
    )
    (train_job_dir / "artifacts" / "dataset_manifest.json").write_text(
        json.dumps({"datasetId": "ds1"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(svc, "TRAINING_JOBS_ROOT", tmp_path / "jobs")
    return train_job_dir


def _detail_assets(train_job_id: str) -> list[dict]:
    payload = svc.list_training_job_model_assets_detail(train_job_id)
    return list(payload.get("modelAssets") or [])


def test_backend_context_prefers_actual_resolved_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    train_job_id = "train_20260720_150020_d08d"
    train_job_dir = _setup_job(tmp_path, monkeypatch, train_job_id)
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"trainingBackend": "robomimic_bc", "epochs": 5}),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "epoch": 5,
                "totalEpochs": 5,
                "trainingBackend": "robomimic_bc",
                "trainingBackendResolved": "torch_bc",
            }
        ),
        encoding="utf-8",
    )

    _, _, _, resolved_backend, framework, _ = svc._resolve_job_backend_context(
        train_job_dir, train_job_id
    )

    assert resolved_backend == "torch_bc"
    assert framework == "BC (PyTorch)"


def test_detail_shows_final_placeholder_while_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260620_120000_abcd"
    train_job_dir = _setup_job(tmp_path, monkeypatch, train_job_id)
    epoch_ckpt = train_job_dir / "checkpoints" / "model_epoch_20.pth"
    epoch_ckpt.write_bytes(b"z")

    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest={"datasetId": "ds1"},
        train_config={"epochs": 100, "saveFinal": True, "taskName": "demo"},
        status={"status": "running", "totalEpochs": 100, "epoch": 20},
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=False,
    )

    detail = _detail_assets(train_job_id)
    kinds = [item.get("checkpointKind") for item in detail]
    assert "final" in kinds
    assert "epoch" not in kinds
    final_row = next(item for item in detail if item.get("checkpointKind") == "final")
    assert final_row.get("isPlaceholder") is True
    assert final_row.get("canEvaluate") is False
    assert final_row.get("displayStatus") == "waiting"

    main_list = svc.list_model_assets_for_training_job(train_job_id)
    assert all(item.get("checkpointKind") != "final" for item in main_list)
    assert any(item.get("checkpointKind") == "epoch" for item in main_list)


def test_detail_hides_epoch_eval_while_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260620_120001_abcd"
    train_job_dir = _setup_job(tmp_path, monkeypatch, train_job_id)
    (train_job_dir / "checkpoints" / "model_epoch_20.pth").write_bytes(b"z")

    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest={"datasetId": "ds1"},
        train_config={"epochs": 100, "saveFinal": True},
        status={"status": "running", "totalEpochs": 100, "epoch": 20},
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=False,
    )

    detail = _detail_assets(train_job_id)
    assert not any(item.get("checkpointKind") == "epoch" for item in detail)


def test_detail_running_with_init_checkpoint_excludes_pretrained_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    train_job_id = "train_20260620_init_test_abcd"
    train_job_dir = _setup_job(tmp_path, monkeypatch, train_job_id)
    init_ckpt = tmp_path / "external" / "model_final.pt"
    init_ckpt.parent.mkdir(parents=True)
    init_ckpt.write_bytes(b"init")

    train_config = {
        "epochs": 200,
        "saveFinal": True,
        "taskName": "joint demo",
        "pretrained": {
            "modelAssetId": "model_ull_pipeline_final",
            "modelAssetName": "线缆穿杆预训练",
            "checkpointPath": str(init_ckpt),
            "sourceTrainJobId": "train_external",
        },
    }
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps(train_config),
        encoding="utf-8",
    )
  # rolling checkpoint inside job dir — should not appear as ready final while running
    rolling_final = train_job_dir / "checkpoints" / "diffusion_policy" / "checkpoints" / "model_final.pt"
    rolling_final.parent.mkdir(parents=True)
    rolling_final.write_bytes(b"rolling")

    (train_job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "logs" / "train.log").write_text(
        f"command: train_dp.py --init-checkpoint {init_ckpt}\nEpoch 1 Loss: 0.2\n",
        encoding="utf-8",
    )

    payload = svc.list_training_job_model_assets_detail(train_job_id)
    detail = list(payload.get("modelAssets") or [])
    assert not any(
        str(item.get("checkpointPath") or "").endswith("external/model_final.pt") for item in detail
    )
    assert all(not item.get("canEvaluate") for item in detail)
    if detail:
        assert all(item.get("isPlaceholder") for item in detail)
    else:
        assert payload.get("listMessage") == "模型资产将在当前训练任务完成后生成。"


def test_init_checkpoint_asset_detection():
    init_path = "/data/pretrain/model_final.pt"
    init_paths = {init_path}
    assert is_init_checkpoint_asset(
        {"checkpointPath": init_path},
        init_paths=init_paths,
        init_asset_ids=set(),
    )
    assert is_init_checkpoint_asset(
        {"modelAssetId": "model_pretrained"},
        init_paths=set(),
        init_asset_ids={"model_pretrained"},
    )


def test_collect_init_checkpoint_paths_reads_pretrained_and_dp_config(tmp_path: Path):
    train_job_dir = tmp_path / "job"
    dp_cfg_dir = train_job_dir / "checkpoints" / "diffusion_policy" / "config"
    dp_cfg_dir.mkdir(parents=True)
    init_path = str(tmp_path / "init.pt")
    (dp_cfg_dir / "train_config.json").write_text(
        json.dumps({"init_checkpoint": init_path}),
        encoding="utf-8",
    )
    paths = collect_init_checkpoint_paths(
        {"pretrained": {"checkpointPath": "/other/init.pt"}},
        train_job_dir=train_job_dir,
    )
    assert "/other/init.pt" in {p for p in paths if p.endswith("other/init.pt")}
    assert init_path in paths or any(p.endswith("init.pt") for p in paths)


def test_detail_final_ready_when_running_but_epoch_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260620_120003_abcd"
    train_job_dir = _setup_job(tmp_path, monkeypatch, train_job_id)
    (train_job_dir / "checkpoints" / "model_final.pth").write_bytes(b"x")
    (train_job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "status.json").write_text(
        json.dumps({"status": "running", "totalEpochs": 100, "epoch": 100}),
        encoding="utf-8",
    )
    lines = [f"Epoch {i} Loss: 0.1\n" for i in range(1, 101)]
    (train_job_dir / "logs" / "train.log").write_text("\n".join(lines), encoding="utf-8")

    detail = _detail_assets(train_job_id)
    final_row = next(item for item in detail if item.get("checkpointKind") == "final")
    assert final_row.get("isPlaceholder") is False
    assert final_row.get("canEvaluate") is True
    assert final_row.get("displayStatus") == "ready"


def test_detail_final_ready_after_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260620_120002_abcd"
    train_job_dir = _setup_job(tmp_path, monkeypatch, train_job_id)
    (train_job_dir / "checkpoints" / "model_final.pth").write_bytes(b"x")
    (train_job_dir / "status.json").write_text(
        json.dumps({"status": "completed", "totalEpochs": 100, "epoch": 100}),
        encoding="utf-8",
    )

    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest={"datasetId": "ds1"},
        train_config={"epochs": 100, "saveFinal": True},
        status={"status": "completed", "totalEpochs": 100, "epoch": 100},
        resolved_backend="robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
        register_final=True,
    )

    detail = _detail_assets(train_job_id)
    final_row = next(item for item in detail if item.get("checkpointKind") == "final")
    assert final_row.get("isPlaceholder") is False
    assert final_row.get("canEvaluate") is True
    assert final_row.get("displayStatus") == "ready"

    main_list = svc.list_model_assets_for_training_job(train_job_id)
    assert any(item.get("checkpointKind") == "final" for item in main_list)


def test_detail_act_joint_final_can_evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_act_joint_eval"
    train_job_dir = _setup_job(tmp_path, monkeypatch, train_job_id)
    final_path = train_job_dir / "checkpoints" / "act" / "checkpoints" / "model_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"x")
    (train_job_dir / "config" / "act_adapted.yaml").write_text(
        "\n".join(
            [
                "action_dim: 8",
                "action_key: actions",
                "eval_executor: joint_position",
                "controller_type: JOINT_POSITION",
                "low_dim_keys: [robot0_joint_pos, robot0_gripper_qpos]",
                "image_keys: [agentview_image, robot0_eye_in_hand_image]",
            ]
        ),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps({"status": "completed", "totalEpochs": 2, "epoch": 2, "trainingBackend": "act"}),
        encoding="utf-8",
    )
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"epochs": 2, "saveFinal": True, "trainingBackend": "act", "downstreamModelType": "ACT"}),
        encoding="utf-8",
    )

    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest={"datasetId": "ds1"},
        train_config={"epochs": 2, "saveFinal": True, "trainingBackend": "act"},
        status={"status": "completed", "totalEpochs": 2, "epoch": 2},
        resolved_backend="act",
        framework_label="ACT",
        model_type="act",
        register_final=True,
    )

    detail = _detail_assets(train_job_id)
    final_row = next(item for item in detail if item.get("checkpointKind") == "final")
    assert final_row.get("canEvaluate") is True
    assert final_row.get("modelType") == "act"


def test_detail_promotes_pending_final_after_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_id = "train_20260620_120004_abcd"
    train_job_dir = _setup_job(tmp_path, monkeypatch, train_job_id)
    final_path = train_job_dir / "checkpoints" / "diffusion_policy" / "checkpoints" / "model_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"x")
    (train_job_dir / "artifacts" / "model_assets_registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "sourceTrainJobId": train_job_id,
                "assets": [
                    {
                        "modelAssetId": "model__120004_abcd_testfinal",
                        "name": "demo · Final",
                        "displayName": "demo · Final",
                        "checkpointKind": "final",
                        "checkpointPath": str(final_path),
                        "status": "pending",
                        "sourceTrainJobId": train_job_id,
                        "trainingBackend": "diffusion_policy",
                        "framework": "Diffusion Policy",
                        "modelType": "diffusion_policy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (train_job_dir / "status.json").write_text(
        json.dumps({"status": "completed", "totalEpochs": 100, "epoch": 100}),
        encoding="utf-8",
    )
    (train_job_dir / "config" / "train_config.json").write_text(
        json.dumps({"epochs": 100, "saveFinal": True, "trainingBackend": "diffusion_policy"}),
        encoding="utf-8",
    )

    detail = _detail_assets(train_job_id)
    final_row = next(item for item in detail if item.get("checkpointKind") == "final")
    assert final_row.get("displayStatus") == "ready"
    assert final_row.get("canEvaluate") is True


def test_filter_excludes_external_checkpoint_registered_to_job(tmp_path: Path):
    train_job_id = "train_filter_external"
    train_job_dir = tmp_path / train_job_id
    train_job_dir.mkdir(parents=True)
    external = tmp_path / "external_final.pt"
    external.write_bytes(b"x")
    status = {"status": "running", "totalEpochs": 100, "epoch": 5}
    train_config = {
        "pretrained": {"checkpointPath": str(external), "modelAssetId": "pretrained_id"},
    }
    assets = [
        {
            "id": "pretrained_id",
            "modelAssetId": "pretrained_id",
            "checkpointPath": str(external),
            "checkpointKind": "final",
            "sourceTrainJobId": train_job_id,
            "displayStatus": "ready",
            "canEvaluate": True,
        }
    ]
    filtered = filter_training_job_detail_model_assets(
        assets,
        train_job_id=train_job_id,
        train_job_dir=train_job_dir,
        status=status,
        train_config=train_config,
    )
    assert filtered == []
