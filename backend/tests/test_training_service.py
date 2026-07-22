from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastapi import HTTPException

from app.services import training_service as svc


def test_probe_training_capabilities_finds_robomimic_script():
    result = svc.probe_training_capabilities()
    assert result["foundTrainingScripts"] is True
    assert "robomimic_bc" in result["supportedTrainingBackends"]
    assert "torch_bc" in result["supportedTrainingBackends"]
    if svc.TRAIN_ACT_SCRIPT.is_file():
        assert "act" in result["supportedTrainingBackends"]
    assert any("train_bc.py" in path for path in result["evidence"])
    assert result["recommendedBackend"] == "robomimic"


def test_nut_assembly_training_rejects_dataset_without_valid_demos(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "nut.hdf5"
    with h5py.File(dataset, "w") as handle:
        demos = handle.create_group("data")
        demo = demos.create_group("demo_0")
        demo.attrs["valid_for_training"] = False

    with pytest.raises(HTTPException) as exc_info:
        svc._validate_nut_assembly_training_dataset(
            {"taskType": "nut_assembly"}, dataset
        )

    assert exc_info.value.status_code == 422
    assert "没有成功且可用于训练的轨迹" in str(exc_info.value.detail)


def test_probe_training_capabilities_finds_diffusion_policy_script():
    result = svc.probe_training_capabilities()
    if not svc.TRAIN_DP_SCRIPT.is_file():
        pytest.skip("train_dp.py missing")
    assert "diffusion_policy" in result["supportedTrainingBackends"]
    assert any("train_dp.py" in path for path in result["evidence"])


def test_resolve_dual_arm_training_backend_torch_bc():
    capabilities = {"supportedTrainingBackends": ["robomimic_bc", "torch_bc"]}
    manifest = {
        "taskType": "dual_arm_cable_manipulation",
        "sourceJobId": "dac_gen_20260614_220258_il1b",
    }
    backend, message = svc._resolve_training_backend(
        downstream_model_type="Robomimic",
        training_backend="torch_bc",
        has_hdf5=True,
        capabilities=capabilities,
        manifest=manifest,
    )
    assert backend == "torch_bc"
    assert message == ""


@pytest.mark.parametrize("requested", ["auto", "robomimic", "robomimic_bc"])
def test_resolve_dual_arm_generic_bc_choice_uses_dedicated_torch_runner(requested: str):
    capabilities = {"supportedTrainingBackends": ["robomimic_bc", "torch_bc"]}
    backend, message = svc._resolve_training_backend(
        downstream_model_type="Robomimic",
        training_backend=requested,
        has_hdf5=True,
        capabilities=capabilities,
        manifest={
            "taskType": "dual_arm_cable_manipulation",
            "sourceJobId": "dac_gen_20260720_144324_6804",
        },
    )
    assert backend == "torch_bc"
    assert message == ""


def test_resolve_hdf5_path_for_dual_arm_job():
    job_id = "dac_gen_20260614_220258_il1b"
    hdf5 = (
        svc.PROJECT_ROOT
        / "runs"
        / "dual_arm_cable"
        / "jobs"
        / job_id
        / "datasets"
        / "dataset.hdf5"
    )
    if not hdf5.is_file():
        pytest.skip("dual-arm sample hdf5 missing")
    resolved = svc._resolve_hdf5_path({"sourceJobId": job_id, "taskType": "dual_arm_cable_manipulation"})
    assert resolved == hdf5.resolve()


def test_resolve_training_backend_act_unavailable():
    capabilities = {"supportedTrainingBackends": ["robomimic_bc"]}
    backend, message = svc._resolve_training_backend(
        downstream_model_type="ACT",
        training_backend="auto",
        has_hdf5=True,
        capabilities=capabilities,
        manifest={"taskType": "cable_threading"},
    )
    assert backend is None
    assert "act" in message.lower()


def test_resolve_device_defaults_to_cuda():
    assert svc._resolve_device("") == "cuda"
    assert svc._resolve_device("cuda_if_available") == "cuda"
    assert svc._resolve_device("l20") == "cuda"
    assert svc._resolve_device("cpu") == "cpu"


def test_resolve_training_backend_robomimic_explicit():
    capabilities = {"supportedTrainingBackends": ["robomimic_bc"]}
    for req in ("robomimic", "robomimic_bc"):
        backend, message = svc._resolve_training_backend(
            downstream_model_type="Robomimic",
            training_backend=req,
            has_hdf5=True,
            capabilities=capabilities,
            manifest={"taskType": "cable_threading"},
        )
        assert backend == "robomimic_bc"
        assert message == ""


def test_resolve_hdf5_path_from_npz_sibling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])
    hdf5 = tmp_path / "datasets" / "dataset.hdf5"
    npz = tmp_path / "datasets" / "dataset.npz"
    hdf5.parent.mkdir(parents=True)
    hdf5.write_bytes(b"hdf5")
    npz.write_bytes(b"npz")
    resolved = svc._resolve_hdf5_path({"artifacts": {"npz": str(npz)}})
    assert resolved == hdf5


def test_validate_dataset_trainable_requires_successful_episodes(tmp_path: Path):
    manifest = {
        "successfulEpisodes": 0,
        "artifacts": {"npz": str(tmp_path / "dataset.npz")},
    }
    (tmp_path / "dataset.npz").write_bytes(b"npz")
    ok, reason = svc._validate_dataset_trainable(manifest)
    assert ok is False
    assert "成功轨迹" in reason


def test_create_training_job_act_rejects_low_dim_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "lowdim.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_eef_pos", data=[[0.0, 0.0, 0.0]])
        demo.create_dataset("actions", data=[[0.0] * 7] * 10)

    manifest = {
        "datasetId": "ds_lowdim",
        "datasetName": "lowdim",
        "successfulEpisodes": 1,
        "artifacts": {"hdf5": str(hdf5)},
    }

    with pytest.raises(HTTPException) as exc:
        svc.create_training_job(
            {
                "datasetId": "ds_lowdim",
                "datasetManifest": manifest,
                "downstreamModelType": "ACT",
                "trainingBackend": "act",
                "epochs": 1,
            }
        )
    assert "image observations" in str(exc.value.detail)
    leftover = list((svc.TRAINING_ROOT / "jobs").glob("train_*"))
    assert leftover == [], "non-adaptable ACT job must not leave runtime job directory"


def test_create_training_job_backend_unavailable_for_act(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "CABLE_WORKING_DIR", tmp_path / "CableThreadingMVP")
    monkeypatch.setattr(svc, "TRAIN_BC_SCRIPT", tmp_path / "CableThreadingMVP/examples/cable_threading/train_bc.py")
    monkeypatch.setattr(svc, "TRAIN_ACT_SCRIPT", tmp_path / "CableThreadingMVP/examples/cable_threading/train_act.py")
    (tmp_path / "CableThreadingMVP/examples/cable_threading").mkdir(parents=True)
    (svc.TRAIN_BC_SCRIPT).write_text("# stub", encoding="utf-8")

    h5py = pytest.importorskip("h5py")
    import numpy as np

    hdf5 = tmp_path / "dataset.hdf5"
    horizon = 16
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((horizon, 64, 64, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eef_pos", data=np.zeros((horizon, 3), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((horizon, 7), dtype=np.float32))

    manifest = {
        "datasetId": "ds_test",
        "datasetName": "test dataset",
        "successfulEpisodes": 3,
        "artifacts": {"hdf5": str(hdf5)},
    }

    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    with patch.object(svc, "_execute_training_job", side_effect=lambda job_id: None):
        result = svc.create_training_job(
            {
                "datasetId": "ds_test",
                "datasetManifest": manifest,
                "downstreamModelType": "ACT",
                "trainingBackend": "auto",
                "dataFormat": "HDF5",
                "epochs": 2,
                "batchSize": 8,
                "learningRate": 0.0001,
                "device": "cpu",
            }
        )

    assert result["trainJobId"].startswith("train_")
    train_job_dir = svc._train_job_dir(result["trainJobId"])
    status_path = train_job_dir / "status.json"
    train_config_path = train_job_dir / "config" / "train_config.json"
    assert status_path.is_file()
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    assert train_config["device"] == "cpu"
    assert train_config["deviceLabel"] == "L20"

    svc._execute_training_job(result["trainJobId"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "backend_unavailable"
    assert status.get("checkpointExists") is not True
    assert "act" in status.get("message", "").lower()


def test_model_manifest_only_when_checkpoint_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_job_dir = tmp_path / "jobs/train_test"
    train_job_dir.mkdir(parents=True)
    checkpoint = train_job_dir / "checkpoints" / "model_final.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ckpt")

    manifest = svc._register_model_manifest(
        train_job_dir=train_job_dir,
        train_job_id="train_test",
        manifest={"datasetId": "ds_1", "taskType": "cable_threading"},
        train_config={"downstreamModelType": "Robomimic"},
        checkpoint_path=checkpoint,
        resolved_backend="robomimic_bc",
    )

    assert manifest["status"] == "ready"
    assert manifest["checkpointPath"] == str(checkpoint)
    assert Path(train_job_dir / "artifacts/model_manifest.json").is_file()


def test_build_train_command_robomimic_without_advanced():
    cmd = svc._build_train_command(
        backend="robomimic_bc",
        hdf5_path=Path("/tmp/dataset.hdf5"),
        out_dir=Path("/tmp/out"),
        train_config={
            "epochs": 10,
            "batchSize": 8,
            "learningRate": 0.001,
            "device": "cuda",
            "seed": 42,
            "advancedEnabled": False,
        },
    )
    assert "--num-epochs" in cmd
    assert cmd[cmd.index("--num-epochs") + 1] == "10"
    assert "--actor-hidden-dims" not in cmd


def test_build_train_command_robomimic_with_advanced():
    cmd = svc._build_train_command(
        backend="robomimic_bc",
        hdf5_path=Path("/tmp/dataset.hdf5"),
        out_dir=Path("/tmp/out"),
        train_config={
            "epochs": 20,
            "batchSize": 16,
            "learningRate": 0.0001,
            "device": "cpu",
            "seed": 7,
            "advancedEnabled": True,
            "modelParams": {
                "actor_hidden_dims": "256,128",
                "l2_regularization": 0.01,
            },
        },
    )
    assert cmd[cmd.index("--actor-hidden-dims") + 1] == "256,128"
    assert cmd[cmd.index("--l2-regularization") + 1] == "0.01"
    assert "--normalize-obs" not in cmd
    assert "--save-every-n-epochs" in cmd
    assert cmd[cmd.index("--save-every-n-epochs") + 1] == "20"
    assert "--num-data-workers" not in cmd


def test_build_train_command_torch_bc_without_advanced():
    cmd = svc._build_train_command(
        backend="torch_bc",
        hdf5_path=Path("/tmp/dataset.hdf5"),
        out_dir=Path("/tmp/out"),
        train_config={
            "epochs": 10,
            "batchSize": 8,
            "learningRate": 0.001,
            "device": "cuda",
            "seed": 42,
            "advancedEnabled": False,
        },
    )
    assert "--num-epochs" in cmd
    assert "--hidden-dims" not in cmd
    assert "--weight-decay" not in cmd


def test_build_train_command_torch_bc_with_advanced():
    cmd = svc._build_train_command(
        backend="torch_bc",
        hdf5_path=Path("/tmp/dataset.hdf5"),
        out_dir=Path("/tmp/out"),
        train_config={
            "epochs": 20,
            "batchSize": 16,
            "learningRate": 0.0001,
            "device": "cpu",
            "seed": 7,
            "advancedEnabled": True,
            "modelParams": {
                "hidden_dims": "256,128",
                "weight_decay": 0.0001,
            },
        },
    )
    assert cmd[cmd.index("--hidden-dims") + 1] == "256,128"
    assert cmd[cmd.index("--weight-decay") + 1] == "0.0001"


def test_build_train_command_appends_init_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])
    checkpoint = tmp_path / "model_final.pth"
    checkpoint.write_bytes(b"ckpt")

    cmd = svc._build_train_command(
        backend="robomimic_bc",
        hdf5_path=Path("/tmp/dataset.hdf5"),
        out_dir=Path("/tmp/out"),
        train_config={
            "epochs": 10,
            "batchSize": 8,
            "learningRate": 0.001,
            "device": "cuda",
            "seed": 42,
            "pretrained": {"checkpointPath": str(checkpoint)},
        },
    )
    assert cmd[cmd.index("--init-checkpoint") + 1] == str(checkpoint.resolve())


def test_validate_pretrained_model_normalizes_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])
    checkpoint = tmp_path / "init.pth"
    checkpoint.write_bytes(b"ckpt")

    asset = {
        "id": "model_asset_1",
        "name": "BC baseline",
        "framework": "robomimic_bc",
        "taskTemplateId": "task_cable_threading_v1",
        "checkpointPath": str(checkpoint),
        "sourceTrainingJobId": "train_prev",
    }

    with patch("app.services.workspace_model_asset_service.get_model_asset_by_id", return_value=asset):
        normalized = svc._validate_pretrained_model(
            pretrained={"modelAssetId": "model_asset_1"},
            resolved_backend="robomimic_bc",
            manifest={"taskType": "cable_threading"},
        )

    assert normalized["checkpointPath"] == str(checkpoint.resolve())
    assert normalized["modelAssetName"] == "BC baseline"
    assert normalized["sourceTrainJobId"] == "train_prev"


def test_validate_pretrained_model_rejects_framework_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])
    checkpoint = tmp_path / "init.pth"
    checkpoint.write_bytes(b"ckpt")
    asset = {
        "id": "model_asset_1",
        "framework": "torch_bc",
        "taskTemplateId": "dual_arm_cable_manipulation",
        "checkpointPath": str(checkpoint),
    }

    with patch("app.services.workspace_model_asset_service.get_model_asset_by_id", return_value=asset):
        with pytest.raises(HTTPException) as exc:
            svc._validate_pretrained_model(
                pretrained={"modelAssetId": "model_asset_1"},
                resolved_backend="robomimic_bc",
                manifest={"taskType": "cable_threading"},
            )
    assert "不一致" in str(exc.value.detail)


def test_register_model_manifest_records_init_model_asset(tmp_path: Path):
    train_job_dir = tmp_path / "jobs/train_test"
    train_job_dir.mkdir(parents=True)
    checkpoint = train_job_dir / "checkpoints" / "model_final.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ckpt")

    manifest = svc._register_model_manifest(
        train_job_dir=train_job_dir,
        train_job_id="train_test",
        manifest={"datasetId": "ds_1", "taskType": "cable_threading"},
        train_config={
            "downstreamModelType": "Robomimic",
            "pretrained": {
                "modelAssetId": "model_asset_prev",
                "sourceTrainJobId": "train_prev",
            },
        },
        checkpoint_path=checkpoint,
        resolved_backend="robomimic_bc",
    )

    assert manifest["initModelAssetId"] == "model_asset_prev"
    assert manifest["initSourceTrainJobId"] == "train_prev"


def test_resolve_training_backend_diffusion_policy_single_arm():
    capabilities = {"supportedTrainingBackends": ["robomimic_bc", "diffusion_policy"]}
    backend, message = svc._resolve_training_backend(
        downstream_model_type="Diffusion Policy",
        training_backend="diffusion_policy",
        has_hdf5=True,
        capabilities=capabilities,
        manifest={"taskType": "cable_threading"},
    )
    assert backend == "diffusion_policy"
    assert message == ""


def test_resolve_training_backend_diffusion_policy_allows_dual_arm():
    capabilities = {"supportedTrainingBackends": ["torch_bc", "diffusion_policy"]}
    backend, message = svc._resolve_training_backend(
        downstream_model_type="Diffusion Policy",
        training_backend="diffusion_policy",
        has_hdf5=True,
        capabilities=capabilities,
        manifest={"taskType": "dual_arm_cable_manipulation"},
    )
    assert backend == "diffusion_policy"
    assert message == ""


def test_build_train_command_diffusion_policy_with_advanced():
    cmd = svc._build_train_command(
        backend="diffusion_policy",
        hdf5_path=Path("/tmp/dataset.hdf5"),
        out_dir=Path("/tmp/out"),
        train_config={
            "epochs": 12,
            "batchSize": 8,
            "learningRate": 0.0002,
            "device": "cuda",
            "seed": 3,
            "advancedEnabled": True,
            "modelParams": {
                "horizon": 16,
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "num_inference_steps": 10,
                "num_diffusion_steps": 15,
                "vision_encoder": "resnet18",
                "image_size": 128,
                "use_ema": True,
                "ema_decay": 0.995,
                "weight_decay": 0.0001,
            },
        },
    )
    assert "--horizon" in cmd
    assert cmd[cmd.index("--horizon") + 1] == "16"
    assert cmd[cmd.index("--n-obs-steps") + 1] == "2"
    assert cmd[cmd.index("--num-diffusion-steps") + 1] == "15"
    assert cmd[cmd.index("--vision-encoder") + 1] == "resnet18"
    assert cmd[cmd.index("--use-ema") + 1] == "true"
    assert "--init-checkpoint" not in cmd


def test_validate_dp_hdf5_requires_configured_image_keys(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "dataset.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=[1])
        demo.create_dataset("actions", data=[[0.0] * 7])

    ok, reason = svc._validate_dp_hdf5(
        hdf5,
        {"dpConfig": {"image_keys": ["agentview_image", "robot0_eye_in_hand_image"], "low_dim_keys": []}},
    )
    assert ok is False
    assert "robot0_eye_in_hand_image" in reason

    with h5py.File(hdf5, "a") as handle:
        handle["data"]["demo_0"]["obs"].create_dataset("robot0_eye_in_hand_image", data=[1])

    ok, reason = svc._validate_dp_hdf5(
        hdf5,
        {"dpConfig": {"image_keys": ["agentview_image", "robot0_eye_in_hand_image"], "low_dim_keys": []}},
    )
    assert ok is True
    assert reason == ""


def test_register_model_manifest_diffusion_policy_model_type(tmp_path: Path):
    train_job_dir = tmp_path / "jobs/train_dp"
    train_job_dir.mkdir(parents=True)
    checkpoint = train_job_dir / "checkpoints" / "diffusion_policy" / "model_final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ckpt")

    manifest = svc._register_model_manifest(
        train_job_dir=train_job_dir,
        train_job_id="train_dp",
        manifest={"datasetId": "ds_1", "taskType": "cable_threading"},
        train_config={"downstreamModelType": "Diffusion Policy"},
        checkpoint_path=checkpoint,
        resolved_backend="diffusion_policy",
    )

    assert manifest["modelType"] == "diffusion_policy"
    assert manifest["backendType"] == "diffusion_policy"


def test_create_training_job_request_accepts_large_epochs():
    from app.schemas.training import CreateTrainingJobRequest

    req = CreateTrainingJobRequest(
        datasetId="ds_test",
        epochs=3000,
        checkpointIntervalEpochs=100,
    )
    assert req.epochs == 3000
    assert req.checkpointIntervalEpochs == 100


def test_create_training_job_request_rejects_epochs_below_one():
    from pydantic import ValidationError

    from app.schemas.training import CreateTrainingJobRequest

    with pytest.raises(ValidationError):
        CreateTrainingJobRequest(datasetId="ds_test", epochs=0)


def test_build_train_command_act_includes_config(tmp_path: Path):
    if not svc.TRAIN_ACT_SCRIPT.is_file():
        pytest.skip("train_act.py not available")
    act_cfg = tmp_path / "act_adapted.yaml"
    act_cfg.write_text("task_name: test\naction_dim: 7\nimage_keys: [agentview_image]\n", encoding="utf-8")
    cmd = svc._build_train_command(
        backend="act",
        hdf5_path=tmp_path / "dataset.hdf5",
        out_dir=tmp_path / "out",
        train_config={
            "epochs": 2,
            "batchSize": 8,
            "learningRate": 1e-4,
            "device": "cpu",
            "seed": 1,
            "actConfigPath": str(act_cfg),
            "advancedEnabled": True,
            "modelParams": {"chunk_size": 16, "kl_weight": 5.0},
        },
    )
    assert "train_act.py" in " ".join(cmd)
    assert str(act_cfg) in cmd
    assert "--chunk-size" in cmd


def test_act_training_job_completes_with_image_hdf5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not svc.TRAIN_ACT_SCRIPT.is_file() or not svc.PYTHON_BIN.is_file():
        pytest.skip("ACT training runtime unavailable")

    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    h5py = pytest.importorskip("h5py")
    import numpy as np
    import time

    hdf5 = tmp_path / "act_dataset.hdf5"
    horizon = 24
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        for idx in range(2):
            demo = data.create_group(f"demo_{idx}")
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_image", data=np.zeros((horizon, 64, 64, 3), dtype=np.uint8))
            obs.create_dataset("robot0_eye_in_hand_image", data=np.zeros((horizon, 64, 64, 3), dtype=np.uint8))
            obs.create_dataset("robot0_eef_pos", data=np.zeros((horizon, 3), dtype=np.float32))
            obs.create_dataset("robot0_gripper_qpos", data=np.zeros((horizon, 2), dtype=np.float32))
            demo.create_dataset("actions", data=np.zeros((horizon, 7), dtype=np.float32))

    manifest = {
        "datasetId": "ds_act_run",
        "datasetName": "ACT smoke dataset",
        "taskType": "cable_threading",
        "successfulEpisodes": 2,
        "artifacts": {"hdf5": str(hdf5)},
    }

    with patch.object(svc, "_execute_training_job", wraps=svc._execute_training_job):
        result = svc.create_training_job(
            {
                "datasetId": "ds_act_run",
                "datasetManifest": manifest,
                "downstreamModelType": "ACT",
                "trainingBackend": "act",
                "epochs": 1,
                "batchSize": 4,
                "learningRate": 1e-4,
                "device": "cpu",
            }
        )

    train_job_id = result["trainJobId"]
    train_job_dir = svc._train_job_dir(train_job_id)
    assert (train_job_dir / "config" / "act_adapted.yaml").is_file()
    assert (train_job_dir / "config" / "train_config.json").is_file()

    deadline = time.time() + 180
    status = {}
    while time.time() < deadline:
        status = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") in {"completed", "failed", "backend_unavailable"}:
            break
        time.sleep(2)

    assert status.get("status") == "completed", status.get("message")
    assert (train_job_dir / "logs" / "train.log").is_file()
    metrics_path = train_job_dir / "artifacts" / "metrics.jsonl"
    if metrics_path.is_file():
        assert metrics_path.read_text(encoding="utf-8").strip()

    registry = train_job_dir / "artifacts" / "model_assets_registry.json"
    if registry.is_file():
        assets = json.loads(registry.read_text(encoding="utf-8")).get("assets") or []
        final_assets = [a for a in assets if (a.get("checkpointKind") or "").lower() == "final"]
        assert final_assets
        assert final_assets[0].get("modelType") == "act"
        assert final_assets[0].get("status") in {"ready", "available"}
        from app.services.checkpoint_registry import compute_asset_can_evaluate, compute_asset_display_status

        assert compute_asset_can_evaluate(final_assets[0], job_status=status) is False
        assert compute_asset_display_status(final_assets[0], job_status=status) == "ready"
