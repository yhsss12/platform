from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import training_service as svc


def _write_block_stacking_low_dim_hdf5(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("joint_pos", data=[[0.0] * 7] * 40)
        obs.create_dataset("eef_pos", data=[[0.0] * 3] * 40)
        demo.create_dataset("actions", data=[[0.0] * 7] * 40)


def test_create_training_job_block_stacking_with_diffusion_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "CABLE_WORKING_DIR", tmp_path / "CableThreadingMVP")
    monkeypatch.setattr(svc, "TRAIN_DP_SCRIPT", tmp_path / "CableThreadingMVP/examples/cable_threading/train_dp.py")
    (tmp_path / "CableThreadingMVP/examples/cable_threading").mkdir(parents=True)
    (svc.TRAIN_DP_SCRIPT).write_text("# stub", encoding="utf-8")
    (tmp_path / "CableThreadingMVP/examples/cable_threading/train_bc.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(svc, "TRAIN_BC_SCRIPT", tmp_path / "CableThreadingMVP/examples/cable_threading/train_bc.py")
    monkeypatch.setattr(svc, "PYTHON_BIN", Path("/usr/bin/python3"))

    hdf5 = tmp_path / "block_stacking.hdf5"
    _write_block_stacking_low_dim_hdf5(hdf5)
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    manifest = {
        "datasetId": "ds_isaac_stack",
        "datasetName": "物块堆叠数据集",
        "taskType": "isaac_block_stacking",
        "taskName": "物块堆叠",
        "simulatorBackend": "isaac_lab",
        "successfulEpisodes": 8,
        "artifacts": {"hdf5": str(hdf5)},
    }

    with patch.object(svc, "_execute_training_job", side_effect=lambda job_id: None):
        with patch.object(svc, "record_workspace_job_start") as record_start:
            result = svc.create_training_job(
                {
                    "datasetId": "ds_isaac_stack",
                    "datasetManifest": manifest,
                    "downstreamModelType": "Diffusion Policy",
                    "trainingBackend": "diffusion_policy",
                    "epochs": 3,
                    "batchSize": 8,
                }
            )

    assert result["status"] == "queued"
    job_dir = tmp_path / "training" / "jobs" / result["trainJobId"]
    train_config = svc._read_json(job_dir / "config" / "train_config.json")
    assert train_config["trainingBackend"] == "diffusion_policy"
    assert train_config.get("adaptationSnapshot")
    assert train_config["adaptationSnapshot"]["modelType"] == "diffusion_policy"
    assert train_config.get("dpConfigPath")
    assert Path(train_config["dpConfigPath"]).is_file()
    dp_config = train_config.get("dpConfig") or {}
    assert dp_config.get("action_dim") == 7
    assert "joint_pos" in (dp_config.get("low_dim_keys") or [])

    metadata = record_start.call_args.kwargs["metadata"]
    assert metadata["adaptationSnapshot"]["modelType"] == "diffusion_policy"


def test_diffusion_policy_low_dim_does_not_require_camera_keys(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "lowdim.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_eef_pos", data=[[0.0] * 3] * 10)
        demo.create_dataset("actions", data=[[0.0] * 7] * 10)

    train_config = {
        "dpConfig": {
            "low_dim_keys": ["robot0_eef_pos"],
            "image_keys": [],
        }
    }
    ok, reason = svc._validate_dp_hdf5(hdf5, train_config)
    assert ok is True
    assert reason == ""


def test_create_training_job_block_stacking_with_act_image_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "CABLE_WORKING_DIR", tmp_path / "CableThreadingMVP")
    monkeypatch.setattr(svc, "TRAIN_ACT_SCRIPT", tmp_path / "CableThreadingMVP/examples/cable_threading/train_act.py")
    (tmp_path / "CableThreadingMVP/examples/cable_threading").mkdir(parents=True)
    (svc.TRAIN_ACT_SCRIPT).write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(svc, "PYTHON_BIN", Path("/usr/bin/python3"))

    h5py = pytest.importorskip("h5py")
    import numpy as np

    hdf5 = tmp_path / "block_stacking_image.hdf5"
    horizon = 20
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.zeros((horizon, 128, 128, 3), dtype=np.uint8))
        obs.create_dataset("eef_pos", data=np.zeros((horizon, 3), dtype=np.float32))
        obs.create_dataset("eef_quat", data=np.zeros((horizon, 4), dtype=np.float32))
        obs.create_dataset("gripper_pos", data=np.zeros((horizon, 2), dtype=np.float32))
        obs.create_dataset("object", data=np.zeros((horizon, 39), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((horizon, 7), dtype=np.float32))

    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    manifest = {
        "datasetId": "ds_isaac_stack_image",
        "datasetName": "物块堆叠图像数据集",
        "taskType": "isaac_block_stacking",
        "taskName": "物块堆叠",
        "simulatorBackend": "isaac_lab",
        "robotType": "Panda",
        "successfulEpisodes": 1,
        "artifacts": {"hdf5": str(hdf5)},
    }

    with patch.object(svc, "_execute_training_job", side_effect=lambda job_id: None):
        result = svc.create_training_job(
            {
                "datasetId": "ds_isaac_stack_image",
                "datasetManifest": manifest,
                "downstreamModelType": "ACT",
                "trainingBackend": "act",
                "epochs": 1,
                "batchSize": 4,
            }
        )

    assert result["status"] == "queued"
    job_dir = tmp_path / "training" / "jobs" / result["trainJobId"]
    train_config = svc._read_json(job_dir / "config" / "train_config.json")
    assert train_config["trainingBackend"] == "act"
    assert train_config.get("actConfigPath")
    assert Path(train_config["actConfigPath"]).is_file()
    adaptation = train_config.get("adaptationSnapshot") or {}
    assert adaptation.get("validation", {}).get("adaptable") is True
    assert "agentview_image" in (adaptation.get("modelAdaptation", {}).get("inputConfig", {}).get("camera_names") or [])


def test_resolve_training_backend_allows_dp_on_isaac_manifest():
    capabilities = {"supportedTrainingBackends": ["robomimic_bc", "diffusion_policy"]}
    manifest = {"taskType": "isaac_block_stacking", "simulatorBackend": "isaac_lab"}
    backend, message = svc._resolve_training_backend(
        downstream_model_type="Diffusion Policy",
        training_backend="diffusion_policy",
        has_hdf5=True,
        capabilities=capabilities,
        manifest=manifest,
    )
    assert backend == "diffusion_policy"
    assert message == ""
