from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import training_service as svc


def test_create_training_job_persists_adaptation_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    h5py = pytest.importorskip("h5py")
    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "CABLE_WORKING_DIR", tmp_path / "CableThreadingMVP")
    monkeypatch.setattr(svc, "TRAIN_BC_SCRIPT", tmp_path / "CableThreadingMVP/examples/cable_threading/train_bc.py")
    (tmp_path / "CableThreadingMVP/examples/cable_threading").mkdir(parents=True)
    (svc.TRAIN_BC_SCRIPT).write_text("# stub", encoding="utf-8")

    hdf5 = tmp_path / "dataset.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_eef_pos", data=[[0.0] * 3] * 5)
        demo.create_dataset("actions", data=[[0.0] * 7] * 5)
    npz = tmp_path / "dataset.npz"
    npz.write_bytes(b"npz")
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    with patch.object(svc, "_execute_training_job", side_effect=lambda job_id: None):
        with patch.object(svc, "record_workspace_job_start") as record_start:
            result = svc.create_training_job(
                {
                    "datasetId": "ds_adapt",
                    "datasetManifest": {
                        "datasetId": "ds_adapt",
                        "backend": "mujoco",
                        "robotType": "Panda",
                        "successfulEpisodes": 3,
                        "artifacts": {"hdf5": str(hdf5), "npz": str(npz)},
                    },
                    "downstreamModelType": "Robomimic",
                    "trainingBackend": "robomimic_bc",
                }
            )

    assert result["status"] == "queued"
    train_config = svc._read_json(tmp_path / "training" / "jobs" / result["trainJobId"] / "config" / "train_config.json")
    snapshot = train_config["adaptationSnapshot"]
    assert snapshot["modelType"] == "robomimic_bc"
    assert snapshot["datasetProfile"]["datasetId"] == "ds_adapt"
    assert train_config["architectureConfig"]
    assert (tmp_path / "training" / "jobs" / result["trainJobId"] / "artifacts" / "training_adaptation.json").is_file()

    metadata = record_start.call_args.kwargs["metadata"]
    assert metadata["adaptationSnapshot"]["modelType"] == "robomimic_bc"
