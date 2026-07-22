from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from app.services import training_service as svc


def _write_low_dim_hdf5(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_eef_pos", data=[[0.0] * 3] * 10)
        demo.create_dataset("actions", data=[[0.0] * 7] * 10)


def test_custom_robomimic_bc_hidden_dims_in_train_config_and_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    from app.db.base import Base
    from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
    from app.services import model_type_service as mt_svc

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, _compiler, **_kw):
        return "JSON"

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(mt_svc, "SessionLocal", TestSession)

    mt_svc.ensure_default_model_types(TestSession())
    mt_svc.create_model_type(
        {
            "name": "Custom Robomimic BC",
            "modelTypeId": "custom-robomimic-bc-256",
            "baseAlgorithm": "robomimic_bc",
            "structureConfig": {"actor_hidden_dims": "256,256"},
            "status": "available",
        }
    )

    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "CABLE_WORKING_DIR", tmp_path / "CableThreadingMVP")
    train_bc = tmp_path / "CableThreadingMVP/examples/cable_threading/train_bc.py"
    train_bc.parent.mkdir(parents=True)
    train_bc.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(svc, "TRAIN_BC_SCRIPT", train_bc)
    monkeypatch.setattr(svc, "PYTHON_BIN", Path("/usr/bin/python3"))

    hdf5 = tmp_path / "dataset.hdf5"
    _write_low_dim_hdf5(hdf5)
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    manifest = {
        "datasetId": "ds_custom_bc",
        "taskType": "cable_threading",
        "artifacts": {"hdf5": str(hdf5)},
    }

    with patch.object(svc, "_execute_training_job", side_effect=lambda job_id: None):
        result = svc.create_training_job(
            {
                "datasetId": "ds_custom_bc",
                "datasetManifest": manifest,
                "modelTypeId": "custom-robomimic-bc-256",
                "epochs": 1,
                "batchSize": 8,
            }
        )

    job_dir = tmp_path / "training" / "jobs" / result["trainJobId"]
    train_config = svc._read_json(job_dir / "config" / "train_config.json")
    assert train_config["modelTypeId"] == "custom-robomimic-bc-256"
    assert train_config["modelParams"]["actor_hidden_dims"] == "256,256"
    assert train_config["structureConfig"]["actor_hidden_dims"] == "256,256"

    cmd = svc._build_train_command(
        backend="robomimic_bc",
        hdf5_path=hdf5,
        out_dir=job_dir / "checkpoints",
        train_config=train_config,
    )
    assert "--actor-hidden-dims" in cmd
    idx = cmd.index("--actor-hidden-dims")
    assert cmd[idx + 1] == "256,256"


def test_custom_dp_horizon_in_yaml_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    from app.db.base import Base
    from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
    from app.services import model_type_service as mt_svc

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, _compiler, **_kw):
        return "JSON"

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(mt_svc, "SessionLocal", TestSession)

    mt_svc.ensure_default_model_types(TestSession())
    mt_svc.create_model_type(
        {
            "name": "Custom DP",
            "modelTypeId": "custom-dp-horizon-32",
            "baseAlgorithm": "diffusion_policy",
            "structureConfig": {
                "horizon": 32,
                "n_obs_steps": 4,
                "n_action_steps": 16,
                "num_inference_steps": 30,
            },
            "status": "available",
        }
    )

    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "CABLE_WORKING_DIR", tmp_path / "CableThreadingMVP")
    train_dp = tmp_path / "CableThreadingMVP/examples/cable_threading/train_dp.py"
    train_dp.parent.mkdir(parents=True)
    train_dp.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(svc, "TRAIN_DP_SCRIPT", train_dp)
    monkeypatch.setattr(svc, "PYTHON_BIN", Path("/usr/bin/python3"))

    hdf5 = tmp_path / "dataset.hdf5"
    _write_low_dim_hdf5(hdf5)
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    manifest = {
        "datasetId": "ds_custom_dp",
        "taskType": "cable_threading",
        "artifacts": {"hdf5": str(hdf5)},
    }

    with patch.object(svc, "_execute_training_job", side_effect=lambda job_id: None):
        result = svc.create_training_job(
            {
                "datasetId": "ds_custom_dp",
                "datasetManifest": manifest,
                "modelTypeId": "custom-dp-horizon-32",
                "epochs": 1,
                "batchSize": 8,
            }
        )

    job_dir = tmp_path / "training" / "jobs" / result["trainJobId"]
    train_config = svc._read_json(job_dir / "config" / "train_config.json")
    assert train_config["modelTypeId"] == "custom-dp-horizon-32"
    assert train_config["modelParams"]["horizon"] == 32

    dp_yaml_path = job_dir / "config" / "dp_adapted.yaml"
    assert dp_yaml_path.is_file()
    dp_cfg = yaml.safe_load(dp_yaml_path.read_text(encoding="utf-8"))
    assert int(dp_cfg["horizon"]) == 32
