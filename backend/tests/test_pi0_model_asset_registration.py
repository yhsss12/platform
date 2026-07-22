from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.checkpoint_registry import explain_model_asset_eval_blocker
from app.services import training_service as training_svc
from app.services.pi0_lerobot_smoke_runner import PI0_EVAL_DISABLED_REASON


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/dataset.manifest.json"
)


@pytest.fixture()
def pi0_platform_env(tmp_path, monkeypatch):
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.base import Base
    from app.models.model_type_definition import ModelTypeDefinition  # noqa: F401
    from app.services import model_type_service as model_svc

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
    monkeypatch.setattr(model_svc, "SessionLocal", TestSession)
    model_svc.ensure_default_model_types(TestSession())

    jobs_root = tmp_path / "runs" / "training" / "jobs"
    jobs_root.mkdir(parents=True)
    monkeypatch.setattr(training_svc, "PROJECT_ROOT", PROJECT_ROOT)
    monkeypatch.setattr(training_svc, "TRAINING_ROOT", tmp_path / "runs" / "training")
    return jobs_root


def test_pi0_model_asset_ready_but_not_evaluable(pi0_platform_env):
    manifest = json.loads(SMOKE_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["datasetId"] = "ds_pi0_asset"
    payload = {
        "datasetId": manifest["datasetId"],
        "modelTypeId": "pi0",
        "datasetManifest": manifest,
        "epochs": 1,
        "batchSize": 2,
        "maxSteps": 10,
        "seed": 1,
    }

    with patch.object(
        training_svc,
        "probe_training_capabilities",
        return_value={"supportedTrainingBackends": ["pi0"], "pi0Capability": {"ready": True}},
    ), patch.object(training_svc, "sync_workspace_job_from_runtime"), patch.object(
        training_svc, "finalize_training_job_sync"
    ), patch.object(training_svc, "record_workspace_job_start"), patch.object(
        training_svc.threading,
        "Thread",
        lambda *args, **kwargs: type("NoopThread", (), {"start": lambda self: None})(),
    ):
        created = training_svc.create_training_job(payload)
        training_svc._execute_training_job(created["trainJobId"])

    train_job_dir = training_svc._train_job_dir(created["trainJobId"])
    model_manifest = json.loads((train_job_dir / "artifacts" / "model_manifest.json").read_text(encoding="utf-8"))
    assert model_manifest["modelType"] == "pi0"
    assert model_manifest["datasetFormat"] == "lerobot"
    assert model_manifest["stateDim"] == 9
    assert model_manifest["actionDim"] == 8
    assert model_manifest["controllerType"] == "JOINT_POSITION"
    assert model_manifest["actionMode"] == "joint_delta_derived"
    assert model_manifest["actionRepresentation"] == "normalized_joint_delta"
    assert model_manifest["taskInstruction"] == "thread the cable through the pole"
    assert model_manifest["canEvaluate"] is False
    assert model_manifest["evalDisabledReason"] == PI0_EVAL_DISABLED_REASON

    blocker = explain_model_asset_eval_blocker(
        model_manifest,
        job_status={"status": "completed"},
    )
    assert blocker in {PI0_EVAL_DISABLED_REASON, "pi0 platform evaluation not enabled"}


def test_pi0_model_asset_blocker_after_eval_adapter_ready(pi0_platform_env, monkeypatch):
    from app.services.policy_schema_resolver import PI0_PLATFORM_EVAL_NOT_ENABLED_REASON, pi0_eval_adapter_ready

    monkeypatch.setattr(
        "app.services.policy_schema_resolver.pi0_eval_adapter_ready",
        lambda: True,
    )
    entry = {
        "modelType": "pi0",
        "policyType": "pi0",
        "datasetFormat": "lerobot",
        "stateDim": 9,
        "actionDim": 8,
        "robot": "Panda",
        "controllerType": "JOINT_POSITION",
        "actionMode": "joint_delta_derived",
        "lowDimKeys": ["robot0_joint_pos", "robot0_gripper_qpos"],
        "canEvaluate": False,
        "status": "ready",
        "checkpointPath": str(
            PROJECT_ROOT
            / "runs/training/jobs/train_20260630_123947_ebd2/checkpoints/pi0/checkpoints/model_final.pt"
        ),
    }
    blocker = explain_model_asset_eval_blocker(entry, job_status={"status": "completed"})
    assert blocker == PI0_PLATFORM_EVAL_NOT_ENABLED_REASON
    assert pi0_eval_adapter_ready() is True
