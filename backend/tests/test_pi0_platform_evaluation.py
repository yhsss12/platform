"""Tests for pi0 platform evaluation wiring (Phase G)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.schemas.evaluation import EvaluateAsyncRequest  # noqa: E402
from app.services.cable_threading_service import _build_eval_command  # noqa: E402
from app.services.evaluation.evaluation_request_resolver import normalize_evaluate_request  # noqa: E402


def _joint_asset() -> dict:
    return {
        "modelType": "pi0",
        "policyType": "pi0",
        "datasetFormat": "lerobot",
        "stateDim": 9,
        "actionDim": 8,
        "robot": "Panda",
        "controllerType": "JOINT_POSITION",
        "actionMode": "joint_delta_derived",
        "taskInstruction": "thread the cable through the pole",
        "sourceTrainingJobId": "train_20260630_123947_ebd2",
    }


def test_pi0_build_eval_command_includes_joint_position_args(tmp_path: Path):
    ckpt = tmp_path / "model_final.pt"
    cfg = tmp_path / "train_config.json"
    ckpt.write_text("{}", encoding="utf-8")
    cfg.write_text(json.dumps({"taskInstruction": "thread the cable through the pole"}), encoding="utf-8")
    job_root = tmp_path / "ct_eval_pi0"
    job_root.mkdir()
    cmd = _build_eval_command(
        job_root,
        episodes=1,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=200,
        seed=1,
        policy="pi0",
        checkpoint=str(ckpt),
        device="cpu",
        record_video=False,
        eval_executor="joint_position",
        controller_type="JOINT_POSITION",
        action_mode="joint_delta_derived",
        train_config_path=str(cfg),
        task_instruction="thread the cable through the pole",
    )
    joined = " ".join(cmd)
    assert "eval" in cmd
    assert "--policy" in cmd and "pi0" in cmd
    assert "--eval-executor" in cmd and "joint_position" in joined
    assert "--controller-type" in cmd and "JOINT_POSITION" in joined
    assert "--train-config" in cmd
    assert "--task-instruction" in cmd


def test_pi0_eval_resolver_returns_platform_fields(monkeypatch, tmp_path: Path):
    ckpt = tmp_path / "model_final.pt"
    cfg = tmp_path / "train_config.json"
    train_job = tmp_path / "runs/training/jobs/train_pi0/config"
    train_job.mkdir(parents=True)
    ckpt.write_text(
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
    cfg.write_text(json.dumps({"taskInstruction": "thread the cable through the pole"}), encoding="utf-8")
    (train_job / "train_config.json").write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    asset = _joint_asset()
    asset["checkpointPath"] = str(ckpt)

    def _fake_get_model_asset_by_id(_model_asset_id: str):
        return asset

    def _fake_resolve_eval_checkpoint_path(*, asset=None, path_hint=None, model_asset_id=None):
        return str(ckpt), True

    def _fake_infer_trained_policy_type(*, model_asset=None, checkpoint_path=None):
        return "pi0"

    monkeypatch.setattr(
        "app.services.evaluation.evaluation_request_resolver.get_model_asset_by_id",
        _fake_get_model_asset_by_id,
    )
    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.resolve_eval_checkpoint_path",
        _fake_resolve_eval_checkpoint_path,
    )
    monkeypatch.setattr(
        "app.services.model_asset_checkpoint_resolver.infer_trained_policy_type",
        _fake_infer_trained_policy_type,
    )
    import app.services.policy_schema_resolver as psr

    monkeypatch.setattr(psr, "PROJECT_ROOT", tmp_path)

    normalized = normalize_evaluate_request(
        EvaluateAsyncRequest(
            taskTemplateId="cable_threading_single_arm",
            evaluationMode="trained_model_evaluation",
            modelAssetId="model__123947_ebd2_final",
            numEpisodes=1,
            horizon=200,
        )
    )
    assert normalized.policy == "pi0"
    assert normalized.eval_executor == "joint_position"
    assert normalized.robot == "Panda"
    assert normalized.task_instruction == "thread the cable through the pole"
    assert normalized.state_dim == 9
    assert normalized.action_dim == 8


def test_pi0_build_eval_command_does_not_use_osc_pose(tmp_path: Path):
    job_root = tmp_path / "job"
    job_root.mkdir()
    ckpt = tmp_path / "model_final.pt"
    ckpt.write_text("{}", encoding="utf-8")
    cmd = _build_eval_command(
        job_root,
        episodes=1,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=200,
        seed=1,
        policy="pi0",
        checkpoint=str(ckpt),
        eval_executor="joint_position",
        controller_type="JOINT_POSITION",
    )
    joined = " ".join(cmd).lower()
    assert "osc_pose" not in joined
    assert "ur5e" not in joined
