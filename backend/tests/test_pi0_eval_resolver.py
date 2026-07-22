"""Tests for pi0 eval resolver (Phase F)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.services.policy_schema_resolver import (  # noqa: E402
    PI0_JOINT_SPACE_ENABLED,
    is_pi0_joint_space_eval_asset,
    resolve_eval_robot_for_policy,
    resolve_pi0_eval_disabled_reason,
    resolve_pi0_eval_executor,
    resolve_pi0_eval_runtime,
)


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
        "imageKeys": ["agentview_image", "robot0_eye_in_hand_image"],
        "lowDimKeys": ["robot0_joint_pos", "robot0_gripper_qpos"],
    }


def test_pi0_eval_resolver_returns_joint_position_for_lerobot_asset():
    spec = resolve_pi0_eval_executor(policy="pi0", model_asset=_joint_asset())
    assert spec.eval_executor == "joint_position"
    assert spec.controller_type == "JOINT_POSITION"
    assert spec.source == "pi0_lerobot_joint_schema"


def test_pi0_eval_resolver_selects_panda():
    robot, _warnings = resolve_eval_robot_for_policy(
        policy="pi0",
        model_asset=_joint_asset(),
        eval_executor="joint_position",
        controller_type="JOINT_POSITION",
        action_mode="joint_delta_derived",
    )
    assert robot == "Panda"


def test_pi0_eval_resolver_does_not_fallback_osc_for_joint_asset(tmp_path: Path):
    assert PI0_JOINT_SPACE_ENABLED is False
    spec = resolve_pi0_eval_executor(policy="pi0", model_asset=_joint_asset())
    assert spec.eval_executor == "joint_position"
    assert spec.eval_executor != "osc_pose"


def test_pi0_eval_resolver_rejects_non_panda_robot():
    asset = _joint_asset()
    asset["robot"] = "UR5e"
    assert is_pi0_joint_space_eval_asset(asset) is False


def test_pi0_eval_runtime_requires_task_instruction(tmp_path: Path):
    ckpt = tmp_path / "model_final.pt"
    cfg = tmp_path / "train_config.json"
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
            }
        ),
        encoding="utf-8",
    )
    cfg.write_text(json.dumps({"modelType": "pi0", "stateDim": 9, "actionDim": 8}), encoding="utf-8")
    try:
        resolve_pi0_eval_runtime(
            policy="pi0",
            model_asset={
                "modelType": "pi0",
                "datasetFormat": "lerobot",
                "stateDim": 9,
                "actionDim": 8,
                "robot": "Panda",
                "controllerType": "JOINT_POSITION",
            },
            checkpoint_path=ckpt,
            train_config_path=cfg,
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "taskInstruction" in str(exc)
    assert raised


def test_pi0_eval_runtime_payload(tmp_path: Path):
    ckpt = tmp_path / "model_final.pt"
    cfg = tmp_path / "train_config.json"
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
    cfg.write_text(
        json.dumps({"taskInstruction": "thread the cable through the pole", "stateDim": 9, "actionDim": 8}),
        encoding="utf-8",
    )
    runtime = resolve_pi0_eval_runtime(
        policy="pi0",
        model_asset=_joint_asset(),
        checkpoint_path=ckpt,
        train_config_path=cfg,
    )
    assert runtime["evalExecutor"] == "joint_position"
    assert runtime["robot"] == "Panda"
    assert runtime["stateDim"] == 9
    assert runtime["actionDim"] == 8
    assert runtime["policyRuntime"] == "pi0"


def test_pi0_eval_disabled_reason_before_platform_enable(monkeypatch):
    import app.services.policy_schema_resolver as psr

    monkeypatch.setattr(psr, "pi0_platform_eval_ready", lambda: False)
    reason = resolve_pi0_eval_disabled_reason(eval_adapter_ready=False)
    assert reason == "pi0 eval adapter not ready"
    reason_ready = resolve_pi0_eval_disabled_reason(eval_adapter_ready=True)
    assert reason_ready == "pi0 platform evaluation not enabled"
