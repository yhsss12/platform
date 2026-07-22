"""Tests for pi0 policy runtime schema (Phase F)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))


def _write_smoke_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
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
                "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
                "field_mapping": {"third_wrist_padding": True},
            }
        ),
        encoding="utf-8",
    )
    cfg.write_text(
        json.dumps(
            {
                "taskInstruction": "thread the cable through the pole",
                "stateDim": 9,
                "actionDim": 8,
            }
        ),
        encoding="utf-8",
    )
    return ckpt, cfg


def _joint_obs() -> dict:
    return {
        "agentview_image": np.zeros((32, 32, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((32, 32, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def test_pi0_policy_runs_8d_action(tmp_path: Path):
    from examples.cable_threading.pi0_lab.policy_runtime import Pi0PolicyAdapter

    ckpt, cfg = _write_smoke_checkpoint(tmp_path)
    policy = Pi0PolicyAdapter(ckpt, device="cpu", train_config_path=cfg)
    action = policy.predict(_joint_obs())
    assert action.shape == (8,)


def test_pi0_policy_runtime_builds_9d_state(tmp_path: Path):
    from examples.cable_threading.pi0_lab.pi0_smoke_inference import build_pi0_state_vector

    state = build_pi0_state_vector(_joint_obs(), state_dim=9)
    assert state.shape == (9,)


def test_pi0_policy_runtime_requires_task_instruction(tmp_path: Path):
    from examples.cable_threading.pi0_lab.policy_runtime import Pi0PolicyAdapter

    ckpt, cfg = _write_smoke_checkpoint(tmp_path)
    payload = json.loads(ckpt.read_text(encoding="utf-8"))
    payload.pop("task_instruction")
    ckpt.write_text(json.dumps(payload), encoding="utf-8")
    cfg.write_text("{}", encoding="utf-8")
    policy = Pi0PolicyAdapter(ckpt, device="cpu", train_config_path=cfg)
    with pytest.raises(ValueError, match="taskInstruction"):
        policy.predict(_joint_obs())


def test_pi0_policy_runtime_rejects_non_8d_checkpoint(tmp_path: Path):
    from examples.cable_threading.pi0_lab.policy_runtime import Pi0PolicyAdapter

    ckpt, cfg = _write_smoke_checkpoint(tmp_path)
    payload = json.loads(ckpt.read_text(encoding="utf-8"))
    payload["action_dim"] = 7
    ckpt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="action_dim"):
        Pi0PolicyAdapter(ckpt, device="cpu", train_config_path=cfg)


def test_pi0_eval_runtime_module_joint_position(tmp_path: Path):
    from examples.cable_threading.pi0_eval_runtime import resolve_pi0_eval_runtime

    ckpt, cfg = _write_smoke_checkpoint(tmp_path)
    runtime = resolve_pi0_eval_runtime(
        policy="pi0",
        checkpoint_path=ckpt,
        train_config_path=cfg,
        eval_executor="joint_position",
        robot="Panda",
    )
    assert runtime["evalExecutor"] == "joint_position"
    assert runtime["actionDim"] == 8
    assert runtime["stateDim"] == 9
