"""Integration-style tests for pi0 eval smoke artifacts (no MuJoCo required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_CABLE_MVP = _BACKEND.parent / "integrations" / "CableThreadingMVP"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from app.services.policy_schema_resolver import PI0_JOINT_SPACE_ENABLED  # noqa: E402
from app.services.pi0_lerobot_smoke_runner import assess_pi0_lerobot_training_capability  # noqa: E402


def test_pi0_capability_updates_after_eval_rollout_flag():
    cap = assess_pi0_lerobot_training_capability(
        dataset_path=_BACKEND.parent
        / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/lerobot_dataset",
        smoke_success=True,
        platform_training_success=True,
        eval_rollout_success=True,
    )
    assert cap["eval_adapter_ready"] is True
    assert cap["joint_position_rollout_ready"] is True
    assert cap["pi0_joint_space_enabled"] is False
    assert PI0_JOINT_SPACE_ENABLED is False


def test_run_py_has_pi0_eval_args():
    run_py = (_CABLE_MVP / "run.py").read_text(encoding="utf-8")
    assert "--train-config" in run_py
    assert "--task-instruction" in run_py
    assert "resolve_pi0_eval_runtime" in run_py


def test_aggregate_result_schema_example(tmp_path: Path):
    aggregate = {
        "policyType": "pi0",
        "modelType": "pi0",
        "evalExecutor": "joint_position",
        "robot": "Panda",
        "controllerType": "JOINT_POSITION",
        "stateDim": 9,
        "actionDim": 8,
        "taskInstruction": "thread the cable through the pole",
        "episodes": 1,
        "horizon": 200,
        "success_rate": 0.0,
        "ever_success_rate": 0.0,
        "rollout_ok": True,
    }
    path = tmp_path / "aggregate_result.json"
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["rollout_ok"] is True
    assert loaded["actionDim"] == 8
