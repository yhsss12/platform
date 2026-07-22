"""pi0 joint-space readiness tests (no training/eval runs)."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_BACKEND = Path(__file__).resolve().parents[1]
_CABLE_MVP = _BACKEND.parent / "integrations" / "CableThreadingMVP"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from app.services.policy_schema_resolver import (  # noqa: E402
    PI0_JOINT_SPACE_DISABLED_REASON,
    PI0_JOINT_SPACE_ENABLED,
    pi0_joint_space_capability,
    resolve_pi0_eval_executor,
)


def test_pi0_joint_config_template_marked_not_enabled():
    cfg_path = (
        _CABLE_MVP
        / "examples"
        / "cable_threading"
        / "pi0_configs"
        / "cable_threading_joint_obs_joint_action.yaml"
    )
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert data.get("not_enabled") is True or data.get("enabled") is False
    assert data.get("status") == "not_enabled"
    assert data["action_dim"] == 8
    assert data["controller_type"] == "JOINT_POSITION"
    assert data["eval_executor"] == "joint_position"
    assert "language_instruction" in data


def test_pi0_joint_space_capability_disabled():
    cap = pi0_joint_space_capability()
    assert cap["jointSpaceEnabled"] is False
    assert cap["reason"] == PI0_JOINT_SPACE_DISABLED_REASON
    assert cap["policySchemaId"] == "joint_state_obs_joint_action"


def test_pi0_eval_resolver_joint_for_lerobot_asset():
    spec = resolve_pi0_eval_executor(
        policy="pi0",
        model_asset={
            "modelType": "pi0",
            "datasetFormat": "lerobot",
            "stateDim": 9,
            "actionDim": 8,
            "robot": "Panda",
            "controllerType": "JOINT_POSITION",
            "actionMode": "joint_delta_derived",
            "lowDimKeys": ["robot0_joint_pos", "robot0_gripper_qpos"],
        },
    )
    assert spec.eval_executor == "joint_position"
    assert spec.source == "pi0_lerobot_joint_schema"


def test_pi0_eval_resolver_legacy_non_joint_still_osc():
    assert PI0_JOINT_SPACE_ENABLED is False
    spec = resolve_pi0_eval_executor(
        policy="pi0",
        model_asset={
            "modelType": "pi0",
            "actionDim": 7,
            "controllerType": "OSC_POSE",
        },
    )
    assert spec.eval_executor == "osc_pose"
    assert spec.source == "pi0_joint_space_not_enabled"


def test_pi0_training_runner_hdf5_not_supported_message():
    from app.services.pi0_training_runner import PI0_HDF5_NOT_SUPPORTED_MESSAGE

    assert "HDF5" in PI0_HDF5_NOT_SUPPORTED_MESSAGE or "LeRobot" in PI0_HDF5_NOT_SUPPORTED_MESSAGE
