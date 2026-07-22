from __future__ import annotations

from app.services.policy_schema_resolver import assess_pi0_lerobot_data_format_readiness


def test_pi0_data_format_ready_when_lerobot_pi0_ready():
    manifest = {
        "availableFormats": ["lerobot"],
        "primaryFormat": "lerobot",
        "lerobot": {
            "status": "ready",
            "robot": "Panda",
            "stateDim": 9,
            "actionDim": 8,
            "taskInstruction": "thread the cable through the pole",
            "pi0Ready": True,
            "pi0ReadyReason": "",
        },
    }
    result = assess_pi0_lerobot_data_format_readiness(manifest)
    assert result["data_format_ready"] is True
    assert result["pi0Ready"] is True


def test_pi0_data_format_not_ready_when_lerobot_only():
    manifest = {
        "availableFormats": ["lerobot"],
        "primaryFormat": "lerobot",
        "lerobot": {
            "status": "ready",
            "robot": "Panda",
            "stateDim": 10,
            "actionDim": 7,
            "taskInstruction": "thread the cable through the pole",
            "pi0Ready": False,
            "pi0ReadyReason": "action_dim is 7 / controller is OSC_POSE, not Panda JOINT_POSITION 8D",
        },
    }
    result = assess_pi0_lerobot_data_format_readiness(manifest)
    assert result["data_format_ready"] is False
    assert result["lerobotReady"] is True
    assert result["pi0Ready"] is False
    assert "action_dim" in result["reason"] or "OSC_POSE" in result["reason"]


def test_pi0_data_format_not_ready_without_lerobot():
    manifest = {"availableFormats": ["hdf5"], "primaryFormat": "hdf5"}
    result = assess_pi0_lerobot_data_format_readiness(manifest)
    assert result["data_format_ready"] is False
    assert "LeRobot" in result["reason"]
