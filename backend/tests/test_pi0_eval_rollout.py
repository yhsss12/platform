from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MOCK_OPENPI_ROOT = FIXTURES / "mock_openpi"
CABLE_THREADING_ROOT = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"


class _MockCableEnv:
    horizon = 12

    def __init__(self) -> None:
        self._step = 0

    def reset(self):
        self._step = 0
        return {
            "agentview_image": np.zeros((32, 32, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        }

    def step(self, action):
        self._step += 1
        done = self._step >= 5
        return self.reset(), 0.0, done, {"phase": "policy"}


@pytest.fixture()
def pi0_eval_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("OPENPI_ROOT", str(MOCK_OPENPI_ROOT))
    monkeypatch.setenv("OPENPI_PYTHON", sys.executable)
    monkeypatch.delenv("OPENPI_INFER_SCRIPT", raising=False)
    return tmp_path


def _write_pi0_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "openpi_ckpt" / "step_000003"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "params").write_text("mock", encoding="utf-8")
    final_path = tmp_path / "model_final.pt"
    final_path.write_text(
        json.dumps(
            {
                "format": "openpi_orbax_v1",
                "backend": "pi0",
                "checkpointPath": str(ckpt_dir),
                "action_dim": 7,
                "action_horizon": 4,
                "camera_keys": ["agentview_image"],
                "low_dim_keys": ["robot0_eef_pos"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return final_path


def test_pi0_policy_adapter_rollout_one_episode(pi0_eval_env: Path):
    sys.path.insert(0, str(CABLE_THREADING_ROOT))
    from examples.cable_threading.pi0_lab.policy_runtime import Pi0PolicyAdapter

    checkpoint = _write_pi0_checkpoint(pi0_eval_env)
    policy = Pi0PolicyAdapter(checkpoint, device="cpu")
    env = _MockCableEnv()
    policy.reset()
    steps = 0
    obs = env.reset()
    done = False
    while not done and steps < env.horizon:
        action = policy.act(obs)
        obs, _reward, done, _info = env.step(action)
        steps += 1
    assert steps > 0


def test_pi0_policy_rejects_shim_checkpoint(pi0_eval_env: Path):
    sys.path.insert(0, str(CABLE_THREADING_ROOT))
    from examples.cable_threading.pi0_lab.policy_runtime import Pi0PolicyAdapter

    shim_path = pi0_eval_env / "shim.pt"
    shim_path.write_bytes(b"PI0_PLATFORM_SHIM_CHECKPOINT")
    with pytest.raises(RuntimeError, match="shim"):
        Pi0PolicyAdapter(shim_path, device="cpu")


def test_pi0_policy_missing_camera_obs_readable_error(pi0_eval_env: Path):
    sys.path.insert(0, str(CABLE_THREADING_ROOT))
    from examples.cable_threading.pi0_lab.policy_runtime import Pi0PolicyAdapter

    checkpoint = _write_pi0_checkpoint(pi0_eval_env)
    policy = Pi0PolicyAdapter(checkpoint, device="cpu")
    with pytest.raises(KeyError, match="camera obs"):
        policy.act({"robot0_eef_pos": np.zeros(3, dtype=np.float32)})
