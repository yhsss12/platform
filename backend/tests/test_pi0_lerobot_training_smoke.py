from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.pi0_lerobot_smoke_runner import run_pi0_lerobot_training_smoke
from app.services.policy_schema_resolver import PI0_JOINT_SPACE_ENABLED

SMOKE_DATASET = (
    Path(__file__).resolve().parents[2]
    / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/lerobot_dataset"
)


@pytest.fixture()
def lerobot_fixture_dir() -> Path:
    if not SMOKE_DATASET.is_dir():
        pytest.skip("smoke LeRobot dataset not present")
    return SMOKE_DATASET


def test_pi0_lerobot_training_smoke(tmp_path: Path, lerobot_fixture_dir: Path):
    output_dir = tmp_path / "pi0_smoke"
    result = run_pi0_lerobot_training_smoke(
        dataset_path=lerobot_fixture_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=2,
        max_steps=5,
    )
    assert result["status"] == "completed", result.get("error")
    assert result["stepsCompleted"] == 5
    assert (output_dir / "logs" / "train.log").is_file()
    assert (output_dir / "artifacts" / "metrics.jsonl").is_file()
    assert (output_dir / "artifacts" / "smoke_result.json").is_file()
    checkpoint = output_dir / "checkpoints" / "pi0" / "checkpoints" / "model_final.pt"
    assert checkpoint.is_file()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["modelType"] == "pi0"
    assert payload["datasetFormat"] == "lerobot"
    assert payload["state_dim"] == 9
    assert payload["action_dim"] == 8
    assert payload["pi0ReadyData"] is True
    assert payload["task_instruction"] == "thread the cable through the pole"
    assert PI0_JOINT_SPACE_ENABLED is False
