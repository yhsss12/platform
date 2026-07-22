from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.pi0_training_runner import PI0_HDF5_NOT_SUPPORTED_MESSAGE

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MOCK_OPENPI_ROOT = FIXTURES / "mock_openpi"
RUNNER_SCRIPT = Path(__file__).resolve().parents[1] / "integrations" / "pi0_runner" / "run_openpi_train.py"


def _write_platform_config(tmp_path: Path, *, base_config: str = "pi0_mock", learning_rate: float = 1e-4) -> Path:
    config = {
        "platform": {"trainJobId": "train_test_job", "backend": "pi0"},
        "openpi_base_config": base_config,
        "openpi": {
            "exp_name": "train_test_job",
            "checkpoint_base_dir": str(tmp_path / "checkpoints" / "pi0"),
            "platform_final_checkpoint": str(tmp_path / "checkpoints" / "pi0" / "checkpoints" / "model_final.pt"),
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": learning_rate,
            "seed": 1,
            "num_train_steps": 3,
        },
        "dataset": {"hdf5_path": str(tmp_path / "dataset.hdf5")},
        "manifest": {"artifacts": {"hdf5": str(tmp_path / "dataset.hdf5")}},
        "paths": {
            "output_dir": str(tmp_path / "checkpoints" / "pi0"),
            "checkpoint_base_dir": str(tmp_path / "checkpoints" / "pi0"),
            "platform_final_checkpoint": str(tmp_path / "checkpoints" / "pi0" / "checkpoints" / "model_final.pt"),
            "metrics_path": str(tmp_path / "artifacts" / "metrics.jsonl"),
        },
    }
    path = tmp_path / "openpi_platform_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _run_bridge(
    tmp_path: Path,
    *,
    platform_config: Path,
    env: dict[str, str],
    train_mode: str = "openpi",
) -> subprocess.CompletedProcess[str]:
    out_dir = tmp_path / "checkpoints" / "pi0"
    metrics_path = tmp_path / "artifacts" / "metrics.jsonl"
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env["PI0_TRAIN_MODE"] = train_mode
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--platform-config",
            str(platform_config),
            "--dataset",
            str(tmp_path / "dataset.hdf5"),
            "--out-dir",
            str(out_dir),
            "--metrics-path",
            str(metrics_path),
        ],
        cwd=str(tmp_path),
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture()
def bridge_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_ROOT", str(MOCK_OPENPI_ROOT))
    monkeypatch.setenv("OPENPI_PYTHON", sys.executable)
    monkeypatch.setenv("OPENPI_BASE_CONFIG", "pi0_mock")
    monkeypatch.delenv("PI0_USE_PLATFORM_SHIM", raising=False)
    return tmp_path


def test_openpi_command_does_not_use_unsupported_flags(bridge_env: Path):
    platform_config = _write_platform_config(bridge_env)
    completed = _run_bridge(
        bridge_env,
        platform_config=platform_config,
        env={"OPENPI_BASE_CONFIG": "pi0_mock"},
    )
    output = completed.stdout + completed.stderr
    assert "--config-path" not in output
    assert "--learning-rate" not in output
    assert "--output-dir" not in output
    assert "Unrecognized options" not in output
    assert "--exp-name" in output
    assert "--checkpoint-base-dir" in output
    assert completed.returncode == 0


def test_openpi_success_materializes_final_checkpoint_and_metrics(bridge_env: Path):
    platform_config = _write_platform_config(bridge_env)
    completed = _run_bridge(bridge_env, platform_config=platform_config, env={})
    assert completed.returncode == 0, completed.stdout + completed.stderr

    final_path = bridge_env / "checkpoints" / "pi0" / "checkpoints" / "model_final.pt"
    assert final_path.is_file()
    assert final_path.stat().st_size > 0

    metrics_path = bridge_env / "artifacts" / "metrics.jsonl"
    assert metrics_path.is_file()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert rows[-1]["trainLoss"] > 0
    assert rows[-1]["totalSteps"] == 3


def test_openpi_nonzero_exit_code(bridge_env: Path, monkeypatch: pytest.MonkeyPatch):
    broken_root = bridge_env / "broken_openpi"
    scripts_dir = broken_root / "scripts"
    scripts_dir.mkdir(parents=True)
    broken_script = scripts_dir / "train.py"
    broken_script.write_text(
        "#!/usr/bin/env python3\nimport sys\nraise SystemExit(3)\n",
        encoding="utf-8",
    )
    platform_config = _write_platform_config(bridge_env)
    completed = _run_bridge(
        bridge_env,
        platform_config=platform_config,
        env={"OPENPI_ROOT": str(broken_root), "OPENPI_TRAIN_SCRIPT": str(broken_script)},
    )
    assert completed.returncode == 3


def test_openpi_zero_exit_without_checkpoint_fails(bridge_env: Path):
    noop_root = bridge_env / "noop_openpi"
    scripts_dir = noop_root / "scripts"
    scripts_dir.mkdir(parents=True)
    noop_script = scripts_dir / "train.py"
    noop_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    platform_config = _write_platform_config(bridge_env)
    completed = _run_bridge(
        bridge_env,
        platform_config=platform_config,
        env={"OPENPI_ROOT": str(noop_root), "OPENPI_TRAIN_SCRIPT": str(noop_script)},
    )
    assert completed.returncode == 1
    assert "Final checkpoint not found" in completed.stdout + completed.stderr


def test_hdf5_with_lerobot_config_rejected_before_openpi(bridge_env: Path):
    platform_config = _write_platform_config(bridge_env, base_config="pi05_libero")
    completed = _run_bridge(
        bridge_env,
        platform_config=platform_config,
        env={"OPENPI_BASE_CONFIG": "pi05_libero"},
    )
    assert completed.returncode == 2
    assert PI0_HDF5_NOT_SUPPORTED_MESSAGE in completed.stdout + completed.stderr
