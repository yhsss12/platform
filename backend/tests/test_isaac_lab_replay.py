from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.services.isaac_lab.cli_runner import IsaacLabCliRunner
from app.services.isaac_lab import isaac_job_service as job_svc
from app.services.isaac_lab import replay_service as replay_svc
from app.services.isaac_lab.job_paths import (
    isaac_job_metadata_dir,
    isaac_job_replay_manifest_path,
    isaac_job_status_path,
    isaac_job_stdout_path,
    is_isaac_replay_job_id,
)
from app.services.isaac_lab.replay_cli import ReplayDemoCliParams, build_replay_demos_cli_args


def test_isaac_replay_job_id_pattern():
    assert is_isaac_replay_job_id("isaac_replay_20260614_120000_ab12")
    assert not is_isaac_replay_job_id("isaac_run_20260614_120000_ab12")


def test_build_replay_demos_cli_args(tmp_path: Path):
    dataset = tmp_path / "dataset.hdf5"
    dataset.write_bytes(b"hdf5")
    params = ReplayDemoCliParams(
        task_id="Isaac-Stack-Cube-Franka-IK-Rel-v0",
        dataset_file=dataset,
        headless=True,
        enable_cameras=True,
    )
    args = build_replay_demos_cli_args(params)
    assert "--task" in args
    assert "Isaac-Stack-Cube-Franka-IK-Rel-v0" in args
    assert "--dataset_file" in args
    assert str(dataset) in args
    assert "--headless" in args
    assert "--enable_cameras" in args
    assert "--video" not in args


def test_start_replay_demo_503_when_unconfigured(monkeypatch: pytest.MonkeyPatch):
    from app.core.config import settings

    settings.ISAACLAB_ROOT = None
    settings.ISAACLAB_RUNTIME_ENABLED = False

    with pytest.raises(HTTPException) as exc:
        replay_svc.start_replay_demo(dataset_file="/tmp/missing.hdf5")
    assert exc.value.status_code == 503


def test_start_replay_demo_400_when_dataset_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "IsaacLab"
    root.mkdir()
    sh = root / "isaaclab.sh"
    sh.write_text("#!/bin/bash\n", encoding="utf-8")
    sh.chmod(0o755)

    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True

    with pytest.raises(HTTPException) as exc:
        replay_svc.start_replay_demo(dataset_file=str(tmp_path / "missing.hdf5"))
    assert exc.value.status_code == 400


def test_replay_job_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "IsaacLab"
    root.mkdir()
    sh = root / "isaaclab.sh"
    sh.write_text("#!/bin/bash\n", encoding="utf-8")
    sh.chmod(0o755)
    dataset = tmp_path / "dataset.hdf5"
    dataset.write_bytes(b"hdf5")

    jobs_root = tmp_path / "jobs"
    monkeypatch.setenv("ISAACLAB_ROOT", str(root))
    monkeypatch.setenv("ISAACLAB_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("ISAACLAB_OUTPUT_ROOT", str(jobs_root))

    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True
    settings.ISAACLAB_OUTPUT_ROOT = str(jobs_root)
    settings.ISAACLAB_REPLAY_TIMEOUT = 30

    def _fake_run(self, script_relative, *args, stdout_path, stderr_path, timeout):
        stdout_path.write_text("Successfully replayed: 1/1\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        from app.services.isaac_lab.cli_runner import IsaacLabCliRunResult

        return IsaacLabCliRunResult(
            returncode=0,
            command=self.build_command(script_relative, *args),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    monkeypatch.setattr(IsaacLabCliRunner, "run_to_files", _fake_run)

    started = replay_svc.start_replay_demo(
        dataset_file=str(dataset),
        video=False,
    )
    job_id = started["jobId"]
    import time

    for _ in range(50):
        status = replay_svc.get_replay_job_status(job_id)
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert is_isaac_replay_job_id(job_id)
    assert isaac_job_status_path(job_id).is_file()
    assert isaac_job_stdout_path(job_id).is_file()
    assert (isaac_job_metadata_dir(job_id) / "request.json").is_file()
    assert isaac_job_replay_manifest_path(job_id).is_file()
    assert status["status"] == "completed"
    assert status.get("videoAvailable") is False

    tail = job_svc.read_job_log_tail(job_id, stream="stdout", lines=10)
    assert "Successfully replayed" in tail
