from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.services.isaac_lab import isaac_job_service as job_svc
from app.services.isaac_lab import smoke_test_service as smoke_svc
from app.services.isaac_lab.cli_runner import IsaacLabCliRunner
from app.services.isaac_lab.isaac_runtime_service import get_runtime_status
from app.services.isaac_lab.job_paths import (
    isaac_job_status_path,
    isaac_job_stderr_path,
    isaac_job_stdout_path,
    is_isaac_run_job_id,
)


def test_isaac_run_job_id_pattern():
    assert is_isaac_run_job_id("isaac_run_20260614_120000_ab12")
    assert not is_isaac_run_job_id("isaac_eval_20260614_120000_ab12")


def test_start_smoke_test_returns_503_when_unconfigured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ISAACLAB_ROOT", "")
    monkeypatch.setenv("ISAACLAB_RUNTIME_ENABLED", "false")
    from app.core.config import settings

    settings.ISAACLAB_ROOT = None
    settings.ISAACLAB_RUNTIME_ENABLED = False

    with pytest.raises(HTTPException) as exc:
        smoke_svc.start_smoke_test()
    assert exc.value.status_code == 503


def test_cli_runner_build_command(tmp_path: Path):
    sh = tmp_path / "isaaclab.sh"
    sh.write_text("#!/bin/bash\n", encoding="utf-8")
    sh.chmod(0o755)
    runner = IsaacLabCliRunner(root=tmp_path, sh_path=sh)
    cmd = runner.build_command("scripts/environments/list_envs.py", "--keyword", "Stack")
    assert cmd[0] == str(sh)
    assert cmd[1:4] == ["-p", "scripts/environments/list_envs.py", "--keyword"]
    assert cmd[4] == "Stack"


def test_smoke_test_job_writes_logs_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "IsaacLab"
    root.mkdir()
    sh = root / "isaaclab.sh"
    sh.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    sh.chmod(0o755)
    (root / "source" / "isaaclab_tasks").mkdir(parents=True)
    (root / "VERSION").write_text("2.3.2\n", encoding="utf-8")

    jobs_root = tmp_path / "jobs"
    monkeypatch.setenv("ISAACLAB_ROOT", str(root))
    monkeypatch.setenv("ISAACLAB_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("ISAACLAB_OUTPUT_ROOT", str(jobs_root))

    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True
    settings.ISAACLAB_OUTPUT_ROOT = str(jobs_root)
    settings.ISAACLAB_SMOKE_TEST_TIMEOUT = 30

    def _fake_run(self, script_relative, *args, stdout_path, stderr_path, timeout):
        stdout_path.write_text("Isaac-Stack-Cube-Franka-v0\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        from app.services.isaac_lab.cli_runner import IsaacLabCliRunResult

        return IsaacLabCliRunResult(
            returncode=0,
            command=self.build_command(script_relative, *args),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    monkeypatch.setattr(IsaacLabCliRunner, "run_to_files", _fake_run)

    started = smoke_svc.start_smoke_test("Stack")
    job_id = started["jobId"]
    import time

    for _ in range(50):
        status = smoke_svc.get_smoke_test_job_status(job_id)
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert is_isaac_run_job_id(job_id)
    assert isaac_job_status_path(job_id).is_file()
    assert isaac_job_stdout_path(job_id).is_file()
    assert isaac_job_stderr_path(job_id).is_file()
    assert status["status"] == "completed"
    assert status.get("stackEnvMatches", 0) >= 1


def test_runtime_status_reports_version_when_root_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "IsaacLab"
    root.mkdir()
    sh = root / "isaaclab.sh"
    sh.write_text("#!/bin/bash\n", encoding="utf-8")
    sh.chmod(0o755)
    (root / "source" / "isaaclab_tasks").mkdir(parents=True)
    task_file = root / "source" / "isaaclab_tasks" / "stack.py"
    task_file.write_text('TASK = "Isaac-Stack-Cube-Franka-IK-Rel-v0"\n', encoding="utf-8")
    (root / "VERSION").write_text("2.3.2\n", encoding="utf-8")

    monkeypatch.setenv("ISAACLAB_ROOT", str(root))
    monkeypatch.setenv("ISAACLAB_RUNTIME_ENABLED", "false")

    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = False
    settings.ISAACLAB_DEFAULT_TASK = "Isaac-Stack-Cube-Franka-IK-Rel-v0"

    status = get_runtime_status()
    assert status["configured"] is True
    assert status["isaacLabVersion"] == "2.3.2"
    assert status["taskRegistered"] is True
    assert status["available"] is False
