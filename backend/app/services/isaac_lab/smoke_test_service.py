"""Isaac Lab smoke test job（list_envs --keyword Stack）。"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings
from app.services.isaac_lab.cli_runner import IsaacLabCliRunner
from app.services.isaac_lab.isaac_job_utils import (
    finalize_status,
    make_isaac_job_id,
    read_json,
    utc_now_iso,
    write_json,
)
from app.services.isaac_lab.isaac_runtime_service import (
    RUNTIME_NOT_CONFIGURED_MSG,
    assert_runtime_configured_for_commands,
)
from app.services.isaac_lab.job_paths import (
    isaac_job_metadata_dir,
    isaac_job_root,
    isaac_job_status_path,
    isaac_job_stderr_path,
    isaac_job_stdout_path,
    is_isaac_run_job_id,
)

logger = logging.getLogger(__name__)

SMOKE_TEST_SCRIPT = "scripts/environments/list_envs.py"
DEFAULT_SMOKE_KEYWORD = "Stack"

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_JOBS: set[str] = set()


def make_isaac_run_job_id() -> str:
    return make_isaac_job_id("isaac_run")


def _execute_smoke_test_job(job_id: str, keyword: str) -> None:
    runner = IsaacLabCliRunner.from_settings()
    stdout_path = isaac_job_stdout_path(job_id)
    stderr_path = isaac_job_stderr_path(job_id)
    timeout = int(getattr(settings, "ISAACLAB_SMOKE_TEST_TIMEOUT", 900) or 900)

    # 不传 --keyword：Isaac Lab list_envs 在 AppLauncher 启动后不会清理 sys.argv，
    # --keyword 会导致 SimulationApp 异常退出且不打印环境表（exit 0 但 stdout 为空）。
    cmd = runner.build_command(SMOKE_TEST_SCRIPT)
    finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "smoke_test",
            "status": "running",
            "phase": "cli",
            "command": cmd,
            "keyword": keyword,
            "startedAt": read_json(isaac_job_status_path(job_id)).get("startedAt"),
            "message": "Running Isaac Lab CLI smoke test…",
        },
    )

    try:
        result = runner.run_to_files(
            SMOKE_TEST_SCRIPT,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
        )
        success = result.returncode == 0 and not result.timed_out
        stdout_text = ""
        try:
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        kw = (keyword or DEFAULT_SMOKE_KEYWORD).strip() or DEFAULT_SMOKE_KEYWORD
        stack_lines = [line for line in stdout_text.splitlines() if kw in line]
        finalize_status(
            job_id,
            {
                "jobId": job_id,
                "kind": "smoke_test",
                "status": "completed" if success else "failed",
                "phase": "done",
                "command": result.command,
                "keyword": keyword,
                "startedAt": read_json(isaac_job_status_path(job_id)).get("startedAt"),
                "finishedAt": utc_now_iso(),
                "exitCode": result.returncode,
                "timedOut": result.timed_out,
                "stackEnvMatches": len(stack_lines),
                "message": (
                    "Smoke test completed"
                    if success
                    else (
                        f"Smoke test timed out after {timeout}s"
                        if result.timed_out
                        else f"Smoke test failed with exit code {result.returncode}"
                    )
                ),
                "paths": {
                    "jobRoot": str(isaac_job_root(job_id)),
                    "stdoutLog": str(stdout_path),
                    "stderrLog": str(stderr_path),
                    "statusJson": str(isaac_job_status_path(job_id)),
                },
            },
        )
    except Exception as exc:
        logger.exception("Isaac Lab smoke test job %s failed", job_id)
        finalize_status(
            job_id,
            {
                "jobId": job_id,
                "kind": "smoke_test",
                "status": "failed",
                "phase": "error",
                "keyword": keyword,
                "startedAt": read_json(isaac_job_status_path(job_id)).get("startedAt"),
                "finishedAt": utc_now_iso(),
                "message": str(exc),
                "paths": {
                    "jobRoot": str(isaac_job_root(job_id)),
                    "stdoutLog": str(stdout_path),
                    "stderrLog": str(stderr_path),
                    "statusJson": str(isaac_job_status_path(job_id)),
                },
            },
        )
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_JOBS.discard(job_id)


def start_smoke_test(keyword: str = DEFAULT_SMOKE_KEYWORD) -> dict[str, Any]:
    assert_runtime_configured_for_commands()

    runner = IsaacLabCliRunner.from_settings()
    if not runner.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=RUNTIME_NOT_CONFIGURED_MSG,
        )

    kw = (keyword or DEFAULT_SMOKE_KEYWORD).strip() or DEFAULT_SMOKE_KEYWORD
    job_id = make_isaac_run_job_id()
    job_root = isaac_job_root(job_id)
    job_root.mkdir(parents=True, exist_ok=True)
    meta_dir = isaac_job_metadata_dir(job_id)
    meta_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        meta_dir / "request.json",
        {
            "kind": "smoke_test",
            "keyword": kw,
            "script": SMOKE_TEST_SCRIPT,
            "submittedAt": utc_now_iso(),
        },
    )

    started_at = utc_now_iso()
    status_payload = finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "smoke_test",
            "status": "queued",
            "phase": "queued",
            "keyword": kw,
            "command": runner.build_command(SMOKE_TEST_SCRIPT),
            "startedAt": started_at,
            "message": "Smoke test queued",
            "paths": {
                "jobRoot": str(job_root),
                "stdoutLog": str(isaac_job_stdout_path(job_id)),
                "stderrLog": str(isaac_job_stderr_path(job_id)),
                "statusJson": str(isaac_job_status_path(job_id)),
            },
        },
    )

    with _ACTIVE_LOCK:
        _ACTIVE_JOBS.add(job_id)

    thread = threading.Thread(
        target=_execute_smoke_test_job,
        args=(job_id, kw),
        name=f"isaac-smoke-{job_id}",
        daemon=True,
    )
    thread.start()

    return {
        "jobId": job_id,
        "kind": "smoke_test",
        "status": status_payload.get("status", "queued"),
        "runtimePath": str(job_root),
        "statusUrl": f"/api/workspace/isaac-lab/jobs/{job_id}/status",
        "logPaths": {
            "stdout": str(isaac_job_stdout_path(job_id)),
            "stderr": str(isaac_job_stderr_path(job_id)),
        },
    }


def get_smoke_test_job_status(job_id: str) -> dict[str, Any]:
    if not is_isaac_run_job_id(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Isaac Lab run job ID format",
        )
    job_root = isaac_job_root(job_id)
    if not job_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Isaac Lab run job not found",
        )
    payload = read_json(isaac_job_status_path(job_id))
    if not payload:
        payload = {
            "jobId": job_id,
            "status": "unknown",
            "message": "status.json missing",
        }
    payload.setdefault("jobId", job_id)
    payload.setdefault(
        "paths",
        {
            "jobRoot": str(job_root),
            "stdoutLog": str(isaac_job_stdout_path(job_id)),
            "stderrLog": str(isaac_job_stderr_path(job_id)),
            "statusJson": str(isaac_job_status_path(job_id)),
        },
    )
    return payload
