from __future__ import annotations

import re
import secrets
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, status

from app.core.platform_paths import is_path_within, platform_paths

PROJECT_ROOT = platform_paths.project_root
EVAL_OUTPUT_ROOT = platform_paths.evaluation_jobs.parent

EVAL_JOB_ID_PATTERN = re.compile(r"^eval_\d{8}_\d{6}_[a-f0-9]{4}$")
ISAAC_EVAL_JOB_ID_PATTERN = re.compile(r"^isaac_eval_\d{8}_\d{6}_[a-f0-9]{4}$")
_CT_EVAL_JOB_SUFFIX = r"(?:\d{8}_\d{6}_[a-f0-9]{4}|[a-z0-9_]+)"
CT_EVAL_JOB_ID_PATTERN = re.compile(rf"^ct_eval_{_CT_EVAL_JOB_SUFFIX}$")
IMPORTED_EVAL_JOB_ID_PATTERN = re.compile(r"^eval_joint_dp_[a-z0-9_]+$")


def make_eval_job_id() -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{ts}_{suffix}"


def make_isaac_eval_job_id() -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"isaac_eval_{ts}_{suffix}"


def is_isaac_eval_job_id(eval_job_id: str) -> bool:
    return ISAAC_EVAL_JOB_ID_PATTERN.match((eval_job_id or "").strip()) is not None


def is_valid_eval_job_id_format(eval_job_id: str) -> bool:
    candidate = (eval_job_id or "").strip()
    return bool(
        EVAL_JOB_ID_PATTERN.match(candidate)
        or ISAAC_EVAL_JOB_ID_PATTERN.match(candidate)
        or CT_EVAL_JOB_ID_PATTERN.match(candidate)
        or IMPORTED_EVAL_JOB_ID_PATTERN.match(candidate)
    )


def validate_eval_job_id(eval_job_id: str) -> str:
    candidate = (eval_job_id or "").strip()
    if is_valid_eval_job_id_format(candidate):
        return candidate
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid eval job ID format",
    )


def _assert_eval_root(job_root: Path) -> Path:
    resolved = job_root.resolve()
    if not is_path_within(resolved, EVAL_OUTPUT_ROOT):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid evaluation job path",
        )
    return resolved


def eval_job_dir(eval_job_id: str) -> Path:
    validated = validate_eval_job_id(eval_job_id)
    from app.services.workspace_runtime_paths import resolve_eval_job_root

    resolved = resolve_eval_job_root(validated)
    if resolved is not None:
        return resolved
    if validated.startswith("ct_eval_"):
        from app.services.cable_threading_service import _job_dir

        cable_root = _job_dir(validated)
        if cable_root.is_dir():
            return cable_root
    return _assert_eval_root(EVAL_OUTPUT_ROOT / "jobs" / validated)


def prepare_eval_job_root(eval_job_id: str) -> Path:
    # Creation always targets the configured data root.
    validated = validate_eval_job_id(eval_job_id)
    job_root = _assert_eval_root(EVAL_OUTPUT_ROOT / "jobs" / validated)
    for sub in ("logs", "results", "videos", "frames", "artifacts", "metadata", "episodes"):
        (job_root / sub).mkdir(parents=True, exist_ok=True)
    return job_root


def resolve_eval_status_path(job_root: Path) -> Path:
    """Return existing status file path, or the default write target."""
    for candidate in (job_root / "status.json", job_root / "metadata" / "status.json"):
        if candidate.is_file():
            return candidate
    return job_root / "status.json"
