"""Resolve workspace job runtime directories (including imported standalone pipelines)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from app.core.platform_paths import (
    is_path_within,
    platform_paths,
    resolve_runtime_reference,
)

PROJECT_ROOT = platform_paths.project_root
RUNTIME_ROOT = platform_paths.runs_root
TRAINING_JOBS_ROOT = RUNTIME_ROOT / "training" / "jobs"
EVAL_JOBS_ROOT = RUNTIME_ROOT / "evaluations" / "jobs"

SAFE_JOB_ID_PATTERN = re.compile(r"^[\w.-]+$")
IMPORTED_EVAL_JOB_PATTERN = re.compile(r"^eval_joint_dp_[a-z0-9_]+$")
IMPORTED_TRAIN_JOB_PATTERN = re.compile(r"^train_joint_dp_[a-z0-9_]+$")


def _safe_job_id(job_id: str) -> Optional[str]:
    candidate = (job_id or "").strip()
    if not candidate or not SAFE_JOB_ID_PATTERN.match(candidate):
        return None
    if ".." in candidate:
        return None
    return candidate


def resolve_workspace_job_runtime_path(
    job_id: str,
    *,
    job_type: Optional[str] = None,
) -> Optional[Path]:
    candidate = _safe_job_id(job_id)
    if not candidate:
        return None
    try:
        from app.core.database import SessionLocal
        from app.models.workspace_job import WorkspaceJob

        with SessionLocal() as db:
            query = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate)
            if job_type:
                query = query.filter(WorkspaceJob.job_type == job_type)
            row = query.one_or_none()
            if row is None or not row.runtime_path:
                return None
            path = resolve_runtime_reference(row.runtime_path)
            if not path.is_dir():
                return None
            if not is_path_within(path, RUNTIME_ROOT):
                return None
            return path
    except Exception:
        return None


def resolve_training_job_root(train_job_id: str) -> Optional[Path]:
    candidate = _safe_job_id(train_job_id)
    if not candidate:
        return None
    standard = TRAINING_JOBS_ROOT / candidate
    if standard.is_dir():
        return standard
    return resolve_workspace_job_runtime_path(candidate, job_type="training")


def resolve_eval_job_root(eval_job_id: str) -> Optional[Path]:
    candidate = _safe_job_id(eval_job_id)
    if not candidate:
        return None
    standard = EVAL_JOBS_ROOT / candidate
    if standard.is_dir():
        return standard
    cable = RUNTIME_ROOT / "cable_threading" / "jobs" / candidate
    if cable.is_dir() and candidate.startswith("ct_eval_"):
        return cable
    return resolve_workspace_job_runtime_path(candidate, job_type="evaluation")


def is_imported_workspace_eval_job_id(eval_job_id: str) -> bool:
    return IMPORTED_EVAL_JOB_PATTERN.match((eval_job_id or "").strip()) is not None


def is_imported_workspace_train_job_id(train_job_id: str) -> bool:
    return IMPORTED_TRAIN_JOB_PATTERN.match((train_job_id or "").strip()) is not None
