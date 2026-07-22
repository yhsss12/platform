#!/usr/bin/env python3
"""评测 running 任务 health 检查与 DB/runtime 状态自愈。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.database import SessionLocal
from app.models.workspace_job import WorkspaceJob
from app.services.evaluation.evaluation_runtime_health import (
    EVALUATION_RUNNING_MAX_AGE_SECONDS,
    inspect_evaluation_runtime_health,
    reconcile_evaluation_runtime_health,
)
from app.services.training_job_sync_service import sync_eval_job_from_runtime

RUNNING_STATUSES = frozenset({"running", "evaluating", "queued", "pending", "unknown"})


def _iso(dt) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _load_jobs(
    job_id: str | None,
    *,
    all_running: bool,
    older_than_hours: int | None,
) -> list[WorkspaceJob]:
    with SessionLocal() as db:
        query = db.query(WorkspaceJob).filter(
            WorkspaceJob.job_type == "evaluation",
            WorkspaceJob.status != "deleted",
        )
        if job_id:
            query = query.filter(WorkspaceJob.job_id == job_id)
        elif all_running:
            query = query.filter(WorkspaceJob.status.in_(sorted(RUNNING_STATUSES)))
        else:
            raise SystemExit("请指定 --job-id 或 --all-running")
        rows = query.order_by(WorkspaceJob.created_at.asc()).all()

    if older_than_hours is None:
        return rows

    cutoff = EVALUATION_RUNNING_MAX_AGE_SECONDS
    if older_than_hours > 0:
        cutoff = older_than_hours * 3600

    filtered: list[WorkspaceJob] = []
    now = datetime.now(timezone.utc).timestamp()
    for row in rows:
        created_ts = row.created_at.timestamp() if row.created_at else None
        started_ts = row.started_at.timestamp() if row.started_at else None
        start_ts = started_ts or created_ts
        if start_ts is None:
            continue
        if int(now - start_ts) >= cutoff:
            filtered.append(row)
    return filtered


def _print_report(row: WorkspaceJob, health: dict, *, apply: bool) -> bool:
    changed = bool(health.get("applied")) if apply else health.get("actualStatus") != str(row.status or "")
    print(
        json.dumps(
            {
                "jobId": row.job_id,
                "taskName": row.task_name,
                "declaredStatus": health.get("declaredStatus") or row.status,
                "actualStatus": health.get("actualStatus"),
                "createdAt": _iso(row.created_at),
                "lastRuntimeUpdateAt": health.get("lastRuntimeUpdateAt"),
                "isProcessAlive": health.get("isProcessAlive"),
                "jobAgeSeconds": health.get("jobAgeSeconds"),
                "reason": health.get("reason"),
                "wouldChange": changed,
                "applied": health.get("applied") if apply else False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("---")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile evaluation job runtime health")
    parser.add_argument("--job-id", action="append", dest="job_ids", default=[])
    parser.add_argument("--all-running", action="store_true")
    parser.add_argument(
        "--older-than-hours",
        type=int,
        default=None,
        help=f"仅处理创建/启动时间超过 N 小时的任务（默认与 --all-running 联用时为 {EVALUATION_RUNNING_MAX_AGE_SECONDS // 3600}）",
    )
    parser.add_argument("--apply", action="store_true", help="写回 DB/runtime；默认 dry-run")
    args = parser.parse_args()

    older_than_hours = args.older_than_hours

    if args.job_ids:
        rows: list[WorkspaceJob] = []
        for job_id in args.job_ids:
            rows.extend(_load_jobs(job_id, all_running=False, older_than_hours=older_than_hours))
    elif args.all_running:
        rows = _load_jobs(None, all_running=True, older_than_hours=older_than_hours)
    else:
        parser.error("请指定 --job-id 或 --all-running")

    if not rows:
        print("未找到匹配的评测任务")
        return 0

    print(f"mode={'apply' if args.apply else 'dry-run'} jobs={len(rows)}")
    changed = 0
    for row in rows:
        if args.apply:
            sync_eval_job_from_runtime(row.job_id)
            health = inspect_evaluation_runtime_health(
                row.job_id,
                row.runtime_path,
                declared_status=row.status,
                created_at=_iso(row.created_at),
                started_at=_iso(row.started_at),
            )
            health["applied"] = True
        else:
            health = reconcile_evaluation_runtime_health(
                row.job_id,
                row.runtime_path,
                declared_status=row.status,
                created_at=_iso(row.created_at),
                started_at=_iso(row.started_at),
                apply=False,
            )
        if _print_report(row, health, apply=args.apply):
            changed += 1

    if args.apply:
        print(f"applied={changed}")
    else:
        print("dry-run 完成；追加 --apply 写回状态")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
