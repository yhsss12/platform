#!/usr/bin/env python3
"""Hard-purge training jobs that were soft-deleted (status='deleted').

Safe by default: dry-run unless --apply is passed.

What it removes for each matching workspace_jobs row:
  - workspace_artifacts rows
  - model_assets / training_metric_summary rows (via delete_workspace_job_async)
  - runs directory (when path is safe and exists)
  - the workspace_jobs row itself

Usage:
  cd backend && python tools/maintenance/purge_soft_deleted_training_jobs.py
  cd backend && python tools/maintenance/purge_soft_deleted_training_jobs.py --apply
  cd backend && python tools/maintenance/purge_soft_deleted_training_jobs.py --apply --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _list_soft_deleted_training_jobs(*, limit: int | None = None) -> list[dict[str, Any]]:
    from app.core.database import SessionLocal
    from app.models.workspace_job import WorkspaceJob

    with SessionLocal() as db:
        query = (
            db.query(WorkspaceJob)
            .filter(
                WorkspaceJob.job_type == "training",
                WorkspaceJob.status == "deleted",
            )
            .order_by(WorkspaceJob.updated_at.desc(), WorkspaceJob.id.desc())
        )
        if limit is not None and limit > 0:
            query = query.limit(limit)
        rows = query.all()
        return [
            {
                "jobId": row.job_id,
                "runtimePath": row.runtime_path or "",
                "taskType": row.task_type,
                "updatedAt": row.updated_at.isoformat() if row.updated_at else None,
                "runtimeExists": bool(row.runtime_path and Path(row.runtime_path).is_dir()),
            }
            for row in rows
        ]


async def _purge_one(job_id: str) -> dict[str, Any]:
    from app.db.session import AsyncSessionLocal
    from app.services.workspace_job_service import (
        RuntimeDeleteFailedError,
        WorkspaceJobDeleteError,
        delete_workspace_job_async,
    )

    async with AsyncSessionLocal() as db:
        try:
            result = await delete_workspace_job_async(db, job_id)
            if result is None:
                return {"jobId": job_id, "ok": False, "error": "not_found"}
            await db.commit()
            return {"jobId": job_id, "ok": True, **result}
        except (WorkspaceJobDeleteError, RuntimeDeleteFailedError) as exc:
            await db.rollback()
            return {"jobId": job_id, "ok": False, "error": str(exc)}
        except Exception as exc:
            await db.rollback()
            return {"jobId": job_id, "ok": False, "error": str(exc)}


async def _purge_all(job_ids: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for job_id in job_ids:
        results.append(await _purge_one(job_id))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hard-purge soft-deleted training jobs (status=deleted)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete DB rows and runtime directories (default is dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the N most recently updated soft-deleted training jobs",
    )
    parser.add_argument(
        "--only-with-runtime",
        action="store_true",
        help="Only target jobs whose runtime_path directory still exists",
    )
    args = parser.parse_args()

    candidates = _list_soft_deleted_training_jobs(limit=args.limit)
    if args.only_with_runtime:
        candidates = [row for row in candidates if row["runtimeExists"]]

    summary: dict[str, Any] = {
        "dryRun": not args.apply,
        "candidateCount": len(candidates),
        "withRuntimeDir": sum(1 for row in candidates if row["runtimeExists"]),
        "candidates": candidates,
    }

    if not args.apply:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        print(
            "\nDry-run only. Re-run with --apply to hard-delete these training jobs.",
            file=sys.stderr,
        )
        return 0

    results = asyncio.run(_purge_all([row["jobId"] for row in candidates]))
    ok = [row for row in results if row.get("ok")]
    failed = [row for row in results if not row.get("ok")]
    summary["applied"] = {
        "ok": len(ok),
        "failed": len(failed),
        "runtimeDeleted": sum(1 for row in ok if row.get("runtimeDeleted")),
        "deletedArtifacts": sum(int(row.get("deletedArtifacts") or 0) for row in ok),
        "deletedModelAssets": sum(int(row.get("deletedModelAssets") or 0) for row in ok),
        "failures": failed,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
