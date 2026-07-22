#!/usr/bin/env python3
"""CLI for workspace runtime reindex / backfill."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Reindex runs into workspace DB indexes")
    parser.add_argument(
        "--job-type",
        choices=["all", "data_generation", "generate", "training", "evaluation"],
        default="all",
    )
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--restore-deleted", action="store_true")
    parser.add_argument(
        "--purge-deleted-train-assets",
        action="store_true",
        help="Hard-delete model_assets revived from deleted training jobs",
    )
    parser.add_argument(
        "--import-joint-dp-pipeline",
        action="store_true",
        help="Import standalone joint-space DP full pipeline only",
    )
    args = parser.parse_args()

    if args.purge_deleted_train_assets:
        from app.services.model_asset_cleanup_service import (
            purge_model_assets_on_deleted_train_jobs,
            purge_soft_deleted_model_assets,
        )

        r1 = purge_model_assets_on_deleted_train_jobs(dry_run=args.dry_run)
        r2 = purge_soft_deleted_model_assets(dry_run=args.dry_run)
        print(json.dumps({"deletedTrainJobAssets": r1, "purgedSoftDeleted": r2}, indent=2, ensure_ascii=False))
        return 0

    if args.import_joint_dp_pipeline:
        from app.services.workspace_joint_dp_import_service import import_joint_dp_full_pipeline

        result = import_joint_dp_full_pipeline(dry_run=args.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0 if not result.get("errors") else 1

    from app.services.training_job_sync_service import reindex_runtime_jobs

    job_type = None if args.job_type == "all" else args.job_type
    result = reindex_runtime_jobs(
        task_type=args.task_type,
        job_type=job_type,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        restore_deleted=args.restore_deleted,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if not result.get("errors") and not result.get("syncErrors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
