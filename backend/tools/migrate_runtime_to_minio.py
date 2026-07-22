#!/usr/bin/env python3
"""历史数据零丢失迁移：runtime_outputs → MinIO → PostgreSQL 索引。

特性：
- 幂等（artifact_storage_objects sha256 + status）
- 可中断恢复（progress JSONL checkpoint）
- hash 校验
- dry-run 模式

用法：
  cd backend && python tools/migrate_runtime_to_minio.py --dry-run
  cd backend && python tools/migrate_runtime_to_minio.py --limit 50
  cd backend && python tools/migrate_runtime_to_minio.py --job-id train_20260101_120000_abcd
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.artifact_storage_registry import file_digest, get_artifact_record, should_skip_upload
from app.services.artifact_upload_service import (
    artifact_upload_enabled,
    batch_upload_jobs,
    discover_runtime_job_ids,
    process_job_artifact_upload,
)
from app.services.storage.storage_service import StorageService
from app.services.workspace_job_service import sync_workspace_job_from_runtime

logger = logging.getLogger(__name__)

DEFAULT_PROGRESS_PATH = BACKEND_ROOT.parent / "runtime_outputs" / ".migration" / "state.jsonl"
LEGACY_PROGRESS_PATH = BACKEND_ROOT.parent / "runtime_outputs" / ".migration" / "migrate_runtime_to_minio.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_completed(progress_path: Path) -> set[str]:
    done: set[str] = set()
    paths = [progress_path]
    if progress_path == DEFAULT_PROGRESS_PATH and LEGACY_PROGRESS_PATH.is_file():
        paths.append(LEGACY_PROGRESS_PATH)
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "completed" and row.get("jobId"):
                done.add(str(row["jobId"]))
    return done


def _append_progress(progress_path: Path, payload: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _scan_orphan_files(runtime_root: Path) -> list[Path]:
    """发现尚未登记、可能需要迁移的本地大文件。"""
    patterns = (
        "**/checkpoints/**/*.ckpt",
        "**/checkpoints/**/*.pt",
        "**/checkpoints/**/*.pth",
        "**/datasets/dataset.hdf5",
        "**/datasets/dataset.npz",
        "**/datasets/dataset.mcap",
        "**/results/aggregate_result.json",
        "**/results/eval.results.json",
        "**/videos/*.mp4",
    )
    found: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in runtime_root.glob(pattern):
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(path.resolve())
    return found


def migrate_jobs(
    *,
    job_ids: list[str],
    dry_run: bool = False,
    progress_path: Path = DEFAULT_PROGRESS_PATH,
    sync_first: bool = True,
) -> dict[str, Any]:
    completed = _load_completed(progress_path)
    summary: dict[str, Any] = {
        "dryRun": dry_run,
        "enabled": artifact_upload_enabled(),
        "total": len(job_ids),
        "skipped": 0,
        "processed": 0,
        "uploaded": 0,
        "errors": [],
    }

    if not artifact_upload_enabled() and not dry_run:
        summary["errors"].append("MinIO not configured; set MINIO_ENDPOINT or use --dry-run")
        return summary

    for job_id in job_ids:
        if job_id in completed:
            summary["skipped"] += 1
            continue
        try:
            if sync_first and not dry_run:
                sync_workspace_job_from_runtime(job_id)
            if dry_run:
                result = {"jobId": job_id, "uploaded": 0, "dryRun": True}
            else:
                result = process_job_artifact_upload(job_id)
            summary["processed"] += 1
            summary["uploaded"] += int(result.get("uploaded", 0) or 0)
            if not dry_run:
                _append_progress(
                    progress_path,
                    {
                        "jobId": job_id,
                        "status": "completed",
                        "result": result,
                        "at": _utc_now_iso(),
                    },
                )
        except Exception as exc:
            logger.warning("migrate job failed job_id=%s: %s", job_id, exc)
            summary["errors"].append(f"{job_id}: {exc}")
            if not dry_run:
                _append_progress(
                    progress_path,
                    {"jobId": job_id, "status": "failed", "error": str(exc), "at": _utc_now_iso()},
                )
    return summary


def verify_file_hashes(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        sha256, size_bytes = file_digest(path)
        rows.append({"path": str(path), "sha256": sha256, "sizeBytes": size_bytes})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate runtime_outputs artifacts to MinIO")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, no upload")
    parser.add_argument("--limit", type=int, default=100, help="Max jobs to process")
    parser.add_argument("--job-id", action="append", default=[], help="Specific job ID(s)")
    parser.add_argument("--include-non-terminal", action="store_true", help="Include running jobs")
    parser.add_argument("--no-sync", action="store_true", help="Skip PG sync before upload")
    parser.add_argument("--progress", type=str, default=str(DEFAULT_PROGRESS_PATH), help="Progress JSONL path")
    parser.add_argument("--scan-orphans", action="store_true", help="List orphan files under runtime_outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runtime_root = BACKEND_ROOT.parent / "runtime_outputs"
    progress_path = Path(args.progress)

    if args.scan_orphans:
        orphans = _scan_orphan_files(runtime_root)
        print(json.dumps({"orphans": len(orphans), "files": [str(p) for p in orphans[:200]]}, ensure_ascii=False, indent=2))
        return

    if args.job_id:
        job_ids = [jid.strip() for jid in args.job_id if jid.strip()]
    else:
        job_ids = discover_runtime_job_ids(include_non_terminal=args.include_non_terminal, limit=args.limit)

    logger.info(
        "migration start jobs=%s dry_run=%s minio=%s progress=%s",
        len(job_ids),
        args.dry_run,
        StorageService.is_remote_storage_enabled(),
        progress_path,
    )
    summary = migrate_jobs(
        job_ids=job_ids,
        dry_run=args.dry_run,
        progress_path=progress_path,
        sync_first=not args.no_sync,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
