"""异步产物上传 worker（standalone 可独立运行）。

扫描 runs → 检测新产物 → 上传 MinIO → 更新 PostgreSQL 索引。

用法：
  cd backend && python -m app.workers.artifact_upload_worker --once
  cd backend && python -m app.workers.artifact_upload_worker --batch train_xxx eval_xxx
  cd backend && python -m app.workers.artifact_upload_worker
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Optional

from app.core.config import settings
from app.services.artifact_upload_service import (
    artifact_upload_enabled,
    batch_upload_jobs,
    run_upload_cycle,
)

logger = logging.getLogger(__name__)


def run_once(*, scan_limit: int = 20, pending_limit: int = 20) -> dict[str, Any]:
    return run_upload_cycle(scan_limit=scan_limit, pending_limit=pending_limit)


def run_loop(*, interval_sec: Optional[int] = None) -> None:
    interval = int(interval_sec or getattr(settings, "ARTIFACT_UPLOAD_WORKER_INTERVAL_SEC", 120) or 120)
    scan_limit = int(getattr(settings, "ARTIFACT_UPLOAD_SCAN_LIMIT", 20) or 20)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("artifact upload worker started interval=%ss scan_limit=%s", interval, scan_limit)
    while True:
        try:
            summary = run_once(scan_limit=scan_limit)
            logger.info("artifact upload worker cycle done: %s", summary)
        except Exception as exc:
            logger.exception("artifact upload worker cycle error: %s", exc)
        time.sleep(max(5, interval))


def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact upload worker (runs → MinIO)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=int, default=None, help="Loop interval seconds")
    parser.add_argument("--batch", nargs="*", help="Upload specific job IDs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.batch and not artifact_upload_enabled():
        logger.warning("MinIO not configured; worker will no-op uploads (file:// fallback only)")
    if args.batch:
        print(batch_upload_jobs(args.batch))
    elif args.once:
        print(run_once())
    else:
        run_loop(interval_sec=args.interval)


if __name__ == "__main__":
    main()
