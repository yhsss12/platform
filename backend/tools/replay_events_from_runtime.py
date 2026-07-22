#!/usr/bin/env python3
"""从 runtime_outputs 重建 event stream（幂等，用于 debug / lineage 修复）。

用法：
  cd backend && python tools/replay_events_from_runtime.py --dry-run
  cd backend && python tools/replay_events_from_runtime.py --limit 20
  cd backend && python tools/replay_events_from_runtime.py --job-id train_xxx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.events.event_bus import get_event_bus
from app.core.events.event_emitter import emit_event
from app.core.events.event_models import EventType, PlatformEvent
from app.services.artifact_upload_service import discover_runtime_job_ids, RUNTIME_SCAN_ROOTS

logger = logging.getLogger(__name__)

TERMINAL = frozenset({"completed", "failed", "canceled", "cancelled", "success", "succeeded"})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_job_root(job_id: str) -> Optional[Path]:
    for root in RUNTIME_SCAN_ROOTS:
        candidate = root / job_id
        if candidate.is_dir():
            return candidate
    return None


def _infer_job_type(job_id: str) -> str:
    if job_id.startswith("train_"):
        return "training"
    if job_id.startswith(("eval_", "ct_eval_", "isaac_eval_")):
        return "evaluation"
    if job_id.startswith(("ct_gen_", "dac_gen_", "dg_gen_")):
        return "generate"
    return "unknown"


def build_events_for_job(job_id: str) -> list[PlatformEvent]:
    root = _resolve_job_root(job_id)
    if root is None:
        return []
    status = _read_json(root / "status.json") or _read_json(root / "live" / "status.json")
    job_type = _infer_job_type(job_id)
    st = str(status.get("status") or "").lower()
    events: list[PlatformEvent] = []
    payload = {"status": st, "jobType": job_type, "replayed": True}

    if job_type == "training":
        if st == "running":
            events.append(PlatformEvent.create(EventType.TRAINING_STARTED, job_id, payload=payload, source="replay"))
        if st in TERMINAL and st == "completed":
            events.append(PlatformEvent.create(EventType.TRAINING_COMPLETED, job_id, payload=payload, source="replay"))
            ckpt_dir = root / "checkpoints"
            if ckpt_dir.is_dir():
                for ckpt in ckpt_dir.rglob("*"):
                    if ckpt.is_file() and ckpt.suffix.lower() in {".ckpt", ".pt", ".pth"}:
                        events.append(
                            PlatformEvent.create(
                                EventType.CHECKPOINT_CREATED,
                                job_id,
                                payload={"checkpointPath": str(ckpt), "replayed": True},
                                source="replay",
                            )
                        )
    elif job_type == "evaluation":
        if st == "running":
            events.append(PlatformEvent.create(EventType.EVAL_STARTED, job_id, payload=payload, source="replay"))
        if st in TERMINAL and st == "completed":
            events.append(PlatformEvent.create(EventType.EVAL_COMPLETED, job_id, payload=payload, source="replay"))
    elif job_type == "generate":
        if st in TERMINAL and st == "completed":
            events.append(PlatformEvent.create(EventType.DATASET_INGESTED, job_id, payload=payload, source="replay"))

    metrics = root / "metrics.jsonl"
    if metrics.is_file() and job_type == "training":
        payload_metrics = dict(payload)
        payload_metrics["hasMetricsJsonl"] = True
        events.append(
            PlatformEvent.create(
                EventType.TRAINING_STARTED,
                job_id,
                payload=payload_metrics,
                source="replay_metrics",
            )
        )
    return events


def replay_jobs(
    job_ids: list[str],
    *,
    dry_run: bool = False,
    dispatch: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"dryRun": dry_run, "jobs": len(job_ids), "events": 0, "details": []}
    bus = get_event_bus()
    for job_id in job_ids:
        events = build_events_for_job(job_id)
        summary["events"] += len(events)
        summary["details"].append({"jobId": job_id, "eventCount": len(events)})
        if dry_run:
            continue
        for event in events:
            emit_event(
                event.event_type,
                event.job_id,
                payload=event.payload,
                source=event.source,
                async_dispatch=False,
            )
            try:
                from app.services.platform_stage2_hooks import after_workspace_job_sync

                after_workspace_job_sync(job_id)
            except Exception as exc:
                logger.warning("lineage replay hook failed job_id=%s: %s", job_id, exc)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay platform events from runtime_outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--job-id", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    job_ids = [j.strip() for j in args.job_id if j.strip()] or discover_runtime_job_ids(include_non_terminal=True, limit=args.limit)
    summary = replay_jobs(job_ids, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
