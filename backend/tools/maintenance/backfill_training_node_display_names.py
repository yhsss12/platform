#!/usr/bin/env python3
"""回填训练任务与节点注册表中的训练节点展示名（L20 · IP）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded

ensure_dotenv_loaded()


def _node_registry_summary() -> list[dict[str, Any]]:
    from app.services.training_node_service import get_training_node_registry, probe_training_node

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for cfg in get_training_node_registry().values():
        if cfg.node_id in seen:
            continue
        seen.add(cfg.node_id)
        probe = probe_training_node(cfg.node_id, refresh=False)
        rows.append(
            {
                "nodeId": cfg.node_id,
                "host": cfg.host or probe.get("host"),
                "displayName": cfg.device_label,
                "executionMode": cfg.execution_mode,
                "gpuModel": cfg.gpu_model,
                "status": probe.get("status"),
            }
        )
    rows.sort(key=lambda item: str(item.get("host") or ""))
    return rows


def _resolve_display_name(
    *,
    training_node_id: Optional[str],
    device_label: Optional[str],
    execution_mode: Optional[str],
) -> str:
    from app.services.training_node_service import resolve_training_node_display_name

    return resolve_training_node_display_name(
        training_node_id=training_node_id,
        device_label=device_label,
        execution_mode=execution_mode,
    )


def _plan_workspace_job_updates(
    *,
    node_id_filter: Optional[str] = None,
) -> list[dict[str, Any]]:
    from app.core.db_session import db_session_scope
    from app.models.workspace_job import WorkspaceJob

    plans: list[dict[str, Any]] = []
    with db_session_scope(label="backfill-plan") as db:
        query = db.query(WorkspaceJob).filter(WorkspaceJob.job_type == "training")
        for row in query.all():
            if str(row.status or "").lower() == "deleted":
                continue
            meta = dict(row.metadata_json or {})
            metrics = dict(row.metrics_json or {})
            train_config = dict(meta.get("trainConfig") or {})
            current_node_id = str(
                metrics.get("trainingNodeId")
                or train_config.get("trainingNodeId")
                or meta.get("trainingNodeId")
                or ""
            ).strip()
            if node_id_filter and current_node_id != node_id_filter:
                continue
            execution_mode = str(
                metrics.get("executionMode")
                or train_config.get("executionMode")
                or meta.get("executionMode")
                or ""
            ).strip()
            current_label = str(
                metrics.get("trainingNodeDisplayName")
                or metrics.get("deviceLabel")
                or train_config.get("trainingNodeDisplayName")
                or train_config.get("deviceLabel")
                or meta.get("trainingNodeDisplayName")
                or meta.get("deviceLabel")
                or ""
            )
            new_label = _resolve_display_name(
                training_node_id=current_node_id or None,
                device_label=current_label or None,
                execution_mode=execution_mode or None,
            )
            needs_update = (
                current_label != new_label
                or metrics.get("trainingNodeDisplayName") != new_label
                or train_config.get("deviceLabel") != new_label
                or train_config.get("trainingNodeDisplayName") != new_label
                or meta.get("deviceLabel") != new_label
                or meta.get("trainingNodeDisplayName") != new_label
            )
            if not needs_update:
                continue
            plans.append(
                {
                    "jobId": row.job_id,
                    "status": row.status,
                    "trainingNodeId": current_node_id or None,
                    "before": {
                        "deviceLabel": current_label or None,
                        "trainingNodeDisplayName": metrics.get("trainingNodeDisplayName"),
                    },
                    "after": {
                        "deviceLabel": new_label,
                        "trainingNodeDisplayName": new_label,
                    },
                }
            )
    return plans


def _apply_workspace_job_updates(plans: list[dict[str, Any]]) -> int:
    from app.core.db_session import db_session_scope
    from app.models.workspace_job import WorkspaceJob

    updated = 0
    with db_session_scope(label="backfill-apply") as db:
        for plan in plans:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == plan["jobId"]).one_or_none()
            if row is None or str(row.status or "").lower() == "deleted":
                continue
            meta = dict(row.metadata_json or {})
            metrics = dict(row.metrics_json or {})
            train_config = dict(meta.get("trainConfig") or {})
            display = plan["after"]["trainingNodeDisplayName"]

            train_config["deviceLabel"] = display
            train_config["trainingNodeDisplayName"] = display
            meta["deviceLabel"] = display
            meta["trainingNodeDisplayName"] = display
            meta["trainConfig"] = train_config

            metrics["deviceLabel"] = display
            metrics["trainingNodeDisplayName"] = display
            if plan.get("trainingNodeId"):
                metrics["trainingNodeId"] = plan["trainingNodeId"]
                meta["trainingNodeId"] = plan["trainingNodeId"]
                train_config["trainingNodeId"] = plan["trainingNodeId"]

            row.metadata_json = meta
            row.metrics_json = metrics
            updated += 1
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill training node display names")
    parser.add_argument("--dry-run", action="store_true", help="only print planned changes")
    parser.add_argument("--apply", action="store_true", help="apply changes to database")
    parser.add_argument("--node-id", default="", help="only update jobs for this trainingNodeId")
    parser.add_argument("--db-timeout", type=float, default=5.0)
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.error("specify --dry-run or --apply")

    from app.core.db_health import check_db_health

    health = check_db_health(
        connect_timeout=args.db_timeout,
        statement_timeout_ms=int(args.db_timeout * 1000),
    )
    print(
        f"DB health: {health.level} idle_in_transaction={health.idle_in_transaction} "
        f"blocking_locks={health.blocking_locks}"
    )
    if health.blocking_locks > 0:
        print("blocking locks detected; aborting", file=sys.stderr)
        return 2
    if health.idle_in_transaction > 5:
        print("too many idle in transaction sessions; aborting", file=sys.stderr)
        return 2

    registry = _node_registry_summary()
    print("=== 当前训练节点注册表 ===")
    print(json.dumps(registry, ensure_ascii=False, indent=2))

    node_filter = args.node_id.strip() or None
    plans = _plan_workspace_job_updates(node_id_filter=node_filter)
    print(f"\n=== 计划更新 workspace_jobs: {len(plans)} 条 ===")
    for plan in plans[:50]:
        print(json.dumps(plan, ensure_ascii=False))
    if len(plans) > 50:
        print(f"... 另有 {len(plans) - 50} 条")

    result = {
        "ok": True,
        "dryRun": bool(args.dry_run),
        "registry": registry,
        "plannedUpdates": len(plans),
        "appliedUpdates": 0,
        "tables": ["workspace_jobs"],
    }

    if args.apply:
        applied = _apply_workspace_job_updates(plans)
        result["appliedUpdates"] = applied
        print(f"\n已更新 workspace_jobs: {applied} 条")
    else:
        print("\n(dry-run) 未写入数据库")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
