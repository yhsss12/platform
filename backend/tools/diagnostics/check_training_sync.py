#!/usr/bin/env python3
"""验证训练任务 runtime → PostgreSQL 同步链路（开发/运维自检）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check training job DB sync status")
    parser.add_argument("train_job_id", nargs="?", help="训练任务 ID，如 train_20260620_150000_abcd")
    parser.add_argument("--sync", action="store_true", help="执行 sync_training_job_from_runtime")
    parser.add_argument("--reindex", action="store_true", help="执行 reindex_runtime_jobs(training)")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    from app.services.training_job_sync_service import (
        get_training_job_summary_from_db,
        reindex_runtime_jobs,
        sync_training_job_from_runtime,
    )
    from app.core.database import SessionLocal
    from app.models.workspace_index import ModelAsset, TrainingMetricSummary

    if args.reindex:
        result = reindex_runtime_jobs(job_type="training", dry_run=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    job_id = (args.train_job_id or "").strip()
    if not job_id:
        parser.error("请提供 train_job_id，或使用 --reindex")

    if args.sync:
        sync_training_job_from_runtime(job_id)

    summary = get_training_job_summary_from_db(job_id)
    with SessionLocal() as db:
        metrics = db.query(TrainingMetricSummary).filter_by(job_id=job_id).one_or_none()
        assets = (
            db.query(ModelAsset)
            .filter(ModelAsset.train_job_id == job_id, ModelAsset.status != "deleted")
            .order_by(ModelAsset.asset_type.asc())
            .all()
        )

    payload = {
        "trainJobId": job_id,
        "summary": summary,
        "trainingMetricSummary": {
            "currentEpoch": metrics.current_epoch if metrics else None,
            "totalEpochs": metrics.total_epochs if metrics else None,
            "currentLoss": metrics.current_loss if metrics else None,
            "bestLoss": metrics.best_loss if metrics else None,
            "lossSeriesPoints": len(metrics.loss_series or []) if metrics else 0,
        }
        if metrics
        else None,
        "modelAssets": [
            {
                "modelAssetId": row.model_asset_id,
                "assetType": row.asset_type,
                "epoch": row.epoch,
                "status": row.status,
                "storageUri": row.storage_uri,
                "sha256": row.sha256,
            }
            for row in assets
        ],
        "readyFinalCount": sum(
            1 for row in assets if row.asset_type == "final" and row.status in {"ready", "available"}
        ),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"训练任务: {job_id}")
        print(f"  状态: {(summary or {}).get('status')}")
        print(f"  Epoch: {(summary or {}).get('epoch')}/{(summary or {}).get('totalEpochs')}")
        print(f"  Loss: {(summary or {}).get('loss')}")
        print(f"  指标摘要: {'有' if metrics else '无'}")
        print(f"  模型资产: {len(assets)} 条 (Final ready: {payload['readyFinalCount']})")
        for row in assets:
            print(
                f"    - {row.model_asset_id} kind={row.asset_type} epoch={row.epoch} "
                f"status={row.status} uri={row.storage_uri}"
            )

    return 0 if summary and assets else 1


if __name__ == "__main__":
    raise SystemExit(main())
