#!/usr/bin/env python3
"""模型资产 DB 记录与文件系统一致性 reconcile。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.database import SessionLocal
from app.models.workspace_index import ModelAsset
from app.services.model_asset_db_service import _row_to_asset
from app.services.model_asset_validation import enrich_model_asset, resolve_effective_asset_status
from app.services.model_asset_validation import check_checkpoint_file, _training_job_exists


def reconcile_model_assets(*, apply: bool = False) -> list[dict]:
    rows_out: list[dict] = []
    with SessionLocal() as db:
        rows = db.query(ModelAsset).filter(ModelAsset.status != "deleted").order_by(ModelAsset.created_at.asc()).all()
        for row in rows:
            asset = enrich_model_asset(_row_to_asset(row))
            manifest = row.manifest_json if isinstance(row.manifest_json, dict) else {}
            file_exists = bool(asset.get("fileExists"))
            train_exists = _training_job_exists(row.train_job_id)
            before = str(row.status or "")
            after = resolve_effective_asset_status(
                db_status=before,
                file_exists=file_exists,
                train_job_exists=train_exists,
                is_placeholder=bool(manifest.get("isPlaceholder")),
            )
            reason = ""
            if after == "missing":
                reason = "checkpoint 文件不存在或大小为 0"
            elif after == "invalid":
                reason = "关联训练任务不存在或为占位资产"
            elif after == "available" and before in {"generating", "ready"}:
                reason = "文件已就绪"

            record = {
                "modelAssetId": row.model_asset_id,
                "displayName": asset.get("displayName"),
                "backendType": asset.get("backendType"),
                "artifactPath": asset.get("artifactPath"),
                "fileExists": file_exists,
                "statusBefore": before,
                "statusAfter": after,
                "reason": reason,
            }
            rows_out.append(record)

            if apply and after != before and after in {"missing", "invalid", "available"}:
                db_status = after
                if after == "available":
                    db_status = "ready"
                row.status = db_status
                merged_manifest = dict(manifest)
                merged_manifest["reconcileStatus"] = after
                merged_manifest["fileExists"] = file_exists
                row.manifest_json = merged_manifest
                exists, size = check_checkpoint_file(str(asset.get("artifactPath") or ""))
                if exists:
                    row.size_bytes = size

        if apply:
            db.commit()
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile model assets against filesystem")
    parser.add_argument("--apply", action="store_true", help="写回 DB status；默认 dry-run")
    args = parser.parse_args()

    rows = reconcile_model_assets(apply=args.apply)
    print(f"mode={'apply' if args.apply else 'dry-run'} assets={len(rows)}")
    print("| modelAssetId | displayName | backendType | fileExists | statusBefore | statusAfter | reason |")
    print("|---|---|---|---|---|---|---|")
    changed = 0
    for row in rows:
        if row["statusBefore"] != row["statusAfter"]:
            changed += 1
        print(
            f"| {row['modelAssetId']} | {row.get('displayName') or ''} | "
            f"{row.get('backendType') or ''} | {row['fileExists']} | "
            f"{row['statusBefore']} | {row['statusAfter']} | {row.get('reason') or ''} |"
        )
    print(json.dumps({"changed": changed, "total": len(rows)}, ensure_ascii=False))
    if not args.apply:
        print("dry-run 完成；追加 --apply 写回状态")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
