#!/usr/bin/env python3
"""Backfill model asset schema fields from checkpoint / train config (dry-run by default)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.services.checkpoint_registry import (  # noqa: E402
    read_registry,
    registry_path,
)
from app.services.training_display_names import apply_training_task_name_backfill  # noqa: E402
from app.services.training_job_sync_service import (  # noqa: E402
    _enrich_registry_entry_from_job_context,
    _registry_entry_to_model_asset_row,
    _upsert_model_assets,
)
from app.services.workspace_model_asset_service import TRAINING_JOBS_ROOT  # noqa: E402


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def backfill_job(job_id: str, *, apply: bool, verbose: bool) -> dict:
    train_job_dir = TRAINING_JOBS_ROOT / job_id
    if not train_job_dir.is_dir():
        raise SystemExit(f"training job dir not found: {train_job_dir}")

    status_path = train_job_dir / "status.json"
    train_config_path = train_job_dir / "config" / "train_config.json"
    status = _read_json(status_path)
    train_config = _read_json(train_config_path)
    manifest = _read_json(train_job_dir / "artifacts" / "dataset_manifest.json")

    registry = read_registry(train_job_dir)
    assets = [dict(item) for item in (registry.get("assets") or []) if isinstance(item, dict)]
    if not assets:
        raise SystemExit(f"no registry assets for {job_id}")

    changes: list[dict] = []
    enriched_assets: list[dict] = []
    for entry in assets:
        before = {key: entry.get(key) for key in (
            "evalExecutor", "controllerType", "actionDim", "actionKey", "lowDimKeys",
            "lowDimDim", "imageKeys", "preferredPolicySchemaId", "robotType", "actionMode",
        )}
        enriched = _enrich_registry_entry_from_job_context(
            entry,
            train_config=train_config,
            train_job_dir=train_job_dir,
        )
        after = {key: enriched.get(key) for key in before}
        delta = {key: {"before": before[key], "after": after[key]} for key in before if before[key] != after[key]}
        if delta:
            changes.append({"modelAssetId": enriched.get("modelAssetId"), "delta": delta})
        enriched_assets.append(enriched)

    status_before = status.get("taskName")
    status_after_payload = apply_training_task_name_backfill(dict(status), dict(train_config))
    task_name_change = None
    if status_after_payload.get("taskName") != status_before:
        task_name_change = {"before": status_before, "after": status_after_payload.get("taskName")}

    report = {
        "jobId": job_id,
        "assetCount": len(enriched_assets),
        "schemaChanges": changes,
        "taskNameChange": task_name_change,
        "apply": apply,
    }

    if verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if not apply:
        print(f"[dry-run] {job_id}: {len(changes)} asset(s) would update schema; taskName change={bool(task_name_change)}")
        return report

    registry_payload = {
        "version": 1,
        "sourceTrainJobId": job_id,
        "updatedAt": status_after_payload.get("updatedAt") or status.get("updatedAt"),
        "assets": enriched_assets,
    }
    _write_json(registry_path(train_job_dir), registry_payload)

    if task_name_change:
        status["taskName"] = status_after_payload.get("taskName")
        train_config["taskName"] = status_after_payload.get("taskName")
        _write_json(status_path, status)
        _write_json(train_config_path, train_config)

    try:
        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            _upsert_model_assets(
                db,
                train_job_id=job_id,
                entries=[_registry_entry_to_model_asset_row(item, train_job_id=job_id, status=status) for item in enriched_assets],
                status=status,
            )
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        print(f"warning: DB upsert skipped or failed: {exc}", file=sys.stderr)

    print(f"[apply] {job_id}: updated registry ({len(changes)} schema deltas)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill model asset schema from checkpoint/train config")
    parser.add_argument("--job-id", required=True, help="train job id, e.g. train_20260626_171542_3cbb")
    parser.add_argument("--dry-run", action="store_true", default=True, help="preview only (default)")
    parser.add_argument("--apply", action="store_true", help="write registry / status / DB")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    apply = bool(args.apply)
    backfill_job(args.job_id, apply=apply, verbose=args.verbose)


if __name__ == "__main__":
    main()
