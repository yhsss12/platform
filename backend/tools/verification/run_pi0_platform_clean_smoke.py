#!/usr/bin/env python3
"""Create a real pi0 LeRobot platform training job via training_service API and verify DB sync."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded

ensure_dotenv_loaded()


def main() -> int:
    from app.core.db_session import db_session_scope
    from app.models.workspace_job import WorkspaceJob
    from app.services import training_service as training_svc
    from app.services.policy_schema_resolver import PI0_JOINT_SPACE_ENABLED
    from app.services.pi0_lerobot_smoke_runner import assess_pi0_lerobot_training_capability
    from app.services.training_job_sync_service import sync_training_job_from_runtime

    project_root = BACKEND_ROOT.parent
    manifest_path = (
        project_root
        / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/dataset.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasetId"] = manifest.get("datasetId") or "ds_ct_gen_20260630_120927_1153"
    manifest["datasetName"] = manifest.get("displayName") or "线缆穿杆 LeRobot pi0 smoke"

    payload = {
        "datasetId": manifest["datasetId"],
        "modelTypeId": "pi0",
        "datasetManifest": manifest,
        "epochs": 1,
        "batchSize": 2,
        "maxSteps": 10,
        "seed": 1,
        "datasetFormat": "lerobot",
        "taskInstruction": "thread the cable through the pole",
        "taskName": "pi0 LeRobot Platform Smoke",
    }

    created = training_svc.create_training_job(payload)
    job_id = created["trainJobId"]
    print(f"created job_id={job_id}")

    thread = training_svc._RUNNING_THREADS.get(job_id)
    deadline = time.time() + 180
    while time.time() < deadline:
        if thread is None or not thread.is_alive():
            break
        time.sleep(0.5)

    train_job_dir = training_svc._train_job_dir(job_id)
    status = json.loads((train_job_dir / "status.json").read_text(encoding="utf-8"))
    print(f"runtime status={status.get('status')} progress={status.get('progress')}")

    sync_training_job_from_runtime(job_id)
    training_svc.finalize_training_job_sync(job_id)

    with db_session_scope() as db:
        row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
        db_status = row.status if row else None
        metrics = dict(row.metrics_json or {}) if row else {}

    mm_path = train_job_dir / "artifacts/model_manifest.json"
    mm = json.loads(mm_path.read_text(encoding="utf-8")) if mm_path.is_file() else {}

    lerobot_path = (
        project_root
        / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/lerobot_dataset"
    )
    cap = assess_pi0_lerobot_training_capability(
        dataset_path=lerobot_path,
        platform_training_success=status.get("status") == "completed",
        smoke_success=status.get("status") == "completed",
    )

    metrics_lines = []
    metrics_path = train_job_dir / "artifacts/metrics.jsonl"
    if metrics_path.is_file():
        metrics_lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()

    report = {
        "job_id": job_id,
        "dataset_id": manifest["datasetId"],
        "lerobot_path": str(lerobot_path),
        "payload": payload,
        "execution_mode": status.get("executionMode") or "local",
        "device": status.get("device"),
        "runtime_status": status.get("status"),
        "runtime_progress": status.get("progress"),
        "runtime_final_loss": status.get("finalLoss"),
        "db_status": db_status,
        "db_progress": metrics.get("progress"),
        "db_loss_series_len": len(metrics.get("lossHistory") or metrics.get("lossSeries") or []),
        "model_asset_id": mm.get("modelAssetId"),
        "can_evaluate": mm.get("canEvaluate"),
        "eval_disabled_reason": mm.get("evalDisabledReason"),
        "PI0_JOINT_SPACE_ENABLED": PI0_JOINT_SPACE_ENABLED,
        "capability": cap,
        "paths": {
            "train_log": str(train_job_dir / "logs/train.log"),
            "metrics_jsonl": str(metrics_path),
            "checkpoint": str(train_job_dir / "checkpoints/pi0/checkpoints/model_final.pt"),
            "train_config": str(train_job_dir / "config/train_config.json"),
        },
        "loss_series": [json.loads(line).get("loss") for line in metrics_lines[:10]],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    ok = (
        status.get("status") == "completed"
        and db_status == "completed"
        and mm.get("canEvaluate") is False
        and mm.get("evalDisabledReason") == "pi0 eval adapter not ready"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
