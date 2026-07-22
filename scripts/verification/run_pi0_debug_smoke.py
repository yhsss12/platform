#!/usr/bin/env python3
"""Run one real openpi debug_pi05 smoke training job and print acceptance report."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
os.chdir(BACKEND_ROOT)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.core.database import SessionLocal
from app.models.workspace_index import ModelAsset, TrainingMetricSummary
from app.models.workspace_job import WorkspaceJob
from app.services import training_service as training_svc
from app.services.pi0_training_runner import probe_pi0_training_capability

HDF5 = PROJECT_ROOT / "runs/cable_threading/datasets/panda_composite_cable_10.hdf5"
MANIFEST = {
    "datasetId": "ds_pi0_debug_smoke",
    "datasetName": "pi0 debug_pi05 smoke",
    "taskType": "cable_threading",
    "taskDescription": "debug_pi05 smoke placeholder dataset",
    "observationSpace": {
        "type": "image",
        "keys": ["agentview_image", "robot0_eye_in_hand_image"],
    },
    "actionSpace": {"type": "continuous", "dim": 7},
    "episodes": 10,
    "successfulEpisodes": 10,
    "artifacts": {"hdf5": str(HDF5)},
}


def _extract_openpi_command(log_text: str) -> str:
    for line in log_text.splitlines():
        if "openpi command:" in line:
            return line.split("openpi command:", 1)[1].strip()
    return ""


def _query_db(job_id: str) -> dict:
    with SessionLocal() as db:
        job = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
        metric = (
            db.query(TrainingMetricSummary)
            .filter(TrainingMetricSummary.job_id == job_id)
            .one_or_none()
        )
        assets = (
            db.query(ModelAsset)
            .filter(ModelAsset.train_job_id == job_id)
            .order_by(ModelAsset.created_at.desc())
            .all()
        )
        return {
            "workspace_job": {
                "job_id": job.job_id if job else None,
                "status": job.status if job else None,
                "job_type": job.job_type if job else None,
                "metadata": job.metadata_json if job else None,
            },
            "training_metric_summary": {
                "final_loss": getattr(metric, "final_loss", None),
                "progress": getattr(metric, "progress", None),
                "total_epochs": getattr(metric, "total_epochs", None),
                "current_epoch": getattr(metric, "current_epoch", None),
            }
            if metric
            else None,
            "model_assets": [
                {
                    "modelAssetId": row.model_asset_id,
                    "modelType": row.model_type,
                    "trainingBackend": row.training_backend,
                    "checkpointKind": row.checkpoint_kind,
                    "status": row.status,
                    "checkpointPath": row.checkpoint_path,
                }
                for row in assets
            ],
        }


def main() -> int:
    print("=== pi0 debug_pi05 smoke acceptance ===")
    print("OPENPI_BASE_CONFIG=", os.environ.get("OPENPI_BASE_CONFIG"))
    print("OPENPI_ROOT=", os.environ.get("OPENPI_ROOT"))
    print("OPENPI_PYTHON=", os.environ.get("OPENPI_PYTHON"))
    cap = probe_pi0_training_capability()
    print("probe ready=", cap.get("ready"), "reason=", cap.get("reason"))

    if not HDF5.is_file():
        print("ERROR: HDF5 placeholder missing:", HDF5)
        return 1

    payload = {
        "datasetId": "ds_pi0_debug_smoke",
        "modelTypeId": "pi0",
        "datasetManifest": MANIFEST,
        "trainingBackend": "pi0",
        "downstreamModelType": "pi0",
        "epochs": 1,
        "batchSize": 2,
        "learningRate": 1e-4,
        "device": "cpu",
        "seed": 1,
        "taskName": "pi0 debug_pi05 smoke",
    }
    result = training_svc.create_training_job(payload)
    job_id = result["trainJobId"]
    job_dir = training_svc._train_job_dir(job_id)
    print("job_id:", job_id)
    print("job_dir:", job_dir)

    deadline = time.time() + 1800
    status: dict = {}
    while time.time() < deadline:
        status = training_svc.get_training_job_status(job_id)
        state = status.get("status")
        print(f"poll status={state} progress={status.get('progress')} epoch={status.get('epoch')}")
        if state in {"completed", "failed", "backend_unavailable", "stopped"}:
            break
        time.sleep(5)

    log_path = job_dir / "logs" / "train.log"
    log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    first_line = log_text.splitlines()[0] if log_text else ""
    openpi_cmd = _extract_openpi_command(log_text)
    metrics_path = job_dir / "artifacts" / "metrics.jsonl"
    metrics_sample = []
    if metrics_path.is_file():
        metrics_sample = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:5]
    final_ckpt = job_dir / "checkpoints" / "pi0" / "checkpoints" / "model_final.pt"
    db_info = _query_db(job_id)

    print("\n=== report ===")
    print("job_id:", job_id)
    print("train.log first line:", first_line)
    print("openpi command:", openpi_cmd)
    print("smoke log present:", "debug_pi05 smoke mode: skip platform dataset conversion" in log_text)
    print("unsupported flags absent:", all(x not in log_text for x in ("--config-path", "--learning-rate", "--output-dir")))
    print("status.json:", json.dumps(status, ensure_ascii=False, indent=2))
    print("metrics.jsonl sample:", json.dumps(metrics_sample, ensure_ascii=False, indent=2))
    print("checkpoint path:", str(final_ckpt), "exists=", final_ckpt.is_file())
    print("workspace/db:", json.dumps(db_info, ensure_ascii=False, indent=2, default=str))

    ok = (
        status.get("status") == "completed"
        and final_ckpt.is_file()
        and metrics_path.is_file()
        and metrics_sample
        and openpi_cmd
        and "debug_pi05 smoke mode: skip platform dataset conversion" in log_text
    )
    final_assets = [
        a for a in db_info.get("model_assets") or [] if (a.get("checkpointKind") or "").lower() == "final"
    ]
    if final_assets:
        print("Final model_asset:", final_assets[0])
    else:
        ok = False
        print("Final model_asset: MISSING")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
