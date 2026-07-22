#!/usr/bin/env python3
"""Run pi0 platform evaluation smoke via evaluation_service (Phase G)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded

ensure_dotenv_loaded()


def main() -> int:
    from app.core.db_session import db_session_scope
    from app.models.workspace_job import WorkspaceJob
    from app.schemas.evaluation import EvaluateAsyncRequest
    from app.services import cable_threading_service as ct_svc
    from app.services.evaluation import evaluation_service as eval_svc
    from app.services.policy_schema_resolver import (
        PI0_JOINT_SPACE_ENABLED,
        is_pi0_joint_space_enabled,
        mark_pi0_platform_eval_ready,
        pi0_platform_eval_ready,
        resolve_pi0_model_asset_eval_fields,
    )
    from app.services.pi0_lerobot_smoke_runner import assess_pi0_lerobot_training_capability
    from app.services.training_job_sync_service import sync_eval_job_from_runtime
    from app.services.workspace_model_asset_service import get_model_asset_by_id

    model_asset_id = "model__123947_ebd2_final"
    asset = get_model_asset_by_id(model_asset_id) or {}
    eval_fields = resolve_pi0_model_asset_eval_fields(asset)
    print(json.dumps({"modelAssetId": model_asset_id, "evalFields": eval_fields}, ensure_ascii=False, indent=2))

    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        taskType="cable_threading",
        evaluationMode="trained_model_evaluation",
        modelAssetId=model_asset_id,
        numEpisodes=1,
        horizon=200,
        cableThreading={
            "robot": "Panda",
            "device": "cpu",
            "horizon": 200,
            "modelName": "pi0 Platform Eval Smoke",
        },
        taskName="pi0 Platform Eval Smoke",
    )

    created = eval_svc.start_evaluate_async(request)
    eval_job_id = str(created["evalJobId"])
    job_root = ct_svc.OUTPUT_ROOT / "jobs" / eval_job_id
    print(f"created eval_job_id={eval_job_id} runtime={job_root}")

    deadline = time.time() + 600
    final_status = "running"
    while time.time() < deadline:
        sync_eval_job_from_runtime(eval_job_id)
        status_payload = ct_svc.get_job_status(eval_job_id)
        final_status = str(status_payload.get("status") or "running")
        if final_status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(2.0)

    sync_eval_job_from_runtime(eval_job_id)
    aggregate_path = job_root / "results" / "aggregate_result.json"
    eval_csv = job_root / "results" / "eval.csv"
    run_log = job_root / "logs" / "run.log"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.is_file() else {}

    db_status = None
    with db_session_scope() as db:
        row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == eval_job_id).one_or_none()
        db_status = row.status if row else None

    ok = (
        final_status == "completed"
        and db_status == "completed"
        and aggregate_path.is_file()
        and eval_csv.is_file()
        and run_log.is_file()
        and aggregate.get("rollout_ok") is True
        and aggregate.get("modelType") == "pi0"
        and aggregate.get("evalExecutor") == "joint_position"
    )

    if ok and not pi0_platform_eval_ready():
        mark_pi0_platform_eval_ready(eval_job_id=eval_job_id, model_asset_id=model_asset_id)

    capability = assess_pi0_lerobot_training_capability(
        dataset_path=PROJECT_ROOT
        / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/lerobot_dataset",
        platform_training_success=True,
    )

    report = {
        "ok": ok,
        "evalJobId": eval_job_id,
        "runtimeStatus": final_status,
        "dbStatus": db_status,
        "evalOutputDir": str(job_root),
        "runLog": str(run_log),
        "evalCsv": str(eval_csv),
        "aggregateResult": str(aggregate_path),
        "successRate": aggregate.get("success_rate"),
        "everSuccessRate": aggregate.get("ever_success_rate"),
        "rolloutOk": aggregate.get("rollout_ok"),
        "canEvaluateAfter": resolve_pi0_model_asset_eval_fields(asset),
        "capability": capability,
        "pi0JointSpaceEnabled": is_pi0_joint_space_enabled(),
        "platformEvalReady": pi0_platform_eval_ready(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
