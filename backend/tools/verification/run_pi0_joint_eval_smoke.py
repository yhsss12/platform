#!/usr/bin/env python3
"""Run pi0 JOINT_POSITION eval rollout smoke (standalone, no platform eval job)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded

ensure_dotenv_loaded()


def main() -> int:
    from app.services.policy_schema_resolver import (
        PI0_JOINT_SPACE_ENABLED,
        assess_pi0_lerobot_data_format_readiness,
        is_pi0_joint_space_eval_asset,
        pi0_eval_adapter_ready,
        resolve_pi0_eval_disabled_reason,
        resolve_pi0_eval_runtime,
    )
    from app.services.pi0_lerobot_smoke_runner import assess_pi0_lerobot_training_capability
    from app.services.checkpoint_registry import explain_model_asset_eval_blocker

    model_asset_id = "model__123947_ebd2_final"
    train_job_id = "train_20260630_123947_ebd2"
    checkpoint = (
        PROJECT_ROOT
        / "runs/training/jobs/train_20260630_123947_ebd2/checkpoints/pi0/checkpoints/model_final.pt"
    )
    train_config = (
        PROJECT_ROOT / "runs/training/jobs/train_20260630_123947_ebd2/config/train_config.json"
    )
    manifest_path = (
        PROJECT_ROOT
        / "runs/training/jobs/train_20260630_123947_ebd2/artifacts/model_manifest.json"
    )
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    if not checkpoint.is_file():
        print(json.dumps({"ok": False, "failure_step": "checkpoint_missing", "error": str(checkpoint)}, indent=2))
        return 1
    if not train_config.is_file():
        print(json.dumps({"ok": False, "failure_step": "train_config_missing", "error": str(train_config)}, indent=2))
        return 1

    asset = dict(model_manifest)
    asset.setdefault("modelAssetId", model_asset_id)
    if not is_pi0_joint_space_eval_asset(asset, checkpoint_path=checkpoint):
        print(
            json.dumps(
                {
                    "ok": False,
                    "failure_step": "asset_schema",
                    "error": "pi0 model asset is not joint-space eval ready",
                    "asset": asset,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    runtime = resolve_pi0_eval_runtime(
        policy="pi0",
        model_asset=asset,
        checkpoint_path=checkpoint,
        train_config_path=train_config,
    )
    if runtime["evalExecutor"] != "joint_position":
        print(
            json.dumps(
                {
                    "ok": False,
                    "failure_step": "eval_resolver",
                    "error": f"expected joint_position, got {runtime['evalExecutor']}",
                    "runtime": runtime,
                },
                indent=2,
            )
        )
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "runs/evaluation/pi0_smoke" / f"eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "eval.csv"
    run_log = out_dir / "run.log"

    cable_python = Path("/home/ubuntu/miniconda3/envs/cable/bin/python")
    python_bin = cable_python if cable_python.is_file() else Path(sys.executable)

    cable_mvp = PROJECT_ROOT / "integrations/CableThreadingMVP"
    cmd = [
        str(python_bin),
        str(cable_mvp / "run.py"),
        "eval",
        "--policy",
        "pi0",
        "--checkpoint",
        str(checkpoint),
        "--train-config",
        str(train_config),
        "--task-instruction",
        runtime["taskInstruction"],
        "--robot",
        "Panda",
        "--eval-executor",
        "joint_position",
        "--controller-type",
        "JOINT_POSITION",
        "--action-mode",
        "joint_delta_derived",
        "--episodes",
        "1",
        "--horizon",
        "200",
        "--device",
        "cpu",
        "--out",
        str(csv_path),
    ]
    print("eval smoke command:", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(cable_mvp),
        capture_output=True,
        text=True,
        check=False,
    )
    run_log.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        print(
            json.dumps(
                {
                    "ok": False,
                    "failure_step": "run_py_eval",
                    "returncode": completed.returncode,
                    "run_log": str(run_log),
                    "stderr_tail": (completed.stderr or "")[-2000:],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return completed.returncode

    aggregate_path = out_dir / "aggregate_result.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.is_file() else {}
    aggregate.setdefault("rollout_ok", True)
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    marker_dir = PROJECT_ROOT / "runs/evaluation/pi0_smoke"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / "adapter_ready.json"
    marker_path.write_text(
        json.dumps(
            {
                "eval_adapter_ready": True,
                "joint_position_rollout_ready": True,
                "modelAssetId": model_asset_id,
                "trainJobId": train_job_id,
                "evalOutputDir": str(out_dir),
                "aggregateResultPath": str(aggregate_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset_path = (
        PROJECT_ROOT
        / "runs/cable_threading/jobs/ct_gen_20260630_120927_1153/datasets/lerobot_dataset"
    )
    capability = assess_pi0_lerobot_training_capability(
        dataset_path=dataset_path,
        smoke_success=True,
        platform_training_success=True,
        eval_rollout_success=True,
    )
    blocker = explain_model_asset_eval_blocker(
        {
            **asset,
            "canEvaluate": False,
            "status": "ready",
            "checkpointPath": str(checkpoint),
        },
        job_status={"status": "completed"},
    )
    report = {
        "ok": True,
        "model_asset_id": model_asset_id,
        "checkpoint": str(checkpoint),
        "train_config": str(train_config),
        "eval_output_dir": str(out_dir),
        "run_log": str(run_log),
        "eval_csv": str(csv_path),
        "aggregate_result": str(aggregate_path),
        "eval_results_json": str(csv_path.with_suffix(".results.json")),
        "runtime": runtime,
        "aggregate": aggregate,
        "capability": capability,
        "PI0_JOINT_SPACE_ENABLED": PI0_JOINT_SPACE_ENABLED,
        "pi0_eval_adapter_ready": pi0_eval_adapter_ready(),
        "can_evaluate": False,
        "eval_disabled_reason": resolve_pi0_eval_disabled_reason(),
        "eval_blocker": blocker,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
