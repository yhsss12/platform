"""Import standalone joint-space DP full pipeline into workspace index (scoped, no bulk reindex)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.platform_paths import platform_paths

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PIPELINE_ROOT = platform_paths.runs_root / "standalone_dp_joint_space_tests" / "20260624_full_pipeline"
DATA_GEN_JOB_ID = "ct_gen_20260624_joint_space_replay_full"
TRAIN_JOB_ID = "train_joint_dp_20260624_full_pipeline"
EVAL_JOB_ID = "eval_joint_dp_20260624_full_pipeline"
MODEL_ASSET_ID = "model_joint_dp_20260624_full_final"

DISPLAY = {
    "datasetName": "Joint-Space DP · 线缆穿杆 81 demo",
    "trainTaskName": "Joint-Space DP · 200ep 训练",
    "modelName": "Joint-Space DP · Final (joint_position)",
    "evalTaskName": "Joint-Space DP · 10ep 评测 (horizon=1200)",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_train_loss_series(train_root: Path) -> list[dict[str, Any]]:
    from app.services.training_metrics import resolve_training_metrics_log_path

    log_path = resolve_training_metrics_log_path(train_root)
    if not log_path.is_file():
        pipeline_log = train_root.parent / "train.log"
        log_path = pipeline_log if pipeline_log.is_file() else log_path
    if not log_path.is_file():
        return []
    import re

    pattern = re.compile(r"Epoch\s+(\d+)\s+Loss:\s+([0-9.eE+-]+)", re.IGNORECASE)
    series: list[dict[str, Any]] = []
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if not match:
                continue
            series.append({"epoch": int(match.group(1)), "trainLoss": float(match.group(2))})
    except OSError:
        return []
    return series


def _ensure_runtime_status_files(
    *,
    train_root: Path,
    eval_root: Path,
    checkpoint: Path,
    train_config: dict[str, Any],
    aggregate: dict[str, Any],
    dataset_id: str,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _write_json(
        train_root / "status.json",
        {
            "trainJobId": TRAIN_JOB_ID,
            "status": "completed",
            "progress": 1.0,
            "epoch": 200,
            "totalEpochs": 200,
            "datasetId": dataset_id,
            "datasetName": DISPLAY["datasetName"],
            "downstreamModelType": "diffusion_policy",
            "trainingBackend": "diffusion_policy",
            "modelAssetId": MODEL_ASSET_ID,
            "checkpointPath": str(checkpoint),
            "checkpointExists": True,
            "trainedActionMode": "joint_delta",
            "evalExecutor": "joint_position",
            "controllerType": "JOINT_POSITION",
            "updatedAt": now,
        },
    )
    _write_json(
        eval_root / "status.json",
        {
            "evalJobId": EVAL_JOB_ID,
            "status": "completed",
            "modelAssetId": MODEL_ASSET_ID,
            "trainJobId": TRAIN_JOB_ID,
            "datasetId": dataset_id,
            "evalExecutor": "joint_position",
            "actionMode": "joint_delta",
            "controllerType": aggregate.get("controller_type") or "JOINT_POSITION",
            "successRate": aggregate.get("final_success_rate") or aggregate.get("success_rate"),
            "updatedAt": now,
        },
    )
    _write_json(
        train_root / "artifacts" / "model_manifest.json",
        {
            "modelAssetId": MODEL_ASSET_ID,
            "status": "ready",
            "checkpointPath": str(checkpoint),
            "modelType": "diffusion_policy",
            "trainedActionMode": "joint_delta",
            "evalExecutor": "joint_position",
            "controllerType": "JOINT_POSITION",
            "datasetId": dataset_id,
            "displayName": DISPLAY["modelName"],
            "trainConfig": train_config,
        },
    )


def _update_dataset_manifest_display(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    merged = dict(manifest)
    merged["displayName"] = DISPLAY["datasetName"]
    merged["datasetName"] = DISPLAY["datasetName"]
    merged["name"] = DISPLAY["datasetName"]
    merged["datasetId"] = f"ds_{DATA_GEN_JOB_ID}"
    merged["trainedActionMode"] = "joint_delta"
    merged["evalExecutor"] = "joint_position"
    merged["jointActionAvailable"] = True
    _write_json(manifest_path, merged)
    return merged


def _register_eval_video_in_db(
    db,
    *,
    eval_job_id: str,
    eval_root: Path,
) -> str:
    """Persist eval mp4 as workspace artifact + replay URI for frontend playback."""
    from app.models.workspace_index import EvalMetricSummary
    from app.models.workspace_job import WorkspaceArtifact

    raw_video = eval_root / "videos" / "eval.mp4"
    browser_video = eval_root / "videos" / "eval.browser.mp4"
    if not raw_video.is_file() and not browser_video.is_file():
        return ""

    api_uri = f"/api/workspace/evaluation/jobs/{eval_job_id}/video"
    db.query(WorkspaceArtifact).filter(
        WorkspaceArtifact.job_id == eval_job_id,
        WorkspaceArtifact.artifact_type == "video",
    ).delete(synchronize_session=False)
    db.flush()
    db.add(
        WorkspaceArtifact(
            job_id=eval_job_id,
            artifact_type="video",
            name=raw_video.name,
            file_path=str(raw_video.resolve()),
            url_path=api_uri,
            metadata_json={
                "source": "joint_dp_import",
                "format": "mp4",
                "browserPath": str(browser_video.resolve()) if browser_video.is_file() else None,
            },
        )
    )

    ev = db.query(EvalMetricSummary).filter(EvalMetricSummary.job_id == eval_job_id).one_or_none()
    if ev is not None:
        ev.replay_uri = api_uri

    return api_uri


def import_joint_dp_full_pipeline(*, dry_run: bool = False) -> dict[str, Any]:
    """Register joint-space DP test dataset / train / model / eval only."""
    from app.core.database import SessionLocal
    from app.models.workspace_index import EvalMetricSummary, ModelAsset, TrainingMetricSummary
    from app.models.workspace_job import WorkspaceJob
    from app.services.training_job_sync_service import path_to_storage_uri
    from app.services.workspace_dataset_backfill_service import backfill_hdf5_dataset_records
    from app.services.workspace_job_service import _sync_job_record

    data_hdf5 = (
        platform_paths.runs_root
        / "cable_threading"
        / "jobs"
        / DATA_GEN_JOB_ID
        / "datasets"
        / "dataset.hdf5"
    )
    train_root = PIPELINE_ROOT / "train_200ep"
    eval_root = PIPELINE_ROOT / "eval"
    checkpoint = train_root / "checkpoints" / "model_final.pt"
    aggregate_path = eval_root / "results" / "aggregate_result.json"
    train_config = _read_json(train_root / "config" / "train_config.json")
    aggregate = _read_json(aggregate_path)
    result_meta = _read_json(PIPELINE_ROOT / "result.json")

    report: dict[str, Any] = {
        "dryRun": dry_run,
        "datasetJobId": DATA_GEN_JOB_ID,
        "datasetId": f"ds_{DATA_GEN_JOB_ID}",
        "datasetName": DISPLAY["datasetName"],
        "trainJobId": TRAIN_JOB_ID,
        "trainTaskName": DISPLAY["trainTaskName"],
        "modelAssetId": MODEL_ASSET_ID,
        "modelName": DISPLAY["modelName"],
        "evalJobId": EVAL_JOB_ID,
        "evalTaskName": DISPLAY["evalTaskName"],
        "checkpointPath": str(checkpoint),
        "videoPath": str(eval_root / "videos" / "eval.mp4"),
        "errors": [],
    }

    missing: list[str] = []
    if not data_hdf5.is_file():
        missing.append(str(data_hdf5))
    if not checkpoint.is_file():
        missing.append(str(checkpoint))
    if not aggregate_path.is_file():
        missing.append(str(aggregate_path))
    if missing:
        report["errors"] = [f"missing required artifact: {p}" for p in missing]
        return report

    if dry_run:
        return report

    dataset_result = backfill_hdf5_dataset_records(dry_run=False, overwrite=True)
    report["datasetBackfill"] = {
        k: dataset_result.get(k)
        for k in ("insertedHdf5Datasets", "updatedHdf5Datasets", "insertedDataAssets", "updatedDataAssets", "errors")
    }

    dataset_id = f"ds_{DATA_GEN_JOB_ID}"
    manifest_path = data_hdf5.parent / "dataset.manifest.json"
    manifest = _update_dataset_manifest_display(manifest_path, _read_json(manifest_path))
    loss_series = _parse_train_loss_series(train_root)
    _ensure_runtime_status_files(
        train_root=train_root,
        eval_root=eval_root,
        checkpoint=checkpoint,
        train_config=train_config,
        aggregate=aggregate,
        dataset_id=dataset_id,
    )

    try:
        with SessionLocal() as db:
            data_gen_root = data_hdf5.parent.parent
            _sync_job_record(
                db,
                job_id=DATA_GEN_JOB_ID,
                job_type="generate",
                task_type="cable_threading",
                runtime_path=str(data_gen_root),
                runner="run.py",
                task_name=DISPLAY["datasetName"],
                metadata={"datasetManifest": manifest, "importSource": "joint_dp_pipeline"},
                overwrite=True,
            )
            gen_row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == DATA_GEN_JOB_ID).one()
            gen_row.status = "completed"
            gen_row.task_name = DISPLAY["datasetName"]
            gen_row.updated_at = _utc_now()

            _sync_job_record(
                db,
                job_id=TRAIN_JOB_ID,
                job_type="training",
                task_type="cable_threading",
                runtime_path=str(train_root),
                runner="train_dp.py",
                task_name=DISPLAY["trainTaskName"],
                metadata={
                    "trainConfig": train_config,
                    "importSource": "joint_dp_pipeline",
                    "datasetId": dataset_id,
                    "datasetName": DISPLAY["datasetName"],
                },
                overwrite=True,
            )
            train_row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == TRAIN_JOB_ID).one()
            train_row.status = "completed"
            train_row.task_name = DISPLAY["trainTaskName"]
            train_row.metrics_json = {
                "datasetId": dataset_id,
                "datasetName": DISPLAY["datasetName"],
                "downstreamModelType": "diffusion_policy",
                "trainingBackend": "diffusion_policy",
                "modelAssetId": MODEL_ASSET_ID,
                "checkpointPath": str(checkpoint),
                "checkpointExists": True,
                "totalEpochs": 200,
                "epoch": 200,
                "progress": 1.0,
                "trainedActionMode": "joint_delta",
                "evalExecutor": "joint_position",
                "controllerType": "JOINT_POSITION",
                "actionDim": 8,
                "lossHistory": loss_series,
            }
            train_row.updated_at = _utc_now()

            tm = db.query(TrainingMetricSummary).filter(TrainingMetricSummary.job_id == TRAIN_JOB_ID).one_or_none()
            tm_payload = {
                "current_epoch": 200,
                "total_epochs": 200,
                "progress": 1.0,
                "loss_series": loss_series,
                "final_loss": loss_series[-1]["trainLoss"] if loss_series else None,
                "best_loss": min((row["trainLoss"] for row in loss_series), default=None),
                "updated_at": _utc_now(),
            }
            if tm is None:
                db.add(TrainingMetricSummary(job_id=TRAIN_JOB_ID, **tm_payload))
            else:
                for key, value in tm_payload.items():
                    setattr(tm, key, value)

            manifest_json = {
                "displayName": DISPLAY["modelName"],
                "modelType": "diffusion_policy",
                "trainedActionMode": "joint_delta",
                "actionMode": "joint_delta",
                "evalExecutor": "joint_position",
                "controllerType": "JOINT_POSITION",
                "actionKey": "joint_actions",
                "gripperActionKey": "gripper_actions",
                "actionDim": 8,
                "datasetId": dataset_id,
                "sourceDatasetId": dataset_id,
                "canEvaluate": True,
                "isPlaceholder": False,
                "displayStatus": "ready",
            }
            size_bytes = checkpoint.stat().st_size if checkpoint.is_file() else None
            ma = db.query(ModelAsset).filter(ModelAsset.model_asset_id == MODEL_ASSET_ID).one_or_none()
            ma_payload = {
                "train_job_id": TRAIN_JOB_ID,
                "dataset_id": dataset_id,
                "model_name": DISPLAY["modelName"],
                "model_type": "diffusion_policy",
                "asset_type": "final",
                "epoch": 200,
                "storage_uri": path_to_storage_uri(checkpoint),
                "manifest_json": manifest_json,
                "size_bytes": size_bytes,
                "status": "ready",
                "updated_at": _utc_now(),
            }
            if ma is None:
                db.add(ModelAsset(model_asset_id=MODEL_ASSET_ID, created_at=_utc_now(), **ma_payload))
            else:
                for key, value in ma_payload.items():
                    setattr(ma, key, value)

            _sync_job_record(
                db,
                job_id=EVAL_JOB_ID,
                job_type="evaluation",
                task_type="cable_threading",
                runtime_path=str(eval_root),
                runner="run.py",
                task_name=DISPLAY["evalTaskName"],
                metadata={
                    "importSource": "joint_dp_pipeline",
                    "modelAssetId": MODEL_ASSET_ID,
                    "modelAssetName": DISPLAY["modelName"],
                    "trainJobId": TRAIN_JOB_ID,
                    "datasetId": dataset_id,
                    "datasetName": DISPLAY["datasetName"],
                    "evalExecutor": "joint_position",
                    "trainedActionMode": "joint_delta",
                    "controllerType": "JOINT_POSITION",
                    "evaluationType": "model",
                    "evaluationTypeLabel": "模型评测",
                    "evaluationMode": "trained_model_evaluation",
                    "evaluationObject": "trained_model",
                    "productEvaluationMode": "model_evaluation",
                },
                overwrite=True,
            )
            eval_row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == EVAL_JOB_ID).one()
            eval_row.status = "completed"
            eval_row.task_name = DISPLAY["evalTaskName"]
            eval_row.metrics_json = {
                "modelAssetId": MODEL_ASSET_ID,
                "modelName": DISPLAY["modelName"],
                "trainJobId": TRAIN_JOB_ID,
                "datasetId": dataset_id,
                "datasetName": DISPLAY["datasetName"],
                "evaluationType": "model",
                "evaluationTypeLabel": "模型评测",
                "evaluationMode": "trained_model_evaluation",
                "evaluationObject": "trained_model",
                "successRate": aggregate.get("final_success_rate") or aggregate.get("success_rate"),
                "everSuccessRate": aggregate.get("ever_success_rate"),
                "threadCompletionMax": aggregate.get("mean_thread_completion_max"),
                "endpointGoalError": aggregate.get("mean_endpoint_goal_error_final"),
                "evalExecutor": "joint_position",
                "controllerType": aggregate.get("controller_type") or "JOINT_POSITION",
                "actionMode": "joint_delta",
                "videoAvailable": True,
                "replayAvailable": True,
                "replayUri": f"/api/workspace/evaluation/jobs/{EVAL_JOB_ID}/video",
            }
            merged_meta = dict(eval_row.metadata_json or {})
            merged_meta.update(
                {
                    "evaluationType": "model",
                    "evaluationTypeLabel": "模型评测",
                    "evaluationMode": "trained_model_evaluation",
                    "evaluationObject": "trained_model",
                    "modelAssetName": DISPLAY["modelName"],
                }
            )
            eval_row.metadata_json = merged_meta
            eval_row.updated_at = _utc_now()

            replay_api_uri = f"/api/workspace/evaluation/jobs/{EVAL_JOB_ID}/video"

            summary_json = dict(aggregate)
            summary_json.update(
                {
                    "status": "completed",
                    "modelAssetId": MODEL_ASSET_ID,
                    "trainJobId": TRAIN_JOB_ID,
                    "datasetId": dataset_id,
                    "evalExecutor": "joint_position",
                    "actionMode": "joint_delta",
                    "controllerType": "JOINT_POSITION",
                    "policyType": "diffusion_policy_joint",
                    "videoPath": str(eval_root / "videos" / "eval.mp4"),
                }
            )
            ev = db.query(EvalMetricSummary).filter(EvalMetricSummary.job_id == EVAL_JOB_ID).one_or_none()
            ev_payload = {
                "model_asset_id": MODEL_ASSET_ID,
                "summary_json": summary_json,
                "report_uri": path_to_storage_uri(aggregate_path),
                "replay_uri": replay_api_uri or f"/api/workspace/evaluation/jobs/{EVAL_JOB_ID}/video",
                "updated_at": _utc_now(),
            }
            if ev is None:
                db.add(EvalMetricSummary(job_id=EVAL_JOB_ID, **ev_payload))
            else:
                for key, value in ev_payload.items():
                    setattr(ev, key, value)

            db.commit()
    except Exception as exc:
        logger.exception("joint dp import failed")
        report["errors"].append(str(exc))
    else:
        try:
            from app.services.training_job_sync_service import sync_eval_job_from_runtime
            from app.services.workspace_job_service import _sync_job_record

            sync_eval_job_from_runtime(EVAL_JOB_ID, overwrite_artifacts=True)
            with SessionLocal() as db:
                for job_id, job_type, runtime_path, runner, task_name in (
                    (DATA_GEN_JOB_ID, "generate", str(data_hdf5.parent.parent), "run.py", DISPLAY["datasetName"]),
                    (TRAIN_JOB_ID, "training", str(train_root), "train_dp.py", DISPLAY["trainTaskName"]),
                    (EVAL_JOB_ID, "evaluation", str(eval_root), "run.py", DISPLAY["evalTaskName"]),
                ):
                    _sync_job_record(
                        db,
                        job_id=job_id,
                        job_type=job_type,
                        task_type="cable_threading",
                        runtime_path=runtime_path,
                        runner=runner,
                        task_name=task_name,
                        overwrite=True,
                    )
                _register_eval_video_in_db(db, eval_job_id=EVAL_JOB_ID, eval_root=eval_root)
                eval_row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == EVAL_JOB_ID).one_or_none()
                if eval_row is not None:
                    metrics = dict(eval_row.metrics_json or {})
                    metrics.update(
                        {
                            "modelAssetId": MODEL_ASSET_ID,
                            "modelName": DISPLAY["modelName"],
                            "evaluationType": "model",
                            "evaluationTypeLabel": "模型评测",
                            "evaluationMode": "trained_model_evaluation",
                            "evaluationObject": "trained_model",
                            "videoAvailable": True,
                            "replayUri": f"/api/workspace/evaluation/jobs/{EVAL_JOB_ID}/video",
                        }
                    )
                    eval_row.metrics_json = metrics
                    meta = dict(eval_row.metadata_json or {})
                    meta.update(
                        {
                            "evaluationType": "model",
                            "evaluationTypeLabel": "模型评测",
                            "evaluationMode": "trained_model_evaluation",
                            "evaluationObject": "trained_model",
                            "modelAssetId": MODEL_ASSET_ID,
                            "modelAssetName": DISPLAY["modelName"],
                        }
                    )
                    eval_row.metadata_json = meta
                db.commit()
        except Exception as exc:
            logger.warning("joint dp artifact sync failed: %s", exc)
            report["errors"].append(f"artifact sync: {exc}")

    report["pipelineRoot"] = str(PIPELINE_ROOT)
    report["jointHdf5"] = str(result_meta.get("joint_hdf5") or data_hdf5)
    return report
