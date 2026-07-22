"""训练/评测任务：runs → PostgreSQL 索引同步。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.platform_paths import platform_paths, resolve_runtime_reference
from app.models.workspace_index import EvalMetricSummary, ModelAsset, TrainingMetricSummary
from app.models.workspace_job import WorkspaceJob
from app.services.training_job_status import normalize_api_training_status

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
RUNTIME_ROOT = platform_paths.runs_root
TRAINING_JOBS_ROOT = RUNTIME_ROOT / "training" / "jobs"
EVAL_JOBS_ROOT = RUNTIME_ROOT / "evaluations" / "jobs"

TRAIN_JOB_ID_PATTERN = re.compile(r"^train_\d{8}_\d{6}_[a-f0-9]{4}$")
TRAIN_JOB_DIR_PATTERN = re.compile(r"^train_")
TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled"})


def _runtime_dir_exists(runtime_path: Optional[str]) -> bool:
    if not runtime_path:
        return False
    path = resolve_runtime_reference(runtime_path)
    return path.is_dir()


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


def path_to_storage_uri(path: Path | str) -> str:
    text = str(path).strip()
    if not text:
        return ""
    if "://" in text:
        return text
    resolved = resolve_runtime_reference(text)
    return f"file://{resolved}"


def _file_digest(path: Path) -> tuple[Optional[str], Optional[int]]:
    if not path.is_file():
        return None, None
    try:
        size = path.stat().st_size
        if size <= 0:
            return None, 0
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), size
    except OSError:
        return None, None


def _loss_value(row: dict[str, Any]) -> Optional[float]:
    raw = row.get("trainLoss", row.get("loss", row.get("validLoss")))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _summarize_loss_series(
    loss_series: list[dict[str, Any]],
    *,
    job_status: Optional[str] = None,
) -> tuple[Optional[float], Optional[float]]:
    best: Optional[float] = None
    final: Optional[float] = None
    for row in sorted(loss_series, key=lambda item: int(item.get("epoch") or 0)):
        train_value = row.get("trainLoss")
        valid_value = row.get("validLoss")
        for value in (train_value, valid_value):
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if best is None or numeric < best:
                best = numeric
        train_final = row.get("trainLoss", row.get("loss"))
        if train_final is not None:
            try:
                final = float(train_final)
            except (TypeError, ValueError):
                pass
    status = str(job_status or "").lower()
    if status not in {"completed", "succeeded", "success"}:
        final = None
    return best, final


def _extract_eval_metric_columns(summary_json: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """从 summary_json / aggregate 提取 success_rate 与 average_score。"""
    payload = summary_json if isinstance(summary_json, dict) else {}
    success_rate: Optional[float] = None
    average_score: Optional[float] = None

    for key in ("successRate", "success_rate", "everSuccessRate"):
        raw = payload.get(key)
        if raw is not None:
            try:
                success_rate = float(raw)
                break
            except (TypeError, ValueError):
                continue

    stats = payload.get("successStats") if isinstance(payload.get("successStats"), dict) else {}
    if success_rate is None and stats.get("successRate") is not None:
        try:
            success_rate = float(stats["successRate"])
        except (TypeError, ValueError):
            pass

    metric_results = payload.get("metricResults") if isinstance(payload.get("metricResults"), dict) else {}
    scores: list[float] = []
    for value in metric_results.values():
        if isinstance(value, dict):
            for score_key in ("average", "mean", "score", "value"):
                if value.get(score_key) is not None:
                    try:
                        scores.append(float(value[score_key]))
                    except (TypeError, ValueError):
                        pass
        elif value is not None:
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                pass
    run_metrics = payload.get("runMetrics") if isinstance(payload.get("runMetrics"), dict) else {}
    for value in run_metrics.values():
        if value is not None:
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                pass
    if scores:
        average_score = sum(scores) / len(scores)
    return success_rate, average_score


def _map_asset_status(entry: dict[str, Any]) -> str:
    checkpoint_path = str(entry.get("checkpointPath") or "").strip()
    if checkpoint_path and Path(checkpoint_path).is_file():
        return "ready"
    raw = str(entry.get("status") or entry.get("displayStatus") or "generating").lower()
    if raw in {"available", "completed"}:
        return "ready"
    if raw in {"ready", "generating", "waiting", "failed", "deleted", "superseded", "pending"}:
        if raw == "pending":
            return "generating"
        if raw == "waiting":
            return "generating"
        return raw
    return "generating"


def _infer_act_schema_from_yaml(train_job_dir: Path) -> dict[str, Any]:
    for rel in ("config/act_adapted.yaml", "checkpoints/act/config/act_adapted.yaml"):
        path = train_job_dir / rel
        if not path.is_file():
            continue
        try:
            import yaml

            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        low_dim_keys = cfg.get("low_dim_keys") or []
        image_keys = cfg.get("image_keys") or []
        action_dim = cfg.get("action_dim")
        schema: dict[str, Any] = {}
        joint_keys = {"robot0_joint_pos", "robot0_joint_pos_rel"}
        if isinstance(low_dim_keys, list) and joint_keys.intersection(set(low_dim_keys)):
            schema.update(
                {
                    "trainedActionMode": cfg.get("trained_action_mode") or cfg.get("action_mode") or "joint_delta_derived",
                    "actionMode": cfg.get("action_mode") or cfg.get("trained_action_mode") or "joint_delta_derived",
                    "evalExecutor": cfg.get("eval_executor") or "joint_position",
                    "controllerType": cfg.get("controller_type") or "JOINT_POSITION",
                    "actionKey": cfg.get("action_key") or "actions",
                    "preferredPolicySchemaId": cfg.get("preferred_policy_schema_id") or cfg.get("policy_schema_id"),
                }
            )
        if action_dim is not None:
            schema["actionDim"] = action_dim
        if low_dim_keys:
            schema["lowDimKeys"] = low_dim_keys
        if image_keys:
            schema["imageKeys"] = image_keys
        if cfg.get("low_dim_dim") is not None:
            schema["lowDimDim"] = cfg.get("low_dim_dim")
        if cfg.get("robot") or cfg.get("robot_type"):
            schema["robotType"] = cfg.get("robot") or cfg.get("robot_type")
        return schema
    return {}


def _infer_dp_schema_from_adaptation(train_job_dir: Path) -> dict[str, Any]:
    payload = _read_json(train_job_dir / "artifacts" / "training_adaptation.json")
    if not payload:
        return {}
    model_adaptation = payload.get("modelAdaptation") if isinstance(payload.get("modelAdaptation"), dict) else {}
    output_cfg = model_adaptation.get("outputConfig") if isinstance(model_adaptation.get("outputConfig"), dict) else {}
    input_cfg = model_adaptation.get("inputConfig") if isinstance(model_adaptation.get("inputConfig"), dict) else {}
    dp_config = payload.get("configPatch") if isinstance(payload.get("configPatch"), dict) else {}
    nested_dp = dp_config.get("dpConfig") if isinstance(dp_config.get("dpConfig"), dict) else {}

    low_dim_keys = input_cfg.get("low_dim_keys") or nested_dp.get("low_dim_keys") or []
    action_space = str(output_cfg.get("action_space") or output_cfg.get("actionSpace") or "").lower()
    action_dim = output_cfg.get("action_dim") or nested_dp.get("action_dim")

    schema: dict[str, Any] = {}
    if action_dim is not None:
        schema["actionDim"] = action_dim
    if low_dim_keys:
        schema["lowDimKeys"] = low_dim_keys

    joint_keys = {"robot0_joint_pos", "robot0_joint_pos_rel"}
    if action_space in {"joint_delta", "joint_position", "joint"} or (
        isinstance(low_dim_keys, list) and joint_keys.intersection(set(low_dim_keys))
    ) or action_dim == 8:
        schema.update(
            {
                "trainedActionMode": "joint_delta",
                "actionMode": "joint_delta",
                "evalExecutor": "joint_position",
                "controllerType": "JOINT_POSITION",
                "actionKey": "joint_actions",
                "gripperActionKey": "gripper_actions",
            }
        )
    elif action_space in {"delta_pose", "eef_delta", "osc_pose_delta_eef", "eef_pose"} or (
        isinstance(low_dim_keys, list) and {"robot0_eef_pos", "robot0_eef_quat"}.intersection(set(low_dim_keys))
    ):
        schema.update(
            {
                "trainedActionMode": "osc_pose_delta_eef",
                "actionMode": "osc_pose_delta_eef",
                "evalExecutor": "osc_pose",
                "controllerType": "OSC_POSE",
                "actionKey": "actions",
            }
        )
    return schema


def _infer_dp_schema_from_yaml(train_job_dir: Path) -> dict[str, Any]:
    for rel in ("config/dp_adapted.yaml", "checkpoints/diffusion_policy/config/dp_adapted.yaml"):
        path = train_job_dir / rel
        if not path.is_file():
            continue
        try:
            import yaml

            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        low_dim_keys = cfg.get("low_dim_keys") or []
        action_dim = cfg.get("action_dim")
        schema: dict[str, Any] = {}
        joint_keys = {"robot0_joint_pos", "robot0_joint_pos_rel"}
        if isinstance(low_dim_keys, list) and joint_keys.intersection(set(low_dim_keys)):
            schema.update(
                {
                    "trainedActionMode": "joint_delta",
                    "actionMode": "joint_delta",
                    "evalExecutor": "joint_position",
                    "controllerType": "JOINT_POSITION",
                    "actionKey": cfg.get("action_key") or "joint_actions",
                    "gripperActionKey": cfg.get("gripper_action_key") or "gripper_actions",
                }
            )
        elif isinstance(low_dim_keys, list) and {"robot0_eef_pos", "robot0_eef_quat"}.intersection(set(low_dim_keys)):
            schema.update(
                {
                    "trainedActionMode": "osc_pose_delta_eef",
                    "actionMode": "osc_pose_delta_eef",
                    "evalExecutor": "osc_pose",
                    "controllerType": "OSC_POSE",
                    "actionKey": cfg.get("action_key") or "actions",
                }
            )
        if action_dim is not None:
            schema["actionDim"] = action_dim
        return schema
    return {}


def _registry_entry_to_model_asset_row(
    entry: dict[str, Any],
    *,
    train_job_id: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = str(entry.get("checkpointPath") or "").strip()
    path_obj = Path(checkpoint_path) if checkpoint_path else None
    sha256, size_bytes = _file_digest(path_obj) if path_obj else (None, None)

    asset_type = str(entry.get("checkpointKind") or "epoch").lower()
    epoch_raw = entry.get("checkpointEpoch")
    epoch = int(epoch_raw) if epoch_raw is not None else None

    manifest = {
        key: entry.get(key)
        for key in (
            "displayName",
            "checkpointMetricName",
            "checkpointMetricValue",
            "framework",
            "trainingBackend",
            "taskType",
            "taskTemplateId",
            "datasetDisplayName",
            "manifestPath",
            "isPlaceholder",
            "canEvaluate",
            "displayStatus",
            "evalExecutor",
            "trainedActionMode",
            "actionMode",
            "controllerType",
            "actionSchema",
            "observationSchema",
            "controllerSchema",
            "sideChannelSchema",
            "actionKey",
            "gripperActionKey",
            "actionDim",
            "lowDimKeys",
            "lowDimDim",
            "imageKeys",
            "preferredPolicySchemaId",
            "robotType",
            "canEvaluateReason",
            "structureConfig",
        )
        if entry.get(key) is not None
    }

    metrics_json: dict[str, Any] = {}
    if entry.get("checkpointMetricName") is not None:
        metrics_json["checkpointMetricName"] = entry.get("checkpointMetricName")
    if entry.get("checkpointMetricValue") is not None:
        metrics_json["checkpointMetricValue"] = entry.get("checkpointMetricValue")

    model_name = str(entry.get("displayName") or entry.get("name") or entry.get("modelAssetId") or "")

    return {
        "model_asset_id": str(entry.get("modelAssetId") or entry.get("id") or ""),
        "train_job_id": train_job_id,
        "dataset_id": str(entry.get("sourceDatasetId") or status.get("datasetId") or "") or None,
        "model_name": model_name,
        "model_type": str(entry.get("modelType") or status.get("downstreamModelType") or "") or None,
        "asset_type": asset_type,
        "checkpoint_kind": asset_type,
        "epoch": epoch,
        "storage_uri": path_to_storage_uri(checkpoint_path) if checkpoint_path else None,
        "manifest_json": manifest,
        "metrics_json": metrics_json or None,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "status": _map_asset_status(entry),
    }


def _upsert_training_metric_summary(
    db: Session,
    *,
    job_id: str,
    normalized: dict[str, Any],
    status: dict[str, Any],
) -> None:
    loss_series = normalized.get("lossSeries") if isinstance(normalized.get("lossSeries"), list) else []
    best_loss, final_loss = _summarize_loss_series(loss_series, job_status=str(status.get("status") or ""))
    current_loss = normalized.get("loss")
    if current_loss is not None:
        try:
            current_loss = float(current_loss)
        except (TypeError, ValueError):
            current_loss = None

    row = db.query(TrainingMetricSummary).filter(TrainingMetricSummary.job_id == job_id).one_or_none()
    payload = {
        "current_epoch": int(normalized.get("epoch") or status.get("epoch") or 0),
        "total_epochs": int(normalized.get("totalEpochs") or status.get("totalEpochs") or 0),
        "progress": float(normalized.get("progress") or status.get("progress") or 0.0),
        "current_loss": current_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "loss_series": loss_series,
        "updated_at": _utc_now(),
    }
    if row is None:
        db.add(TrainingMetricSummary(job_id=job_id, **payload))
    else:
        for key, value in payload.items():
            setattr(row, key, value)


def _enrich_registry_entry_from_train_config(entry: dict[str, Any], train_config: dict[str, Any]) -> dict[str, Any]:
    row = dict(entry)
    dp_config = train_config.get("dpConfig") if isinstance(train_config.get("dpConfig"), dict) else {}
    act_config = train_config.get("actConfig") if isinstance(train_config.get("actConfig"), dict) else {}
    for key, sources in (
        ("trainedActionMode", (train_config.get("trainedActionMode"), train_config.get("actionMode"), dp_config.get("trained_action_mode"), dp_config.get("action_mode"), act_config.get("trained_action_mode"), act_config.get("action_mode"))),
        ("actionMode", (train_config.get("actionMode"), train_config.get("trainedActionMode"), dp_config.get("action_mode"), act_config.get("action_mode"))),
        ("evalExecutor", (train_config.get("evalExecutor"), dp_config.get("eval_executor"), act_config.get("eval_executor"))),
        ("controllerType", (train_config.get("controllerType"), dp_config.get("controller_type"), act_config.get("controller_type"))),
        ("actionSchema", (train_config.get("actionSchema"), dp_config.get("action_schema"), act_config.get("action_schema"))),
        ("observationSchema", (train_config.get("observationSchema"), dp_config.get("observation_schema"), act_config.get("observation_schema"))),
        ("controllerSchema", (train_config.get("controllerSchema"), dp_config.get("controller_schema"), act_config.get("controller_schema"))),
        ("sideChannelSchema", (train_config.get("sideChannelSchema"), dp_config.get("side_channel_schema"), act_config.get("side_channel_schema"))),
        ("actionKey", (dp_config.get("action_key"), act_config.get("action_key"))),
        ("gripperActionKey", (dp_config.get("gripper_action_key"), act_config.get("gripper_action_key"))),
        ("actionDim", (train_config.get("actionDim"), dp_config.get("action_dim"), act_config.get("action_dim"))),
        ("lowDimDim", (train_config.get("lowDimDim"), dp_config.get("low_dim_dim"), act_config.get("low_dim_dim"), act_config.get("state_dim"))),
        ("lowDimKeys", (train_config.get("lowDimKeys"), dp_config.get("low_dim_keys"), act_config.get("low_dim_keys"))),
        ("imageKeys", (train_config.get("imageKeys"), dp_config.get("image_keys"), act_config.get("image_keys"))),
        ("preferredPolicySchemaId", (train_config.get("preferredPolicySchemaId"), dp_config.get("preferred_policy_schema_id"), act_config.get("preferred_policy_schema_id"), act_config.get("policy_schema_id"))),
        ("robotType", (train_config.get("robot"), train_config.get("robotType"), act_config.get("robot"), act_config.get("robot_type"))),
    ):
        if row.get(key):
            continue
        for source in sources:
            if source not in (None, "", [], {}):
                row[key] = source
                break
    return row


def _enrich_registry_entry_from_checkpoint(entry: dict[str, Any]) -> dict[str, Any]:
    from app.services.policy_schema_resolver import extract_model_schema_fields_from_checkpoint
    from app.services.model_asset_checkpoint_resolver import resolve_local_checkpoint_path

    checkpoint_path = str(entry.get("checkpointPath") or "").strip()
    if not checkpoint_path:
        return entry
    local_path = resolve_local_checkpoint_path(
        asset=entry,
        path_hint=checkpoint_path,
        model_asset_id=str(entry.get("modelAssetId") or entry.get("id") or ""),
    )
    resolved = local_path or checkpoint_path
    try:
        fields = extract_model_schema_fields_from_checkpoint(resolved)
    except Exception:
        return entry
    if not fields:
        return entry
    row = dict(entry)
    for key, value in fields.items():
        if value not in (None, "", [], {}):
            row[key] = value
    return row


def _enrich_registry_entry_from_job_context(
    entry: dict[str, Any],
    *,
    train_config: dict[str, Any],
    train_job_dir: Path,
) -> dict[str, Any]:
    row = _enrich_registry_entry_from_train_config(entry, train_config)
    for key, value in _infer_dp_schema_from_adaptation(train_job_dir).items():
        if row.get(key) in (None, "", {}):
            row[key] = value
    for key, value in _infer_act_schema_from_yaml(train_job_dir).items():
        if row.get(key) in (None, "", {}):
            row[key] = value
    for key, value in _infer_dp_schema_from_yaml(train_job_dir).items():
        if row.get(key) in (None, "", {}):
            row[key] = value
    row = _enrich_registry_entry_from_checkpoint(row)
    return row


def _upsert_model_assets(
    db: Session,
    *,
    train_job_id: str,
    entries: list[dict[str, Any]],
    status: dict[str, Any],
) -> int:
    job_row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == train_job_id).one_or_none()
    if job_row is not None and job_row.status == "deleted":
        return 0

    seen_ids: set[str] = set()
    upserted = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        asset_id = str(entry.get("modelAssetId") or entry.get("id") or "").strip()
        if not asset_id:
            continue
        seen_ids.add(asset_id)
        row_data = _registry_entry_to_model_asset_row(entry, train_job_id=train_job_id, status=status)
        if job_row is not None and job_row.project_id and not row_data.get("project_id"):
            row_data["project_id"] = job_row.project_id
        row = db.query(ModelAsset).filter(ModelAsset.model_asset_id == asset_id).one_or_none()
        now = _utc_now()
        if row is None:
            db.add(ModelAsset(created_at=now, updated_at=now, **row_data))
            upserted += 1
        else:
            for key, value in row_data.items():
                setattr(row, key, value)
            row.updated_at = now
            upserted += 1

    stale = (
        db.query(ModelAsset)
        .filter(
            ModelAsset.train_job_id == train_job_id,
            ModelAsset.status != "deleted",
        )
        .all()
    )
    for row in stale:
        if row.model_asset_id not in seen_ids:
            row.status = "deleted"
            row.updated_at = _utc_now()
    return upserted


def _resolve_train_job_dir(job_id: str, runtime_path: Optional[str] = None) -> Optional[Path]:
    if runtime_path:
        path = resolve_runtime_reference(runtime_path)
        if path.is_dir():
            return path.resolve()
    candidate = TRAINING_JOBS_ROOT / job_id
    if candidate.is_dir():
        return candidate.resolve()
    return None


def _collect_registry_entries(train_job_dir: Path, train_job_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from app.services.workspace_model_asset_service import _resolve_job_backend_context
    from app.services.checkpoint_registry import discover_checkpoints, list_training_job_detail_registry_entries

    status, train_config, manifest, resolved_backend, framework_label, model_type = (
        _resolve_job_backend_context(train_job_dir, train_job_id)
    )
    registry_status = dict(status)
    if str(registry_status.get("status") or "").lower() in {"canceled", "cancelled"}:
        if any(record.kind == "final" for record in discover_checkpoints(train_job_dir)):
            registry_status = {**registry_status, "status": "completed"}
    entries = list_training_job_detail_registry_entries(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest=manifest,
        train_config=train_config,
        status=registry_status,
        resolved_backend=resolved_backend,
        framework_label=framework_label,
        model_type=model_type,
    )
    return entries, registry_status, train_config, manifest


def sync_training_job_from_runtime(
    job_id: str,
    *,
    overwrite_artifacts: bool = False,
) -> dict[str, Any]:
    """读取 runtime 训练目录，同步 workspace_jobs / training_metric_summary / model_assets。"""
    from app.core.db_session import db_session_scope
    from app.services.workspace_job_service import _sync_job_record, infer_job_identity
    from app.services.training_metrics import normalized_training_metrics
    from app.services.training_job_status import enrich_and_persist_training_job_status

    started = time.perf_counter()
    result: dict[str, Any] = {
        "jobId": job_id,
        "ok": True,
        "steps": {},
        "warnings": [],
    }

    def _step(name: str, ok: bool, *, warning: Optional[str] = None, extra: Optional[dict[str, Any]] = None) -> None:
        payload: dict[str, Any] = {
            "ok": ok,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if warning:
            payload["warning"] = warning
            result["warnings"].append(warning)
        if extra:
            payload.update(extra)
        result["steps"][name] = payload
        if not ok:
            result["ok"] = False

    candidate = (job_id or "").strip()
    if not candidate or not TRAIN_JOB_DIR_PATTERN.match(candidate):
        _step("validate", False, warning="invalid job id")
        return result

    runtime_path: Optional[str] = None
    try:
        with db_session_scope(label=f"sync-read-{candidate}") as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate).one_or_none()
            if row is not None:
                runtime_path = row.runtime_path
        _step("load_runtime_path", True)
    except Exception as exc:
        _step("load_runtime_path", False, warning=str(exc))

    train_job_dir = _resolve_train_job_dir(candidate, runtime_path)
    if train_job_dir is None:
        _step("resolve_runtime_dir", False, warning="runtime directory not found")
        return result
    _step("resolve_runtime_dir", True)

    status_data = _read_json(train_job_dir / "status.json")
    original_terminal = normalize_api_training_status(str(status_data.get("status") or "")) in TERMINAL_STATUSES
    original_completed = normalize_api_training_status(str(status_data.get("status") or "")) == "completed"

    train_config_early = _read_json(train_job_dir / "config" / "train_config.json")
    execution_mode = str(
        status_data.get("executionMode") or train_config_early.get("executionMode") or ""
    ).lower()
    api_status_early = normalize_api_training_status(str(status_data.get("status") or ""))
    if execution_mode == "remote_ssh" and api_status_early in {"running", "starting"}:
        try:
            from app.services.training_remote_runner import reconcile_remote_training_job_runtime

            reconcile_remote_training_job_runtime(candidate)
            status_data = _read_json(train_job_dir / "status.json") or status_data
            _step("reconcile_remote_runtime", True)
        except Exception as exc:
            _step("reconcile_remote_runtime", False, warning=str(exc))

    try:
        from app.services.training_service import _reconcile_stale_running_training_job

        status_data = _reconcile_stale_running_training_job(candidate, train_job_dir, status_data)
        _step("reconcile_stale_status", True)
    except Exception as exc:
        _step("reconcile_stale_status", False, warning=str(exc))

    if status_data.get("deleted") is True or str(status_data.get("lifecycleStatus") or "").lower() == "deleted":
        try:
            with db_session_scope(label=f"sync-delete-{candidate}") as db:
                row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate).one_or_none()
                if row is not None:
                    row.status = "deleted"
                    row.updated_at = _utc_now()
            _step("mark_deleted", True)
        except Exception as exc:
            _step("mark_deleted", False, warning=str(exc))
        return result

    try:
        from app.services.training_node_service import enrich_training_node_display_fields

        train_config_for_display = _read_json(train_job_dir / "config" / "train_config.json")
        status_data = enrich_training_node_display_fields(status_data, train_config=train_config_for_display)
        status_data = enrich_and_persist_training_job_status(candidate, train_job_dir, status_data)
        _step("enrich_runtime_status", True)
    except Exception as exc:
        _step("enrich_runtime_status", False, warning=str(exc))
        logger.warning("sync enrich failed job_id=%s: %s", candidate, exc)

    if original_completed and normalize_api_training_status(str(status_data.get("status") or "")) == "running":
        from app.services.training_job_status import infer_training_job_completed

        if infer_training_job_completed(status_data, train_job_dir=train_job_dir):
            status_data = dict(status_data)
            status_data["status"] = "completed"
            status_data["progress"] = 1.0
    if original_terminal and normalize_api_training_status(str(status_data.get("status") or "")) == "running":
        from app.services.training_job_status import infer_training_job_completed

        if infer_training_job_completed(status_data, train_job_dir=train_job_dir):
            restored = _read_json(train_job_dir / "status.json")
            if normalize_api_training_status(str(restored.get("status") or "")) in TERMINAL_STATUSES:
                status_data = restored

    normalized = normalized_training_metrics(train_job_dir, status_data)
    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    metadata: dict[str, Any] = {"trainConfig": train_config} if train_config else {}
    manifest = _read_json(train_job_dir / "artifacts" / "dataset_manifest.json")
    if manifest:
        metadata["datasetManifest"] = manifest

    inferred = infer_job_identity(candidate)
    if inferred is None:
        task_type = str(manifest.get("taskType") or status_data.get("taskType") or "unknown")
        runner = "train_bc.py"
    else:
        _, inferred_task_type, runtime_rel, runner = inferred
        task_type = str(
            manifest.get("taskType") or status_data.get("taskType") or inferred_task_type or "unknown"
        )
        runtime_path = runtime_rel
    _step("parse_runtime_files", True)

    asset_entries: list[dict[str, Any]] = []
    asset_status: dict[str, Any] = dict(status_data)
    try:
        entries, reg_status, reg_train_config, _ = _collect_registry_entries(train_job_dir, candidate)
        asset_entries = [
            _enrich_registry_entry_from_job_context(
                entry,
                train_config=reg_train_config,
                train_job_dir=train_job_dir,
            )
            for entry in entries
        ]
        asset_status = reg_status or status_data
        _step("collect_model_assets", True, extra={"count": len(asset_entries)})
    except Exception as exc:
        _step("collect_model_assets", False, warning=str(exc))

    from app.services.workspace_job_service import _collect_artifact_specs

    artifact_specs = _collect_artifact_specs(
        job_id=candidate,
        job_type="training",
        task_type=task_type,
        job_root=train_job_dir,
    )

    import time as _time

    for attempt in range(3):
        try:
            with db_session_scope(label=f"sync-job-{candidate}") as db:
                _sync_job_record(
                    db,
                    job_id=candidate,
                    job_type="training",
                    task_type=task_type,
                    runtime_path=str(train_job_dir),
                    runner=runner,
                    metadata=metadata,
                    overwrite=overwrite_artifacts,
                    artifact_specs=artifact_specs,
                )

                row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate).one_or_none()
                if row is not None:
                    api_status = normalize_api_training_status(str(status_data.get("status") or "unknown"))
                    row.status = api_status
                    if api_status == "running":
                        row.finished_at = None
                    elif api_status == "completed" and row.finished_at is None:
                        row.finished_at = _utc_now()
                    merged_metrics = dict(row.metrics_json or {})
                    merged_metrics.update(
                        {
                            "epoch": normalized.get("epoch"),
                            "totalEpochs": normalized.get("totalEpochs"),
                            "loss": normalized.get("loss"),
                            "progress": normalized.get("progress"),
                            "lossHistory": normalized.get("lossSeries", []),
                            "lossSeries": normalized.get("lossSeries", []),
                            "bestLoss": normalized.get("bestLoss"),
                            "finalLoss": normalized.get("finalLoss") if api_status == "completed" else None,
                            "datasetId": status_data.get("datasetId"),
                            "datasetName": status_data.get("datasetName"),
                            "downstreamModelType": status_data.get("downstreamModelType"),
                            "trainingBackend": status_data.get("trainingBackend"),
                            "modelAssetId": status_data.get("modelAssetId"),
                            "checkpointPath": status_data.get("checkpointPath"),
                            "checkpointExists": status_data.get("checkpointExists"),
                            "dataFormat": status_data.get("dataFormat"),
                            "deviceLabel": status_data.get("deviceLabel"),
                            "trainingNodeId": status_data.get("trainingNodeId") or train_config_for_display.get("trainingNodeId"),
                            "trainingNodeDisplayName": status_data.get("trainingNodeDisplayName")
                            or train_config_for_display.get("trainingNodeDisplayName"),
                            "taskName": status_data.get("taskName"),
                            "message": status_data.get("message"),
                        }
                    )
                    row.metrics_json = {k: v for k, v in merged_metrics.items() if v is not None}
                    if status_data.get("taskName") and not row.task_name:
                        row.task_name = str(status_data.get("taskName"))

                _upsert_training_metric_summary(db, job_id=candidate, normalized=normalized, status=status_data)
            _step("sync_status_metrics", True, extra={"status": status_data.get("status"), "attempt": attempt + 1})
            break
        except Exception as exc:
            if attempt >= 2:
                _step("sync_status_metrics", False, warning=str(exc))
                logger.warning("sync_training_job status/metrics failed job_id=%s: %s", candidate, exc)
            else:
                _time.sleep(0.3 * (attempt + 1))

    if asset_entries:
        try:
            with db_session_scope(label=f"sync-assets-{candidate}") as db:
                _upsert_model_assets(
                    db,
                    train_job_id=candidate,
                    entries=asset_entries,
                    status=asset_status,
                )
            _step("sync_model_assets", True, extra={"count": len(asset_entries)})
        except Exception as exc:
            _step("sync_model_assets", False, warning=str(exc))
            logger.warning("sync model_assets failed job_id=%s: %s", candidate, exc)

    result["elapsedMs"] = round((time.perf_counter() - started) * 1000, 1)
    try:
        from app.services.platform_stage2_hooks import after_workspace_job_sync

        after_workspace_job_sync(candidate)
    except Exception as exc:
        logger.warning("sync_training stage2 hook failed job_id=%s: %s", candidate, exc)
    return result


def _sync_model_assets_for_train_job(
    db: Session,
    train_job_dir: Path,
    train_job_id: str,
    *,
    overwrite: bool = False,
) -> None:
    try:
        entries, reg_status, train_config, _ = _collect_registry_entries(train_job_dir, train_job_id)
        entries = [
            _enrich_registry_entry_from_job_context(entry, train_config=train_config, train_job_dir=train_job_dir)
            for entry in entries
        ]
        _upsert_model_assets(db, train_job_id=train_job_id, entries=entries, status=reg_status)
    except Exception as exc:
        logger.warning("sync model_assets failed job_id=%s: %s", train_job_id, exc)


def sync_training_assets_from_runtime(job_id: str, *, overwrite_artifacts: bool = False) -> None:
    """Backfill model_assets / training_metric_summary for deleted training jobs without reviving job status."""
    candidate = (job_id or "").strip()
    if not candidate or not TRAIN_JOB_DIR_PATTERN.match(candidate):
        return

    from app.services.training_metrics import normalized_training_metrics
    from app.services.training_job_status import enrich_and_persist_training_job_status

    train_job_dir = _resolve_train_job_dir(candidate)
    if train_job_dir is None:
        return

    status_data = _read_json(train_job_dir / "status.json")
    status_data = enrich_and_persist_training_job_status(candidate, train_job_dir, status_data)
    normalized = normalized_training_metrics(train_job_dir, status_data)

    try:
        with SessionLocal() as db:
            _upsert_training_metric_summary(db, job_id=candidate, normalized=normalized, status=status_data)
            _sync_model_assets_for_train_job(
                db, train_job_dir, candidate, overwrite=overwrite_artifacts
            )
            db.commit()
    except Exception as exc:
        logger.warning("sync_training_assets_from_runtime failed job_id=%s: %s", candidate, exc)


def _resolve_eval_job_dir(job_id: str, runtime_path: Optional[str] = None) -> Optional[Path]:
    if runtime_path:
        path = resolve_runtime_reference(runtime_path)
        if path.is_dir():
            return path.resolve()
    if job_id.startswith("ct_eval_") or job_id.startswith("eval_"):
        for runtime_root in (RUNTIME_ROOT,):
            ct_root = runtime_root / "cable_threading" / "jobs" / job_id
            if ct_root.is_dir():
                return ct_root.resolve()
    if job_id.startswith("isaac_eval_"):
        for runtime_root in (RUNTIME_ROOT,):
            isaac_root = runtime_root / "isaac_lab" / "jobs" / job_id
            if isaac_root.is_dir():
                return isaac_root.resolve()
    for runtime_root in (RUNTIME_ROOT,):
        candidate = runtime_root / "evaluations" / "jobs" / job_id
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _is_eval_job_id(job_id: str) -> bool:
    text = (job_id or "").strip()
    return text.startswith("eval_") or text.startswith("isaac_eval_") or text.startswith("ct_eval_")


def _iso_datetime(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _apply_eval_health_to_db(
    job_id: str,
    health: dict[str, Any],
    *,
    db_row_status: Optional[str] = None,
) -> None:
    """runtime 不可达时，仅根据 health 判定写回 workspace_jobs / eval_metric_summary。"""
    actual_status = str(health.get("actualStatus") or db_row_status or "unknown").strip()
    health_reason = str(health.get("reason") or "").strip()
    declared = str(health.get("declaredStatus") or db_row_status or "").strip().lower()
    if actual_status == declared and actual_status not in TERMINAL_STATUSES:
        return
    if actual_status not in TERMINAL_STATUSES:
        return

    try:
        with SessionLocal() as db:
            job_row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
            if job_row is None or job_row.status == "deleted":
                return
            if job_row.status not in TERMINAL_STATUSES:
                job_row.status = actual_status
            merged_metrics = dict(job_row.metrics_json or {}) if isinstance(job_row.metrics_json, dict) else {}
            if health_reason:
                merged_metrics["runtimeHealthReason"] = health_reason
            merged_metrics["runtimeHealth"] = {
                key: health.get(key)
                for key in (
                    "actualStatus",
                    "isProcessAlive",
                    "matchedPids",
                    "lastRuntimeUpdateAt",
                    "staleSeconds",
                    "jobAgeSeconds",
                    "reason",
                )
                if health.get(key) is not None
            }
            job_row.metrics_json = merged_metrics
            if actual_status == "failed":
                job_row.error_message = health_reason or job_row.error_message or "评测失败"
            merged_meta = dict(job_row.metadata_json or {}) if isinstance(job_row.metadata_json, dict) else {}
            merged_meta["runtimeHealth"] = health
            job_row.metadata_json = merged_meta
            job_row.updated_at = _utc_now()

            summary_row = db.query(EvalMetricSummary).filter(EvalMetricSummary.job_id == job_id).one_or_none()
            if summary_row is not None:
                summary_payload = dict(summary_row.summary_json or {}) if isinstance(summary_row.summary_json, dict) else {}
                summary_payload["status"] = actual_status
                if health_reason:
                    summary_payload["message"] = health_reason
                summary_row.summary_json = summary_payload
                summary_row.updated_at = _utc_now()
            db.commit()
    except Exception as exc:
        logger.warning("_apply_eval_health_to_db failed job_id=%s: %s", job_id, exc)


def sync_eval_job_from_runtime(job_id: str, *, overwrite_artifacts: bool = False) -> None:
    """同步评测任务指标摘要到 eval_metric_summary。"""
    candidate = (job_id or "").strip()
    if not candidate or not _is_eval_job_id(candidate):
        return

    from app.services.workspace_job_service import _sync_job_record, infer_job_identity
    from app.services.evaluation.evaluation_runtime_health import reconcile_evaluation_runtime_health

    runtime_path: Optional[str] = None
    db_row_status: Optional[str] = None
    db_created_at: Optional[str] = None
    db_started_at: Optional[str] = None
    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate).one_or_none()
            if row is not None:
                if row.status == "deleted":
                    return
                db_row_status = row.status
                runtime_path = row.runtime_path
                db_created_at = _iso_datetime(row.created_at)
                db_started_at = _iso_datetime(row.started_at)
    except Exception:
        pass

    job_root = _resolve_eval_job_dir(candidate, runtime_path)

    health = reconcile_evaluation_runtime_health(
        candidate,
        str(job_root) if job_root else runtime_path,
        declared_status=db_row_status,
        created_at=db_created_at,
        started_at=db_started_at,
        apply=bool(job_root),
    )

    if job_root is None:
        _apply_eval_health_to_db(candidate, health, db_row_status=db_row_status)
        return

    status_data = (
        _read_json(job_root / "live" / "status.json")
        or _read_json(job_root / "status.json")
        or _read_json(job_root / "metadata" / "status.json")
    )
    from app.services.runtime_job_lifecycle import is_job_deleted

    if is_job_deleted(status_data):
        return

    aggregate = _read_json(job_root / "results" / "aggregate_result.json")
    eval_request = _read_json(job_root / "metadata" / "evaluation_request.json")
    if not eval_request:
        context = _read_json(job_root / "metadata" / "evaluation_context.json")
        nested = context.get("evaluationRequest") if isinstance(context.get("evaluationRequest"), dict) else {}
        eval_request = nested or context
    results_data = _read_json(job_root / "results" / "eval.results.json")
    model_asset_id = str(
        status_data.get("modelAssetId")
        or eval_request.get("modelAssetId")
        or aggregate.get("modelAssetId")
        or ""
    ).strip() or None

    report_path = job_root / "results" / "aggregate_result.json"
    if not report_path.is_file():
        report_path = job_root / "results" / "eval.results.json"
    report_uri = path_to_storage_uri(report_path) if report_path.is_file() else None

    replay_uri: Optional[str] = None
    replay_info: dict[str, Any] = {}
    if candidate.startswith(("ct_eval_", "eval_", "isaac_eval_")):
        from app.services.evaluation_replay_info import (
            build_evaluation_replay_info,
            resolve_replay_api_prefix,
        )

        replay_info = build_evaluation_replay_info(
            candidate,
            job_root,
            live=status_data,
            results_data=results_data,
            aggregate_file=aggregate,
            status_value=str(status_data.get("status") or ""),
            api_prefix=resolve_replay_api_prefix(candidate),
        )
        replay_items = replay_info.get("replayUris") if isinstance(replay_info.get("replayUris"), list) else []
        replay_uri = replay_info.get("replayUri")
        if replay_uri is None and replay_items:
            first_name = replay_items[0].get("fileName") if isinstance(replay_items[0], dict) else None
            if first_name:
                replay_path = job_root / "videos" / str(first_name)
                if replay_path.is_file():
                    replay_uri = path_to_storage_uri(replay_path)
    if replay_uri is None:
        videos_dir = job_root / "videos"
        if videos_dir.is_dir():
            episode_videos = sorted(
                p for p in videos_dir.glob("episode_*.mp4") if p.is_file()
            )
            if episode_videos:
                replay_uri = path_to_storage_uri(episode_videos[0])
            else:
                videos = sorted(videos_dir.glob("*.mp4"))
                if videos:
                    replay_uri = path_to_storage_uri(videos[0])

    summary_json: dict[str, Any] = {}
    if aggregate:
        summary_json = dict(aggregate)
    elif status_data.get("metrics"):
        summary_json = status_data.get("metrics") if isinstance(status_data.get("metrics"), dict) else {}
    if replay_info:
        summary_json.update(
            {
                key: replay_info.get(key)
                for key in (
                    "requestedEpisodes",
                    "completedEpisodes",
                    "successfulEpisodes",
                    "failedEpisodes",
                    "successRate",
                    "recordedVideoCount",
                    "replayUri",
                    "replayUris",
                    "videoAvailable",
                    "isRepresentativeVideo",
                    "warning",
                    "selectedMetrics",
                )
                if replay_info.get(key) is not None
            }
        )
    metrics_list = eval_request.get("metrics") if isinstance(eval_request.get("metrics"), list) else []
    selected_metric_ids = eval_request.get("selectedMetricIds")
    if isinstance(selected_metric_ids, list) and selected_metric_ids:
        summary_json["selectedMetricIds"] = selected_metric_ids
        summary_json["selectedMetrics"] = selected_metric_ids
    elif metrics_list:
        summary_json["selectedMetricIds"] = metrics_list
        summary_json["selectedMetrics"] = metrics_list
    if isinstance(summary_json.get("metricResults"), dict):
        pass
    elif isinstance(aggregate, dict) and isinstance(aggregate.get("metricResults"), dict):
        summary_json["metricResults"] = aggregate["metricResults"]
    if isinstance(aggregate, dict) and isinstance(aggregate.get("runMetrics"), dict):
        summary_json["runMetrics"] = aggregate["runMetrics"]

    inferred = infer_job_identity(candidate)
    if inferred:
        _, task_type, _, runner = inferred
    else:
        task_type = str(eval_request.get("taskType") or status_data.get("taskType") or "unknown")
        runner = "evaluation_adapter"

    metadata = {"evaluationRequest": eval_request} if eval_request else {}
    task_name = (
        eval_request.get("taskName")
        or eval_request.get("modelName")
        or status_data.get("taskName")
    )

    metrics_payload: dict[str, Any] = dict(status_data.get("metrics") or {}) if isinstance(status_data.get("metrics"), dict) else {}
    for key in (
        "evaluationMode",
        "evaluationType",
        "datasetId",
        "datasetName",
        "modelAssetId",
        "modelName",
        "taskType",
    ):
        if status_data.get(key) is not None:
            metrics_payload[key] = status_data.get(key)
        elif eval_request.get(key) is not None:
            metrics_payload[key] = eval_request.get(key)

    actual_status = str(health.get("actualStatus") or status_data.get("status") or db_row_status or "unknown")
    runtime_status = str(status_data.get("status") or "").strip().lower()
    if health.get("actualStatus") and health["actualStatus"] != runtime_status:
        actual_status = str(health["actualStatus"])
    health_reason = str(health.get("reason") or "").strip()

    from app.services.evaluation.evaluation_progress import resolve_evaluation_progress

    progress_info = resolve_evaluation_progress(
        status=actual_status,
        metrics=metrics_payload,
        summary_json=summary_json,
        runtime_status=status_data,
        job_id=candidate,
        runtime_path=str(job_root),
    )
    for key, value in progress_info.items():
        if value is not None:
            metrics_payload[key] = value
            summary_json[key] = value

    try:
        from app.services.evaluation.success_stats import resolve_success_stats

        success_stats = resolve_success_stats(
            candidate,
            summary_json=summary_json,
            aggregate_result=aggregate if isinstance(aggregate, dict) else None,
            status_json=status_data if isinstance(status_data, dict) else None,
            context_json=eval_request if isinstance(eval_request, dict) else None,
            runtime_path=str(job_root),
            metrics=metrics_payload,
        )
        summary_json["successStats"] = success_stats
    except Exception as exc:
        logger.warning("resolve_success_stats failed job_id=%s: %s", candidate, exc)

    try:
        with SessionLocal() as db:
            _sync_job_record(
                db,
                job_id=candidate,
                job_type="evaluation",
                task_type=task_type,
                runtime_path=str(job_root),
                runner=runner,
                task_name=str(task_name) if task_name else None,
                metadata=metadata,
                overwrite=overwrite_artifacts,
            )

            job_row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate).one_or_none()
            if job_row is not None:
                if job_row.status == "deleted":
                    pass
                elif overwrite_artifacts or job_row.status not in TERMINAL_STATUSES:
                    job_row.status = actual_status
                if task_name and not job_row.task_name:
                    job_row.task_name = str(task_name)
                merged_metrics = dict(job_row.metrics_json or {})
                merged_metrics.update(metrics_payload)
                if health_reason:
                    merged_metrics["runtimeHealthReason"] = health_reason
                merged_metrics["runtimeHealth"] = {
                    key: health.get(key)
                    for key in (
                        "actualStatus",
                        "isProcessAlive",
                        "matchedPids",
                        "lastRuntimeUpdateAt",
                        "staleSeconds",
                        "reason",
                    )
                    if health.get(key) is not None
                }
                job_row.metrics_json = merged_metrics
                if actual_status == "failed":
                    job_row.error_message = health_reason or str(
                        status_data.get("error") or status_data.get("message") or "评测失败"
                    )
                elif status_data.get("error") or status_data.get("message"):
                    if str(status_data.get("status") or "").lower() == "failed":
                        job_row.error_message = str(status_data.get("error") or status_data.get("message"))

                merged_meta = dict(job_row.metadata_json or {}) if isinstance(job_row.metadata_json, dict) else {}
                merged_meta["runtimeHealth"] = health
                job_row.metadata_json = merged_meta

            row = db.query(EvalMetricSummary).filter(EvalMetricSummary.job_id == candidate).one_or_none()
            summary_payload = dict(summary_json or {})
            summary_payload["status"] = actual_status
            if health_reason:
                summary_payload["message"] = health_reason
            success_rate, average_score = _extract_eval_metric_columns(summary_payload)
            payload = {
                "model_asset_id": model_asset_id,
                "summary_json": summary_payload or None,
                "report_uri": report_uri,
                "replay_uri": replay_uri,
                "success_rate": success_rate,
                "average_score": average_score,
                "updated_at": _utc_now(),
            }
            if row is None:
                db.add(EvalMetricSummary(job_id=candidate, **payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            db.commit()
    except Exception as exc:
        logger.warning("sync_eval_job_from_runtime failed job_id=%s: %s", candidate, exc)
        return

    if actual_status in TERMINAL_STATUSES:
        try:
            from app.services.artifact_upload_service import schedule_artifact_upload

            schedule_artifact_upload(candidate)
        except Exception as exc:
            logger.warning("sync_eval_job_from_runtime artifact upload schedule failed job_id=%s: %s", candidate, exc)

    try:
        from app.services.platform_stage2_hooks import after_workspace_job_sync

        after_workspace_job_sync(candidate)
    except Exception as exc:
        logger.warning("sync_eval stage2 hook failed job_id=%s: %s", candidate, exc)


def reindex_runtime_jobs(
    *,
    task_type: Optional[str] = None,
    job_type: Optional[str] = None,
    dry_run: bool = False,
    overwrite: bool = False,
    restore_deleted: bool = False,
) -> dict[str, Any]:
    """扫描 runs，回填 workspace 索引与训练/评测摘要表。"""
    from app.services.workspace_reindex_service import reindex_workspace_all

    return reindex_workspace_all(
        task_type=task_type,
        job_type=job_type,
        dry_run=dry_run,
        overwrite=overwrite,
        restore_deleted=restore_deleted,
    )


def get_training_job_summary_from_db(job_id: str) -> Optional[dict[str, Any]]:
    """从 PostgreSQL 读取训练任务列表/详情摘要（不依赖 runtime 目录可读）。"""
    candidate = (job_id or "").strip()
    if not candidate:
        return None
    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate).one_or_none()
            if row is None or row.job_type != "training" or row.status == "deleted":
                return None
            metrics_row = (
                db.query(TrainingMetricSummary)
                .filter(TrainingMetricSummary.job_id == candidate)
                .one_or_none()
            )
            meta = row.metadata_json if isinstance(row.metadata_json, dict) else {}
            metrics_json = row.metrics_json if isinstance(row.metrics_json, dict) else {}
            train_config = meta.get("trainConfig") if isinstance(meta.get("trainConfig"), dict) else {}
            dataset_manifest = (
                meta.get("datasetManifest") if isinstance(meta.get("datasetManifest"), dict) else {}
            )

            loss_series: list[dict[str, Any]] = []
            if metrics_row and isinstance(metrics_row.loss_series, list):
                loss_series = metrics_row.loss_series

            runtime_available = _runtime_dir_exists(row.runtime_path)
            status = str(row.status or "unknown")
            message = metrics_json.get("message")
            if not runtime_available:
                if status not in TERMINAL_STATUSES:
                    status = "failed"
                    message = message or "运行时工作目录不可用，展示索引数据可能不完整"
                elif not message:
                    message = "运行时工作目录不可用，展示索引数据"

            api_status = normalize_api_training_status(status)
            epoch = int(metrics_row.current_epoch if metrics_row else metrics_json.get("epoch") or 0)
            total_epochs = int(metrics_row.total_epochs if metrics_row else metrics_json.get("totalEpochs") or 0)
            progress_value = float(metrics_row.progress if metrics_row else metrics_json.get("progress") or 0.0)
            final_loss = metrics_row.final_loss if metrics_row else metrics_json.get("finalLoss")
            if api_status != "completed":
                final_loss = None

            dataset_name = (
                metrics_json.get("datasetName")
                or train_config.get("datasetName")
                or meta.get("datasetName")
                or dataset_manifest.get("datasetName")
                or dataset_manifest.get("displayName")
            )
            dataset_id = (
                metrics_json.get("datasetId")
                or train_config.get("datasetId")
                or meta.get("datasetId")
                or dataset_manifest.get("datasetId")
            )

            summary = {
                "trainJobId": candidate,
                "status": api_status,
                "datasetId": dataset_id,
                "datasetName": dataset_name,
                "downstreamModelType": (
                    metrics_json.get("downstreamModelType")
                    or train_config.get("downstreamModelType")
                    or meta.get("downstreamModelType")
                ),
                "trainingBackend": (
                    metrics_json.get("trainingBackend")
                    or train_config.get("trainingBackend")
                    or meta.get("trainingBackend")
                ),
                "createdAt": row.created_at.isoformat() if row.created_at else metrics_json.get("createdAt"),
                "updatedAt": row.updated_at.isoformat() if row.updated_at else metrics_json.get("updatedAt"),
                "startedAt": row.started_at.isoformat() if row.started_at else metrics_json.get("startedAt"),
                "finishedAt": row.finished_at.isoformat() if row.finished_at else metrics_json.get("finishedAt"),
                "checkpointExists": bool(metrics_json.get("checkpointExists")),
                "modelAssetId": metrics_json.get("modelAssetId"),
                "epoch": epoch,
                "totalEpochs": total_epochs,
                "loss": metrics_row.current_loss if metrics_row else metrics_json.get("loss"),
                "bestLoss": metrics_row.best_loss if metrics_row else metrics_json.get("bestLoss"),
                "finalLoss": final_loss,
                "progress": progress_value,
                "lossHistory": loss_series or metrics_json.get("lossHistory") or [],
                "message": message,
                "dataFormat": metrics_json.get("dataFormat") or train_config.get("dataFormat"),
                "deviceLabel": metrics_json.get("deviceLabel") or train_config.get("deviceLabel"),
                "trainingNodeId": (
                    metrics_json.get("trainingNodeId")
                    or train_config.get("trainingNodeId")
                    or meta.get("trainingNodeId")
                ),
                "trainingNodeDisplayName": (
                    metrics_json.get("trainingNodeDisplayName")
                    or train_config.get("trainingNodeDisplayName")
                    or meta.get("trainingNodeDisplayName")
                ),
                "taskName": row.task_name or metrics_json.get("taskName") or train_config.get("taskName"),
                "runtimeAvailable": runtime_available,
            }
            from app.services.training_node_service import enrich_training_node_display_fields

            return enrich_training_node_display_fields(summary, train_config=train_config)
    except Exception as exc:
        logger.warning("get_training_job_summary_from_db failed job_id=%s: %s", candidate, exc)
        return None


def get_training_job_filter_context_batch(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    """批量读取训练任务筛选/展示上下文（轻量，无 metrics 展开）。"""
    unique = [jid for jid in dict.fromkeys((job_id or "").strip() for job_id in job_ids) if jid]
    if not unique:
        return {}
    result: dict[str, dict[str, Any]] = {}
    try:
        with SessionLocal() as db:
            rows = (
                db.query(WorkspaceJob)
                .filter(
                    WorkspaceJob.job_id.in_(unique),
                    WorkspaceJob.job_type == "training",
                    WorkspaceJob.status != "deleted",
                )
                .all()
            )
            for row in rows:
                meta = row.metadata_json if isinstance(row.metadata_json, dict) else {}
                metrics_json = row.metrics_json if isinstance(row.metrics_json, dict) else {}
                train_config = meta.get("trainConfig") if isinstance(meta.get("trainConfig"), dict) else {}
                dataset_manifest = (
                    meta.get("datasetManifest") if isinstance(meta.get("datasetManifest"), dict) else {}
                )
                result[row.job_id] = {
                    "taskName": row.task_name or metrics_json.get("taskName") or train_config.get("taskName"),
                    "datasetName": (
                        metrics_json.get("datasetName")
                        or train_config.get("datasetName")
                        or meta.get("datasetName")
                        or dataset_manifest.get("datasetName")
                        or dataset_manifest.get("displayName")
                    ),
                    "taskType": metrics_json.get("taskType") or train_config.get("taskType") or meta.get("taskType"),
                }
    except Exception as exc:
        logger.warning("get_training_job_filter_context_batch failed: %s", exc)
    return result


def list_training_jobs_from_db(*, sync_stale: bool = True) -> list[dict[str, Any]]:
    """训练中心列表：优先 PostgreSQL；运行中任务可选同步 runtime。"""
    rows: list[dict[str, Any]] = []
    try:
        with SessionLocal() as db:
            jobs = (
                db.query(WorkspaceJob)
                .filter(
                    WorkspaceJob.job_type == "training",
                    WorkspaceJob.status != "deleted",
                )
                .order_by(WorkspaceJob.created_at.desc())
                .all()
            )
            if sync_stale:
                for job in jobs:
                    if job.status not in TERMINAL_STATUSES:
                        sync_training_job_from_runtime(job.job_id)
                db.expire_all()
                jobs = (
                    db.query(WorkspaceJob)
                    .filter(
                        WorkspaceJob.job_type == "training",
                        WorkspaceJob.status != "deleted",
                    )
                    .order_by(WorkspaceJob.created_at.desc())
                    .all()
                )

            for job in jobs:
                summary = get_training_job_summary_from_db(job.job_id)
                if summary:
                    rows.append(summary)
    except Exception as exc:
        logger.warning("list_training_jobs_from_db failed: %s", exc)
    return rows


def finalize_training_job_sync(train_job_id: str) -> None:
    """训练完成或状态收敛后：同步 PostgreSQL 索引并可选异步归档 MinIO。"""
    sync_training_job_from_runtime(train_job_id)
    try:
        from app.services.artifact_upload_service import schedule_artifact_upload

        schedule_artifact_upload(train_job_id)
    except Exception as exc:
        logger.warning("finalize_training_job_sync artifact upload schedule failed job_id=%s: %s", train_job_id, exc)
    try:
        from app.services.platform_stage2_hooks import after_workspace_job_sync

        after_workspace_job_sync(train_job_id)
    except Exception as exc:
        logger.warning("finalize_training stage2 hook failed job_id=%s: %s", train_job_id, exc)


def mark_training_job_deleted_in_db(job_id: str) -> None:
    candidate = (job_id or "").strip()
    if not candidate:
        return
    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate).one_or_none()
            if row is not None:
                row.status = "deleted"
                row.updated_at = _utc_now()
                db.commit()
    except Exception as exc:
        logger.warning("mark_training_job_deleted_in_db failed job_id=%s: %s", candidate, exc)
