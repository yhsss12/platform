from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.platform_paths import is_path_within, platform_paths
from app.services.checkpoint_registry import (
    discover_checkpoints,
    is_training_job_complete,
    list_displayable_registry_assets,
    list_training_job_detail_registry_entries,
    normalize_registry_assets,
    parse_save_policy,
    read_registry,
    register_checkpoint_assets,
    registry_path,
    resolve_training_job_model_assets_list_message,
)
from app.services.model_asset_naming import (
    build_checkpoint_asset_display_name,
    resolve_model_asset_context_label,
    resolve_model_asset_display_name,
)

logger = logging.getLogger(__name__)

MODEL_TYPE_FILTER_PRESETS = ("Robomimic BC", "Diffusion Policy", "BC (PyTorch)", "ACT")


@dataclass
class ModelAssetListTiming:
    db_query_ms: float = 0.0
    count_ms: float = 0.0
    training_assoc_ms: float = 0.0
    dataset_assoc_ms: float = 0.0
    fs_scan_ms: float = 0.0
    json_serialize_ms: float = 0.0
    total_ms: float = 0.0
    cache_hit: bool = False


def log_model_asset_list_timing(timing: ModelAssetListTiming, **context: object) -> None:
    ctx = " ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    suffix = f" {ctx}" if ctx else ""
    logger.info(
        "[api-timing] GET /workspace/model-assets db=%.1fms count=%.1fms training=%.1fms "
        "dataset=%.1fms fs=%.1fms json=%.1fms total=%.1fms cache=%s%s",
        timing.db_query_ms,
        timing.count_ms,
        timing.training_assoc_ms,
        timing.dataset_assoc_ms,
        timing.fs_scan_ms,
        timing.json_serialize_ms,
        timing.total_ms,
        "hit" if timing.cache_hit else "miss",
        suffix,
    )

PROJECT_ROOT = platform_paths.project_root
RUNTIME_ROOT = platform_paths.runs_root
TRAINING_JOBS_ROOT = platform_paths.training_jobs

TASK_TYPE_TO_TEMPLATE: dict[str, str] = {
    "cable_threading": "task_cable_threading_v1",
    "dual_arm_cable_manipulation": "dual_arm_cable_manipulation",
    "isaac_block_stacking": "isaac_block_stacking",
}


def _training_job_roots() -> tuple[Path, ...]:
    return (TRAINING_JOBS_ROOT,)


def _find_training_job_dir(train_job_id: str) -> Path:
    for root in _training_job_roots():
        candidate = root / train_job_id
        if candidate.is_dir():
            return candidate
    return TRAINING_JOBS_ROOT / train_job_id


def _has_training_jobs_root() -> bool:
    return any(root.is_dir() for root in _training_job_roots())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _backend_labels(resolved_backend: str) -> tuple[str, str]:
    if resolved_backend == "diffusion_policy":
        return "Diffusion Policy", "diffusion_policy"
    if resolved_backend == "act":
        return "ACT", "act"
    if resolved_backend == "pi0":
        return "pi0", "pi0"
    if resolved_backend == "torch_bc":
        return "BC (PyTorch)", "bc"
    return "Robomimic BC", "bc"


def _infer_model_type(manifest: dict[str, Any], status: dict[str, Any]) -> str:
    explicit = manifest.get("modelType")
    if explicit:
        return str(explicit)
    value = manifest.get("downstreamModelType") or status.get("downstreamModelType")
    return str(value or "unknown")


def _infer_framework(manifest: dict[str, Any], status: dict[str, Any]) -> str:
    explicit = manifest.get("framework")
    if explicit:
        return str(explicit)
    value = manifest.get("backendType") or manifest.get("trainingBackend") or status.get("trainingBackend")
    if value == "isaac_robomimic_bc":
        return "Robomimic BC"
    return str(value or "unknown")


def _entry_to_asset(
    entry: dict[str, Any],
    *,
    train_job_id: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    model_asset_id = str(entry.get("modelAssetId") or entry.get("id") or "")
    manifest_path = (
        Path(entry.get("manifestPath") or "")
        if entry.get("manifestPath")
        else TRAINING_JOBS_ROOT / train_job_id / "artifacts" / "checkpoint_manifests" / f"{model_asset_id}.json"
    )
    task_type = str(entry.get("taskType") or status.get("taskType") or "")
    task_template_id = str(entry.get("taskTemplateId") or "").strip() or TASK_TYPE_TO_TEMPLATE.get(task_type)
    created_at = str(entry.get("createdAt") or status.get("createdAt") or _utc_now_iso())
    asset_status = str(entry.get("status") or "ready")
    if asset_status == "completed":
        asset_status = "available"

    framework = _infer_framework(entry, status)
    model_type = _infer_model_type(entry, status)
    training_backend = (
        entry.get("trainingBackend")
        or status.get("trainingBackendResolved")
        or status.get("trainingBackend")
    )
    display_name = str(entry.get("displayName") or entry.get("name") or "").strip()
    if entry.get("checkpointKind"):
        context_label = resolve_model_asset_context_label(
            training_task_name=str(entry.get("trainingTaskName") or status.get("taskName") or "") or None,
            dataset_name=str(entry.get("datasetDisplayName") or status.get("datasetName") or "") or None,
            dataset_id=str(entry.get("sourceDatasetId") or status.get("datasetId") or "") or None,
            task_template_id=task_template_id,
            task_type=task_type or None,
        )
        display_name = build_checkpoint_asset_display_name(
            context_label=context_label,
            kind=str(entry.get("checkpointKind") or ""),
            epoch=int(entry["checkpointEpoch"]) if entry.get("checkpointEpoch") is not None else None,
            metric_name=str(entry.get("checkpointMetricName") or "") or None,
        )
    elif not display_name:
        display_name = resolve_model_asset_display_name(
            stored_name=str(entry.get("name") or ""),
            display_name=str(entry.get("displayName") or ""),
            training_task_name=str(entry.get("trainingTaskName") or status.get("taskName") or "") or None,
            dataset_name=str(entry.get("datasetDisplayName") or status.get("datasetName") or "") or None,
            dataset_id=str(entry.get("sourceDatasetId") or status.get("datasetId") or "") or None,
            task_template_id=task_template_id,
            task_type=task_type or None,
            framework=framework,
            model_type=model_type,
            training_backend=str(training_backend or "") or None,
            created_at=created_at,
        )

    return {
        "id": model_asset_id,
        "name": display_name,
        "displayName": display_name,
        "sourceTrainingJobId": str(entry.get("sourceTrainJobId") or train_job_id),
        "sourceDatasetId": entry.get("sourceDatasetId") or status.get("datasetId"),
        "taskTemplateId": task_template_id,
        "modelType": model_type,
        "framework": framework,
        "trainingBackend": entry.get("trainingBackend") or training_backend,
        "backendType": entry.get("backendType") or training_backend,
        "checkpointPath": str(entry.get("checkpointPath") or ""),
        "checkpointKind": entry.get("checkpointKind"),
        "checkpointEpoch": entry.get("checkpointEpoch"),
        "checkpointMetricName": entry.get("checkpointMetricName"),
        "checkpointMetricValue": entry.get("checkpointMetricValue"),
        "datasetDisplayName": entry.get("datasetDisplayName") or status.get("datasetName"),
        "manifestPath": str(manifest_path),
        "version": "v1",
        "status": asset_status,
        "createdAt": created_at,
        "updatedAt": created_at,
        "modelTypeId": entry.get("modelTypeId"),
        "modelTypeName": entry.get("modelTypeName"),
        "baseAlgorithm": entry.get("baseAlgorithm"),
        "adapterId": entry.get("adapterId"),
        "structureConfig": entry.get("structureConfig"),
        "trainingDefaults": entry.get("trainingDefaults"),
        "resolvedModelParams": entry.get("resolvedModelParams"),
        "openpiEnvironment": entry.get("openpiEnvironment"),
        "taskType": task_type or None,
        "artifactKind": entry.get("checkpointKind"),
        "evalExecutor": entry.get("evalExecutor"),
        "trainedActionMode": entry.get("trainedActionMode") or entry.get("actionMode"),
        "actionMode": entry.get("actionMode") or entry.get("trainedActionMode"),
        "controllerType": entry.get("controllerType"),
        "actionSchema": entry.get("actionSchema"),
        "actionKey": entry.get("actionKey"),
        "gripperActionKey": entry.get("gripperActionKey"),
        "actionDim": entry.get("actionDim"),
        "lowDimKeys": entry.get("lowDimKeys"),
        "lowDimDim": entry.get("lowDimDim"),
        "imageKeys": entry.get("imageKeys"),
        "preferredPolicySchemaId": entry.get("preferredPolicySchemaId"),
        "robotType": entry.get("robotType"),
        "canEvaluateReason": entry.get("canEvaluateReason"),
    }


def _sync_registry(train_job_dir: Path, train_job_id: str, status: dict[str, Any], train_config: dict[str, Any]) -> list[dict[str, Any]]:
    registry = read_registry(train_job_dir)
    total_epochs = int(status.get("totalEpochs") or train_config.get("epochs") or 0)
    raw_assets = [item for item in (registry.get("assets") or []) if isinstance(item, dict)]
    if not raw_assets:
        return []
    normalized = normalize_registry_assets(
        raw_assets,
        status=status,
        total_epochs=total_epochs,
        train_job_dir=train_job_dir,
    )
    if normalized != raw_assets:
        registry_payload = {
            "version": 1,
            "sourceTrainJobId": train_job_id,
            "updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assets": normalized,
        }
        reg_path = registry_path(train_job_dir)
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg_path.write_text(
            json.dumps(registry_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return list_displayable_registry_assets(
        normalized,
        status=status,
        total_epochs=total_epochs,
        train_job_dir=train_job_dir,
    )


def _ensure_registry_for_job(train_job_dir: Path, train_job_id: str) -> list[dict[str, Any]]:
    registry = read_registry(train_job_dir)
    status = _read_json(train_job_dir / "status.json")
    train_config = _read_json(train_job_dir / "config" / "train_config.json")

    manifest = _read_json(train_job_dir / "artifacts" / "dataset_manifest.json")
    resolved_backend = str(
        status.get("trainingBackendResolved")
        or train_config.get("trainingBackendResolved")
        or train_config.get("trainingBackend")
        or status.get("trainingBackend")
        or "robomimic_bc"
    )
    framework_label, model_type = _backend_labels(resolved_backend)

    training_complete = is_training_job_complete(status, train_job_dir=train_job_dir)
    discovered = discover_checkpoints(train_job_dir)
    assets = registry.get("assets")
    registry_empty = not isinstance(assets, list) or not assets

    if discovered or training_complete or registry_empty:
        return register_checkpoint_assets(
            train_job_dir=train_job_dir,
            train_job_id=train_job_id,
            manifest=manifest,
            train_config=train_config,
            status=status,
            resolved_backend=resolved_backend,
            framework_label=framework_label,
            model_type=model_type,
            checkpoints=discovered or None,
            register_final=training_complete,
        )

    return _sync_registry(train_job_dir, train_job_id, status, train_config)


def _needs_training_context_for_filter(
    *,
    search: Optional[str] = None,
    dataset_label: Optional[str] = None,
    source_task: Optional[str] = None,
) -> bool:
    return bool((search or "").strip() or (dataset_label or "").strip() or (source_task or "").strip())


def _load_model_asset_rows_for_list(
    *,
    for_evaluation: bool,
    evaluation_task_type: Optional[str],
    timing: ModelAssetListTiming,
) -> list[dict[str, Any]]:
    from app.services.workspace_model_asset_list_cache import get_or_load_model_asset_list_rows

    def loader() -> list[dict[str, Any]]:
        return _load_model_asset_rows_uncached(
            for_evaluation=for_evaluation,
            evaluation_task_type=evaluation_task_type,
            timing=timing,
        )

    rows, cache_hit = get_or_load_model_asset_list_rows(loader=loader)
    timing.cache_hit = cache_hit
    if cache_hit:
        timing.db_query_ms = 0.0
        timing.fs_scan_ms = 0.0
    return rows


def _load_model_asset_rows_uncached(
    *,
    for_evaluation: bool,
    evaluation_task_type: Optional[str],
    timing: ModelAssetListTiming,
) -> list[dict[str, Any]]:
    from app.services.model_asset_db_service import list_model_assets_from_db
    from app.services.model_asset_validation import enrich_model_asset, filter_evaluable_model_assets

    db_started = time.perf_counter()
    assets = list_model_assets_from_db(
        for_evaluation=for_evaluation,
        evaluation_task_type=evaluation_task_type,
        for_list=True,
    )
    timing.db_query_ms = (time.perf_counter() - db_started) * 1000
    if assets:
        timing.fs_scan_ms = 0.0
        return assets

    if not _has_training_jobs_root():
        timing.fs_scan_ms = 0.0
        return []

    fs_started = time.perf_counter()
    from app.services.training_job_sync_service import reindex_runtime_jobs

    reindex_runtime_jobs(job_type="training", dry_run=False)
    assets = list_model_assets_from_db(
        for_evaluation=for_evaluation,
        evaluation_task_type=evaluation_task_type,
        for_list=True,
    )
    if assets:
        timing.fs_scan_ms = (time.perf_counter() - fs_started) * 1000
        return assets

    fs_assets = _list_model_assets_from_filesystem()
    timing.fs_scan_ms = (time.perf_counter() - fs_started) * 1000
    if for_evaluation:
        return filter_evaluable_model_assets(fs_assets, evaluation_task_type=evaluation_task_type)
    return [enrich_model_asset(item, for_list=True) for item in fs_assets]


def _batch_training_context_for_rows(
    rows: list[dict[str, Any]],
    *,
    timing: Optional[ModelAssetListTiming] = None,
) -> dict[str, dict[str, Any]]:
    from app.services.training_job_sync_service import get_training_job_filter_context_batch

    job_ids = [str(asset.get("sourceTrainingJobId") or "") for asset in rows]
    started = time.perf_counter()
    cache = get_training_job_filter_context_batch(job_ids)
    if timing is not None:
        timing.training_assoc_ms = (time.perf_counter() - started) * 1000
    return cache


def filter_model_asset_rows(
    rows: list[dict[str, Any]],
    *,
    search: Optional[str] = None,
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    training_job_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    source: Optional[str] = None,
    dataset_label: Optional[str] = None,
    source_task: Optional[str] = None,
    timing: Optional[ModelAssetListTiming] = None,
) -> list[dict[str, Any]]:
    q = (search or "").strip().lower()
    status_filter = (status or "").strip().lower()
    model_type_filter = (model_type or "").strip()
    training_job_filter = (training_job_id or "").strip()
    dataset_id_filter = (dataset_id or "").strip()
    source_filter = (source or "").strip().lower()
    dataset_label_filter = (dataset_label or "").strip()
    source_task_filter = (source_task or "").strip()

    needs_training = _needs_training_context_for_filter(
        search=search,
        dataset_label=dataset_label,
        source_task=source_task,
    )
    training_cache: dict[str, dict[str, Any]] = {}
    if needs_training:
        training_cache = _batch_training_context_for_rows(rows, timing=timing)

    filter_started = time.perf_counter()
    filtered: list[dict[str, Any]] = []
    dataset_assoc_ms = 0.0
    for asset in rows:
        if asset.get("isPlaceholder"):
            continue
        job_id = str(asset.get("sourceTrainingJobId") or "")
        training_ctx = training_cache.get(job_id, {}) if needs_training else {}

        if status_filter:
            asset_status = str(asset.get("status") or "").lower()
            if asset_status != status_filter:
                continue

        if training_job_filter and job_id != training_job_filter:
            continue

        if dataset_id_filter:
            source_dataset = str(asset.get("sourceDatasetId") or "")
            if source_dataset != dataset_id_filter:
                continue

        if source_filter:
            asset_source = _resolve_asset_source_kind(asset)
            if asset_source != source_filter:
                continue

        recipe = _resolve_asset_recipe_label(asset)
        if model_type_filter and recipe != model_type_filter:
            continue

        dataset = ""
        task_label = job_id
        if needs_training:
            ds_started = time.perf_counter()
            dataset = _resolve_asset_dataset_label_for_filter(asset, training_ctx)
            dataset_assoc_ms += (time.perf_counter() - ds_started) * 1000
            task_label = _resolve_asset_training_task_label_for_filter(asset, training_ctx)
            if dataset_label_filter and dataset != dataset_label_filter:
                continue
            if source_task_filter and task_label != source_task_filter:
                continue
        elif dataset_label_filter or source_task_filter:
            continue

        if q:
            haystack = " ".join(
                filter(
                    None,
                    [
                        asset.get("name"),
                        asset.get("displayName"),
                        asset.get("id"),
                        job_id,
                        asset.get("sourceDatasetId"),
                        recipe,
                        dataset or asset.get("datasetDisplayName"),
                        task_label,
                        training_ctx.get("taskName"),
                        training_ctx.get("datasetName"),
                    ],
                )
            ).lower()
            if q not in haystack:
                continue

        filtered.append(asset)

    if timing is not None:
        timing.dataset_assoc_ms = dataset_assoc_ms
        timing.count_ms = (time.perf_counter() - filter_started) * 1000
    return filtered


def list_model_assets_for_list_api(
    *,
    for_evaluation: bool = False,
    evaluation_task_type: Optional[str] = None,
    search: Optional[str] = None,
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    training_job_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    source: Optional[str] = None,
    dataset_label: Optional[str] = None,
    source_task: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int, ModelAssetListTiming]:
    from app.core.api_timing import paginate_rows

    total_started = time.perf_counter()
    timing = ModelAssetListTiming()
    rows = _load_model_asset_rows_for_list(
        for_evaluation=for_evaluation,
        evaluation_task_type=evaluation_task_type,
        timing=timing,
    )
    filtered = filter_model_asset_rows(
        rows,
        search=search,
        status=status,
        model_type=model_type,
        training_job_id=training_job_id,
        dataset_id=dataset_id,
        source=source,
        dataset_label=dataset_label,
        source_task=source_task,
        timing=timing,
    )
    total = len(filtered)
    page_rows = paginate_rows(filtered, limit=limit, offset=offset)
    timing.total_ms = (time.perf_counter() - total_started) * 1000
    return page_rows, total, timing


def list_model_asset_filter_options() -> dict[str, list[str]]:
    from app.core.database import SessionLocal
    from app.models.workspace_index import ModelAsset
    from app.models.workspace_job import WorkspaceJob
    from app.services.model_asset_naming import format_model_recipe_label

    model_types: set[str] = set(MODEL_TYPE_FILTER_PRESETS)
    datasets: set[str] = set()
    source_tasks: set[str] = set()
    job_ids: set[str] = set()

    with SessionLocal() as db:
        rows = (
            db.query(ModelAsset)
            .filter(ModelAsset.status.notin_(["deleted", "superseded"]))
            .all()
        )
        for row in rows:
            manifest = row.manifest_json if isinstance(row.manifest_json, dict) else {}
            if manifest.get("isPlaceholder"):
                continue
            framework = str(manifest.get("framework") or "unknown")
            model_type = str(row.model_type or manifest.get("modelType") or "unknown")
            training_backend = manifest.get("trainingBackend")
            model_types.add(
                format_model_recipe_label(
                    framework=framework,
                    model_type=model_type,
                    training_backend=str(training_backend or "") or None,
                )
            )
            dataset_label = str(manifest.get("datasetDisplayName") or row.dataset_id or "").strip()
            if dataset_label and dataset_label not in {"未知数据集", "未知任务数据"}:
                datasets.add(dataset_label)
            if row.train_job_id:
                job_ids.add(row.train_job_id)

        if job_ids:
            jobs = (
                db.query(WorkspaceJob.job_id, WorkspaceJob.task_name)
                .filter(
                    WorkspaceJob.job_id.in_(list(job_ids)),
                    WorkspaceJob.job_type == "training",
                    WorkspaceJob.status != "deleted",
                )
                .all()
            )
            for job in jobs:
                task_name = str(job.task_name or "").strip()
                if task_name:
                    source_tasks.add(task_name)

    return {
        "modelTypes": sorted(model_types, key=lambda value: value.casefold()),
        "datasets": sorted(datasets, key=lambda value: value.casefold()),
        "sourceTasks": sorted(source_tasks, key=lambda value: value.casefold()),
    }


def _resolve_asset_recipe_label(asset: dict[str, Any]) -> str:
    from app.services.model_asset_naming import format_model_recipe_label

    return format_model_recipe_label(
        framework=str(asset.get("framework") or "") or None,
        model_type=str(asset.get("modelType") or "") or None,
        training_backend=str(asset.get("trainingBackend") or asset.get("backendType") or "") or None,
    )


def _resolve_asset_dataset_label_for_filter(
    asset: dict[str, Any],
    training_ctx: dict[str, Any],
) -> str:
    from app.services.dataset_naming import is_canonical_dataset_display_name, normalize_dataset_display_name
    from app.services.model_asset_naming import is_internal_context_label

    candidates = [
        training_ctx.get("datasetName"),
        asset.get("datasetDisplayName"),
        asset.get("sourceDatasetId"),
    ]
    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        normalized = normalize_dataset_display_name(
            display_name=raw,
            name=raw,
            task_type=training_ctx.get("taskType") or asset.get("taskTemplateId"),
            source_job_id=asset.get("sourceDatasetId"),
        )
        if normalized and normalized not in {"未知数据集", "未知任务数据"}:
            return normalized
        if is_canonical_dataset_display_name(raw):
            return raw
        if not is_internal_context_label(raw):
            return raw
    return "未知数据集"


def _resolve_asset_training_task_label_for_filter(
    asset: dict[str, Any],
    training_ctx: dict[str, Any],
) -> str:
    from app.services.model_asset_naming import is_internal_context_label

    task_name = str(training_ctx.get("taskName") or "").strip()
    if task_name and not is_internal_context_label(task_name):
        return task_name
    return str(asset.get("sourceTrainingJobId") or "")


def _resolve_asset_source_kind(asset: dict[str, Any]) -> str:
    if asset.get("assetSource") == "imported" or asset.get("checkpointKind") == "imported":
        return "imported"
    if str(asset.get("sourceTrainingJobId") or "") == "model_asset_import_hub":
        return "imported"
    return "training"


def list_model_assets(
    *,
    for_evaluation: bool = False,
    evaluation_task_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    from app.services.model_asset_db_service import list_model_assets_from_db
    from app.services.model_asset_validation import enrich_model_asset, filter_evaluable_model_assets
    from app.services.training_job_sync_service import reindex_runtime_jobs

    assets = list_model_assets_from_db(
        for_evaluation=for_evaluation,
        evaluation_task_type=evaluation_task_type,
    )
    if assets:
        return assets

    if not _has_training_jobs_root():
        return []

    reindex_runtime_jobs(job_type="training", dry_run=False)
    assets = list_model_assets_from_db(
        for_evaluation=for_evaluation,
        evaluation_task_type=evaluation_task_type,
    )
    if assets:
        return assets

    fs_assets = _list_model_assets_from_filesystem()
    if for_evaluation:
        return filter_evaluable_model_assets(fs_assets, evaluation_task_type=evaluation_task_type)
    return [enrich_model_asset(item) for item in fs_assets]


def _list_model_assets_from_filesystem() -> list[dict[str, Any]]:
    from app.services.model_asset_validation import enrich_model_asset

    assets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    job_dirs: dict[str, Path] = {}
    for root in reversed(_training_job_roots()):
        if not root.is_dir():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir():
                job_dirs[candidate.name] = candidate

    for train_job_dir in sorted(job_dirs.values(), key=lambda p: p.name, reverse=True):
        train_job_id = train_job_dir.name
        status = _read_json(train_job_dir / "status.json") if (train_job_dir / "status.json").is_file() else {}

        entries = _ensure_registry_for_job(train_job_dir, train_job_id)
        if not entries:
            continue

        for entry in entries:
            asset = enrich_model_asset(_entry_to_asset(entry, train_job_id=train_job_id, status=status))
            asset_id = asset["id"]
            if asset_id in seen_ids:
                continue
            seen_ids.add(asset_id)
            assets.append(asset)

    return assets


def get_model_asset_by_id(model_asset_id: str) -> Optional[dict[str, Any]]:
    from app.services.model_asset_db_service import get_model_asset_from_db

    candidate = (model_asset_id or "").strip()
    if not candidate:
        return None
    asset = get_model_asset_from_db(candidate)
    if asset:
        from app.services.model_asset_validation import enrich_model_asset

        return enrich_model_asset(asset)
    for item in _list_model_assets_from_filesystem():
        if item.get("id") == candidate:
            from app.services.model_asset_validation import enrich_model_asset

            return enrich_model_asset(item)
    return None


def _entry_to_detail_asset(
    entry: dict[str, Any],
    *,
    train_job_id: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    base = _entry_to_asset(entry, train_job_id=train_job_id, status=status)
    is_placeholder = bool(entry.get("isPlaceholder"))
    can_evaluate = bool(entry.get("canEvaluate")) if "canEvaluate" in entry else False
    display_status = str(entry.get("displayStatus") or "waiting")
    if not is_placeholder and "canEvaluate" not in entry:
        from app.services.checkpoint_registry import (
            compute_asset_can_evaluate,
            compute_asset_display_status,
        )

        can_evaluate = compute_asset_can_evaluate(entry, job_status=status)
        display_status = compute_asset_display_status(entry, job_status=status)
    return {
        **base,
        "isPlaceholder": is_placeholder,
        "canEvaluate": can_evaluate,
        "displayStatus": display_status,
    }


def _resolve_job_backend_context(
    train_job_dir: Path,
    train_job_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str, str]:
    status = _read_json(train_job_dir / "status.json")
    from app.services.training_job_status import enrich_and_persist_training_job_status

    status = enrich_and_persist_training_job_status(train_job_id, train_job_dir, status)
    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    manifest = _read_json(train_job_dir / "artifacts" / "dataset_manifest.json")
    resolved_backend = str(
        status.get("trainingBackendResolved")
        or train_config.get("trainingBackendResolved")
        or train_config.get("trainingBackend")
        or status.get("trainingBackend")
        or "robomimic_bc"
    )
    framework_label, model_type = _backend_labels(resolved_backend)
    return status, train_config, manifest, resolved_backend, framework_label, model_type


def list_model_assets_for_training_job(
    train_job_id: str,
    *,
    for_evaluation: bool = False,
) -> list[dict[str, Any]]:
    """主列表/初始化权重：默认仅真实可用资产；detail 接口可看全部。"""
    from app.services.model_asset_db_service import list_model_assets_for_job_from_db
    from app.services.model_asset_validation import filter_evaluable_model_assets
    from app.services.training_job_sync_service import sync_training_job_from_runtime

    job_id = (train_job_id or "").strip()
    if not job_id:
        return []
    train_job_dir = _find_training_job_dir(job_id)
    if train_job_dir.is_dir():
        sync_training_job_from_runtime(job_id)

    assets = list_model_assets_for_job_from_db(job_id)
    if not assets and train_job_dir.is_dir():
        status, _, _, _, _, _ = _resolve_job_backend_context(train_job_dir, job_id)
        entries = _ensure_registry_for_job(train_job_dir, job_id)
        from app.services.model_asset_validation import enrich_model_asset

        assets = [
            enrich_model_asset(_entry_to_asset(entry, train_job_id=job_id, status=status))
            for entry in entries
        ]

    if for_evaluation:
        return filter_evaluable_model_assets(assets)
    return assets


def list_training_job_model_assets_detail(
    train_job_id: str,
    *,
    sync_db: bool = False,
) -> dict[str, Any]:
    """训练任务详情：含 Final 占位与状态字段。"""
    from app.services.model_asset_db_service import list_training_job_model_assets_detail_from_db

    job_id = (train_job_id or "").strip()
    if not job_id:
        return {"modelAssets": [], "total": 0, "listMessage": None, "warning": None}
    train_job_dir = _find_training_job_dir(job_id)
    db_warning: Optional[str] = None

    if sync_db and train_job_dir.is_dir():
        try:
            from app.services.training_job_sync_service import sync_training_job_from_runtime

            sync_result = sync_training_job_from_runtime(job_id)
            if not sync_result.get("ok"):
                db_warning = "; ".join(sync_result.get("warnings") or []) or "DB sync partial failure"
        except Exception as exc:
            db_warning = f"DB sync failed: {exc}"

    status: dict[str, Any] = {}
    if train_job_dir.is_dir():
        status, train_config, manifest, resolved_backend, framework_label, model_type = (
            _resolve_job_backend_context(train_job_dir, job_id)
        )
    else:
        from app.services.training_job_sync_service import get_training_job_summary_from_db

        summary = get_training_job_summary_from_db(job_id)
        if summary:
            status = summary
        train_config = {}
        manifest = {}
        resolved_backend = str(summary.get("trainingBackend") or "robomimic_bc") if summary else "robomimic_bc"
        framework_label, model_type = _backend_labels(resolved_backend)

    db_assets: list[dict[str, Any]] = []
    try:
        db_assets = list_training_job_model_assets_detail_from_db(job_id, status=status)
    except Exception as exc:
        db_warning = db_warning or f"DB_UNAVAILABLE: {exc}"
        db_assets = []
    if db_assets:
        from app.services.training_job_generated_assets import filter_training_job_detail_model_assets

        db_assets = filter_training_job_detail_model_assets(
            db_assets,
            train_job_id=job_id,
            train_job_dir=train_job_dir,
            status=status,
            train_config=train_config if train_job_dir.is_dir() else {},
        )
        list_message = resolve_training_job_model_assets_list_message(
            rows=db_assets,
            train_config=train_config if train_job_dir.is_dir() else {},
            status=status,
            train_job_dir=train_job_dir if train_job_dir.is_dir() else None,
        )
        if db_warning and not list_message:
            list_message = db_warning
        return {
            "modelAssets": db_assets,
            "total": len(db_assets),
            "listMessage": list_message,
            "warning": db_warning,
        }

    if not train_job_dir.is_dir():
        return {"modelAssets": [], "total": 0, "listMessage": db_warning, "warning": db_warning}

    status, train_config, manifest, resolved_backend, framework_label, model_type = (
        _resolve_job_backend_context(train_job_dir, job_id)
    )
    entries = list_training_job_detail_registry_entries(
        train_job_dir=train_job_dir,
        train_job_id=job_id,
        manifest=manifest,
        train_config=train_config,
        status=status,
        resolved_backend=resolved_backend,
        framework_label=framework_label,
        model_type=model_type,
    )
    assets = [_entry_to_detail_asset(entry, train_job_id=job_id, status=status) for entry in entries]
    from app.services.training_job_generated_assets import filter_training_job_detail_model_assets

    assets = filter_training_job_detail_model_assets(
        assets,
        train_job_id=job_id,
        train_job_dir=train_job_dir,
        status=status,
        train_config=train_config,
    )
    list_message = resolve_training_job_model_assets_list_message(
        rows=assets,
        train_config=train_config,
        status=status,
        train_job_dir=train_job_dir,
    )
    if db_warning and not list_message:
        list_message = db_warning
    return {
        "modelAssets": assets,
        "total": len(assets),
        "listMessage": list_message,
        "warning": db_warning,
    }


def delete_model_asset(model_asset_id: str) -> dict[str, Any]:
    from app.services.model_asset_db_service import get_model_asset_from_db, mark_model_asset_deleted
    from app.services.training_service import _resolve_safe_path
    from app.services.workspace_model_asset_list_cache import invalidate_model_asset_list_cache

    invalidate_model_asset_list_cache()

    candidate = (model_asset_id or "").strip()
    if not candidate:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="modelAssetId is required")

    asset = get_model_asset_from_db(candidate) or get_model_asset_by_id(candidate)
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="model asset not found")

    mark_model_asset_deleted(candidate)

    train_job_id = str(asset.get("sourceTrainingJobId") or "")
    train_job_dir = _find_training_job_dir(train_job_id)

    registry = read_registry(train_job_dir) if train_job_dir.is_dir() else {"assets": []}
    assets = registry.get("assets") if isinstance(registry.get("assets"), list) else []
    remaining = [item for item in assets if isinstance(item, dict) and item.get("modelAssetId") != candidate]

    warnings: list[str] = []
    checkpoint_path = str(asset.get("checkpointPath") or "")
    asset_source = str(asset.get("assetSource") or "")
    if checkpoint_path:
        try:
            safe_path = _resolve_safe_path(checkpoint_path)
            if safe_path.is_file():
                safe_path.unlink()
            elif safe_path.exists():
                warnings.append(f"checkpoint path is not a file: {checkpoint_path}")
            if asset_source == "imported" or str(asset.get("checkpointKind") or "") == "imported":
                asset_dir = safe_path.parent
                imported_roots = (
                    platform_paths.models / "imported",
                    RUNTIME_ROOT / "model_assets" / "imported",
                )
                if asset_dir.is_dir() and any(is_path_within(asset_dir, root) for root in imported_roots):
                    shutil.rmtree(asset_dir, ignore_errors=True)
        except Exception as exc:
            warnings.append(f"checkpoint not deleted: {exc}")

    if train_job_dir.is_dir():
        manifest_path = train_job_dir / "artifacts" / "checkpoint_manifests" / f"{candidate}.json"
        if manifest_path.is_file():
            try:
                manifest_path.unlink()
            except OSError as exc:
                warnings.append(f"manifest not deleted: {exc}")

        registry_payload = {
            "version": 1,
            "sourceTrainJobId": train_job_id,
            "updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assets": remaining,
        }
        reg_path = registry_path(train_job_dir)
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg_path.write_text(json.dumps(registry_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        primary_manifest = train_job_dir / "artifacts" / "model_manifest.json"
        if primary_manifest.is_file():
            try:
                primary_data = _read_json(primary_manifest)
                if primary_data.get("modelAssetId") == candidate:
                    if remaining:
                        _write_json(primary_manifest, remaining[-1])
                    else:
                        primary_manifest.unlink(missing_ok=True)
            except OSError:
                pass

        status_path = train_job_dir / "status.json"
        if status_path.is_file():
            status_data = _read_json(status_path)
            if status_data.get("modelAssetId") == candidate:
                next_primary = remaining[-1] if remaining else None
                status_data["modelAssetId"] = next_primary.get("modelAssetId") if next_primary else None
                status_data["checkpointPath"] = next_primary.get("checkpointPath") if next_primary else None
                status_data["checkpointExists"] = bool(remaining)
                status_path.write_text(
                    json.dumps(status_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    elif checkpoint_path:
        warnings.append("training job directory missing; database record removed only")

    return {
        "modelAssetId": candidate,
        "deleted": True,
        "warnings": warnings,
    }
