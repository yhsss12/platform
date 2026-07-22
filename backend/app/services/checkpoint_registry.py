"""训练 checkpoint 发现、注册与模型资产列表。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.services.model_asset_naming import build_checkpoint_asset_display_name

CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
REGISTRY_FILENAME = "model_assets_registry.json"

CHECKPOINT_FINAL_RE = re.compile(r"^model_final\.(pth|pt|ckpt)$", re.IGNORECASE)
CHECKPOINT_FINAL_ALT_RE = re.compile(r"^(checkpoint_final|latest|last)\.(pth|pt|ckpt)$", re.IGNORECASE)
CHECKPOINT_MODEL_PT_RE = re.compile(r"^model\.pt$", re.IGNORECASE)
CHECKPOINT_EPOCH_BEST_RE = re.compile(
    r"^model_epoch_(\d+)_best_validation_([\d.+-]+)\.(pth|pt)$",
    re.IGNORECASE,
)
CHECKPOINT_EPOCH_RE = re.compile(r"^model_epoch_(\d+)\.(pth|pt)$", re.IGNORECASE)

SAVE_CAPABILITIES: dict[str, dict[str, bool]] = {
    "robomimic_bc": {"final": True, "best": True, "interval": True},
    "isaac_robomimic_bc": {"final": True, "best": False, "interval": True},
    "diffusion_policy": {"final": True, "best": True, "interval": False},
    "act": {"final": True, "best": True, "interval": False},
    "pi0": {"final": True, "best": True, "interval": False},
    "torch_bc": {"final": True, "best": False, "interval": False},
}

KIND_DISPLAY_ORDER = {"final": 0, "best": 1, "epoch": 2}
KIND_PRIORITY = {"best": 3, "final": 2, "epoch": 1}

COMPLETED_JOB_STATUSES = frozenset({"completed", "success", "succeeded", "finished", "done"})
IN_PROGRESS_JOB_STATUSES = frozenset({"running", "training", "queued", "pending"})


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    kind: str  # final | best | epoch
    epoch: Optional[int] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None


def _utc_now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_training_job_complete(
    status: dict[str, Any],
    *,
    train_job_dir: Path | None = None,
) -> bool:
    from app.services.training_job_status import infer_training_job_completed, normalize_training_status_token

    raw = normalize_training_status_token(str(status.get("status") or ""))
    if raw in COMPLETED_JOB_STATUSES:
        return True
    if raw in {"failed", "error", "canceled", "cancelled"}:
        return False
    return infer_training_job_completed(status, train_job_dir=train_job_dir)


def is_training_job_in_progress(status: dict[str, Any]) -> bool:
    return str(status.get("status") or "").lower() in IN_PROGRESS_JOB_STATUSES


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_save_policy(train_config: dict[str, Any]) -> dict[str, Any]:
    interval = train_config.get("checkpointIntervalEpochs")
    parsed_interval: Optional[int] = None
    if interval is not None:
        try:
            value = int(interval)
            if value > 0:
                parsed_interval = value
        except (TypeError, ValueError):
            parsed_interval = None
    return {
        "saveFinal": bool(train_config.get("saveFinal", True)),
        "saveBest": bool(train_config.get("saveBest", False)),
        "checkpointIntervalEpochs": parsed_interval,
    }


def save_capabilities_for_backend(backend: str) -> dict[str, bool]:
    return SAVE_CAPABILITIES.get(backend, {"final": True, "best": False, "interval": False})


def _checkpoint_file_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def classify_checkpoint(path: Path) -> Optional[CheckpointRecord]:
    if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        return None
    if path.stat().st_size <= 0:
        return None
    name = path.name
    if (
        CHECKPOINT_FINAL_RE.match(name)
        or CHECKPOINT_FINAL_ALT_RE.match(name)
        or CHECKPOINT_MODEL_PT_RE.match(name)
    ):
        return CheckpointRecord(path=path, kind="final")
    best_match = CHECKPOINT_EPOCH_BEST_RE.match(name)
    if best_match:
        metric_raw = best_match.group(2)
        try:
            metric_value = float(metric_raw)
        except ValueError:
            metric_value = None
        return CheckpointRecord(
            path=path,
            kind="best",
            epoch=int(best_match.group(1)),
            metric_name="Loss",
            metric_value=metric_value,
        )
    epoch_match = CHECKPOINT_EPOCH_RE.match(name)
    if epoch_match:
        return CheckpointRecord(path=path, kind="epoch", epoch=int(epoch_match.group(1)))
    return None


def discover_checkpoints(train_job_dir: Path) -> list[CheckpointRecord]:
    search_roots = [
        train_job_dir / "checkpoints",
        train_job_dir / "artifacts",
    ]
    found: dict[str, CheckpointRecord] = {}
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            record = classify_checkpoint(path)
            if record is None:
                continue
            key = str(record.path.resolve())
            found[key] = record
    return list(found.values())


def _pick_best_final_checkpoint(records: list[CheckpointRecord]) -> CheckpointRecord:
    def score(record: CheckpointRecord) -> tuple[int, int]:
        name = record.path.name.lower()
        explicit = 1 if name.startswith("model_final") or name == "model.pt" else 0
        try:
            size = record.path.stat().st_size if record.path.is_file() else 0
        except OSError:
            size = 0
        return explicit, size

    return max(records, key=score)


def normalize_discovered_checkpoints(
    checkpoints: list[CheckpointRecord],
    total_epochs: int,
    *,
    training_complete: bool,
) -> list[CheckpointRecord]:
    """末轮 epoch 仅在训练完成时归为 final；合并重复 final。"""
    if not checkpoints:
        return []

    others: list[CheckpointRecord] = []
    final_candidates: list[CheckpointRecord] = []

    for record in checkpoints:
        if (
            training_complete
            and total_epochs > 0
            and record.kind == "epoch"
            and record.epoch is not None
            and record.epoch == total_epochs
        ):
            final_candidates.append(
                CheckpointRecord(path=record.path, kind="final", epoch=None, metric_name=record.metric_name)
            )
        elif record.kind == "final":
            if training_complete:
                final_candidates.append(record)
        else:
            others.append(record)

    if not final_candidates:
        return others

    unique_by_path: dict[str, CheckpointRecord] = {}
    for record in final_candidates:
        unique_by_path[str(record.path.resolve())] = record
    unique_finals = list(unique_by_path.values())
    if len(unique_finals) > 1:
        unique_finals = [_pick_best_final_checkpoint(unique_finals)]

    return others + unique_finals


def _dedupe_discovered_by_path(checkpoints: list[CheckpointRecord]) -> list[CheckpointRecord]:
    """同一路径若同时满足 best/epoch，优先保留 best。"""
    by_path: dict[str, CheckpointRecord] = {}
    for record in checkpoints:
        key = str(record.path.resolve())
        existing = by_path.get(key)
        if existing is None:
            by_path[key] = record
            continue
        existing_pri = KIND_PRIORITY.get(existing.kind, 0)
        record_pri = KIND_PRIORITY.get(record.kind, 0)
        if record_pri > existing_pri:
            by_path[key] = record
    return list(by_path.values())


def _path_key(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def _stable_path_asset_id(train_job_id: str, checkpoint_path: Path) -> str:
    digest = hashlib.sha1(str(checkpoint_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"model_{train_job_id[-12:]}_{digest}"


def _stable_best_asset_id(train_job_id: str, metric_name: str | None) -> str:
    metric = (metric_name or "Loss").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", metric).strip("_") or "loss"
    return f"model_{train_job_id[-12:]}_best_{slug}"


def _stable_final_asset_id(train_job_id: str) -> str:
    return f"model_{train_job_id[-12:]}_final"


def stable_asset_id_for_checkpoint(train_job_id: str, checkpoint: CheckpointRecord) -> str:
    if checkpoint.kind == "best":
        return _stable_best_asset_id(train_job_id, checkpoint.metric_name)
    if checkpoint.kind == "final":
        return _stable_final_asset_id(train_job_id)
    return _stable_path_asset_id(train_job_id, checkpoint.path)


def registry_path(train_job_dir: Path) -> Path:
    return train_job_dir / "artifacts" / REGISTRY_FILENAME


def read_registry(train_job_dir: Path) -> dict[str, Any]:
    return _read_json(registry_path(train_job_dir))


def _context_label(train_config: dict[str, Any], status: dict[str, Any], manifest: dict[str, Any]) -> str:
    from app.services.model_asset_naming import _normalize_context_label

    return _normalize_context_label(
        training_task_name=str(train_config.get("taskName") or status.get("taskName") or "") or None,
        dataset_name=str(status.get("datasetName") or manifest.get("datasetName") or "") or None,
        dataset_id=str(manifest.get("datasetId") or status.get("datasetId") or "") or None,
        task_template_id=str(manifest.get("taskTemplateId") or "") or None,
        task_type=str(manifest.get("taskType") or status.get("taskType") or "") or None,
    )


def _resolve_asset_status(checkpoint: CheckpointRecord, *, allow_final: bool) -> str:
    if checkpoint.kind == "final" and not allow_final:
        return "pending"
    if not _checkpoint_file_ready(checkpoint.path):
        return "pending"
    return "ready"


def _manifest_for_checkpoint(
    *,
    model_asset_id: str,
    train_job_id: str,
    checkpoint: CheckpointRecord,
    display_name: str,
    manifest: dict[str, Any],
    status: dict[str, Any],
    train_config: dict[str, Any],
    resolved_backend: str,
    framework_label: str,
    model_type: str,
    allow_final: bool,
    created_at: str | None = None,
) -> dict[str, Any]:
    now = created_at or _utc_now_label()
    asset_status = _resolve_asset_status(checkpoint, allow_final=allow_final)
    from app.services.model_type_traceability import build_model_type_traceability_fields

    traceability = build_model_type_traceability_fields(train_config)
    result = {
        "modelAssetId": model_asset_id,
        "name": display_name,
        "displayName": display_name,
        "checkpointKind": checkpoint.kind,
        "checkpointEpoch": checkpoint.epoch,
        "checkpointMetricName": checkpoint.metric_name,
        "checkpointMetricValue": checkpoint.metric_value,
        "sourceTrainJobId": train_job_id,
        "sourceDatasetId": manifest.get("datasetId") or status.get("datasetId"),
        "taskType": manifest.get("taskType") or status.get("taskType"),
        "taskTemplateId": manifest.get("taskTemplateId"),
        "trainingBackend": resolved_backend,
        "backendType": resolved_backend,
        "framework": framework_label,
        "modelType": model_type,
        "checkpointPath": str(checkpoint.path),
        "trainingTaskName": train_config.get("taskName") or status.get("taskName"),
        "datasetDisplayName": status.get("datasetName"),
        "status": asset_status,
        "createdAt": now,
        "updatedAt": now,
    }
    result.update(traceability)
    return result


def _find_asset_index(assets: list[dict[str, Any]], asset_id: str) -> int | None:
    for index, item in enumerate(assets):
        if isinstance(item, dict) and str(item.get("modelAssetId") or "") == asset_id:
            return index
    return None


def _mark_superseded(assets: list[dict[str, Any]], asset_id: str) -> None:
    index = _find_asset_index(assets, asset_id)
    if index is None:
        return
    item = dict(assets[index])
    if str(item.get("status") or "").lower() == "superseded":
        return
    item["status"] = "superseded"
    item["updatedAt"] = _utc_now_label()
    assets[index] = item


def _mark_other_best_superseded(
    assets: list[dict[str, Any]],
    *,
    train_job_id: str,
    metric_name: str | None,
    keep_asset_id: str,
) -> None:
    metric = (metric_name or "Loss").strip().lower()
    for index, item in enumerate(assets):
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("modelAssetId") or "")
        if asset_id == keep_asset_id:
            continue
        if str(item.get("checkpointKind") or "").lower() != "best":
            continue
        item_metric = str(item.get("checkpointMetricName") or "Loss").strip().lower()
        if item_metric != metric:
            continue
        if str(item.get("sourceTrainJobId") or train_job_id) != train_job_id:
            continue
        superseded = dict(item)
        superseded["status"] = "superseded"
        superseded["updatedAt"] = _utc_now_label()
        assets[index] = superseded


def _pick_current_best_asset(group: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        item
        for item in group
        if str(item.get("status") or "").lower() not in {"superseded", "pending"}
    ] or list(group)

    def sort_key(item: dict[str, Any]) -> tuple[float, int, int]:
        metric_raw = item.get("checkpointMetricValue")
        try:
            metric_value = float(metric_raw) if metric_raw is not None else float("inf")
        except (TypeError, ValueError):
            metric_value = float("inf")
        epoch = int(item.get("checkpointEpoch") or 0)
        stable = 1 if "_best_" in str(item.get("modelAssetId") or "") else 0
        return metric_value, -epoch, stable

    return min(candidates, key=sort_key)


def _pick_best_final_asset_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    def score(entry: dict[str, Any]) -> tuple[int, int]:
        path = str(entry.get("checkpointPath") or "").lower()
        explicit = 1 if "model_final" in path or path.endswith("/model.pt") else 0
        return explicit, len(path)

    return max(entries, key=score)


def dedupe_registry_assets(
    assets: list[dict[str, Any]],
    total_epochs: int,
    *,
    training_complete: bool,
) -> list[dict[str, Any]]:
    """注册表去重：末轮 epoch 仅在完成时视为 final；多个 final 仅保留一个。"""
    if not assets:
        return []

    normalized: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for entry in assets:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("checkpointPath") or "").strip()
        path_key = _path_key(path) if path else ""
        if path_key:
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

        kind = str(entry.get("checkpointKind") or "").lower()
        epoch_raw = entry.get("checkpointEpoch")
        epoch = int(epoch_raw) if epoch_raw is not None else None

        item = dict(entry)
        if training_complete and total_epochs > 0 and kind == "epoch" and epoch == total_epochs:
            item["checkpointKind"] = "final"
            item["checkpointEpoch"] = None
            final_candidates.append(item)
        elif kind == "final":
            if training_complete:
                final_candidates.append(item)
            else:
                item["status"] = "pending"
                normalized.append(item)
        else:
            normalized.append(item)

    if final_candidates:
        if len(final_candidates) > 1:
            final_candidates = [_pick_best_final_asset_entry(final_candidates)]
        normalized.extend(final_candidates)

    return normalized


def save_policy_has_outputs(train_config: dict[str, Any]) -> bool:
    policy = parse_save_policy(train_config)
    return bool(
        policy.get("saveFinal", True)
        or policy.get("saveBest", False)
        or policy.get("checkpointIntervalEpochs")
    )


def _promote_asset_readiness(
    rows: list[dict[str, Any]],
    *,
    training_complete: bool,
) -> None:
    for row in rows:
        path = str(row.get("checkpointPath") or "").strip()
        if not path or not _checkpoint_file_ready(Path(path)):
            continue
        kind = str(row.get("checkpointKind") or "").lower()
        asset_status = str(row.get("status") or "").lower()
        if asset_status == "superseded":
            continue
        if training_complete and kind == "final":
            row["status"] = "ready"
        elif kind != "final" and asset_status in {"pending", "generating", "unknown", ""}:
            row["status"] = "ready"


def normalize_registry_assets(
    assets: list[dict[str, Any]],
    *,
    status: dict[str, Any],
    total_epochs: int,
    train_job_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """清理历史脏数据：best 单条、path 去重、训练中隐藏 final。"""
    training_complete = is_training_job_complete(status, train_job_dir=train_job_dir)
    rows = dedupe_registry_assets(
        [dict(item) for item in assets if isinstance(item, dict)],
        total_epochs,
        training_complete=training_complete,
    )

    best_groups: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        if str(item.get("checkpointKind") or "").lower() != "best":
            continue
        metric = str(item.get("checkpointMetricName") or "Loss").strip().lower()
        best_groups.setdefault(metric, []).append(item)

    superseded_ids: set[str] = set()
    for group in best_groups.values():
        if len(group) <= 1:
            continue
        winner = _pick_current_best_asset(group)
        winner_id = str(winner.get("modelAssetId") or "")
        for item in group:
            asset_id = str(item.get("modelAssetId") or "")
            if asset_id and asset_id != winner_id:
                superseded_ids.add(asset_id)

    path_winners: dict[str, dict[str, Any]] = {}
    for item in rows:
        asset_id = str(item.get("modelAssetId") or "")
        if asset_id in superseded_ids:
            continue
        path = str(item.get("checkpointPath") or "").strip()
        if not path:
            continue
        path_key = _path_key(path)
        kind = str(item.get("checkpointKind") or "").lower()
        priority = KIND_PRIORITY.get(kind, 0)
        existing = path_winners.get(path_key)
        if existing is None:
            path_winners[path_key] = item
            continue
        existing_kind = str(existing.get("checkpointKind") or "").lower()
        if priority > KIND_PRIORITY.get(existing_kind, 0):
            superseded_ids.add(str(existing.get("modelAssetId") or ""))
            path_winners[path_key] = item
        elif priority == KIND_PRIORITY.get(existing_kind, 0):
            existing_id = str(existing.get("modelAssetId") or "")
            if asset_id and existing_id and asset_id != existing_id:
                superseded_ids.add(asset_id)

    normalized: list[dict[str, Any]] = []
    for item in rows:
        asset_id = str(item.get("modelAssetId") or "")
        row = dict(item)
        if asset_id in superseded_ids:
            row["status"] = "superseded"
        if str(row.get("checkpointKind") or "").lower() == "final":
            if not training_complete:
                row["status"] = "pending"
            else:
                path = str(row.get("checkpointPath") or "").strip()
                if path and _checkpoint_file_ready(Path(path)):
                    row["status"] = "ready"
        normalized.append(row)

    _promote_asset_readiness(normalized, training_complete=training_complete)
    return normalized


def list_displayable_registry_assets(
    assets: list[dict[str, Any]],
    *,
    status: dict[str, Any],
    total_epochs: int,
    train_job_dir: Path | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_registry_assets(
        assets,
        status=status,
        total_epochs=total_epochs,
        train_job_dir=train_job_dir,
    )
    training_complete = is_training_job_complete(status, train_job_dir=train_job_dir)
    displayable: list[dict[str, Any]] = []

    for item in normalized:
        row = dict(item)
        asset_status = str(row.get("status") or "").lower()
        kind = str(row.get("checkpointKind") or "").lower()
        path = str(row.get("checkpointPath") or "").strip()
        file_ready = bool(path) and _checkpoint_file_ready(Path(path))

        if asset_status == "superseded":
            continue
        if asset_status == "pending":
            if training_complete and kind == "final" and file_ready:
                row["status"] = "ready"
                asset_status = "ready"
            else:
                continue
        if kind == "final" and not training_complete:
            continue
        if path and not file_ready:
            continue
        if asset_status not in {"ready", "available", "active", "completed"}:
            continue
        displayable.append(row)

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        kind = str(item.get("checkpointKind") or "").lower()
        epoch = int(item.get("checkpointEpoch") or 0)
        return KIND_DISPLAY_ORDER.get(kind, 9), epoch

    return sorted(displayable, key=sort_key)


def final_checkpoint_exists(train_job_dir: Path) -> bool:
    for record in discover_checkpoints(train_job_dir):
        if record.kind == "final":
            return _checkpoint_file_ready(record.path)
    return False


def _final_checkpoint_exists(train_job_dir: Path) -> bool:
    return final_checkpoint_exists(train_job_dir)


def build_final_placeholder_entry(
    *,
    train_job_id: str,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    status: dict[str, Any],
    resolved_backend: str,
    framework_label: str,
    model_type: str,
    train_job_dir: Path,
) -> dict[str, Any]:
    context = _context_label(train_config, status, manifest)
    display_name = build_checkpoint_asset_display_name(context_label=context, kind="final")
    asset_id = _stable_final_asset_id(train_job_id)
    display_status = "waiting"
    return {
        "modelAssetId": asset_id,
        "name": display_name,
        "displayName": display_name,
        "checkpointKind": "final",
        "checkpointEpoch": None,
        "checkpointMetricName": None,
        "checkpointMetricValue": None,
        "sourceTrainJobId": train_job_id,
        "sourceDatasetId": manifest.get("datasetId") or status.get("datasetId"),
        "taskType": manifest.get("taskType") or status.get("taskType"),
        "taskTemplateId": manifest.get("taskTemplateId"),
        "trainingBackend": resolved_backend,
        "backendType": resolved_backend,
        "framework": framework_label,
        "modelType": model_type,
        "checkpointPath": "",
        "trainingTaskName": train_config.get("taskName") or status.get("taskName"),
        "datasetDisplayName": status.get("datasetName"),
        "status": "pending",
        "isPlaceholder": True,
        "canEvaluate": False,
        "displayStatus": display_status,
        "createdAt": _utc_now_label(),
        "updatedAt": _utc_now_label(),
    }


def build_missing_final_entry(
    *,
    train_job_id: str,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    status: dict[str, Any],
    resolved_backend: str,
    framework_label: str,
    model_type: str,
) -> dict[str, Any]:
    context = _context_label(train_config, status, manifest)
    display_name = build_checkpoint_asset_display_name(context_label=context, kind="final")
    return {
        "modelAssetId": _stable_final_asset_id(train_job_id),
        "name": display_name,
        "displayName": display_name,
        "checkpointKind": "final",
        "checkpointEpoch": None,
        "checkpointMetricName": None,
        "checkpointMetricValue": None,
        "sourceTrainJobId": train_job_id,
        "sourceDatasetId": manifest.get("datasetId") or status.get("datasetId"),
        "taskType": manifest.get("taskType") or status.get("taskType"),
        "taskTemplateId": manifest.get("taskTemplateId"),
        "trainingBackend": resolved_backend,
        "backendType": resolved_backend,
        "framework": framework_label,
        "modelType": model_type,
        "checkpointPath": "",
        "trainingTaskName": train_config.get("taskName") or status.get("taskName"),
        "datasetDisplayName": status.get("datasetName"),
        "status": "missing",
        "isPlaceholder": True,
        "canEvaluate": False,
        "displayStatus": "missing",
        "createdAt": _utc_now_label(),
        "updatedAt": _utc_now_label(),
    }


def _entry_from_model_manifest(
    model_manifest: dict[str, Any],
    *,
    train_job_id: str,
) -> Optional[dict[str, Any]]:
    if not model_manifest:
        return None
    checkpoint_path = str(model_manifest.get("checkpointPath") or "").strip()
    if not checkpoint_path or not _checkpoint_file_ready(Path(checkpoint_path)):
        return None
    asset_id = str(model_manifest.get("modelAssetId") or _stable_final_asset_id(train_job_id))
    return {
        "modelAssetId": asset_id,
        "name": model_manifest.get("displayName") or model_manifest.get("name") or asset_id,
        "displayName": model_manifest.get("displayName") or model_manifest.get("name"),
        "checkpointKind": "final",
        "checkpointEpoch": None,
        "checkpointMetricName": None,
        "sourceTrainJobId": train_job_id,
        "sourceDatasetId": model_manifest.get("sourceDatasetId"),
        "taskType": model_manifest.get("taskType"),
        "taskTemplateId": model_manifest.get("taskTemplateId"),
        "trainingBackend": model_manifest.get("trainingBackend"),
        "backendType": model_manifest.get("backendType"),
        "framework": model_manifest.get("framework"),
        "modelType": model_manifest.get("modelType"),
        "checkpointPath": checkpoint_path,
        "trainingTaskName": model_manifest.get("trainingTaskName"),
        "datasetDisplayName": model_manifest.get("datasetDisplayName"),
        "status": "ready",
        "createdAt": model_manifest.get("createdAt") or _utc_now_label(),
        "evalExecutor": model_manifest.get("evalExecutor"),
        "trainedActionMode": model_manifest.get("trainedActionMode") or model_manifest.get("actionMode"),
        "actionMode": model_manifest.get("actionMode") or model_manifest.get("trainedActionMode"),
        "controllerType": model_manifest.get("controllerType"),
        "actionSchema": model_manifest.get("actionSchema"),
        "observationSchema": model_manifest.get("observationSchema"),
        "controllerSchema": model_manifest.get("controllerSchema"),
        "sideChannelSchema": model_manifest.get("sideChannelSchema"),
        "canEvaluate": model_manifest.get("canEvaluate"),
        "evalDisabledReason": model_manifest.get("evalDisabledReason"),
        "datasetFormat": model_manifest.get("datasetFormat"),
        "policyType": model_manifest.get("policyType"),
        "stateDim": model_manifest.get("stateDim"),
        "actionDim": model_manifest.get("actionDim"),
        "actionRepresentation": model_manifest.get("actionRepresentation"),
        "taskInstruction": model_manifest.get("taskInstruction"),
        "imageKeys": model_manifest.get("imageKeys"),
        "lowDimKeys": model_manifest.get("lowDimKeys"),
    }


def _merge_registry_with_manifest_and_discovered(
    *,
    train_job_dir: Path,
    train_job_id: str,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    status: dict[str, Any],
    resolved_backend: str,
    framework_label: str,
    model_type: str,
    normalized_assets: list[dict[str, Any]],
    total_epochs: int,
    training_complete: bool,
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}

    for item in normalized_assets:
        row = dict(item)
        asset_id = str(row.get("modelAssetId") or "")
        if asset_id:
            by_id[asset_id] = row
        path = str(row.get("checkpointPath") or "").strip()
        if path:
            by_path[_path_key(path)] = row

    model_manifest = _read_json(train_job_dir / "artifacts" / "model_manifest.json")
    manifest_entry = _entry_from_model_manifest(model_manifest, train_job_id=train_job_id)
    if manifest_entry:
        path_key = _path_key(str(manifest_entry.get("checkpointPath") or ""))
        asset_id = str(manifest_entry.get("modelAssetId") or "")
        if path_key not in by_path and asset_id not in by_id:
            normalized_assets.append(manifest_entry)
            by_path[path_key] = manifest_entry
            by_id[asset_id] = manifest_entry

    discovered = normalize_discovered_checkpoints(
        discover_checkpoints(train_job_dir),
        total_epochs,
        training_complete=training_complete,
    )
    context = _context_label(train_config, status, manifest)
    for checkpoint in discovered:
        path_key = _path_key(checkpoint.path)
        if path_key in by_path:
            continue
        asset_id = stable_asset_id_for_checkpoint(train_job_id, checkpoint)
        if asset_id in by_id:
            continue
        display_name = build_checkpoint_asset_display_name(
            context_label=context,
            kind=checkpoint.kind,
            epoch=checkpoint.epoch,
            metric_name=checkpoint.metric_name,
        )
        entry = _manifest_for_checkpoint(
            model_asset_id=asset_id,
            train_job_id=train_job_id,
            checkpoint=checkpoint,
            display_name=display_name,
            manifest=manifest,
            status=status,
            train_config=train_config,
            resolved_backend=resolved_backend,
            framework_label=framework_label,
            model_type=model_type,
            allow_final=training_complete,
        )
        normalized_assets.append(entry)
        by_path[path_key] = entry
        by_id[asset_id] = entry

    merged = normalize_registry_assets(
        normalized_assets,
        status=status,
        total_epochs=total_epochs,
        train_job_dir=train_job_dir,
    )
    return merged


def resolve_training_job_model_assets_list_message(
    *,
    rows: list[dict[str, Any]],
    train_config: dict[str, Any],
    status: dict[str, Any],
    train_job_dir: Path,
) -> Optional[str]:
    if rows:
        if any(str(row.get("displayStatus") or "") == "missing" for row in rows):
            return None
        return None
    from app.services.training_job_generated_assets import resolve_training_job_detail_empty_message

    in_progress_message = resolve_training_job_detail_empty_message(
        status=status,
        train_job_dir=train_job_dir,
    )
    if in_progress_message:
        return in_progress_message
    training_complete = is_training_job_complete(status, train_job_dir=train_job_dir)
    if not training_complete:
        return None
    if not save_policy_has_outputs(train_config):
        return "未启用任何模型保存策略，暂无模型资产"
    if bool(parse_save_policy(train_config).get("saveFinal", True)):
        return "训练已完成，但未找到最终模型文件"
    if discover_checkpoints(train_job_dir):
        return "训练已完成，但未找到可展示的模型资产"
    return "暂无模型资产"


def explain_model_asset_eval_blocker(entry: dict[str, Any], *, job_status: dict[str, Any]) -> str | None:
    from app.services.policy_schema_resolver import is_joint_space_policy_schema, _norm_low_dim_keys

    if entry.get("isPlaceholder"):
        return "最终模型尚未生成"
    asset_status = str(entry.get("status") or entry.get("displayStatus") or "").lower()
    if asset_status in {"superseded"}:
        return "该 checkpoint 已被更新的最佳模型替换"
    if asset_status in {"pending", "generating", "waiting"}:
        return "模型资产生成中"
    if asset_status not in {"ready", "available", "active", "completed"}:
        return f"模型资产状态不可用（{asset_status or 'unknown'}）"

    path = str(entry.get("checkpointPath") or "").strip()
    if not path:
        return "checkpoint 路径缺失"
    if not _checkpoint_file_ready(Path(path)):
        return "checkpoint 文件不存在或不可读"

    model_type = str(entry.get("modelType") or entry.get("framework") or "").lower()
    eval_executor = str(entry.get("evalExecutor") or "").strip()
    controller_type = str(entry.get("controllerType") or "").strip()
    action_dim_raw = entry.get("actionDim")
    try:
        action_dim = int(action_dim_raw) if action_dim_raw is not None else None
    except (TypeError, ValueError):
        action_dim = None

    if model_type in {"pi0", "openpi"} or str(entry.get("policyType") or "").lower() == "pi0":
        from app.services.policy_schema_resolver import (
            explain_pi0_model_asset_eval_blocker,
            is_pi0_joint_space_eval_asset,
        )

        if is_pi0_joint_space_eval_asset(entry, checkpoint_path=entry.get("checkpointPath")):
            blocker = explain_pi0_model_asset_eval_blocker(
                entry,
                checkpoint_path=entry.get("checkpointPath"),
            )
            if blocker:
                return blocker
        elif entry.get("canEvaluate") is False:
            return str(
                entry.get("evalDisabledReason")
                or entry.get("canEvaluateReason")
                or "pi0 eval adapter not ready"
            )
        if not eval_executor or eval_executor.lower() in {"pi0_not_ready", "null", "none"}:
            return "pi0 eval adapter not ready"

    if model_type in {"act", "diffusion_policy"} and is_joint_space_policy_schema(
        eval_executor=eval_executor,
        controller_type=controller_type,
        action_mode=str(entry.get("actionMode") or entry.get("trainedActionMode") or ""),
        action_dim=action_dim,
        low_dim_keys=_norm_low_dim_keys(entry.get("lowDimKeys") or []),
        preferred_policy_schema_id=str(entry.get("preferredPolicySchemaId") or ""),
    ):
        if not eval_executor:
            return "evalExecutor 缺失，无法确定 joint-space 评测执行器"
        if not controller_type:
            return "controllerType 缺失，无法确定控制器类型"
        if action_dim is None:
            return "actionDim 缺失，无法确定动作维度"

    kind = str(entry.get("checkpointKind") or "").lower()
    if kind == "final" and not is_training_job_complete(job_status):
        return "训练尚未完成，Final 模型暂不可评测"
    return None


def compute_asset_can_evaluate(
    entry: dict[str, Any],
    *,
    job_status: dict[str, Any],
) -> bool:
    return explain_model_asset_eval_blocker(entry, job_status=job_status) is None


def compute_asset_display_status(
    entry: dict[str, Any],
    *,
    job_status: dict[str, Any],
) -> str:
    if entry.get("isPlaceholder"):
        return str(entry.get("displayStatus") or "waiting")
    asset_status = str(entry.get("status") or "").lower()
    if asset_status == "superseded":
        return "superseded"
    kind = str(entry.get("checkpointKind") or "").lower()
    path = str(entry.get("checkpointPath") or "").strip()
    file_ready = bool(path) and _checkpoint_file_ready(Path(path))
    if kind == "final" and not is_training_job_complete(job_status):
        if file_ready:
            return "ready"
        return "waiting"
    if compute_asset_can_evaluate(entry, job_status=job_status):
        return "ready"
    if (
        is_training_job_complete(job_status)
        and file_ready
        and asset_status in {"ready", "available", "active", "completed"}
    ):
        return "ready"
    if asset_status == "generating":
        return "generating"
    return "waiting"


def list_training_job_detail_registry_entries(
    *,
    train_job_dir: Path,
    train_job_id: str,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    status: dict[str, Any],
    resolved_backend: str,
    framework_label: str,
    model_type: str,
) -> list[dict[str, Any]]:
    """训练任务详情：合并注册表、model_manifest、磁盘 checkpoint。"""
    total_epochs = int(status.get("totalEpochs") or train_config.get("epochs") or 0)
    training_complete = is_training_job_complete(status, train_job_dir=train_job_dir)
    save_final = bool(parse_save_policy(train_config).get("saveFinal", True))

    register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest=manifest,
        train_config=train_config,
        status=status,
        resolved_backend=resolved_backend,
        framework_label=framework_label,
        model_type=model_type,
        register_final=training_complete,
    )

    registry = read_registry(train_job_dir)
    raw_assets = [dict(item) for item in (registry.get("assets") or []) if isinstance(item, dict)]
    normalized = normalize_registry_assets(
        raw_assets,
        status=status,
        total_epochs=total_epochs,
        train_job_dir=train_job_dir,
    )
    merged_assets = _merge_registry_with_manifest_and_discovered(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest=manifest,
        train_config=train_config,
        status=status,
        resolved_backend=resolved_backend,
        framework_label=framework_label,
        model_type=model_type,
        normalized_assets=normalized,
        total_epochs=total_epochs,
        training_complete=training_complete,
    )
    ready_assets = list_displayable_registry_assets(
        merged_assets,
        status=status,
        total_epochs=total_epochs,
        train_job_dir=train_job_dir,
    )
    job_active = is_training_job_in_progress(status) or not training_complete

    rows: list[dict[str, Any]] = []
    final_ready = next(
        (item for item in ready_assets if str(item.get("checkpointKind") or "").lower() == "final"),
        None,
    )

    if final_ready and training_complete:
        item = dict(final_ready)
        item["isPlaceholder"] = False
        item["canEvaluate"] = compute_asset_can_evaluate(item, job_status=status)
        blocker = explain_model_asset_eval_blocker(item, job_status=status)
        if blocker:
            item["canEvaluateReason"] = blocker
        item["displayStatus"] = compute_asset_display_status(item, job_status=status)
        rows.append(item)
    elif save_final and job_active:
        rows.append(
            build_final_placeholder_entry(
                train_job_id=train_job_id,
                manifest=manifest,
                train_config=train_config,
                status=status,
                resolved_backend=resolved_backend,
                framework_label=framework_label,
                model_type=model_type,
                train_job_dir=train_job_dir,
            )
        )
    elif training_complete and save_final:
        rows.append(
            build_missing_final_entry(
                train_job_id=train_job_id,
                manifest=manifest,
                train_config=train_config,
                status=status,
                resolved_backend=resolved_backend,
                framework_label=framework_label,
                model_type=model_type,
            )
        )

    for item in ready_assets:
        if str(item.get("checkpointKind") or "").lower() == "final":
            continue
        if job_active:
            continue
        enriched = dict(item)
        enriched["isPlaceholder"] = False
        enriched["canEvaluate"] = compute_asset_can_evaluate(enriched, job_status=status)
        blocker = explain_model_asset_eval_blocker(enriched, job_status=status)
        if blocker:
            enriched["canEvaluateReason"] = blocker
        enriched["displayStatus"] = compute_asset_display_status(enriched, job_status=status)
        rows.append(enriched)

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        kind = str(item.get("checkpointKind") or "").lower()
        epoch = int(item.get("checkpointEpoch") or 0)
        return KIND_DISPLAY_ORDER.get(kind, 9), epoch

    return sorted(rows, key=sort_key)


def register_checkpoint_assets(
    *,
    train_job_dir: Path,
    train_job_id: str,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    status: dict[str, Any],
    resolved_backend: str,
    framework_label: str,
    model_type: str,
    checkpoints: Optional[list[CheckpointRecord]] = None,
    register_final: bool | None = None,
) -> list[dict[str, Any]]:
    training_complete = register_final if register_final is not None else is_training_job_complete(status)
    total_epochs = int(status.get("totalEpochs") or train_config.get("epochs") or 0)

    discovered = checkpoints if checkpoints is not None else discover_checkpoints(train_job_dir)
    discovered = normalize_discovered_checkpoints(
        discovered,
        total_epochs,
        training_complete=training_complete,
    )
    if not training_complete:
        discovered = [item for item in discovered if item.kind != "final"]
    discovered = _dedupe_discovered_by_path(discovered)
    if not discovered:
        registry = read_registry(train_job_dir)
        assets = list(registry.get("assets") or [])
        assets = normalize_registry_assets(assets, status=status, total_epochs=total_epochs)
        _write_json(
            registry_path(train_job_dir),
            {
                "version": 1,
                "sourceTrainJobId": train_job_id,
                "updatedAt": _utc_now_label(),
                "assets": assets,
            },
        )
        return list_displayable_registry_assets(assets, status=status, total_epochs=total_epochs)

    context = _context_label(train_config, status, manifest)
    registry = read_registry(train_job_dir)
    assets: list[dict[str, Any]] = [
        dict(item) for item in (registry.get("assets") or []) if isinstance(item, dict)
    ]
    active_paths = {
        _path_key(str(item.get("checkpointPath") or ""))
        for item in assets
        if isinstance(item, dict)
        and str(item.get("status") or "").lower() not in {"superseded", "pending"}
        and str(item.get("checkpointPath") or "").strip()
    }

    manifests_dir = train_job_dir / "artifacts" / "checkpoint_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint in discovered:
        if checkpoint.kind == "final" and not training_complete:
            continue

        resolved_path = _path_key(checkpoint.path)
        asset_id = stable_asset_id_for_checkpoint(train_job_id, checkpoint)
        display_name = build_checkpoint_asset_display_name(
            context_label=context,
            kind=checkpoint.kind,
            epoch=checkpoint.epoch,
            metric_name=checkpoint.metric_name,
        )
        entry = _manifest_for_checkpoint(
            model_asset_id=asset_id,
            train_job_id=train_job_id,
            checkpoint=checkpoint,
            display_name=display_name,
            manifest=manifest,
            status=status,
            train_config=train_config,
            resolved_backend=resolved_backend,
            framework_label=framework_label,
            model_type=model_type,
            allow_final=training_complete,
        )

        if checkpoint.kind == "best":
            index = _find_asset_index(assets, asset_id)
            if index is not None:
                previous = dict(assets[index])
                entry["createdAt"] = previous.get("createdAt") or entry["createdAt"]
                assets[index] = entry
            else:
                assets.append(entry)
            _mark_other_best_superseded(
                assets,
                train_job_id=train_job_id,
                metric_name=checkpoint.metric_name,
                keep_asset_id=asset_id,
            )
            active_paths.add(resolved_path)
            _write_json(manifests_dir / f"{asset_id}.json", entry)
            continue

        if checkpoint.kind == "final":
            index = _find_asset_index(assets, asset_id)
            if index is not None:
                previous = dict(assets[index])
                entry["createdAt"] = previous.get("createdAt") or entry["createdAt"]
                assets[index] = entry
            else:
                assets.append(entry)
            active_paths.add(resolved_path)
            _write_json(manifests_dir / f"{asset_id}.json", entry)
            continue

        if resolved_path in active_paths:
            continue
        if _find_asset_index(assets, asset_id) is not None:
            continue

        assets.append(entry)
        active_paths.add(resolved_path)
        _write_json(manifests_dir / f"{asset_id}.json", entry)

    assets = normalize_registry_assets(assets, status=status, total_epochs=total_epochs)

    registry_payload = {
        "version": 1,
        "sourceTrainJobId": train_job_id,
        "updatedAt": _utc_now_label(),
        "assets": assets,
    }
    _write_json(registry_path(train_job_dir), registry_payload)

    displayable = list_displayable_registry_assets(assets, status=status, total_epochs=total_epochs)
    primary = next((item for item in displayable if item.get("checkpointKind") == "final"), None)
    if primary is None and displayable:
        primary = displayable[0]
    if primary and training_complete:
        existing = _read_json(train_job_dir / "artifacts" / "model_manifest.json")
        merged = dict(primary)
        if existing:
            preserve_keys = (
                "datasetFormat",
                "datasetPath",
                "policyType",
                "stateDim",
                "actionDim",
                "robot",
                "controllerType",
                "actionMode",
                "actionRepresentation",
                "trainedActionMode",
                "taskInstruction",
                "imageKeys",
                "lowDimKeys",
                "observationSchema",
                "actionSchema",
                "evalExecutor",
                "canEvaluate",
                "evalDisabledReason",
                "field_mapping",
                "norm_stats_source",
                "pi0ReadyData",
            )
            for key in preserve_keys:
                if existing.get(key) is not None:
                    merged[key] = existing[key]
        _write_json(train_job_dir / "artifacts" / "model_manifest.json", merged)

    return displayable
