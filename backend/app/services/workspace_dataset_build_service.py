"""将已导入 HDF5 整理为标准训练数据集（built 目录）。"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.platform_paths import platform_paths
from app.services.dataset_naming import apply_dataset_row_display_fields, task_display_name
from app.services.workspace_dataset_import_service import (
    IMPORT_ROOT,
    _matches_field_category,
    _parse_hdf5_file,
    _read_import_metadata,
    _resolve_import_dir as _resolve_import_dataset_dir,
    _walk_hdf5_tree,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
BUILT_ROOT = platform_paths.runs_root / "datasets" / "built"

AUTO_FIELD_DETECT_FAILED_MSG = "未能自动识别训练所需字段，请打开高级配置手动指定。"

BUILD_AUTO_FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "action": ("action", "actions", "cmd", "target_qpos"),
    "qpos": ("qpos", "joint_pos", "joint_positions", "state"),
    "image": ("image", "rgb", "camera", "cam"),
    "qvel": ("qvel",),
    "done": ("done",),
}

STANDARD_FIELD_FALLBACKS: dict[str, tuple[str, ...]] = {
    "action": ("data/demo_0/actions", "data/demo_0/action"),
    "qpos": ("data/demo_0/obs/qpos", "data/demo_0/obs/state"),
    "image": ("data/demo_0/obs/image", "data/demo_0/obs/rgb"),
    "qvel": ("data/demo_0/obs/qvel",),
    "done": ("data/demo_0/done",),
}

TARGET_FORMAT_STANDARD_HDF5 = "standard_hdf5"
EPISODE_RULE_SINGLE = "single_episode"


def _built_roots() -> tuple[Path, ...]:
    return (BUILT_ROOT,)


def _resolve_built_dir(dataset_id: str) -> Path:
    for root in _built_roots():
        candidate = root / dataset_id
        if candidate.is_dir():
            return candidate
    return BUILT_ROOT / dataset_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_built_dataset_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"ds_built_{stamp}_{suffix}"


def make_built_source_job_id(built_dataset_id: str) -> str:
    return f"built_{built_dataset_id}"


def _resolve_import_dir(source_dataset_id: str) -> Path:
    candidate = (source_dataset_id or "").strip()
    if not candidate.startswith("ds_import_"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sourceDatasetId 无效")
    import_dir = _resolve_import_dataset_dir(candidate)
    if not import_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="源数据集不存在")
    hdf5_path = import_dir / "source.hdf5"
    if not hdf5_path.is_file():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="源 HDF5 文件不存在")
    return import_dir


def _load_schema_fields(import_dir: Path, hdf5_path: Path) -> list[dict[str, Any]]:
    schema_path = import_dir / "schema.json"
    if schema_path.is_file():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            fields = schema.get("fields")
            if isinstance(fields, list) and fields:
                return [item for item in fields if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            pass
    parsed = _parse_hdf5_file(hdf5_path)
    return list(parsed.get("tree") or [])


def get_import_dataset_schema(source_dataset_id: str) -> dict[str, Any]:
    import_dir = _resolve_import_dir(source_dataset_id)
    hdf5_path = import_dir / "source.hdf5"
    metadata = _read_import_metadata(import_dir)
    fields = _load_schema_fields(import_dir, hdf5_path)
    return {
        "datasetId": source_dataset_id,
        "fields": fields,
        "recognizedFields": metadata.get("recognizedFields"),
    }


def _detect_build_auto_fields(tree: list[dict[str, Any]]) -> dict[str, list[str]]:
    recognized: dict[str, list[str]] = {key: [] for key in BUILD_AUTO_FIELD_PATTERNS}
    for entry in tree:
        path = str(entry.get("path") or "")
        if not path:
            continue
        for category, hints in BUILD_AUTO_FIELD_PATTERNS.items():
            if _matches_field_category(path, hints):
                recognized[category].append(path)
    return recognized


def _recognized_lists_from_metadata(metadata: dict[str, Any]) -> dict[str, list[str]]:
    recognized: dict[str, list[str]] = {key: [] for key in BUILD_AUTO_FIELD_PATTERNS}
    raw = metadata.get("recognizedFields")
    if not isinstance(raw, dict):
        return recognized
    for key in BUILD_AUTO_FIELD_PATTERNS:
        value = raw.get(key)
        if isinstance(value, list):
            recognized[key] = [str(item) for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            recognized[key] = [value.strip()]
    return recognized


def _merge_recognized_field_lists(*sources: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {key: [] for key in BUILD_AUTO_FIELD_PATTERNS}
    seen: dict[str, set[str]] = {key: set() for key in BUILD_AUTO_FIELD_PATTERNS}
    for source in sources:
        for key, paths in source.items():
            if key not in merged:
                continue
            for path in paths:
                clean = path.strip()
                if not clean or clean in seen[key]:
                    continue
                seen[key].add(clean)
                merged[key].append(clean)
    return merged


def _pick_primary_field(paths: list[str]) -> Optional[str]:
    if not paths:
        return None
    return sorted(paths, key=lambda item: (item.count("/"), len(item)))[0]


def _extract_manual_field_mapping(raw: dict[str, Any]) -> dict[str, str]:
    manual: dict[str, str] = {}
    for key in ("action", "qpos", "image", "qvel", "done"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            manual[key] = _normalize_field_path(value)
    return manual


def _auto_resolve_field_mapping(
    handle: Any,
    import_dir: Path,
    hdf5_path: Path,
    *,
    fail_if_incomplete: bool,
) -> dict[str, str]:
    tree = _load_schema_fields(import_dir, hdf5_path)
    metadata = _read_import_metadata(import_dir)

    recognized = _merge_recognized_field_lists(
        _recognized_lists_from_metadata(metadata),
        _detect_build_auto_fields(tree),
    )

    for category, fallbacks in STANDARD_FIELD_FALLBACKS.items():
        for candidate in fallbacks:
            if candidate in recognized.get(category, []):
                continue
            try:
                _resolve_hdf5_node(handle, candidate)
            except KeyError:
                continue
            recognized.setdefault(category, []).append(candidate)

    resolved: dict[str, str] = {}
    action_path = _pick_primary_field(recognized.get("action") or [])
    if action_path:
        resolved["action"] = action_path

    qpos_path = _pick_primary_field(recognized.get("qpos") or [])
    if qpos_path:
        resolved["qpos"] = qpos_path

    image_path = _pick_primary_field(recognized.get("image") or [])
    if image_path:
        resolved["image"] = image_path

    qvel_path = _pick_primary_field(recognized.get("qvel") or [])
    if qvel_path:
        resolved["qvel"] = qvel_path

    done_path = _pick_primary_field(recognized.get("done") or [])
    if done_path:
        resolved["done"] = done_path

    has_action = bool(resolved.get("action"))
    has_obs = bool(resolved.get("qpos") or resolved.get("image"))
    if fail_if_incomplete and (not has_action or not has_obs):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=AUTO_FIELD_DETECT_FAILED_MSG,
        )
    return resolved


def _resolve_build_field_mapping(
    handle: Any,
    import_dir: Path,
    hdf5_path: Path,
    field_mapping_raw: dict[str, Any],
    *,
    auto: bool,
) -> dict[str, Any]:
    manual = _extract_manual_field_mapping(field_mapping_raw)
    if manual:
        auto_mapping = _auto_resolve_field_mapping(
            handle,
            import_dir,
            hdf5_path,
            fail_if_incomplete=False,
        )
        return {**auto_mapping, **manual}
    if auto:
        return _auto_resolve_field_mapping(
            handle,
            import_dir,
            hdf5_path,
            fail_if_incomplete=True,
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=AUTO_FIELD_DETECT_FAILED_MSG,
    )


def _normalize_field_path(raw: str) -> str:
    return raw.strip().lstrip("/")


def _resolve_hdf5_node(handle: Any, path: str) -> Any:
    import h5py

    clean = _normalize_field_path(path)
    if not clean:
        raise KeyError("empty path")
    node: Any = handle
    for part in clean.split("/"):
        if part not in node:
            raise KeyError(path)
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node
    raise KeyError(path)


def _first_dim(shape: Any) -> int:
    if shape is None or len(shape) < 1:
        return 0
    return int(shape[0])


def _validate_field_mapping(
    handle: Any,
    field_mapping: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    action_path = field_mapping.get("action")
    if not isinstance(action_path, str) or not action_path.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Action 字段必选")

    resolved: dict[str, str] = {"action": _normalize_field_path(action_path)}
    optional_keys = ("qpos", "image", "qvel", "done")
    for key in optional_keys:
        raw = field_mapping.get(key)
        if isinstance(raw, str) and raw.strip():
            resolved[key] = _normalize_field_path(raw)

    if "qpos" not in resolved and "image" not in resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Qpos 或 Image 至少选择一个",
        )

    shapes: dict[str, Any] = {}
    try:
        action_ds = _resolve_hdf5_node(handle, resolved["action"])
        action_t = _first_dim(action_ds.shape)
        shapes["action"] = list(action_ds.shape)
        if action_t <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Action 时间维无效")

        obs_t: Optional[int] = None
        for obs_key in ("qpos", "image"):
            if obs_key not in resolved:
                continue
            obs_ds = _resolve_hdf5_node(handle, resolved[obs_key])
            obs_t = _first_dim(obs_ds.shape)
            shapes[obs_key] = list(obs_ds.shape)
            if obs_t <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{obs_key} 时间维无效",
                )
            if obs_t != action_t:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Action 时间维({action_t})与 {obs_key} 时间维({obs_t})不一致",
                )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"字段路径不存在: {exc.args[0]}",
        ) from exc

    return resolved, shapes


def _copy_dataset_to_group(target_group: Any, source_ds: Any) -> None:
    import numpy as np

    data = np.array(source_ds)
    target_group.create_dataset(source_ds.name, data=data, dtype=source_ds.dtype)


def build_dataset_from_import(payload: dict[str, Any]) -> dict[str, Any]:
    source_dataset_id = str(payload.get("sourceDatasetId") or "").strip()
    output_name = str(payload.get("outputName") or "").strip()
    task_type = str(payload.get("taskType") or "custom").strip() or "custom"
    target_format = str(payload.get("targetFormat") or TARGET_FORMAT_STANDARD_HDF5).strip()
    field_mapping_raw = payload.get("fieldMapping") or {}
    auto = bool(payload.get("auto", True))
    episode_rule = payload.get("episodeRule") or {}

    if target_format != TARGET_FORMAT_STANDARD_HDF5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="targetFormat 仅支持 standard_hdf5")
    if field_mapping_raw is not None and not isinstance(field_mapping_raw, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="fieldMapping 无效")
    rule_type = str(episode_rule.get("type") or EPISODE_RULE_SINGLE).strip() or EPISODE_RULE_SINGLE
    if rule_type != EPISODE_RULE_SINGLE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="episodeRule 仅支持 single_episode")
    if not output_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请填写构建后名称")

    import_dir = _resolve_import_dir(source_dataset_id)
    source_hdf5 = import_dir / "source.hdf5"
    source_metadata = _read_import_metadata(import_dir)
    source_name = str(source_metadata.get("displayName") or source_metadata.get("name") or source_dataset_id)

    built_dataset_id = make_built_dataset_id()
    built_dir = BUILT_ROOT / built_dataset_id
    built_dir.mkdir(parents=True, exist_ok=True)
    output_hdf5 = built_dir / "dataset.hdf5"

    import h5py
    import numpy as np

    shapes: dict[str, Any] = {}
    checked_fields: list[str] = []

    with h5py.File(source_hdf5, "r") as src, h5py.File(output_hdf5, "w") as dst:
        resolved_raw = _resolve_build_field_mapping(
            src,
            import_dir,
            source_hdf5,
            field_mapping_raw if isinstance(field_mapping_raw, dict) else {},
            auto=auto,
        )
        resolved_mapping, shapes = _validate_field_mapping(src, resolved_raw)
        data_group = dst.create_group("data")
        demo = data_group.create_group("demo_0")
        obs_group = demo.create_group("obs")

        action_ds = _resolve_hdf5_node(src, resolved_mapping["action"])
        demo.create_dataset("actions", data=np.array(action_ds), dtype=action_ds.dtype)
        checked_fields.append(resolved_mapping["action"])

        if "qpos" in resolved_mapping:
            qpos_ds = _resolve_hdf5_node(src, resolved_mapping["qpos"])
            obs_group.create_dataset("qpos", data=np.array(qpos_ds), dtype=qpos_ds.dtype)
            checked_fields.append(resolved_mapping["qpos"])
        if "image" in resolved_mapping:
            image_ds = _resolve_hdf5_node(src, resolved_mapping["image"])
            obs_group.create_dataset("image", data=np.array(image_ds), dtype=image_ds.dtype)
            checked_fields.append(resolved_mapping["image"])
        if "qvel" in resolved_mapping:
            qvel_ds = _resolve_hdf5_node(src, resolved_mapping["qvel"])
            obs_group.create_dataset("qvel", data=np.array(qvel_ds), dtype=qvel_ds.dtype)
            checked_fields.append(resolved_mapping["qvel"])
        if "done" in resolved_mapping:
            done_ds = _resolve_hdf5_node(src, resolved_mapping["done"])
            demo.create_dataset("done", data=np.array(done_ds), dtype=done_ds.dtype)
            checked_fields.append(resolved_mapping["done"])

    file_size = output_hdf5.stat().st_size
    created_at = _utc_now_iso()

    with h5py.File(output_hdf5, "r") as built_handle:
        built_tree = _walk_hdf5_tree(built_handle)
    built_schema = {
        "fields": built_tree,
        "sourceDatasetId": source_dataset_id,
        "fieldMapping": resolved_mapping,
        "targetFormat": target_format,
    }
    validation_report = {
        "status": "ready",
        "passed": True,
        "trainable": True,
        "directTrainable": True,
        "errors": [],
        "warnings": [],
        "checkedFields": checked_fields,
        "actionShape": shapes.get("action"),
        "qposShape": shapes.get("qpos"),
        "imageShape": shapes.get("image"),
        "generatedAt": created_at,
    }

    manifest = {
        "builtDatasetId": built_dataset_id,
        "id": built_dataset_id,
        "datasetId": built_dataset_id,
        "sourceDatasetId": source_dataset_id,
        "sourceDatasetName": source_name,
        "name": output_name,
        "displayName": output_name,
        "outputName": output_name,
        "taskType": task_type,
        "taskDisplayName": task_display_name(task_type),
        "sourceType": "real_robot_built",
        "dataSourceLabel": "真实数据构建",
        "format": "hdf5",
        "datasetFormat": "hdf5",
        "targetFormat": target_format,
        "status": "ready",
        "episodeCount": 1,
        "dataCount": 1,
        "episodeParsed": True,
        "totalEpisodes": 1,
        "generationRounds": 1,
        "datasetFile": str(output_hdf5),
        "builtDatasetPath": str(output_hdf5),
        "storagePath": str(built_dir),
        "manifestPath": str(built_dir / "manifest.json"),
        "sourceJobId": make_built_source_job_id(built_dataset_id),
        "fieldMapping": resolved_mapping,
        "episodeRule": {"type": EPISODE_RULE_SINGLE},
        "trainable": True,
        "directTrainable": True,
        "needsBuild": False,
        "needsMapping": False,
        "fileSizeBytes": file_size,
        "createdAt": created_at,
        "updatedAt": created_at,
    }

    _write_json(built_dir / "manifest.json", manifest)
    _write_json(built_dir / "schema.json", built_schema)
    _write_json(built_dir / "validation_report.json", validation_report)

    row = dict(manifest)
    apply_dataset_row_display_fields(
        row,
        task_type=task_type,
        manifest=manifest,
        source_job_id=str(manifest["sourceJobId"]),
    )
    return {
        "builtDatasetId": built_dataset_id,
        "status": "ready",
        "trainable": True,
        "directTrainable": True,
        "dataset": row,
    }


def built_record_to_dataset_row(built_dir: Path) -> Optional[dict[str, Any]]:
    manifest_path = built_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None

    dataset_id = str(manifest.get("builtDatasetId") or manifest.get("id") or built_dir.name)
    hdf5_path = built_dir / "dataset.hdf5"
    if not hdf5_path.is_file():
        alt = manifest.get("datasetFile")
        if alt and Path(str(alt)).is_file():
            hdf5_path = Path(str(alt))
        else:
            return None

    task_type = str(manifest.get("taskType") or "custom")
    row: dict[str, Any] = {
        "id": dataset_id,
        "name": str(manifest.get("name") or manifest.get("displayName") or dataset_id),
        "displayName": str(manifest.get("displayName") or manifest.get("name") or dataset_id),
        "sourceJobId": str(manifest.get("sourceJobId") or make_built_source_job_id(dataset_id)),
        "sourceType": "real_robot_built",
        "dataSourceLabel": "真实数据构建",
        "taskType": task_type,
        "taskDisplayName": str(manifest.get("taskDisplayName") or task_display_name(task_type)),
        "manifestPath": str(manifest_path),
        "episodeCount": int(manifest.get("episodeCount") or 1),
        "dataCount": int(manifest.get("dataCount") or 1),
        "totalEpisodes": int(manifest.get("totalEpisodes") or 1),
        "generationRounds": int(manifest.get("generationRounds") or 1),
        "storagePath": str(built_dir),
        "format": "hdf5",
        "status": str(manifest.get("status") or "ready"),
        "replayAvailable": False,
        "datasetFile": str(hdf5_path),
        "builtDatasetPath": str(hdf5_path),
        "fileSizeBytes": int(manifest.get("fileSizeBytes") or hdf5_path.stat().st_size),
        "trainable": True,
        "directTrainable": True,
        "needsBuild": False,
        "needsMapping": False,
        "episodeParsed": True,
        "sourceDatasetId": manifest.get("sourceDatasetId"),
        "createdAt": str(manifest.get("createdAt") or _utc_now_iso()),
        "updatedAt": str(manifest.get("updatedAt") or manifest.get("createdAt") or _utc_now_iso()),
    }
    apply_dataset_row_display_fields(
        row,
        task_type=task_type,
        manifest=manifest,
        source_job_id=str(row["sourceJobId"]),
    )
    return row


def list_built_datasets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in _built_roots():
        if not root.is_dir():
            continue
        for built_dir in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
            if (
                not built_dir.is_dir()
                or not built_dir.name.startswith("ds_built_")
                or built_dir.name in seen
            ):
                continue
            row = built_record_to_dataset_row(built_dir)
            if row:
                seen.add(built_dir.name)
                rows.append(row)
    return rows


def delete_built_dataset(built_dataset_id: str) -> dict[str, Any]:
    candidate = (built_dataset_id or "").strip()
    if not candidate.startswith("ds_built_"):
        raise ValueError("builtDatasetId 无效")
    built_dir = _resolve_built_dir(candidate)
    if not built_dir.is_dir():
        raise FileNotFoundError("构建数据集不存在")
    import shutil

    shutil.rmtree(built_dir)
    return {"ok": True, "builtDatasetId": candidate}
