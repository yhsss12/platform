"""Workspace 数据中心 — 本地 HDF5 文件上传导入。"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, UploadFile, status

from app.core.platform_paths import platform_paths
from app.services.adapter_layer.hdf5_inspector import inspect_hdf5
from app.services.dataset_naming import apply_dataset_row_display_fields, task_display_name

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
IMPORT_ROOT = platform_paths.runs_root / "datasets" / "imports"
TRAINING_JOBS_ROOT = platform_paths.training_jobs
MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
ALLOWED_SUFFIXES = {".hdf5", ".h5"}

DATA_SOURCE_LABELS: dict[str, str] = {
    "real_collection": "真实导入",
    "simulation_export": "仿真导出",
    "public_dataset": "外部公开数据",
    "other": "其他",
}

TASK_TYPE_MAP: dict[str, str] = {
    "cable_threading": "cable_threading",
    "dual_arm_cable": "dual_arm_cable_manipulation",
    "stack_cube": "isaaclab_franka_stack_cube",
    "custom": "custom",
}

TASK_DISPLAY_MAP: dict[str, str] = {
    "cable_threading": "线缆穿杆",
    "dual_arm_cable_manipulation": "线缆整理",
    "isaaclab_franka_stack_cube": "Stack Cube",
    "custom": "自定义",
}

TASK_TEMPLATE_MAP: dict[str, str] = {
    "cable_threading": "task_cable_threading_v1",
    "dual_arm_cable_manipulation": "task_dual_arm_cable_manipulation_v1",
    "isaaclab_franka_stack_cube": "task_isaaclab_franka_stack_cube_v1",
    "custom": "custom_import",
}

FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "action": ("action", "actions"),
    "qpos": ("qpos", "state"),
    "qvel": ("qvel",),
    "image": ("image", "rgb", "camera", "cam"),
    "depth": ("depth",),
}

IMPORT_STATUS_READY = "ready"
IMPORT_STATUS_NEEDS_MAPPING = "needs_mapping"
IMPORT_STATUS_NEEDS_BUILD = "needs_build"
IMPORT_STATUS_FAILED = "failed"
IMPORT_STATUS_PARSING = "parsing"

# 兼容旧测试与历史 metadata
IMPORT_STATUS_AVAILABLE = IMPORT_STATUS_READY
IMPORT_STATUS_PENDING_MAPPING = IMPORT_STATUS_NEEDS_MAPPING

LEGACY_STATUS_TO_CURRENT: dict[str, str] = {
    "available": IMPORT_STATUS_READY,
    "pending_field_mapping": IMPORT_STATUS_NEEDS_MAPPING,
    "import_failed": IMPORT_STATUS_FAILED,
}


def _import_roots() -> tuple[Path, ...]:
    return (IMPORT_ROOT,)


def _resolve_import_dir(dataset_id: str) -> Path:
    for root in _import_roots():
        candidate = root / dataset_id
        if candidate.is_dir():
            return candidate
    return IMPORT_ROOT / dataset_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_dataset_import_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"ds_import_{stamp}_{suffix}"


def make_import_source_job_id(dataset_id: str) -> str:
    return f"import_{dataset_id}"


def _normalize_key(path: str) -> str:
    return path.lower().replace("\\", "/")


def _matches_field_category(path: str, hints: tuple[str, ...]) -> bool:
    normalized = _normalize_key(path)
    segments = re.split(r"[/.]", normalized)
    for hint in hints:
        for segment in segments:
            if segment == hint or hint in segment:
                return True
    return False


def _dtype_name(dtype: Any) -> str:
    try:
        return str(dtype)
    except Exception:
        return "unknown"


def _shape_list(shape: Any) -> list[int]:
    if shape is None:
        return []
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return []


def _walk_hdf5_tree(handle: Any, prefix: str = "") -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in handle.keys():
        node = handle[key]
        path = f"{prefix}/{key}" if prefix else str(key)
        if hasattr(node, "keys"):
            entries.extend(_walk_hdf5_tree(node, path))
        else:
            entries.append(
                {
                    "path": path,
                    "shape": _shape_list(getattr(node, "shape", None)),
                    "dtype": _dtype_name(getattr(node, "dtype", None)),
                }
            )
    return entries


def _count_episodes(handle: Any, tree: list[dict[str, Any]]) -> int:
    data_group = handle.get("data")
    if data_group is not None and hasattr(data_group, "keys"):
        demo_keys = [k for k in data_group.keys() if str(k).startswith("demo_")]
        if demo_keys:
            return len(demo_keys)
        if len(list(data_group.keys())) > 0:
            return len(list(data_group.keys()))

    episodes_group = handle.get("episodes")
    if episodes_group is not None and hasattr(episodes_group, "keys"):
        ep_keys = list(episodes_group.keys())
        if ep_keys:
            return len(ep_keys)

    demo_ids = {
        match.group(1)
        for entry in tree
        if (match := re.search(r"(demo_\d+)", entry["path"]))
    }
    if demo_ids:
        return len(demo_ids)
    return 0


def _detect_recognized_fields(tree: list[dict[str, Any]]) -> dict[str, list[str]]:
    recognized: dict[str, list[str]] = {key: [] for key in FIELD_PATTERNS}
    for entry in tree:
        path = entry["path"]
        for category, hints in FIELD_PATTERNS.items():
            if _matches_field_category(path, hints):
                recognized[category].append(path)
    return recognized


def _normalize_import_status(raw: str) -> str:
    key = (raw or "").strip()
    return LEGACY_STATUS_TO_CURRENT.get(key, key or IMPORT_STATUS_READY)


def _first_time_dim(shape: Any) -> Optional[int]:
    if shape is None or len(shape) < 1:
        return None
    try:
        value = int(shape[0])
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _check_time_dimension_consistency(handle: Any, recognized: dict[str, list[str]]) -> tuple[bool, Optional[str]]:
    data_group = handle.get("data")
    if data_group is None:
        return False, "缺少 data 分组，无法校验 action/observation 时间维"

    demo_keys = sorted(k for k in data_group.keys() if str(k).startswith("demo_"))
    if demo_keys:
        demo = data_group[demo_keys[0]]
    else:
        keys = list(data_group.keys())
        if not keys:
            return False, "data 分组内无 demo 轨迹"
        demo = data_group[keys[0]]

    action_node = demo.get("actions")
    if action_node is None:
        action_node = demo.get("action")
    if action_node is None:
        return False, "缺少 actions 时间序列"

    action_t = _first_time_dim(getattr(action_node, "shape", None))
    if action_t is None:
        return False, "actions 时间维无效"

    obs_lengths: list[int] = []
    obs_group = demo.get("obs")
    if obs_group is not None:
        for key in obs_group.keys():
            key_str = str(key).lower()
            is_state = any(hint in key_str for hint in ("qpos", "state"))
            is_image = any(hint in key_str for hint in ("image", "rgb", "camera", "cam"))
            if not is_state and not is_image:
                continue
            ds = obs_group[key]
            obs_t = _first_time_dim(getattr(ds, "shape", None))
            if obs_t is not None:
                obs_lengths.append(obs_t)

    if not obs_lengths:
        return False, "无法读取 observation 时间维"

    for obs_t in obs_lengths:
        if obs_t != action_t:
            return False, f"action 时间维({action_t})与 observation 时间维({obs_t})不一致"
    return True, None


def _has_observation_fields(recognized: dict[str, list[str]]) -> bool:
    return bool(recognized.get("qpos")) or bool(recognized.get("image"))


def _assess_import_dataset(
    *,
    parse_error: Optional[str],
    recognized: dict[str, list[str]],
    episode_count: int,
    time_consistent: bool,
    time_dim_error: Optional[str],
) -> dict[str, Any]:
    if parse_error:
        return {
            "status": IMPORT_STATUS_FAILED,
            "trainable": False,
            "directTrainable": False,
            "needsBuild": False,
            "needsMapping": False,
            "episodeParsed": False,
            "errors": [parse_error],
        }

    errors: list[str] = []
    if episode_count <= 0:
        return {
            "status": IMPORT_STATUS_NEEDS_BUILD,
            "trainable": False,
            "directTrainable": False,
            "needsBuild": True,
            "needsMapping": False,
            "episodeParsed": False,
            "errors": ["未检测到 episode / demo 轨迹"],
        }

    if not recognized.get("action"):
        errors.append("未自动识别 action / actions 字段")
    if not _has_observation_fields(recognized):
        errors.append("未自动识别 qpos/state 或 image / rgb / camera 观测字段")
    if not time_consistent and time_dim_error:
        errors.append(time_dim_error)

    if errors:
        return {
            "status": IMPORT_STATUS_NEEDS_MAPPING,
            "trainable": False,
            "directTrainable": False,
            "needsBuild": False,
            "needsMapping": True,
            "episodeParsed": True,
            "errors": errors,
        }

    return {
        "status": IMPORT_STATUS_READY,
        "trainable": True,
        "directTrainable": True,
        "needsBuild": False,
        "needsMapping": False,
        "episodeParsed": True,
        "errors": [],
    }


def _build_data_scale_label(file_size: int, episode_count: int, assessment: dict[str, Any]) -> str:
    size_label = _format_file_size_label(file_size)
    if episode_count > 0:
        return f"{size_label} / {episode_count} episodes"
    status = str(assessment.get("status") or "")
    if status == IMPORT_STATUS_NEEDS_BUILD or assessment.get("needsBuild"):
        return f"{size_label} / 待构建"
    return f"{size_label} / episode 未解析"


def _format_file_size_label(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    unit_idx = 0
    while value >= 1024 and unit_idx < len(units) - 1:
        value /= 1024
        unit_idx += 1
    if unit_idx >= 3:
        return f"{value:.2f}{units[unit_idx]}"
    return f"{value:.2f} {units[unit_idx]}"


def _parse_hdf5_file(hdf5_path: Path) -> dict[str, Any]:
    parse_error: Optional[str] = None
    tree: list[dict[str, Any]] = []
    episode_count = 0
    recognized: dict[str, list[str]] = {key: [] for key in FIELD_PATTERNS}
    time_consistent = False
    time_dim_error: Optional[str] = None
    inspection = inspect_hdf5(hdf5_path)

    try:
        import h5py
    except ImportError as exc:
        parse_error = f"h5py 不可用: {exc}"
        return {
            "parseError": parse_error,
            "tree": tree,
            "episodeCount": 0,
            "recognizedFields": recognized,
            "inspection": inspection,
            "timeConsistent": False,
            "timeDimError": parse_error,
        }

    try:
        with h5py.File(hdf5_path, "r") as handle:
            tree = _walk_hdf5_tree(handle)
            episode_count = _count_episodes(handle, tree)
            if episode_count <= 0:
                episode_count = inspection.episode_count
            recognized = _detect_recognized_fields(tree)
            if episode_count > 0 and recognized.get("action") and _has_observation_fields(recognized):
                time_consistent, time_dim_error = _check_time_dimension_consistency(handle, recognized)
            elif episode_count > 0:
                time_consistent = False
                time_dim_error = "缺少 action 或 observation 字段，无法校验时间维"
    except OSError as exc:
        parse_error = f"HDF5 解析失败: {exc}"
        logger.warning("import hdf5 parse failed path=%s err=%s", hdf5_path, exc)

    return {
        "parseError": parse_error,
        "tree": tree,
        "episodeCount": episode_count,
        "recognizedFields": recognized,
        "inspection": inspection,
        "timeConsistent": time_consistent,
        "timeDimError": time_dim_error,
    }


def _build_schema_json(
    recognized: dict[str, list[str]],
    inspection: Any,
    tree: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "fields": tree,
        "recognizedFields": recognized,
        "observationKeys": list(getattr(inspection, "observation_keys", []) or []),
        "cameraKeys": list(getattr(inspection, "camera_keys", []) or []),
        "stateKeys": list(getattr(inspection, "state_keys", []) or []),
        "actionDim": getattr(inspection, "action_dim", None),
        "stateDim": getattr(inspection, "state_dim", None),
        "jointActionAvailable": bool(getattr(inspection, "joint_action_available", False)),
        "imageShape": getattr(inspection, "image_shape", None),
    }


def _build_validation_report(
    *,
    assessment: dict[str, Any],
    recognized: dict[str, list[str]],
    inspection_warnings: list[str],
    episode_count: int,
    time_consistent: bool,
) -> dict[str, Any]:
    errors: list[str] = list(assessment.get("errors") or [])
    warnings: list[str] = list(inspection_warnings)
    status = str(assessment.get("status") or IMPORT_STATUS_FAILED)
    trainable = bool(assessment.get("trainable"))
    issues = errors + warnings

    return {
        "status": status,
        "passed": trainable,
        "trainable": trainable,
        "directTrainable": bool(assessment.get("directTrainable")),
        "needsBuild": bool(assessment.get("needsBuild")),
        "needsMapping": bool(assessment.get("needsMapping")),
        "episodeParsed": bool(assessment.get("episodeParsed")),
        "timeConsistent": time_consistent,
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
        "episodeCount": episode_count,
        "recognizedFields": recognized,
        "generatedAt": _utc_now_iso(),
    }


def _resolve_data_source_label(data_source: str) -> str:
    return DATA_SOURCE_LABELS.get(data_source, data_source or "其他")


def _sanitize_upload_basename(filename: str) -> str:
    """Strip path segments from client-provided filename; never used for on-disk path."""
    name = (filename or "").strip().replace("\\", "/")
    return Path(name).name or "upload.hdf5"


def _dataset_id_referenced_in_payload(payload: Any, dataset_id: str) -> bool:
    if not isinstance(payload, dict):
        return False
    target = dataset_id.strip()
    for key in ("datasetId", "sourceDatasetId", "dataset_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() == target:
            return True
    dataset_ids = payload.get("datasetIds")
    if isinstance(dataset_ids, list) and target in {str(item).strip() for item in dataset_ids}:
        return True
    train_config = payload.get("trainConfig") or payload.get("train_config")
    if isinstance(train_config, dict) and _dataset_id_referenced_in_payload(train_config, target):
        return True
    manifest = payload.get("datasetManifest") or payload.get("dataset_manifest")
    if isinstance(manifest, dict) and _dataset_id_referenced_in_payload(manifest, target):
        return True
    return False


def is_dataset_referenced_by_training(dataset_id: str) -> bool:
    target = (dataset_id or "").strip()
    roots = (TRAINING_JOBS_ROOT,)
    if not target or not any(root.is_dir() for root in roots):
        return False
    for root in roots:
        if not root.is_dir():
            continue
        for job_dir in root.iterdir():
            if not job_dir.is_dir():
                continue
            for rel in ("status.json", "request.json", "artifacts/dataset_manifest.json"):
                path = job_dir / rel
                if not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if _dataset_id_referenced_in_payload(payload, target):
                    return True
    return False


def _resolve_task_type(task_type: str) -> str:
    return TASK_TYPE_MAP.get(task_type, task_type or "custom")


async def import_hdf5_dataset_upload(
    *,
    name: str,
    data_source: str,
    task_type: str,
    robot_type: str,
    file: UploadFile,
) -> dict[str, Any]:
    display_name = (name or "").strip()
    if not display_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请填写数据集名称")

    original_name = _sanitize_upload_basename(file.filename or "")
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持 .hdf5 / .h5 文件",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件超过大小限制")

    dataset_id = make_dataset_import_id()
    source_job_id = make_import_source_job_id(dataset_id)
    import_dir = IMPORT_ROOT / dataset_id
    hdf5_path = import_dir / "source.hdf5"
    import_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path.write_bytes(content)

    file_size = hdf5_path.stat().st_size
    created_at = _utc_now_iso()

    parsed = _parse_hdf5_file(hdf5_path)
    inspection = parsed["inspection"]
    recognized = parsed["recognizedFields"]
    episode_count = int(parsed["episodeCount"] or 0)
    parse_error = parsed["parseError"]
    inspection_warnings = list(getattr(inspection, "warnings", []) or [])
    time_consistent = bool(parsed.get("timeConsistent"))
    time_dim_error = parsed.get("timeDimError")

    assessment = _assess_import_dataset(
        parse_error=parse_error,
        recognized=recognized,
        episode_count=episode_count,
        time_consistent=time_consistent,
        time_dim_error=str(time_dim_error) if time_dim_error else None,
    )
    import_status = str(assessment["status"])

    resolved_task_type = _resolve_task_type(task_type)
    data_source_label = _resolve_data_source_label(data_source)
    schema_json = _build_schema_json(recognized, inspection, parsed["tree"])
    validation_report = _build_validation_report(
        assessment=assessment,
        recognized=recognized,
        inspection_warnings=inspection_warnings,
        episode_count=episode_count,
        time_consistent=time_consistent,
    )

    trainable = bool(assessment.get("trainable"))
    direct_trainable = bool(assessment.get("directTrainable"))
    needs_build = bool(assessment.get("needsBuild"))
    needs_mapping = bool(assessment.get("needsMapping"))
    episode_parsed = bool(assessment.get("episodeParsed"))
    data_scale_label = _build_data_scale_label(file_size, episode_count, assessment)

    metadata = {
        "datasetId": dataset_id,
        "id": dataset_id,
        "name": display_name,
        "displayName": display_name,
        "sourceJobId": source_job_id,
        "sourceType": "real_robot_imported" if data_source == "real_collection" else "mixed",
        "dataSource": data_source,
        "dataSourceLabel": data_source_label,
        "taskType": resolved_task_type,
        "taskDisplayName": TASK_DISPLAY_MAP.get(resolved_task_type, task_display_name(resolved_task_type)),
        "taskTemplateId": TASK_TEMPLATE_MAP.get(resolved_task_type, "custom_import"),
        "robotType": robot_type,
        "format": "hdf5",
        "datasetFormat": "hdf5",
        "status": import_status,
        "episodeCount": episode_count if episode_parsed else 0,
        "episodeParsed": episode_parsed,
        "successfulEpisodes": episode_count if direct_trainable else 0,
        "totalEpisodes": episode_count if episode_parsed else 0,
        "validTrajectories": episode_count if episode_parsed else None,
        "generationRounds": episode_count if episode_parsed else None,
        "fileSizeBytes": file_size,
        "dataScaleLabel": data_scale_label,
        "storagePath": str(import_dir),
        "manifestPath": str(import_dir / "metadata.json"),
        "schemaPath": str(import_dir / "schema.json"),
        "validationReportPath": str(import_dir / "validation_report.json"),
        "datasetFile": str(hdf5_path),
        "builtDatasetPath": str(hdf5_path),
        "originalFileName": original_name,
        "trainable": trainable,
        "directTrainable": direct_trainable,
        "needsBuild": needs_build,
        "needsMapping": needs_mapping,
        "trainingBackends": ["torch_bc", "diffusion_policy"] if direct_trainable else [],
        "replayAvailable": False,
        "createdAt": created_at,
        "updatedAt": created_at,
        "importKind": "hdf5_upload",
    }

    if direct_trainable:
        metadata["observationSchema"] = getattr(inspection, "observation_schema_id", None) or "imported_hdf5"
        metadata["actionSchema"] = getattr(inspection, "action_schema_id", None) or "imported_hdf5"
        if getattr(inspection, "joint_action_available", False):
            metadata["jointActionAvailable"] = True
            metadata["trainedActionMode"] = getattr(inspection, "trained_action_mode", None) or "joint_delta"

    _write_json(import_dir / "metadata.json", metadata)
    _write_json(import_dir / "schema.json", schema_json)
    _write_json(import_dir / "validation_report.json", validation_report)

    if import_status == IMPORT_STATUS_FAILED:
        logger.warning(
            "dataset import failed dataset_id=%s name=%s issues=%s",
            dataset_id,
            display_name,
            validation_report.get("errors"),
        )

    row = dict(metadata)
    apply_dataset_row_display_fields(
        row,
        task_type=resolved_task_type,
        manifest=metadata,
        source_job_id=source_job_id,
    )
    from app.services.workspace_dataset_stats_service import enrich_dataset_list_stats

    enrich_dataset_list_stats(row, persist_size=True)
    return {
        "dataset": row,
        "datasetId": dataset_id,
        "status": import_status,
        "validationReport": validation_report,
    }


def _read_import_metadata(import_dir: Path) -> dict[str, Any]:
    metadata_path = import_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def import_record_to_dataset_row(import_dir: Path) -> Optional[dict[str, Any]]:
    metadata = _read_import_metadata(import_dir)
    if not metadata:
        return None

    dataset_id = str(metadata.get("datasetId") or metadata.get("id") or import_dir.name)
    hdf5_path = import_dir / "source.hdf5"
    if not hdf5_path.is_file():
        hdf5_alt = metadata.get("datasetFile")
        if hdf5_alt and Path(str(hdf5_alt)).is_file():
            hdf5_path = Path(str(hdf5_alt))
        else:
            return None

    source_job_id = str(metadata.get("sourceJobId") or make_import_source_job_id(dataset_id))
    episode_count = int(metadata.get("episodeCount") or 0)
    file_size = int(metadata.get("fileSizeBytes") or hdf5_path.stat().st_size)
    import_status = _normalize_import_status(str(metadata.get("status") or IMPORT_STATUS_READY))
    task_type = str(metadata.get("taskType") or "custom")
    created_at = str(metadata.get("createdAt") or _utc_now_iso())
    direct_trainable = bool(metadata.get("directTrainable"))
    trainable = bool(metadata.get("trainable")) and direct_trainable
    episode_parsed = bool(metadata.get("episodeParsed", episode_count > 0))

    row: dict[str, Any] = {
        "id": dataset_id,
        "name": str(metadata.get("name") or metadata.get("displayName") or dataset_id),
        "displayName": str(metadata.get("displayName") or metadata.get("name") or dataset_id),
        "sourceJobId": source_job_id,
        "sourceType": str(metadata.get("sourceType") or "real_robot_imported"),
        "dataSourceLabel": str(metadata.get("dataSourceLabel") or _resolve_data_source_label(str(metadata.get("dataSource") or ""))),
        "taskType": task_type,
        "taskDisplayName": str(metadata.get("taskDisplayName") or TASK_DISPLAY_MAP.get(task_type, task_display_name(task_type))),
        "taskTemplateId": str(metadata.get("taskTemplateId") or TASK_TEMPLATE_MAP.get(task_type, "custom_import")),
        "robotType": metadata.get("robotType"),
        "manifestPath": str(import_dir / "metadata.json"),
        "episodeCount": episode_count,
        "validTrajectories": metadata.get("validTrajectories"),
        "generationRounds": metadata.get("generationRounds"),
        "successfulEpisodes": metadata.get("successfulEpisodes"),
        "totalEpisodes": metadata.get("totalEpisodes"),
        "storagePath": str(import_dir),
        "format": "hdf5",
        "status": import_status,
        "replayAvailable": False,
        "datasetFile": str(hdf5_path),
        "builtDatasetPath": str(hdf5_path),
        "fileSizeBytes": file_size,
        "dataScaleLabel": str(
            metadata.get("dataScaleLabel")
            or _build_data_scale_label(
                file_size,
                episode_count,
                {
                    "status": import_status,
                    "needsBuild": metadata.get("needsBuild"),
                    "episodeParsed": episode_parsed,
                },
            )
        ),
        "trainable": trainable,
        "directTrainable": direct_trainable,
        "needsBuild": bool(metadata.get("needsBuild")),
        "needsMapping": bool(metadata.get("needsMapping")),
        "episodeParsed": episode_parsed,
        "trainingBackends": metadata.get("trainingBackends"),
        "observationSchema": metadata.get("observationSchema"),
        "actionSchema": metadata.get("actionSchema"),
        "jointActionAvailable": metadata.get("jointActionAvailable"),
        "trainedActionMode": metadata.get("trainedActionMode"),
        "createdAt": created_at,
        "updatedAt": str(metadata.get("updatedAt") or created_at),
        "simulatorBackend": metadata.get("simulatorBackend") or "imported",
    }
    apply_dataset_row_display_fields(
        row,
        task_type=task_type,
        manifest=metadata,
        source_job_id=source_job_id,
    )
    from app.services.workspace_dataset_stats_service import enrich_dataset_list_stats

    enrich_dataset_list_stats(row, persist_size=True)
    return row


def list_imported_datasets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in _import_roots():
        if not root.is_dir():
            continue
        for import_dir in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
            if not import_dir.is_dir() or import_dir.name in seen:
                continue
            row = import_record_to_dataset_row(import_dir)
            if row:
                seen.add(import_dir.name)
                rows.append(row)
    return rows


def delete_imported_dataset(dataset_id: str) -> dict[str, Any]:
    candidate = (dataset_id or "").strip()
    if not candidate.startswith("ds_import_"):
        raise ValueError(f"invalid imported dataset id: {candidate}")

    if is_dataset_referenced_by_training(candidate):
        raise RuntimeError("该数据集已被训练任务引用，暂不可删除")

    import_dir = _resolve_import_dir(candidate)
    if not import_dir.is_dir():
        raise FileNotFoundError(f"import dataset not found: {candidate}")

    shutil.rmtree(import_dir)
    return {"ok": True, "datasetId": candidate, "deleted": True}
