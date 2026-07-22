"""Workspace 数据中心列表 — dataCount / fileSizeBytes 统一补齐。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

LEGACY_IMPORT_STATUS_MAP = {
    "available": "ready",
    "pending_field_mapping": "needs_mapping",
    "import_failed": "failed",
}


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _normalize_import_status(raw: Any) -> str:
    key = str(raw or "").strip().lower()
    return LEGACY_IMPORT_STATUS_MAP.get(key, key or "ready")


def _is_imported_row(row: dict[str, Any]) -> bool:
    dataset_id = str(row.get("id") or "").strip()
    source_job_id = str(row.get("sourceJobId") or "").strip()
    return dataset_id.startswith("ds_import_") or source_job_id.startswith("import_ds_import_")


def _normalized_format(row: dict[str, Any]) -> str:
    return str(row.get("format") or row.get("datasetFormat") or "").strip().lower()


def _is_manifest_only_row(row: dict[str, Any]) -> bool:
    fmt = _normalized_format(row)
    if fmt in {"manifest", "episode_manifest"}:
        return True
    dataset_format = str(row.get("datasetFormat") or "").strip().lower()
    return dataset_format in {"manifest", "episode_manifest", "manifest-only"}


def _path_candidate(raw: Any) -> Optional[Path]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return Path(text).expanduser()
    except (TypeError, ValueError):
        return None


def _existing_path(raw: Any) -> Optional[Path]:
    path = _path_candidate(raw)
    if path is None or not path.exists():
        return None
    return path


def _file_size_bytes(path: Path) -> Optional[int]:
    try:
        if path.is_file():
            size = path.stat().st_size
            return size if size > 0 else None
        if path.is_dir():
            return _directory_size_bytes(path)
    except OSError as exc:
        logger.debug("dataset size stat failed for %s: %s", path, exc)
    return None


def _directory_size_bytes(path: Path) -> Optional[int]:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError as exc:
        logger.debug("dataset directory size failed for %s: %s", path, exc)
        return None
    return total if total > 0 else None


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _job_root_from_row(row: dict[str, Any]) -> Optional[Path]:
    manifest_path = _path_candidate(row.get("manifestPath"))
    if manifest_path is not None:
        if manifest_path.is_file():
            parent = manifest_path.parent
            if manifest_path.name == "dataset.manifest.json" and parent.name == "datasets":
                return parent.parent
            if manifest_path.name == "dataset_manifest.json":
                return parent
            if manifest_path.name == "metadata.json":
                return parent
            if manifest_path.name in {"episode_manifest.json", "episode_result.json"}:
                if parent.name in {"results", "episode"}:
                    return parent.parent
                return parent
        elif manifest_path.is_dir():
            return manifest_path

    storage_path = _existing_path(row.get("storagePath"))
    if storage_path is not None:
        if storage_path.name == "datasets":
            return storage_path.parent
        return storage_path
    return None


def _resolve_relative_data_path(job_root: Optional[Path], raw: Any) -> Optional[Path]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_file() or path.is_dir():
        return path
    if job_root is not None:
        candidate = (job_root / text).resolve()
        if candidate.exists():
            return candidate
    return None


def _load_row_manifest(row: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _existing_path(row.get("manifestPath"))
    if manifest_path is None or not manifest_path.is_file():
        return {}
    if manifest_path.name == "registry.json":
        return {}
    return _read_json_dict(manifest_path)


def _imported_hdf5_file_exists(row: dict[str, Any]) -> bool:
    for key in ("datasetFile", "builtDatasetPath"):
        path = _existing_path(row.get(key))
        if path is not None and path.is_file():
            return True
    storage_path = _existing_path(row.get("storagePath"))
    if storage_path is not None and storage_path.is_dir():
        for name in ("source.hdf5", "dataset.hdf5"):
            if (storage_path / name).is_file():
                return True
    return False


def _is_built_row(row: dict[str, Any]) -> bool:
    dataset_id = str(row.get("id") or "").strip()
    source_job_id = str(row.get("sourceJobId") or "").strip()
    return dataset_id.startswith("ds_built_") or source_job_id.startswith("built_ds_built_")


def resolve_dataset_data_count(row: dict[str, Any]) -> Optional[int]:
    if _is_built_row(row):
        value = _positive_int(row.get("dataCount"))
        if value is not None and value > 0:
            return value
        episode_count = _positive_int(row.get("episodeCount"))
        return episode_count if episode_count and episode_count > 0 else None

    if _is_imported_row(row):
        status = _normalize_import_status(row.get("status"))
        if status in {"failed", "parsing"}:
            return None

        episode_parsed = row.get("episodeParsed")
        episode_count = _positive_int(row.get("episodeCount"))
        if episode_parsed is not False and episode_count is not None and episode_count > 0:
            return episode_count

        # 列表展示：已成功导入的 HDF5 文件在 episode 未解析时记为 1 份数据集，不改 episodeCount。
        if _imported_hdf5_file_exists(row) and status in {"needs_build", "needs_mapping", "ready"}:
            return 1
        return None

    if _is_manifest_only_row(row):
        return None

    for key in ("totalEpisodes", "generationRounds", "episodeCount"):
        value = _positive_int(row.get(key))
        if value is not None and value > 0:
            return value
    return None


def resolve_dataset_size_bytes(row: dict[str, Any]) -> Optional[int]:
    existing = _positive_int(row.get("fileSizeBytes"))
    if existing is not None and existing > 0:
        return existing

    fmt = _normalized_format(row)
    job_root = _job_root_from_row(row)
    manifest = _load_row_manifest(row)

    if fmt == "lerobot":
        for key in ("lerobotPath", "datasetFile", "builtDatasetPath"):
            path = _existing_path(row.get(key))
            if path is not None:
                size = _file_size_bytes(path)
                if size is not None:
                    return size
        lerobot_block = manifest.get("lerobot")
        if isinstance(lerobot_block, dict):
            rel = lerobot_block.get("path")
            path = _resolve_relative_data_path(job_root, rel)
            if path is not None:
                size = _file_size_bytes(path)
                if size is not None:
                    return size

    for key in ("datasetFile", "builtDatasetPath"):
        path = _existing_path(row.get(key))
        if path is not None and path.is_file():
            size = _file_size_bytes(path)
            if size is not None:
                return size

    storage_path = _existing_path(row.get("storagePath"))
    if storage_path is not None:
        candidates: list[Path] = []
        if storage_path.is_dir():
            candidates.extend(
                [
                    storage_path / "dataset.hdf5",
                    storage_path / "source.hdf5",
                    storage_path / "datasets" / "dataset.hdf5",
                ]
            )
        for candidate in candidates:
            if candidate.is_file():
                size = _file_size_bytes(candidate)
                if size is not None:
                    return size

    if manifest:
        artifacts = manifest.get("artifacts")
        if isinstance(artifacts, dict):
            for artifact_key in ("hdf5", "dataset", "datasetFile"):
                path = _resolve_relative_data_path(job_root, artifacts.get(artifact_key))
                if path is not None and path.is_file():
                    size = _file_size_bytes(path)
                    if size is not None:
                        return size

        for manifest_key in ("dataset_hdf5_path", "datasetFile", "dataset"):
            path = _resolve_relative_data_path(job_root, manifest.get(manifest_key))
            if path is not None:
                size = _file_size_bytes(path)
                if size is not None:
                    return size

    if job_root is not None:
        for relative in (
            "datasets/dataset.hdf5",
            "source.hdf5",
            "dataset.hdf5",
        ):
            candidate = job_root / relative
            if candidate.is_file():
                size = _file_size_bytes(candidate)
                if size is not None:
                    return size

        episodes_dir = job_root / "episodes"
        if episodes_dir.is_dir() and _is_manifest_only_row(row):
            size = _directory_size_bytes(episodes_dir)
            if size is not None:
                return size

    if _is_manifest_only_row(row):
        return None

    return None


def _maybe_persist_file_size_bytes(row: dict[str, Any], size_bytes: int) -> None:
    if size_bytes <= 0:
        return
    manifest_path = _path_candidate(row.get("manifestPath"))
    if manifest_path is None or not manifest_path.is_file():
        return
    if manifest_path.name == "registry.json":
        return
    try:
        manifest = _read_json_dict(manifest_path)
        if _positive_int(manifest.get("fileSizeBytes")) == size_bytes:
            return
        manifest["fileSizeBytes"] = size_bytes
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("persist fileSizeBytes skipped for %s: %s", manifest_path, exc)


def enrich_dataset_count_stats(row: dict[str, Any]) -> None:
    row["dataCount"] = resolve_dataset_data_count(row)


def enrich_dataset_storage_stats(row: dict[str, Any], *, persist: bool = True) -> None:
    size_bytes = resolve_dataset_size_bytes(row)
    row["fileSizeBytes"] = size_bytes if size_bytes is not None else 0
    if persist and size_bytes is not None and size_bytes > 0:
        existing = _positive_int(row.get("fileSizeBytes"))
        if existing == size_bytes:
            _maybe_persist_file_size_bytes(row, size_bytes)


def enrich_dataset_list_stats(row: dict[str, Any], *, persist_size: bool = True) -> None:
    enrich_dataset_count_stats(row)
    enrich_dataset_storage_stats(row, persist=persist_size)
