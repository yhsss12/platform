"""Isaac Lab HDF5 demo 数据集注册表（imported_demo）。"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.platform_paths import is_path_within, platform_paths
from app.services.isaac_lab.job_paths import isaac_job_metadata_dir, isaac_job_preview_video_path
from app.services.isaac_lab.replay_service import resolve_dataset_file
from app.services.dataset_naming import (
    is_canonical_dataset_display_name,
    normalize_dataset_display_name,
    resolve_unique_dataset_display_name,
    task_display_name,
)
from integrations.isaac_lab.hdf5_image_obs import build_observation_manifest_fields

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
DEFAULT_ISAAC_DATASET_REGISTRY_PATH = (
    platform_paths.assets_root / "datasets" / "isaac_lab" / "registry.json"
)
ISAAC_DATASET_REGISTRY_PATH = DEFAULT_ISAAC_DATASET_REGISTRY_PATH

ISAAC_BLOCK_STACKING_TEMPLATE_ID = "isaac_block_stacking"
ISAAC_BLOCK_STACKING_REGISTRY_ID = "task_isaac_block_stacking_v1"
DEFAULT_REPLAY_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
DEFAULT_TRAIN_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
ISAAC_STACK_DATASET_ENV = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"
ISAAC_STACK_OBS_KEYS = ["eef_pos", "eef_quat", "gripper_pos", "object"]
ISAAC_STACK_ACTION_DIM = 7
ISAAC_STACK_TRAINING_BACKEND = "isaac_robomimic_bc"


def _resolve_generation_mode_from_manifest(
    gen_manifest_data: dict[str, Any],
    *,
    source_job_id: str = "",
) -> Optional[str]:
    """Resolve generationMode from manifest top-level, metrics, or job request.json."""
    mode = gen_manifest_data.get("generationMode")
    if mode:
        return str(mode)
    metrics = gen_manifest_data.get("metrics")
    if isinstance(metrics, dict) and metrics.get("generationMode"):
        return str(metrics["generationMode"])
    if source_job_id.startswith("isaac_gen_"):
        request_path = isaac_job_metadata_dir(source_job_id) / "request.json"
        if request_path.is_file():
            try:
                req = json.loads(request_path.read_text(encoding="utf-8"))
                if req.get("generationMode"):
                    return str(req["generationMode"])
            except (OSError, json.JSONDecodeError):
                pass
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_registry_file() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    registry_paths = [ISAAC_DATASET_REGISTRY_PATH]
    for registry_path in dict.fromkeys(registry_paths):
        if not registry_path.is_file():
            continue
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            candidates = (
                data
                if isinstance(data, list)
                else data.get("datasets", [])
                if isinstance(data, dict)
                else []
            )
            for row in candidates:
                if not isinstance(row, dict):
                    continue
                dataset_id = str(row.get("id") or "")
                if dataset_id and dataset_id in seen:
                    continue
                if dataset_id:
                    seen.add(dataset_id)
                rows.append(row)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("isaac dataset registry read failed (%s): %s", registry_path, exc)
    return rows


def _write_registry_file(rows: list[dict[str, Any]]) -> None:
    ISAAC_DATASET_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "updatedAt": _utc_now_iso(), "datasets": rows}
    ISAAC_DATASET_REGISTRY_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_isaac_dataset_id() -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"isaac_ds_{ts}_{suffix}"


def make_isaac_import_source_job_id(dataset_id: str) -> str:
    return f"isaac_import_{dataset_id}"


def _count_hdf5_episodes(dataset_path: Path) -> int:
    try:
        import h5py
    except ImportError:
        return 0
    try:
        with h5py.File(dataset_path, "r") as handle:
            if "data" in handle and hasattr(handle["data"], "keys"):
                return len(list(handle["data"].keys()))
    except Exception as exc:
        logger.debug("hdf5 episode count skipped for %s: %s", dataset_path, exc)
    return 0


def _record_to_dataset_row(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    dataset_file = Path(str(record.get("datasetFile") or "")).expanduser()
    if not dataset_file.is_file():
        return None

    dataset_id = str(record.get("id") or "")
    if not dataset_id:
        return None

    source_type = str(record.get("sourceType") or "imported_demo")
    source_job_id = str(record.get("sourceJobId") or make_isaac_import_source_job_id(dataset_id))
    created_at = str(record.get("createdAt") or _utc_now_iso())
    updated_at = str(record.get("updatedAt") or created_at)
    manifest_path = str(
        record.get("manifestPath")
        or record.get("generationManifestPath")
        or ISAAC_DATASET_REGISTRY_PATH
    )
    replay_available = bool(record.get("replayAvailable", source_type == "imported_demo"))

    preview_video_path = record.get("previewVideoPath")
    if not preview_video_path and source_job_id.startswith("isaac_gen_"):
        preview = isaac_job_preview_video_path(source_job_id)
        if preview.is_file() and preview.stat().st_size > 0:
            preview_video_path = str(preview.resolve())
    preview_path_obj = Path(str(preview_video_path)).expanduser() if preview_video_path else None
    video_available = bool(preview_path_obj and preview_path_obj.is_file())

    quality_status = record.get("qualityStatus")
    quality_warnings = record.get("qualityWarnings")
    quality_display_tier = record.get("qualityDisplayTier")
    quality_display_label = record.get("qualityDisplayLabel")
    quality_display_hint = record.get("qualityDisplayHint")
    quality_display_severity = record.get("qualityDisplaySeverity")
    generation_mode = record.get("generationMode")
    preview_status = record.get("previewStatus")
    browser_preview_path = record.get("browserPreviewVideoPath")
    if source_job_id.startswith("isaac_gen_"):
        gen_manifest = Path(str(record.get("generationManifestPath") or "")).expanduser()
        if not gen_manifest.is_file():
            gen_manifest = dataset_file.parent.parent / "generation_manifest.json"
        if gen_manifest.is_file():
            try:
                gm = json.loads(gen_manifest.read_text(encoding="utf-8"))
                quality_status = quality_status or gm.get("qualityStatus")
                quality_warnings = quality_warnings or gm.get("qualityWarnings")
                quality_display_tier = quality_display_tier or gm.get("qualityDisplayTier")
                quality_display_label = quality_display_label or gm.get("qualityDisplayLabel")
                quality_display_hint = quality_display_hint or gm.get("qualityDisplayHint")
                quality_display_severity = quality_display_severity or gm.get("qualityDisplaySeverity")
                generation_mode = generation_mode or _resolve_generation_mode_from_manifest(
                    gm,
                    source_job_id=source_job_id,
                )
                preview_status = preview_status or gm.get("previewStatus")
                browser_preview_path = browser_preview_path or gm.get("browserPreviewVideoPath")
                if gm.get("previewVideoPath") and not preview_video_path:
                    preview_video_path = gm.get("previewVideoPath")
            except (OSError, json.JSONDecodeError):
                pass

    episode_count = int(record.get("episodeCount") or _count_hdf5_episodes(dataset_file))
    successful: Optional[int] = None
    for key in ("successfulEpisodes", "numDemos", "validTrajectories", "num_successful"):
        val = record.get(key)
        if isinstance(val, int) and val >= 0:
            successful = val
            break

    total: Optional[int] = None
    for key in ("totalEpisodes", "numEpisodes", "generationRounds", "episodeCount"):
        val = record.get(key)
        if isinstance(val, int) and val > 0:
            total = val
            break
    if total is None and episode_count > 0:
        total = episode_count
    if successful is None and episode_count > 0:
        successful = episode_count

    task_type = str(record.get("taskType") or "block_stacking")
    display_name = normalize_dataset_display_name(
        task_type=task_type,
        display_name=record.get("displayName"),
        name=record.get("name"),
        created_at=created_at,
        source_job_id=source_job_id,
    )

    row = {
        "id": dataset_id,
        "name": display_name,
        "displayName": display_name,
        "taskType": task_type,
        "taskDisplayName": task_display_name(task_type),
        "sourceJobId": source_job_id,
        "sourceTaskTemplateId": ISAAC_BLOCK_STACKING_REGISTRY_ID,
        "taskTemplateId": ISAAC_BLOCK_STACKING_TEMPLATE_ID,
        "sourceType": source_type,
        "simulatorBackend": "isaac_lab",
        "datasetFile": str(dataset_file.resolve()),
        "datasetFormat": "hdf5",
        "manifestPath": manifest_path,
        "episodeCount": episode_count,
        "validTrajectories": successful,
        "generationRounds": total,
        "successfulEpisodes": successful,
        "totalEpisodes": total,
        "storagePath": str(dataset_file.parent.resolve()),
        "format": "hdf5",
        "status": "available",
        "trainable": True,
        "trainingBackends": [ISAAC_STACK_TRAINING_BACKEND],
        "taskEnv": str(record.get("taskEnv") or DEFAULT_TRAIN_TASK_ID),
        "datasetEnv": str(record.get("datasetEnv") or ISAAC_STACK_DATASET_ENV),
        "obsKeys": list(record.get("obsKeys") or ISAAC_STACK_OBS_KEYS),
        "actionDim": int(record.get("actionDim") or ISAAC_STACK_ACTION_DIM),
        "replayAvailable": replay_available,
        "replayBackend": "isaac_lab",
        "taskId": str(record.get("taskId") or DEFAULT_REPLAY_TASK_ID),
        "previewVideoPath": str(preview_path_obj.resolve()) if video_available and preview_path_obj else None,
        "videoAvailable": video_available,
        "qualityStatus": quality_status,
        "qualityWarnings": list(quality_warnings) if isinstance(quality_warnings, list) else None,
        "qualityDisplayTier": quality_display_tier,
        "qualityDisplayLabel": quality_display_label,
        "qualityDisplayHint": quality_display_hint,
        "qualityDisplaySeverity": quality_display_severity,
        "generationMode": generation_mode,
        "previewStatus": preview_status,
        "browserPreviewVideoPath": browser_preview_path,
        "createdAt": created_at,
        "updatedAt": updated_at,
    }
    obs_meta = build_observation_manifest_fields(dataset_file)
    if obs_meta.get("obsKeys"):
        row["obsKeys"] = obs_meta["obsKeys"]
    if obs_meta.get("observationType"):
        row["observationType"] = obs_meta["observationType"]
    if obs_meta.get("cameraKeys"):
        row["cameraKeys"] = obs_meta["cameraKeys"]
        row["imageKeys"] = obs_meta.get("imageKeys") or obs_meta["cameraKeys"]
    if obs_meta.get("imageShape"):
        row["imageShape"] = obs_meta["imageShape"]
    if obs_meta.get("actionDim") is not None:
        row["actionDim"] = obs_meta["actionDim"]
    quality = dict(row.get("quality") or {})
    quality.update(obs_meta.get("quality") or {})
    row["quality"] = quality
    row["simulator"] = obs_meta.get("simulator") or "Isaac"
    row["robotType"] = obs_meta.get("robotType") or "Panda"
    row["observationSpace"] = obs_meta.get("observationSpace") or {
        "type": row.get("observationType") or "low_dim",
        "keys": row.get("obsKeys") or [],
    }
    return row


def list_isaac_datasets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _read_registry_file():
        built = _record_to_dataset_row(record)
        if built:
            rows.append(built)
    return rows


def get_isaac_dataset(dataset_id: str) -> dict[str, Any]:
    candidate = (dataset_id or "").strip()
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="datasetId is required",
        )
    for record in _read_registry_file():
        if str(record.get("id")) == candidate:
            built = _record_to_dataset_row(record)
            if built is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"dataset file missing for datasetId={candidate}",
                )
            return built
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Isaac dataset not found: {candidate}",
    )


def register_generated_dataset(
    *,
    job_id: str,
    dataset_name: str,
    dataset_file: Path,
    task_id: str = DEFAULT_REPLAY_TASK_ID,
    episode_count: int = 0,
    replay_available: bool = True,
) -> dict[str, Any]:
    if not dataset_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="generated dataset.hdf5 missing after successful job",
        )

    now = _utc_now_iso()
    dataset_id = make_isaac_dataset_id()
    generation_manifest = dataset_file.parent.parent / "generation_manifest.json"
    preview_path = isaac_job_preview_video_path(job_id)
    preview_video_path = (
        str(preview_path.resolve())
        if preview_path.is_file() and preview_path.stat().st_size > 0
        else None
    )
    canonical_name = dataset_name.strip()
    if not is_canonical_dataset_display_name(canonical_name):
        canonical_name = resolve_unique_dataset_display_name(
            task_type="block_stacking",
            created_at=now,
            source_job_id=job_id,
        )
    record = {
        "id": dataset_id,
        "name": canonical_name,
        "displayName": canonical_name,
        "taskDisplayName": task_display_name("block_stacking"),
        "taskType": "block_stacking",
        "sourceJobId": job_id,
        "taskTemplateId": ISAAC_BLOCK_STACKING_TEMPLATE_ID,
        "sourceTaskTemplateId": ISAAC_BLOCK_STACKING_REGISTRY_ID,
        "sourceType": "simulation_generated",
        "simulatorBackend": "isaac_lab",
        "physicsBackend": "physx",
        "datasetFormat": "hdf5",
        "datasetFile": str(dataset_file.resolve()),
        "taskId": task_id,
        "replayAvailable": replay_available,
        "replayBackend": "isaac_lab",
        "previewVideoPath": preview_video_path,
        "videoAvailable": preview_video_path is not None,
        "episodeCount": episode_count or _count_hdf5_episodes(dataset_file),
        "manifestPath": str(
            generation_manifest if generation_manifest.is_file() else ISAAC_DATASET_REGISTRY_PATH
        ),
        "generationManifestPath": str(generation_manifest),
        "generationMode": None,
        "createdAt": now,
        "updatedAt": now,
    }

    gen_manifest_data: dict[str, Any] = {}
    if generation_manifest.is_file():
        try:
            loaded = json.loads(generation_manifest.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                gen_manifest_data = loaded
        except (OSError, json.JSONDecodeError):
            gen_manifest_data = {}
    if gen_manifest_data:
        record["generationMode"] = _resolve_generation_mode_from_manifest(
            gen_manifest_data,
            source_job_id=job_id,
        )
        record["qualityStatus"] = gen_manifest_data.get("qualityStatus")
        record["qualityWarnings"] = gen_manifest_data.get("qualityWarnings")
        record["qualityDisplayTier"] = gen_manifest_data.get("qualityDisplayTier")
        record["qualityDisplayLabel"] = gen_manifest_data.get("qualityDisplayLabel")
        record["qualityDisplayHint"] = gen_manifest_data.get("qualityDisplayHint")
        for field in (
            "observationType",
            "cameraKeys",
            "imageKeys",
            "imageShape",
            "observationSpace",
            "obsKeys",
            "robotType",
            "simulator",
        ):
            if gen_manifest_data.get(field) is not None:
                record[field] = gen_manifest_data[field]
        quality = gen_manifest_data.get("quality")
        if isinstance(quality, dict):
            record["quality"] = quality

    rows = _read_registry_file()
    rows = [row for row in rows if str(row.get("sourceJobId")) != job_id]
    rows.append(record)
    _write_registry_file(rows)

    built = _record_to_dataset_row(record)
    if built is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to register generated Isaac dataset",
        )
    return built


def import_demo_hdf5(
    *,
    dataset_file: str,
    display_name: str,
    task_id: str = DEFAULT_REPLAY_TASK_ID,
) -> dict[str, Any]:
    name = (display_name or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="displayName is required",
        )

    resolved = resolve_dataset_file(dataset_file)
    if not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"dataset_file not found: {dataset_file}",
        )

    for existing in _read_registry_file():
        if str(existing.get("datasetFile")) == str(resolved):
            built = _record_to_dataset_row(existing)
            if built:
                return built
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="dataset file path exists in registry but file is unavailable",
            )

    now = _utc_now_iso()
    dataset_id = make_isaac_dataset_id()
    record = {
        "id": dataset_id,
        "name": name,
        "sourceJobId": make_isaac_import_source_job_id(dataset_id),
        "taskTemplateId": ISAAC_BLOCK_STACKING_TEMPLATE_ID,
        "sourceTaskTemplateId": ISAAC_BLOCK_STACKING_REGISTRY_ID,
        "sourceType": "imported_demo",
        "simulatorBackend": "isaac_lab",
        "datasetFormat": "hdf5",
        "datasetFile": str(resolved),
        "taskId": (task_id or DEFAULT_REPLAY_TASK_ID).strip() or DEFAULT_REPLAY_TASK_ID,
        "replayAvailable": True,
        "replayBackend": "isaac_lab",
        "episodeCount": _count_hdf5_episodes(resolved),
        "manifestPath": str(ISAAC_DATASET_REGISTRY_PATH),
        "createdAt": now,
        "updatedAt": now,
    }

    rows = _read_registry_file()
    rows.append(record)
    _write_registry_file(rows)

    built = _record_to_dataset_row(record)
    if built is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to register Isaac dataset",
        )
    return built


def delete_isaac_dataset(dataset_id: str) -> None:
    candidate = (dataset_id or "").strip()
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="datasetId is required",
        )

    rows = _read_registry_file()
    record = next((row for row in rows if str(row.get("id")) == candidate), None)
    kept = [row for row in rows if str(row.get("id")) != candidate]
    if len(kept) == len(rows):
        # Deletion is intentionally idempotent.
        return
    if isinstance(record, dict):
        source_job_id = str(record.get("sourceJobId") or "").strip()
        if source_job_id.startswith("isaac_gen_"):
            jobs_root = platform_paths.runs_root / "isaac_lab" / "jobs"
            job_root = jobs_root / source_job_id
            if job_root.is_dir() and is_path_within(job_root, jobs_root):
                shutil.rmtree(job_root)
    _write_registry_file(kept)
