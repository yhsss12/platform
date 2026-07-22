"""Backfill simulation HDF5 datasets into hdf5_datasets / data_assets tables."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import func

from app.core.platform_paths import platform_paths
from app.models.data_asset import DataAsset
from app.models.hdf5_dataset import HDF5Dataset
from app.services.asset_registration_service import DataAssetsSyncSessionLocal
from app.services.hdf5_platform_metadata import build_dataset_row_from_hdf5

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNTIME_ROOT = platform_paths.runs_root


def _read_manifest_for_hdf5(hdf5_path: Path) -> tuple[dict[str, Any], str | None]:
    datasets_dir = hdf5_path.parent
    job_dir = datasets_dir.parent
    for name in ("dataset.manifest.json", "manifest.json"):
        manifest_path = datasets_dir / name
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            manifest["manifestPath"] = str(manifest_path)
            return manifest, job_dir.name
    return {}, job_dir.name if job_dir.is_dir() else None


def discover_dataset_hdf5_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved in seen or not path.is_file():
            return
        seen.add(resolved)
        paths.append(path.resolve())

    runtime_roots = (RUNTIME_ROOT,)
    for runtime_root in runtime_roots:
        ct_jobs = runtime_root / "cable_threading" / "jobs"
        if ct_jobs.is_dir():
            for job_dir in ct_jobs.iterdir():
                if not job_dir.is_dir():
                    continue
                candidate = job_dir / "datasets" / "dataset.hdf5"
                if candidate.is_file():
                    add(candidate)

    for runtime_root in runtime_roots:
        dac_jobs = runtime_root / "dual_arm_cable" / "jobs"
        if dac_jobs.is_dir():
            for job_dir in dac_jobs.iterdir():
                if not job_dir.is_dir():
                    continue
                candidate = job_dir / "datasets" / "dataset.hdf5"
                if candidate.is_file():
                    add(candidate)

    for runtime_root in runtime_roots:
        for pattern in ("**/datasets/dataset.hdf5",):
            for candidate in runtime_root.glob(pattern):
                if "node_modules" in candidate.parts:
                    continue
                add(candidate)

    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def backfill_hdf5_dataset_records(
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    hdf5_paths: list[Path] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scannedDatasets": 0,
        "insertedHdf5Datasets": 0,
        "updatedHdf5Datasets": 0,
        "insertedDataAssets": 0,
        "updatedDataAssets": 0,
        "skippedDatasets": 0,
        "errors": [],
    }

    paths = hdf5_paths if hdf5_paths is not None else discover_dataset_hdf5_paths()
    result["scannedDatasets"] = len(paths)
    if dry_run:
        return result

    from app.core.database import SessionLocal

    for hdf5_path in paths:
        try:
            manifest, source_job_id = _read_manifest_for_hdf5(hdf5_path)
            row = build_dataset_row_from_hdf5(hdf5_path, manifest=manifest, source_job_id=source_job_id)
            storage_uri = f"file://{hdf5_path}"
            file_size = hdf5_path.stat().st_size if hdf5_path.is_file() else 0
            display_name = str(
                manifest.get("displayName")
                or manifest.get("datasetName")
                or manifest.get("name")
                or row.get("id")
            )

            with SessionLocal() as db:
                existing_hdf5 = (
                    db.query(HDF5Dataset).filter(HDF5Dataset.storage_uri == storage_uri).one_or_none()
                )
                if existing_hdf5 is None:
                    db.add(
                        HDF5Dataset(
                            name=display_name,
                            project=str(row.get("taskType") or "simulation"),
                            task=str(row.get("taskTemplateId") or ""),
                            device=str(row.get("robotType") or "Panda"),
                            source="simulation_generated",
                            file_size_bytes=int(file_size),
                            format="HDF5",
                            storage_type="local",
                            storage_uri=storage_uri,
                            qc_status="passed",
                            label_status="unlabeled",
                            assign_status="unassigned",
                            tags=json.dumps(
                                {
                                    "datasetId": row.get("id"),
                                    "sourceJobId": source_job_id,
                                    "actionMode": row.get("trainedActionMode"),
                                    "evalExecutor": row.get("evalExecutor"),
                                },
                                ensure_ascii=False,
                            ),
                        )
                    )
                    result["insertedHdf5Datasets"] += 1
                elif overwrite:
                    existing_hdf5.name = display_name
                    existing_hdf5.file_size_bytes = int(file_size)
                    existing_hdf5.tags = json.dumps(
                        {
                            "datasetId": row.get("id"),
                            "sourceJobId": source_job_id,
                            "actionMode": row.get("trainedActionMode"),
                            "evalExecutor": row.get("evalExecutor"),
                        },
                        ensure_ascii=False,
                    )
                    result["updatedHdf5Datasets"] += 1
                else:
                    result["skippedDatasets"] += 1
                db.commit()

            dataset_id = str(row.get("id") or "")
            meta_payload = {
                "workspaceDatasetId": dataset_id,
                "datasetId": dataset_id,
                "datasetName": display_name,
                "displayName": display_name,
                "sourceJobId": source_job_id,
                "format": "hdf5",
                "datasetFormat": "hdf5",
                "datasetFile": str(hdf5_path),
                "hdf5Path": str(hdf5_path),
                "manifestPath": manifest.get("manifestPath"),
                "taskTemplateId": row.get("taskTemplateId"),
                "taskType": row.get("taskType"),
                "observationSchema": row.get("observationSchema"),
                "actionSchema": row.get("actionSchema"),
                "controllerSchema": row.get("controllerSchema"),
                "sideChannelSchema": row.get("sideChannelSchema"),
                "trainedActionMode": row.get("trainedActionMode"),
                "evalExecutor": row.get("evalExecutor"),
                "episodeCount": row.get("episodeCount"),
                "successfulEpisodes": int(
                    manifest.get("successfulEpisodes")
                    or manifest.get("num_successful")
                    or row.get("episodeCount")
                    or 0
                ),
                "trainable": bool(
                    manifest.get("trainable", True)
                    and int(
                        manifest.get("successfulEpisodes")
                        or manifest.get("num_successful")
                        or row.get("episodeCount")
                        or 0
                    )
                    > 0
                ),
                "directTrainable": bool(
                    manifest.get("directTrainable", manifest.get("trainable", True))
                ),
                "availableFormats": ["hdf5"],
                "jointActionAvailable": row.get("jointActionAvailable"),
            }
            # data_assets.dataset_id is varchar(32); long workspace ids live in meta only.
            asset_dataset_id = dataset_id if len(dataset_id) <= 32 else None

            with DataAssetsSyncSessionLocal() as assets_db:
                existing_asset = None
                if asset_dataset_id:
                    existing_asset = (
                        assets_db.query(DataAsset)
                        .filter(DataAsset.dataset_id == asset_dataset_id)
                        .one_or_none()
                    )
                if existing_asset is None and dataset_id:
                    existing_asset = (
                        assets_db.query(DataAsset)
                        .filter(DataAsset.meta.contains(f'"workspaceDatasetId": "{dataset_id}"'))
                        .one_or_none()
                    )
                if existing_asset is None:
                    existing_asset = (
                        assets_db.query(DataAsset)
                        .filter(DataAsset.file_path == str(hdf5_path))
                        .one_or_none()
                    )

                if existing_asset is None:
                    max_code = assets_db.query(func.max(DataAsset.code)).scalar()
                    next_num = int(max_code or 0) + 1 if str(max_code or "").isdigit() else (
                        assets_db.query(func.count(DataAsset.id)).scalar() or 0
                    ) + 1
                    code = f"{next_num:04d}"
                    assets_db.add(
                        DataAsset(
                            dataset_id=asset_dataset_id,
                            code=code,
                            filename=hdf5_path.name,
                            format="hdf5",
                            source="simulation_generated",
                            file_path=str(hdf5_path),
                            file_size_bytes=int(file_size),
                            meta=json.dumps(meta_payload, ensure_ascii=False),
                            parse_status="成功",
                            sync_status="synced",
                        )
                    )
                    result["insertedDataAssets"] += 1
                elif overwrite:
                    existing_asset.meta = json.dumps(meta_payload, ensure_ascii=False)
                    existing_asset.file_size_bytes = int(file_size)
                    if asset_dataset_id and not existing_asset.dataset_id:
                        existing_asset.dataset_id = asset_dataset_id
                    result["updatedDataAssets"] += 1
                assets_db.commit()

            try:
                from app.services.artifact_upload_service import upload_data_asset_file

                owner = asset_dataset_id or source_job_id or dataset_id
                if owner:
                    upload_data_asset_file(owner, hdf5_path)
                elif source_job_id:
                    from app.services.artifact_upload_service import upload_dataset_artifacts

                    upload_dataset_artifacts(source_job_id, hdf5_path.parent.parent)
            except Exception as upload_exc:
                logger.warning("dataset minio upload skipped for %s: %s", hdf5_path, upload_exc)
        except Exception as exc:
            logger.warning("dataset backfill failed for %s: %s", hdf5_path, exc)
            result["errors"].append(f"{hdf5_path}: {exc}")

    return result
