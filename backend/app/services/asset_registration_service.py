
import os
import json
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from app.models.data_asset import DataAsset, ConversionJobAsset
from app.models.project_asset import Project
from app.core.config import settings
from app.services.asset_meta_parser import parse_meta_for_asset
from app.services.minio_service import (
    upload_file_to_project_bucket,
    upload_dir_to_project_bucket,
    MinioBucketError,
)
from app.services.storage_meta_merge import merge_storage_meta

# Configure logging
logger = logging.getLogger(__name__)

# 同步引擎（PostgreSQL），供后台任务/线程使用
data_assets_sync_engine = create_engine(settings.sync_database_url, echo=False, future=True)
DataAssetsSyncSessionLocal = sessionmaker(bind=data_assets_sync_engine, autoflush=False, autocommit=False)

def _format_for_asset(output_format: str) -> str:
    """
    Map output format to data_assets table format field.
    """
    upper = (output_format or "").upper()
    if upper == "HDF5":
        return "hdf5"
    if "LEROBOT" in upper:
        return "lerobot"
    return "hdf5"

def _build_derived_meta(existing_meta: Optional[str], parent_asset_id: str, input_path: str, project_id: str, project_name: str) -> str:
    """
    Record source asset info in meta JSON.
    """
    base: Dict[str, Any] = {}
    if existing_meta:
        try:
            base = json.loads(existing_meta)
        except Exception:
            base = {}
    base["derived_from"] = {
        "asset_id": str(parent_asset_id) if parent_asset_id is not None else None,
        "input_path": input_path,
        "project_id": project_id,
        "project_name": project_name,
    }
    return json.dumps(base, ensure_ascii=False)


def get_directory_size(path: str) -> int:
    """
    Calculate total size of directory in bytes.
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except Exception as e:
        logger.warning(f"Failed to calculate directory size for {path}: {e}")
    return total_size

def upsert_converted_asset(job: Any, input_path: str, output_path: str) -> None:
    """
    Register converted output file/directory as a new data asset.
    """
    try:
        if not os.path.exists(output_path):
            logger.warning(f"Converted output not found, skip asset registration: {output_path}")
            return

        is_dir = os.path.isdir(output_path)
        if is_dir:
            size = get_directory_size(output_path)
        else:
            size = os.path.getsize(output_path)
            
        fmt = _format_for_asset(getattr(job, "outputFormat", None))
        project_id = (getattr(job, "projectId", None) or "").strip()
        project_name = (getattr(job, "projectName", None) or project_id or "").strip()
        parent_asset_id = job.assetId
        filename = os.path.basename(output_path)
        job_id = str(getattr(job, "jobId", "") or "")

        # 后台产物写入必须保证 project_id 明确且项目仍可写（存在且未归档）。
        if not project_id:
            logger.error(f"Converted asset registration aborted: missing projectId (jobId={job_id})")
            return

        session = DataAssetsSyncSessionLocal()
        try:
            p = session.query(Project).filter(Project.id == project_id).one_or_none()
            if not p:
                logger.error(
                    f"Converted asset registration aborted: project not found (projectId={project_id}, jobId={job_id})"
                )
                return
            if (getattr(p, "status", None) or "").strip() == "已归档":
                logger.error(
                    f"Converted asset registration aborted: project archived (projectId={project_id}, jobId={job_id})"
                )
                return
        except Exception:
            logger.exception(
                f"Converted asset registration aborted: failed to validate project (projectId={project_id}, jobId={job_id})"
            )
            return

        # 上传转换产物到 MinIO，并以 MinIO 路径作为资产主路径
        # 上传转换产物到 MinIO，并以 MinIO 路径作为资产主路径
        if is_dir:
            object_prefix = f"projects/{project_id}/convert/{job_id}/{filename}".strip("/")
            minio_uri = upload_dir_to_project_bucket(
                project_name=project_name or project_id,
                local_dir_path=output_path,
                object_prefix=object_prefix,
            )
        else:
            object_name = f"projects/{project_id}/convert/{job_id}/{filename}".strip("/")
            minio_uri = upload_file_to_project_bucket(
                project_name=project_name or project_id,
                local_file_path=output_path,
                object_name=object_name,
            )

        # 任务记录返回 MinIO 路径，前端后续删除/查看统一使用 MinIO 地址
        try:
            job.outputPath = minio_uri
            job.outputFileName = filename
            job.fileName = filename
        except Exception:
            pass

        conversion_task_name = getattr(job, "taskName", None) or getattr(job, "task_name", None)
        if conversion_task_name and isinstance(conversion_task_name, str):
            conversion_task_name = conversion_task_name.strip() or None
        else:
            conversion_task_name = None
        operator_name = (
            (getattr(job, "operatorName", None) or getattr(job, "operator_name", None) or "").strip() or None
        )
        if not operator_name and job_id:
            rec = session.query(ConversionJobAsset).filter(ConversionJobAsset.job_id == job_id).one_or_none()
            operator_name = ((getattr(rec, "operator_name", None) or "").strip() or None) if rec else None

        try:
            existing = session.query(DataAsset).filter(DataAsset.file_path == minio_uri).one_or_none()
            if existing:
                # Update existing
                existing.filename = filename
                existing.file_size_bytes = size
                existing.format = fmt
                existing.source = "convert"
                existing.project_id = project_id
                existing.project_name = project_name or existing.project_name
                existing.meta = _build_derived_meta(existing.meta, parent_asset_id, input_path, project_id, project_name)
                existing.meta = merge_storage_meta(existing.meta, output_path, minio_uri) or existing.meta
                existing.conversion_task_name = conversion_task_name
                if not (getattr(existing, "operator_name", None) or "").strip():
                    existing.operator_name = operator_name
                session.commit()
                logger.info(f"Updated existing converted asset record for {minio_uri}")
                return

            # Create new
            count = session.query(func.count(DataAsset.id)).scalar() or 0
            code = str(count + 1).zfill(4)

            asset = DataAsset(
                code=code,
                filename=filename,
                format=fmt,
                source="convert",
                project_id=project_id,
                project_name=project_name or None,
                file_path=minio_uri,
                file_size_bytes=size,
                parse_status="解析中",
                error_msg=None,
                conversion_task_name=conversion_task_name,
                operator_name=operator_name,
            )

            # Parse meta if applicable (mainly for HDF5/files)
            # For directories (LeRobot), parse_meta_for_asset might need adjustment or return default
            if not is_dir:
                meta_json, parse_status, err_msg = parse_meta_for_asset(output_path, fmt)
            else:
                # For directories, simple default meta
                meta_json, parse_status, err_msg = None, "已完成", None

            if meta_json:
                try:
                    base_meta = json.loads(meta_json)
                except Exception:
                    base_meta = {}
                base_meta["derived_from"] = {
                    "asset_id": str(parent_asset_id) if parent_asset_id is not None else None,
                    "input_path": input_path,
                    "project_id": project_id,
                    "project_name": project_name,
                }
                asset.meta = merge_storage_meta(
                    json.dumps(base_meta, ensure_ascii=False),
                    output_path,
                    minio_uri,
                ) or json.dumps(base_meta, ensure_ascii=False)
                asset.parse_status = parse_status
                asset.error_msg = err_msg
            else:
                base_meta = _build_derived_meta(None, parent_asset_id, input_path, project_id, project_name)
                asset.meta = merge_storage_meta(base_meta, output_path, minio_uri) or base_meta
                # If directory, mark as parsed/ready unless error
                asset.parse_status = parse_status if parse_status else "已完成"
                asset.error_msg = err_msg

            session.add(asset)
            session.commit()
            logger.info(f"Registered new converted asset for {minio_uri}")
        finally:
            session.close()
    except MinioBucketError as e:
        logger.exception(f"Failed to upload converted output to MinIO for job {job.jobId}: {e}")
        raise
    except Exception as e:
        logger.exception(f"Failed to upsert converted asset for job {job.jobId}: {e}")
