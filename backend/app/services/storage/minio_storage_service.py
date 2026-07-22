"""统一对象存储服务：minio:// 为主，file:// 本地 fallback。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class MinioStorageService:
    """封装 MinIO 上传/下载/删除；MinIO 未配置时 upload 返回 file:// URI。"""

    @staticmethod
    def is_minio_configured() -> bool:
        return bool((settings.MINIO_ENDPOINT or "").strip())

    @staticmethod
    def normalize_uri(uri: str) -> str:
        text = (uri or "").strip()
        if not text:
            return ""
        if text.startswith("minio://") or text.startswith("file://"):
            return text
        path = Path(text)
        if path.is_absolute():
            return f"file://{path.resolve()}"
        return text

    @staticmethod
    def local_path_from_uri(storage_uri: str) -> Optional[Path]:
        text = (storage_uri or "").strip()
        if text.startswith("file://"):
            return Path(text[len("file://") :])
        if text and "://" not in text:
            return Path(text)
        return None

    @classmethod
    def upload_file(
        cls,
        local_path: Path | str,
        object_key: str,
        *,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"local file not found: {path}")

        key = (object_key or "").strip().lstrip("/")
        if not key:
            raise ValueError("object_key is required")

        if not cls.is_minio_configured():
            return f"file://{path.resolve()}"

        bucket_name = (bucket or getattr(settings, "WORKSPACE_ARTIFACT_BUCKET", None) or "eai-workspace-artifacts").strip()
        try:
            from app.services.minio_service import MinioConfigError, build_minio_uri, ensure_project_bucket, _client

            ensure_project_bucket(bucket_name)
            client = _client()
            client.fput_object(
                bucket_name=bucket_name,
                object_name=key,
                file_path=str(path),
                content_type=content_type,
            )
            return build_minio_uri(bucket_name, key)
        except Exception as exc:
            logger.warning("MinIO upload failed path=%s key=%s error=%s", path, key, exc)
            return f"file://{path.resolve()}"

    @classmethod
    def download_file(cls, storage_uri: str, local_path: Path | str) -> Path:
        uri = cls.normalize_uri(storage_uri)
        dest = Path(local_path)
        if uri.startswith("file://"):
            src = Path(uri[len("file://") :])
            if not src.is_file():
                raise FileNotFoundError(f"file:// source missing: {src}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.resolve() != src.resolve():
                import shutil

                shutil.copy2(src, dest)
            return dest.resolve()

        if uri.startswith("minio://"):
            from app.services.minio_service import download_by_minio_uri

            downloaded = download_by_minio_uri(uri, str(dest.parent))
            downloaded_path = Path(downloaded)
            if downloaded_path.is_file():
                return downloaded_path.resolve()
            candidate = dest.parent / Path(uri.split("/")[-1])
            if candidate.is_file():
                return candidate.resolve()
            raise FileNotFoundError(f"minio download failed: {uri}")

        raise ValueError(f"unsupported storage_uri: {storage_uri}")

    @classmethod
    def delete_file(cls, storage_uri: str) -> None:
        uri = cls.normalize_uri(storage_uri)
        if uri.startswith("file://"):
            path = Path(uri[len("file://") :])
            if path.is_file():
                path.unlink(missing_ok=True)
            return
        if uri.startswith("minio://"):
            from app.services.minio_service import delete_by_minio_uri

            delete_by_minio_uri(uri)
            return
        raise ValueError(f"unsupported storage_uri: {storage_uri}")

    @classmethod
    def exists(cls, storage_uri: str) -> bool:
        uri = cls.normalize_uri(storage_uri)
        if not uri:
            return False
        if uri.startswith("file://"):
            return Path(uri[len("file://") :]).is_file()
        if uri.startswith("minio://"):
            try:
                from app.services.minio_service import _parse_minio_uri, stat_object

                bucket, key = _parse_minio_uri(uri)
                stat_object(bucket, key)
                return True
            except Exception:
                return False
        return Path(uri).is_file()
