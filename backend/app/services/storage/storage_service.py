"""工业级统一对象存储服务（业务层唯一入口）。

所有 MinIO / 本地文件 URI 操作必须通过本模块，禁止业务代码直接调用 minio client。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from app.core.config import settings
from app.services.storage.minio_storage_service import MinioStorageService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedStorageUri:
    """解析后的 storage URI。"""

    scheme: str
    raw: str
    bucket: Optional[str] = None
    object_key: Optional[str] = None
    local_path: Optional[Path] = None

    @property
    def is_minio(self) -> bool:
        return self.scheme == "minio"

    @property
    def is_file(self) -> bool:
        return self.scheme == "file"

    def to_dict(self) -> dict[str, str | None]:
        """解析结果为 {scheme, bucket, key} 字典（key 对 file:// 为本地路径字符串）。"""
        if self.scheme == "minio":
            return {"scheme": self.scheme, "bucket": self.bucket, "key": self.object_key}
        local = str(self.local_path) if self.local_path is not None else None
        return {"scheme": self.scheme, "bucket": None, "key": local}


class StorageService:
    """统一 storage 接口：upload / download / delete / exists / parse_uri。"""

    @staticmethod
    def is_remote_storage_enabled() -> bool:
        return MinioStorageService.is_minio_configured()

    @staticmethod
    def normalize_uri(uri: str) -> str:
        return MinioStorageService.normalize_uri(uri)

    @staticmethod
    def to_file_uri(local_path: Path | str) -> str:
        path = Path(local_path).resolve()
        return f"file://{path}"

    @classmethod
    def parse_uri(cls, storage_uri: str) -> ParsedStorageUri:
        text = cls.normalize_uri(storage_uri)
        if not text:
            raise ValueError("storage_uri is empty")

        if text.startswith("minio://"):
            remainder = text[len("minio://") :]
            if "/" not in remainder:
                return ParsedStorageUri(scheme="minio", raw=text, bucket=remainder, object_key="")
            bucket, key = remainder.split("/", 1)
            return ParsedStorageUri(
                scheme="minio",
                raw=text,
                bucket=bucket.strip(),
                object_key=unquote(key.lstrip("/")),
            )

        if text.startswith("file://"):
            local = Path(unquote(text[len("file://") :]))
            return ParsedStorageUri(scheme="file", raw=text, local_path=local)

        parsed = urlparse(text)
        if parsed.scheme in {"", "file"} and parsed.path:
            local = Path(unquote(parsed.path))
            return ParsedStorageUri(scheme="file", raw=text, local_path=local)

        if "://" not in text:
            local = Path(text)
            return ParsedStorageUri(scheme="file", raw=cls.to_file_uri(local), local_path=local)

        raise ValueError(f"unsupported storage_uri scheme: {text}")

    @classmethod
    def parse_uri_dict(cls, storage_uri: str) -> dict[str, str | None]:
        """parse_uri 的字典形式：{scheme, bucket, key}。"""
        return cls.parse_uri(storage_uri).to_dict()

    @staticmethod
    def local_path_from_uri(storage_uri: str) -> Optional[Path]:
        try:
            parsed = StorageService.parse_uri(storage_uri)
        except ValueError:
            return MinioStorageService.local_path_from_uri(storage_uri)
        if parsed.local_path is not None:
            return parsed.local_path
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
        return MinioStorageService.upload_file(
            local_path,
            object_key,
            bucket=bucket,
            content_type=content_type,
        )

    @classmethod
    def download_file(cls, storage_uri: str, local_path: Path | str) -> Path:
        return MinioStorageService.download_file(storage_uri, local_path)

    @classmethod
    def delete_file(cls, storage_uri: str) -> None:
        MinioStorageService.delete_file(storage_uri)

    @classmethod
    def exists(cls, storage_uri: str) -> bool:
        return MinioStorageService.exists(storage_uri)

    @staticmethod
    def default_checkpoint_bucket() -> str:
        return (getattr(settings, "CHECKPOINT_ARCHIVE_BUCKET", None) or "eai-checkpoints").strip()

    @staticmethod
    def default_workspace_bucket() -> str:
        return (getattr(settings, "WORKSPACE_ARTIFACT_BUCKET", None) or "eai-workspace-artifacts").strip()
