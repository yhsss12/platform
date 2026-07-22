import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error
from pypinyin import lazy_pinyin

from app.core.config import settings


BUCKET_RE = re.compile(r"^(?!\d+\.\d+\.\d+\.\d+$)[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")

_DOWNLOAD_LOCKS_GUARD = threading.Lock()
_DOWNLOAD_LOCKS: dict[str, threading.Lock] = {}


def _download_lock(key: str) -> threading.Lock:
    k = (key or "").strip()
    if not k:
        k = "unknown"
    with _DOWNLOAD_LOCKS_GUARD:
        lk = _DOWNLOAD_LOCKS.get(k)
        if lk is None:
            lk = threading.Lock()
            _DOWNLOAD_LOCKS[k] = lk
        return lk


class MinioConfigError(RuntimeError):
    pass


class MinioBucketError(RuntimeError):
    pass


@dataclass
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool


def _load_config() -> MinioConfig:
    endpoint = (settings.MINIO_ENDPOINT or "").strip()
    access_key = (settings.MINIO_ACCESS_KEY or "").strip()
    secret_key = (settings.MINIO_SECRET_KEY or "").strip()
    if not endpoint or not access_key or not secret_key:
        raise MinioConfigError("MinIO 未配置，请设置 MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY")
    return MinioConfig(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=bool(settings.MINIO_SECURE),
    )


def _validate_bucket_name(bucket_name: str) -> None:
    name = (bucket_name or "").strip()
    if not BUCKET_RE.match(name):
        raise MinioBucketError(
            "项目名不符合 MinIO bucket 命名规则。请使用 3-63 位小写字母/数字/.-，且首尾为字母或数字。"
        )


def _normalize_bucket_name(project_name: str) -> str:
    """
    将项目名转换为可用 bucket 名：
    - 中文转拼音
    - 仅保留 a-z0-9.-（其他字符替换为 -）
    - 长度限制在 3~63
    """
    raw = (project_name or "").strip()
    if not raw:
        raise MinioBucketError("项目名不能为空")

    pinyin_parts = lazy_pinyin(raw, errors="ignore")
    if pinyin_parts:
        base = "-".join(pinyin_parts).lower()
    else:
        base = raw.lower()

    base = re.sub(r"[^a-z0-9.-]+", "-", base)
    base = re.sub(r"-{2,}", "-", base).strip("-.")

    if not base:
        raise MinioBucketError("项目名转换后为空，请使用可识别名称")

    if len(base) < 3:
        base = f"{base}-bucket"
    if len(base) > 63:
        base = base[:63].rstrip("-.")
        if len(base) < 3:
            base = "proj-bucket"

    if not BUCKET_RE.match(base):
        raise MinioBucketError("项目名转换后的 bucket 名不合法，请修改项目名")
    return base


def _client() -> Minio:
    cfg = _load_config()
    return Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )


def _presign_client() -> Minio:
    """生成浏览器 PUT 预签名 URL 使用的客户端；主机名须与浏览器实际访问一致，否则勿用 127.0.0.1。"""
    cfg = _load_config()
    pub = (settings.MINIO_PUBLIC_ENDPOINT or "").strip()
    ep = pub if pub else cfg.endpoint
    return Minio(
        ep,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )


def ensure_project_bucket(bucket_name: str) -> None:
    """
    确保项目同名 bucket 存在。
    - bucket 不存在：创建
    - 已存在：直接返回
    """
    bucket = _normalize_bucket_name(bucket_name)
    _validate_bucket_name(bucket)
    client = _client()
    try:
        exists = client.bucket_exists(bucket)
        if not exists:
            client.make_bucket(bucket)
    except S3Error as e:
        raise MinioBucketError(f"创建 MinIO bucket 失败: {e.code} {e.message}") from e


def remove_project_bucket(project_name: str, *, force: bool = True, batch_size: int = 1000) -> None:
    """
    删除项目对应 bucket。
    - 默认 force=True：先递归删除桶内对象，再删除桶
    - 桶不存在视为成功
    """
    bucket = project_bucket_name(project_name)
    client = _client()
    try:
        exists = client.bucket_exists(bucket)
    except S3Error as e:
        raise MinioBucketError(f"检查 MinIO bucket 失败: {e.code} {e.message}") from e
    if not exists:
        return

    try:
        if force:
            pending: List[DeleteObject] = []
            for obj in client.list_objects(bucket, recursive=True):
                pending.append(DeleteObject(obj.object_name))
                if len(pending) >= max(1, int(batch_size)):
                    errors = client.remove_objects(bucket, pending)
                    for err in errors:
                        raise MinioBucketError(f"清空 MinIO bucket 失败: {err.code} {err.message}")
                    pending = []
            if pending:
                errors = client.remove_objects(bucket, pending)
                for err in errors:
                    raise MinioBucketError(f"清空 MinIO bucket 失败: {err.code} {err.message}")
        client.remove_bucket(bucket)
    except S3Error as e:
        # 桶不存在按已删除处理
        if e.code in ("NoSuchBucket", "NoSuchResource"):
            return
        raise MinioBucketError(f"删除 MinIO bucket 失败: {e.code} {e.message}") from e


def project_bucket_name(project_name: str) -> str:
    """项目名 -> MinIO bucket 名（含中文转拼音）。"""
    bucket = _normalize_bucket_name(project_name)
    _validate_bucket_name(bucket)
    return bucket


def upload_file_to_project_bucket(
    project_name: str,
    local_file_path: str,
    object_name: str,
    content_type: Optional[str] = None,
) -> str:
    """
    上传单文件到项目 bucket，返回 minio://bucket/object_name。
    """
    bucket = project_bucket_name(project_name)
    ensure_project_bucket(project_name)
    local = Path(local_file_path)
    if not local.is_file():
        raise MinioBucketError(f"本地文件不存在: {local_file_path}")
    object_key = object_name.strip().lstrip("/")
    if not object_key:
        raise MinioBucketError("object_name 不能为空")
    client = _client()
    try:
        client.fput_object(
            bucket_name=bucket,
            object_name=object_key,
            file_path=str(local),
            content_type=content_type,
        )
    except S3Error as e:
        raise MinioBucketError(f"上传文件到 MinIO 失败: {e.code} {e.message}") from e
    return f"minio://{bucket}/{object_key}"


def upload_dir_to_project_bucket(
    project_name: str,
    local_dir_path: str,
    object_prefix: str,
) -> str:
    """
    递归上传目录到项目 bucket，返回前缀 minio://bucket/prefix/。
    """
    bucket = project_bucket_name(project_name)
    ensure_project_bucket(project_name)
    local_dir = Path(local_dir_path)
    if not local_dir.is_dir():
        raise MinioBucketError(f"本地目录不存在: {local_dir_path}")
    prefix = object_prefix.strip().strip("/")
    if not prefix:
        raise MinioBucketError("object_prefix 不能为空")
    client = _client()
    try:
        for p in local_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(local_dir).as_posix()
            object_key = f"{prefix}/{rel}"
            client.fput_object(
                bucket_name=bucket,
                object_name=object_key,
                file_path=str(p),
            )
    except S3Error as e:
        raise MinioBucketError(f"上传目录到 MinIO 失败: {e.code} {e.message}") from e
    return f"minio://{bucket}/{prefix}/"


def build_minio_uri(bucket: str, object_key: str) -> str:
    """构建 minio://bucket/object_key（object_key 不含前导 /）。"""
    b = (bucket or "").strip()
    key = (object_key or "").strip().lstrip("/")
    if not b or not key:
        raise MinioBucketError("build_minio_uri: bucket 或 object_key 为空")
    return f"minio://{b}/{key}"


def build_minio_prefix_uri(bucket: str, prefix: str) -> str:
    """目录型前缀 URI：minio://bucket/prefix/（末尾必有 /，供前缀删除与下载）。"""
    b = (bucket or "").strip()
    p = (prefix or "").strip().lstrip("/").rstrip("/")
    if not b or not p:
        raise MinioBucketError("build_minio_prefix_uri: bucket 或 prefix 为空")
    return f"minio://{b}/{p}/"


def normalize_relative_path(relative_path: str) -> str:
    """
    规范化浏览器提供的相对路径：仅允许非空相对路径，禁止 ..、绝对路径与空段。
    返回 posix 风格，不含前导 /。
    """
    raw = (relative_path or "").strip()
    if not raw:
        raise MinioBucketError("相对路径为空")
    s = raw.replace("\\", "/").strip().lstrip("/")
    if s.startswith("..") or "/../" in f"/{s}/" or s.endswith("/.."):
        raise MinioBucketError("非法相对路径：不允许 ..")
    parts: List[str] = []
    for seg in s.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise MinioBucketError("非法相对路径：不允许 ..")
        if seg.startswith(".") and len(seg) > 1 and seg[1] == ".":
            raise MinioBucketError("非法相对路径")
        safe = re.sub(r"[^\w.\-+\s\u4e00-\u9fff]", "_", seg).strip()
        if not safe or safe in (".", ".."):
            raise MinioBucketError(f"非法路径段: {seg!r}")
        parts.append(safe)
    if not parts:
        raise MinioBucketError("相对路径无效")
    return "/".join(parts)


def list_object_names_under_prefix(bucket: str, prefix: str) -> List[str]:
    """列出 prefix 下所有对象名（recursive）。"""
    b = (bucket or "").strip()
    p = (prefix or "").strip().lstrip("/").rstrip("/")
    if not b or not p:
        raise MinioBucketError("list_object_names_under_prefix: 参数无效")
    client = _client()
    out: List[str] = []
    try:
        for obj in client.list_objects(b, prefix=p, recursive=True):
            out.append(obj.object_name)
    except S3Error as e:
        raise MinioBucketError(f"列举对象失败: {e.code} {e.message}") from e
    return out


def presigned_put_many(
    bucket: str,
    object_keys: List[str],
    *,
    expires_seconds: int = 3600,
) -> Tuple[List[str], datetime]:
    """批量生成 PUT 预签名 URL，过期时间取最后一次生成值（与单次一致）。"""
    if not object_keys:
        raise MinioBucketError("presigned_put_many: object_keys 为空")
    urls: List[str] = []
    exp: Optional[datetime] = None
    for key in object_keys:
        url, e = generate_presigned_put_url(bucket, key, expires_seconds=expires_seconds)
        urls.append(url)
        exp = e
    assert exp is not None
    return urls, exp


def generate_presigned_put_url(
    bucket: str,
    object_key: str,
    *,
    expires_seconds: int = 3600,
) -> Tuple[str, datetime]:
    """
    生成 PUT 预签名 URL，返回 (url, expires_at_utc)。
    """
    b = (bucket or "").strip()
    key = (object_key or "").strip().lstrip("/")
    if not b or not key:
        raise MinioBucketError("预签名参数无效：bucket 或 object_key 为空")
    client = _presign_client()
    try:
        url = client.presigned_put_object(
            bucket_name=b,
            object_name=key,
            expires=timedelta(seconds=max(60, min(expires_seconds, 86400 * 7))),
        )
    except S3Error as e:
        raise MinioBucketError(f"生成预签名 URL 失败: {e.code} {e.message}") from e
    exp = datetime.now(timezone.utc) + timedelta(seconds=max(60, min(expires_seconds, 86400 * 7)))
    return url, exp


def stat_object(bucket: str, object_key: str) -> Any:
    """返回 MinIO stat 结果（含 .size 等）；对象不存在时抛 MinioBucketError。"""
    b = (bucket or "").strip()
    key = (object_key or "").strip().lstrip("/")
    if not b or not key:
        raise MinioBucketError("stat_object: bucket 或 object_key 为空")
    client = _client()
    try:
        return client.stat_object(b, key)
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchObject"):
            raise MinioBucketError("对象不存在或尚未上传完成") from e
        raise MinioBucketError(f"读取对象信息失败: {e.code} {e.message}") from e


def _parse_minio_uri(minio_uri: str) -> tuple[str, str]:
    raw = (minio_uri or "").strip()
    if not raw.startswith("minio://"):
        raise MinioBucketError("minio_path 格式不合法，必须以 minio:// 开头")
    body = raw.removeprefix("minio://")
    parts = body.split("/", 1)
    bucket = (parts[0] if parts else "").strip()
    key = (parts[1] if len(parts) > 1 else "").strip()
    if not bucket:
        raise MinioBucketError("minio_path 缺少 bucket")
    return bucket, key


def delete_by_minio_uri(minio_uri: str) -> None:
    """
    删除 minio://bucket/key 或 minio://bucket/prefix/ 对应的数据。
    - 以 / 结尾按前缀递归删除
    - 否则按单对象删除
    """
    bucket, key = _parse_minio_uri(minio_uri)
    client = _client()
    try:
        if not key:
            # 禁止删除整个 bucket
            raise MinioBucketError("禁止直接删除 bucket，请提供对象或前缀路径")
        if key.endswith("/"):
            objs = client.list_objects(bucket, prefix=key, recursive=True)
            to_remove = [DeleteObject(obj.object_name) for obj in objs]
            if not to_remove:
                return
            errors = client.remove_objects(bucket, to_remove)
            for err in errors:
                # 首个错误即失败，避免部分删除后误报成功
                raise MinioBucketError(f"删除 MinIO 前缀失败: {err.code} {err.message}")
            return
        client.remove_object(bucket, key)
    except S3Error as e:
        # 对象不存在视为已删除
        if e.code in ("NoSuchKey", "NoSuchObject"):
            return
        raise MinioBucketError(f"删除 MinIO 对象失败: {e.code} {e.message}") from e


def download_by_minio_uri(minio_uri: str, dest_root: str) -> str:
    """
    下载 minio://bucket/key 或 minio://bucket/prefix/ 到 dest_root，返回本地路径。
    - key: 返回本地文件绝对路径
    - prefix/: 返回本地目录绝对路径
    """
    bucket, key = _parse_minio_uri(minio_uri)
    if not key:
        raise MinioBucketError("minio_path 缺少对象路径")
    root = Path(dest_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    client = _client()
    try:
        if key.endswith("/"):
            local_dir = root / key.rstrip("/")
            with _download_lock(str(local_dir)):
                local_dir.mkdir(parents=True, exist_ok=True)
                if local_dir.exists() and any(local_dir.rglob("*")):
                    return str(local_dir)
                has_any = False
                for obj in client.list_objects(bucket, prefix=key, recursive=True):
                    has_any = True
                    rel = obj.object_name[len(key) :].lstrip("/")
                    if not rel:
                        continue
                    local_file = local_dir / rel
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    tmp = Path(str(local_file) + ".part.minio")
                    if tmp.exists():
                        try:
                            tmp.unlink()
                        except Exception:
                            pass
                    if local_file.exists() and local_file.is_file() and local_file.stat().st_size > 0:
                        continue
                    client.fget_object(bucket, obj.object_name, str(local_file))
                if not has_any:
                    raise MinioBucketError("MinIO 前缀下无可下载对象")
                return str(local_dir)

        local_file = root / key
        with _download_lock(str(local_file)):
            local_file.parent.mkdir(parents=True, exist_ok=True)
            if local_file.exists() and local_file.is_file() and local_file.stat().st_size > 0:
                return str(local_file)
            tmp = Path(str(local_file) + ".part.minio")
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            client.fget_object(bucket, key, str(local_file))
            return str(local_file)
    except S3Error as e:
        raise MinioBucketError(f"下载 MinIO 对象失败: {e.code} {e.message}") from e
