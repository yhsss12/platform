"""
数据仓库（MinIO）为本机读取的唯一来源。

业务规则：
- 查看 / 标注 / 帧读取等所需的「本机路径」一律通过 minio:// 从对象存储下载到
  DATA_ASSETS_ROOT/.minio_view_cache 后使用；禁止把「本地磁盘 file_path」当作源数据读取
  （上传/同步仍可能产生本地临时文件，但不作为读路径）。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from typing import Dict, Optional, Tuple

from app.db.data_assets_session import DATA_ASSETS_ROOT
from app.models.data_asset import DataAsset
from app.services.asset_registration_service import DataAssetsSyncSessionLocal
from app.services.minio_service import MinioBucketError, _parse_minio_uri, download_by_minio_uri

_minio_view_cache: Dict[str, str] = {}
_minio_view_locks: Dict[str, threading.Lock] = {}
_minio_view_locks_guard = threading.Lock()

_WAREHOUSE_MISSING_MSG = (
    "资产缺少 MinIO 对象地址（file_path 或 meta.storage.minio_path 需为 minio://...），"
    "数据仅从对象仓库读取"
)


def _extract_minio_path(meta_json: Optional[str]) -> Optional[str]:
    if not meta_json:
        return None
    try:
        parsed = json.loads(meta_json)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    storage = parsed.get("storage")
    if not isinstance(storage, dict):
        return None
    minio_path = storage.get("minio_path")
    if isinstance(minio_path, str) and minio_path.strip():
        return minio_path.strip()
    return None


def minio_uri_from_fields(file_path: Optional[str], meta_json: Optional[str]) -> Optional[str]:
    """返回 minio://...，若无则 None。"""
    fp = (file_path or "").strip()
    if fp.startswith("minio://"):
        return fp
    mp = _extract_minio_path(meta_json)
    if mp and mp.startswith("minio://"):
        return mp
    return None


def _cached_download(minio_uri: str) -> str:
    if minio_uri in _minio_view_cache:
        cached = _minio_view_cache[minio_uri]
        if cached and os.path.exists(cached):
            return cached
    with _minio_view_locks_guard:
        lock = _minio_view_locks.get(minio_uri)
        if lock is None:
            lock = threading.Lock()
            _minio_view_locks[minio_uri] = lock
    with lock:
        # Double-check：等待锁期间可能已有其它请求下载完成
        cached = _minio_view_cache.get(minio_uri)
        if cached and os.path.exists(cached):
            return cached
        cache_root = str(DATA_ASSETS_ROOT / ".minio_view_cache")
        local_path = download_by_minio_uri(minio_uri, cache_root)
        _minio_view_cache[minio_uri] = local_path
        return local_path


def resolve_read_local_from_warehouse_uri(minio_uri: str) -> str:
    """从数据仓库 URI 解析出本机可读绝对路径（缓存目录）。"""
    u = (minio_uri or "").strip()
    if not u.startswith("minio://"):
        raise MinioBucketError(f"数据仓库路径必须以 minio:// 开头: {u!r}")
    return _cached_download(u)


def local_cache_path_for_minio_uri(minio_uri: str) -> str:
    """
    根据 minio://bucket/key 计算 MinIO 对象在本机缓存目录中的路径（不触发下载）。
    """
    u = (minio_uri or "").strip()
    if not u.startswith("minio://"):
        raise MinioBucketError(f"数据仓库路径必须以 minio:// 开头: {u!r}")
    bucket, key = _parse_minio_uri(u)
    cache_root = os.path.abspath(str(DATA_ASSETS_ROOT / ".minio_view_cache"))
    if not key:
        raise MinioBucketError("minio_uri 缺少 key")
    # prefix（以 / 结尾）缓存到目录；单对象缓存到文件
    if key.endswith("/"):
        local_dir = os.path.join(cache_root, key.rstrip("/"))
        return os.path.abspath(local_dir)
    return os.path.abspath(os.path.join(cache_root, key))


def evict_minio_view_cache(minio_uri: str) -> None:
    """
    清理某个 minio_uri 对应的本地缓存文件/目录，释放磁盘。
    """
    u = (minio_uri or "").strip()
    if not u:
        return
    try:
        cache_root = os.path.abspath(str(DATA_ASSETS_ROOT / ".minio_view_cache"))
        local_path = _minio_view_cache.pop(u, None) or local_cache_path_for_minio_uri(u)
        if not os.path.exists(local_path):
            return
        # 安全保护：只允许清理 cache_root 内的文件
        if os.path.commonpath([cache_root, os.path.abspath(local_path)]) != cache_root:
            return
        if os.path.isdir(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
        else:
            try:
                os.remove(local_path)
            except FileNotFoundError:
                pass
    except Exception:
        # 清理失败不影响业务主流程
        return


def clear_all_minio_view_cache() -> Dict[str, int]:
    """
    清理 DATA_ASSETS_ROOT/.minio_view_cache 全部缓存，并重置进程内缓存索引。
    返回粗略统计信息，便于接口层回显。
    """
    cache_root = os.path.abspath(str(DATA_ASSETS_ROOT / ".minio_view_cache"))
    removed_files = 0
    removed_dirs = 0
    removed_bytes = 0
    try:
        if os.path.isdir(cache_root):
            for root, dirs, files in os.walk(cache_root, topdown=False):
                for name in files:
                    fp = os.path.join(root, name)
                    try:
                        removed_bytes += os.path.getsize(fp)
                    except Exception:
                        pass
                    try:
                        os.remove(fp)
                        removed_files += 1
                    except FileNotFoundError:
                        pass
                for name in dirs:
                    dp = os.path.join(root, name)
                    try:
                        os.rmdir(dp)
                        removed_dirs += 1
                    except Exception:
                        pass
    finally:
        _minio_view_cache.clear()
        _minio_view_locks.clear()
    return {
        "removed_files": int(removed_files),
        "removed_dirs": int(removed_dirs),
        "removed_bytes": int(removed_bytes),
    }


def clear_minio_conversion_download_cache() -> Dict[str, int]:
    """
    清理 DATA_ASSETS_ROOT/_minio_conversion_cache（MCAP 转换前从 MinIO 下载到本地的副本）。
    删除后下次转换会重新下载；不影响对象仓库与数据库中的 minio:// 记录。
    """
    cache_root = os.path.abspath(str(DATA_ASSETS_ROOT / "_minio_conversion_cache"))
    removed_files = 0
    removed_dirs = 0
    removed_bytes = 0
    try:
        if os.path.isdir(cache_root):
            for root, dirs, files in os.walk(cache_root, topdown=False):
                for name in files:
                    fp = os.path.join(root, name)
                    try:
                        removed_bytes += os.path.getsize(fp)
                    except Exception:
                        pass
                    try:
                        os.remove(fp)
                        removed_files += 1
                    except FileNotFoundError:
                        pass
                for name in dirs:
                    dp = os.path.join(root, name)
                    try:
                        os.rmdir(dp)
                        removed_dirs += 1
                    except Exception:
                        pass
    except Exception:
        pass
    return {
        "removed_files": int(removed_files),
        "removed_dirs": int(removed_dirs),
        "removed_bytes": int(removed_bytes),
    }


def warehouse_uri_for_local_under_prefix(prefix_minio_uri: str, local_root: str, local_file: str) -> str:
    """
    目录型资产下载到本地后，将某个缓存文件路径还原为对象仓库中单对象的 minio URI。
    prefix_minio_uri 须为前缀形式 minio://bucket/prefix/（以 / 结尾）。
    """
    bucket, key = _parse_minio_uri(prefix_minio_uri)
    lr = os.path.normpath(local_root)
    lf = os.path.normpath(local_file)
    if not key.endswith("/"):
        return prefix_minio_uri.strip()
    prefix_key = key.rstrip("/")
    rel = os.path.relpath(lf, lr)
    if rel.startswith(".."):
        raise ValueError(f"local_file not under local_root: {lf} vs {lr}")
    rel_posix = rel.replace("\\", "/")
    if rel_posix == ".":
        full_key = prefix_key
    else:
        full_key = f"{prefix_key}/{rel_posix}"
    return f"minio://{bucket}/{full_key}"


def resolve_local_path_from_fields(
    file_path: Optional[str],
    meta_json: Optional[str],
) -> str:
    """
    返回可用于 os.path / h5py / mcap 的本机绝对路径（数据仅来自 MinIO）。
    失败时抛出 FileNotFoundError（缺 URI）或 MinioBucketError。
    """
    uri = minio_uri_from_fields(file_path, meta_json)
    if not uri:
        raise FileNotFoundError(_WAREHOUSE_MISSING_MSG)
    return _cached_download(uri)


def resolve_local_path_for_data_asset(asset: DataAsset) -> str:
    return resolve_local_path_from_fields(getattr(asset, "file_path", None), getattr(asset, "meta", None))


def resolve_label_task_warehouse_and_local(config_path: str) -> Tuple[str, str]:
    """
    标注任务 dataset_path 的一项：仅允许 MinIO 或已在库中登记且含 MinIO 的资产。
    返回 (本地缓存根路径, 资产在仓库中的 URI — 单文件或前缀目录)。
    """
    p = (config_path or "").strip()
    if not p:
        raise FileNotFoundError("empty dataset path")
    if p.startswith("minio://"):
        local = _cached_download(p)
        return local, p
    session = DataAssetsSyncSessionLocal()
    try:
        row = session.query(DataAsset).filter(DataAsset.file_path == p).one_or_none()
        if row is None:
            raise FileNotFoundError(p)
        uri = minio_uri_from_fields(row.file_path, row.meta)
        if not uri:
            raise FileNotFoundError(_WAREHOUSE_MISSING_MSG)
        local = _cached_download(uri)
        return local, uri
    finally:
        session.close()


def resolve_label_task_warehouse_uri(config_path: str) -> str:
    """
    仅解析标注任务路径对应的 MinIO URI（不触发下载）。
    """
    p = (config_path or "").strip()
    if not p:
        raise FileNotFoundError("empty dataset path")
    if p.startswith("minio://"):
        return p
    session = DataAssetsSyncSessionLocal()
    try:
        row = session.query(DataAsset).filter(DataAsset.file_path == p).one_or_none()
        if row is None:
            raise FileNotFoundError(p)
        uri = minio_uri_from_fields(row.file_path, row.meta)
        if not uri:
            raise FileNotFoundError(_WAREHOUSE_MISSING_MSG)
        return uri
    finally:
        session.close()


def resolve_label_task_dataset_path(path: str) -> str:
    """兼容旧调用：仅返回本地缓存路径。"""
    local, _ = resolve_label_task_warehouse_and_local(path)
    return local
