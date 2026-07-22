"""
唯一的数据读取入口：EpisodeStorage -> 可读本地文件路径。

目标：
- 业务代码不关心 abs_path / minio://warehouse_path
- MinIO 下载必须走 resolve_read_local_from_warehouse_uri（含缓存与去重锁）
"""

from __future__ import annotations

import os
import logging
from app.services.episode_storage import EpisodeStorage

from app.services.data_asset_path_resolver import resolve_read_local_from_warehouse_uri


logger = logging.getLogger(__name__)


class EpisodeResolveError(Exception):
    def __init__(self, code: str, message: str, storage: EpisodeStorage):
        self.code = code
        self.message = message
        self.episode_id = storage.episode_id
        self.warehouse_path = storage.warehouse_path
        self.abs_path = storage.abs_path
        super().__init__(message)


def assert_not_direct_path_access(storage: EpisodeStorage):
    # ⚠️ Do not access abs_path / warehouse_path directly.
    # Use EpisodeStorage instead.
    if not (storage.abs_path or "").strip() and not (storage.warehouse_path or "").strip():
        episode_id = (storage.episode_id or "").strip() or "<unknown>"
        raise RuntimeError(f"Invalid episode: no valid path source (episode_id={episode_id})")


def resolve_episode_file(storage: EpisodeStorage) -> str:
    """
    返回可读取的本地文件路径：
    - 优先使用 warehouse_path（通过 MinIO 下载到本地缓存）
    - fallback 使用 abs_path
    - 如果都不存在，抛出明确异常（包含 episode_id 和路径信息）
    """
    assert_not_direct_path_access(storage)
    episode_id = (storage.episode_id or "").strip() or "<unknown>"
    # ⚠️ DO NOT access abs_path / warehouse_path directly.
    # Use EpisodeStorage methods only.
    wh = (storage.warehouse_path or "").strip()
    ap = (storage.abs_path or "").strip()

    if wh.startswith("minio://"):
        local = resolve_read_local_from_warehouse_uri(wh)
        if not local or not os.path.exists(local):
            ex = EpisodeResolveError(
                code="EPISODE_FILE_NOT_FOUND",
                message="Episode file not found after MinIO resolve",
                storage=storage,
            )
            logger.error(
                f"[episode_resolve_error] code={ex.code} "
                f"episode_id={ex.episode_id} "
                f"warehouse={ex.warehouse_path} "
                f"abs={ex.abs_path}"
            )
            raise ex
        return local

    if ap:
        if not os.path.exists(ap):
            ex = EpisodeResolveError(
                code="EPISODE_FILE_NOT_FOUND",
                message="Episode file not found",
                storage=storage,
            )
            logger.error(
                f"[episode_resolve_error] code={ex.code} "
                f"episode_id={ex.episode_id} "
                f"warehouse={ex.warehouse_path} "
                f"abs={ex.abs_path}"
            )
            raise ex
        return ap

    ex = EpisodeResolveError(
        code="EPISODE_PATH_INVALID",
        message="No valid path source",
        storage=storage,
    )
    logger.error(
        f"[episode_resolve_error] code={ex.code} "
        f"episode_id={ex.episode_id} "
        f"warehouse={ex.warehouse_path} "
        f"abs={ex.abs_path}"
    )
    raise ex

