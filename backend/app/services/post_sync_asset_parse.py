"""
采集同步到 MinIO 之后：合并 meta.storage 并做轻量解析，刷新 parse_status / error_msg。

采集登记时平台常无本地文件，仅写入 meta.collect +「等待落盘后再解析」；同步完成后应从 MinIO
拉缓存文件并 parse_meta_for_asset，与「采集端已有文件」登记路径对齐。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.data_asset import get_asset_by_id, update_asset
from app.models.data_asset import DataAsset

logger = logging.getLogger(__name__)


def _storage_minio_path_from_meta(meta_json: Optional[str]) -> str:
    if not meta_json:
        return ""
    try:
        parsed = json.loads(meta_json)
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    storage = parsed.get("storage")
    if not isinstance(storage, dict):
        return ""
    v = storage.get("minio_path")
    if isinstance(v, str) and v.strip().startswith("minio://"):
        return v.strip()
    return ""


def _minio_uri_from_asset(meta_json: Optional[str], file_path: Optional[str]) -> str:
    """优先 meta.storage.minio_path，否则 file_path（minio://）。"""
    u = _storage_minio_path_from_meta(meta_json)
    if u:
        return u
    fp = (file_path or "").strip()
    return fp if fp.startswith("minio://") else ""


async def refresh_parse_from_minio_for_asset(db: AsyncSession, asset: DataAsset) -> Dict[str, Any]:
    """
    对已落在 MinIO 的资产重新执行「同步后解析」逻辑（不重新上传）。
    用于修复历史数据中仍保留「等待落盘后再解析」等占位状态。
    """
    aid = int(asset.id)
    meta = getattr(asset, "meta", None)
    meta_s = meta if isinstance(meta, str) else None
    fp = (getattr(asset, "file_path", None) or "").strip()
    mu = _minio_uri_from_asset(meta_s, fp)
    fmt = (getattr(asset, "format", "") or "").strip().lower()
    if not mu.startswith("minio://"):
        return {"ok": False, "asset_id": aid, "reason": "no_minio_uri"}
    if fmt not in ("mcap", "hdf5"):
        return {"ok": False, "asset_id": aid, "reason": "unsupported_format", "format": fmt}
    base_meta = meta_s.strip() if (meta_s and meta_s.strip()) else "{}"
    await update_asset_after_minio_sync_with_parse(
        db,
        asset_id=aid,
        merged_meta_json=base_meta,
        minio_path=mu,
        file_format=fmt,
    )
    fresh = await get_asset_by_id(db, aid)
    em = (getattr(fresh, "error_msg", None) or "") if fresh else ""
    still_stale = "等待落盘" in em
    return {
        "ok": True,
        "asset_id": aid,
        "parse_status": getattr(fresh, "parse_status", None) if fresh else None,
        "still_stale_error": still_stale,
    }


async def refresh_parse_from_minio_for_asset_id(db: AsyncSession, asset_id: int) -> Dict[str, Any]:
    asset = await get_asset_by_id(db, asset_id)
    if not asset:
        return {"ok": False, "asset_id": asset_id, "reason": "not_found"}
    return await refresh_parse_from_minio_for_asset(db, asset)


async def batch_refresh_parse_from_minio(
    db: AsyncSession,
    *,
    allowed_project_ids: Optional[List[str]],
    project_id: Optional[str],
    limit: int,
    stale_error_filter: bool,
    asset_ids: Optional[List[int]],
) -> Dict[str, Any]:
    """
    批量重新解析。stale_error_filter=True 时仅 error_msg 含「等待落盘」的记录。
    asset_ids 非空时按 ID 列表处理（仍受 allowed_project_ids 限制），并忽略 stale 条件。
    """
    items: List[Dict[str, Any]] = []
    if asset_ids:
        unique_ids = sorted({int(i) for i in asset_ids if i is not None})[:300]
        if not unique_ids:
            return {"ok": True, "processed": 0, "items": []}
        stmt = select(DataAsset).where(DataAsset.id.in_(unique_ids))
        if allowed_project_ids is not None:
            if not allowed_project_ids:
                return {"ok": True, "processed": 0, "items": [], "note": "无可见项目"}
            stmt = stmt.where(DataAsset.project_id.in_(allowed_project_ids))
        r = await db.execute(stmt)
        rows = list(r.scalars().all())
    else:
        conds = [
            DataAsset.file_path.startswith("minio://"),
            func.lower(func.coalesce(DataAsset.format, "")).in_(["mcap", "hdf5"]),
        ]
        if stale_error_filter:
            conds.append(DataAsset.error_msg.ilike("%等待落盘%"))
        if allowed_project_ids is not None:
            if not allowed_project_ids:
                return {"ok": True, "processed": 0, "items": [], "note": "无可见项目"}
            conds.append(DataAsset.project_id.in_(allowed_project_ids))
        if project_id and project_id.strip():
            conds.append(DataAsset.project_id == project_id.strip())
        stmt = select(DataAsset).where(*conds).order_by(DataAsset.id.asc()).limit(max(1, min(limit, 2000)))
        r = await db.execute(stmt)
        rows = list(r.scalars().all())

    for asset in rows:
        out = await refresh_parse_from_minio_for_asset(db, asset)
        items.append(out)
    ok_n = sum(1 for x in items if x.get("ok"))
    fixed_n = sum(1 for x in items if x.get("ok") and not x.get("still_stale_error"))
    return {
        "ok": True,
        "processed": len(items),
        "succeeded": ok_n,
        "cleared_stale_message": fixed_n,
        "failed": len(items) - ok_n,
        "items": items,
    }


async def update_asset_after_minio_sync_with_parse(
    db: AsyncSession,
    *,
    asset_id: int,
    merged_meta_json: str,
    minio_path: str,
    file_format: str,
) -> None:
    """
    写入合并后的 meta、synced，并在 mcap/hdf5 时尽量做一次轻量解析以清除「等待落盘」类占位文案。
    """
    fmt = (file_format or "").strip().lower()
    mu = (minio_path or "").strip()
    if not mu.startswith("minio://") or fmt not in ("mcap", "hdf5"):
        await update_asset(
            db,
            asset_id,
            meta=merged_meta_json,
            sync_status="synced",
            sync_error=None,
        )
        return

    try:
        from app.services.data_asset_path_resolver import resolve_read_local_from_warehouse_uri
        from app.services.asset_meta_parser import parse_meta_for_asset

        local_parse = await asyncio.to_thread(resolve_read_local_from_warehouse_uri, mu)
        meta_parse_json, parse_status, err_msg = parse_meta_for_asset(local_parse, fmt)
        try:
            m_full: dict = json.loads(merged_meta_json) if merged_meta_json else {}
        except Exception:
            m_full = {}
        if not isinstance(m_full, dict):
            m_full = {}
        if meta_parse_json:
            try:
                m_parse = json.loads(meta_parse_json)
                if isinstance(m_parse, dict):
                    for k, v in m_parse.items():
                        m_full[k] = v
            except Exception:
                pass
        final_meta = json.dumps(m_full, ensure_ascii=False)
        await update_asset(
            db,
            asset_id,
            meta=final_meta,
            sync_status="synced",
            sync_error=None,
            parse_status=parse_status,
            # 必须传非 None 才能覆盖登记阶段的「等待落盘后再解析」占位文案
            error_msg="" if not err_msg else err_msg,
        )
    except Exception as e:
        logger.warning(
            "post_sync_parse: 同步后解析失败 asset_id=%s minio=%r err=%s",
            asset_id,
            mu[:120],
            e,
            exc_info=True,
        )
        await update_asset(
            db,
            asset_id,
            meta=merged_meta_json,
            sync_status="synced",
            sync_error=None,
        )
