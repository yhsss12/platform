"""
采集作业完成数与 data_assets 表对账：删除采集资产后回写 collection_jobs 进度/状态。
与 routes_jobs.py 中按 job_id 子串统计资产条数的逻辑保持一致。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.data_asset import CollectionJobAsset, DataAsset
from app.services.collect_progress import apply_progress_guard

logger = logging.getLogger(__name__)


def extract_collect_job_id_from_asset(asset: DataAsset) -> Optional[str]:
    if (getattr(asset, "source", None) or "").lower() != "collect":
        return None
    return extract_collect_job_id_from_meta_text(getattr(asset, "meta", None))


def extract_collect_job_id_from_meta_text(meta_text: Optional[str]) -> Optional[str]:
    if not meta_text or not str(meta_text).strip():
        return None
    try:
        m = json.loads(meta_text)
        if isinstance(m, dict):
            c = m.get("collect")
            if isinstance(c, dict):
                jid = c.get("job_id")
                if jid:
                    s = str(jid).strip()
                    if s:
                        return s
    except Exception:
        pass
    return None


def _status_after_reconcile(
    old_status: str,
    *,
    next_current: int,
    next_total: int,
    percent: int,
) -> str:
    o = (old_status or "").strip().upper()
    if o in ("CANCELED", "FAILED"):
        return o
    if next_total > 0 and next_current >= next_total and percent >= 100:
        return "SUCCEEDED" if o == "SUCCEEDED" else "COMPLETED"
    if next_current > 0:
        return "RUNNING"
    return "PENDING"


def derive_job_status_after_reconcile(
    old_status: str,
    *,
    next_current: int,
    next_total: int,
    percent: int,
) -> str:
    """供磁盘对账等模块复用，与 _status_after_reconcile 一致。"""
    return _status_after_reconcile(
        old_status,
        next_current=next_current,
        next_total=next_total,
        percent=percent,
    )


async def reconcile_collection_job_progress_from_data_assets(
    db: AsyncSession,
    job_id: str,
) -> None:
    """
    按与 PATCH 作业后「资产条数对账」相同的方式统计仍存在的采集资产，并回写
    completed_count / progress；必要时将 COMPLETED 降回 RUNNING 等。
    """
    probe_id = (job_id or "").strip()
    if not probe_id:
        return
    try:
        db_job = await db.get(CollectionJobAsset, probe_id)
        if not db_job:
            return
        actual_count = (
            await db.execute(
                select(func.count(DataAsset.id)).where(
                    DataAsset.source == "collect",
                    DataAsset.meta.is_not(None),
                    DataAsset.meta.contains("job_id"),
                    DataAsset.meta.contains(probe_id),
                )
            )
        ).scalar_one() or 0
        actual_count = int(actual_count)
        cur = int(getattr(db_job, "completed_count", 0) or 0)
        prev_total = int(getattr(db_job, "collection_quantity", 0) or 0)
        existing_percent = int(getattr(db_job, "progress", 0) or 0)
        next_current, next_total, percent, _, _ = apply_progress_guard(
            existing_current=cur,
            existing_total=prev_total,
            existing_percent=existing_percent,
            desired_current=actual_count,
            desired_total=None,
            allow_reset=True,
            protect_total_regression=False,
        )
        old_status = (db_job.status or "").strip().upper()
        db_job.completed_count = next_current
        db_job.collection_quantity = next_total
        db_job.progress = percent
        if old_status not in ("CANCELED", "FAILED"):
            db_job.status = _status_after_reconcile(
                old_status,
                next_current=next_current,
                next_total=next_total,
                percent=percent,
            )
        await db.commit()
    except Exception as e:
        logger.warning(
            "reconcile_collection_job_progress_from_data_assets failed job_id=%s err=%s",
            probe_id,
            e,
            exc_info=True,
        )
        try:
            await db.rollback()
        except Exception:
            pass


async def decrement_collection_job_completed_for_removed_episode(
    db: AsyncSession,
    job_id: str,
    *,
    decrement_by: int = 1,
) -> None:
    """
    仅删除采集端目录但保留平台 data_assets / MinIO 时，无法用「资产条数」对账，
    因此按完成数减一（下限 0）回写作业进度与状态。
    """
    probe_id = (job_id or "").strip()
    if not probe_id or decrement_by <= 0:
        return
    try:
        db_job = await db.get(CollectionJobAsset, probe_id)
        if not db_job:
            return
        cur = int(getattr(db_job, "completed_count", 0) or 0)
        prev_total = int(getattr(db_job, "collection_quantity", 0) or 0)
        existing_percent = int(getattr(db_job, "progress", 0) or 0)
        desired = max(0, cur - int(decrement_by))
        next_current, next_total, percent, _, _ = apply_progress_guard(
            existing_current=cur,
            existing_total=prev_total,
            existing_percent=existing_percent,
            desired_current=desired,
            desired_total=None,
            allow_reset=True,
            protect_total_regression=False,
        )
        old_status = (db_job.status or "").strip().upper()
        db_job.completed_count = next_current
        db_job.collection_quantity = next_total
        db_job.progress = percent
        if old_status not in ("CANCELED", "FAILED"):
            db_job.status = _status_after_reconcile(
                old_status,
                next_current=next_current,
                next_total=next_total,
                percent=percent,
            )
        await db.commit()
        logger.info(
            "collection_job decremented after remote-only episode delete job_id=%s %s->%s",
            probe_id,
            cur,
            next_current,
        )
    except Exception as e:
        logger.warning(
            "decrement_collection_job_completed_for_removed_episode failed job_id=%s err=%s",
            probe_id,
            e,
            exc_info=True,
        )
        try:
            await db.rollback()
        except Exception:
            pass
