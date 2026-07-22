"""
转换批量任务（父任务）统计与状态汇总。
子任务仍为 conversion_jobs；父任务为 conversion_batch_jobs。
"""
from __future__ import annotations

from typing import List, Optional

from app.models.data_asset import ConversionBatchJob, ConversionJobAsset
from app.services.asset_registration_service import DataAssetsSyncSessionLocal


def _norm_status(raw: Optional[str]) -> str:
    return (raw or "queued").strip().lower()


def _compute_overall_status(states: List[str]) -> str:
    if not states:
        return "PENDING"
    n = len(states)
    if all(s == "queued" for s in states):
        return "PENDING"
    if all(s == "succeeded" for s in states):
        return "SUCCESS"
    if all(s == "canceled" for s in states):
        return "CANCELED"

    pending = sum(1 for s in states if s == "queued")
    running = sum(1 for s in states if s == "running")
    succ = sum(1 for s in states if s == "succeeded")
    fail = sum(1 for s in states if s == "failed")
    canc = sum(1 for s in states if s == "canceled")

    if pending > 0 or running > 0:
        return "RUNNING"

    # 全部已结束
    if succ == n:
        return "SUCCESS"
    if succ > 0 and (fail > 0 or canc > 0):
        return "PARTIAL_SUCCESS"
    if succ == 0 and fail > 0:
        return "FAILED"
    if succ == 0 and canc > 0:
        return "CANCELED"
    return "FAILED"


def recompute_conversion_batch_stats(batch_id: str) -> None:
    """根据子任务行重算父任务计数、进度与 overall_status。"""
    bid = (batch_id or "").strip()
    if not bid:
        return

    session = DataAssetsSyncSessionLocal()
    try:
        batch = session.query(ConversionBatchJob).filter(ConversionBatchJob.batch_id == bid).one_or_none()
        if batch is None:
            return
        rows: List[ConversionJobAsset] = (
            session.query(ConversionJobAsset).filter(ConversionJobAsset.batch_id == bid).order_by(ConversionJobAsset.id.asc()).all()
        )
        if not rows:
            batch.total_count = 0
            batch.success_count = 0
            batch.failed_count = 0
            batch.canceled_count = 0
            batch.running_count = 0
            batch.pending_count = 0
            batch.progress_percent = 0.0
            batch.overall_status = "PENDING"
            session.add(batch)
            session.commit()
            return

        states = [_norm_status(r.status) for r in rows]
        n = len(states)
        succ = sum(1 for s in states if s == "succeeded")
        fail = sum(1 for s in states if s == "failed")
        canc = sum(1 for s in states if s == "canceled")
        pending = sum(1 for s in states if s == "queued")
        running = sum(1 for s in states if s == "running")

        batch.total_count = n
        batch.success_count = succ
        # 父任务 failed_count：失败 + 取消（与「已结束」口径一致，避免前端把取消误算成进行中）
        batch.failed_count = fail + canc
        batch.canceled_count = canc
        batch.running_count = running
        batch.pending_count = pending

        # 进度采用子任务 progress_percent 的平均值，避免“长时间运行但始终 0%”。
        # 对异常值做边界收敛（0-100）。
        progress_values = []
        for r in rows:
            try:
                p = float(getattr(r, "progress_percent", 0.0) or 0.0)
            except Exception:
                p = 0.0
            progress_values.append(max(0.0, min(100.0, p)))
        avg_progress = (sum(progress_values) / n) if n else 0.0
        if succ + fail + canc == n and n > 0:
            # 全部结束时强制置 100，防止个别子任务未回写最终进度。
            batch.progress_percent = 100.0
        else:
            batch.progress_percent = round(avg_progress, 2)
        batch.overall_status = _compute_overall_status(states)

        session.add(batch)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()
