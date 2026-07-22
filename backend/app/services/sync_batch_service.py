"""
批量数据同步：PostgreSQL 持久化 + 全局/按采集端并发上限。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.crud.data_asset import get_asset_by_id, update_asset, try_mark_asset_syncing
from app.crud.user import get_user_by_id
from app.db.data_assets_session import DataAssetsSessionLocal
from app.db.session import AsyncSessionLocal as MainSessionLocal
from app.db.session import AsyncSessionLocal
from app.models.data_asset import DataAsset, SyncBatchJob, SyncBatchJobItem
from app.models.user import User
from app.models.project_asset import Project
from app.services.agent_data_sync_proxy import (
    effective_collect_device_id_for_sync,
    resolve_sync_agent_id,
    sync_asset_via_agent,
)
from app.services.dispatcher import dispatch_task
from app.services.post_sync_asset_parse import update_asset_after_minio_sync_with_parse
from app.services.storage_meta_merge import merge_storage_meta
from app.services.task_job_store import is_cancelled

logger = logging.getLogger(__name__)

_active_batch_jobs: Set[str] = set()
_scheduled_batch_jobs: Set[str] = set()
_global_sync_sem: Optional[asyncio.Semaphore] = None
_agent_sync_sems: Dict[str, asyncio.Semaphore] = {}
_asset_sync_locks: Dict[int, asyncio.Lock] = {}
_asset_sync_locks_guard = asyncio.Lock()


def _filename_from_minio_uri(minio_uri: str) -> str:
    u = (minio_uri or "").strip()
    if not u.startswith("minio://"):
        return ""
    body = u.removeprefix("minio://")
    if "/" not in body:
        return ""
    key = body.split("/", 1)[1].strip().rstrip("/")
    if not key:
        return ""
    return key.split("/")[-1].strip()


def _get_global_sem() -> asyncio.Semaphore:
    global _global_sync_sem
    if _global_sync_sem is None:
        n = max(1, int(settings.SYNC_BATCH_MAX_CONCURRENT_GLOBAL))
        _global_sync_sem = asyncio.Semaphore(n)
    return _global_sync_sem


def _get_agent_sem(agent_key: str) -> asyncio.Semaphore:
    if agent_key not in _agent_sync_sems:
        n = max(1, int(settings.SYNC_BATCH_MAX_CONCURRENT_PER_AGENT))
        _agent_sync_sems[agent_key] = asyncio.Semaphore(n)
    return _agent_sync_sems[agent_key]


def _now() -> datetime:
    # DB 字段类型为 TIMESTAMP WITHOUT TIME ZONE（无 tzinfo）。
    # 使用 offset-naive 的 UTC 时间，避免 asyncpg 报
    # "can't subtract offset-naive and offset-aware datetimes"。
    return datetime.utcnow()


async def _get_asset_sync_lock(asset_id: int) -> asyncio.Lock:
    async with _asset_sync_locks_guard:
        lock = _asset_sync_locks.get(asset_id)
        if lock is None:
            lock = asyncio.Lock()
            _asset_sync_locks[asset_id] = lock
        return lock


async def acquire_asset_sync_lock(asset_id: int) -> asyncio.Lock:
    """
    对外暴露：单资产同步与批量同步共用同一把锁。
    调用方需在 finally 中 release()。
    """
    lock = await _get_asset_sync_lock(int(asset_id))
    await lock.acquire()
    return lock


async def _sync_one_asset(
    db: AsyncSession,
    device_db: AsyncSession,
    asset: DataAsset,
    agent_id_query: Optional[str],
    cancel_task_id: Optional[str],
) -> str:
    """执行单条同步，成功返回 minio_path；已同步返回空串。"""
    from app.api.routes_data_assets import (  # 避免循环 import
        _asset_is_synced,
        _extract_backend_local_path,
        _episode_delete_target_local,
    )

    if _asset_is_synced(asset):
        return ""

    raw_source = _extract_backend_local_path(getattr(asset, "meta", None)) or (asset.file_path or "").strip()
    if not raw_source:
        raise RuntimeError("资产缺少可同步的源路径")
    source_path = _episode_delete_target_local(raw_source)
    if not source_path:
        raise RuntimeError("资产缺少可同步的源路径")
    project_id = (asset.project_id or "").strip()
    project_name = (asset.project_name or project_id).strip()
    if not project_id:
        raise RuntimeError("资产缺少所属项目，无法同步")
    # 后台同步属于对 data_assets 的写入：要求项目仍存在且未归档
    p = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if not p:
        raise RuntimeError("项目不存在，无法同步")
    if (p.status or "").strip() == "已归档":
        raise RuntimeError("项目已归档，禁止同步")
    ret = await sync_asset_via_agent(
        device_db,
        asset_id=asset.id,
        source_path=source_path,
        project_id=project_id,
        project_name=project_name,
        agent_id=agent_id_query,
        meta_json=getattr(asset, "meta", None),
        collect_device_id=(getattr(asset, "device_id", None) or "").strip() or None,
        cancel_task_id=cancel_task_id,
    )
    minio_path = str(ret.get("minio_path") or "").strip()
    merged_meta = merge_storage_meta(getattr(asset, "meta", None), source_path, minio_path)
    if not merged_meta:
        merged_meta = merge_storage_meta(None, source_path, minio_path)
    asset.file_path = minio_path
    # 历史采集数据可能以无后缀文件名入库；同步后若对象名含 .mcap，则补齐显示名。
    fmt = (getattr(asset, "format", "") or "").strip().lower()
    if fmt == "mcap":
        fn = (getattr(asset, "filename", "") or "").strip()
        if fn and not fn.lower().endswith(".mcap"):
            key_name = _filename_from_minio_uri(minio_path)
            if key_name.lower().endswith(".mcap"):
                asset.filename = key_name
            else:
                asset.filename = f"{fn}.mcap"
    await db.commit()
    await db.refresh(asset)
    await update_asset_after_minio_sync_with_parse(
        db,
        asset_id=asset.id,
        merged_meta_json=merged_meta,
        minio_path=minio_path,
        file_format=fmt,
    )
    return minio_path


async def run_batch_job(job_id: str) -> None:
    _scheduled_batch_jobs.discard(job_id)
    if job_id in _active_batch_jobs:
        logger.warning("sync_batch: 任务已在执行，忽略重复启动 job_id=%s", job_id)
        return
    _active_batch_jobs.add(job_id)

    try:
        async with DataAssetsSessionLocal() as db:
            stmt = select(SyncBatchJob).where(SyncBatchJob.job_id == job_id)
            res = await db.execute(stmt)
            job = res.scalar_one_or_none()
            if not job:
                logger.error("sync_batch: 任务不存在 job_id=%s", job_id)
                return
            if is_cancelled(job_id):
                job.status = "canceled"
                job.current_step = "已取消"
                job.updated_at = _now()
                await db.commit()
                return

            stmt_items = (
                select(SyncBatchJobItem)
                .where(SyncBatchJobItem.job_id == job_id)
                .order_by(SyncBatchJobItem.sort_order)
            )
            res_items = await db.execute(stmt_items)
            items: List[SyncBatchJobItem] = list(res_items.scalars().all())

            async with MainSessionLocal() as main_db:
                user = await get_user_by_id(main_db, str(job.user_id))
            if not user:
                job.status = "failed"
                job.error_message = "用户不存在，任务中止"
                job.updated_at = _now()
                await db.commit()
                return

            from app.api.routes_data_assets import _ensure_asset_visible

            # 仅允许 queued/running 状态进入执行，避免误执行终态任务
            if (job.status or "").strip().lower() not in ("queued", "running"):
                logger.info("sync_batch: 跳过非可执行状态任务 job_id=%s status=%s", job_id, job.status)
                return

            job.status = "running"
            job.current_step = "开始处理"
            job.progress_percent = 0.0
            job.updated_at = _now()
            await db.commit()

            succeeded = sum(1 for i in items if i.status in ("succeeded", "skipped"))
            failed = sum(1 for i in items if i.status == "failed")
            total = max(1, len(items))

            async with AsyncSessionLocal() as device_db:
                for idx, item in enumerate(items):
                    if is_cancelled(job_id):
                        print(f"[Cancel] Task {job_id} cancelled")
                        job.status = "canceled"
                        job.current_step = "已取消"
                        job.updated_at = _now()
                        await db.commit()
                        return
                    if item.status in ("succeeded", "failed", "skipped"):
                        continue
                    asset = await get_asset_by_id(db, item.asset_id)
                    job.current_step = f"同步资产 id={item.asset_id} ({idx + 1}/{total})"
                    job.progress_percent = min(99.0, (idx / total) * 100.0)
                    await db.commit()

                    item.status = "running"
                    item.started_at = _now()
                    item.error_message = None
                    await db.commit()

                    if not asset:
                        item.status = "failed"
                        item.error_message = "资产不存在"
                        item.finished_at = _now()
                        failed += 1
                        await db.commit()
                        continue

                    if not await _ensure_asset_visible(db, user, asset):
                        item.status = "failed"
                        item.error_message = "无权限或资产不可见"
                        item.finished_at = _now()
                        failed += 1
                        await db.commit()
                        continue

                    from app.api.routes_data_assets import _asset_is_synced

                    if _asset_is_synced(asset):
                        item.status = "skipped"
                        item.error_message = None
                        item.finished_at = _now()
                        succeeded += 1
                        job.succeeded = succeeded
                        job.failed = failed
                        await db.commit()
                        continue

                    try:
                        asset_lock = await _get_asset_sync_lock(int(item.asset_id))
                        async with asset_lock:
                            # 拿到同资产锁后再二次检查，避免并发任务重复同步同一资产
                            fresh_asset = await get_asset_by_id(db, item.asset_id)
                            if fresh_asset is None:
                                raise RuntimeError("资产不存在")
                            if _asset_is_synced(fresh_asset):
                                item.status = "skipped"
                                item.error_message = None
                                item.finished_at = _now()
                                succeeded += 1
                                job.succeeded = succeeded
                                job.failed = failed
                                await db.commit()
                                continue
                            if (getattr(fresh_asset, "sync_status", "") or "").strip().lower() == "syncing":
                                item.status = "skipped"
                                item.error_message = "已在同步中"
                                item.finished_at = _now()
                                succeeded += 1
                                job.succeeded = succeeded
                                job.failed = failed
                                await db.commit()
                                continue
                            if not await try_mark_asset_syncing(db, fresh_asset.id):
                                await db.refresh(fresh_asset)
                                if _asset_is_synced(fresh_asset):
                                    item.status = "skipped"
                                    item.error_message = None
                                else:
                                    item.status = "skipped"
                                    item.error_message = "已在同步中"
                                item.finished_at = _now()
                                succeeded += 1
                                job.succeeded = succeeded
                                job.failed = failed
                                await db.commit()
                                continue

                            meta_json = getattr(fresh_asset, "meta", None)
                            cid = effective_collect_device_id_for_sync(
                                meta_json,
                                (getattr(fresh_asset, "device_id", None) or "").strip() or None,
                            )
                            agent_hint = (job.agent_id_query or "").strip() or None
                            # 按采集端 agent_id（hardware_uuid）分桶，与隧道同步一致；不再解析 HTTP base_url（避免多台设备均有 IP 时歧义）
                            agent_bucket_key = await resolve_sync_agent_id(
                                device_db,
                                agent_id=agent_hint,
                                meta_json=meta_json,
                                collect_device_id=cid,
                            )

                            async with _get_global_sem():
                                async with _get_agent_sem(agent_bucket_key):
                                    minio_path = await _sync_one_asset(db, device_db, fresh_asset, agent_hint, job_id)

                        item.status = "succeeded"
                        item.minio_path = minio_path or None
                        item.error_message = None
                        item.finished_at = _now()
                        succeeded += 1
                    except Exception as e:
                        err = str(e)[:800]
                        if is_cancelled(job_id) or "已取消" in err:
                            item.status = "canceled"
                            item.error_message = "已取消"
                            item.finished_at = _now()
                            try:
                                await update_asset(db, item.asset_id, sync_status="failed", sync_error="已取消")
                            except Exception:
                                await db.rollback()
                            job.status = "canceled"
                            job.current_step = "已取消"
                            job.updated_at = _now()
                            await db.commit()
                            return
                        logger.exception("sync_batch: 单条失败 job_id=%s asset_id=%s", job_id, item.asset_id)
                        item.status = "failed"
                        item.error_message = err
                        item.finished_at = _now()
                        failed += 1
                        try:
                            await update_asset(db, item.asset_id, sync_status="failed", sync_error=err[:400])
                        except Exception:
                            await db.rollback()
                            asset2 = await get_asset_by_id(db, item.asset_id)
                            if asset2:
                                await update_asset(db, item.asset_id, sync_status="failed", sync_error=err[:400])

                    job.succeeded = succeeded
                    job.failed = failed
                    job.progress_percent = min(100.0, ((idx + 1) / total) * 100.0)
                    await db.commit()

            job.progress_percent = 100.0
            job.current_step = "已完成"
            job.status = "succeeded" if failed == 0 else "failed"
            first_failed_msg = next(
                (
                    (it.error_message or "").strip()
                    for it in items
                    if (it.status or "").strip().lower() == "failed" and (it.error_message or "").strip()
                ),
                "",
            )
            if failed and succeeded == 0:
                job.error_message = f"全部失败：{first_failed_msg}" if first_failed_msg else "全部失败"
            elif failed:
                suffix = f"，原因：{first_failed_msg}" if first_failed_msg else ""
                job.error_message = f"部分失败：成功 {succeeded}，失败 {failed}{suffix}"
            else:
                job.error_message = None
            job.updated_at = _now()
            await db.commit()
            logger.info(
                "sync_batch: 结束 job_id=%s status=%s ok=%s fail=%s",
                job_id,
                job.status,
                succeeded,
                failed,
            )

    except Exception as e:
        logger.exception("sync_batch: 任务异常终止 job_id=%s err=%s", job_id, e)
        try:
            async with DataAssetsSessionLocal() as db2:
                stmt = select(SyncBatchJob).where(SyncBatchJob.job_id == job_id)
                res = await db2.execute(stmt)
                j2 = res.scalar_one_or_none()
                if j2:
                    j2.status = "failed"
                    j2.error_message = str(e)[:800]
                    j2.updated_at = _now()
                    await db2.commit()
        except Exception:
            pass
    finally:
        _active_batch_jobs.discard(job_id)


def schedule_batch_job(job_id: str, *, user_id: str | None = None) -> None:
    if job_id in _scheduled_batch_jobs or job_id in _active_batch_jobs:
        return
    _scheduled_batch_jobs.add(job_id)
    dispatch_task({
        "type": "batch",
        "task_id": job_id,
        "job_id": job_id,
        "user_id": user_id,
    })


async def build_job_status_payload(db: AsyncSession, job_id: str, user: User) -> Optional[Dict]:
    from app.core.roles import is_super_admin_or_team_admin

    stmt = select(SyncBatchJob).where(SyncBatchJob.job_id == job_id)
    res = await db.execute(stmt)
    job = res.scalar_one_or_none()
    if not job:
        return None
    if str(job.user_id) != str(user.id) and not is_super_admin_or_team_admin(user.role):
        return None

    stmt_items = (
        select(SyncBatchJobItem).where(SyncBatchJobItem.job_id == job_id).order_by(SyncBatchJobItem.sort_order)
    )
    res_items = await db.execute(stmt_items)
    items = list(res_items.scalars().all())
    return {
        "jobId": job.job_id,
        "status": job.status,
        "total": job.total,
        "succeeded": job.succeeded,
        "failed": job.failed,
        "progress": job.progress_percent,
        "currentStep": job.current_step or "",
        "errorMessage": job.error_message or "",
        "agentId": job.agent_id_query or None,
        "items": [
            {
                "assetId": it.asset_id,
                "status": it.status,
                "errorMessage": it.error_message or "",
                "minioPath": it.minio_path or "",
            }
            for it in items
        ],
    }


async def recover_sync_batch_jobs_on_startup() -> int:
    """
    进程启动补偿：
    - queued：直接重新调度
    - running：回滚为 queued 后重新调度（上次进程中断）
    返回本次重调度数量。
    """
    recovered = 0
    async with DataAssetsSessionLocal() as db:
        stmt = select(SyncBatchJob).where(SyncBatchJob.status.in_(["queued", "running"]))
        res = await db.execute(stmt)
        jobs = list(res.scalars().all())

        for job in jobs:
            status = (job.status or "").strip().lower()
            if status == "running":
                job.status = "queued"
                if not (job.current_step or "").strip():
                    job.current_step = "等待恢复调度"
                job.updated_at = _now()
        if jobs:
            await db.commit()

    for job in jobs:
        schedule_batch_job(job.job_id, user_id=str(getattr(job, "user_id", "") or "") or None)
        recovered += 1
    return recovered
