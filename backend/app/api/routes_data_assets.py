"""
数据资产 API：真实读取本地 HDF5/MCAP/LeRobot，资产元数据存 PostgreSQL
"""
import csv
import logging
import io
import os
import json
import re
import asyncio
import threading
import shutil
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, UploadFile, File, Form, Body, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, text, update

from app.db.data_assets_session import DATA_ASSETS_ROOT, DataAssetsSessionLocal, get_data_assets_db
from app.services.asset_registration_service import data_assets_sync_engine
from app.db.session import get_db as get_main_db
from app.api.routes_fs import FS_LIST_WHITELIST, validate_path_whitelist
from app.models.data_asset import CollectionTaskAsset, DataAsset, DataAssetUploadSession, SyncBatchJob, SyncBatchJobItem
from app.models.project_asset import Project
from app.crud.data_asset import (
    get_assets,
    get_asset_by_id,
    get_assets_by_ids,
    get_asset_by_file_path,
    create_asset,
    delete_asset,
    next_code,
    update_asset,
    try_mark_asset_syncing,
)
from app.schemas.data_asset import (
    DataAssetQueryParams,
    DataAssetListResponse,
    DataAssetResponse,
    DataAssetCreate,
    LocalFileItem,
    RegisterAssetRequest,
    ExportRequest,
    DeleteExportResultRequest,
    SyncBatchCreateBody,
    ReparseFromMinioBatchBody,
    DeleteAssetsBatchBody,
    DirectUploadInitBody,
    DirectUploadCompleteBody,
)
from app.schemas.common import ApiResponse
from app.services.asset_meta_parser import parse_meta_for_asset
from app.core.deps import get_current_user
from app.models.user import User
# refresh_token cookie 已禁用（多 session 并行）；保留 import 会造成未使用
from app.crud.user import get_user_by_account_id as get_user_by_account_id_async
from app.core.roles import is_super_admin, is_super_admin_or_team_admin
from app.core.data_asset_access import (
    assert_may_write_project_for_data_asset_import,
    data_assets_allowed_project_ids,
    data_asset_visible_to_user,
    user_cannot_delete_data_asset,
)
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.security import decode_token, require_token_type
from app.core.database import SessionLocal
from jose import JWTError
from app.services.user_service import get_user_by_account_id
from app.services.user_team_access import (
    user_blocked_by_all_teams_inactive_async,
    user_blocked_by_all_teams_inactive_sync,
)
from app.services.minio_service import (
    upload_file_to_project_bucket,
    upload_dir_to_project_bucket,
    download_by_minio_uri,
    delete_by_minio_uri,
    project_bucket_name,
    ensure_project_bucket,
    generate_presigned_put_url,
    presigned_put_many,
    stat_object,
    build_minio_uri,
    build_minio_prefix_uri,
    normalize_relative_path,
    MinioBucketError,
    MinioConfigError,
)
from app.services.episode_storage import EpisodeStorage
from app.services.storage_resolver import EpisodeResolveError
from app.services.storage_meta_merge import merge_storage_meta
from app.services.agent_data_sync_proxy import sync_asset_via_agent
from app.crud.device import get_device_by_id
from app.services.agent_registry import agent_registry
from app.services.agent_tunnel_manager import agent_tunnel_manager
from app.services.post_sync_asset_parse import (
    batch_refresh_parse_from_minio,
    refresh_parse_from_minio_for_asset,
    update_asset_after_minio_sync_with_parse,
)
from app.services.sync_batch_service import (
    acquire_asset_sync_lock,
    build_job_status_payload,
    schedule_batch_job,
    _filename_from_minio_uri,
)
from app.services.collect_disk_reconcile import build_collect_disk_presence_map
from app.services.collection_job_reconcile import (
    decrement_collection_job_completed_for_removed_episode,
    extract_collect_job_id_from_asset,
    reconcile_collection_job_progress_from_data_assets,
)
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import enqueue_audit_log
from app.services.task_job_store import (
    create_task_job,
    delete_task_job,
    get_task_job,
    is_cancelled,
    update_task_status,
)
from app.services.data_asset_path_resolver import (
    resolve_local_path_from_fields,
    minio_uri_from_fields,
)
from app.services.cache_cleanup_service import clear_data_disk_caches, clear_project_cache

# 复用标注模块的 HDF5/MCAP 读帧能力
from app.api.routes_label import (  # type: ignore
    get_task_config_service,
    get_hdf5_service,
    get_mcap_service,
    _file_format,
    _ws_playback_resolve_file,
)

router = APIRouter()
logger = logging.getLogger(__name__)


async def _resolve_asset_local_base_path(asset: DataAsset) -> str:
    """解析资产在本机可读根路径（含从 MinIO 拉取到 .minio_view_cache）。"""
    return await asyncio.to_thread(
        resolve_local_path_from_fields,
        getattr(asset, "file_path", None),
        getattr(asset, "meta", None),
    )


def _data_asset_response_for_viewer(asset: DataAsset, viewer_username: Optional[str] = None) -> DataAssetResponse:
    """序列化资产；operator_name 仅来自资产持久化字段，不回落当前会话用户。"""
    row = DataAssetResponse.model_validate(asset)
    wh = minio_uri_from_fields(getattr(asset, "file_path", None), getattr(asset, "meta", None))
    updates: Dict[str, Any] = {}
    fmt = (getattr(asset, "format", None) or "").strip().lower()
    filename = (getattr(asset, "filename", None) or "").strip()
    # 历史兼容：仅对 mcap 显示名补后缀，避免影响目录型资产。
    if fmt == "mcap" and filename and not filename.lower().endswith(".mcap"):
        updates["filename"] = f"{filename}.mcap"
    if wh:
        updates["warehouse_uri"] = wh
        sync_status = (getattr(asset, "sync_status", None) or "").strip().lower()
        if sync_status != "synced":
            updates["sync_status"] = "synced"
            updates["sync_error"] = None
    stored = getattr(asset, "created_by_username", None) or getattr(asset, "operator_name", None)
    if isinstance(stored, str) and stored.strip():
        updates["operator_name"] = stored.strip()
    if updates:
        return row.model_copy(update=updates)
    return row


async def _resolve_collect_task_display_name(db: AsyncSession, asset: DataAsset) -> Optional[str]:
    """采集资产：优先使用 collect_task_name 列；为空时根据 meta.collect.task_id 查采集任务名称（兼容历史数据）。"""
    col = (getattr(asset, "collect_task_name", None) or "").strip()
    if col:
        return col
    src = (getattr(asset, "source", None) or "").strip().lower()
    if src != "collect":
        return None
    meta_raw = getattr(asset, "meta", None)
    if not meta_raw:
        return None
    try:
        obj = json.loads(meta_raw)
        c = obj.get("collect") if isinstance(obj, dict) else None
        tid = ""
        if isinstance(c, dict):
            tid = str(c.get("task_id") or "").strip()
        if not tid:
            return None
        ct = await db.get(CollectionTaskAsset, tid)
        if ct is None:
            return None
        nm = (getattr(ct, "name", None) or "").strip()
        return nm or None
    except Exception:
        return None


async def _project_display_names_for_assets(
    db: AsyncSession, assets: List[DataAsset]
) -> Dict[str, str]:
    """按 project_id 批量读取 projects.name，用于 API 展示与冗余 project_name 纠偏。"""
    pids: set[str] = set()
    for a in assets:
        pid = (getattr(a, "project_id", None) or "").strip()
        if pid:
            pids.add(pid)
    if not pids:
        return {}
    r = await db.execute(select(Project.id, Project.name).where(Project.id.in_(pids)))
    out: Dict[str, str] = {}
    for pid, pname in r.all():
        if pname and str(pname).strip():
            out[str(pid)] = str(pname).strip()
    return out


def _data_asset_response_with_resolved_project(
    asset: DataAsset,
    viewer_username: Optional[str],
    project_name_by_id: Dict[str, str],
) -> DataAssetResponse:
    row = _data_asset_response_for_viewer(asset, viewer_username)
    pid = (getattr(asset, "project_id", None) or "").strip()
    canon = project_name_by_id.get(pid) if pid else None
    if canon:
        return row.model_copy(update={"project_name": canon})
    return row


def _asset_is_synced(asset: Optional[DataAsset]) -> bool:
    if asset is None:
        return False
    return (getattr(asset, "sync_status", "synced") or "synced").strip().lower() == "synced"


async def _ensure_asset_visible(db: AsyncSession, user: User, asset: Optional[DataAsset]) -> bool:
    """
    兼容旧调用：批量同步服务会从本模块导入该函数做可见性校验。
    """
    return await data_asset_visible_to_user(db, user, asset)


# 兼容历史导入名（无下划线）
ensure_asset_visible = _ensure_asset_visible


def _raise_if_unsynced(asset: Optional[DataAsset]) -> None:
    if not _asset_is_synced(asset):
        raise HTTPException(status_code=409, detail="该数据尚未同步，暂不可操作")


def _get_current_user_for_ws(websocket: WebSocket) -> Optional[User]:
    auth = websocket.headers.get("authorization")
    token: Optional[str] = None
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token:
        token = websocket.query_params.get("token")
    if not token:
        return None
    try:
        payload = decode_token(token)
        require_token_type(payload, "access")
    except JWTError:
        return None
    account_id = payload.get("sub")
    if not account_id:
        return None
    db = SessionLocal()
    try:
        user = get_user_by_account_id(db, account_id)
        if user is None or not getattr(user, "is_active", True):
            return None
        if user_blocked_by_all_teams_inactive_sync(db, user):
            return None
        return user
    finally:
        db.close()


async def get_current_user_or_cookie(
    request: Request,
    db: AsyncSession = Depends(get_main_db),
) -> User:
    """
    帧图等只读接口用：优先 Bearer，其次 query token（GET 帧图时 Next 代理可能不转发 Authorization），最后 refresh_token Cookie。
    """
    auth_header = request.headers.get("Authorization")
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        token = request.query_params.get("token")
        if token:
            logger.info("get_current_user_or_cookie: using token from query")
    if token:
        try:
            payload = decode_token(token)
            require_token_type(payload, "access")
        except JWTError as e:
            logger.warning("get_current_user_or_cookie: JWT validation failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        account_id = payload.get("sub")
        if not account_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = await get_user_by_account_id_async(db, account_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not getattr(user, "is_active", True):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User disabled")
        if await user_blocked_by_all_teams_inactive_async(db, user):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Team disabled")
        await db.refresh(user)
        return user

    # 改造说明：不再允许 refresh_token Cookie 作为兜底登录态来源（避免多标签页串号）。
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing Authorization header or token query param",
        headers={"WWW-Authenticate": "Bearer"},
    )


# 任务下拉候选项：必须在 get_current_user_or_cookie 定义之后，否则 Depends 默认值会 NameError
@router.get("/task-options", response_model=ApiResponse)
async def list_task_options(
    project: str = Query(None),
    format: str = Query(None, alias="format"),
    source: str = Query(None, alias="source"),
    current_user: User = Depends(get_current_user_or_cookie),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """按筛选条件返回「任务」下拉候选项（按项目/来源/格式滚动缩小范围）。"""
    try:
        await asyncio.to_thread(_ensure_data_assets_task_columns_sync)
    except Exception:
        pass
    try:
        allowed_project_ids = await data_assets_allowed_project_ids(db, current_user)
        conditions = []
        if allowed_project_ids is not None:
            ids = [str(x).strip() for x in allowed_project_ids if str(x).strip()]
            if not ids:
                return ApiResponse(ok=True, data={"items": []})
            conditions.append(DataAsset.project_id.in_(ids))
        if project and project.strip():
            pv = project.strip()
            conditions.append((DataAsset.project_id == pv) | (DataAsset.project_name == pv))
        if format and format.strip():
            conditions.append(DataAsset.format == format.strip().lower())

        src = (source or "").strip().lower()
        if src:
            if src == "import":
                conditions.append(DataAsset.source.in_(["import", "本地", "local"]))
            else:
                conditions.append(DataAsset.source == src)

        if src == "label":
            stmt = select(text("DISTINCT label_task_name")).select_from(DataAsset).where(text("label_task_name IS NOT NULL AND label_task_name <> ''"))
            for c in conditions:
                stmt = stmt.where(c)
            rows = (await db.execute(stmt)).scalars().all()
            items = [{"value": str(x), "label": str(x)} for x in rows if x]
            return ApiResponse(ok=True, data={"items": items})
        if src == "collect":
            stmt = select(text("DISTINCT collect_task_name")).select_from(DataAsset).where(text("collect_task_name IS NOT NULL AND collect_task_name <> ''"))
            for c in conditions:
                stmt = stmt.where(c)
            rows = (await db.execute(stmt)).scalars().all()
            items = [{"value": str(x), "label": str(x)} for x in rows if x]
            return ApiResponse(ok=True, data={"items": items})
        if src == "convert":
            stmt = select(text("DISTINCT conversion_task_name")).select_from(DataAsset).where(text("conversion_task_name IS NOT NULL AND conversion_task_name <> ''"))
            for c in conditions:
                stmt = stmt.where(c)
            rows = (await db.execute(stmt)).scalars().all()
            items = [{"value": str(x), "label": str(x)} for x in rows if x]
            return ApiResponse(ok=True, data={"items": items})

        # 未限定来源：返回三类任务名（label 区分展示；value 带来源前缀，避免同名任务重复 key）
        def _q(col: str):
            stmt0 = select(text(f"DISTINCT {col}")).select_from(DataAsset).where(text(f"{col} IS NOT NULL AND {col} <> ''"))
            for c in conditions:
                stmt0 = stmt0.where(c)
            return stmt0

        out = []
        seen_values: set[str] = set()
        for col, tag, prefix in (
            ("label_task_name", "标注", "label"),
            ("collect_task_name", "采集", "collect"),
            ("conversion_task_name", "转换", "convert"),
        ):
            rows = (await db.execute(_q(col))).scalars().all()
            for x in rows:
                if not x:
                    continue
                name = str(x)
                value = f"{prefix}:{name}"
                if value in seen_values:
                    continue
                seen_values.add(value)
                out.append({"value": value, "label": f"{name}（{tag}）"})
        return ApiResponse(ok=True, data={"items": out})
    except Exception as e:
        return ApiResponse(ok=False, error=str(e)[:300])


# 导出任务（内存态）：第一版用轮询，不做任务中心/持久化
_EXPORT_JOB_TTL_SECONDS = 60 * 60  # 1h，允许成功后多次重新下载
_export_jobs: Dict[str, Dict] = {}
# 线程锁：导出在 worker 线程 asyncio.run 中跑，与 API 主 loop 不同，不能用 asyncio.Lock 保护 _export_jobs
_export_jobs_lock = threading.RLock()
_export_asset_locks: Dict[int, asyncio.Lock] = {}
_export_asset_locks_guard = asyncio.Lock()
_export_run_semaphore = asyncio.Semaphore(max(1, int(os.getenv("EXPORT_MAX_CONCURRENCY", "1"))))
_export_asset_workers = max(1, int(os.getenv("EXPORT_ASSET_WORKERS", "4")))
_LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc


def _datetime_to_iso_utc(dt: Optional[datetime]) -> str:
    """统一输出 ISO 8601 UTC（毫秒 + Z）。"""
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_LOCAL_TZ)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _datetime_to_epoch_ms(dt: Optional[datetime]) -> int:
    """统一输出 UTC 毫秒时间戳（13 位）。"""
    if not dt:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_LOCAL_TZ)
    dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1000)


def _export_percent_by_status(status: str) -> int:
    """仅用于无 progress_state 的旧路径（如无 outputPath 的 zip 下载）。"""
    return {
        "validating": 5,
        "collecting_files": 15,
        "collecting_annotations": 25,
        "generating_asset_list": 35,
        "packaging": 90,
        "writing": 90,
        "ready": 100,
        "failed": 0,
    }.get(status, 0)


def _export_step_title(status: str) -> str:
    return {
        "validating": "校验导出项",
        "collecting_files": "整理原始数据",
        "collecting_annotations": "整理标注文件",
        "generating_asset_list": "生成资产清单",
        "packaging": "打包压缩文件",
        "writing": "写入导出目录",
        "ready": "导出完成",
        "ready_zip": "导出完成，可下载压缩包",
        "failed": "导出失败",
    }.get(status, "正在处理")


def _normalize_compression_mode(mode: Optional[str]) -> str:
    raw = (mode or os.getenv("EXPORT_ZIP_COMPRESSION", "deflated")).strip().lower()
    return "store" if raw == "store" else "deflated"


def _zipfile_params_for_mode(mode: Optional[str]) -> Dict[str, Any]:
    normalized = _normalize_compression_mode(mode)
    if normalized == "store":
        return {"compression": zipfile.ZIP_STORED}
    compress_level = max(1, min(9, int(os.getenv("EXPORT_ZIP_COMPRESSLEVEL", "1"))))
    return {"compression": zipfile.ZIP_DEFLATED, "compresslevel": compress_level}


async def _cleanup_expired_export_jobs() -> None:
    """清理过期任务（ready/failed 且超过 TTL）。"""
    now_ms = int(time.time() * 1000)
    with _export_jobs_lock:
        expired = []
        for job_id, job in _export_jobs.items():
            created_at = int(job.get("createdAtTs") or 0)
            if not created_at:
                continue
            if now_ms - created_at <= _EXPORT_JOB_TTL_SECONDS * 1000:
                continue
            # 仅清理终态
            if job.get("status") not in ("ready", "failed", "cancelled"):
                continue
            expired.append(job_id)
        for job_id in expired:
            job = _export_jobs.pop(job_id, None)
            try:
                tmp_dir = job.get("tmpDir") if job else None
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


async def _set_export_job(job_id: str, patch: Dict) -> None:
    with _export_jobs_lock:
        if job_id not in _export_jobs:
            return
        _export_jobs[job_id].update(patch)


async def _get_export_asset_locks(asset_ids: List[int]) -> List[asyncio.Lock]:
    ordered = sorted({int(aid) for aid in asset_ids})
    locks: List[asyncio.Lock] = []
    async with _export_asset_locks_guard:
        for aid in ordered:
            lock = _export_asset_locks.get(aid)
            if lock is None:
                lock = asyncio.Lock()
                _export_asset_locks[aid] = lock
            locks.append(lock)
    return locks


def _export_count_work_units(assets: List[DataAsset], fmt: str) -> int:
    """按工作量单位计算导出总步数，用于半真实进度。"""
    n = len(assets)
    if n == 0:
        return 1
    ann_count = sum(1 for a in assets if _platform_annotation_path(a))
    if fmt == "hdf5" or fmt == "mcap":
        return n + ann_count + 2
    if fmt == "lerobot" or fmt == "directory":
        return 5 * n + ann_count + 1
    return n + ann_count + 2


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    try:
        if src.stat().st_dev == dst.parent.stat().st_dev:
            os.link(src, dst)
            return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _copytree_fast(src: Path, dst: Path) -> None:
    def _copy_fn(s: str, d: str) -> str:
        _link_or_copy_file(Path(s), Path(d))
        return d

    shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=_copy_fn)


def _prefetch_export_minio_assets(
    assets: List[DataAsset],
    minio_cache_root: Path,
    minio_download_cache: Dict[str, str],
) -> None:
    uris = sorted({
        u for u in (_minio_uri_for_storage_ops(a) for a in assets)
        if isinstance(u, str) and u.strip()
    })
    if not uris:
        return
    workers = min(_export_asset_workers, max(1, len(uris)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(download_by_minio_uri, uri, str(minio_cache_root)): uri
            for uri in uris
            if not (minio_download_cache.get(uri) and os.path.exists(str(minio_download_cache.get(uri))))
        }
        for future in as_completed(future_map):
            uri = future_map[future]
            local_path = future.result()
            minio_download_cache[uri] = local_path


def _build_export_tree(
    assets: List[DataAsset],
    export_root: Path,
    progress_ctx: Optional[Dict[str, Any]] = None,
    cancel_checker: Optional[Any] = None,
) -> None:
    """
    在 export_root 下按既定结构创建导出目录：hdf5/mcap/lerobot、annotations（如有）、asset_list.xlsx。
    progress_ctx 为可变的 dict：{"state": [progress_pct, current_step], "total_units": int}，仅对 state 写。
    """
    if not assets:
        raise ValueError("资产列表为空")
    export_root.mkdir(parents=True, exist_ok=True)
    annotation_entries: List[Tuple[str, str]] = []
    fmt = _normalize_format(assets[0].format, assets[0].file_path or "")
    minio_cache_root = export_root.parent / "_minio_export_cache"
    minio_download_cache: Dict[str, str] = {}
    progress_lock = threading.Lock()
    total = progress_ctx["total_units"] if progress_ctx else 0
    state = progress_ctx["state"] if progress_ctx and isinstance(progress_ctx.get("state"), list) else []
    done = [0]
    asset_progress_mode = bool(progress_ctx and progress_ctx.get("asset_progress_mode"))
    assets_done_ref: Optional[List[int]] = (
        progress_ctx.get("assets_done") if progress_ctx and isinstance(progress_ctx.get("assets_done"), list) else None
    )
    assets_total_n = int(progress_ctx.get("assets_total") or 0) if progress_ctx else 0

    def report(step: str) -> None:
        if callable(cancel_checker) and cancel_checker():
            raise RuntimeError("Task cancelled")
        if asset_progress_mode:
            return
        if progress_ctx and total > 0 and len(state) >= 2:
            with progress_lock:
                done[0] += 1
                state[0] = min(100, int(100 * done[0] / total))
                state[1] = step

    def bump_asset_export(step: str) -> None:
        if callable(cancel_checker) and cancel_checker():
            raise RuntimeError("Task cancelled")
        if not asset_progress_mode or assets_done_ref is None or len(state) < 2:
            return
        with progress_lock:
            assets_done_ref[0] += 1
            d = assets_done_ref[0]
            if assets_total_n > 0:
                state[0] = min(99, int(100 * d / assets_total_n))
                state[1] = f"{d} / {assets_total_n} · {step}"

    def set_post_asset_phase(step: str) -> None:
        if callable(cancel_checker) and cancel_checker():
            raise RuntimeError("Task cancelled")
        if not asset_progress_mode or len(state) < 2 or assets_total_n <= 0:
            return
        with progress_lock:
            d = assets_done_ref[0] if assets_done_ref else assets_total_n
            state[0] = min(99, int(100 * d / assets_total_n))
            state[1] = f"{d} / {assets_total_n} · {step}"

    _prefetch_export_minio_assets(assets, minio_cache_root, minio_download_cache)

    if fmt == "hdf5":
        data_dir = export_root / "hdf5"
        data_dir.mkdir(parents=True, exist_ok=True)
        def _worker_hdf5(a: DataAsset) -> Optional[Tuple[str, str]]:
            local_path = _resolve_export_local_path(a, minio_cache_root, minio_download_cache)
            src_file = _resolve_export_single_file_path(local_path, a.filename, "hdf5")
            dest_name = f"{a.code}_{a.filename}"
            _link_or_copy_file(Path(src_file), data_dir / dest_name)
            ann = _export_annotation_src(a, src_file)
            return (a.code, ann) if ann else None

        with ThreadPoolExecutor(max_workers=min(_export_asset_workers, max(1, len(assets)))) as pool:
            futures = [pool.submit(_worker_hdf5, a) for a in assets]
            for f in as_completed(futures):
                item = f.result()
                report("整理原始数据")
                bump_asset_export("整理原始数据")
                if item:
                    annotation_entries.append(item)
    elif fmt == "mcap":
        data_dir = export_root / "mcap"
        data_dir.mkdir(parents=True, exist_ok=True)
        def _worker_mcap(a: DataAsset) -> Optional[Tuple[str, str]]:
            local_path = _resolve_export_local_path(a, minio_cache_root, minio_download_cache)
            src_file = _resolve_export_single_file_path(local_path, a.filename, "mcap")
            dest_name = f"{a.code}_{a.filename}"
            _link_or_copy_file(Path(src_file), data_dir / dest_name)
            ann = _export_annotation_src(a, src_file)
            return (a.code, ann) if ann else None

        with ThreadPoolExecutor(max_workers=min(_export_asset_workers, max(1, len(assets)))) as pool:
            futures = [pool.submit(_worker_mcap, a) for a in assets]
            for f in as_completed(futures):
                item = f.result()
                report("整理原始数据")
                bump_asset_export("整理原始数据")
                if item:
                    annotation_entries.append(item)
    elif fmt == "directory":
        bundle = export_root / "directory"
        bundle.mkdir(parents=True, exist_ok=True)
        def _worker_directory(a: DataAsset) -> Optional[Tuple[str, str]]:
            local_path = _resolve_export_local_path(a, minio_cache_root, minio_download_cache)
            root = Path(local_path or "")
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError(f"未找到原始目录: {a.filename}")
            safe_name = re.sub(r"[^\w.\-+\u4e00-\u9fff]", "_", (a.filename or "dataset").strip()) or "dataset"
            dest = bundle / f"{a.code}_{safe_name}"
            _copytree_fast(root, dest)
            ann = _export_annotation_src(a, local_path)
            return (a.code, ann) if ann else None

        with ThreadPoolExecutor(max_workers=min(_export_asset_workers, max(1, len(assets)))) as pool:
            futures = [pool.submit(_worker_directory, a) for a in assets]
            for f in as_completed(futures):
                item = f.result()
                report("复制目录资产")
                bump_asset_export("复制目录资产")
                if item:
                    annotation_entries.append(item)
    else:
        lerobot_root = export_root / "lerobot"
        conversion_meta_dir = lerobot_root / "conversion_metadata"
        data_dir = lerobot_root / "data"
        labels_dir = lerobot_root / "labels"
        meta_dir = lerobot_root / "meta"
        videos_dir = lerobot_root / "videos"
        conversion_meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)
        def _worker_lerobot(a: DataAsset) -> Optional[Tuple[str, str]]:
            local_path = _resolve_export_local_path(a, minio_cache_root, minio_download_cache)
            root = Path(local_path or "")
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError(f"未找到原始目录: {a.filename}")
            code = a.code
            cm = root / "conversion_metadata.json"
            if cm.is_file():
                _link_or_copy_file(cm, conversion_meta_dir / f"{code}_conversion_metadata.json")
            report("conversion_metadata")
            lb = root / "labels.json"
            if lb.is_file():
                _link_or_copy_file(lb, labels_dir / f"{code}_labels.json")
            report("labels")
            d = root / "data"
            if d.is_dir():
                _copytree_fast(d, data_dir / code)
            report("data")
            m = root / "meta"
            if m.is_dir():
                _copytree_fast(m, meta_dir / code)
            report("meta")
            v = root / "videos"
            if v.is_dir():
                _copytree_fast(v, videos_dir / code)
            report("videos")
            ann = _export_annotation_src(a, local_path)
            return (a.code, ann) if ann else None

        with ThreadPoolExecutor(max_workers=min(_export_asset_workers, max(1, len(assets)))) as pool:
            futures = [pool.submit(_worker_lerobot, a) for a in assets]
            for f in as_completed(futures):
                item = f.result()
                bump_asset_export("整理 LeRobot 数据")
                if item:
                    annotation_entries.append(item)

    if annotation_entries:
        set_post_asset_phase("整理标注文件")
        ann_dir = export_root / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        for code, src_path in annotation_entries:
            base = os.path.basename(src_path)
            dest_name = f"{code}_{base}"
            _link_or_copy_file(Path(src_path), ann_dir / dest_name)
            report("整理标注文件")

    try:
        from openpyxl import Workbook
    except ImportError:
        raise RuntimeError("服务端未安装 openpyxl，无法生成 asset_list.xlsx")

    wb = Workbook()
    ws = wb.active
    ws.title = "资产清单"
    headers = ["编号", "文件名", "数据格式", "来源", "上传时间", "所属项目", "文件大小", "标注文件", "转换记录"]
    ws.append(headers)
    for a in assets:
        ann_path = _platform_annotation_path(a)
        ann_display = f"{a.code}_{os.path.basename(ann_path)}" if ann_path else "无"
        created = a.created_at.strftime("%Y-%m-%d %H:%M:%S") if a.created_at else ""
        source_display = {"import": "导入", "collect": "采集", "label": "标注", "convert": "转换"}.get(
            (a.source or "").lower(), "导入"
        )
        format_display = (a.format or "").upper() if a.format else "—"
        ws.append([
            a.code,
            a.filename,
            format_display,
            source_display,
            created,
            a.project_name or a.project_id or "—",
            _file_size_display(a.file_size_bytes),
            ann_display,
            _conversion_record_display(a),
        ])
    set_post_asset_phase("生成资产清单")
    wb.save(export_root / "asset_list.xlsx")
    if progress_ctx:
        report("生成资产清单")


def _build_export_zip_to_file(
    assets: List[DataAsset],
    zip_path: str,
    progress_state: Optional[List[Any]] = None,
    compression_mode: Optional[str] = None,
    cancel_checker: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    构建导出 zip 写入到 zip_path。
    返回 (zip_filename, export_root_name)。
    progress_state 可选：[pct, current_step]，在 asset_progress_mode 下按「已完成资产数/总数」更新。
    """
    if not assets:
        raise ValueError("资产列表为空")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_root_name = f"export_{ts}"
    zip_filename = f"export_{ts}.zip"
    n = len(assets)

    tmp_dir = tempfile.mkdtemp(prefix="data_assets_export_build_")
    try:
        export_root = Path(tmp_dir) / export_root_name
        progress_ctx = None
        if progress_state is not None:
            if len(progress_state) < 2:
                progress_state.extend([0, f"0 / {n} · 准备"])
            progress_state[0] = 0
            progress_state[1] = f"0 / {n} · 准备"
            progress_ctx = {
                "state": progress_state,
                "total_units": 1,
                "asset_progress_mode": True,
                "assets_total": n,
                "assets_done": [0],
            }
        _build_export_tree(assets, export_root, progress_ctx, cancel_checker=cancel_checker)

        if progress_state is not None and len(progress_state) >= 2:
            progress_state[0] = 95
            progress_state[1] = f"{n} / {n} · 打包压缩文件"
        if callable(cancel_checker) and cancel_checker():
            raise RuntimeError("Task cancelled")

        Path(zip_path).parent.mkdir(parents=True, exist_ok=True)
        zip_kwargs = _zipfile_params_for_mode(compression_mode)
        with zipfile.ZipFile(zip_path, "w", **zip_kwargs) as zf:
            for item in export_root.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(export_root.parent)
                    zf.write(item, arcname.as_posix())

        if progress_state is not None and len(progress_state) >= 2:
            progress_state[0] = 100
            progress_state[1] = f"{n} / {n} · 完成"

        return zip_filename, export_root_name
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def build_export_to_directory(
    assets: List[DataAsset],
    output_base_path: str,
    progress_state: Optional[List] = None,
    compression_mode: Optional[str] = None,
    cancel_checker: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    将导出结果写入指定输出路径：
    - HDF5 / MCAP：按既定结构在临时目录下生成内容，然后整体打包为 export_yyyyMMdd_HHmmss.zip，写入 output_base_path；
      返回 (zip_filename, full_zip_path)。
    - LeRobot：按既定结构直接在 output_base_path/export_yyyyMMdd_HHmmss 下生成目录结构；
      返回 (export_dir_name, full_dir_path)。
    """
    if not assets:
        raise ValueError("资产列表为空")
    output_base_path = (output_base_path or "").strip()
    if not output_base_path:
        raise ValueError("输出路径不能为空")
    try:
        resolved = validate_path_whitelist(output_base_path)
    except HTTPException as e:
        raise ValueError(e.detail if isinstance(e.detail, str) else str(e.detail))
    if not os.path.isdir(resolved):
        raise ValueError("输出路径不是目录")

    fmt = _normalize_format(assets[0].format, assets[0].file_path or "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_root_name = f"export_{ts}"
    total_units = _export_count_work_units(assets, fmt)
    if progress_state is not None and len(progress_state) < 2:
        progress_state.extend([0, "准备中"])
    progress_ctx = (
        {"state": progress_state, "total_units": total_units}
        if progress_state is not None
        else None
    )

    if fmt == "lerobot" or fmt == "directory":
        export_root = Path(resolved) / export_root_name
        _build_export_tree(assets, export_root, progress_ctx, cancel_checker=cancel_checker)
        return export_root_name, str(export_root)

    tmp_dir = tempfile.mkdtemp(prefix="data_assets_export_build_to_dir_")
    try:
        export_root = Path(tmp_dir) / export_root_name
        _build_export_tree(assets, export_root, progress_ctx, cancel_checker=cancel_checker)

        if progress_state is not None and len(progress_state) >= 2:
            progress_state[0] = 95
            progress_state[1] = "打包压缩文件"
        zip_filename = f"{export_root_name}.zip"
        dest_zip_path = Path(resolved) / zip_filename
        Path(dest_zip_path).parent.mkdir(parents=True, exist_ok=True)
        zip_kwargs = _zipfile_params_for_mode(compression_mode)
        with zipfile.ZipFile(dest_zip_path, "w", **zip_kwargs) as zf:
            for item in export_root.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(export_root.parent)
                    zf.write(item, arcname.as_posix())
        if progress_state is not None and len(progress_state) >= 2:
            progress_state[0] = 100
            progress_state[1] = "导出内容已保存到指定路径"
        return zip_filename, str(dest_zip_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _run_export_job(job_id: str, asset_ids: List[int]) -> None:
    """后台执行导出任务，更新内存状态。若 job 含 outputPath 则写入该目录；否则走 zip 落盘（旧逻辑）。"""
    output_path: Optional[str] = None
    compression_mode: Optional[str] = None
    with _export_jobs_lock:
        job = _export_jobs.get(job_id) or {}
        output_path = job.get("outputPath")
        compression_mode = _normalize_compression_mode(job.get("compressionMode"))

    try:
        if is_cancelled(job_id):
            print(f"[Cancel] Task {job_id} cancelled")
            await _set_export_job(
                job_id,
                {
                    "status": "cancelled",
                    "progress": 0,
                    "currentStep": "已取消",
                    "errorMessage": "任务已取消",
                },
            )
            return
        print(f"[Export] {job_id} 开始：加载资产 {len(asset_ids)} 条", flush=True)
        await _set_export_job(job_id, {
            "status": "validating",
            "progress": 5,
            "currentStep": _export_step_title("validating"),
            "errorMessage": "",
        })
        async with _export_run_semaphore:
            locks = await _get_export_asset_locks(asset_ids)
            async with AsyncExitStack() as stack:
                for lock in locks:
                    await stack.enter_async_context(lock)

                print(f"[Export] {job_id} 已获取执行槽与资产锁，查询数据库", flush=True)
                async with DataAssetsSessionLocal() as db:
                    assets = await get_assets_by_ids(db, asset_ids)
                if len(assets) != len(asset_ids):
                    found_ids = {a.id for a in assets}
                    missing = [i for i in asset_ids if i not in found_ids]
                    raise FileNotFoundError(f"未找到资产: {missing}")

                formats = [_normalize_format(a.format, a.file_path or "") for a in assets]
                if len(set(formats)) > 1:
                    raise ValueError(
                        "当前仅支持同一种数据格式的批量导出，请按 HDF5、MCAP、LeRobot 或目录型资产分别导出。"
                    )

                if output_path:
                    print(f"[Export] {job_id} 本地目录模式：output_path={output_path}", flush=True)
                    progress_state: List = [0, "整理原始数据"]

                    def _run_build() -> Tuple[str, str]:
                        return build_export_to_directory(
                            assets,
                            output_path,
                            progress_state,
                            compression_mode=compression_mode,
                            cancel_checker=lambda: is_cancelled(job_id),
                        )

                    loop = asyncio.get_running_loop()
                    future = loop.run_in_executor(None, _run_build)

                    async def _poll_progress() -> None:
                        while not future.done():
                            if is_cancelled(job_id):
                                print(f"[Cancel] Task {job_id} cancelled")
                                future.cancel()
                                break
                            await _set_export_job(job_id, {
                                "progress": min(99, progress_state[0]),
                                "currentStep": progress_state[1] if len(progress_state) > 1 else "",
                            })
                            await asyncio.sleep(0.25)

                    poll_task = asyncio.create_task(_poll_progress())
                    try:
                        export_name, full_output_path = await future
                    finally:
                        poll_task.cancel()
                        try:
                            await poll_task
                        except asyncio.CancelledError:
                            pass

                    if is_cancelled(job_id):
                        print(f"[Cancel] Task {job_id} cancelled")
                        return
                    await _set_export_job(job_id, {
                        "status": "ready",
                        "progress": 100,
                        "currentStep": _export_step_title("ready"),
                        "exportDirName": export_name,
                        "fullOutputPath": full_output_path,
                        "fileName": export_name,
                    })
                    try:
                        async with DataAssetsSessionLocal() as db:
                            for aid in asset_ids:
                                asset = await get_asset_by_id(db, aid)
                                if asset:
                                    asset.updated_at = func.now()
                            await db.commit()
                    except Exception:
                        pass
                else:
                    if is_cancelled(job_id):
                        print(f"[Cancel] Task {job_id} cancelled")
                        return
                    with _export_jobs_lock:
                        job = _export_jobs.get(job_id) or {}
                        tmp_dir = job.get("tmpDir")
                    if not tmp_dir:
                        raise RuntimeError("导出任务临时目录丢失")
                    zip_path = os.path.join(tmp_dir, "export.zip")
                    n_assets = len(assets)
                    progress_state: List[Any] = [0, f"0 / {n_assets} · 准备"]

                    def _run_zip() -> Tuple[str, str]:
                        return _build_export_zip_to_file(
                            assets,
                            zip_path,
                            progress_state,
                            compression_mode=compression_mode,
                            cancel_checker=lambda: is_cancelled(job_id),
                        )

                    print(f"[Export] {job_id} 浏览器 zip 模式：线程池构建 zip -> {zip_path}", flush=True)
                    loop = asyncio.get_running_loop()
                    future = loop.run_in_executor(None, _run_zip)

                    async def _poll_zip_progress() -> None:
                        while not future.done():
                            if is_cancelled(job_id):
                                future.cancel()
                                break
                            pct = 0
                            step = ""
                            done_n = 0
                            if isinstance(progress_state, list) and len(progress_state) >= 2:
                                pct = int(progress_state[0]) if isinstance(progress_state[0], (int, float)) else 0
                                step = str(progress_state[1] or "")
                                m = re.match(r"^(\d+)\s*/\s*(\d+)", step)
                                if m:
                                    done_n = int(m.group(1))
                            await _set_export_job(job_id, {
                                "progress": min(99, pct),
                                "currentStep": step,
                                "completedAssets": done_n,
                                "totalAssets": n_assets,
                            })
                            await asyncio.sleep(0.25)

                    poll_task = asyncio.create_task(_poll_zip_progress())
                    try:
                        zip_filename, _ = await future
                    finally:
                        poll_task.cancel()
                        try:
                            await poll_task
                        except asyncio.CancelledError:
                            pass

                    if is_cancelled(job_id):
                        print(f"[Cancel] Task {job_id} cancelled")
                        return
                    print(f"[Export] {job_id} zip 构建完成 file={zip_filename}", flush=True)
                    try:
                        async with DataAssetsSessionLocal() as db:
                            for aid in asset_ids:
                                asset = await get_asset_by_id(db, aid)
                                if asset:
                                    asset.updated_at = func.now()
                            await db.commit()
                    except Exception:
                        pass
                    await _set_export_job(job_id, {
                        "status": "ready",
                        "progress": 100,
                        "currentStep": _export_step_title("ready_zip"),
                        "fileName": zip_filename,
                        "downloadUrl": f"/api/data-assets/export/download?jobId={job_id}",
                        "zipPath": zip_path,
                        "completedAssets": n_assets,
                        "totalAssets": n_assets,
                        "deliveryMode": "browser_zip",
                        "compressionMode": compression_mode or "deflated",
                    })
    except Exception as e:
        msg = str(e) or "导出包生成失败，请稍后重试"
        print(f"[Export] {job_id} 失败: {msg}", flush=True)
        if is_cancelled(job_id):
            await _set_export_job(
                job_id,
                {
                    "status": "cancelled",
                    "progress": 0,
                    "currentStep": "已取消",
                    "errorMessage": "任务已取消",
                },
            )
        else:
            await _set_export_job(
                job_id,
                {
                    "status": "failed",
                    "progress": 0,
                    "currentStep": _export_step_title("failed"),
                    "errorMessage": msg[:400],
                },
            )


def _cleanup_export_job_disk(job: Optional[Dict[str, Any]]) -> None:
    """删除导出临时目录与 zip（不触碰已写入白名单的最终目录导出）。"""
    if not job:
        return
    tmp_dir = job.get("tmpDir")
    if tmp_dir and isinstance(tmp_dir, str) and tmp_dir.strip():
        shutil.rmtree(tmp_dir.strip(), ignore_errors=True)
    zip_p = (job.get("zipPath") or "").strip()
    if zip_p and os.path.isfile(zip_p):
        try:
            os.remove(zip_p)
        except OSError:
            pass


async def _export_job_background(job_id: str, asset_ids: List[int]) -> None:
    """在 FastAPI 应用事件循环中执行导出（与 AsyncSession/engine 同 loop），并写回 TaskJob 表。"""
    finished = datetime.now(timezone.utc)
    try:
        update_task_status(job_id, "running", started_at=finished)
        await _run_export_job(job_id, asset_ids)
    except Exception as e:
        msg = str(e) or "导出任务异常"
        print(f"[Export] {job_id} 后台包装异常: {msg}", flush=True)
        row = get_task_job(job_id)
        row_st = (row.status or "").strip().lower() if row else ""
        if row_st == "cancelled" or "cancel" in msg.lower():
            with _export_jobs_lock:
                popped = _export_jobs.pop(job_id, None) or {}
            _cleanup_export_job_disk(popped)
            update_task_status(job_id, "cancelled", finished_at=datetime.now(timezone.utc))
            return
        with _export_jobs_lock:
            if job_id in _export_jobs and (_export_jobs[job_id].get("status") or "") not in ("ready", "failed", "cancelled"):
                _export_jobs[job_id].update(
                    {
                        "status": "failed",
                        "progress": 0,
                        "errorMessage": msg[:400],
                        "currentStep": _export_step_title("failed"),
                    }
                )
        update_task_status(job_id, "failed", error=msg[:2000], finished_at=datetime.now(timezone.utc))
        return

    row = get_task_job(job_id)
    row_st = (row.status or "").strip().lower() if row else ""

    with _export_jobs_lock:
        in_map = job_id in _export_jobs
        j = dict(_export_jobs[job_id]) if in_map else {}

    final_status = (j.get("status") or "").strip().lower()
    err_msg = (j.get("errorMessage") or "")[:2000]
    finished = datetime.now(timezone.utc)

    if row_st == "cancelled" or final_status == "cancelled":
        with _export_jobs_lock:
            popped = _export_jobs.pop(job_id, None) or j
        _cleanup_export_job_disk(popped)
        update_task_status(job_id, "cancelled", finished_at=finished)
        return

    if not in_map:
        if row_st not in ("cancelled", "failed"):
            update_task_status(job_id, "failed", error="任务记录已删除", finished_at=finished)
        return

    if final_status == "failed":
        update_task_status(job_id, "failed", error=err_msg or "导出失败", finished_at=finished)
        return

    # 协作式取消与 zip 写完之间存在竞态：提交 success 前再读库，避免覆盖 cancelled
    if is_cancelled(job_id):
        with _export_jobs_lock:
            popped = _export_jobs.pop(job_id, None) or j
        _cleanup_export_job_disk(popped)
        update_task_status(job_id, "cancelled", finished_at=finished)
        return

    update_task_status(
        job_id,
        "success",
        result={"exportJobId": job_id, "status": final_status or "ready"},
        finished_at=finished,
    )


def _format_from_filename(filename: str) -> Optional[str]:
    """hdf5 | mcap | lerobot"""
    lower = filename.lower().strip()
    if lower.endswith(".hdf5") or lower.endswith(".h5"):
        return "hdf5"
    if lower.endswith(".mcap"):
        return "mcap"
    if lower.endswith(".zip"):
        return "lerobot"
    return None


def _safe_filename_for_minio_object(filename: str) -> str:
    """对象名中的文件名段：仅 basename，去掉路径与危险字符。"""
    base = Path(filename).name.strip() or "file.dat"
    if base in (".", ".."):
        base = "file.dat"
    base = re.sub(r"[^\w.\-+\s\u4e00-\u9fff]", "_", base)
    base = base.strip()[:240] or "file.dat"
    return base


_MAX_DIRECT_UPLOAD_FILES = 300


def _direct_upload_parse_items_json(us: DataAssetUploadSession) -> List[Dict[str, Any]]:
    raw = (us.items_json or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _direct_upload_size_tolerance(expected: int) -> int:
    if expected <= 0:
        return 4096
    return max(4096, int(expected * 0.0001))


def _merge_directory_asset_meta(
    parsed_meta_json: Optional[str],
    minio_prefix_uri: str,
    bucket: str,
    storage_prefix_slash: str,
    root_dir_name: str,
    manifest_paths: List[Dict[str, Any]],
    total_files: int,
    total_size_bytes: int,
    *,
    dataset_shape: str = "generic_directory",
) -> str:
    base: Dict[str, Any] = {}
    if parsed_meta_json:
        try:
            p = json.loads(parsed_meta_json)
            if isinstance(p, dict):
                base = dict(p)
        except Exception:
            base = {}
    prev = base.get("storage")
    storage: Dict[str, Any] = dict(prev) if isinstance(prev, dict) else {}
    storage["minio_path"] = minio_prefix_uri
    storage["bucket"] = bucket
    storage["prefix"] = storage_prefix_slash.rstrip("/") + "/"
    storage["root_dir_name"] = root_dir_name
    storage["manifest"] = manifest_paths
    storage["total_files"] = total_files
    storage["total_size_bytes"] = total_size_bytes
    storage["upload_mode"] = "directory"
    storage["backend_local_path"] = minio_prefix_uri
    storage["dataset_shape"] = dataset_shape
    base["storage"] = storage
    return json.dumps(base, ensure_ascii=False)


def _directory_suffix_for_object_key(relative_path: str, root_name: str) -> str:
    fr = (relative_path or "").replace("\\", "/").strip().lstrip("/")
    r = (root_name or "").strip().strip("/")
    if r and fr.startswith(r + "/"):
        return fr[len(r) + 1 :]
    if r and fr == r:
        return ""
    return fr


def _infer_directory_root_name(items: List[Any], explicit: Optional[str]) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    roots: set[str] = set()
    for it in items:
        rel = (getattr(it, "relative_path", None) or "").replace("\\", "/").strip()
        if "/" in rel:
            roots.add(rel.split("/")[0].strip())
    if len(roots) > 1:
        raise ValueError("目录批次包含多个不同的根文件夹名")
    if len(roots) == 1:
        return next(iter(roots))
    return "dataset"


def _sanitize_project_path(project: str) -> Path:
    if not project or not project.strip():
        return Path(".")
    first = project.strip().replace(" ", "_").split("/")[0].strip()
    safe = "".join(c for c in first if c.isalnum() or c in "._-").strip()
    return Path(safe) if safe and safe != ".." else Path(".")


def _unique_path(root: Path, filename: str, project_subdir: Optional[str]) -> Path:
    """在 root 下生成唯一路径，避免重名覆盖。"""
    if project_subdir:
        rel = _sanitize_project_path(project_subdir)
        base = (root / rel).resolve()
        try:
            base.resolve().relative_to(root.resolve())
        except ValueError:
            base = root
        base.mkdir(parents=True, exist_ok=True)
    else:
        base = root
    fp = base / filename
    if not fp.exists():
        return fp
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        name, ext = parts
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = base / f"{name}_{ts}.{ext}"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = base / f"{filename}_{ts}"
    return fp


def _get_path_size_bytes(path_str: str) -> int:
    """
    读取路径的总大小（字节数）：
    - 文件：直接返回 file size
    - 目录：递归累加所有子文件大小
    - 其它或异常：返回 0
    """
    try:
        if os.path.isfile(path_str):
            return os.path.getsize(path_str)
        if os.path.isdir(path_str):
            total = 0
            for root, _, files in os.walk(path_str):
                for name in files:
                    fp = os.path.join(root, name)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        # 单个文件读不到时忽略，尽量返回其余文件大小
                        continue
            return total
    except Exception:
        return 0
    return 0


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


def _extract_backend_local_path(meta_json: Optional[str]) -> Optional[str]:
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
    v = storage.get("backend_local_path")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


async def _rollback_failed_import_attempt(
    db: AsyncSession,
    *,
    asset_id: Optional[int],
    minio_uri: Optional[str],
    local_path: Optional[str],
    is_dir: bool,
) -> None:
    """导入失败后的尽力回滚，避免留下缺 meta.storage.minio_path 的半成品或孤儿 MinIO 对象。"""
    mu = (minio_uri or "").strip()
    if mu.startswith("minio://"):
        try:
            delete_by_minio_uri(mu)
        except Exception:
            logger.exception("import rollback: MinIO delete failed")
    lp = (local_path or "").strip()
    if lp and os.path.exists(lp):
        try:
            if is_dir and os.path.isdir(lp):
                shutil.rmtree(lp, ignore_errors=True)
            elif os.path.isfile(lp):
                os.unlink(lp)
        except Exception:
            logger.exception("import rollback: local delete failed")
    if asset_id is not None:
        try:
            await delete_asset(db, asset_id)
        except Exception:
            logger.exception("import rollback: DB asset delete failed")


async def _apply_import_storage_meta(
    db: AsyncSession,
    asset_id: int,
    local_path: str,
    minio_uri: str,
    meta_json: Optional[str],
    parse_status: str,
    err_msg: Optional[str],
) -> None:
    """合并解析 meta 并强制写入 meta.storage.minio_path，供导出从 MinIO 取数使用。"""
    mu = (minio_uri or "").strip()
    if not mu.startswith("minio://"):
        raise ValueError(f"内部错误：无效 minio_uri {mu[:120]}")
    merged = merge_storage_meta(meta_json, local_path, mu)
    if _extract_minio_path(merged) is None:
        merged = merge_storage_meta(None, local_path, mu)
    await update_asset(db, asset_id, parse_status=parse_status, error_msg=err_msg, meta=merged)


def _minio_uri_for_storage_ops(asset: DataAsset) -> Optional[str]:
    """删除等操作：优先 meta.storage.minio_path，否则直传资产的 file_path（minio://）。"""
    meta_minio = _extract_minio_path(getattr(asset, "meta", None))
    if isinstance(meta_minio, str) and meta_minio.startswith("minio://"):
        return meta_minio
    fp = (getattr(asset, "file_path", None) or "").strip()
    if fp.startswith("minio://"):
        return fp
    return None


def _legacy_export_local_path(asset: DataAsset) -> Optional[str]:
    """
    旧版兼容：无 MinIO 身份时，尝试服务器本地路径。
    顺序：meta.storage.backend_local_path（历史登记机路径）→ asset.file_path（非 minio://）。
    """
    bl = _extract_backend_local_path(getattr(asset, "meta", None))
    for candidate in (bl, (getattr(asset, "file_path", None) or "").strip()):
        if not candidate or candidate.startswith("minio://"):
            continue
        try:
            abs_p = os.path.abspath(candidate)
        except Exception:
            continue
        if os.path.isfile(abs_p) or os.path.isdir(abs_p):
            return abs_p
    return None


def _resolve_export_local_path(
    asset: DataAsset,
    minio_cache_root: Path,
    minio_download_cache: Dict[str, str],
) -> str:
    """
    导出前解析资产实际可读路径。
    正式路径（优先）：meta.storage.minio_path 或 file_path=minio://… → 从 MinIO 下载到临时缓存。
    兼容路径：无上述信息时使用 meta.storage.backend_local_path 或本地 file_path。
    """
    minio_uri = _minio_uri_for_storage_ops(asset)
    if minio_uri:
        cached = minio_download_cache.get(minio_uri)
        if cached and os.path.exists(cached):
            return cached
        local_path = download_by_minio_uri(minio_uri, str(minio_cache_root))
        minio_download_cache[minio_uri] = local_path
        return local_path
    legacy = _legacy_export_local_path(asset)
    if legacy:
        return legacy
    raise FileNotFoundError(
        f"资产无可读存储：无 MinIO 路径且本地文件不存在 — {asset.filename}（id={getattr(asset, 'id', '')}）"
    )


def _resolve_export_single_file_path(local_path: str, filename: str, fmt: str) -> str:
    """
    导出单文件资产（hdf5/mcap）时，兼容 MinIO 前缀下载返回目录的情况。
    """
    p = Path((local_path or "").strip())
    if p.is_file():
        return str(p)
    if not p.is_dir():
        raise FileNotFoundError(f"未找到原始文件: {filename}")

    target_name = (filename or "").strip()
    if target_name:
        exact_files = [x for x in p.rglob(target_name) if x.is_file()]
        if len(exact_files) == 1:
            return str(exact_files[0])
        if len(exact_files) > 1:
            exact_files.sort(key=lambda x: len(x.parts))
            return str(exact_files[0])

    suffixes = {".hdf5", ".h5"} if (fmt or "").lower() == "hdf5" else {".mcap"}
    candidates = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in suffixes]
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1 and target_name:
        stem = Path(target_name).stem
        preferred = [x for x in candidates if x.stem == stem]
        if len(preferred) == 1:
            return str(preferred[0])
        if preferred:
            preferred.sort(key=lambda x: len(x.parts))
            return str(preferred[0])
    raise FileNotFoundError(f"未找到原始文件: {filename}")


def _export_annotation_src(asset: DataAsset, resolved_data_path: str) -> Optional[str]:
    """
    导出用标注文件路径：优先与已解析的数据同目录（覆盖 MinIO 缓存目录场景），否则回退 _platform_annotation_path。
    """
    rp = (resolved_data_path or "").strip()
    if rp and os.path.exists(rp):
        base_dir = os.path.dirname(rp) if os.path.isfile(rp) else rp
        for name in ("instruction.json", "instructions.json"):
            cand = os.path.join(base_dir, name)
            if os.path.isfile(cand):
                return cand
    return _platform_annotation_path(asset)


_data_assets_task_columns_checked = False


def _ensure_data_assets_task_columns_sync() -> None:
    """Best-effort: 为 data_assets 表增加任务名称列（若不存在）。"""
    global _data_assets_task_columns_checked
    if _data_assets_task_columns_checked:
        return
    _data_assets_task_columns_checked = True
    try:
        with data_assets_sync_engine.begin() as conn:
            conn.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS label_task_name VARCHAR(256)"))
            conn.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS collect_task_name VARCHAR(256)"))
            conn.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS conversion_task_name VARCHAR(256)"))
            conn.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS sync_status VARCHAR(32) DEFAULT 'synced'"))
            conn.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS sync_error TEXT"))
            conn.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS device_id VARCHAR(64)"))
            conn.execute(text("ALTER TABLE data_assets ADD COLUMN IF NOT EXISTS operator_name VARCHAR(256)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_data_assets_device_id ON data_assets (device_id)"))
            conn.execute(text("UPDATE data_assets SET sync_status='synced' WHERE sync_status IS NULL OR sync_status = ''"))
    except Exception:
        pass


@router.get("", response_model=ApiResponse)
async def list_data_assets(
    keyword: str = Query(None),
    project: str = Query(None),
    format: str = Query(None, alias="format"),
    source: str = Query(None, alias="source"),
    task_id: str = Query(None),
    task_name: str = Query(None),
    created_from: str = Query(None, description="按入库创建时间筛选下限 YYYY-MM-DD（UTC 日界，含当日）"),
    created_to: str = Query(None, description="按入库创建时间筛选上限 YYYY-MM-DD（UTC 日界，含当日）"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    reconcile_collect_disk: bool = Query(
        False,
        description="为 true 且采集端在线时，校验每条采集资产对应 episode 目录是否在盘上，并填充 collect_episode_on_device / collect_episode_rel_path",
    ),
    current_user: User = Depends(get_current_user_or_cookie),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """数据资产列表（筛选 + 分页）。支持 format=mcap|hdf5|lerobot 等缩小格式范围。支持 query token 避免代理未转发 Authorization 导致 401。"""
    try:
        await asyncio.to_thread(_ensure_data_assets_task_columns_sync)
        params = DataAssetQueryParams(
            keyword=keyword,
            project=project,
            format=format,
            source=source,
            task_id=task_id,
            task_name=task_name,
            created_from=created_from,
            created_to=created_to,
            page=page,
            page_size=page_size,
        )
        allowed_project_ids = await data_assets_allowed_project_ids(db, current_user)
        items, total = await get_assets(db, params, allowed_project_ids=allowed_project_ids)

        # 兼容修复：为历史资产补充 file_size_bytes（LeRobot 目录型 + 采集 MCAP 单文件）
        changed = False
        for asset in items:
            try:
                wh = minio_uri_from_fields(getattr(asset, "file_path", None), getattr(asset, "meta", None))
                if wh:
                    ss = (getattr(asset, "sync_status", None) or "").strip().lower()
                    if ss != "synced":
                        asset.sync_status = "synced"
                        asset.sync_error = None
                        changed = True
                if (
                    (asset.format or "").lower() == "lerobot"
                    and (asset.file_size_bytes or 0) <= 0
                    and asset.file_path
                    and os.path.exists(asset.file_path)
                ):
                    size = _get_path_size_bytes(asset.file_path)
                    if size > 0:
                        asset.file_size_bytes = size
                        changed = True
                elif (
                    (asset.source or "").lower() == "collect"
                    and (asset.format or "").lower() == "mcap"
                    and (asset.file_size_bytes or 0) <= 0
                    and asset.file_path
                    and os.path.isfile(asset.file_path)
                ):
                    size = int(os.path.getsize(asset.file_path))
                    if size > 0:
                        asset.file_size_bytes = size
                        changed = True
            except Exception:
                continue
        if changed:
            try:
                await db.commit()
            except Exception:
                pass
        proj_name_map = await _project_display_names_for_assets(db, items)
        presence: Dict[int, Tuple[Optional[bool], Optional[str]]] = {}
        if reconcile_collect_disk and items:
            try:
                presence = await build_collect_disk_presence_map(items, db=db)
            except Exception as ex:
                logger.warning("list_data_assets reconcile_collect_disk failed: %s", ex)
        row_items: List[DataAssetResponse] = []
        for x in items:
            row = _data_asset_response_with_resolved_project(x, current_user.username, proj_name_map)
            aid = int(getattr(x, "id", 0) or 0)
            if aid and aid in presence:
                on_d, rel = presence[aid]
                row = row.model_copy(
                    update={
                        "collect_episode_on_device": on_d,
                        "collect_episode_rel_path": rel,
                    }
                )
            row_items.append(row)
        return ApiResponse(
            ok=True,
            data=DataAssetListResponse(
                items=row_items,
                total=total,
                page=page,
                page_size=page_size,
            ),
        )
    except Exception as e:
        return ApiResponse(ok=False, error=str(e)[:300])


@router.get("/local-files", response_model=ApiResponse)
async def list_local_files(
    path: str = Query("", description="相对平台数据资产根目录的子路径"),
    current_user: User = Depends(get_current_user),
):
    """列出平台数据资产根目录及子目录下的文件/目录，供导入时选择。"""
    if not is_super_admin_or_team_admin(current_user.role):
        raise HTTPException(status_code=403, detail="无权限访问")
    try:
        sub = path.strip().strip("/")
        base = DATA_ASSETS_ROOT / sub if sub else DATA_ASSETS_ROOT
        base = base.resolve()
        if not str(base).startswith(str(DATA_ASSETS_ROOT.resolve())):
            return ApiResponse(ok=False, error="路径不允许")
        if not base.exists() or not base.is_dir():
            return ApiResponse(ok=True, data=[])
        result: List[LocalFileItem] = []
        for entry in sorted(base.iterdir()):
            rel = entry.relative_to(DATA_ASSETS_ROOT)
            result.append(
                LocalFileItem(
                    name=entry.name,
                    path=str(rel).replace("\\", "/"),
                    is_dir=entry.is_dir(),
                    size=entry.stat().st_size if entry.is_file() else None,
                )
            )
        return ApiResponse(ok=True, data=result)
    except Exception as e:
        return ApiResponse(ok=False, error=str(e)[:300])


@router.get("/by-id/{asset_id}", response_model=ApiResponse)
async def get_data_asset(
    asset_id: int,
    current_user: User = Depends(get_current_user_or_cookie),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """获取单条资产（供标注/转换页解析 file_path）。只读接口支持 query token 避免代理未转发 Authorization 导致 401。"""
    asset = await get_asset_by_id(db, asset_id)
    if not await data_asset_visible_to_user(db, current_user, asset):
        return ApiResponse(ok=False, error="资产不存在")
    _raise_if_unsynced(asset)
    # 解析自愈：历史记录因 mcap 版本差异（decoder_factory/iter_messages 返回结构）写入了解析错误，
    # 这里遇到时自动重试一次并覆盖 error_msg（不要求重新导入）。
    try:
        fmt0 = (getattr(asset, "format", None) or "").strip().lower()
        err0 = (getattr(asset, "error_msg", None) or "").strip()
        ps0 = (getattr(asset, "parse_status", None) or "").strip()
        mcap_compat_err = (
            "decoder_factory" in err0
            or "decoder_factories" in err0
            or "tuple' object has no attribute 'channel'" in err0
            or "has no attribute 'channel'" in err0
        )
        if fmt0 == "mcap" and ps0 in ("失败", "failed") and mcap_compat_err:
            local_path = await _resolve_asset_local_base_path(asset)
            meta_json, parse_status, err_msg = parse_meta_for_asset(local_path, fmt0)
            # update_asset 中 error_msg=None 不会覆盖旧值，这里用空字符串显式清理历史错误
            await update_asset(db, asset_id, parse_status=parse_status, error_msg=(err_msg or ""), meta=meta_json)
            asset = await get_asset_by_id(db, asset_id)
    except Exception:
        # 自愈失败不影响详情查看
        pass
    proj_name_map = await _project_display_names_for_assets(db, [asset])
    row = _data_asset_response_with_resolved_project(asset, current_user.username, proj_name_map)
    enriched_name = await _resolve_collect_task_display_name(db, asset)
    if enriched_name:
        row = row.model_copy(update={"collect_task_name": enriched_name})
    return ApiResponse(ok=True, data=row)


@router.get("/episodes", response_model=ApiResponse)
async def list_asset_episodes(
    assetId: int = Query(..., description="数据资产 ID"),
    current_user: User = Depends(get_current_user_or_cookie),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    按数据资产获取 episode 列表（只读查看用）。支持 query token 避免 401。
    - 如果 file_path 是单个 HDF5/MCAP 文件：返回单一 episode
    - 如果是目录：扫描目录下所有 HDF5/MCAP 文件，每个文件一条 episode
    - instruction_text 来自 data_assets 表中与路径匹配的记录
    """
    asset = await get_asset_by_id(db, assetId)
    if not await data_asset_visible_to_user(db, current_user, asset):
        return ApiResponse(ok=False, error="资产不存在")
    _raise_if_unsynced(asset)

    try:
        base_path = await _resolve_asset_local_base_path(asset)
    except FileNotFoundError:
        return ApiResponse(ok=False, error="资产路径不存在")
    except (MinioBucketError, MinioConfigError) as e:
        return ApiResponse(ok=False, error=str(e)[:300])

    episodes: List[Dict[str, Any]] = []

    def _is_data_file(p: str) -> bool:
        lower = p.lower()
        return lower.endswith(".hdf5") or lower.endswith(".h5") or lower.endswith(".mcap")

    if os.path.isfile(base_path):
        if not _is_data_file(base_path):
            return ApiResponse(ok=False, error="当前仅支持 HDF5/MCAP 资产的数据可视化")
        episodes.append(
            {
                "episode_id": str(asset.id),
                "filename": asset.filename or os.path.basename(base_path),
                "abs_path": base_path,
            }
        )
    elif os.path.isdir(base_path):
        # 目录：递归扫描子文件
        for root, _, files in os.walk(base_path):
            for name in files:
                fp = os.path.join(root, name)
                if not _is_data_file(fp):
                    continue
                rel = os.path.relpath(fp, base_path)
                episodes.append(
                    {
                        "episode_id": f"{asset.id}:{rel.replace(os.sep, '/')}",
                        "filename": name,
                        "abs_path": fp,
                    }
                )
    else:
        return ApiResponse(ok=False, error="资产路径不存在")

    # 按路径批量查询 instruction_text（缓存路径与库内 file_path 可能不一致，补充库内规范化路径 + 行级 fallback）
    path_to_instruction: Dict[str, str] = {}
    if episodes:
        try:
            from app.crud.data_asset import get_instruction_text_by_paths

            # ⚠️ Do not access abs_path / warehouse_path directly.
            # Use EpisodeStorage instead.
            path_list = [os.path.normpath((EpisodeStorage(ep).resolve_local_path() or "")) for ep in episodes]
            canon = os.path.normpath((asset.file_path or "").strip())
            if canon and canon not in path_list:
                path_list.append(canon)
            path_to_instruction = await get_instruction_text_by_paths(db, path_list)
        except Exception:
            path_to_instruction = {}

    fallback_inst = (getattr(asset, "instruction_text", None) or "").strip()
    canon_fp = os.path.normpath((asset.file_path or "").strip())
    result = []
    for ep in episodes:
        abs_path = EpisodeStorage(ep).resolve_local_path() or ""
        norm_path = os.path.normpath(abs_path)
        instruction_text = (
            path_to_instruction.get(norm_path, "")
            or (path_to_instruction.get(canon_fp, "") if canon_fp else "")
            or fallback_inst
        )
        result.append(
            {
                "id": ep.get("episode_id"),
                "name": ep.get("filename"),
                "path": abs_path,
                "instruction_text": instruction_text,
            }
        )

    return ApiResponse(ok=True, data=result)


def _asset_episodes_with_paths(asset, base_path: str) -> List[Dict[str, Any]]:
    """与 list_asset_episodes 一致的 episode 列表（含 episode_id, filename, abs_path），用于标注下载。"""
    episodes: List[Dict[str, Any]] = []

    def _is_data_file(p: str) -> bool:
        lower = p.lower()
        return lower.endswith(".hdf5") or lower.endswith(".h5") or lower.endswith(".mcap")

    if os.path.isfile(base_path):
        if _is_data_file(base_path):
            episodes.append({
                "episode_id": str(asset.id),
                "filename": asset.filename or os.path.basename(base_path),
                "abs_path": base_path,
            })
    elif os.path.isdir(base_path):
        for root, _, files in os.walk(base_path):
            for name in files:
                fp = os.path.join(root, name)
                if not _is_data_file(fp):
                    continue
                rel = os.path.relpath(fp, base_path)
                episodes.append({
                    "episode_id": f"{asset.id}:{rel.replace(os.sep, '/')}",
                    "filename": name,
                    "abs_path": fp,
                })
    return episodes


@router.get("/annotations/download_one", response_model=ApiResponse)
async def download_asset_annotation_one(
    assetId: int = Query(..., description="数据资产 ID"),
    episodeId: str = Query(..., description="Episode ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """从数据仓库（data_assets.instruction_text）查询单条标注并返回，用于「下载当前条」"""
    asset = await get_asset_by_id(db, assetId)
    if not await data_asset_visible_to_user(db, current_user, asset):
        raise HTTPException(status_code=404, detail="资产不存在")
    _raise_if_unsynced(asset)
    try:
        base_path = await _resolve_asset_local_base_path(asset)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="资产路径不存在")
    except (MinioBucketError, MinioConfigError) as e:
        raise HTTPException(status_code=502, detail=str(e)[:500])
    # 解析 episodeId -> abs_path
    if os.path.isfile(base_path):
        if ":" in episodeId:
            raise HTTPException(status_code=400, detail="无效的 episode_id")
        abs_path = base_path
        if str(asset.id) != episodeId:
            raise HTTPException(status_code=404, detail="Episode not found")
    else:
        if ":" not in episodeId:
            raise HTTPException(status_code=400, detail="无效的 episode_id")
        _, rel = episodeId.split(":", 1)
        abs_path = os.path.join(base_path, rel.replace("/", os.sep))
    # ⚠️ Do not access abs_path / warehouse_path directly.
    # Use EpisodeStorage instead.
    try:
        abs_path = EpisodeStorage({"episode_id": episodeId, "abs_path": abs_path, "warehouse_path": ""}).resolve_local_path()
    except EpisodeResolveError as ex:
        raise HTTPException(
            status_code=404,
            detail={"error_code": ex.code, "message": ex.message, "episode_id": ex.episode_id},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    from app.crud.data_asset import get_instruction_text_by_paths
    fp_asset = os.path.normpath((asset.file_path or "").strip())
    path_to_instruction = await get_instruction_text_by_paths(
        db, [abs_path] + ([fp_asset] if fp_asset else [])
    )
    np = os.path.normpath(abs_path)
    instruction = (
        path_to_instruction.get(np, "")
        or (path_to_instruction.get(fp_asset, "") if fp_asset else "")
        or (getattr(asset, "instruction_text", None) or "")
    )
    return ApiResponse(ok=True, data={"instruction": instruction})


@router.get("/annotations/download_batch", response_model=ApiResponse)
async def download_asset_annotations_batch(
    assetId: int = Query(..., description="数据资产 ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """从数据仓库（data_assets.instruction_text）批量查询该资产下所有 episode 的标注，用于「下载整个数据集」"""
    asset = await get_asset_by_id(db, assetId)
    if not await data_asset_visible_to_user(db, current_user, asset):
        raise HTTPException(status_code=404, detail="资产不存在")
    _raise_if_unsynced(asset)
    try:
        base_path = await _resolve_asset_local_base_path(asset)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="资产路径不存在")
    except (MinioBucketError, MinioConfigError) as e:
        raise HTTPException(status_code=502, detail=str(e)[:500])
    episodes = _asset_episodes_with_paths(asset, base_path)
    if not episodes:
        return ApiResponse(ok=True, data={"items": []})
    # ⚠️ Do not access abs_path / warehouse_path directly.
    # Use EpisodeStorage instead.
    path_list = [os.path.normpath((EpisodeStorage(ep).resolve_local_path() or "")) for ep in episodes]
    canon = os.path.normpath((asset.file_path or "").strip())
    if canon and canon not in path_list:
        path_list.append(canon)
    from app.crud.data_asset import get_instruction_text_by_paths
    path_to_instruction = await get_instruction_text_by_paths(db, path_list)
    fallback_inst = (getattr(asset, "instruction_text", None) or "").strip()
    items = []
    for ep in episodes:
        abs_path = EpisodeStorage(ep).resolve_local_path() or ""
        norm_path = os.path.normpath(abs_path)
        instruction = (
            path_to_instruction.get(norm_path, "")
            or (path_to_instruction.get(canon, "") if canon else "")
            or fallback_inst
        )
        items.append({"episode_id": ep.get("episode_id"), "path": abs_path, "instruction": instruction})
    return ApiResponse(ok=True, data={"items": items})


@router.get("/episodes/{episode_id}", response_model=ApiResponse)
async def get_asset_episode_detail(
    assetId: int,
    episode_id: str,
    camera_candidates: bool = Query(
        True,
        description="HDF5 为 true 时返回路径名含 camera 的所有节点；为 false 时仅返回校验过的相机",
    ),
    current_user: User = Depends(get_current_user_or_cookie),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    获取数据资产下某个 episode 的详细信息（相机列表、帧数等），供只读可视化页面使用。支持 query token 避免 401。
    实现上复用标注模块的 HDF5/MCAP 解析能力。
    """
    asset = await get_asset_by_id(db, assetId)
    if not await data_asset_visible_to_user(db, current_user, asset):
        raise HTTPException(status_code=404, detail="资产不存在")
    _raise_if_unsynced(asset)

    try:
        base_path = await _resolve_asset_local_base_path(asset)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="资产路径不存在")
    except (MinioBucketError, MinioConfigError) as e:
        raise HTTPException(status_code=502, detail=str(e)[:500])

    # 解析 episode 对应的文件路径
    if os.path.isfile(base_path):
        file_path = base_path
    elif os.path.isdir(base_path):
        # episode_id 形如 "assetId:relative/path/to/file"
        if ":" not in episode_id:
            raise HTTPException(status_code=400, detail="无效的 episode_id")
        _, rel = episode_id.split(":", 1)
        file_path = os.path.join(base_path, rel.replace("/", os.sep))
    else:
        raise HTTPException(status_code=404, detail="资产路径不存在")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Episode file not found: {file_path}")

    fmt = _file_format(file_path)
    hdf5_service = get_hdf5_service()
    try:
        data: Dict[str, Any] = {
            "id": episode_id,
            "name": os.path.basename(file_path),
            "path": file_path,
            "cameras": [],
            "frameCount": 0,
        }
        if fmt == "mcap":
            mcap_service = get_mcap_service()
            cameras = mcap_service.list_camera_candidate_topics(file_path)
            if not cameras:
                cameras = mcap_service.list_cameras(file_path)
            with_frames: List[str] = []
            no_frames: List[str] = []
            frame_count = 0
            for cam in (cameras or []):
                n = mcap_service.get_frame_count(file_path, cam)
                if n > 0:
                    with_frames.append(cam)
                    if n > frame_count:
                        frame_count = n
                else:
                    no_frames.append(cam)
            data["cameras"] = with_frames + no_frames
            data["frameCount"] = frame_count
            if data["cameras"]:
                start_ns, end_ns = mcap_service.get_time_range(file_path, data["cameras"][0])
                if start_ns > 0 or end_ns > 0:
                    data["startTimeNs"] = start_ns
                    data["endTimeNs"] = end_ns
        else:
            import h5py

            with h5py.File(file_path, "r") as f:
                if camera_candidates:
                    cameras = hdf5_service.list_camera_candidate_paths(f, image_only=True)
                    if not cameras:
                        cameras = hdf5_service.list_camera_candidate_paths(f, image_only=False)
                    if not cameras:
                        cameras = hdf5_service.list_cameras(f)
                else:
                    cameras = hdf5_service.list_cameras(f)
                with_frames: List[str] = []
                no_frames: List[str] = []
                frame_count = 0
                for cam in (cameras or []):
                    n = hdf5_service.get_frame_count(file_path, cam)
                    if n > 0:
                        with_frames.append(cam)
                        if n > frame_count:
                            frame_count = n
                    else:
                        no_frames.append(cam)
                data["cameras"] = with_frames + no_frames
                data["frameCount"] = frame_count

        return ApiResponse(ok=True, data=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file ({fmt}): {str(e)}")


@router.get("/frames/{episode_id}")
async def get_asset_frame(
    assetId: int,
    episode_id: str,
    camera: str = Query(..., description="相机名称"),
    frame: int = Query(..., description="帧索引", ge=0),
    quality: int = Query(85, description="JPEG 质量", ge=1, le=100),
    current_user: User = Depends(get_current_user_or_cookie),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    获取数据资产下某个 episode 的单帧图像（只读查看用）。
    实现上与 /api/label/frames/{episode_id} 的无 taskId 分支保持一致。
    """
    asset = await get_asset_by_id(db, assetId)
    if not await data_asset_visible_to_user(db, current_user, asset):
        raise HTTPException(status_code=404, detail="资产不存在")
    _raise_if_unsynced(asset)

    try:
        base_path = await _resolve_asset_local_base_path(asset)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="资产路径不存在")
    except (MinioBucketError, MinioConfigError) as e:
        raise HTTPException(status_code=502, detail=str(e)[:500])

    if os.path.isfile(base_path):
        file_path = base_path
    elif os.path.isdir(base_path):
        if ":" not in episode_id:
            raise HTTPException(status_code=400, detail="无效的 episode_id")
        _, rel = episode_id.split(":", 1)
        file_path = os.path.join(base_path, rel.replace("/", os.sep))
    else:
        raise HTTPException(status_code=404, detail="资产路径不存在")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Episode file not found: {file_path}")

    hdf5_service = get_hdf5_service()
    fmt = _file_format(file_path)
    camera_norm = (camera or "").strip().lstrip("/")
    try:
        if fmt == "mcap":
            mcap_service = get_mcap_service()
            cameras = mcap_service.list_camera_candidate_topics(file_path)
            if not cameras:
                cameras = mcap_service.list_cameras(file_path)
            frame_count = 0
            if cameras:
                # 必须传原始 camera（完整 topic），get_frame_count 内 _camera_to_topic 才可匹配
                frame_count = mcap_service.get_frame_count(file_path, camera or camera_norm)
        else:
            if not hdf5_service._is_hdf5_file(file_path):
                raise HTTPException(
                    status_code=400,
                    detail="文件不是有效的 HDF5 格式，无法读取帧",
                )
            import h5py

            with h5py.File(file_path, "r") as f:
                cameras = hdf5_service.list_camera_candidate_paths(f, image_only=False)
                if not cameras:
                    cameras = hdf5_service.list_cameras(f)
                frame_count = 0
                if cameras:
                    frame_count = hdf5_service.get_frame_count(file_path, camera_norm or camera)

        if not cameras:
            raise HTTPException(status_code=404, detail="No cameras found in file")
        cameras_norm = [c.lstrip("/") for c in cameras]
        if camera_norm and camera_norm not in cameras_norm:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera}' not found. Available cameras: {', '.join(cameras[:10])}{'...' if len(cameras) > 10 else ''}",
            )
        if frame_count == 0:
            raise HTTPException(status_code=404, detail=f"Camera '{camera}' has no frames")
        if frame >= frame_count:
            raise HTTPException(
                status_code=404,
                detail=f"Frame index {frame} out of range. Camera '{camera}' has {frame_count} frames (0-{frame_count-1})",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 真正读取帧并编码为 JPEG（与 routes_label 一致：使用 get_frame_image 直接得到 JPEG 字节）
    # MCAP 需传原始 camera（完整 topic），以便 _camera_to_topic 中 "camera_name in cameras" 能匹配
    try:
        if fmt == "mcap":
            mcap_service = get_mcap_service()
            image_bytes = mcap_service.get_frame_image(
                file_path, camera or camera_norm, frame, quality=int(quality)
            )
        else:
            hdf5_service = get_hdf5_service()
            image_bytes = hdf5_service.get_frame_image(
                file_path, camera_norm or camera, frame, quality=int(quality)
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取帧失败: {str(e)}")

    if image_bytes is None:
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found: episode={episode_id}, camera={camera}, frame={frame}",
        )

    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/jpeg",
    )


@router.websocket("/ws/playback/{episode_id}")
async def websocket_asset_playback(
    websocket: WebSocket,
    episode_id: str,
    camera: str = Query(...),
    assetId: int = Query(...),
):
    """数据资产 WebSocket 播放，支持 MCAP 与 HDF5，与标注页 /api/label/ws/playback 协议一致。"""
    from app.services.episode_storage import clear_episode_cache
    import logging
    logger = logging.getLogger(__name__)
    await websocket.accept()
    logger.info("[data-assets WS playback] episode=%s camera=%s assetId=%s", episode_id, camera[:80] if len(camera) > 80 else camera, assetId)

    current_user = _get_current_user_for_ws(websocket)
    if current_user is None:
        await websocket.close(code=4401, reason="未认证")
        return

    async with DataAssetsSessionLocal() as db:
        asset = await get_asset_by_id(db, assetId)
        if not await data_asset_visible_to_user(db, current_user, asset):
            await websocket.close(code=4404, reason="资产不存在")
            return
        if not _asset_is_synced(asset):
            await websocket.close(code=4409, reason="该数据尚未同步，暂不可操作")
            return
    try:
        base_path = await _resolve_asset_local_base_path(asset)
    except FileNotFoundError:
        await websocket.close(code=4404, reason="资产路径不存在")
        return
    except (MinioBucketError, MinioConfigError) as e:
        await websocket.close(code=4404, reason=str(e)[:200])
        return
    if os.path.isfile(base_path):
        file_path = base_path
    elif os.path.isdir(base_path):
        if ":" not in episode_id:
            await websocket.close(code=4400, reason="无效的 episode_id")
            return
        _, rel = episode_id.split(":", 1)
        file_path = os.path.join(base_path, rel.replace("/", os.sep))
    else:
        await websocket.close(code=4404, reason="资产路径不存在")
        return
    if not os.path.exists(file_path):
        await websocket.close(code=4404, reason="文件不存在")
        return
    # 连接级缓存：后续播放循环禁止再 resolve
    websocket.state.local_path = file_path
    assert hasattr(websocket.state, "local_path")
    file_path = websocket.state.local_path
    fmt = _file_format(file_path)
    if fmt not in ("mcap", "hdf5"):
        await websocket.close(code=4404, reason="仅支持 MCAP/HDF5")
        return
    frame_count, get_frame_fn, iter_frames_fn = _ws_playback_resolve_file(file_path, fmt, camera)
    if frame_count <= 0:
        await websocket.close(code=4404, reason="该相机无可用帧")
        return

    # 默认播放帧率稍高一些，提升流畅度；允许前端在 1~60fps 范围内调节
    state = {"playing": False, "fps": 24, "seek_to": None}
    loop = asyncio.get_event_loop()

    async def receive_loop():
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action", "")
                if action == "play":
                    state["playing"] = True
                    # 限制帧率在 1~60fps，避免过高占用 CPU
                    state["fps"] = max(1, min(60, int(data.get("fps", state["fps"] or 24))))
                elif action == "pause":
                    state["playing"] = False
                elif action == "seek":
                    state["seek_to"] = max(0, int(data.get("frame", 0)))
        except WebSocketDisconnect:
            state["playing"] = False
        except Exception:
            state["playing"] = False

    async def send_loop():
        current_index = 0
        frame_gen = None
        try:
            while True:
                if state["seek_to"] is not None:
                    seek_frame = min(state["seek_to"], frame_count - 1)
                    state["seek_to"] = None
                    frame_gen = None  # 重置生成器，下次播放从新位置顺序读
                    jpeg = await loop.run_in_executor(None, lambda sf=seek_frame: get_frame_fn(sf))
                    if jpeg is not None:
                        msg = seek_frame.to_bytes(4, "big") + jpeg
                        await websocket.send_bytes(msg)
                    current_index = seek_frame + 1
                if not state["playing"]:
                    await asyncio.sleep(0.05)
                    continue
                if current_index >= frame_count:
                    state["playing"] = False
                    try:
                        await websocket.send_json({"type": "ended", "frame": current_index})
                    except Exception:
                        pass
                    continue
                # MCAP：使用顺序 iter_frames，避免 get_frame_image O(n) 导致越播越慢
                if fmt == "mcap" and iter_frames_fn is not None:
                    if frame_gen is None:
                        frame_gen = iter_frames_fn(current_index)

                    def _next():
                        try:
                            return next(frame_gen)
                        except StopIteration:
                            return None

                    jpeg = await loop.run_in_executor(None, _next)
                    await asyncio.sleep(0)
                else:
                    jpeg = await loop.run_in_executor(None, lambda i=current_index: get_frame_fn(i))
                    await asyncio.sleep(0)

                # 到达结尾：停止播放并发送一次 ended，避免反复 0/1 抖动
                if jpeg is None:
                    state["playing"] = False
                    frame_gen = None
                    try:
                        await websocket.send_json({"type": "ended", "frame": current_index})
                    except Exception:
                        pass
                    continue
                msg = current_index.to_bytes(4, "big") + jpeg
                await websocket.send_bytes(msg)
                current_index += 1
                await asyncio.sleep(1.0 / state["fps"])
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    recv_task = asyncio.create_task(receive_loop())
    send_task = asyncio.create_task(send_loop())
    try:
        await asyncio.gather(recv_task, send_task)
    except asyncio.CancelledError:
        pass
    finally:
        recv_task.cancel()
        send_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        # WS 连接结束时清理缓存，避免泄漏（连接级生命周期）
        clear_episode_cache()


@router.post("/import", response_model=ApiResponse)
async def import_data_assets(
    http_request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project: Optional[str] = Form(None),
    project_name: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """
    数据页标准导入：multipart 上传 → 落盘 DATA_ASSETS_ROOT → 上传 MinIO → 写入 data_assets。
    每条成功导入的记录均包含 meta.storage.minio_path（与 meta.storage.backend_local_path），导出链路依赖前者。
    支持 .hdf5/.h5、.mcap、.zip（LeRobot）；文件名含路径分隔符时按 LeRobot 目录聚合还原。
    """
    if not files:
        return ApiResponse(ok=False, error="请至少上传一个文件")
    raw_proj = (project or "").strip()
    if not raw_proj:
        return ApiResponse(ok=False, error="请选择所属项目")
    p, perr = await assert_may_write_project_for_data_asset_import(db, current_user, raw_proj)
    if perr or not p:
        return ApiResponse(ok=False, error=perr or "项目校验失败")
    project_id = str(p.id)
    proj_name = (project_name or p.name or project_id).strip()

    imported = []
    failed = []
    # 如果上传的文件名包含路径分隔符（来自 webkitdirectory / 拖拽目录），则视为目录型导入（LeRobot 等），
    # 将同一根目录下的文件聚合为一个资产，并在服务端还原目录结构。
    has_dir_like = any(("/" in (u.filename or "")) or ("\\" in (u.filename or "")) for u in files)

    if has_dir_like:
        # 分组：按第一级目录名聚合为一个数据集
        groups: Dict[str, List[UploadFile]] = {}
        for u in files:
            raw = (u.filename or "unknown").replace("\\", "/").strip().lstrip("/")
            if not raw:
                raw = "unknown"
            parts = raw.split("/", 1)
            root_name = parts[0] or "dataset"
            groups.setdefault(root_name, []).append(u)

        for root_name, group_files in groups.items():
            abs_root = ""
            minio_uri: Optional[str] = None
            asset_id: Optional[int] = None
            try:
                # 在数据资产根目录下为该数据集分配唯一目录（含项目子目录隔离）
                safe_root = root_name.strip().rstrip("/\\") or "dataset"
                dest_root_base = _unique_path(DATA_ASSETS_ROOT, safe_root, project_id)
                # _unique_path 返回的是「文件」路径，这里改为目录：去掉后缀，按目录使用
                dest_root = dest_root_base
                if dest_root.suffix:
                    dest_root = dest_root.with_suffix("")
                dest_root.mkdir(parents=True, exist_ok=True)

                # 还原每个文件的相对路径到 dest_root 下（multipart 文件名规则与 importDataAssetFiles 一致：含 / 的相对路径）
                for u in group_files:
                    raw = (u.filename or u.filename or "unknown").replace("\\", "/").strip().lstrip("/")
                    if not raw:
                        raw = safe_root
                    parts = raw.split("/", 1)
                    rel_path = parts[1] if len(parts) > 1 else parts[0]
                    target_path = dest_root / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    content = await u.read()
                    with open(target_path, "wb") as out:
                        out.write(content)

                abs_root = str(dest_root.resolve())
                # LeRobot 目录有效性校验：避免导入成一堆散文件
                if not _inspect_lerobot_dir(abs_root):
                    try:
                        shutil.rmtree(abs_root, ignore_errors=True)
                    except Exception:
                        pass
                    failed.append(
                        {"name": safe_root, "reason": "当前文件夹不是有效的 LeRobot 数据集目录"}
                    )
                    continue

                # 仅限制“同一项目”内重复导入；允许同一条数据绑定到不同项目
                exists = (await db.execute(
                    select(DataAsset).where(DataAsset.file_path == abs_root, DataAsset.project_id == project_id)
                )).scalar_one_or_none()
                if exists:
                    try:
                        shutil.rmtree(abs_root, ignore_errors=True)
                    except Exception:
                        pass
                    failed.append({"name": safe_root, "reason": "该目录已登记，不能重复导入"})
                    continue

                size = _get_path_size_bytes(abs_root)
                code = await next_code(db)
                minio_prefix = f"projects/{project_id}/import/{code}/{safe_root}"
                minio_uri = upload_dir_to_project_bucket(
                    project_name=(p.name or project_id),
                    local_dir_path=abs_root,
                    object_prefix=minio_prefix,
                )
                create_data = DataAssetCreate(
                    code=code,
                    filename=os.path.basename(abs_root.rstrip(os.sep)) or safe_root,
                    format="lerobot",
                    source="import",
                    project_id=project_id,
                    project_name=proj_name,
                    file_path=abs_root,
                    file_size_bytes=size,
                    meta=None,
                    parse_status="解析中",
                    error_msg=None,
                    sync_status="unsynced",
                    sync_error="对象存储已上传，正在写入元数据",
                    operator_name=(getattr(current_user, "username", None) or "").strip() or None,
                )
                asset = await create_asset(db, create_data)
                asset_id = asset.id
                meta_json, parse_status, err_msg = parse_meta_for_asset(abs_root, "lerobot")
                await _apply_import_storage_meta(
                    db, asset.id, abs_root, minio_uri, meta_json, parse_status, err_msg
                )
                await update_asset(db, asset.id, sync_status="synced", sync_error=None)
                imported.append({"name": create_data.filename, "id": asset.id, "minio_path": minio_uri})
            except MinioBucketError as e:
                await _rollback_failed_import_attempt(
                    db,
                    asset_id=None,
                    minio_uri=minio_uri,
                    local_path=abs_root or None,
                    is_dir=True,
                )
                failed.append({"name": root_name, "reason": str(e)[:200]})
            except Exception as e:
                await _rollback_failed_import_attempt(
                    db,
                    asset_id=asset_id,
                    minio_uri=minio_uri,
                    local_path=abs_root or None,
                    is_dir=True,
                )
                failed.append({"name": root_name, "reason": str(e)[:200]})
    else:
        # 纯文件导入：一文件一资产（HDF5/MCAP/LeRobot 压缩包）
        for u in files:
            raw = u.filename or "unknown"
            filename = raw.replace("\\", "/").split("/")[-1].strip() or "unknown"
            fmt = _format_from_filename(filename)
            if not fmt:
                failed.append({"name": filename, "reason": "仅支持 .hdf5/.h5、.mcap、.zip（LeRobot）"})
                continue
            abs_path = ""
            minio_uri: Optional[str] = None
            asset_id: Optional[int] = None
            try:
                dest = _unique_path(DATA_ASSETS_ROOT, filename, project_id)
                content = await u.read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as out:
                    out.write(content)
                abs_path = str(dest.resolve())
                # 仅限制“同一项目”内重复导入；允许同一条数据绑定到不同项目
                exists = (await db.execute(
                    select(DataAsset).where(DataAsset.file_path == abs_path, DataAsset.project_id == project_id)
                )).scalar_one_or_none()
                if exists:
                    try:
                        dest.unlink()
                    except Exception:
                        pass
                    failed.append({"name": filename, "reason": "该路径已登记，不能重复导入"})
                    continue
                size = dest.stat().st_size
                code = await next_code(db)
                minio_object = f"projects/{project_id}/import/{code}/{dest.name}"
                minio_uri = upload_file_to_project_bucket(
                    project_name=(p.name or project_id),
                    local_file_path=abs_path,
                    object_name=minio_object,
                )
                create_data = DataAssetCreate(
                    code=code,
                    filename=dest.name,
                    format=fmt,
                    source="import",
                    project_id=project_id,
                    project_name=proj_name,
                    file_path=abs_path,
                    file_size_bytes=size,
                    meta=None,
                    parse_status="解析中",
                    error_msg=None,
                    sync_status="unsynced",
                    sync_error="对象存储已上传，正在写入元数据",
                    operator_name=(getattr(current_user, "username", None) or "").strip() or None,
                )
                asset = await create_asset(db, create_data)
                asset_id = asset.id
                meta_json, parse_status, err_msg = parse_meta_for_asset(abs_path, fmt)
                await _apply_import_storage_meta(
                    db, asset.id, abs_path, minio_uri, meta_json, parse_status, err_msg
                )
                await update_asset(db, asset.id, sync_status="synced", sync_error=None)
                imported.append({"name": dest.name, "id": asset.id, "minio_path": minio_uri})
            except MinioBucketError as e:
                await _rollback_failed_import_attempt(
                    db,
                    asset_id=None,
                    minio_uri=minio_uri,
                    local_path=abs_path or None,
                    is_dir=False,
                )
                failed.append({"name": filename, "reason": str(e)[:200]})
            except Exception as e:
                await _rollback_failed_import_attempt(
                    db,
                    asset_id=asset_id,
                    minio_uri=minio_uri,
                    local_path=abs_path or None,
                    is_dir=False,
                )
                failed.append({"name": filename, "reason": str(e)[:200]})

    primary_name = imported[0]["name"] if len(imported) == 1 else None
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=http_request,
        action_type=AA.IMPORT_DATA_ASSET,
        project_id=project_id,
        project_name=proj_name,
        resource_type=AR.DATA_ASSET,
        resource_id=str(imported[0]["id"]) if len(imported) == 1 else None,
        resource_name=primary_name or (f"批量导入({len(imported)}个文件)" if imported else "批量导入"),
        detail_json={
            "imported_count": len(imported),
            "failed_count": len(failed),
            "imported": imported[:50],
            "failed": failed[:20],
        },
        result="SUCCESS" if imported else ("FAIL" if failed else "SUCCESS"),
    )
    return ApiResponse(
        ok=True,
        data={"imported": imported, "failed": failed},
    )


@router.post("/upload-init", response_model=ApiResponse)
async def data_assets_upload_init(
    body: DirectUploadInitBody = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """浏览器直传 MinIO：预签名 PUT（single_file / multi_file / directory）。"""
    mode = (body.upload_mode or "single_file").strip().lower()
    raw_pid = (body.project_id or "").strip()
    if not raw_pid:
        return ApiResponse(ok=False, error="请选择所属项目")
    p, uerr = await assert_may_write_project_for_data_asset_import(db, current_user, raw_pid)
    if uerr or not p:
        return ApiResponse(ok=False, error=uerr or "项目校验失败")
    project_id = str(p.id)
    logger.info(
        "upload-init request: mode=%s project_id=%s items_in=%s",
        mode,
        project_id,
        len(list(body.items or [])),
    )

    uid = str(current_user.id)

    upload_session_id = uuid.uuid4().hex
    proj_bucket = project_bucket_name(p.name or project_id)

    try:
        await asyncio.to_thread(ensure_project_bucket, p.name or project_id)
    except MinioConfigError as e:
        return ApiResponse(ok=False, error=str(e)[:200])
    except MinioBucketError as e:
        return ApiResponse(ok=False, error=str(e)[:200])

    if mode == "single_file":
        items_one = list(body.items or [])
        if len(items_one) == 1:
            o = items_one[0]
            fn_src = Path((o.relative_path or "").replace("\\", "/")).name
            try:
                sz = int(o.size_bytes)
            except (TypeError, ValueError):
                return ApiResponse(ok=False, error="size_bytes 无效")
            ct = o.content_type
            cid = (o.client_file_id or "").strip() or uuid.uuid4().hex
        else:
            fn_src = body.filename or ""
            try:
                sz = int(body.size_bytes) if body.size_bytes is not None else 0
            except (TypeError, ValueError):
                return ApiResponse(ok=False, error="size_bytes 无效")
            ct = body.content_type
            cid = uuid.uuid4().hex
        if not fn_src or sz <= 0:
            return ApiResponse(ok=False, error="单文件请提供 filename+size_bytes 或 items[1]")
        safe_fn = _safe_filename_for_minio_object(fn_src)
        fmt = _format_from_filename(safe_fn)
        if not fmt:
            return ApiResponse(ok=False, error="仅支持 .hdf5/.h5、.mcap、.zip（LeRobot）")
        object_key = f"projects/{project_id}/import/v2/{upload_session_id}/{safe_fn}"
        try:
            upload_url, expires_at = await asyncio.to_thread(
                generate_presigned_put_url,
                proj_bucket,
                object_key,
                expires_seconds=3600,
            )
        except (MinioConfigError, MinioBucketError) as e:
            return ApiResponse(ok=False, error=str(e)[:200])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        headers: Dict[str, str] = {}
        ctn = (ct or "").strip()
        if ctn:
            headers["Content-Type"] = ctn
        us = DataAssetUploadSession(
            id=upload_session_id,
            user_id=uid,
            project_id=project_id,
            status="presigned",
            bucket=proj_bucket,
            object_key=object_key,
            filename=safe_fn,
            size_bytes=sz,
            content_type=ctn or None,
            expires_at=expires_at,
            upload_mode="single_file",
        )
        db.add(us)
        try:
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("upload-init: persist upload_sessions failed")
            return ApiResponse(ok=False, error="创建上传会话失败")
        exp_str = expires_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        item_out = {
            "client_file_id": cid,
            "relative_path": safe_fn,
            "object_key": object_key,
            "upload_url": upload_url,
            "method": "PUT",
            "headers": headers,
        }
        logger.info(
            "upload-init ok: session=%s mode=single_file project=%s object_key=%s",
            upload_session_id,
            project_id,
            object_key,
        )
        return ApiResponse(
            ok=True,
            data={
                "upload_session_id": upload_session_id,
                "bucket": proj_bucket,
                "expires_at": exp_str,
                "upload_mode": "single_file",
                "upload_items": [item_out],
                "object_key": object_key,
                "upload_url": upload_url,
                "method": "PUT",
                "headers": headers,
            },
        )

    if mode not in ("multi_file", "directory"):
        return ApiResponse(ok=False, error="upload_mode 无效")

    items_in = list(body.items or [])
    if not items_in:
        return ApiResponse(ok=False, error="请提供 items")
    if len(items_in) > _MAX_DIRECT_UPLOAD_FILES:
        return ApiResponse(ok=False, error=f"单批最多 {_MAX_DIRECT_UPLOAD_FILES} 个文件")

    if mode == "multi_file" and len(items_in) < 2:
        return ApiResponse(ok=False, error="multi_file 至少需要 2 个文件")

    # webkitdirectory 等场景会把子目录当成 size=0 的条目混入；此类项不应导致整批 upload-init 失败
    items_valid: List[Tuple[Any, int]] = []
    for it in items_in:
        try:
            z = int(it.size_bytes)
        except (TypeError, ValueError):
            return ApiResponse(ok=False, error="size_bytes 无效")
        if z <= 0:
            continue
        items_valid.append((it, z))

    _skipped_zero = len(items_in) - len(items_valid)
    if _skipped_zero > 0:
        logger.info(
            "upload-init: skipped %s item(s) with size_bytes<=0 (upload_mode=%s, kept=%s)",
            _skipped_zero,
            mode,
            len(items_valid),
        )

    if mode == "multi_file" and len(items_valid) < 2:
        return ApiResponse(
            ok=False,
            error="multi_file 至少需要 2 个有效文件（size_bytes>0）；0 字节目录占位等无效项已跳过",
        )
    if mode == "directory" and len(items_valid) == 0:
        return ApiResponse(
            ok=False,
            error="目录下没有可上传的有效文件（size_bytes≤0 的项已跳过，常见于文件夹内的目录占位条目）",
        )

    planned_rows: List[Dict[str, Any]] = []
    object_keys: List[str] = []
    total_sz = 0

    if mode == "multi_file":
        for idx, (it, z) in enumerate(items_valid):
            fn = Path((it.relative_path or "").replace("\\", "/")).name
            safe_fn = _safe_filename_for_minio_object(fn)
            fmt = _format_from_filename(safe_fn)
            if not fmt:
                return ApiResponse(ok=False, error=f"不支持的文件类型: {fn}")
            ok = f"projects/{project_id}/import/v2/{upload_session_id}/files/{idx:04d}_{safe_fn}"
            object_keys.append(ok)
            total_sz += z
            planned_rows.append(
                {
                    "client_file_id": (it.client_file_id or "").strip() or uuid.uuid4().hex,
                    "relative_path": (it.relative_path or "").strip(),
                    "object_key": ok,
                    "size_bytes": z,
                    "filename": safe_fn,
                    "format": fmt,
                }
            )
        primary_object_key = object_keys[0]
        filename_row = f"multi×{len(items_valid)}"
        root_dn = None
        aname = None
    else:
        try:
            root_name = _infer_directory_root_name(items_in, body.root_dir_name)
        except ValueError as e:
            return ApiResponse(ok=False, error=str(e)[:200])
        safe_root = _safe_filename_for_minio_object(root_name)
        base_prefix = f"projects/{project_id}/import/v2/{upload_session_id}/dir/{safe_root}"
        for it, z in items_valid:
            sfx = _directory_suffix_for_object_key(it.relative_path or "", root_name)
            raw_name = Path((it.relative_path or "").replace("\\", "/")).name
            try:
                rel_norm = normalize_relative_path(sfx if sfx else raw_name)
            except MinioBucketError as e:
                return ApiResponse(ok=False, error=str(e)[:200])
            leaf = Path(rel_norm).name
            safe_leaf = _safe_filename_for_minio_object(leaf)
            fmt = _format_from_filename(safe_leaf)
            # 目录型整包上传：允许 data/meta/videos 下任意叶子类型；占位 format 供会话清单使用，资产 format 由 upload-complete 整目录判定
            if not fmt:
                fmt = "lerobot"
            okey = f"{base_prefix}/{rel_norm}"
            object_keys.append(okey)
            total_sz += z
            planned_rows.append(
                {
                    "client_file_id": (it.client_file_id or "").strip() or uuid.uuid4().hex,
                    "relative_path": rel_norm,
                    "object_key": okey,
                    "size_bytes": z,
                    "filename": safe_leaf,
                    "format": fmt,
                }
            )
        primary_object_key = base_prefix
        filename_row = safe_root
        root_dn = root_name
        aname = safe_root

    try:
        urls, expires_at = await asyncio.to_thread(
            presigned_put_many,
            proj_bucket,
            object_keys,
            expires_seconds=3600,
        )
    except (MinioConfigError, MinioBucketError) as e:
        return ApiResponse(ok=False, error=str(e)[:200])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    upload_items_out: List[Dict[str, Any]] = []
    for idx, row in enumerate(planned_rows):
        it0 = items_valid[idx][0]
        hdrs: Dict[str, str] = {}
        cti = (it0.content_type or "").strip()
        if cti:
            hdrs["Content-Type"] = cti
        upload_items_out.append(
            {
                "client_file_id": row["client_file_id"],
                "relative_path": row["relative_path"],
                "object_key": row["object_key"],
                "upload_url": urls[idx],
                "method": "PUT",
                "headers": hdrs,
            }
        )

    store_json = json.dumps(planned_rows, ensure_ascii=False)
    us = DataAssetUploadSession(
        id=upload_session_id,
        user_id=uid,
        project_id=project_id,
        status="presigned",
        bucket=proj_bucket,
        object_key=primary_object_key,
        filename=filename_row,
        size_bytes=total_sz,
        content_type=None,
        expires_at=expires_at,
        upload_mode=mode,
        items_json=store_json,
        expected_count=len(items_valid),
        expected_total_size=total_sz,
        root_dir_name=root_dn,
        asset_name=aname,
    )
    db.add(us)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("upload-init: persist upload_sessions failed")
        return ApiResponse(ok=False, error="创建上传会话失败")

    exp_str = expires_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    out_init: Dict[str, Any] = {
        "upload_session_id": upload_session_id,
        "bucket": proj_bucket,
        "expires_at": exp_str,
        "upload_mode": mode,
        "upload_items": upload_items_out,
    }
    if mode == "directory" and root_dn:
        out_init["root_dir_name"] = root_dn
    logger.info(
        "upload-init ok: session=%s mode=%s project=%s valid_items=%s presigned=%s total_bytes=%s",
        upload_session_id,
        mode,
        project_id,
        len(items_valid),
        len(upload_items_out),
        total_sz,
    )
    return ApiResponse(ok=True, data=out_init)


async def _upload_complete_multi_file(
    db: AsyncSession,
    us: DataAssetUploadSession,
    sid: str,
    proj_name: str,
    viewer: Optional[str],
) -> ApiResponse:
    items = _direct_upload_parse_items_json(us)
    if not items:
        return ApiResponse(ok=False, error="会话缺少文件清单")
    failed_items: List[Dict[str, Any]] = []
    imported_assets: List[Dict[str, Any]] = []
    for it in items:
        oid = str(it.get("object_key") or "")
        rel = str(it.get("relative_path") or "")
        try:
            st = await asyncio.to_thread(stat_object, us.bucket, oid)
        except MinioBucketError as e:
            failed_items.append({"object_key": oid, "relative_path": rel, "reason": str(e)[:160]})
            continue
        if int(st.size) != int(it.get("size_bytes", -1)):
            failed_items.append(
                {
                    "object_key": oid,
                    "relative_path": rel,
                    "reason": f"大小不一致: stat={st.size} session={it.get('size_bytes')}",
                }
            )
            continue
        minio_u = build_minio_uri(us.bucket, oid)
        ex = (
            await db.execute(
                select(DataAsset)
                .where(DataAsset.file_path == minio_u, DataAsset.project_id == us.project_id)
                .limit(1)
            )
        ).scalars().first()
        if ex:
            imported_assets.append(_data_asset_response_for_viewer(ex, viewer).model_dump(mode="json"))
            continue
        fmt = it.get("format") or _format_from_filename(str(it.get("filename") or rel))
        if not fmt:
            failed_items.append({"object_key": oid, "relative_path": rel, "reason": "无法识别格式"})
            continue
        tmp_d = tempfile.mkdtemp(prefix="data_assets_direct_mf_")
        aid: Optional[int] = None
        try:
            try:
                local_p = await asyncio.to_thread(download_by_minio_uri, minio_u, tmp_d)
            except MinioBucketError as e:
                failed_items.append({"object_key": oid, "relative_path": rel, "reason": str(e)[:160]})
                continue
            code = await next_code(db)
            cd = DataAssetCreate(
                code=code,
                filename=str(it.get("filename") or Path(rel).name),
                format=str(fmt),
                source="import",
                project_id=us.project_id,
                project_name=proj_name,
                file_path=minio_u,
                file_size_bytes=int(st.size),
                meta=None,
                parse_status="解析中",
                error_msg=None,
                sync_status="unsynced",
                sync_error="直传对象已就绪，正在写入元数据",
                operator_name=(viewer or "").strip() or None,
            )
            asset = await create_asset(db, cd)
            aid = asset.id
            mj, ps, em = parse_meta_for_asset(local_p, str(fmt))
            await _apply_import_storage_meta(db, asset.id, minio_u, minio_u, mj, ps, em)
            await update_asset(db, asset.id, sync_status="synced", sync_error=None)
            ref = await get_asset_by_id(db, aid)
            imported_assets.append(_data_asset_response_for_viewer(ref, viewer).model_dump(mode="json"))
        except Exception as e:
            logger.exception("upload-complete multi_file item failed")
            if aid is not None:
                try:
                    await delete_asset(db, aid)
                except Exception:
                    pass
            failed_items.append({"object_key": oid, "relative_path": rel, "reason": str(e)[:160]})
        finally:
            shutil.rmtree(tmp_d, ignore_errors=True)

    if not imported_assets:
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
        )
        await db.commit()
        logger.info(
            "upload-complete multi_file session=%s outcome=all_failed stat_failures=%s",
            sid,
            len(failed_items),
        )
        return ApiResponse(
            ok=False,
            error="全部文件登记失败",
            data={"assets": [], "failed_items": failed_items},
        )

    logger.info(
        "upload-complete multi_file session=%s success_assets=%s failed_items=%s",
        sid,
        len(imported_assets),
        len(failed_items),
    )
    out_data = {"assets": imported_assets, "failed_items": failed_items}
    await db.execute(
        update(DataAssetUploadSession)
        .where(DataAssetUploadSession.id == sid)
        .values(status="completed", result_payload_json=json.dumps(out_data, ensure_ascii=False, default=str))
    )
    await db.commit()
    return ApiResponse(ok=True, data=out_data)


async def _upload_complete_directory(
    db: AsyncSession,
    body: DirectUploadCompleteBody,
    us: DataAssetUploadSession,
    sid: str,
    proj_name: str,
    viewer: Optional[str],
) -> ApiResponse:
    man = body.manifest
    if man is None:
        return ApiResponse(ok=False, error="directory 需提供 manifest")
    sess_root = (us.root_dir_name or "").strip()
    if (man.root_dir_name or "").strip() != sess_root:
        return ApiResponse(ok=False, error="manifest.root_dir_name 与会话不一致")
    if int(man.total_files) != int(us.expected_count or -1):
        return ApiResponse(ok=False, error="total_files 与会话 expected_count 不一致")
    exp_sz = int(us.expected_total_size or 0)
    tol = _direct_upload_size_tolerance(exp_sz)
    if abs(int(man.total_size_bytes) - exp_sz) > tol:
        return ApiResponse(ok=False, error=f"total_size_bytes 与会话偏差过大（允许±{tol}）")

    sess_items = _direct_upload_parse_items_json(us)
    if len(sess_items) != len(man.paths):
        return ApiResponse(ok=False, error="manifest.paths 条数与会话不符")
    by_rel: Dict[str, Dict[str, Any]] = {str(s["relative_path"]): s for s in sess_items}
    sum_stat = 0
    manifest_rows: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for pe in man.paths:
        try:
            rpn = normalize_relative_path((pe.relative_path or "").strip())
        except MinioBucketError as e:
            return ApiResponse(ok=False, error=str(e)[:200])
        if rpn not in by_rel:
            return ApiResponse(ok=False, error=f"manifest 含未授权路径: {pe.relative_path}")
        row = by_rel[rpn]
        if int(pe.size_bytes) != int(row["size_bytes"]):
            return ApiResponse(ok=False, error=f"manifest 大小与会话不符: {rpn}")
        try:
            st = await asyncio.to_thread(stat_object, us.bucket, row["object_key"])
        except MinioBucketError as e:
            return ApiResponse(ok=False, error=f"对象不存在: {rpn} ({str(e)[:120]})")
        if int(st.size) != int(row["size_bytes"]):
            return ApiResponse(ok=False, error=f"MinIO 大小与会话不符: {rpn}")
        sum_stat += int(st.size)
        seen_paths.add(rpn)
        manifest_rows.append(
            {"relative_path": rpn, "size_bytes": int(st.size), "object_key": row["object_key"]}
        )
    if seen_paths != set(by_rel.keys()):
        return ApiResponse(ok=False, error="manifest.paths 必须与会话路径集合完全一致")
    tol2 = _direct_upload_size_tolerance(int(man.total_size_bytes))
    if abs(sum_stat - int(man.total_size_bytes)) > tol2:
        return ApiResponse(ok=False, error="对象实际大小合计与 manifest.total_size_bytes 不一致")

    prefix_uri = build_minio_prefix_uri(us.bucket, us.object_key)
    ex = (
        await db.execute(
            select(DataAsset)
            .where(DataAsset.file_path == prefix_uri, DataAsset.project_id == us.project_id)
            .limit(1)
        )
    ).scalars().first()
    if ex:
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="completed")
        )
        await db.commit()
        return ApiResponse(
            ok=True,
            data={"asset": _data_asset_response_for_viewer(ex, viewer).model_dump(mode="json")},
        )

    tmp_d = tempfile.mkdtemp(prefix="data_assets_direct_dir_")
    aid: Optional[int] = None
    try:
        try:
            local_root = await asyncio.to_thread(download_by_minio_uri, prefix_uri, tmp_d)
        except MinioBucketError as e:
            await db.execute(
                update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
            )
            await db.commit()
            return ApiResponse(ok=False, error=f"下载目录失败: {str(e)[:200]}")
        is_lerobot = await asyncio.to_thread(_inspect_lerobot_dir, local_root)
        if is_lerobot:
            meta_json, parse_status, err_msg = parse_meta_for_asset(local_root, "lerobot")
            fmt = "lerobot"
            dataset_shape = "lerobot"
        else:
            meta_json = None
            parse_status = "未解析"
            err_msg = "目录直传：未识别为 LeRobot 数据集结构（需含 meta、data、videos 等约定布局）"
            fmt = "directory"
            dataset_shape = "generic_directory"
        storage_prefix = us.object_key.rstrip("/") + "/"
        merged = _merge_directory_asset_meta(
            meta_json,
            prefix_uri,
            us.bucket,
            storage_prefix,
            sess_root,
            manifest_rows,
            int(man.total_files),
            int(man.total_size_bytes),
            dataset_shape=dataset_shape,
        )
        code = await next_code(db)
        cd = DataAssetCreate(
            code=code,
            filename=us.asset_name or us.filename or sess_root,
            format=fmt,
            source="import",
            project_id=us.project_id,
            project_name=proj_name,
            file_path=prefix_uri,
            file_size_bytes=int(man.total_size_bytes),
            meta=merged,
            parse_status=parse_status,
            error_msg=err_msg,
            sync_status="synced",
            sync_error=None,
            operator_name=(viewer or "").strip() or None,
        )
        asset = await create_asset(db, cd)
        aid = asset.id
        one = {"asset": _data_asset_response_for_viewer(asset, viewer).model_dump(mode="json")}
        await db.execute(
            update(DataAssetUploadSession)
            .where(DataAssetUploadSession.id == sid)
            .values(
                status="completed",
                result_payload_json=json.dumps(one, ensure_ascii=False, default=str),
            )
        )
        await db.commit()
        return ApiResponse(ok=True, data=one)
    except Exception as e:
        logger.exception("upload-complete directory failed")
        if aid is not None:
            try:
                await delete_asset(db, aid)
            except Exception:
                pass
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
        )
        await db.commit()
        return ApiResponse(ok=False, error=str(e)[:200])
    finally:
        shutil.rmtree(tmp_d, ignore_errors=True)


@router.get("/upload-sessions", response_model=ApiResponse)
async def list_data_asset_upload_sessions(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """
    列出当前用户的浏览器直传上传会话（upload_sessions），供任务中心刷新后按真实状态恢复导入任务。
    仅返回 user_id 命中且项目在当前用户数据资产可见范围内的会话。
    """
    uid = str(current_user.id)
    allowed = await data_assets_allowed_project_ids(db, current_user)
    stmt = select(DataAssetUploadSession).where(DataAssetUploadSession.user_id == uid)
    if allowed is not None:
        if not allowed:
            return ApiResponse(ok=True, data={"items": []})
        stmt = stmt.where(DataAssetUploadSession.project_id.in_(allowed))
    stmt = stmt.order_by(DataAssetUploadSession.created_at.desc()).limit(int(limit))
    rows = (await db.execute(stmt)).scalars().all()
    items: List[Dict[str, Any]] = []
    for r in rows:
        exp = r.expires_at
        if exp is not None and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        items.append(
            {
                "upload_session_id": r.id,
                "project_id": r.project_id or "",
                "status": (r.status or "").strip(),
                "upload_mode": (getattr(r, "upload_mode", None) or "single_file").strip(),
                "filename": (r.filename or "").strip(),
                "expected_count": getattr(r, "expected_count", None),
                "size_bytes": int(r.size_bytes or 0),
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "expires_at": exp.isoformat() if exp else None,
                "root_dir_name": (getattr(r, "root_dir_name", None) or None),
                "asset_name": (getattr(r, "asset_name", None) or None),
            }
        )
    return ApiResponse(ok=True, data={"items": items})


@router.post("/upload-sessions/{upload_session_id}/cancel", response_model=ApiResponse)
async def cancel_data_asset_upload_session(
    upload_session_id: str,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """取消浏览器直传上传会话，供任务中心“取消导入”使用。"""
    sid = (upload_session_id or "").strip()
    if not sid:
        return ApiResponse(ok=False, error="upload_session_id 不能为空")
    row = (
        await db.execute(select(DataAssetUploadSession).where(DataAssetUploadSession.id == sid))
    ).scalar_one_or_none()
    if row is None:
        return ApiResponse(ok=False, error="上传会话不存在")
    uid = str(getattr(current_user, "id", "") or "")
    if str(getattr(row, "user_id", "") or "") != uid:
        return ApiResponse(ok=False, error="上传会话不匹配")

    st = (getattr(row, "status", None) or "").strip().lower()
    if st == "completed":
        return ApiResponse(ok=False, error="上传会话已完成，不能取消")
    if st == "cancelled":
        return ApiResponse(ok=True, data={"upload_session_id": sid, "status": "cancelled"})

    await db.execute(
        update(DataAssetUploadSession)
        .where(DataAssetUploadSession.id == sid)
        .values(status="cancelled")
    )
    await db.commit()
    return ApiResponse(ok=True, data={"upload_session_id": sid, "status": "cancelled"})


@router.post("/upload-complete", response_model=ApiResponse)
async def data_assets_upload_complete(
    body: DirectUploadCompleteBody = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """直传完成：single_file / multi_file / directory。"""
    sid = (body.upload_session_id or "").strip()
    if not sid:
        return ApiResponse(ok=False, error="upload_session_id 不能为空")
    logger.info("upload-complete request: upload_session_id=%s", sid)

    body_sz_opt: Optional[int] = None
    if body.size_bytes is not None:
        try:
            body_sz_opt = int(body.size_bytes)
        except (TypeError, ValueError):
            return ApiResponse(ok=False, error="size_bytes 无效")
        if body_sz_opt <= 0:
            return ApiResponse(ok=False, error="size_bytes 无效")

    result = await db.execute(select(DataAssetUploadSession).where(DataAssetUploadSession.id == sid))
    us = result.scalar_one_or_none()
    if not us:
        return ApiResponse(ok=False, error="上传会话不存在")

    uid = str(current_user.id)
    if us.user_id != uid:
        return ApiResponse(ok=False, error="上传会话不匹配")

    mode = (getattr(us, "upload_mode", None) or "single_file").strip().lower()
    if mode == "single_file" and body_sz_opt is None:
        return ApiResponse(ok=False, error="single_file 需提供 size_bytes")

    now = datetime.now(timezone.utc)
    exp = us.expires_at
    if exp is not None and exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)

    uname0 = getattr(current_user, "username", None)
    viewer0 = uname0.strip() if isinstance(uname0, str) else None

    if us.status == "completed":
        if mode == "multi_file" and (us.result_payload_json or "").strip():
            try:
                cached = json.loads(us.result_payload_json or "{}")
                return ApiResponse(ok=True, data=cached)
            except Exception:
                pass
        if mode == "directory":
            prefix_uri = build_minio_prefix_uri(us.bucket, us.object_key)
            da = (
                await db.execute(
                    select(DataAsset)
                    .where(DataAsset.file_path == prefix_uri, DataAsset.project_id == us.project_id)
                    .limit(1)
                )
            ).scalars().first()
            if da:
                return ApiResponse(
                    ok=True,
                    data={"asset": _data_asset_response_for_viewer(da, viewer0).model_dump()},
                )
            if (us.result_payload_json or "").strip():
                try:
                    return ApiResponse(ok=True, data=json.loads(us.result_payload_json or "{}"))
                except Exception:
                    pass
        minio_done = build_minio_uri(us.bucket, us.object_key)
        done_asset = (
            await db.execute(
                select(DataAsset)
                .where(
                    DataAsset.file_path == minio_done,
                    DataAsset.project_id == us.project_id,
                )
                .limit(1)
            )
        ).scalars().first()
        if not done_asset:
            return ApiResponse(ok=False, error="会话已完成但未找到对应资产，请联系管理员")
        return ApiResponse(
            ok=True,
            data={"asset": _data_asset_response_for_viewer(done_asset, viewer0).model_dump()},
        )

    if us.status == "expired":
        return ApiResponse(ok=False, error="上传会话已过期，请重新发起上传")

    if us.status == "presigned" and exp is not None and now > exp:
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="expired")
        )
        await db.commit()
        return ApiResponse(ok=False, error="上传会话已过期，请重新发起上传")

    if us.status not in ("presigned", "failed"):
        return ApiResponse(ok=False, error="该会话不可用")

    p2, werr = await assert_may_write_project_for_data_asset_import(db, current_user, us.project_id)
    if werr or not p2:
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
        )
        await db.commit()
        return ApiResponse(ok=False, error=werr or "项目校验失败")

    proj_name = (p2.name or us.project_id).strip()

    if mode == "multi_file":
        return await _upload_complete_multi_file(db, us, sid, proj_name, viewer0)

    if mode == "directory":
        return await _upload_complete_directory(db, body, us, sid, proj_name, viewer0)

    assert body_sz_opt is not None
    if body_sz_opt != int(us.size_bytes):
        return ApiResponse(ok=False, error="文件大小与会话登记不一致")

    try:
        st = await asyncio.to_thread(stat_object, us.bucket, us.object_key)
    except MinioBucketError as e:
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
        )
        await db.commit()
        return ApiResponse(ok=False, error=str(e)[:200])

    if int(st.size) != int(us.size_bytes):
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
        )
        await db.commit()
        return ApiResponse(
            ok=False,
            error=f"对象大小 {st.size} 与声明的 {us.size_bytes} 不一致",
        )

    minio_uri = build_minio_uri(us.bucket, us.object_key)

    exists_asset = (
        await db.execute(
            select(DataAsset)
            .where(DataAsset.file_path == minio_uri, DataAsset.project_id == us.project_id)
            .limit(1)
        )
    ).scalars().first()
    if exists_asset:
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="completed")
        )
        await db.commit()
        return ApiResponse(
            ok=True,
            data={"asset": _data_asset_response_for_viewer(exists_asset, viewer0).model_dump()},
        )

    fmt = _format_from_filename(us.filename)
    if not fmt:
        await db.execute(
            update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
        )
        await db.commit()
        return ApiResponse(ok=False, error="无法识别文件格式")

    tmp_dir = tempfile.mkdtemp(prefix="data_assets_direct_upload_")
    try:
        try:
            local_parse_path = await asyncio.to_thread(download_by_minio_uri, minio_uri, tmp_dir)
        except MinioBucketError as e:
            await db.execute(
                update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
            )
            await db.commit()
            return ApiResponse(ok=False, error=f"校验对象失败: {str(e)[:200]}")

        asset_id: Optional[int] = None
        try:
            code = await next_code(db)
            create_data = DataAssetCreate(
                code=code,
                filename=us.filename,
                format=fmt,
                source="import",
                project_id=us.project_id,
                project_name=proj_name,
                file_path=minio_uri,
                file_size_bytes=int(st.size),
                meta=None,
                parse_status="解析中",
                error_msg=None,
                sync_status="unsynced",
                sync_error="直传对象已就绪，正在写入元数据",
                operator_name=(viewer0 or "").strip() or None,
            )
            asset = await create_asset(db, create_data)
            asset_id = asset.id
            meta_json, parse_status, err_msg = parse_meta_for_asset(local_parse_path, fmt)
            await _apply_import_storage_meta(
                db, asset.id, minio_uri, minio_uri, meta_json, parse_status, err_msg
            )
            await update_asset(db, asset.id, sync_status="synced", sync_error=None)
            ref_pl = await get_asset_by_id(db, asset_id)
            pl = json.dumps(
                {"asset": _data_asset_response_for_viewer(ref_pl, viewer0).model_dump(mode="json")},
                ensure_ascii=False,
            )
            await db.execute(
                update(DataAssetUploadSession)
                .where(DataAssetUploadSession.id == sid)
                .values(status="completed", result_payload_json=pl)
            )
            await db.commit()
        except Exception as e:
            logger.exception("upload-complete: register asset failed")
            if asset_id is not None:
                try:
                    await delete_asset(db, asset_id)
                except Exception:
                    logger.exception("upload-complete: delete_asset rollback failed")
            try:
                await db.execute(
                    update(DataAssetUploadSession).where(DataAssetUploadSession.id == sid).values(status="failed")
                )
                await db.commit()
            except Exception:
                await db.rollback()
            return ApiResponse(ok=False, error=str(e)[:200])

        refreshed = await get_asset_by_id(db, asset_id)
        payload = _data_asset_response_for_viewer(refreshed, viewer0).model_dump()
        return ApiResponse(ok=True, data={"asset": payload})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _inspect_lerobot_dir(resolved: str) -> bool:
    """目录是否为有效 LeRobot：2+ 个 meta/data/videos 或存在 meta 文件。"""
    p = Path(resolved)
    if not p.is_dir():
        return False
    has_meta = (p / "meta").is_dir()
    has_data = (p / "data").is_dir()
    has_videos = (p / "videos").is_dir()
    if sum([has_meta, has_data, has_videos]) >= 2:
        return True
    if (p / "meta.json").is_file() or (p / "dataset_info.json").is_file():
        return True
    if has_meta and (p / "meta" / "info.json").is_file():
        return True
    return False


def _normalize_format(fmt: Optional[str], file_path: str) -> str:
    """统一为 hdf5 | mcap | lerobot。"""
    if fmt and fmt.strip():
        return fmt.strip().lower()
    return _format_from_filename(os.path.basename(file_path)) or "hdf5"


def _platform_annotation_path(asset: DataAsset) -> Optional[str]:
    """平台标注文件：与资产同目录的 instruction.json 或 instructions.json。"""
    fp = (asset.file_path or "").strip()
    if not fp:
        return None
    if os.path.isfile(fp):
        base_dir = os.path.dirname(fp)
    elif os.path.isdir(fp):
        base_dir = fp
    else:
        return None
    for name in ("instruction.json", "instructions.json"):
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _file_size_display(bytes_val: Optional[int]) -> str:
    """文件大小展示：与前端一致，MB/GB。"""
    if not bytes_val:
        return "—"
    mb = bytes_val / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def _conversion_record_display(asset: DataAsset) -> str:
    """转换记录列：未转换 或 已转换为 HDF5/MCAP/LeRobot。"""
    src = (asset.source or "").strip().lower()
    if src != "convert":
        return "无"
    fmt = _normalize_format(asset.format, asset.file_path or "")
    if fmt == "hdf5":
        return "已转换为 HDF5"
    if fmt == "mcap":
        return "已转换为 MCAP"
    if fmt == "lerobot":
        return "已转换为 LeRobot"
    return "已转换"


def _build_export_zip(assets: List[DataAsset], compression_mode: Optional[str] = None) -> Tuple[bytes, str]:
    """
    根据资产列表构建导出 zip，返回 (zip_bytes, zip_filename)。
    仅支持同一种数据格式；调用方已校验。复用 _build_export_tree。
    """
    if not assets:
        raise ValueError("资产列表为空")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_root_name = f"export_{ts}"
    zip_filename = f"export_{ts}.zip"
    tmp_dir = tempfile.mkdtemp(prefix="data_assets_export_")
    try:
        export_root = Path(tmp_dir) / export_root_name
        _build_export_tree(assets, export_root)
        buf = io.BytesIO()
        zip_kwargs = _zipfile_params_for_mode(compression_mode)
        with zipfile.ZipFile(buf, "w", **zip_kwargs) as zf:
            for item in export_root.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(export_root.parent)
                    zf.write(item, arcname.as_posix())
        buf.seek(0)
        return buf.getvalue(), zip_filename
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/register", response_model=ApiResponse)
async def register_asset(
    http_request: Request,
    background_tasks: BackgroundTasks,
    body: RegisterAssetRequest = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """
    【服务器侧登记 / 工具接口】从平台资源浏览器选择白名单内路径登记资产（不复制本机文件）。
    普通用户导入请走 POST /api/data-assets/import（浏览器上传 + MinIO）。
    type=file：后缀 mcap/hdf5/h5；type=dir：LeRobot 目录结构。
    """
    raw_reg = (body.project_id or "").strip()
    if not raw_reg:
        return ApiResponse(ok=False, error="请选择所属项目")
    p, reg_err = await assert_may_write_project_for_data_asset_import(db, current_user, raw_reg)
    if reg_err or not p:
        return ApiResponse(ok=False, error=reg_err or "项目校验失败")
    project_id = str(p.id)
    project_name = (body.project_name or p.name or project_id).strip()
    try:
        resolved = validate_path_whitelist(body.path)
    except HTTPException as e:
        return ApiResponse(ok=False, error=(e.detail if isinstance(e.detail, str) else str(e.detail))[:200])
    except Exception as e:
        return ApiResponse(ok=False, error=str(e)[:200])

    if body.type == "file":
        if not os.path.isfile(resolved):
            return ApiResponse(ok=False, error="路径不是文件")
        ext = Path(resolved).suffix.lower()
        if ext not in (".mcap", ".hdf5", ".h5"):
            return ApiResponse(ok=False, error="仅支持 .mcap、.hdf5、.h5 文件")
        fmt = "mcap" if ext == ".mcap" else "hdf5"
        size = _get_path_size_bytes(resolved)
        filename = os.path.basename(resolved)
    elif body.type == "dir":
        if not os.path.isdir(resolved):
            return ApiResponse(ok=False, error="路径不是目录")
        if not _inspect_lerobot_dir(resolved):
            return ApiResponse(ok=False, error="当前目录不是有效的 LeRobot 数据集")
        fmt = "lerobot"
        filename = os.path.basename(resolved).rstrip("/") or "dataset"
        size = _get_path_size_bytes(resolved)
    else:
        return ApiResponse(ok=False, error="type 必须为 file 或 dir")

    # 仅限制“同一项目”内重复登记；允许同一条数据绑定到不同项目
    exists = (await db.execute(
        select(DataAsset).where(DataAsset.file_path == resolved, DataAsset.project_id == project_id)
    )).scalar_one_or_none()
    if exists:
        return ApiResponse(ok=False, error="该路径在当前项目下已登记，不能重复导入")

    try:
        if body.type == "file":
            object_key = f"projects/{project_id}/import/{uuid.uuid4().hex[:12]}/{filename}"
            minio_uri = upload_file_to_project_bucket(
                project_name=(p.name or project_id),
                local_file_path=resolved,
                object_name=object_key,
            )
        else:
            object_prefix = f"projects/{project_id}/import/{uuid.uuid4().hex[:12]}/{filename}"
            minio_uri = upload_dir_to_project_bucket(
                project_name=(p.name or project_id),
                local_dir_path=resolved,
                object_prefix=object_prefix,
            )
    except MinioBucketError as e:
        return ApiResponse(ok=False, error=str(e)[:200])

    code = await next_code(db)
    create_data = DataAssetCreate(
        code=code,
        filename=filename,
        format=fmt,
        source="import",
        project_id=project_id,
        project_name=project_name,
        file_path=resolved,
        file_size_bytes=size,
        meta=None,
        parse_status="解析中",
        error_msg=None,
        sync_status="synced",
        sync_error=None,
        operator_name=(getattr(current_user, "username", None) or "").strip() or None,
    )
    asset = await create_asset(db, create_data)
    meta_json, parse_status, err_msg = parse_meta_for_asset(resolved, fmt)
    await _apply_import_storage_meta(db, asset.id, resolved, minio_uri, meta_json, parse_status, err_msg)
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=http_request,
        action_type=AA.IMPORT_DATA_ASSET,
        project_id=project_id,
        project_name=project_name,
        resource_type=AR.DATA_ASSET,
        resource_id=str(asset.id),
        resource_name=filename,
        detail_json={"path": body.path, "type": body.type, "minio_registered": True},
    )
    return ApiResponse(ok=True, data={"id": asset.id, "name": filename, "minio_path": minio_uri})


def _looks_like_bag_data_file(p: str) -> bool:
    low = (p or "").strip().lower()
    return (
        low.endswith(".mcap")
        or low.endswith(".mca")
        or low.endswith(".db3")
        or low.endswith(".bag")
    )


def _dir_has_collect_episode_markers(d: str) -> bool:
    """与采集端 / routes_script.delete_data 对齐：目录内含 bag 或质检报告则视为可整夹删除的 episode。"""
    try:
        if not os.path.isdir(d):
            return False
        for name in os.listdir(d):
            low = name.lower()
            if low.endswith((".mcap", ".mca", ".db3")):
                full = os.path.join(d, name)
                if os.path.isfile(full):
                    return True
        if os.path.isfile(os.path.join(d, "validation_report.json")):
            return True
    except OSError:
        return False
    return False


def _episode_delete_target_local(file_path: str) -> str:
    """
    登记路径为具体 bag 文件时，目标为其所在 episode 目录（整夹语义）。
    - 删除：与 /api/script/data DELETE 一致；
    - 同步到 MinIO：采集端按目录上传时上传整夹内所有文件（见 Agent _upload_path_to_minio）。
    """
    fp = (file_path or "").strip()
    if not fp or fp.startswith("minio://"):
        return fp
    norm = os.path.normpath(fp.replace("\\", "/"))
    if os.path.isdir(norm):
        return norm
    if _looks_like_bag_data_file(norm):
        parent = os.path.dirname(norm)
        # 避免 /x.mcap 的父目录为根目录时整夹删除
        root = os.path.abspath(os.sep)
        if parent and parent not in ("/", "\\", root) and not (
            len(parent) == 3 and parent[1] == ":" and parent[2] in "/\\"
        ):
            return parent
        return norm
    return norm


def _delete_local_asset_files_best_effort(asset: DataAsset) -> None:
    """delete_file=true：删除采集 episode 整目录（若符合标记），避免仅删单个 mcap 残留文件夹。"""
    fp = (getattr(asset, "file_path", None) or "").strip()
    if not fp or fp.startswith("minio://"):
        return
    target = _episode_delete_target_local(fp)
    if not target:
        return
    try:
        ap = os.path.abspath(os.path.normpath(target.replace("\\", "/")))
        root = os.path.abspath(os.sep)
        if ap == root:
            return
        if os.path.isdir(ap):
            if _dir_has_collect_episode_markers(ap):
                shutil.rmtree(ap, ignore_errors=True)
            return
        if os.path.isfile(ap):
            parent = os.path.dirname(ap)
            if parent and os.path.isdir(parent) and _dir_has_collect_episode_markers(parent):
                shutil.rmtree(parent, ignore_errors=True)
            else:
                try:
                    os.unlink(ap)
                except Exception:
                    pass
    except Exception:
        pass


async def _resolve_agent_id_for_tunnel_by_device_id(db_main: AsyncSession, device_id: Optional[str]) -> Optional[str]:
    """根据 platform devices.id 尽量解析到在线采集端 agent_id。"""
    did_raw = (device_id or "").strip() if device_id is not None else ""
    if not did_raw:
        return None
    try:
        did = int(did_raw)
    except ValueError:
        return None
    try:
        info = agent_registry.get_by_device_id_strict(did)
        aid = getattr(info, "agent_id", None) if info else None
        if aid:
            return str(aid).strip()
    except Exception:
        pass
    # 回退：通过 devices.hardware_uuid 反查 agent_registry
    try:
        dev = await get_device_by_id(db_main, did)
        hw = str(getattr(dev, "hardware_uuid", "") or "").strip() if dev else ""
        if not hw:
            return None
        agent = agent_registry.get_by_id(hw)
        if agent and getattr(agent, "agent_id", None):
            try:
                agent_registry.bind_device_to_agent(device_id=did, agent_id=hw)
            except Exception:
                pass
            return str(agent.agent_id).strip()
    except Exception:
        return None
    return None


async def _try_remote_delete_collect_asset(
    *,
    db_main: AsyncSession,
    asset: DataAsset,
    timeout_sec: float = 30.0,
    allow_when_synced: bool = False,
) -> tuple[bool, str]:
    """尝试通过采集端 Agent 删除采集端远程目录/文件。

    - 未同步（unsynced）：默认可删。
    - 已同步（synced）：仅当 allow_when_synced=True 时执行（与云端副本独立）。
    """
    if not asset:
        return False, "asset missing"
    if (asset.source or "").lower() != "collect":
        return True, "skip (not collect)"

    ss = (asset.sync_status or "").lower()
    if ss == "unsynced":
        pass
    elif ss == "synced" and allow_when_synced:
        pass
    else:
        return True, f"skip (sync_status={ss or 'unknown'}, remote_delete_not_applicable)"

    if not getattr(asset, "file_path", None):
        return False, "asset.file_path missing"

    agent_id = await _resolve_agent_id_for_tunnel_by_device_id(db_main, getattr(asset, "device_id", None))
    if not agent_id:
        return False, "no agent_id resolved for device"
    did_for_tunnel: Optional[int] = None
    try:
        raw_did = getattr(asset, "device_id", None)
        if raw_did is not None:
            did_for_tunnel = int(str(raw_did).strip())
    except Exception:
        did_for_tunnel = None
    socket_key = await agent_tunnel_manager.resolve_connected_socket_key(
        agent_id,
        platform_device_id=did_for_tunnel,
    )
    if not socket_key:
        return False, f"agent tunnel not connected (agent_id={agent_id})"

    # 已同步后 file_path 多为 minio://，不得作为 SCRIPT_DELETE_DATA 的路径发给采集端；
    # 须用 meta.storage.backend_local_path（同步时 merge_storage_meta 已写入），否则采集端报 Path does not exist。
    raw_collect = (_extract_backend_local_path(getattr(asset, "meta", None)) or "").strip()
    fp_main = (getattr(asset, "file_path", None) or "").strip()
    if not raw_collect and fp_main and not fp_main.startswith("minio://"):
        raw_collect = fp_main
    if not raw_collect:
        return False, "缺少采集端可删路径：已同步数据需 meta.storage.backend_local_path，或未同步时 file_path 为本地路径"

    ep = _episode_delete_target_local(raw_collect)
    root = os.path.abspath(os.sep)
    if not ep or ep in ("/", root):
        ep = raw_collect
    result = await agent_tunnel_manager.send_cmd_and_wait(
        agent_id=socket_key,
        cmd="SCRIPT_DELETE_DATA",
        payload={"path": ep},
        timeout_sec=timeout_sec,
        retry_times=1,
    )
    ok = bool(result.get("success", False))
    if ok:
        return True, result.get("msg") or result.get("message") or "ok"
    return False, result.get("msg") or result.get("message") or "remote delete failed"


def _collect_remote_delete_failure_allows_db_cleanup(msg: str) -> bool:
    """
    采集端 SCRIPT_DELETE_DATA 失败时，若属于「路径已不存在 / 无效」等，仍允许删除平台 data_assets 记录，
    并向用户返回提示文案。
    """
    m = (msg or "").strip().lower()
    if not m:
        return False
    if "asset.file_path missing" in m:
        return True
    needles = (
        "path does not exist",
        "does not exist",
        "不存在",
        "no such file",
        "not found",
        "enoent",
        "invalid path",
        "路径无效",
        "找不到",
        "refuse to delete non-bag",
        "non-bag dir",
        "无法删除",
    )
    return any(x in m for x in needles)


async def _delete_data_asset_row_and_reconcile_collect_job(
    db: AsyncSession,
    asset: DataAsset,
    asset_id: int,
) -> bool:
    """删除 data_assets 行；若为采集资产且 meta 含 job_id，则回写对应采集作业进度。"""
    linked_job_id = extract_collect_job_id_from_asset(asset)
    ok = await delete_asset(db, asset_id)
    if ok and linked_job_id:
        await reconcile_collection_job_progress_from_data_assets(db, linked_job_id)
    return ok


async def _apply_data_asset_delete(
    *,
    db: AsyncSession,
    db_main: AsyncSession,
    asset: DataAsset,
    asset_id: int,
    delete_file: bool,
    delete_cloud: bool,
    delete_remote: bool,
) -> tuple[bool, Optional[str], Optional[str]]:
    """按来源与同步状态执行删除。

    返回 (成功, 错误信息, 提示信息)：成功时若有「采集端路径无效但仍删除平台记录」则 third 非空。
    """
    delete_warning: Optional[str] = None
    src_collect = (asset.source or "").lower() == "collect"
    ss = (asset.sync_status or "").lower()
    synced_collect = src_collect and ss == "synced"

    if not src_collect:
        if not delete_cloud:
            return False, "非采集来源仅支持删除云端记录", None
        minio_path = _minio_uri_for_storage_ops(asset)
        if minio_path:
            try:
                delete_by_minio_uri(minio_path)
            except MinioBucketError as e:
                return False, f"MinIO: {str(e)[:200]}", None
        if delete_file:
            _delete_local_asset_files_best_effort(asset)
        ok = await delete_asset(db, asset_id)
        return (True, None, None) if ok else (False, "删除失败", None)

    if not synced_collect:
        okr, msg = await _try_remote_delete_collect_asset(
            db_main=db_main,
            asset=asset,
            allow_when_synced=False,
        )
        if not okr:
            if _collect_remote_delete_failure_allows_db_cleanup(msg):
                delete_warning = (
                    "采集端路径无效或文件已不存在，已跳过远端删除并删除平台记录。"
                    f"（详情：{msg[:160]}）"
                )
            else:
                return False, msg, None
        minio_path = _minio_uri_for_storage_ops(asset)
        if minio_path:
            try:
                delete_by_minio_uri(minio_path)
            except MinioBucketError as e:
                return False, f"MinIO: {str(e)[:200]}", None
        if delete_file:
            _delete_local_asset_files_best_effort(asset)
        ok = await _delete_data_asset_row_and_reconcile_collect_job(db, asset, asset_id)
        return (True, None, delete_warning) if ok else (False, "删除失败", None)

    if not delete_cloud and not delete_remote:
        return False, "请至少选择删除云端或采集端一项", None
    if delete_remote:
        okr, msg = await _try_remote_delete_collect_asset(
            db_main=db_main,
            asset=asset,
            allow_when_synced=True,
        )
        if not okr:
            if _collect_remote_delete_failure_allows_db_cleanup(msg):
                delete_warning = (
                    "采集端路径无效或文件已不存在，已跳过远端删除。"
                    f"（详情：{msg[:160]}）"
                )
            else:
                return False, msg, None
    if delete_cloud:
        minio_path = _minio_uri_for_storage_ops(asset)
        if minio_path:
            try:
                delete_by_minio_uri(minio_path)
            except MinioBucketError as e:
                return False, f"MinIO: {str(e)[:200]}", None
        if delete_file:
            _delete_local_asset_files_best_effort(asset)
        ok = await _delete_data_asset_row_and_reconcile_collect_job(db, asset, asset_id)
        return (True, None, delete_warning) if ok else (False, "删除失败", None)
    # 已同步：仅删除采集端目录、保留云端与平台记录；远端删除失败若为路径无效则仍递减作业计数。
    linked_job_id = extract_collect_job_id_from_asset(asset)
    if linked_job_id:
        await decrement_collection_job_completed_for_removed_episode(db, linked_job_id)
    return True, None, delete_warning


@router.post("/delete-batch", response_model=ApiResponse)
async def delete_data_assets_batch(
    http_request: Request,
    background_tasks: BackgroundTasks,
    body: DeleteAssetsBatchBody,
    db: AsyncSession = Depends(get_data_assets_db),
    db_main: AsyncSession = Depends(get_main_db),
    current_user: User = Depends(get_current_user),
):
    """批量删除资产（单接口），并记 BATCH_DELETE_DATA_ASSET 审计。"""
    if user_cannot_delete_data_asset(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前角色无权限删除数据资产",
        )
    if not body.asset_ids:
        return ApiResponse(ok=False, error="请至少选择一个资产")
    delete_remote = bool(getattr(body, "delete_remote", False))
    delete_cloud = bool(getattr(body, "delete_cloud", True))
    delete_file = bool(getattr(body, "delete_file", False))
    seen: set[int] = set()
    ids: List[int] = []
    for aid in body.asset_ids:
        if aid not in seen:
            seen.add(aid)
            ids.append(aid)

    deleted_ids: List[int] = []
    deleted_names: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []
    project_ids: set[str] = set()
    for asset_id in ids:
        asset = await get_asset_by_id(db, asset_id)
        if not asset:
            errors.append(f"id={asset_id} 不存在")
            continue
        if not await data_asset_visible_to_user(db, current_user, asset):
            errors.append(f"id={asset_id} 无权限")
            continue
        if asset.project_id:
            project_ids.add(str(asset.project_id))
        ok_del, err, warn = await _apply_data_asset_delete(
            db=db,
            db_main=db_main,
            asset=asset,
            asset_id=asset_id,
            delete_file=delete_file,
            delete_cloud=delete_cloud,
            delete_remote=delete_remote,
        )
        if not ok_del:
            errors.append(f"id={asset_id} {err or '删除失败'}")
            continue
        deleted_ids.append(asset_id)
        if warn:
            warnings.append(f"id={asset_id} {warn}")
        fn = (getattr(asset, "filename", None) or "").strip()
        if fn:
            deleted_names.append(fn)

    n_del = len(deleted_ids)
    batch_rid = f"batch:{n_del}" if n_del else "batch:0"
    pname_for_audit: Optional[str] = None
    if len(project_ids) == 1:
        pid_one = next(iter(project_ids))
        pr = (await db.execute(select(Project).where(Project.id == pid_one))).scalar_one_or_none()
        if pr is not None:
            pname_for_audit = getattr(pr, "name", None)
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=http_request,
        action_type=AA.BATCH_DELETE_DATA_ASSET,
        project_id=next(iter(project_ids), None) if project_ids else None,
        project_name=pname_for_audit,
        resource_type=AR.DATA_ASSET,
        resource_id=batch_rid,
        resource_name=f"共{n_del}条数据资产" if n_del else "批量删除数据资产",
        detail_json={
            "requested_count": len(ids),
            "deleted_count": n_del,
            "deleted_asset_ids": deleted_ids[:80],
            "deleted_filenames": deleted_names[:30],
            "errors": errors[:20],
            "delete_cloud": delete_cloud,
            "delete_remote": delete_remote,
            "delete_file": delete_file,
            "warnings": warnings[:30],
        },
        result="SUCCESS" if not errors else ("FAIL" if not deleted_ids else "SUCCESS"),
    )
    return ApiResponse(
        ok=True,
        data={"deleted": deleted_ids, "errors": errors, "warnings": warnings},
        warning="；".join(warnings[:8]) if warnings else None,
    )


@router.delete("/{asset_id}", response_model=ApiResponse)
async def delete_data_asset(
    asset_id: int,
    http_request: Request,
    background_tasks: BackgroundTasks,
    delete_file: bool = Query(False, description="是否同时删除本地文件"),
    delete_remote: bool = Query(False, description="是否删除采集端远程数据（已同步采集资产可与 delete_cloud 分项勾选）"),
    delete_cloud: bool = Query(True, description="是否删除云端 MinIO 与平台记录"),
    db: AsyncSession = Depends(get_data_assets_db),
    db_main: AsyncSession = Depends(get_main_db),
    current_user: User = Depends(get_current_user),
):
    """删除资产：采集+已同步时可单独删除云端或采集端；采集+未同步为整单删除。"""
    if user_cannot_delete_data_asset(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前角色无权限删除数据资产",
        )
    asset = await get_asset_by_id(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="资产不存在")
    if not await data_asset_visible_to_user(db, current_user, asset):
        raise HTTPException(status_code=404, detail="资产不存在")

    ok_del, err, warn = await _apply_data_asset_delete(
        db=db,
        db_main=db_main,
        asset=asset,
        asset_id=asset_id,
        delete_file=delete_file,
        delete_cloud=delete_cloud,
        delete_remote=delete_remote,
    )
    if not ok_del:
        detail = err or "删除失败"
        if detail.startswith("请至少") or detail.startswith("非采集来源"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail[:300])
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail[:300])

    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=http_request,
        action_type=AA.DELETE_DATA_ASSET,
        project_id=(asset.project_id or None),
        project_name=getattr(asset, "project_name", None),
        resource_type=AR.DATA_ASSET,
        resource_id=str(asset_id),
        resource_name=(getattr(asset, "filename", None) or str(asset_id)),
        detail_json={
            "delete_file": bool(delete_file),
            "delete_cloud": bool(delete_cloud),
            "delete_remote": bool(delete_remote),
            "remote_path_warning": warn[:400] if warn else None,
        },
    )
    return ApiResponse(ok=True, data=None, warning=warn)


@router.post("/reparse-from-minio/batch", response_model=ApiResponse)
async def batch_reparse_data_assets_from_minio(
    body: ReparseFromMinioBatchBody = Body(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    对已同步到 MinIO 的历史数据重新解析（如仍显示「等待落盘后再解析」）。
    默认仅处理 error_msg 含「等待落盘」的资产；超级管理员可将 stale_only=false 以处理可见范围内全部 minio+mcap/hdf5。
    """
    aids = body.asset_ids if body.asset_ids else None
    if aids is not None and len(aids) == 0:
        aids = None

    if not body.stale_only and aids is None:
        if not is_super_admin(current_user.role):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="关闭 stale_only 时仅超级管理员可执行")

    allowed = await data_assets_allowed_project_ids(db, current_user)
    if body.project_id:
        pid = body.project_id.strip()
        if allowed is not None and pid not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权限访问该项目")

    use_stale = bool(body.stale_only) and aids is None
    res = await batch_refresh_parse_from_minio(
        db,
        allowed_project_ids=allowed,
        project_id=body.project_id,
        limit=body.limit,
        stale_error_filter=use_stale,
        asset_ids=aids,
    )
    return ApiResponse(ok=True, data=res)


@router.post("/{asset_id}/reparse-from-minio", response_model=ApiResponse)
async def reparse_single_data_asset_from_minio(
    asset_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """对单条已同步资产从 MinIO 拉取并重新解析 meta / parse_status。"""
    asset = await get_asset_by_id(db, asset_id)
    if not asset or not await data_asset_visible_to_user(db, current_user, asset):
        raise HTTPException(status_code=404, detail="资产不存在")
    res = await refresh_parse_from_minio_for_asset(db, asset)
    return ApiResponse(ok=True, data=res)


@router.post("/{asset_id}/sync", response_model=ApiResponse)
async def sync_data_asset(
    asset_id: int,
    agent_id: Optional[str] = Query(
        None,
        description="可选，指定采集端，取值为设备的 hardware_uuid（与设备连接时落库一致）；多台设备时建议传入",
    ),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """
    将未同步数据通过采集端 Agent 同步到 MinIO。
    """
    asset = await get_asset_by_id(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="资产不存在")
    if not await data_asset_visible_to_user(db, current_user, asset):
        raise HTTPException(status_code=404, detail="资产不存在")
    lock = await acquire_asset_sync_lock(asset.id)
    try:
        await db.refresh(asset)
        if _asset_is_synced(asset):
            return ApiResponse(ok=True, data={"id": asset.id, "sync_status": "synced", "message": "数据已同步"})
        if not await try_mark_asset_syncing(db, asset.id):
            await db.refresh(asset)
            if _asset_is_synced(asset):
                return ApiResponse(ok=True, data={"id": asset.id, "sync_status": "synced", "message": "数据已同步"})
            if (getattr(asset, "sync_status", "") or "").strip().lower() == "syncing":
                raise HTTPException(status_code=409, detail="该数据正在同步中，请勿重复发起同步")
            raise HTTPException(status_code=409, detail="同步状态冲突，请稍后重试")
        raw_source = _extract_backend_local_path(getattr(asset, "meta", None)) or (asset.file_path or "").strip()
        if not raw_source:
            raise HTTPException(status_code=400, detail="资产缺少可同步的源路径")
        # 单文件登记（*.mcap 等）时改为同步其所在 episode 目录，包含质检报告等附属文件
        source_path = _episode_delete_target_local(raw_source)
        if not source_path:
            raise HTTPException(status_code=400, detail="资产缺少可同步的源路径")
        project_id = (asset.project_id or "").strip()
        project_name = (asset.project_name or project_id).strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="资产缺少所属项目，无法同步")
        try:
            logger.info(
                "data_sync API: 开始 asset_id=%s user_id=%s project_id=%s agent_id_query=%r source_path=%r",
                asset.id,
                getattr(current_user, "id", None),
                project_id,
                agent_id,
                source_path,
            )
            ret = await sync_asset_via_agent(
                db,
                asset_id=asset.id,
                source_path=source_path,
                project_id=project_id,
                project_name=project_name,
                agent_id=agent_id,
                meta_json=getattr(asset, "meta", None),
                collect_device_id=(getattr(asset, "device_id", None) or "").strip() or None,
            )
            minio_path = str(ret.get("minio_path") or "").strip()
            merged_meta = merge_storage_meta(getattr(asset, "meta", None), source_path, minio_path)
            asset.file_path = minio_path
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
            logger.info(
                "data_sync API: 完成并已落库 asset_id=%s minio_path=%r（若前端仍报 500，多为 Next 代理超时，见 next.config experimental.proxyTimeout）",
                asset.id,
                minio_path,
            )
            return ApiResponse(
                ok=True,
                data={"id": asset.id, "sync_status": "synced", "minio_path": minio_path, "message": "同步成功"},
            )
        except Exception as e:
            err_text = str(e)[:400]
            logger.exception(
                "data_sync API: 失败 asset_id=%s user_id=%s project_id=%s agent_id_query=%r error=%s",
                asset.id,
                getattr(current_user, "id", None),
                project_id,
                agent_id,
                err_text,
            )
            await update_asset(db, asset.id, sync_status="failed", sync_error=err_text)
            return ApiResponse(ok=False, error=f"同步失败: {str(e)[:200]}")
    finally:
        lock.release()


@router.post("/sync/batch", response_model=ApiResponse)
async def create_sync_batch_job(
    body: SyncBatchCreateBody = Body(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    创建批量同步任务：立即返回 jobId，后台按全局/按 Agent 并发上限执行；轮询 GET /sync/batch/status。
    持久化于 PostgreSQL（sync_batch_jobs / sync_batch_job_items）。
    """
    if not body.asset_ids:
        return ApiResponse(ok=False, error="请至少选择一个资产")
    seen: set[int] = set()
    asset_ids: List[int] = []
    for aid in body.asset_ids:
        if aid not in seen:
            seen.add(aid)
            asset_ids.append(aid)

    assets = await get_assets_by_ids(db, asset_ids)
    if len(assets) != len(asset_ids):
        return ApiResponse(ok=False, error="部分资产不存在")
    allowed_project_ids = await data_assets_allowed_project_ids(db, current_user)
    if allowed_project_ids is not None:
        allowed_set = set(allowed_project_ids)
        if any((a.project_id or "").strip() not in allowed_set for a in assets):
            return ApiResponse(ok=False, error="部分资产不存在或无权限")
    for a in assets:
        if not await data_asset_visible_to_user(db, current_user, a):
            return ApiResponse(ok=False, error="部分资产不存在或无权限")

    job_id = f"sync_batch_{uuid.uuid4().hex[:12]}"
    agent_q = (body.agent_id or "").strip() or None
    job = SyncBatchJob(
        job_id=job_id,
        user_id=str(current_user.id),
        status="queued",
        agent_id_query=agent_q,
        total=len(asset_ids),
        succeeded=0,
        failed=0,
        progress_percent=0.0,
        current_step="排队中",
        error_message=None,
    )
    db.add(job)
    for i, aid in enumerate(asset_ids):
        db.add(
            SyncBatchJobItem(
                job_id=job_id,
                asset_id=aid,
                sort_order=i,
                status="pending",
            )
        )
    await db.commit()
    schedule_batch_job(job_id, user_id=str(current_user.id))
    return ApiResponse(ok=True, data={"jobId": job_id, "status": "queued"})


@router.get("/sync/batch/status", response_model=ApiResponse)
async def get_sync_batch_status(
    jobId: str = Query(..., description="批量同步任务 ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """查询批量同步任务状态（轮询）。"""
    payload = await build_job_status_payload(db, jobId.strip(), current_user)
    if not payload:
        return ApiResponse(ok=False, error="任务不存在或已过期")
    return ApiResponse(ok=True, data=payload)


@router.post("/export/preview", response_model=ApiResponse)
async def export_preview(
    body: ExportRequest = Body(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """导出预览：根据待导出资产返回是否有平台标注文件，供弹窗摘要展示。"""
    if not body.asset_ids:
        return ApiResponse(ok=False, error="请至少选择一个资产")
    assets = await get_assets_by_ids(db, body.asset_ids)
    if len(assets) != len(body.asset_ids):
        return ApiResponse(ok=False, error="部分资产不存在")
    allowed_project_ids = await data_assets_allowed_project_ids(db, current_user)
    if allowed_project_ids is not None:
        allowed_set = set(allowed_project_ids)
        if any((a.project_id or "").strip() not in allowed_set for a in assets):
            return ApiResponse(ok=False, error="部分资产不存在")
    if any(not _asset_is_synced(a) for a in assets):
        return ApiResponse(ok=False, error="包含未同步数据，暂不可导出")
    has_annotations = any(_platform_annotation_path(a) for a in assets)
    return ApiResponse(ok=True, data={"has_annotations": has_annotations})


@router.get("/export")
async def export_data_assets(
    http_request: Request,
    background_tasks: BackgroundTasks,
    keyword: str = Query(None),
    project: str = Query(None),
    format: str = Query(None, alias="format"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """导出资产清单为 CSV。"""
    params = DataAssetQueryParams(
        keyword=keyword,
        project=project,
        format=format,
        page=1,
        page_size=10000,
    )
    allowed_project_ids = await data_assets_allowed_project_ids(db, current_user)
    items, _ = await get_assets(db, params, allowed_project_ids=allowed_project_ids)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([
        "id", "code", "filename", "format", "source", "device_id", "project_id", "project_name",
        "file_path", "file_size_bytes", "created_at", "created_at_ms", "parse_status", "error_msg",
    ])
    for a in items:
        w.writerow([
            a.id,
            a.code,
            a.filename,
            a.format,
            a.source,
            getattr(a, "device_id", None) or "",
            a.project_id or "",
            a.project_name or "",
            a.file_path,
            a.file_size_bytes,
            _datetime_to_iso_utc(a.created_at),
            _datetime_to_epoch_ms(a.created_at),
            a.parse_status or "",
            (a.error_msg or "")[:200],
        ])
    out.seek(0)
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=http_request,
        action_type=AA.EXPORT_DATA_ASSET,
        project_id=(project or None),
        resource_type=AR.DATA_ASSET,
        resource_id="manifest:csv",
        resource_name="数据资产清单 CSV",
        detail_json={"row_count": len(items), "format": "csv"},
    )
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="data_assets.csv"'},
    )


@router.post("/export")
async def export_data_assets_zip(
    http_request: Request,
    background_tasks: BackgroundTasks,
    body: ExportRequest = Body(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    按资产 ID 列表导出为 zip（单条或批量）。
    仅支持同一种数据格式；zip 内包含原始数据、平台标注（若有）、asset_list.xlsx。
    """
    if not body.asset_ids:
        raise HTTPException(status_code=400, detail="请至少选择一个资产")
    assets = await get_assets_by_ids(db, body.asset_ids)
    if len(assets) != len(body.asset_ids):
        found_ids = {a.id for a in assets}
        missing = [i for i in body.asset_ids if i not in found_ids]
        raise HTTPException(status_code=404, detail=f"未找到资产: {missing}")
    allowed_project_ids = await data_assets_allowed_project_ids(db, current_user)
    if allowed_project_ids is not None:
        allowed_set = set(allowed_project_ids)
        if any((a.project_id or "").strip() not in allowed_set for a in assets):
            raise HTTPException(status_code=404, detail="未找到资产")
    if any(not _asset_is_synced(a) for a in assets):
        raise HTTPException(status_code=409, detail="包含未同步数据，暂不可导出")
    formats = [_normalize_format(a.format, a.file_path or "") for a in assets]
    if len(set(formats)) > 1:
        raise HTTPException(
            status_code=400,
            detail="当前仅支持同一种数据格式的批量导出，请按 HDF5、MCAP 或 LeRobot 分别导出。",
        )
    try:
        zip_bytes, zip_filename = _build_export_zip(assets, compression_mode=body.compression_mode)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败，请稍后重试: {str(e)[:200]}")

    atype = AA.BATCH_EXPORT_DATA_ASSET if len(body.asset_ids) > 1 else AA.EXPORT_DATA_ASSET
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=http_request,
        action_type=atype,
        project_id=(assets[0].project_id if assets else None),
        project_name=(getattr(assets[0], "project_name", None) if assets else None),
        resource_type=AR.DATA_ASSET,
        resource_id=f"export:zip:{len(body.asset_ids)}",
        resource_name="ZIP 导出",
        detail_json={"asset_ids": body.asset_ids, "count": len(body.asset_ids)},
    )
    return StreamingResponse(
        iter([zip_bytes]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@router.post("/export/jobs", response_model=ApiResponse)
async def create_export_job(
    http_request: Request,
    background_tasks: BackgroundTasks,
    body: ExportRequest = Body(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """创建导出任务。不传 output_path 时生成临时 zip 供浏览器下载；传入 output_path 时仍写入白名单目录（兼容）。"""
    await _cleanup_expired_export_jobs()
    if not body.asset_ids:
        return ApiResponse(ok=False, error="请至少选择一个资产")
    target = (body.target or "local").strip().lower()
    if target != "local":
        return ApiResponse(ok=False, error="当前仅支持本地导出，云端暂未开放")
    output_path_raw = (body.output_path or "").strip()
    compression_mode = _normalize_compression_mode(body.compression_mode)
    resolved: Optional[str] = None
    if output_path_raw:
        try:
            resolved = validate_path_whitelist(output_path_raw)
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            return ApiResponse(ok=False, error=detail[:200])
        if not os.path.isdir(resolved):
            return ApiResponse(ok=False, error="输出路径不是目录")

    assets = await get_assets_by_ids(db, list(body.asset_ids))
    if len(assets) != len(body.asset_ids):
        return ApiResponse(ok=False, error="部分资产不存在")
    allowed_project_ids = await data_assets_allowed_project_ids(db, current_user)
    if allowed_project_ids is not None:
        allowed_set = set(allowed_project_ids)
        if any((a.project_id or "").strip() not in allowed_set for a in assets):
            return ApiResponse(ok=False, error="部分资产不存在")
    if any(not _asset_is_synced(a) for a in assets):
        return ApiResponse(ok=False, error="包含未同步数据，暂不可导出")

    job_id = f"export_{uuid.uuid4().hex[:12]}"
    tmp_dir = tempfile.mkdtemp(prefix=f"data_assets_export_{job_id}_")
    job = {
        "jobId": job_id,
        "status": "validating",
        "progress": _export_percent_by_status("validating"),
        "currentStep": _export_step_title("validating"),
        "fileName": "",
        "downloadUrl": "",
        "errorMessage": "",
        "assetIds": list(body.asset_ids),
        "userId": str(current_user.id),
        "createdAtTs": int(time.time() * 1000),
        "tmpDir": tmp_dir,
        "zipPath": "",
        "outputPath": resolved or "",
        "exportDirName": "",
        "fullOutputPath": "",
        "deliveryMode": "browser_zip" if not resolved else "local_path",
        "compressionMode": compression_mode,
        "completedAssets": 0,
        "totalAssets": len(body.asset_ids),
    }
    with _export_jobs_lock:
        _export_jobs[job_id] = job

    uid = str(getattr(current_user, "id", "") or "")
    dispatch_payload: Dict[str, Any] = {
        "type": "export",
        "task_id": job_id,
        "user_id": uid,
        "job_id": job_id,
        "asset_ids": list(body.asset_ids),
        "job_payload": job,
    }
    normalized_task: Dict[str, Any] = {
        "task_id": job_id,
        "type": "export",
        "user_id": uid,
        "params": dict(dispatch_payload),
    }
    normalized_task["params"]["task_id"] = job_id
    create_task_job(
        task_id=job_id,
        task_type="export",
        status="pending",
        user_id=uid if uid else None,
        queue_name="io_queue",
        payload=normalized_task,
    )
    update_task_status(job_id, "queued", rq_job_id=job_id)
    asyncio.create_task(_export_job_background(job_id, list(body.asset_ids)))
    atype = AA.BATCH_EXPORT_DATA_ASSET if len(body.asset_ids) > 1 else AA.EXPORT_DATA_ASSET
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=http_request,
        action_type=atype,
        project_id=(assets[0].project_id if assets else None),
        project_name=(getattr(assets[0], "project_name", None) if assets else None),
        resource_type=AR.DATA_ASSET,
        resource_id=f"export_job:{job_id}",
        resource_name="数据资产导出任务（浏览器下载）" if not resolved else "本地目录导出任务",
        detail_json={
            "job_id": job_id,
            "asset_ids": list(body.asset_ids),
            "output_path": resolved,
            "delivery_mode": job["deliveryMode"],
            "compression_mode": compression_mode,
        },
    )
    return ApiResponse(ok=True, data={"jobId": job_id, "status": "validating"})


@router.post("/export/delete-result", response_model=ApiResponse)
async def delete_export_result(
    body: DeleteExportResultRequest = Body(...),
    current_user: User = Depends(get_current_user),
):
    """删除导出任务产物（zip 或目录）。仅允许删除白名单内的路径。删除成功后任务记录移除。"""
    await _cleanup_expired_export_jobs()
    job_id = (body.job_id or "").strip()
    if not job_id:
        return ApiResponse(ok=False, error="缺少 jobId")
    with _export_jobs_lock:
        job = _export_jobs.get(job_id)
        if not job:
            return ApiResponse(ok=False, error="导出任务不存在或已过期")
        if not is_super_admin_or_team_admin(current_user.role) and str(job.get("userId") or "") != str(current_user.id):
            return ApiResponse(ok=False, error="导出任务不存在或已过期")
        st = (job.get("status") or "").strip().lower()
        if st not in ("ready", "failed", "cancelled"):
            return ApiResponse(ok=False, error="进行中任务请先在任务中心点击「取消」，勿使用删除")
        full_output = (job.get("fullOutputPath") or "").strip()
        zip_p = (job.get("zipPath") or "").strip()
    if full_output:
        try:
            resolved = validate_path_whitelist(full_output)
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            return ApiResponse(ok=False, error=detail[:200])
        try:
            if os.path.isfile(resolved):
                os.remove(resolved)
            elif os.path.isdir(resolved):
                shutil.rmtree(resolved, ignore_errors=False)
        except FileNotFoundError:
            pass
        except PermissionError as e:
            return ApiResponse(ok=False, error="没有权限删除该路径")
        except OSError as e:
            return ApiResponse(ok=False, error=(str(e) or "删除失败")[:200])
    elif zip_p and os.path.isfile(zip_p):
        try:
            os.remove(zip_p)
        except OSError:
            pass
    with _export_jobs_lock:
        _export_jobs.pop(job_id, None)
    delete_task_job(job_id)
    return ApiResponse(ok=True, data=None)


@router.get("/export/jobs", response_model=ApiResponse)
async def list_export_jobs(
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
):
    """列出当前用户最近的导出任务（内存态），供刷新页面后恢复任务中心。"""
    await _cleanup_expired_export_jobs()
    uid = str(getattr(current_user, "id", "") or "")
    rows: List[Dict[str, Any]] = []
    with _export_jobs_lock:
        for _jid, job in list(_export_jobs.items()):
            if not is_super_admin_or_team_admin(current_user.role) and str(job.get("userId") or "") != uid:
                continue
            rows.append(
                {
                    "jobId": job.get("jobId"),
                    "status": job.get("status"),
                    "progress": int(job.get("progress") or 0),
                    "currentStep": job.get("currentStep") or "",
                    "fileName": job.get("fileName") or "",
                    "downloadUrl": job.get("downloadUrl") or "",
                    "errorMessage": job.get("errorMessage") or "",
                    "exportDirName": job.get("exportDirName") or "",
                    "fullOutputPath": job.get("fullOutputPath") or "",
                    "completedAssets": int(job.get("completedAssets") or 0),
                    "totalAssets": int(job.get("totalAssets") or 0),
                    "deliveryMode": job.get("deliveryMode") or "",
                    "compressionMode": job.get("compressionMode") or "deflated",
                    "createdAtTs": int(job.get("createdAtTs") or 0),
                }
            )
    rows.sort(key=lambda r: -r["createdAtTs"])
    return ApiResponse(ok=True, data=rows[:limit])


@router.get("/export/status", response_model=ApiResponse)
async def get_export_job_status(
    jobId: str = Query(..., description="导出任务 ID"),
    current_user: User = Depends(get_current_user),
):
    """查询导出任务状态（轮询）。本地导出时返回 exportDirName、fullOutputPath。"""
    await _cleanup_expired_export_jobs()
    with _export_jobs_lock:
        job = _export_jobs.get(jobId)
        if not job:
            return ApiResponse(ok=False, error="导出任务不存在或已过期")
        if not is_super_admin_or_team_admin(current_user.role) and str(job.get("userId") or "") != str(current_user.id):
            return ApiResponse(ok=False, error="导出任务不存在或已过期")
        data = {
            "jobId": job.get("jobId"),
            "status": job.get("status"),
            "progress": int(job.get("progress") or 0),
            "currentStep": job.get("currentStep") or "",
            "fileName": job.get("fileName") or "",
            "downloadUrl": job.get("downloadUrl") or "",
            "errorMessage": job.get("errorMessage") or "",
            "exportDirName": job.get("exportDirName") or "",
            "fullOutputPath": job.get("fullOutputPath") or "",
            "completedAssets": int(job.get("completedAssets") or 0),
            "totalAssets": int(job.get("totalAssets") or 0),
            "deliveryMode": job.get("deliveryMode") or "",
            "compressionMode": job.get("compressionMode") or "deflated",
        }
    return ApiResponse(ok=True, data=data)


@router.get("/export/download")
async def download_export_job_zip(
    jobId: str = Query(..., description="导出任务 ID"),
    current_user: User = Depends(get_current_user_or_cookie),
):
    """下载导出 zip（ready 后可多次下载）。需登录且为任务创建者或管理员；支持 query token。"""
    await _cleanup_expired_export_jobs()
    with _export_jobs_lock:
        job = _export_jobs.get(jobId)
        if not job:
            raise HTTPException(status_code=404, detail="导出任务不存在或已过期")
        if not is_super_admin_or_team_admin(current_user.role) and str(job.get("userId") or "") != str(current_user.id):
            raise HTTPException(status_code=404, detail="导出任务不存在或已过期")
        if job.get("status") != "ready":
            raise HTTPException(status_code=400, detail="导出包尚未准备完成")
        zip_path = job.get("zipPath") or ""
        file_name = job.get("fileName") or "export.zip"
    if not zip_path or not os.path.isfile(zip_path):
        raise HTTPException(status_code=404, detail="未找到导出文件")

    file_size = os.path.getsize(zip_path)

    def _iter_file(p: str, chunk_size: int = 1024 * 1024):
        with open(p, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        _iter_file(zip_path),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{file_name}"',
            "Content-Length": str(file_size),
        },
    )


@router.post("/cache/clear", response_model=ApiResponse)
async def clear_backend_cache(
    scope: str = Query("all", description="清理范围：all | data | project"),
    current_user: User = Depends(get_current_user),
):
    """
    清理平台缓存目录，当前覆盖：
    - data: DATA_ASSETS_ROOT 下 .minio_view_cache 与 _minio_conversion_cache
    - project: backend/project
    """
    if not is_super_admin_or_team_admin(current_user.role):
        raise HTTPException(status_code=403, detail="无权限清理缓存")
    selected = (scope or "all").strip().lower()
    if selected not in {"all", "data", "project"}:
        raise HTTPException(status_code=400, detail="scope 仅支持 all/data/project")

    data_cache_result: Optional[Dict[str, Any]] = None
    project_cache_result: Optional[Dict[str, int]] = None

    if selected in {"all", "data"}:
        data_cache_result = await asyncio.to_thread(clear_data_disk_caches)
    if selected in {"all", "project"}:
        project_cache_result = await asyncio.to_thread(clear_project_cache)

    return ApiResponse(
        ok=True,
        data={
            "scope": selected,
            "data_cache": data_cache_result,
            "project_cache": project_cache_result,
        },
    )
