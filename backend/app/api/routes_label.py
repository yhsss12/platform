"""
标注相关 API 路由
支持 HDF5 与 MCAP 两种数据集格式，按文件扩展名自动选择服务。
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, UploadFile, File, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from typing import Optional, List, Literal, Tuple
from pydantic import BaseModel, field_validator
from app.services.hdf5_service import HDF5Service
from app.services.mcap_service import MCAPService
from app.services.episode_storage import EpisodeStorage
from app.services.storage_resolver import EpisodeResolveError
from app.services.annotation_service import AnnotationService
from app.services.batch_annotation_service import BatchAnnotationService
from app.services.task_job_store import get_task_job
from app.services.task_config_service import TaskConfigService
from app.schemas.common import ApiResponse
from app.core.instruction_path import get_instruction_path_for_data_path
from app.core.deps import get_current_user, get_current_user_ws
from app.core.roles import is_super_admin
from app.core.label_conversion_access import (
    assert_label_task_in_execute_scope,
    assert_platform_task_manage_project,
    scoped_project_ids_for_platform_tasks,
)
from app.core.label_task_actor_permissions import (
    assert_labeler_reviewer_are_project_members,
    assert_user_may_annotate_label_task,
    assert_user_may_review_label_task,
)
from app.crud.project import get_project_by_id
from app.db.data_assets_session import get_data_assets_db, DataAssetsSessionLocal
from app.models.label_task_asset import LabelTask
from app.models.user import User
from app.crud.data_asset import (
    find_asset_by_instruction_path_candidates,
    find_data_asset_for_label_episode,
    get_asset_by_file_path,
    persist_annotation_instruction_to_data_asset,
    update_asset,
    normalize_storage_path_key,
)
from app.services.data_asset_path_resolver import (
    minio_uri_from_fields,
    resolve_label_task_warehouse_and_local,
    resolve_label_task_warehouse_uri,
    resolve_read_local_from_warehouse_uri,
)
from app.services.minio_service import MinioBucketError, MinioConfigError, list_object_names_under_prefix
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import enqueue_audit_log
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
import os
import json
import hashlib
import asyncio
from datetime import datetime


class GenerateAnnotationRequest(BaseModel):
    episode_id: str
    camera_name: Optional[str] = None
    taskId: Optional[str] = None
    model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None


class BatchAnnotationRequest(BaseModel):
    episode_ids: List[str]
    camera_name: Optional[str] = None
    fallback_first_camera: bool = True


class BatchAnnotationByTaskRequest(BaseModel):
    taskId: str
    camera_name: Optional[str] = None
    fallback_first_camera: bool = True
    model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None


class CreateLabelTaskRequest(BaseModel):
    name: str
    dataset_path: Optional[str] = None  # 改为可选，如果提供了 dataset_ids 则不需要
    dataset_ids: Optional[List[int]] = None  # 数据集 ID 列表（可为 data_assets 或 hdf5_datasets 的 ID）
    dataset_source: Optional[str] = None  # "data_assets" | None(默认 hdf5_datasets)
    data_count: Optional[int] = None
    device_type: Optional[str] = None  # 已废弃，保留兼容；任务以所属项目为准
    project_id: Optional[str] = None  # 所属项目 ID
    labeler: Optional[str] = None
    reviewer: Optional[str] = None
    collector: Optional[str] = None

    @field_validator("dataset_ids", mode="before")
    @classmethod
    def coerce_dataset_ids(cls, v):
        """允许前端传 number[] 或 numeric string[]，统一转为 int 列表"""
        if v is None:
            return None
        if not isinstance(v, list):
            return v
        out = []
        for x in v:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str):
                try:
                    out.append(int(x))
                except (ValueError, TypeError):
                    pass
        return out if out else None

router = APIRouter()
logger = logging.getLogger(__name__)


async def _assert_label_task_visible_or_404(db: AsyncSession, current_user: User, task_id: str) -> LabelTask:
    """列表/详情/执行链：任务须在平台任务 project 范围内（与 scoped_project_ids_for_platform_tasks 一致）。"""
    tid = (task_id or "").strip()
    if not tid:
        raise HTTPException(status_code=400, detail="taskId required")
    row = (await db.execute(select(LabelTask).where(LabelTask.task_id == tid))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    pid = (getattr(row, "project_id", None) or "").strip()
    scoped = await scoped_project_ids_for_platform_tasks(db, current_user)
    if scoped is not None:
        if not pid or pid not in scoped:
            raise HTTPException(status_code=404, detail="任务不存在")
    return row


async def _assert_label_task_manage(db: AsyncSession, current_user: User, task_id: str) -> LabelTask:
    """创建/编辑/删除标注任务：须具备对所属项目的管理权限（USER 不可）。"""
    row = await _assert_label_task_visible_or_404(db, current_user, task_id)
    pid = (getattr(row, "project_id", None) or "").strip()
    await assert_platform_task_manage_project(db, current_user, pid)
    return row


async def _label_audit_project_display_name(
    db: AsyncSession,
    *,
    project_id: Optional[str],
    label_task_name: Optional[str] = None,
) -> Optional[str]:
    """
    审计日志「所属项目」列展示用：优先 projects.name（人类可读）；
    查不到项目时用当前标注任务名兜底，避免界面只显示 p_ 时间戳类 project_id。
    """
    pid = (project_id or "").strip()
    if pid:
        proj = await get_project_by_id(db, pid)
        pname = (getattr(proj, "name", None) or "").strip() if proj else ""
        if pname:
            return pname
    tname = (label_task_name or "").strip()
    return tname or None


def _label_task_row_to_config_dict(task_id: str, row: LabelTask) -> dict:
    """从 label_tasks 表行构造与 task.json 一致的结构（用于磁盘 task.json 丢失时按库自愈）。"""
    ds_path = (getattr(row, "dataset_path", None) or "").strip()
    ds_ids_raw = getattr(row, "dataset_ids", None)
    dataset_ids_parsed = None
    if ds_ids_raw:
        if isinstance(ds_ids_raw, str):
            try:
                dataset_ids_parsed = json.loads(ds_ids_raw)
            except Exception:
                dataset_ids_parsed = None
        elif isinstance(ds_ids_raw, list):
            dataset_ids_parsed = ds_ids_raw
    return {
        "task_id": task_id,
        "name": getattr(row, "name", None) or "",
        "dataset_path": ds_path,
        "dataset_ids": dataset_ids_parsed,
        "dataset_source": getattr(row, "dataset_source", None),
        "data_count": getattr(row, "data_count", None),
        "device_type": (getattr(row, "device_type", None) or "") or "",
        "project_id": (getattr(row, "project_id", None) or "") or "",
        "labeler": getattr(row, "labeler", None),
        "reviewer": getattr(row, "reviewer", None),
        "collector": getattr(row, "collector", None) or "",
        "completed": bool(getattr(row, "completed", False)),
        "verified": bool(getattr(row, "verified", False)),
    }


async def _require_super_or_task_scope(db: AsyncSession, current_user: User, task_id: Optional[str]) -> None:
    """无 taskId 的遗留接口：仅超级管理员可用。"""
    tid = (task_id or "").strip()
    if tid:
        await _assert_label_task_visible_or_404(db, current_user, tid)
        return
    if not is_super_admin(current_user.role):
        raise HTTPException(status_code=403, detail="taskId required")


async def _path_in_data_assets(file_path: str) -> bool:
    """检查路径是否在数据资产表中（标注仅允许数据资产中的数据）"""
    if not file_path:
        return False
    try:
        from app.db.data_assets_session import DataAssetsSessionLocal
        from app.models.data_asset import DataAsset
        from sqlalchemy import select, or_
        async with DataAssetsSessionLocal() as db:
            norm = normalize_storage_path_key(file_path)
            r = await db.execute(
                select(DataAsset).where(or_(DataAsset.file_path == file_path, DataAsset.file_path == norm))
            )
            asset = r.scalar_one_or_none()
            if asset is None:
                # 兼容历史数据：data_assets.file_path 仍是本地路径，但 meta.storage.minio_path 已存在。
                if str(file_path).startswith("minio://"):
                    rows = (
                        await db.execute(
                            select(DataAsset).where(DataAsset.sync_status == "synced").limit(2000)
                        )
                    ).scalars().all()
                    for row in rows:
                        wh = minio_uri_from_fields(getattr(row, "file_path", None), getattr(row, "meta", None))
                        if wh and normalize_storage_path_key(wh) == norm:
                            return True
                return False
            return (getattr(asset, "sync_status", "synced") or "synced").strip().lower() == "synced"
    except Exception:
        return False


async def _label_episode_minio_allowed(db: AsyncSession, warehouse_uri: str) -> bool:
    """
    单条 episode 的 MinIO 路径是否允许进入标注读链路。
    须与 GET /episodes 列表过滤口径一致：除 file_path 精确命中外，还支持
    get_file_paths_in_assets 的父路径/规范化匹配（避免列表能点、详情 403）。
    """
    wh = (warehouse_uri or "").strip()
    if not wh.startswith("minio://"):
        return False
    if await _path_in_data_assets(wh):
        return True
    from app.crud.data_asset import get_file_paths_in_assets

    valid = await get_file_paths_in_assets(db, [wh])
    return normalize_storage_path_key(wh) in valid


async def _task_episode_local_read_path(episode_info: dict) -> str:
    """从数据仓库 URI 解析本机缓存路径；缺 warehouse_path 时要求重新加载数据集。"""
    wh = EpisodeStorage(episode_info).get_storage_key()
    if not wh.startswith("minio://"):
        raise HTTPException(
            status_code=400,
            detail="任务索引缺少 MinIO 地址（warehouse_path）。请重新执行「加载数据集」。",
        )
    try:
        return await asyncio.to_thread(resolve_read_local_from_warehouse_uri, wh)
    except MinioBucketError as e:
        raise HTTPException(status_code=502, detail=str(e)[:500])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)[:300])
    except MinioConfigError as e:
        raise HTTPException(status_code=503, detail=str(e)[:300])


async def _resolve_task_episode_file_with_visibility(
    db: AsyncSession,
    current_user: User,
    task_id: str,
    episode_id: str,
) -> tuple[dict, str, str]:
    """统一解析 task episode 本地读路径 + 可见性校验，避免各接口口径漂移。"""
    config_service = get_task_config_service()
    await _assert_label_task_visible_or_404(db, current_user, task_id)
    episode_info = config_service.find_episode_by_id(task_id, episode_id)
    if not episode_info:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    storage = EpisodeStorage(episode_info)
    wh = storage.get_storage_key()
    try:
        file_path = storage.resolve_local_path()
    except EpisodeResolveError as ex:
        raise HTTPException(
            status_code=404,
            detail={"error_code": ex.code, "message": ex.message, "episode_id": ex.episode_id},
        )

    if wh.startswith("minio://"):
        if not await _label_episode_minio_allowed(db, wh):
            raise HTTPException(status_code=403, detail="无数据：该条数据不在数据资产中，不允许标注")
    else:
        if not await _path_in_data_assets(file_path):
            raise HTTPException(status_code=403, detail="无数据：该条数据不在数据资产中，不允许标注")
    return episode_info, wh, file_path


async def _resolve_project_llm_config(
    db: AsyncSession,
    project_id: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """按团队优先读取共享 LLM 配置（团队管理员维护）：返回 (model, api_key, api_base)。"""
    from app.api.routes_llm import _resolve_llm_scope_key

    pid = (project_id or "").strip()
    if not pid:
        return None, None, None

    scope_key = f"project:{pid}"
    try:
        async with db.begin_nested():
            scope_key = await _resolve_llm_scope_key(db, pid)
    except Exception:
        logger.debug(
            "_resolve_project_llm_config: scope key fallback to project:%s",
            pid,
            exc_info=True,
        )
        scope_key = f"project:{pid}"

    try:
        async with db.begin_nested():
            await db.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS project_llm_bindings (
                      id BIGSERIAL PRIMARY KEY,
                      project_id VARCHAR NOT NULL,
                      provider_id BIGINT NOT NULL,
                      provider_code VARCHAR NOT NULL,
                      provider_name VARCHAR NOT NULL,
                      model_name VARCHAR NOT NULL,
                      api_base TEXT,
                      api_key TEXT,
                      api_key_masked VARCHAR,
                      editable_roles VARCHAR NOT NULL DEFAULT 'SUPER_ADMIN,ADMIN',
                      is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                      is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                      verified_at TIMESTAMP NULL,
                      created_by VARCHAR NULL,
                      updated_by VARCHAR NULL,
                      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      UNIQUE(project_id, provider_id, model_name)
                    )
                    """
                )
            )
    except Exception:
        logger.debug(
            "_resolve_project_llm_config: CREATE project_llm_bindings skipped",
            exc_info=True,
        )

    b = (
        await db.execute(
            text(
                """
                SELECT model_name, api_key, api_base
                FROM project_llm_bindings
                WHERE project_id = :scope_key
                  AND is_enabled = TRUE
                ORDER BY updated_at DESC NULLS LAST, id DESC
                LIMIT 1
                """
            ),
            {"scope_key": scope_key},
        )
    ).fetchone()
    if b:
        model = (getattr(b, "model_name", None) or "").strip() or None
        api_key = (getattr(b, "api_key", None) or "").strip() or None
        api_base = (getattr(b, "api_base", None) or "").strip() or None
        return model, api_key, api_base

    # 旧版 user_providers / user_models：库未迁移时表可能不存在。先用 to_regclass 避免触发 ProgrammingError
    # （部分会话下仅依赖 begin_nested 仍可能把错误冒泡到业务层）。
    has_up = (
        await db.execute(text("SELECT to_regclass('public.user_providers') IS NOT NULL"))
    ).scalar()
    has_um = (
        await db.execute(text("SELECT to_regclass('public.user_models') IS NOT NULL"))
    ).scalar()
    if not has_up or not has_um:
        logger.debug(
            "_resolve_project_llm_config: skip legacy LLM tables (user_providers=%s user_models=%s)",
            bool(has_up),
            bool(has_um),
        )
        return None, None, None

    legacy_model: Optional[str] = None
    legacy_key: Optional[str] = None
    legacy_base: Optional[str] = None
    try:
        async with db.begin_nested():
            await db.execute(text("ALTER TABLE user_providers ADD COLUMN IF NOT EXISTS project_id VARCHAR"))
            await db.execute(text("ALTER TABLE user_models ADD COLUMN IF NOT EXISTS project_id VARCHAR"))
            up = (
                await db.execute(
                    text(
                        """
                        SELECT provider_id, api_key, api_base
                        FROM user_providers
                        WHERE user_id = 0
                          AND COALESCE(project_id,'') = :pid
                          AND is_enabled = 1
                        ORDER BY updated_at DESC NULLS LAST, id DESC
                        LIMIT 1
                        """
                    ),
                    {"pid": pid},
                )
            ).fetchone()
            if up:
                mid = (
                    await db.execute(
                        text(
                            """
                            SELECT m.name AS model_name
                            FROM user_models um
                            JOIN models m ON m.id = um.model_id
                            WHERE um.user_id = 0
                              AND um.provider_id = :provider_id
                              AND COALESCE(um.project_id,'') = :pid
                              AND um.is_enabled = 1
                              AND m.is_active = 1
                            ORDER BY m.sort_order ASC, m.id ASC
                            LIMIT 1
                            """
                        ),
                        {"provider_id": up.provider_id, "pid": pid},
                    )
                ).fetchone()
                legacy_model = (getattr(mid, "model_name", None) or "").strip() if mid else None
                legacy_key = (getattr(up, "api_key", None) or "").strip() or None
                legacy_base = (getattr(up, "api_base", None) or "").strip() or None
    except Exception:
        logger.debug(
            "_resolve_project_llm_config: legacy user_providers/user_models skipped (DDL 或查询失败)",
            exc_info=True,
        )
    return legacy_model, legacy_key, legacy_base


def _file_format(path: str) -> Literal["mcap", "hdf5", "unknown"]:
    """根据文件扩展名判断数据集格式"""
    if not path:
        return "unknown"
    p = path.lower()
    if p.endswith(".mcap"):
        return "mcap"
    if p.endswith((".hdf5", ".h5")):
        return "hdf5"
    return "unknown"


# 初始化服务（单例模式）
_hdf5_service: Optional[HDF5Service] = None
_mcap_service: Optional[MCAPService] = None
_annotation_service: Optional[AnnotationService] = None
_batch_annotation_service: Optional[BatchAnnotationService] = None


def get_hdf5_service() -> HDF5Service:
    """获取 HDF5 服务实例"""
    global _hdf5_service
    if _hdf5_service is None:
        data_dir = os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data")
        _hdf5_service = HDF5Service(data_dir=data_dir)
    return _hdf5_service


def get_mcap_service() -> MCAPService:
    """获取 MCAP 服务实例"""
    global _mcap_service
    if _mcap_service is None:
        _mcap_service = MCAPService()
    return _mcap_service


def get_annotation_service() -> AnnotationService:
    """获取标注服务实例"""
    global _annotation_service
    if _annotation_service is None:
        _annotation_service = AnnotationService(get_hdf5_service())
    return _annotation_service


def get_batch_annotation_service() -> BatchAnnotationService:
    """获取批量标注服务实例"""
    global _batch_annotation_service
    if _batch_annotation_service is None:
        _batch_annotation_service = BatchAnnotationService(get_hdf5_service())
    return _batch_annotation_service


_task_config_service: Optional[TaskConfigService] = None


def get_task_config_service() -> TaskConfigService:
    """获取任务配置服务实例"""
    global _task_config_service
    if _task_config_service is None:
        base_dir = os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data")
        _task_config_service = TaskConfigService(base_dir=base_dir)
    return _task_config_service


@router.get("/episodes", response_model=ApiResponse)
async def get_episodes(
    taskId: Optional[str] = Query(None, description="任务 ID，可选"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """获取所有 episode 列表（可选按 taskId 过滤）"""
    try:
        await _require_super_or_task_scope(db, current_user, taskId)
        if taskId:
            # 从 episodes_index.json 读取
            config_service = get_task_config_service()
            config = config_service.load_task_config(taskId)
            episodes = config_service.load_episodes_index(taskId)
            if episodes is None:
                return ApiResponse(ok=True, data=[])

            # 若任务有 dataset_ids（继承自 data_assets.dataset_id），先按 dataset_id 校验数据是否存在
            if config.get("dataset_source") == "data_assets" and config.get("dataset_ids"):
                try:
                    from app.db.data_assets_session import DataAssetsSessionLocal
                    from app.crud.data_asset import are_dataset_ids_valid
                    ds_json = json.dumps(config["dataset_ids"]) if isinstance(config["dataset_ids"], list) else config.get("dataset_ids")
                    async with DataAssetsSessionLocal() as db:
                        if not await are_dataset_ids_valid(db, ds_json):
                            return ApiResponse(ok=False, error="无数据：关联的数据资产已被删除", data=[])
                except Exception:
                    pass

            # 必须：仅允许在数据资产表中存在的数据（file_path 为 minio://...）
            def _ep_asset_key(ep: dict) -> str:
                return EpisodeStorage(ep).get_storage_key()

            paths_to_check = [_ep_asset_key(ep) for ep in episodes if _ep_asset_key(ep)]
            valid_paths: set = set()
            try:
                from app.db.data_assets_session import DataAssetsSessionLocal
                from app.crud.data_asset import get_file_paths_in_assets
                async with DataAssetsSessionLocal() as db:
                    valid_paths = await get_file_paths_in_assets(db, paths_to_check)
            except Exception:
                valid_paths = set()

            valid_episodes = [
                ep for ep in episodes
                if _ep_asset_key(ep) and normalize_storage_path_key(_ep_asset_key(ep)) in valid_paths
            ]
            # dataset_ids 仍有效但路径键因 minio/本地写法不一致未命中时，不再误判为「资产已删除」
            if episodes and not valid_episodes and config.get("dataset_source") == "data_assets" and config.get("dataset_ids"):
                try:
                    from app.db.data_assets_session import DataAssetsSessionLocal
                    from app.crud.data_asset import are_dataset_ids_valid
                    ds_raw = config["dataset_ids"]
                    ds_json = json.dumps(ds_raw) if isinstance(ds_raw, list) else (ds_raw if isinstance(ds_raw, str) else "[]")
                    async with DataAssetsSessionLocal() as db_chk:
                        if await are_dataset_ids_valid(db_chk, ds_json):
                            valid_episodes = list(episodes)
                except Exception:
                    pass
            if episodes and not valid_episodes:
                return ApiResponse(ok=False, error="无数据：关联的数据资产已被删除", data=[])

            # 批量查询 data_assets.instruction_text，用于左侧「已标注/未标注」与任务描述
            path_to_instruction: dict = {}
            try:
                from app.crud.data_asset import get_instruction_text_by_paths
                path_list = [_ep_asset_key(ep) for ep in valid_episodes if _ep_asset_key(ep)]
                async with DataAssetsSessionLocal() as db:
                    path_to_instruction = await get_instruction_text_by_paths(db, path_list)
            except Exception:
                path_to_instruction = {}

            result = []
            for ep in valid_episodes:
                path_key = _ep_asset_key(ep)
                norm_path = normalize_storage_path_key(path_key) if path_key else ""
                instruction_text = path_to_instruction.get(norm_path, "") or ""
                result.append({
                    "id": ep.get("episode_id"),
                    "name": ep.get("filename"),
                    "path": path_key,
                    "instruction_text": instruction_text,
                })
            return ApiResponse(ok=True, data=result)
        else:
            # 兼容旧逻辑：从 data_dir 扫描
            service = get_hdf5_service()
            episodes = service.get_episodes(task_id=None)
            return ApiResponse(ok=True, data=episodes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{episode_id}", response_model=ApiResponse)
async def get_episode(
    episode_id: str,
    taskId: Optional[str] = Query(None, description="任务 ID"),
    camera_candidates: bool = Query(True, description="HDF5 为 true 时返回路径名含 camera 的所有节点（默认）；为 false 时仅返回校验过的相机"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """获取 episode 详细信息。HDF5 默认返回所有含 camera 的路径供下拉选择。"""
    try:
        config_service = get_task_config_service()
        hdf5_service = get_hdf5_service()
        
        # 如果提供了 taskId，从 episodes_index 查找
        if taskId:
            episode_info, wh, file_path = await _resolve_task_episode_file_with_visibility(
                db, current_user, taskId, episode_id
            )

            fmt = _file_format(file_path)
            try:
                data: dict = {
                    "id": episode_id,
                    "name": episode_info.get("filename"),
                    "path": wh if wh.startswith("minio://") else file_path,
                    "cameras": [],
                    "frameCount": 0,
                }
                if fmt == "mcap":
                    mcap_service = get_mcap_service()
                    cameras = mcap_service.list_camera_candidate_topics(file_path)
                    if not cameras:
                        cameras = mcap_service.list_cameras(file_path)
                    # 把有帧的相机排前面，避免第一个视窗默认选到无帧相机（如 depth）
                    with_frames: list = []
                    no_frames: list = []
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
                        start_ns, end_ns = mcap_service.get_time_range(
                            file_path, data["cameras"][0]
                        )
                        if start_ns > 0 or end_ns > 0:
                            data["startTimeNs"] = start_ns
                            data["endTimeNs"] = end_ns
                else:
                    import h5py
                    with h5py.File(file_path, "r") as f:
                        # HDF5 默认只列出能出图的相机路径（列表更短）；camera_candidates=false 时仅返回校验过的相机
                        if camera_candidates:
                            cameras = hdf5_service.list_camera_candidate_paths(f, image_only=True)
                            if not cameras:
                                cameras = hdf5_service.list_camera_candidate_paths(f, image_only=False)
                            if not cameras:
                                cameras = hdf5_service.list_cameras(f)
                        else:
                            cameras = hdf5_service.list_cameras(f)
                        # 把有帧的相机排前面，无帧的排后面；frameCount 取有帧相机中的最大帧数
                        with_frames = []
                        no_frames = []
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
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read file ({fmt}): {str(e)}")
        else:
            if not is_super_admin(current_user.role):
                raise HTTPException(status_code=403, detail="taskId required")
            # 兼容旧逻辑
            episode = hdf5_service.get_episode_info(episode_id, camera_candidates=camera_candidates)
            if not episode:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
            return ApiResponse(ok=True, data=episode)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frames/{episode_id}")
async def get_frame(
    episode_id: str,
    camera: str = Query(..., description="相机名称"),
    frame: int = Query(..., description="帧索引", ge=0),
    quality: int = Query(85, description="JPEG 质量", ge=1, le=100),
    taskId: Optional[str] = Query(None, description="任务 ID"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """获取指定帧的图像"""
    try:
        config_service = get_task_config_service()
        hdf5_service = get_hdf5_service()
        
        # 如果提供了 taskId，从 episodes_index 查找
        if taskId:
            _episode_info, _wh, file_path = await _resolve_task_episode_file_with_visibility(
                db, current_user, taskId, episode_id
            )
        else:
            if not is_super_admin(current_user.role):
                raise HTTPException(status_code=403, detail="taskId required")
            # 兼容旧逻辑
            episode = hdf5_service._find_episode_by_id(episode_id)
            if not episode:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
            file_path = episode["path"]
            if not file_path or not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"Episode file not found: {file_path}")
            if not await _path_in_data_assets(file_path):
                raise HTTPException(status_code=403, detail="无数据：该条数据不在数据资产中，不允许标注")

        fmt = _file_format(file_path)
        cameras = []
        frame_count = 0
        # 统一去掉 camera 前导斜杠，与后端返回的路径格式一致（如 camera_candidates 返回 "camera1/..."）
        camera_norm = (camera or "").strip().lstrip("/")
        try:
            if fmt == "mcap":
                mcap_service = get_mcap_service()
                cameras = mcap_service.list_camera_candidate_topics(file_path)
                if not cameras:
                    cameras = mcap_service.list_cameras(file_path)
                if cameras:
                    frame_count = mcap_service.get_frame_count(file_path, camera_norm or camera)
            else:
                # HDF5：先校验文件魔数，再打开，避免非 HDF5 文件导致 h5py 抛错
                if not hdf5_service._is_hdf5_file(file_path):
                    raise HTTPException(
                        status_code=400,
                        detail="文件不是有效的 HDF5 格式，无法读取帧",
                    )
                import h5py
                with h5py.File(file_path, "r") as f:
                    # HDF5：用“含 camera 的路径”列表校验（含非图像路径，选则返回黑帧）
                    cameras = hdf5_service.list_camera_candidate_paths(f, image_only=False)
                    if not cameras:
                        cameras = hdf5_service.list_cameras(f)
                    if cameras:
                        frame_count = hdf5_service.get_frame_count(file_path, camera_norm or camera)
            
            if not cameras:
                raise HTTPException(status_code=404, detail="No cameras found in file")
            # 支持带或不带前导 / 的路径：用归一化后的 camera 与列表比较（列表项统一无前导 /）
            cameras_norm = [c.lstrip("/") for c in cameras]
            if camera_norm and camera_norm not in cameras_norm:
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera '{camera}' not found. Available cameras: {', '.join(cameras[:10])}{'...' if len(cameras) > 10 else ''}"
                )
            if frame_count == 0:
                raise HTTPException(status_code=404, detail=f"Camera '{camera}' has no frames")
            if frame >= frame_count:
                raise HTTPException(
                    status_code=404,
                    detail=f"Frame index {frame} out of range. Camera '{camera}' has {frame_count} frames (0-{frame_count-1})"
                )
        except HTTPException:
            raise
        except Exception as val_err:
            logger.exception("get_frame 验证失败")
            err_msg = str(val_err).strip() or getattr(val_err, "message", "未知错误")
            if len(err_msg) > 400:
                err_msg = err_msg[:400] + "..."
            raise HTTPException(
                status_code=500,
                detail=f"验证或打开文件失败: {err_msg}",
            )
        
        try:
            if fmt == "mcap":
                mcap_service = get_mcap_service()
                image_bytes = mcap_service.get_frame_image(file_path, camera_norm or camera, frame, quality=quality)
            else:
                image_bytes = hdf5_service.get_frame_image(
                    file_path,
                    camera_norm or camera,
                    frame,
                    quality=quality
                )
        except Exception as img_err:
            logger.exception("get_frame 读取帧失败")
            err_msg = str(img_err).strip() or getattr(img_err, "message", "未知错误")
            if len(err_msg) > 400:
                err_msg = err_msg[:400] + "..."
            raise HTTPException(
                status_code=500,
                detail=f"读取帧失败: {err_msg}",
            )
        
        if image_bytes is None:
            raise HTTPException(
                status_code=404,
                detail=f"Frame not found: episode={episode_id}, camera={camera}, frame={frame}. Please check file structure."
            )
        
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=frame_{frame}.jpg"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e).strip() or getattr(e, "message", "未知错误")
        if len(err_msg) > 400:
            err_msg = err_msg[:400] + "..."
        raise HTTPException(status_code=500, detail=err_msg)


@router.get("/frames/{episode_id}/batch")
async def get_frames_batch(
    episode_id: str,
    camera: str = Query(..., description="相机名称"),
    start: int = Query(0, description="起始帧索引", ge=0),
    count: int = Query(50, description="预加载帧数", ge=1, le=200),
    taskId: Optional[str] = Query(None, description="任务 ID"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """批量获取帧（HDF5 / MCAP 通用），用于预加载缓存"""
    import base64

    try:
        config_service = get_task_config_service()
        if not taskId:
            raise HTTPException(status_code=400, detail="batch 接口需要 taskId")
        _episode_info, _wh, file_path = await _resolve_task_episode_file_with_visibility(
            db, current_user, taskId, episode_id
        )

        fmt = _file_format(file_path)
        frames_b64 = []
        camera_norm = (camera or "").strip().lstrip("/")

        if fmt == "mcap":
            mcap_service = get_mcap_service()
            cameras = mcap_service.list_camera_candidate_topics(file_path)
            if not cameras:
                cameras = mcap_service.list_cameras(file_path)
            cameras_norm = [c.lstrip("/") for c in cameras]
            if not cameras or (camera_norm and camera_norm not in cameras_norm):
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera '{camera}' not found. Available: {', '.join(cameras[:10])}{'...' if len(cameras) > 10 else ''}",
                )
            camera_to_use = cameras[cameras_norm.index(camera_norm)] if camera_norm in cameras_norm else cameras[0]
            frames_bytes = mcap_service.get_frames_batch(file_path, camera_to_use, start, count)
            frames_b64 = [base64.b64encode(b).decode("ascii") for b in frames_bytes]
        else:
            # HDF5：逐帧读取，在线程池中执行避免阻塞事件循环；用含 camera 的路径列表校验
            hdf5_service = get_hdf5_service()
            import h5py
            with h5py.File(file_path, "r") as f:
                cameras = hdf5_service.list_camera_candidate_paths(f, image_only=False)
                if not cameras:
                    cameras = hdf5_service.list_cameras(f)
                cameras_norm = [c.lstrip("/") for c in cameras]
                if not cameras or (camera_norm and camera_norm not in cameras_norm):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Camera '{camera}' not found. Available: {', '.join(cameras[:10])}{'...' if len(cameras) > 10 else ''}",
                    )

            def _hdf5_batch():
                out = []
                for i in range(count):
                    b = hdf5_service.get_frame_image(file_path, camera_norm or camera, start + i, quality=85)
                    if b is None:
                        break
                    out.append(base64.b64encode(b).decode("ascii"))
                return out

            frames_b64 = await asyncio.to_thread(_hdf5_batch)

        return {"start": start, "count": len(frames_b64), "frames": frames_b64}
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _mjpeg_stream_gen(file_path: str, camera: str, start_frame: int, fps: int):
    """生成 MJPEG 流（multipart/x-mixed-replace）"""
    boundary = b"frame"
    mcap_service = get_mcap_service()
    import time
    frame_interval = 1.0 / fps if fps > 0 else 0.1
    for jpeg_bytes in mcap_service.iter_frames(file_path, camera, start_frame):
        header = (
            b"--" + boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n"
        )
        yield header + jpeg_bytes + b"\r\n"
        time.sleep(frame_interval)


@router.get("/stream/mcap/{episode_id}")
async def stream_mcap_mjpeg(
    episode_id: str,
    camera: str = Query(..., description="相机 topic"),
    fps: int = Query(10, description="帧率", ge=1, le=30),
    start_frame: int = Query(0, description="起始帧", ge=0),
    taskId: Optional[str] = Query(None, description="任务 ID"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """MCAP MJPEG 流式播放，供 <img src> 使用，实现类视频连续播放"""
    config_service = get_task_config_service()
    if not taskId:
        raise HTTPException(status_code=400, detail="taskId required")
    _episode_info, _wh, file_path = await _resolve_task_episode_file_with_visibility(
        db, current_user, taskId, episode_id
    )
    if not file_path.lower().endswith(".mcap"):
        raise HTTPException(status_code=404, detail="MCAP file not found")
    mcap_service = get_mcap_service()
    cameras = mcap_service.list_cameras(file_path)
    if not cameras or camera not in cameras:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera}")

    def gen():
        return _mjpeg_stream_gen(file_path, camera, start_frame, fps)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


from typing import Callable, Iterator


def _ws_playback_resolve_file(file_path: str, fmt: str, camera: str) -> Tuple[int, Callable[[int], Optional[bytes]], Optional[Callable[[int], Iterator[bytes]]]]:
    """解析 WebSocket 播放用：返回 (frame_count, get_frame_fn, iter_frames_fn)。

    - 对 MCAP：优先使用 iter_frames 顺序推流，避免 get_frame_image 每帧从头遍历导致越播越慢
    - 对 HDF5：仅提供随机访问的 get_frame_fn
    """
    if fmt == "mcap":
        mcap_service = get_mcap_service()
        cameras = mcap_service.list_camera_candidate_topics(file_path)
        if not cameras:
            cameras = mcap_service.list_cameras(file_path)
        if not cameras or camera not in cameras:
            return 0, (lambda _idx: None), None
        frame_count = mcap_service.get_frame_count(file_path, camera)
        if frame_count <= 0:
            return 0, (lambda _idx: None), None
        def get_frame(idx: int):
            return mcap_service.get_frame_image(file_path, camera, idx)
        def iter_frames(start: int = 0):
            return mcap_service.iter_frames(file_path, camera, start)
        return frame_count, get_frame, iter_frames
    if fmt == "hdf5":
        hdf5_service = get_hdf5_service()
        if not hdf5_service._is_hdf5_file(file_path):
            return 0, (lambda _idx: None), None
        import h5py
        with h5py.File(file_path, "r") as f:
            cameras = hdf5_service.list_camera_candidate_paths(f, image_only=False)
            if not cameras:
                cameras = hdf5_service.list_cameras(f)
        if not cameras or camera not in cameras:
            return 0, (lambda _idx: None), None
        frame_count = hdf5_service.get_frame_count(file_path, camera)
        if frame_count <= 0:
            return 0, (lambda _idx: None), None
        def get_frame(idx: int):
            return hdf5_service.get_frame_image(file_path, camera, idx, quality=85)
        return frame_count, get_frame, None
    return 0, (lambda _idx: None), None


@router.websocket("/ws/playback/{episode_id}")
async def websocket_playback(
    websocket: WebSocket,
    episode_id: str,
    camera: str = Query(...),
    taskId: str = Query(...),
    current_user: User = Depends(get_current_user_ws),
):
    """WebSocket 推流播放，支持 MCAP 与 HDF5，协议一致：服务端按 fps 推帧，前端显示。"""
    from app.services.episode_storage import clear_episode_cache
    try:
        # 仅在握手阶段短暂占用连接，避免 WebSocket 长连接长期持有 DB 会话
        async with DataAssetsSessionLocal() as db:
            await _assert_label_task_visible_or_404(db, current_user, taskId)
    except HTTPException:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info(
        "[WS playback] connected episode=%s camera=%s taskId=%s",
        episode_id,
        camera[:80] if len(camera) > 80 else camera,
        taskId,
    )
    config_service = get_task_config_service()
    try:
        async with DataAssetsSessionLocal() as adb:
            _episode_info, _wh, file_path = await _resolve_task_episode_file_with_visibility(
                adb, current_user, taskId, episode_id
            )
        websocket.state.local_path = file_path
    except HTTPException:
        await websocket.close(code=4404)
        return
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
                    # seek 使用随机访问，MCAP 走 get_frame_image（命中后端 LRU 缓存），HDF5 亦然
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
                    # HDF5 或缺省：仍使用随机访问
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


@router.post("/annotation/generate", response_model=ApiResponse)
async def generate_annotation(
    request: GenerateAnnotationRequest = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """生成任务描述（异步）。可选 model / openai_api_key / openai_base_url 覆盖环境变量。"""
    episode_id = request.episode_id
    camera_name = request.camera_name
    task_id = request.taskId
    model = request.model
    openai_api_key = request.openai_api_key
    openai_base_url = request.openai_base_url
    try:
        tid = (task_id or "").strip()
        row = None
        if tid:
            row = await _assert_label_task_visible_or_404(db, current_user, tid)
            await assert_user_may_annotate_label_task(db, current_user, row)
            cfg_model, cfg_key, cfg_base = await _resolve_project_llm_config(
                db, (getattr(row, "project_id", None) or "").strip()
            )
            model = model or cfg_model
            openai_api_key = openai_api_key or cfg_key
            openai_base_url = openai_base_url or cfg_base
        service = get_annotation_service()
        job_id = await service.generate_description_async(
            episode_id, camera_name, task_id,
            user_id=str(getattr(current_user, "id", "") or ""),
            model=model, openai_api_key=openai_api_key, openai_base_url=openai_base_url,
        )
        return ApiResponse(ok=True, data={"jobId": job_id})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotation/status/{job_id}", response_model=ApiResponse)
async def get_annotation_status(job_id: str, db: AsyncSession = Depends(get_data_assets_db)):
    """获取标注任务状态。任务成功时尽力将结果写入 data_assets.instruction_text（与 worker 是否回写无关）。"""
    try:
        # 首选：task_jobs 持久层（避免 in-memory 状态在不同进程间不一致）
        row = get_task_job(job_id)
        if row is not None:
            raw_status = (getattr(row, "status", None) or "").strip().lower()

            # 前端期望的状态集合：pending/running/completed/failed/cancelled
            if raw_status in ("queued", "pending"):
                ui_status = "pending"
                progress = 0
            elif raw_status == "running":
                ui_status = "running"
                progress = 10
            elif raw_status in ("success", "succeeded", "completed"):
                ui_status = "completed"
                progress = 100
            elif raw_status == "failed":
                ui_status = "failed"
                progress = 0
            elif raw_status == "cancelled":
                ui_status = "cancelled"
                progress = 0
            else:
                ui_status = raw_status or "running"
                progress = 10 if ui_status == "running" else 0

            result = None
            if ui_status == "completed":
                r = getattr(row, "result", None)
                if isinstance(r, dict):
                    # dispatcher/worker 通常写入形如 {"value": "..."}
                    if "value" in r:
                        result = r.get("value")
                    elif "result" in r:
                        result = r.get("result")
                    else:
                        # 尽量兜底：把结构化结果当作字符串展示
                        result = str(r)
                elif isinstance(r, str):
                    result = r

            error = getattr(row, "error", None)
            if ui_status != "failed":
                error = None

            if ui_status == "completed" and result is not None and str(result).strip():
                try:
                    payload = getattr(row, "payload", None) or {}
                    params = payload.get("params") if isinstance(payload, dict) else {}
                    if isinstance(params, dict) and str(params.get("type") or "").strip().lower() == "annotation":
                        await persist_annotation_instruction_to_data_asset(db, params, str(result))
                except Exception:
                    logger.exception("annotation/status 补写 data_assets.instruction_text 失败 job_id=%s", job_id)

            return ApiResponse(
                ok=True,
                data={
                    "status": ui_status,
                    "progress": int(progress),
                    **({"result": result} if result is not None else {}),
                    **({"error": error} if error else {}),
                },
            )

        # 兼容旧逻辑：回退到内存状态（如果 task_jobs 里没找到）
        service = get_annotation_service()
        status = service.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return ApiResponse(ok=True, data=status)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/annotation/cancel/{job_id}", response_model=ApiResponse)
async def cancel_annotation(job_id: str):
    """取消标注任务"""
    try:
        service = get_annotation_service()
        success = service.cancel_job(job_id)
        return ApiResponse(ok=success, data={"cancelled": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instructions/{episode_id}", response_model=ApiResponse)
async def get_instruction(
    episode_id: str,
    taskId: Optional[str] = Query(None, description="任务 ID")
):
    """获取标注内容。带 taskId 时从该任务专属的 instructions 读取（按任务隔离）"""
    try:
        config_service = get_task_config_service()
        hdf5_service = get_hdf5_service()
        
        if taskId:
            # 按任务：从任务目录 instructions.json 读，与 episodes 顺序一致
            instructions = config_service.load_task_instructions(taskId)
            return ApiResponse(ok=True, data={"instructions": instructions})
        
        # 兼容旧逻辑：从数据集目录 instruction.json 读（规定：与数据路径同目录）
        episode = hdf5_service._find_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        out_path = get_instruction_path_for_data_path(episode["path"])
        if not os.path.exists(out_path):
            return ApiResponse(ok=True, data={"instructions": []})
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return ApiResponse(ok=True, data={"instructions": []})
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "instructions" in data:
                    return ApiResponse(ok=True, data=data)
            except json.JSONDecodeError:
                pass
        return ApiResponse(ok=True, data={"instructions": []})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/instructions/{episode_id}", response_model=ApiResponse)
async def save_instruction(
    episode_id: str,
    instruction: str = Body(..., embed=True),
    episode_index: Optional[int] = Body(None, embed=True),
    taskId: Optional[str] = Query(None, description="任务 ID"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """保存标注。带 taskId 时写入该任务专属的 instructions（按任务隔离）；同时将标注信息同步到 data_assets.instruction_text"""
    try:
        config_service = get_task_config_service()
        hdf5_service = get_hdf5_service()
        
        if taskId:
            lt_row = await _assert_label_task_visible_or_404(db, current_user, taskId)
            await assert_user_may_annotate_label_task(db, current_user, lt_row)
            episode_info = config_service.find_episode_by_id(taskId, episode_id)
            if not episode_info:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
            episodes_list = config_service.load_episodes_index(taskId) or []
            idx = episode_index
            if idx is None:
                idx = next((i for i, e in enumerate(episodes_list) if e.get("episode_id") == episode_id), 0)
            instructions = config_service.load_task_instructions(taskId)
            while len(instructions) <= idx:
                instructions.append("")
            instructions[idx] = instruction
            config_service.save_task_instructions(taskId, instructions)

            # 同时写入缓存目录侧方 instruction.json（与解析后的本机数据文件同目录）
            storage = EpisodeStorage(episode_info)
            wh = storage.get_storage_key()
            try:
                local_read = storage.resolve_local_path()
            except EpisodeResolveError:
                local_read = ""

            async def _episode_sidecar_parent(e: dict) -> str:
                w = EpisodeStorage(e).get_storage_key()
                if w.startswith("minio://"):
                    lf = await asyncio.to_thread(resolve_read_local_from_warehouse_uri, w)
                    return os.path.dirname(lf)
                try:
                    ap = EpisodeStorage(e).resolve_local_path()
                    return os.path.dirname(ap) if ap else ""
                except EpisodeResolveError:
                    return ""

            if local_read:
                out_path = get_instruction_path_for_data_path(local_read)
                out_dir = os.path.dirname(local_read)
                same_dir_idx: List[Tuple[int, dict]] = []
                for i, e in enumerate(episodes_list):
                    parent = await _episode_sidecar_parent(e)
                    if parent == out_dir:
                        same_dir_idx.append((i, e))
                same_dir_idx.sort(key=lambda x: EpisodeStorage(x[1]).get_storage_key())
                dir_idx = next((i for i, (_, e) in enumerate(same_dir_idx) if e.get("episode_id") == episode_id), 0)
                existing = []
                if os.path.exists(out_path):
                    try:
                        with open(out_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            existing = data.get("instructions") or []
                    except Exception:
                        pass
                while len(existing) <= dir_idx:
                    existing.append("")
                existing[dir_idx] = instruction
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"instructions": existing}, f, ensure_ascii=False, indent=2)

            # 同步到数据资产表：多候选匹配 file_path（与批量标注 / episodes 列表一致，避免键写法不一致导致未写库）
            candidates: List[str] = []
            if wh.startswith("minio://"):
                candidates.append(wh)
            sk = storage.get_storage_key()
            if sk and sk not in candidates:
                candidates.append(sk)
            if local_read and local_read not in candidates:
                candidates.append(local_read)
            meta_raw = episode_info.get("meta")
            if isinstance(meta_raw, dict):
                meta_json = json.dumps(meta_raw, ensure_ascii=False)
            elif isinstance(meta_raw, str):
                meta_json = meta_raw.strip() or None
            else:
                meta_json = None
            uri_meta = minio_uri_from_fields((episode_info.get("file_path") or "").strip() or None, meta_json)
            if uri_meta and uri_meta not in candidates:
                candidates.append(uri_meta)
            if candidates:
                asset = await find_asset_by_instruction_path_candidates(db, candidates)
            else:
                asset = None
            if asset is None:
                asset = await find_data_asset_for_label_episode(db, label_task_id=taskId, episode_id=episode_id)
            if asset:
                await update_asset(db, asset.id, instruction_text=instruction)
            return ApiResponse(ok=True, data={"saved": True})
        
        if not is_super_admin(current_user.role):
            raise HTTPException(status_code=403, detail="taskId required")
        # 兼容旧逻辑：写入数据集目录 instruction.json（规定：与数据路径同目录）
        episode = hdf5_service._find_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        file_path = episode["path"]
        out_path = get_instruction_path_for_data_path(file_path)
        existing_instructions = []
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        if isinstance(data, dict) and "instructions" in data:
                            existing_instructions = data["instructions"]
            except Exception:
                pass
        episode_index_val = episode_index if episode_index is not None else (int(episode_id) if episode_id.isdigit() else 0)
        while len(existing_instructions) <= episode_index_val:
            existing_instructions.append("")
        existing_instructions[episode_index_val] = instruction
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"instructions": existing_instructions}, f, ensure_ascii=False, indent=2)
        # 同步到数据资产表
        asset = await get_asset_by_file_path(db, file_path)
        if asset:
            await update_asset(db, asset.id, instruction_text=instruction)
        return ApiResponse(ok=True, data={"saved": True})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/instructions_file", response_model=ApiResponse)
async def get_task_instructions_file(
    task_id: str,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """获取任务数据集目录下的 instructions.json 内容（用于查看），若不存在或无标注则返回空数组"""
    try:
        await _assert_label_task_visible_or_404(db, current_user, task_id)
        config_service = get_task_config_service()
        config = config_service.load_task_config(task_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        dataset_path = config.get("dataset_path")
        if not dataset_path:
            return ApiResponse(ok=True, data={"content": json.dumps({"instructions": []}, ensure_ascii=False, indent=2)})
        paths = [p.strip() for p in dataset_path.split(",") if p.strip()]
        if not paths:
            return ApiResponse(ok=True, data={"content": json.dumps({"instructions": []}, ensure_ascii=False, indent=2)})
        first_path = paths[0]
        try:
            if first_path.startswith("minio://"):
                local_ref = await asyncio.to_thread(resolve_read_local_from_warehouse_uri, first_path)
            else:
                local_ref, _ = await asyncio.to_thread(resolve_label_task_warehouse_and_local, first_path)
        except FileNotFoundError:
            return ApiResponse(
                ok=True,
                data={"content": json.dumps({"instructions": []}, ensure_ascii=False, indent=2)},
            )
        except (MinioBucketError, MinioConfigError) as e:
            return ApiResponse(ok=False, error=str(e)[:300])
        out_path = get_instruction_path_for_data_path(local_ref)
        if not os.path.exists(out_path):
            return ApiResponse(ok=True, data={"content": json.dumps({"instructions": []}, ensure_ascii=False, indent=2)})
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return ApiResponse(ok=True, data={"content": json.dumps({"instructions": []}, ensure_ascii=False, indent=2)})
        try:
            data = json.loads(content)
            if not isinstance(data, dict) or "instructions" not in data:
                return ApiResponse(ok=True, data={"content": json.dumps({"instructions": []}, ensure_ascii=False, indent=2)})
            return ApiResponse(ok=True, data={"content": json.dumps(data, ensure_ascii=False, indent=2)})
        except json.JSONDecodeError:
            return ApiResponse(ok=True, data={"content": json.dumps({"instructions": []}, ensure_ascii=False, indent=2)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/instructions", response_model=ApiResponse)
async def get_task_instructions(
    task_id: str,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """获取该任务下所有 episode 的标注结果（按任务隔离，与 episodes 顺序一致）"""
    try:
        await _assert_label_task_visible_or_404(db, current_user, task_id)
        config_service = get_task_config_service()
        episodes = config_service.load_episodes_index(task_id)
        if not episodes:
            return ApiResponse(ok=True, data={"instructions": []})
        instructions = config_service.load_task_instructions(task_id)
        # 长度与 episodes 一致，不足补空串
        while len(instructions) < len(episodes):
            instructions.append("")
        return ApiResponse(ok=True, data={"instructions": instructions[:len(episodes)]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotations/download_one", response_model=ApiResponse)
async def download_annotation_one(
    taskId: str = Query(..., description="任务 ID"),
    episodeId: str = Query(..., description="Episode ID"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """从数据仓库（data_assets.instruction_text）查询单条标注并返回，用于「下载当前条」"""
    try:
        await _assert_label_task_visible_or_404(db, current_user, taskId)
        config_service = get_task_config_service()
        episode_info = config_service.find_episode_by_id(taskId, episodeId)
        if not episode_info:
            raise HTTPException(status_code=404, detail="Episode not found")
        # 这里取“资产 key”（用于 DB 匹配 instruction_text），不需要本机路径
        path_key = EpisodeStorage(episode_info).get_storage_key()
        if not path_key:
            raise HTTPException(status_code=404, detail="Episode path not found")
        from app.crud.data_asset import get_instruction_text_by_paths
        path_to_instruction = await get_instruction_text_by_paths(db, [path_key])
        instruction = path_to_instruction.get(normalize_storage_path_key(path_key), "") or ""
        return ApiResponse(ok=True, data={"instruction": instruction})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotations/download_batch", response_model=ApiResponse)
async def download_annotations_batch(
    taskId: str = Query(..., description="任务 ID"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """从数据仓库（data_assets.instruction_text）批量查询当前任务下所有 episode 的标注，用于「下载整个数据集」"""
    try:
        await _assert_label_task_visible_or_404(db, current_user, taskId)
        config_service = get_task_config_service()
        episodes = config_service.load_episodes_index(taskId)
        if not episodes:
            return ApiResponse(ok=True, data={"items": []})
        path_list = []
        for ep in episodes:
            p = EpisodeStorage(ep).get_storage_key()
            if p:
                path_list.append(p)
        from app.crud.data_asset import get_instruction_text_by_paths
        path_to_instruction = await get_instruction_text_by_paths(db, path_list)
        items = []
        for ep in episodes:
            episode_id = ep.get("episode_id", "")
            path = EpisodeStorage(ep).get_storage_key()
            instruction = (path_to_instruction.get(normalize_storage_path_key(path), "") or "") if path else ""
            items.append({"episode_id": episode_id, "path": path, "instruction": instruction})
        return ApiResponse(ok=True, data={"items": items})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=ApiResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    task_id: Optional[str] = None,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """上传 HDF5 或 MCAP 文件"""
    try:
        tid = (task_id or "").strip()
        if tid:
            await _assert_label_task_manage(db, current_user, tid)
        elif not is_super_admin(current_user.role):
            raise HTTPException(status_code=403, detail="Admin privilege required")
        if not file.filename.lower().endswith((".hdf5", ".h5", ".mcap")):
            raise HTTPException(status_code=400, detail="Only .hdf5, .h5 and .mcap files are allowed")
        
        service = get_hdf5_service()
        
        # 如果指定了 task_id，保存到任务目录
        if task_id:
            task_dir = os.path.join(service.data_dir, task_id)
            os.makedirs(task_dir, exist_ok=True)
            file_path = os.path.join(task_dir, file.filename)
        else:
            file_path = os.path.join(service.data_dir, file.filename)
        
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 提取 episode_id
        episode_id = service._extract_episode_id(file.filename)
        
        return ApiResponse(ok=True, data={
            "episodeId": episode_id,
            "filename": file.filename,
            "path": file_path
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/import_status", response_model=ApiResponse)
async def get_task_import_status(
    task_id: str,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """获取任务导入状态（与 load_dataset 口径一致：基于 tasks/{task_id}/episodes_index.json）"""
    try:
        await _assert_label_task_visible_or_404(db, current_user, task_id)
        config_service = get_task_config_service()
        episodes = config_service.load_episodes_index(task_id) or []
        return ApiResponse(ok=True, data={
            "imported": len(episodes) > 0,
            "episode_count": len(episodes)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=ApiResponse)
async def list_label_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=1000),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """从 label_tasks 表获取标注任务列表（PostgreSQL），严格按表结构返回"""
    stmt = select(LabelTask).order_by(LabelTask.created_at.desc())
    scoped = await scoped_project_ids_for_platform_tasks(db, current_user)
    if scoped is not None:
        if not scoped:
            return ApiResponse(ok=True, data=[])
        stmt = stmt.where(LabelTask.project_id.in_(list(scoped)))
    stmt = stmt.offset(skip).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()
    # 按数据库字段返回，便于前端严格按表显示
    out = []
    for r in rows:
        def _iso(t) -> str:
            if t is None:
                return ""
            if hasattr(t, "isoformat"):
                return t.isoformat()
            return str(t)
        out.append({
            "id": r.id,
            "task_id": r.task_id,
            "name": r.name,
            "dataset_path": r.dataset_path or "",
            "dataset_ids": r.dataset_ids,  # JSON 字符串，前端可 parse
            "dataset_source": r.dataset_source,
            "data_count": r.data_count,
            "device_type": r.device_type,
            "project_id": r.project_id,
            "labeler": r.labeler,
            "reviewer": r.reviewer,
            "collector": r.collector or "",
            "completed": bool(r.completed),
            "verified": bool(r.verified),
            "created_at": _iso(r.created_at),
            "updated_at": _iso(r.updated_at),
        })
    return ApiResponse(ok=True, data=out)


@router.get("/tasks/{task_id}/actor", response_model=ApiResponse)
async def get_label_task_actor_context(
    task_id: str,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """前端门禁用：返回任务标注/审核相关字段及项目 owner_id（不改变可见性规则）。"""
    row = await _assert_label_task_visible_or_404(db, current_user, task_id)
    pid = (getattr(row, "project_id", None) or "").strip()
    owner_id: Optional[str] = None
    if pid:
        proj = await get_project_by_id(db, pid)
        if proj is not None:
            oid = getattr(proj, "owner_id", None)
            owner_id = (str(oid).strip() if oid else None) or None
    return ApiResponse(
        ok=True,
        data={
            "task_id": getattr(row, "task_id", None) or task_id,
            "name": getattr(row, "name", None),
            "project_id": pid or None,
            "project_owner_id": owner_id,
            "labeler": getattr(row, "labeler", None),
            "reviewer": getattr(row, "reviewer", None),
        },
    )


@router.post("/tasks", response_model=ApiResponse)
async def create_label_task(
    request: CreateLabelTaskRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """创建标注任务并保存配置到 task.json，同时写入 label_tasks 表（PostgreSQL）；列表接口从同表读取并展示。"""
    try:
        config_service = get_task_config_service()

        project_id = (request.project_id or "").strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="请选择所属项目")
        await assert_platform_task_manage_project(db, current_user, project_id)
        await assert_labeler_reviewer_are_project_members(
            db, project_id, labeler=request.labeler, reviewer=request.reviewer
        )
        project = (await db.execute(text("SELECT id, status FROM projects WHERE id = :pid"), {"pid": project_id})).first()
        if not project:
            raise HTTPException(status_code=404, detail="项目不存在")
        if (str(project[1] or "").strip()) == "已归档":
            raise HTTPException(status_code=403, detail="项目已归档，禁止该操作")
        
        # 处理 dataset_ids：如果提供了 dataset_ids，则从数据库获取 MinIO 路径
        dataset_path = None
        dataset_ids_to_store = None  # 继承自 data_assets.dataset_id，供执行时校验
        if request.dataset_ids and len(request.dataset_ids) > 0:
            dataset_paths = []
            if request.dataset_source == "data_assets":
                # 从 data_assets 表解析路径，并收集 dataset_id（继承作为 label_task 的存在性校验依据）
                from app.crud.data_asset import get_asset_by_id
                dataset_ids_from_assets = []  # data_assets.dataset_id 列表，如 ["DS000001"]
                for asset_id in request.dataset_ids:
                    asset = await get_asset_by_id(db, asset_id)
                    if asset:
                        sync_status = (getattr(asset, "sync_status", "synced") or "synced").strip().lower()
                        if sync_status != "synced":
                            raise HTTPException(status_code=409, detail=f"数据资产未同步，暂不可标注: {asset.filename}")
                        wh = minio_uri_from_fields(getattr(asset, "file_path", None), getattr(asset, "meta", None))
                        if not wh:
                            raise HTTPException(
                                status_code=409,
                                detail=f"数据资产缺少 MinIO 地址，暂不可标注: {asset.filename}",
                            )
                        dataset_paths.append(wh)
                        if asset.dataset_id:
                            dataset_ids_from_assets.append(asset.dataset_id)
            else:
                # 默认从 hdf5_datasets 表解析路径（与数据资产同库 PostgreSQL）
                from app.crud.hdf5_dataset import get_dataset_by_id
                for dataset_id in request.dataset_ids:
                    dataset = await get_dataset_by_id(db, dataset_id)
                    if dataset and dataset.storage_uri:
                        p = str(dataset.storage_uri).strip()
                        if not p.startswith("minio://"):
                            raise HTTPException(status_code=400, detail=f"仅支持 MinIO 数据路径: {p}")
                        dataset_paths.append(p)

            if not dataset_paths:
                raise HTTPException(status_code=404, detail="未找到指定的数据集")

            if request.dataset_source == "data_assets":
                dataset_ids_to_store = dataset_ids_from_assets
            if len(dataset_paths) == 1:
                dataset_path = dataset_paths[0]
            else:
                dataset_path = ','.join(dataset_paths)
        elif request.dataset_path:
            dataset_path = request.dataset_path.strip()
        else:
            raise HTTPException(status_code=400, detail="必须提供 dataset_path 或 dataset_ids")
        
        # 验证 dataset_path
        if not dataset_path:
            raise HTTPException(status_code=400, detail="dataset_path 不能为空")
        
        # 处理多个路径（逗号分隔）
        if ',' in dataset_path:
            dataset_paths_to_validate = [p.strip() for p in dataset_path.split(',') if p.strip()]
        else:
            dataset_paths_to_validate = [dataset_path.strip()]
        
        if not dataset_paths_to_validate:
            raise HTTPException(status_code=400, detail="数据集路径为空")
        
        # 验证所有路径（硬规则：仅允许 minio://）
        invalid_paths = []
        for path in dataset_paths_to_validate:
            if not str(path).startswith("minio://"):
                invalid_paths.append(path)
        
        if invalid_paths:
            raise HTTPException(
                status_code=400,
                detail=f"仅支持 MinIO 数据路径（minio://...）: {','.join(invalid_paths)}"
            )
        
        # 生成 task_id（简单实现，实际可用 UUID）
        task_id = hashlib.md5(f"{request.name}_{dataset_path}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # 保存任务配置
        config = {
            "task_id": task_id,
            "name": request.name,
            "dataset_path": dataset_path,
            "dataset_ids": dataset_ids_to_store if dataset_ids_to_store is not None else request.dataset_ids,  # 优先用 data_assets.dataset_id
            "dataset_source": request.dataset_source,  # "data_assets" 表示来自数据资产
            "data_count": request.data_count,
            "device_type": request.device_type or "",  # 可选，已废弃
            "project_id": request.project_id or "",
            "labeler": request.labeler,
            "reviewer": request.reviewer,
            "collector": request.collector,
            "created_at": datetime.now().isoformat(),
        }
        
        config_service.save_task_config(task_id, config)

        # label_tasks 表由启动时 create_all 或迁移脚本创建（PostgreSQL）
        # 写入 label_tasks 数据库表
        # label_tasks.dataset_ids 存“可回填编辑弹窗”的原始资产 ID 列表（数字）
        # 任务执行/校验仍使用 task.json 中的 dataset_ids（可为 DSxxxx）逻辑，不受影响
        dataset_ids_json = json.dumps(request.dataset_ids) if request.dataset_ids else None
        label_task = LabelTask(
            task_id=task_id,
            name=request.name,
            dataset_path=dataset_path,
            dataset_ids=dataset_ids_json,
            dataset_source=request.dataset_source,
            data_count=request.data_count,
            device_type=request.device_type or None,
            project_id=request.project_id or None,
            labeler=request.labeler,
            reviewer=request.reviewer,
            collector=request.collector,
            completed=False,
            verified=False,
        )
        db.add(label_task)
        await db.flush()
        await db.commit()

        # 若来自数据资产页，则将相关数据资产的更新时间刷新并写入标注任务名称（便于按任务划分展示）
        if request.dataset_ids and request.dataset_source == "data_assets":
            try:
                from app.crud.data_asset import get_asset_by_id
                for asset_id in request.dataset_ids:
                    asset = await get_asset_by_id(db, asset_id)
                    if asset:
                        asset.updated_at = datetime.utcnow()
                        asset.label_task_name = request.name or asset.label_task_name
                await db.commit()
            except Exception:
                # 更新时间失败不影响标注任务创建
                pass

        # 在数据集目录下创建 instruction.json（规定：与数据路径同目录；若不存在则创建空文件）
        seen_dirs = set()
        for path in dataset_paths_to_validate:
            out_path = get_instruction_path_for_data_path(path)
            out_dir = os.path.dirname(out_path)
            if out_dir in seen_dirs:
                continue
            seen_dirs.add(out_dir)
            if not os.path.exists(out_path):
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"instructions": []}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        audit_pname = await _label_audit_project_display_name(
            db,
            project_id=request.project_id,
            label_task_name=request.name,
        )
        enqueue_audit_log(
            background_tasks,
            user=current_user,
            request=http_request,
            action_type=AA.CREATE_TASK,
            project_id=request.project_id,
            project_name=audit_pname,
            resource_type=AR.LABEL_JOB,
            resource_id=task_id,
            resource_name=request.name,
            detail_json={"dataset_source": request.dataset_source, "domain": "label"},
        )
        return ApiResponse(ok=True, data={
            "task_id": task_id,
            "name": request.name,
            "dataset_path": dataset_path,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateLabelTaskRequest(BaseModel):
    name: Optional[str] = None
    data_count: Optional[int] = None
    device_type: Optional[str] = None
    labeler: Optional[str] = None
    reviewer: Optional[str] = None
    collector: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_ids: Optional[List[int]] = None
    dataset_source: Optional[str] = None
    project_id: Optional[str] = None
    completed: Optional[bool] = None
    verified: Optional[bool] = None

    @field_validator("dataset_ids", mode="before")
    @classmethod
    def coerce_dataset_ids(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            return v
        out = []
        for x in v:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str):
                try:
                    out.append(int(x))
                except (ValueError, TypeError):
                    pass
        return out if out else None


@router.patch("/tasks/{task_id}", response_model=ApiResponse)
async def update_label_task(
    task_id: str,
    body: UpdateLabelTaskRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """更新标注任务配置（部分字段），同时更新 label_tasks 表"""
    try:
        row = await _assert_label_task_visible_or_404(db, current_user, task_id)
        updates = body.model_dump(exclude_unset=True)
        # 权限拆分：
        # - completed：标注员/可标注角色可切换
        # - verified：审核员可切换
        # - 其余字段：项目管理权限可修改
        manage_keys = {
            "name",
            "data_count",
            "device_type",
            "labeler",
            "reviewer",
            "collector",
            "dataset_path",
            "dataset_ids",
            "dataset_source",
            "project_id",
        }
        need_manage = bool(set(updates.keys()) & manage_keys)
        need_annotate = "completed" in updates
        need_review = "verified" in updates
        if need_manage:
            pid0 = (getattr(row, "project_id", None) or "").strip()
            await assert_platform_task_manage_project(db, current_user, pid0)
            if "project_id" in updates:
                new_pid = (updates.get("project_id") or "").strip()
                if new_pid:
                    await assert_platform_task_manage_project(db, current_user, new_pid)
        if need_annotate:
            await assert_user_may_annotate_label_task(db, current_user, row)
        if need_review:
            await assert_user_may_review_label_task(db, current_user, row)
        if need_manage:
            if "labeler" in updates or "reviewer" in updates or "project_id" in updates:
                eff_pid = updates["project_id"] if "project_id" in updates else row.project_id
                eff_pid = (eff_pid or "").strip()
                if eff_pid:
                    lab = updates["labeler"] if "labeler" in updates else row.labeler
                    rev = updates["reviewer"] if "reviewer" in updates else row.reviewer
                    await assert_labeler_reviewer_are_project_members(
                        db, eff_pid, labeler=lab, reviewer=rev
                    )
        config_service = get_task_config_service()
        config = config_service.load_task_config(task_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

        # 允许编辑任务时更新 dataset_ids / dataset_path；dataset_ids 优先。
        if "dataset_ids" in updates:
            ds_ids = updates.get("dataset_ids") or []
            if ds_ids:
                ds_source = (updates.get("dataset_source") or config.get("dataset_source") or "data_assets").strip()
                dataset_paths = []
                dataset_ids_from_assets = []
                if ds_source == "data_assets":
                    from app.crud.data_asset import get_asset_by_id
                    for asset_id in ds_ids:
                        asset = await get_asset_by_id(db, int(asset_id))
                        if asset:
                            sync_status = (getattr(asset, "sync_status", "synced") or "synced").strip().lower()
                            if sync_status != "synced":
                                raise HTTPException(status_code=409, detail=f"数据资产未同步，暂不可标注: {asset.filename}")
                            wh = minio_uri_from_fields(getattr(asset, "file_path", None), getattr(asset, "meta", None))
                            if not wh:
                                raise HTTPException(
                                    status_code=409,
                                    detail=f"数据资产缺少 MinIO 地址，暂不可标注: {asset.filename}",
                                )
                            dataset_paths.append(wh)
                            if asset.dataset_id:
                                dataset_ids_from_assets.append(asset.dataset_id)
                else:
                    from app.crud.hdf5_dataset import get_dataset_by_id
                    for dataset_id in ds_ids:
                        dataset = await get_dataset_by_id(db, int(dataset_id))
                        if dataset and dataset.storage_uri:
                            p = str(dataset.storage_uri).strip()
                            if not p.startswith("minio://"):
                                raise HTTPException(status_code=400, detail=f"仅支持 MinIO 数据路径: {p}")
                            dataset_paths.append(p)
                if not dataset_paths:
                    raise HTTPException(status_code=404, detail="未找到指定的数据集")
                config["dataset_path"] = dataset_paths[0] if len(dataset_paths) == 1 else ",".join(dataset_paths)
                config["dataset_source"] = ds_source
                config["dataset_ids"] = dataset_ids_from_assets if ds_source == "data_assets" else ds_ids
                updates["dataset_path"] = config["dataset_path"]
                updates["dataset_source"] = ds_source
            elif "dataset_path" in updates and updates.get("dataset_path"):
                config["dataset_path"] = str(updates["dataset_path"]).strip()
                config["dataset_source"] = updates.get("dataset_source") or config.get("dataset_source")
                config["dataset_ids"] = None
            else:
                raise HTTPException(status_code=400, detail="必须提供 dataset_path 或 dataset_ids")
        elif "dataset_path" in updates and updates.get("dataset_path"):
            config["dataset_path"] = str(updates["dataset_path"]).strip()
            if "dataset_source" in updates and updates.get("dataset_source") is not None:
                config["dataset_source"] = updates.get("dataset_source")
            config["dataset_ids"] = config.get("dataset_ids")
        if "project_id" in updates:
            config["project_id"] = updates["project_id"]
        for key in ("name", "data_count", "device_type", "labeler", "reviewer", "collector", "completed", "verified", "dataset_path", "dataset_source"):
            if key in updates and updates[key] is not None:
                config[key] = updates[key]
        config_service.save_task_config(task_id, config)

        # 同步更新 label_tasks 表
        from sqlalchemy import select, update
        stmt = select(LabelTask).where(LabelTask.task_id == task_id)
        result = await db.execute(stmt)
        db_task = result.scalar_one_or_none()
        old_completed = bool(getattr(db_task, "completed", False)) if db_task else False
        old_verified = bool(getattr(db_task, "verified", False)) if db_task else False
        if db_task:
            update_data = {
                k: updates[k]
                for k in ("name", "data_count", "device_type", "labeler", "reviewer", "collector", "project_id", "completed", "verified", "dataset_path", "dataset_source")
                if k in updates
            }
            if "dataset_ids" in updates:
                update_data["dataset_ids"] = json.dumps(updates.get("dataset_ids") or [])
            if update_data:
                await db.execute(update(LabelTask).where(LabelTask.task_id == task_id).values(**update_data))
                await db.commit()

        stmt2 = select(LabelTask).where(LabelTask.task_id == task_id)
        result2 = await db.execute(stmt2)
        db_task2 = result2.scalar_one_or_none()
        pid = (getattr(db_task2, "project_id", None) or "").strip() or None
        task_display_name = str((getattr(db_task2, "name", None) or task_id) if db_task2 else task_id)
        audit_pname = await _label_audit_project_display_name(
            db,
            project_id=pid,
            label_task_name=task_display_name,
        )
        base_audit = dict(
            user=current_user,
            request=http_request,
            project_id=pid,
            project_name=audit_pname,
            resource_type=AR.LABEL_JOB,
            resource_id=task_id,
            resource_name=task_display_name,
        )
        if db_task2:
            new_completed = bool(db_task2.completed)
            new_verified = bool(db_task2.verified)
            if not old_completed and new_completed:
                enqueue_audit_log(background_tasks, action_type=AA.SUBMIT_LABEL_RESULT, **base_audit)
            if not old_verified and new_verified:
                enqueue_audit_log(background_tasks, action_type=AA.APPROVE_LABEL_REVIEW, **base_audit)
            if old_verified and not new_verified and "verified" in updates:
                enqueue_audit_log(background_tasks, action_type=AA.REJECT_LABEL_REVIEW, **base_audit)

        return ApiResponse(ok=True, data={"updated": True})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}", response_model=ApiResponse)
async def delete_label_task(
    task_id: str,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """删除标注任务（同步删除 label_tasks 表及 task.json 等文件）"""
    try:
        from sqlalchemy import select, delete as sql_delete
        config_service = get_task_config_service()

        db_task = await _assert_label_task_manage(db, current_user, task_id)
        task_name = getattr(db_task, "name", None) or task_id
        pid = (getattr(db_task, "project_id", None) or "").strip() or None
        audit_pname = await _label_audit_project_display_name(
            db,
            project_id=pid,
            label_task_name=str(task_name) if task_name else None,
        )

        # 1. 删除 label_tasks 表记录
        await db.execute(sql_delete(LabelTask).where(LabelTask.task_id == task_id))
        await db.commit()

        # 2. 删除 task.json、episodes_index.json、instructions.json 等文件
        task_dir = os.path.join(config_service.tasks_dir, task_id)
        if os.path.isdir(task_dir):
            import shutil
            try:
                shutil.rmtree(task_dir)
            except Exception as e:
                logger.warning("删除任务目录失败 %s: %s", task_dir, e)

        enqueue_audit_log(
            background_tasks,
            user=current_user,
            request=http_request,
            action_type=AA.DELETE_TASK,
            project_id=pid,
            project_name=audit_pname,
            resource_type=AR.LABEL_JOB,
            resource_id=task_id,
            resource_name=task_name,
            detail_json={"domain": "label"},
        )

        return ApiResponse(ok=True, data={"deleted": True})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/load_dataset", response_model=ApiResponse)
async def load_task_dataset(
    task_id: str,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """扫描任务的数据集路径并生成 episode 列表"""
    try:
        config_service = get_task_config_service()
        hdf5_service = get_hdf5_service()
        
        # 读取任务配置（磁盘）；若缺失但 PostgreSQL 仍有记录，则按表字段重建 task.json（常见于数据目录迁移或卷清空）
        config = config_service.load_task_config(task_id)
        if not config:
            row = await _assert_label_task_visible_or_404(db, current_user, task_id)
            config = _label_task_row_to_config_dict(task_id, row)
            if not (config.get("dataset_path") or "").strip():
                raise HTTPException(
                    status_code=404,
                    detail=f"任务 {task_id} 无数据集路径，无法扫描；请在库中检查 label_tasks.dataset_path",
                )
            config_service.save_task_config(task_id, config)
        pid = (config.get("project_id") or "").strip()
        await assert_label_task_in_execute_scope(db, current_user, pid)
        
        dataset_path = config.get("dataset_path")
        if not dataset_path:
            raise HTTPException(status_code=400, detail="任务未设置 dataset_path")

        # 若任务有 dataset_ids（继承自 data_assets.dataset_id），先按 dataset_id 校验数据是否存在
        if config.get("dataset_source") == "data_assets" and config.get("dataset_ids"):
            try:
                from app.db.data_assets_session import DataAssetsSessionLocal
                from app.crud.data_asset import are_dataset_ids_valid
                ds_json = json.dumps(config["dataset_ids"]) if isinstance(config["dataset_ids"], list) else config.get("dataset_ids")
                async with DataAssetsSessionLocal() as db:
                    if not await are_dataset_ids_valid(db, ds_json):
                        raise HTTPException(status_code=404, detail="无数据：关联的数据资产已被删除")
            except HTTPException:
                raise
            except Exception:
                pass

        # 处理多个路径（逗号分隔）
        # 如果 dataset_path 包含逗号，说明是多个路径
        if ',' in dataset_path:
            dataset_paths = [p.strip() for p in dataset_path.split(',') if p.strip()]
        else:
            dataset_paths = [dataset_path.strip()]
        
        if not dataset_paths:
            raise HTTPException(status_code=400, detail="数据集路径为空")
        
        # 先轻量解析 MinIO URI，避免在「进入标注页」阶段预下载超大文件
        resolved_uris: List[str] = []
        missing_paths: List[str] = []
        for path in dataset_paths:
            try:
                wh_uri = await asyncio.to_thread(resolve_label_task_warehouse_uri, path)
                resolved_uris.append(wh_uri)
            except FileNotFoundError:
                missing_paths.append(path)
            except (MinioBucketError, MinioConfigError) as e:
                raise HTTPException(status_code=502, detail=str(e)[:500])

        if missing_paths:
            msg = "无数据：关联的数据资产已被删除" if len(missing_paths) == len(dataset_paths) else f"部分数据集路径不存在: {','.join(missing_paths)}"
            raise HTTPException(status_code=404, detail=msg)
        
        # 扫描 HDF5 与 MCAP 文件
        SUPPORTED_EXTENSIONS = (".hdf5", ".h5", ".mcap")
        episodes = []
        
        for warehouse_uri in resolved_uris:
            uri = (warehouse_uri or "").strip()
            if not uri.startswith("minio://"):
                continue
            body = uri.removeprefix("minio://")
            if "/" not in body:
                continue
            bucket, key = body.split("/", 1)
            key = key.strip()
            if not key:
                continue
            # 前缀路径：列举对象并按扩展名过滤（不下载对象）
            if key.endswith("/"):
                try:
                    object_names = await asyncio.to_thread(list_object_names_under_prefix, bucket, key)
                except MinioBucketError as e:
                    raise HTTPException(status_code=502, detail=str(e)[:500])
                for object_name in object_names:
                    wh_path = f"minio://{bucket}/{object_name}"
                    if not wh_path.lower().endswith(SUPPORTED_EXTENSIONS):
                        continue
                    episode_id = hashlib.md5(f"{task_id}_{wh_path}".encode()).hexdigest()[:16]
                    episodes.append({
                        "episode_id": episode_id,
                        "filename": os.path.basename(object_name),
                        "abs_path": "",  # 按需下载，真正读取时再解析到本地缓存路径
                        "warehouse_path": wh_path,
                        "mtime": None,
                        "size_bytes": None,
                    })
                continue

            # 单对象路径：直接生成 episode（不预下载）
            if uri.lower().endswith(SUPPORTED_EXTENSIONS):
                episode_id = hashlib.md5(f"{task_id}_{uri}".encode()).hexdigest()[:16]
                episodes.append({
                    "episode_id": episode_id,
                    "filename": os.path.basename(key),
                    "abs_path": "",  # 按需下载，真正读取时再解析到本地缓存路径
                    "warehouse_path": uri,
                    "mtime": None,
                    "size_bytes": None,
                })
        
        # 保存到 episodes_index.json
        config_service.save_episodes_index(task_id, episodes)
        
        return ApiResponse(ok=True, data={
            "task_id": task_id,
            "dataset_path": dataset_path,
            "episodes": episodes,
            "count": len(episodes)
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/annotation/batch", response_model=ApiResponse)
async def batch_annotation(
    request: BatchAnnotationRequest = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """批量自动标注（无 task 上下文，仅超级管理员可用）。"""
    if not is_super_admin(current_user.role):
        raise HTTPException(status_code=403, detail="无权限执行批量标注")
    try:
        service = get_batch_annotation_service()
        hdf5_service = get_hdf5_service()
        
        # 将 episode_ids 转换为文件路径
        dataset_paths = []
        for episode_id in request.episode_ids:
            episode = hdf5_service._find_episode_by_id(episode_id)
            if episode:
                dataset_paths.append(episode["path"])
            else:
                # 如果找不到，尝试直接使用 episode_id 作为路径
                if os.path.exists(episode_id):
                    dataset_paths.append(episode_id)
        
        if not dataset_paths:
            raise HTTPException(status_code=404, detail="No valid datasets found")
        
        # 执行批量标注
        results = service.perform_batch_annotation(
            dataset_paths,
            camera_name=request.camera_name,
            fallback_first_camera=request.fallback_first_camera
        )
        
        # 同步到数据资产表：按 dataset_path 更新 instruction_text
        for r in results:
            if "error" in r:
                continue
            path = EpisodeStorage(r).get_storage_key() or (r.get("dataset_path") or "").strip()
            if not path:
                continue
            ep = r.get("episode_data") or {}
            text = (ep.get("tasks") or [""])[0] if ep.get("tasks") else ""
            asset = await get_asset_by_file_path(db, path)
            if asset:
                await update_asset(db, asset.id, instruction_text=text)
        
        return ApiResponse(ok=True, data={"results": results})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/annotation/batch_by_task", response_model=ApiResponse)
async def batch_annotation_by_task(
    request: BatchAnnotationByTaskRequest = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """按任务批量自动标注：对任务下所有 episode 依次生成标注并写入 instruction.json。
    在线程池中执行，避免阻塞事件循环导致 WebSocket 播放卡顿；成功后同步标注信息到 data_assets.instruction_text。"""
    try:
        tid = (request.taskId or "").strip()
        if not tid:
            raise HTTPException(status_code=400, detail="taskId required")
        lt = (
            await db.execute(select(LabelTask).where(LabelTask.task_id == tid))
        ).scalar_one_or_none()
        if lt is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        await _assert_label_task_visible_or_404(db, current_user, tid)
        await assert_user_may_annotate_label_task(db, current_user, lt)
        cfg_model, cfg_key, cfg_base = await _resolve_project_llm_config(
            db, (getattr(lt, "project_id", None) or "").strip()
        )
        if not request.model:
            request.model = cfg_model
        if not request.openai_api_key:
            request.openai_api_key = cfg_key
        if not request.openai_base_url:
            request.openai_base_url = cfg_base

        config_service = get_task_config_service()
        service = get_batch_annotation_service()

        def _run():
            return service.perform_batch_annotation_by_task(
                request.taskId,
                config_service,
                camera_name=request.camera_name,
                fallback_first_camera=request.fallback_first_camera,
                model=request.model,
                openai_api_key=request.openai_api_key,
                openai_base_url=request.openai_base_url,
            )

        results = await asyncio.to_thread(_run)
        # 同步到数据资产表：instruction_text 供 GET /episodes 左侧「已标注」使用；需兼容 minio 键 / 本地路径 / 规范化差异
        # 与 POST /instructions/{episode_id}（save_instruction）保持同一套候选路径与回退逻辑，避免批量仅写任务 JSON 而 data_assets 未更新
        for r in results:
            if "error" in r:
                continue
            text = r.get("instruction") or ""
            wp = (r.get("warehouse_path") or "").strip()
            lp = (r.get("path") or "").strip()
            storage_key = (EpisodeStorage(r).get_storage_key() or "").strip()
            candidates: list = []
            if storage_key:
                candidates.append(storage_key)
            if wp and wp not in candidates:
                candidates.append(wp)
            if lp and lp not in candidates:
                candidates.append(lp)
            episode_info = config_service.find_episode_by_id(tid, str(r.get("episode_id") or ""))
            if episode_info:
                meta_raw = episode_info.get("meta")
                if isinstance(meta_raw, dict):
                    meta_json = json.dumps(meta_raw, ensure_ascii=False)
                elif isinstance(meta_raw, str):
                    meta_json = meta_raw.strip() or None
                else:
                    meta_json = None
                uri_meta = minio_uri_from_fields((episode_info.get("file_path") or "").strip() or None, meta_json)
                if uri_meta and uri_meta not in candidates:
                    candidates.append(uri_meta)
            if not candidates:
                continue
            asset = await find_asset_by_instruction_path_candidates(db, candidates)
            if asset is None and r.get("episode_id") is not None:
                asset = await find_data_asset_for_label_episode(
                    db, label_task_id=tid, episode_id=str(r.get("episode_id"))
                )
            if asset:
                await update_asset(db, asset.id, instruction_text=text)
            else:
                logger.warning(
                    "batch_by_task: 未匹配 data_assets 行，左侧「已标注」不会更新 episode_id=%s candidates=%s",
                    r.get("episode_id"),
                    candidates[:5],
                )
        return ApiResponse(ok=True, data={"results": results})
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.exception("batch_annotation_by_task 失败 taskId=%s: %s", request.taskId, e)
        raise HTTPException(
            status_code=500,
            detail=str(e) or "批量标注执行失败，请查看后端日志获取详细错误信息",
        )
