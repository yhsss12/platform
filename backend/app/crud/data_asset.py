"""
数据资产 CRUD（data_assets 表，PostgreSQL）
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, text, update
from typing import Any, Dict, List, Optional

from app.models.data_asset import DataAsset
from app.models.label_task_asset import LabelTask
from app.schemas.data_asset import DataAssetCreate, DataAssetQueryParams


def _parse_created_date_utc_start(value: Optional[str]) -> Optional[datetime]:
    """将 YYYY-MM-DD 或 YYYY/MM/DD 解析为当日 00:00:00 UTC。"""
    if not value or not str(value).strip():
        return None
    raw = str(value).strip()[:10].replace("/", "-")
    try:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _iter_parent_dirs(path: str) -> List[str]:
    p = Path(path)
    out: List[str] = []
    for parent in p.parents:
        s = str(parent)
        if s and s != ".":
            out.append(os.path.normpath(s))
    return out


def _chunks(items: List[str], size: int = 900) -> List[List[str]]:
    if size <= 0:
        size = 900
    return [items[i : i + size] for i in range(0, len(items), size)]


def normalize_storage_path_key(p: str) -> str:
    """路径键比较。minio:// 不得使用 os.path.normpath（会把 // 收成 /）。"""
    s = (p or "").strip()
    if not s:
        return ""
    if s.startswith("minio://"):
        return s
    return os.path.normpath(s)


def _storage_path_parent_dirs(path: str) -> List[str]:
    """目录前缀匹配用父路径：本地沿用 Path.parents；minio:// 按 bucket/key 分段回退。"""
    s = (path or "").strip()
    if not s:
        return []
    if s.startswith("minio://"):
        rest = s[8:]
        if "/" not in rest:
            return []
        bucket, _, key = rest.partition("/")
        key = key.rstrip("/")
        if not key:
            return []
        parts = key.split("/")
        out: List[str] = []
        for i in range(len(parts) - 1, 0, -1):
            prefix = "/".join(parts[:i])
            out.append(f"minio://{bucket}/{prefix}")
            out.append(f"minio://{bucket}/{prefix}/")
        out.append(f"minio://{bucket}/")
        return out
    return _iter_parent_dirs(s)


def _expand_annotation_path_candidates(task_params: Dict[str, Any]) -> List[str]:
    """自动标注任务 params：展开为可与 data_assets.file_path 匹配的候选键（含规范化）。"""
    p = task_params or {}
    wh = (p.get("annotation_warehouse_path") or "").strip()
    fp = (p.get("file_path") or "").strip()
    amu = (p.get("annotation_asset_minio_uri") or "").strip()
    candidates: List[str] = []
    if wh:
        candidates.append(wh)
    if amu and amu not in candidates:
        candidates.append(amu)
    if fp.startswith("minio://") and fp not in candidates:
        candidates.append(fp)
    expanded: List[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        s = str(raw).strip()
        for x in (s, normalize_storage_path_key(s)):
            if x and x not in seen:
                seen.add(x)
                expanded.append(x)
    return expanded


def _pick_data_asset_for_episode_dict(assets: List[DataAsset], episode_info: Dict[str, Any]) -> Optional[DataAsset]:
    """在 label_tasks.dataset_ids 命中的多行中，用 episodes_index 单条 episode 对齐到唯一 data_assets 行。"""
    if not assets or not isinstance(episode_info, dict):
        return None
    fn = (episode_info.get("filename") or "").strip().lower()
    if fn:
        for a in assets:
            afn = (a.filename or "").strip().lower()
            if afn and (afn == fn or afn in fn or fn in afn):
                return a
    candidates: List[str] = []
    wh = (episode_info.get("warehouse_path") or "").strip()
    if wh.startswith("minio://"):
        candidates.append(wh)
    fp = (episode_info.get("file_path") or "").strip()
    if fp.startswith("minio://") and fp not in candidates:
        candidates.append(fp)
    meta_raw = episode_info.get("meta")
    if isinstance(meta_raw, dict):
        meta_json = json.dumps(meta_raw, ensure_ascii=False)
    elif isinstance(meta_raw, str):
        meta_json = meta_raw.strip() or None
    else:
        meta_json = None
    try:
        from app.services.data_asset_path_resolver import minio_uri_from_fields

        uri = minio_uri_from_fields(fp or None, meta_json)
        if uri and uri not in candidates:
            candidates.append(uri)
    except Exception:
        pass
    try:
        from app.services.episode_storage import EpisodeStorage

        sk = EpisodeStorage(episode_info).get_storage_key()
        if sk and sk not in candidates:
            candidates.append(sk)
    except Exception:
        pass
    expanded: List[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        s = str(raw).strip()
        for x in (s, normalize_storage_path_key(s)):
            if x and x not in seen:
                seen.add(x)
                expanded.append(x)
    for a in assets:
        apk = normalize_storage_path_key((a.file_path or "").strip())
        for x in expanded:
            if apk and apk == normalize_storage_path_key(x):
                return a
    if len(assets) == 1:
        return assets[0]
    return None


async def find_data_asset_for_label_episode(
    db: AsyncSession,
    *,
    label_task_id: str,
    episode_id: str,
) -> Optional[DataAsset]:
    """按标注任务 dataset_ids + episodes_index 定位 data_assets（路径候选失败时的回退）。"""
    ltid = (label_task_id or "").strip()
    epid = (episode_id or "").strip()
    if not ltid or not epid:
        return None
    r = await db.execute(select(LabelTask).where(LabelTask.task_id == ltid))
    lt = r.scalar_one_or_none()
    if lt is None:
        return None
    from app.services.task_config_service import TaskConfigService

    svc = TaskConfigService(base_dir=os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data"))
    episode_info = svc.find_episode_by_id(ltid, epid)
    if not episode_info:
        return None
    raw = (lt.dataset_ids or "").strip()
    if not raw:
        return None
    try:
        ids = json.loads(raw)
    except Exception:
        return None
    if not isinstance(ids, list) or not ids:
        return None
    str_ids = [str(x).strip() for x in ids if str(x).strip()]
    if not str_ids:
        return None
    is_ds = any(s.startswith("DS") for s in str_ids)
    if is_ds:
        res = await db.execute(select(DataAsset).where(DataAsset.dataset_id.in_(str_ids)))
    else:
        try:
            int_ids = [int(x) for x in str_ids]
        except (ValueError, TypeError):
            return None
        res = await db.execute(select(DataAsset).where(DataAsset.id.in_(int_ids)))
    assets = list(res.scalars().all())
    return _pick_data_asset_for_episode_dict(assets, episode_info)


async def persist_annotation_instruction_to_data_asset(
    db: AsyncSession,
    task_params: Dict[str, Any],
    instruction_text: str,
) -> bool:
    """
    将自动标注结果写入 data_assets.instruction_text。
    供 GET /annotation/status 在任务成功时补写（不依赖 worker 进程/任务目录是否可见）。
    """
    import logging

    logger = logging.getLogger(__name__)
    text = (instruction_text or "").strip()
    if not text:
        return False
    p = task_params or {}
    expanded = _expand_annotation_path_candidates(p)
    asset: Optional[DataAsset] = None
    if expanded:
        asset = await find_asset_by_instruction_path_candidates(db, expanded)
    if asset is None:
        ltid = (p.get("annotation_label_task_id") or "").strip()
        epid = (p.get("annotation_episode_id") or "").strip()
        if ltid and epid:
            try:
                asset = await find_data_asset_for_label_episode(db, label_task_id=ltid, episode_id=epid)
            except Exception:
                logger.debug("persist_annotation_instruction: dataset_ids 回退匹配失败", exc_info=True)
    if asset is None:
        logger.warning(
            "persist_annotation_instruction: 未找到 data_assets 行 params_keys=%s expanded=%s",
            list(p.keys())[:20],
            (expanded or [])[:6],
        )
        return False
    await update_asset(db, asset.id, instruction_text=text)
    return True


def sync_find_data_asset_for_label_episode_session(session, *, label_task_id: str, episode_id: str) -> Optional[DataAsset]:
    """sync 会话内：dataset_ids + episodes_index 定位资产（供 worker 使用）。"""
    from app.services.task_config_service import TaskConfigService

    ltid = (label_task_id or "").strip()
    epid = (episode_id or "").strip()
    if not ltid or not epid:
        return None
    lt = session.scalars(select(LabelTask).where(LabelTask.task_id == ltid)).first()
    if lt is None:
        return None
    svc = TaskConfigService(base_dir=os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data"))
    episode_info = svc.find_episode_by_id(ltid, epid)
    if not episode_info:
        return None
    raw = (lt.dataset_ids or "").strip()
    if not raw:
        return None
    try:
        ids = json.loads(raw)
    except Exception:
        return None
    if not isinstance(ids, list) or not ids:
        return None
    str_ids = [str(x).strip() for x in ids if str(x).strip()]
    if not str_ids:
        return None
    is_ds = any(s.startswith("DS") for s in str_ids)
    if is_ds:
        res = session.execute(select(DataAsset).where(DataAsset.dataset_id.in_(str_ids)))
        assets = list(res.scalars().all())
    else:
        try:
            int_ids = [int(x) for x in str_ids]
        except (ValueError, TypeError):
            return None
        res = session.execute(select(DataAsset).where(DataAsset.id.in_(int_ids)))
        assets = list(res.scalars().all())
    return _pick_data_asset_for_episode_dict(assets, episode_info)


async def next_code(db: AsyncSession) -> str:
    """按导入顺序生成展示编号 0001, 0002, ..."""
    r = await db.execute(select(func.count()).select_from(DataAsset))
    n = r.scalar() or 0
    return str(n + 1).zfill(4)


async def next_dataset_id(db: AsyncSession) -> str:
    """生成数据专属唯一标识 DS000001, DS000002, ...，标注/转换/导出均以此为存在依据"""
    r = await db.execute(
        select(func.max(DataAsset.id)).select_from(DataAsset)
    )
    max_id = r.scalar() or 0
    return f"DS{(max_id + 1):06d}"


async def get_assets(
    db: AsyncSession,
    params: DataAssetQueryParams,
    allowed_project_ids: Optional[List[str]] = None,
) -> tuple[List[DataAsset], int]:
    """列表（筛选 + 分页）"""
    query = select(DataAsset)
    count_query = select(func.count()).select_from(DataAsset)
    conditions = []

    if allowed_project_ids is not None:
        ids = [str(x).strip() for x in allowed_project_ids if str(x).strip()]
        if not ids:
            return [], 0
        conditions.append(DataAsset.project_id.in_(ids))

    if params.keyword and params.keyword.strip():
        conditions.append(DataAsset.filename.like(f"%{params.keyword.strip()}%"))
    if params.project and params.project.strip():
        pv = params.project.strip()
        conditions.append((DataAsset.project_id == pv) | (DataAsset.project_name == pv))
    if params.format and params.format.strip():
        conditions.append(DataAsset.format == params.format.strip().lower())
    if params.source and params.source.strip():
        src = params.source.strip()
        # 兼容历史：import / 本地 / local 统一视为“导入”
        if src == "import":
            conditions.append(DataAsset.source.in_(["import", "本地", "local"]))
        else:
            conditions.append(DataAsset.source == src)

    # 按任务名称过滤（推荐）：直接按 data_assets 上的任务名列过滤
    if params.task_name and params.task_name.strip():
        tn = params.task_name.strip()
        src0 = (params.source or "").strip().lower()
        if src0 == "label":
            conditions.append(DataAsset.label_task_name == tn)
        elif src0 == "collect":
            conditions.append(DataAsset.collect_task_name == tn)
        elif src0 == "convert":
            conditions.append(DataAsset.conversion_task_name == tn)
        elif not src0 and ":" in tn:
            kind, name = tn.split(":", 1)
            name = name.strip()
            if kind == "label" and name:
                conditions.append(DataAsset.label_task_name == name)
            elif kind == "collect" and name:
                conditions.append(DataAsset.collect_task_name == name)
            elif kind == "convert" and name:
                conditions.append(DataAsset.conversion_task_name == name)
            else:
                conditions.append(
                    (DataAsset.label_task_name == tn)
                    | (DataAsset.collect_task_name == tn)
                    | (DataAsset.conversion_task_name == tn)
                )
        else:
            # 未限定来源：匹配任一任务名（兼容旧版纯任务名筛选）
            conditions.append(
                (DataAsset.label_task_name == tn)
                | (DataAsset.collect_task_name == tn)
                | (DataAsset.conversion_task_name == tn)
            )

    # 按任务缩小范围：标注任务用 dataset_ids，采集任务用 meta.collect.task_id
    if params.task_id and params.task_id.strip():
        tid = params.task_id.strip()
        if params.source and params.source.strip() == "collect":
            # 采集：meta JSON 中 collect.task_id 等于当前任务 ID
            conditions.append(text("meta::json->'collect'->>'task_id' = :tid").bindparams(tid=tid))
        else:
            # 标注（或未指定 source）：从 label_tasks 取 dataset_ids，过滤 data_assets.dataset_id
            label_task = (await db.execute(select(LabelTask).where(LabelTask.task_id == tid))).scalar_one_or_none()
            if label_task and getattr(label_task, "dataset_ids", None):
                try:
                    ids = json.loads(label_task.dataset_ids or "[]")
                    if isinstance(ids, list) and ids:
                        id_list = [str(x) for x in ids]
                        is_ds_format = any(s.startswith("DS") for s in id_list)
                        if is_ds_format:
                            conditions.append(DataAsset.dataset_id.in_(id_list))
                        else:
                            try:
                                int_ids = [int(x) for x in id_list]
                                conditions.append(DataAsset.id.in_(int_ids))
                            except (ValueError, TypeError):
                                conditions.append(DataAsset.id == -1)
                    else:
                        conditions.append(DataAsset.id == -1)  # 无有效 dataset_ids，不返回
                except Exception:
                    conditions.append(DataAsset.id == -1)
            else:
                conditions.append(DataAsset.id == -1)

    created_from_dt = _parse_created_date_utc_start(getattr(params, "created_from", None))
    created_to_dt = _parse_created_date_utc_start(getattr(params, "created_to", None))
    if created_from_dt is not None:
        conditions.append(DataAsset.created_at >= created_from_dt)
    if created_to_dt is not None:
        # 含结束日全天： [created_from, created_to 24:00) 左闭右开
        end_exclusive = created_to_dt + timedelta(days=1)
        conditions.append(DataAsset.created_at < end_exclusive)

    if conditions:
        where = and_(*conditions)
        query = query.where(where)
        count_query = count_query.where(where)

    total_r = await db.execute(count_query)
    total = total_r.scalar() or 0

    # 统一按时间戳语义排序：最近业务更新时间优先，其次创建时间、主键
    query = query.order_by(DataAsset.updated_at.desc(), DataAsset.created_at.desc(), DataAsset.id.desc())
    skip = (params.page - 1) * params.page_size
    query = query.offset(skip).limit(params.page_size)
    result = await db.execute(query)
    items = list(result.scalars().all())
    return items, total


async def get_asset_by_dataset_id(db: AsyncSession, dataset_id: str) -> Optional[DataAsset]:
    """按数据专属唯一标识查询"""
    result = await db.execute(select(DataAsset).where(DataAsset.dataset_id == dataset_id))
    return result.scalar_one_or_none()


async def get_asset_by_id(db: AsyncSession, asset_id: int) -> Optional[DataAsset]:
    result = await db.execute(select(DataAsset).where(DataAsset.id == asset_id))
    return result.scalar_one_or_none()


async def get_assets_by_ids(db: AsyncSession, asset_ids: List[int]) -> List[DataAsset]:
    """按 ID 列表查询资产，保持与 asset_ids 一致的顺序（按 code 排序）。"""
    if not asset_ids:
        return []
    result = await db.execute(
        select(DataAsset).where(DataAsset.id.in_(asset_ids)).order_by(DataAsset.code.asc())
    )
    return list(result.scalars().all())


async def get_asset_by_file_path(db: AsyncSession, file_path: str) -> Optional[DataAsset]:
    result = await db.execute(select(DataAsset).where(DataAsset.file_path == file_path))
    return result.scalar_one_or_none()


async def find_asset_by_instruction_path_candidates(
    db: AsyncSession,
    candidates: List[str],
) -> Optional[DataAsset]:
    """
    按多种可能的路径键查找资产（原始 minio://、本地解析路径、规范化键），用于批量标注回写 instruction_text。
    data_assets.file_path 可能与 episodes_index 中的写法略有差异，单靠一次精确匹配容易失败。
    """
    expanded: List[str] = []
    seen: set = set()
    for raw in candidates:
        if not raw or not str(raw).strip():
            continue
        s = str(raw).strip()
        for x in (s, normalize_storage_path_key(s)):
            if x and x not in seen:
                seen.add(x)
                expanded.append(x)
    if not expanded:
        return None
    for chunk in _chunks(expanded, 900):
        result = await db.execute(select(DataAsset).where(DataAsset.file_path.in_(chunk)))
        rows = list(result.scalars().all())
        if rows:
            return rows[0]
    return None


async def are_dataset_ids_valid(db: AsyncSession, dataset_ids_json: Optional[str]) -> bool:
    """校验 dataset_ids 是否仍有效（按 data_assets.dataset_id 查询，对应记录未被删除则有效）。
    label_tasks.dataset_ids 存储的是 data_assets.dataset_id 列表（如 [\"DS000001\"]）。"""
    if not dataset_ids_json:
        return True  # 无 dataset_ids 视为有效（如手动路径任务）
    try:
        ids = json.loads(dataset_ids_json)
    except (json.JSONDecodeError, TypeError):
        return False
    if not ids:
        return True
    # 支持 dataset_id 字符串（DS000001）或旧格式整数 id
    id_list = [str(x) for x in ids]
    is_ds_format = any(s.startswith("DS") for s in id_list)
    if is_ds_format:
        result = await db.execute(select(func.count()).select_from(DataAsset).where(DataAsset.dataset_id.in_(id_list)))
    else:
        try:
            int_ids = [int(x) for x in id_list]
            result = await db.execute(select(func.count()).select_from(DataAsset).where(DataAsset.id.in_(int_ids)))
        except (ValueError, TypeError):
            return False
    cnt = result.scalar() or 0
    return cnt == len(ids)


async def get_file_paths_in_assets(db: AsyncSession, paths: List[str]) -> set:
    """批量查询哪些路径在数据资产表中存在，返回存在的路径集合（用于标注任务校验）。
    使用 idx_data_assets_file_path 索引加速查询。"""
    if not paths:
        return set()
    expanded: set = set()
    norm_inputs: List[str] = []
    for raw in paths:
        if not raw:
            continue
        raw_s = raw.strip()
        nk = normalize_storage_path_key(raw_s)
        norm_inputs.append(nk)
        expanded.add(raw_s)
        expanded.add(nk)
        for parent in _storage_path_parent_dirs(raw_s):
            expanded.add(parent)
            expanded.add(normalize_storage_path_key(parent))

    if not expanded:
        return set()

    expanded_list = list(expanded)
    matched: set = set()
    for chunk in _chunks(expanded_list, 900):
        r = await db.execute(select(DataAsset.file_path).where(DataAsset.file_path.in_(chunk)))
        rows = r.scalars().all()
        for fp in rows:
            if not fp:
                continue
            fps = fp.strip()
            matched.add(fps)
            matched.add(normalize_storage_path_key(fps))
            for parent in _storage_path_parent_dirs(fps):
                matched.add(parent)
                matched.add(normalize_storage_path_key(parent))

    valid: set = set()
    for p in norm_inputs:
        if p in matched:
            valid.add(p)
            continue
        for parent in _storage_path_parent_dirs(p):
            if parent in matched or normalize_storage_path_key(parent) in matched:
                valid.add(p)
                break
    return valid


async def get_instruction_text_by_paths(db: AsyncSession, paths: List[str]) -> dict:
    """批量按 file_path 查询 instruction_text，返回 规范化路径 -> instruction_text（无则不在字典中或为空）。"""
    if not paths:
        return {}
    expanded: set = set()
    norm_inputs: List[str] = []
    for raw in paths:
        if not raw:
            continue
        raw_s = raw.strip()
        p0 = normalize_storage_path_key(raw_s)
        norm_inputs.append(p0)
        expanded.add(raw_s)
        expanded.add(p0)
        for parent in _storage_path_parent_dirs(raw_s):
            expanded.add(parent)
            expanded.add(normalize_storage_path_key(parent))

    if not expanded:
        return {}

    expanded_list = list(expanded)
    path_to_text: dict = {}
    for chunk in _chunks(expanded_list, 900):
        r = await db.execute(
            select(DataAsset.file_path, DataAsset.instruction_text).where(DataAsset.file_path.in_(chunk))
        )
        for file_path, instruction_text in r.all():
            if not file_path:
                continue
            val = instruction_text if instruction_text is not None else ""
            path_to_text[normalize_storage_path_key(file_path.strip())] = val

    out: dict = {}
    for p in norm_inputs:
        if p in path_to_text:
            out[p] = path_to_text.get(p, "") or ""
            continue
        chosen = None
        for parent in _storage_path_parent_dirs(p):
            pk = normalize_storage_path_key(parent)
            if pk in path_to_text:
                chosen = pk
                break
        if chosen is not None:
            out[p] = path_to_text.get(chosen, "") or ""
    return out


async def create_asset(db: AsyncSession, data: DataAssetCreate) -> DataAsset:
    payload = data.model_dump()
    # 业务时间由数据库统一生成，禁止请求方/调用方注入
    payload.pop("updated_at", None)
    obj = DataAsset(**payload)
    obj.dataset_id = await next_dataset_id(db)
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_asset(
    db: AsyncSession,
    asset_id: int,
    parse_status: Optional[str] = None,
    error_msg: Optional[str] = None,
    meta: Optional[str] = None,
    instruction_text: Optional[str] = None,
    label_task_name: Optional[str] = None,
    collect_task_name: Optional[str] = None,
    conversion_task_name: Optional[str] = None,
    sync_status: Optional[str] = None,
    sync_error: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Optional[DataAsset]:
    asset = await get_asset_by_id(db, asset_id)
    if not asset:
        return None
    if parse_status is not None:
        asset.parse_status = parse_status
    if error_msg is not None:
        # 空字符串表示「无错误」并落库为 NULL，与新建/导入成功（error_msg 未改写）一致
        asset.error_msg = None if error_msg == "" else error_msg
    if meta is not None:
        asset.meta = meta
    if instruction_text is not None:
        asset.instruction_text = instruction_text
    if label_task_name is not None:
        asset.label_task_name = label_task_name
    if collect_task_name is not None:
        asset.collect_task_name = collect_task_name
    if conversion_task_name is not None:
        asset.conversion_task_name = conversion_task_name
    if sync_status is not None:
        asset.sync_status = sync_status
    if sync_error is not None:
        asset.sync_error = sync_error
    if device_id is not None:
        asset.device_id = device_id if str(device_id).strip() else None
    await db.commit()
    await db.refresh(asset)
    return asset


async def try_mark_asset_syncing(db: AsyncSession, asset_id: int) -> bool:
    """
    以幂等/并发安全方式将资产标记为 syncing：
    - 仅允许从 unsynced/failed/空值 进入 syncing
    - 若已 syncing/synced，则不变更并返回 False
    """
    stmt = (
        update(DataAsset)
        .where(
            DataAsset.id == asset_id,
            func.lower(func.coalesce(DataAsset.sync_status, "")) != "syncing",
            func.lower(func.coalesce(DataAsset.sync_status, "")) != "synced",
        )
        .values(
            sync_status="syncing",
            sync_error=None,
            updated_at=func.now(),
        )
    )
    res = await db.execute(stmt)
    await db.commit()
    return bool(getattr(res, "rowcount", 0) or 0)


async def delete_asset(db: AsyncSession, asset_id: int) -> bool:
    asset = await get_asset_by_id(db, asset_id)
    if not asset:
        return False
    await db.delete(asset)
    await db.commit()
    return True


def sync_persist_instruction_text_after_annotation(task_params: Dict[str, Any], instruction_text: str) -> bool:
    """
    供 worker / 同步线程使用：单条自动标注成功后，将描述写回 data_assets.instruction_text，
    与 save_instruction / batch_by_task 一致，供 GET episodes 左侧「已标注」与 instruction_text 展示。
    """
    import logging

    from sqlalchemy.orm import Session
    from app.services.asset_registration_service import DataAssetsSyncSessionLocal

    logger = logging.getLogger(__name__)
    p = task_params or {}
    expanded = _expand_annotation_path_candidates(p)

    text = (instruction_text or "").strip()
    if not text:
        logger.warning("sync_persist_instruction_text_after_annotation: 空文本，跳过")
        return False

    session: Session = DataAssetsSyncSessionLocal()
    try:
        asset: Optional[DataAsset] = None
        if expanded:
            for chunk in _chunks(expanded, 900):
                res = session.execute(select(DataAsset).where(DataAsset.file_path.in_(chunk)))
                rows = list(res.scalars().all())
                if rows:
                    asset = rows[0]
                    break
        if asset is None:
            ltid = (p.get("annotation_label_task_id") or "").strip()
            epid = (p.get("annotation_episode_id") or "").strip()
            if ltid and epid:
                try:
                    asset = sync_find_data_asset_for_label_episode_session(session, label_task_id=ltid, episode_id=epid)
                except Exception:
                    logger.debug("sync_persist_instruction_text_after_annotation: dataset_ids 回退失败", exc_info=True)
        if asset is None:
            logger.warning(
                "sync_persist_instruction_text_after_annotation: 未匹配到 data_assets expanded=%s lt=%s ep=%s",
                expanded[:8],
                (p.get("annotation_label_task_id") or "")[:32],
                (p.get("annotation_episode_id") or "")[:48],
            )
            return False
        asset.instruction_text = text
        session.commit()
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
