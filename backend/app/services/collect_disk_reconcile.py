"""
采集作业目录与采集端磁盘 episode 目录对账（通过 Agent 隧道 FS_LIST）。
用于：作业列表进度与采集端实际文件夹数量一致；数据资产列表标注是否在盘上。
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes_fs import _resolve_agent_id_for_tunnel
from app.models.data_asset import CollectionJobAsset, CollectionTaskAsset, DataAsset
from app.services.agent_tunnel_manager import agent_tunnel_manager
from app.services.collect_progress import apply_progress_guard
from app.services.collect_storage_layout import resolve_collect_job_workspace_path
from app.services.collection_job_reconcile import (
    derive_job_status_after_reconcile,
    extract_collect_job_id_from_asset,
)

logger = logging.getLogger(__name__)

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EPISODE_DIR_TS_RE = re.compile(r"^episode_\d+_(\d{8})_(\d{6})$", re.IGNORECASE)


def _episode_dir_name_mtime_ts(name: str) -> float:
    """从 episode_<n>_YYYYMMDD_HHMMSS 目录名解析 UTC 时间戳（旧 Agent 无时区 mtime 时的兜底）。"""
    m = _EPISODE_DIR_TS_RE.match(str(name or "").strip())
    if not m:
        return 0.0
    try:
        from datetime import datetime, timezone

        dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return 0.0


def _fs_list_item_mtime_ts(mtime_val: object) -> float:
    """FS_LIST 返回的 mtime 为 UTC ISO8601（建议带 Z）；无法解析时返回 0。"""
    if mtime_val is None:
        return 0.0
    s = str(mtime_val).strip()
    if not s:
        return 0.0
    try:
        from datetime import datetime

        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return float(datetime.fromisoformat(s).timestamp())
    except Exception:
        return 0.0


async def find_latest_episode_dir_for_incomplete_cleanup(
    agent_id: str,
    workspace_abs: str,
    since_ms: int,
    *,
    slack_sec: float = 45.0,
) -> tuple[Optional[str], Optional[str]]:
    """
    在采集端作业 workspace 下，查找「自 since_ms（Unix 毫秒）起本机时间之后」修改过的 episode 目录，
    取 mtime 最新的一条绝对路径。用于脚本尚未打印 OUTPUT_PATH 时停止采集并清理半成品。

    返回 (episode_abs_path_or_None, error_message_or_None)；无候选时 (None, None) 表示无需删除。
    """
    ws = os.path.normpath(str(workspace_abs or "").strip().replace("\\", "/"))
    if not ws or ws == ".":
        return None, "workspace empty"
    try:
        since_sec = max(0.0, float(since_ms) / 1000.0 - slack_sec)
    except Exception:
        since_sec = 0.0

    items, err = await agent_fs_list_items(agent_id, ws)
    if err:
        return None, err

    candidates: list[tuple[float, str]] = []

    for it in items:
        if it.get("type") != "dir":
            continue
        name = str(it.get("name") or "").strip()
        if not name:
            continue
        ts = _fs_list_item_mtime_ts(it.get("mtime"))
        if name.lower().startswith("episode_"):
            if ts <= 0:
                ts = _episode_dir_name_mtime_ts(name)
            if ts >= since_sec:
                candidates.append((ts, f"{ws.rstrip('/')}/{name}"))
            continue
        if _DATE_DIR_RE.match(name):
            subpath = f"{ws.rstrip('/')}/{name}"
            subitems, err2 = await agent_fs_list_items(agent_id, subpath)
            if err2:
                logger.debug("incomplete_cleanup: skip date dir %s: %s", subpath, err2)
                continue
            for sit in subitems:
                if sit.get("type") != "dir":
                    continue
                sn = str(sit.get("name") or "").strip()
                if not sn.lower().startswith("episode_"):
                    continue
                ts2 = _fs_list_item_mtime_ts(sit.get("mtime"))
                if ts2 <= 0:
                    ts2 = _episode_dir_name_mtime_ts(sn)
                if ts2 >= since_sec:
                    candidates.append((ts2, f"{subpath}/{sn}"))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], None


def _unwrap_fs_list_payload(result: dict) -> Tuple[List[dict], Optional[str]]:
    """解析隧道 CMD_RESULT 中带回的 FS_LIST 目录列表。"""
    if not bool(result.get("success", False)):
        return [], str(result.get("msg") or result.get("message") or "FS_LIST failed")
    raw = result.get("data")
    if isinstance(raw, dict):
        inner = raw.get("data") if raw.get("ok") else None
        if isinstance(inner, dict) and isinstance(inner.get("items"), list):
            return inner["items"], None
        items = raw.get("items")
        if isinstance(items, list):
            return items, None
    return [], "FS_LIST 返回格式不正确"


async def agent_fs_list_items(agent_id: str, path: str) -> Tuple[List[dict], Optional[str]]:
    path = (path or "").strip().replace("\\", "/")
    if not path:
        return [], "path empty"
    try:
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=agent_id,
            cmd="FS_LIST",
            payload={"path": path},
            timeout_sec=25.0,
            retry_times=0,
        )
        return _unwrap_fs_list_payload(result if isinstance(result, dict) else {})
    except Exception as e:
        logger.warning("agent_fs_list_items failed path=%s err=%s", path, e)
        return [], str(e)[:200]


async def scan_collect_workspace_episodes(
    agent_id: str,
    workspace_abs: str,
) -> Tuple[List[str], Optional[str]]:
    """
    列出作业 workspace 下的 episode 目录相对路径（posix）。
    支持两种布局：
    - workspace/date_dir/episode_* /
    - workspace/episode_* /
    """
    ws = os.path.normpath((workspace_abs or "").strip().replace("\\", "/"))
    items, err = await agent_fs_list_items(agent_id, ws)
    if err:
        return [], err
    out: List[str] = []
    for it in items:
        if it.get("type") != "dir":
            continue
        name = str(it.get("name") or "").strip()
        if not name:
            continue
        if name.startswith("episode_"):
            out.append(name)
            continue
        if _DATE_DIR_RE.match(name):
            subpath = f"{ws.rstrip('/')}/{name}"
            subitems, err2 = await agent_fs_list_items(agent_id, subpath)
            if err2:
                logger.debug("skip date dir scan %s: %s", subpath, err2)
                continue
            for sit in subitems:
                if sit.get("type") != "dir":
                    continue
                sn = str(sit.get("name") or "").strip()
                if sn.startswith("episode_"):
                    out.append(f"{name}/{sn}")
    out.sort()
    return out, None


def episode_relpath_under_workspace(workspace_abs: str, file_path: str) -> Optional[str]:
    """从采集端绝对路径解析相对于 workspace 的 episode 段（date/episode 或单层 episode）。"""
    if not file_path or file_path.strip().lower().startswith("minio://"):
        return None
    wa = os.path.normpath(str(workspace_abs).replace("\\", "/"))
    fp = os.path.normpath(str(file_path).replace("\\", "/"))
    if len(fp) < len(wa) + 1:
        return None
    prefix = wa.rstrip("/") + "/"
    if not fp.replace("\\", "/").startswith(prefix.replace("\\", "/")):
        return None
    rel = fp[len(prefix) :].replace("\\", "/")
    parts = [p for p in rel.split("/") if p]
    if len(parts) >= 2 and _DATE_DIR_RE.match(parts[0]) and parts[1].startswith("episode_"):
        return f"{parts[0]}/{parts[1]}"
    if parts and parts[0].startswith("episode_"):
        return parts[0]
    return None


def _meta_hint_local_path(meta_text: Optional[str]) -> Optional[str]:
    """已同步 MinIO 资产有时在 meta 里保留原始采集路径，用于与磁盘 episode 对齐。"""
    if not meta_text or not str(meta_text).strip():
        return None
    try:
        m = json.loads(meta_text)
        if not isinstance(m, dict):
            return None
        for key in ("original_file_path", "local_file_path", "collect_file_path"):
            v = m.get(key)
            if isinstance(v, str) and v.strip() and not v.strip().lower().startswith("minio://"):
                return v.strip()
        st = m.get("storage")
        if isinstance(st, dict):
            for key in ("local_path", "source_path", "path"):
                v = st.get(key)
                if isinstance(v, str) and v.strip() and not v.strip().lower().startswith("minio://"):
                    return v.strip()
        c = m.get("collect")
        if isinstance(c, dict):
            v = c.get("mcap_path") or c.get("output_path")
            if isinstance(v, str) and v.strip() and not v.strip().lower().startswith("minio://"):
                return v.strip()
    except Exception:
        pass
    return None


def collect_asset_episode_relpath(asset: DataAsset, workspace_abs: str) -> Optional[str]:
    fp = getattr(asset, "file_path", None) or ""
    rel = episode_relpath_under_workspace(workspace_abs, str(fp))
    if rel:
        return rel
    hint = _meta_hint_local_path(getattr(asset, "meta", None))
    if hint:
        return episode_relpath_under_workspace(workspace_abs, hint)
    return None


async def reconcile_collection_job_progress_from_agent_disk(
    db: AsyncSession,
    *,
    job_row: CollectionJobAsset,
    task_description_json: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """
    以采集端磁盘 episode 目录数为真源，回写 collection_jobs 的 completed_count / progress / status。
    返回 (updated, error_message)。隧道不可用或路径失败时返回 (False, reason)，不写库。
    """
    jid = str(getattr(job_row, "id", "") or "").strip()
    jn = int(getattr(job_row, "job_number", 0) or 0)
    workspace = resolve_collect_job_workspace_path(task_description_json, jn)
    dev_raw = getattr(job_row, "device_id", None)
    try:
        did = int(str(dev_raw).strip()) if dev_raw is not None and str(dev_raw).strip().isdigit() else None
    except Exception:
        did = None
    agent_id = await _resolve_agent_id_for_tunnel(did, None)
    if not agent_id:
        return False, "no_agent"
    if not await agent_tunnel_manager.has_connection(agent_id, platform_device_id=did):
        return False, "agent_offline"
    disk_eps, err = await scan_collect_workspace_episodes(agent_id, workspace)
    if err:
        return False, err
    disk_count = len(disk_eps)
    cur = int(getattr(job_row, "completed_count", 0) or 0)
    prev_total = int(getattr(job_row, "collection_quantity", 0) or 0)
    existing_percent = int(getattr(job_row, "progress", 0) or 0)
    next_current, next_total, percent, _, _ = apply_progress_guard(
        existing_current=cur,
        existing_total=prev_total,
        existing_percent=existing_percent,
        desired_current=disk_count,
        desired_total=None,
        allow_reset=True,
        protect_total_regression=False,
    )
    old_status = (job_row.status or "").strip().upper()
    job_row.completed_count = next_current
    job_row.collection_quantity = next_total
    job_row.progress = percent
    if old_status not in ("CANCELED", "FAILED"):
        job_row.status = derive_job_status_after_reconcile(
            old_status,
            next_current=next_current,
            next_total=next_total,
            percent=percent,
        )
    await db.commit()
    logger.info(
        "collect_disk_reconcile job_id=%s disk_eps=%s db_completed %s->%s",
        jid,
        disk_count,
        cur,
        next_current,
    )
    return True, None


async def build_collect_disk_presence_map(
    assets: List[DataAsset],
    *,
    db: AsyncSession,
) -> Dict[int, Tuple[Optional[bool], Optional[str]]]:
    """
    对当前页的采集资产，按 job_id 分组各做一次磁盘扫描，返回 asset_id -> (是否在盘, episode 相对路径)。
    无法连接 Agent 或无法解析路径时为 (None, rel_or_None)。
    """
    out: Dict[int, Tuple[Optional[bool], Optional[str]]] = {}
    by_job: Dict[str, List[DataAsset]] = {}
    for a in assets:
        if (getattr(a, "source", None) or "").lower() != "collect":
            continue
        jid = extract_collect_job_id_from_asset(a)
        if not jid:
            continue
        by_job.setdefault(jid, []).append(a)

    scan_cache: Dict[str, Tuple[List[str], str]] = {}

    for job_id, group in by_job.items():
        db_job = await db.get(CollectionJobAsset, job_id)
        if not db_job:
            for a in group:
                out[int(a.id)] = (None, None)
            continue
        tid = str(getattr(db_job, "task_id", "") or "").strip()
        task_row = await db.get(CollectionTaskAsset, tid) if tid else None
        task_desc = getattr(task_row, "description", None) if task_row else None
        ws = resolve_collect_job_workspace_path(task_desc, int(getattr(db_job, "job_number", 0) or 0))
        dev_raw = getattr(db_job, "device_id", None)
        try:
            did = int(str(dev_raw).strip()) if dev_raw is not None and str(dev_raw).strip().isdigit() else None
        except Exception:
            did = None
        agent_id = await _resolve_agent_id_for_tunnel(did, None)
        if not agent_id or not await agent_tunnel_manager.has_connection(agent_id, platform_device_id=did):
            for a in group:
                rel = collect_asset_episode_relpath(a, ws)
                out[int(a.id)] = (None, rel)
            continue
        if job_id not in scan_cache:
            eps, err = await scan_collect_workspace_episodes(agent_id, ws)
            scan_cache[job_id] = (eps, err or "")
        eps, scan_err_msg = scan_cache[job_id]
        if scan_err_msg:
            for a in group:
                rel = collect_asset_episode_relpath(a, ws)
                out[int(a.id)] = (None, rel)
            continue
        disk_set = set(eps)
        for a in group:
            rel = collect_asset_episode_relpath(a, ws)
            if not rel:
                out[int(a.id)] = (None, None)
                continue
            out[int(a.id)] = (rel in disk_set, rel)
    return out
