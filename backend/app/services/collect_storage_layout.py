"""
采集任务/作业的存储目录约定：
- 任务：`{parent}/{sanitize(任务名)}`，parent 来自用户选择的根目录或环境变量默认。
- 作业输出目录由前端在启动脚本时拼接：`{任务 storagePath}/{job_number 四位补零}`。

同步仍依赖采集端 OUTPUT_PATH 落在上述路径下，逻辑不变。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PARENT = "/home/rm/data/rosbags/aloha_recordings"


def default_collect_storage_parent() -> str:
    return (os.getenv("EAI_COLLECT_STORAGE_PARENT") or _DEFAULT_PARENT).strip() or _DEFAULT_PARENT


def sanitize_filesystem_segment(name: str, *, max_len: int = 120) -> str:
    """目录名单段：去掉路径非法字符，避免空串。"""
    s = (name or "").strip()
    if not s:
        return "unnamed_task"
    # Windows / POSIX 非法字符
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)
    s = s.strip(" .")
    if not s:
        return "unnamed_task"
    return s[:max_len]


def _parse_storage_path_from_description(description_json: Optional[str]) -> Optional[str]:
    if not description_json:
        return None
    try:
        cfg = json.loads(description_json)
        if isinstance(cfg, dict):
            v = cfg.get("storagePath")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass
    return None


def resolve_task_storage_full_path(
    task_name: str,
    *,
    user_parent: Optional[str],
    existing_description_json: Optional[str],
    previous_task_name: Optional[str] = None,
) -> str:
    """
    计算任务级 storagePath（持久化到 description JSON）。

    - user_parent 非空：视为用户选择的「父目录」（尚未含任务名子目录）。
    - 否则从 existing_description_json 解析当前 storagePath，若最后一级等于
      sanitize(previous_task_name)，则父目录为其 dirname；否则整段视为父目录。
    """
    seg = sanitize_filesystem_segment(task_name)
    default_parent = default_collect_storage_parent()

    parent: Optional[str] = None
    raw_user = (user_parent or "").strip()
    if raw_user:
        parent = os.path.normpath(raw_user.replace("\\", "/"))
    else:
        cur_full = _parse_storage_path_from_description(existing_description_json)
        if cur_full:
            pn = os.path.normpath(cur_full.replace("\\", "/"))
            bn = os.path.basename(pn.rstrip(os.sep))
            prev_seg = sanitize_filesystem_segment(previous_task_name or "") if previous_task_name else ""
            if prev_seg and bn == prev_seg:
                parent = os.path.dirname(pn) or default_parent
            elif bn == seg:
                return pn
            else:
                parent = pn
        if not parent:
            parent = default_parent

    parent = os.path.normpath(parent or default_parent)
    if os.path.basename(parent.rstrip(os.sep)) == seg:
        return parent
    return os.path.normpath(os.path.join(parent, seg))


def job_dir_segment(job_number: int) -> str:
    """作业目录名：与前端 padStart(4,'0') 一致。"""
    n = int(job_number or 0)
    if n < 0:
        n = 0
    return str(n).zfill(4)


def resolve_collect_job_workspace_path(
    task_description_json: Optional[str],
    job_number: int,
) -> str:
    """
    单次作业在采集盘上的根目录：{任务 storagePath}/{四位作业编号}。
    与前端 collectOutputRootForJob、脚本 -o 约定一致。
    """
    base = _parse_storage_path_from_description(task_description_json)
    if not base:
        base = default_collect_storage_parent()
    base = os.path.normpath(str(base).replace("\\", "/"))
    seg = job_dir_segment(job_number)
    return os.path.normpath(os.path.join(base, seg))


def _posix_norm_path(p: str) -> str:
    s = (p or "").strip().replace("\\", "/")
    if not s:
        return ""
    return str(PurePosixPath(s))


def collect_path_is_under_workspace(workspace: str, path: str) -> bool:
    """路径（POSIX）是否落在给定作业 workspace 之下（含等于）。"""
    try:
        ws = PurePosixPath(_posix_norm_path(workspace))
        pt = PurePosixPath(_posix_norm_path(path))
        if not str(ws):
            return False
        pt.relative_to(ws)
        return True
    except ValueError:
        return False


# 形如 /0002/... 或 /0002 —— 根下误把作业号当成绝对路径前缀（采集端 cwd 非任务根时常见）
_WRONG_ROOT_JOB_PREFIX = re.compile(r"^/(\d{4})(?:/(.*))?$")


class CollectDeletePathError(ValueError):
    """删除路径无法安全归一化或越界。"""


def normalize_collect_delete_path(
    raw_path: str,
    *,
    workspace: str,
    job_number: int,
    allowed_exact_paths: Optional[list[str]] = None,
) -> str:
    """
    将脚本/日志可能输出的「相对 / 假绝对」路径，归一成采集约定的绝对路径，且必须落在本作业 workspace 下。

    策略（偏安全、少猜测）：
    - 已在 workspace 下：原样返回（支持完整绝对路径）。
    - 与 allowed_exact_paths 中任一项规范化后完全一致：原样返回（用于平台同步后的 jobs.mcap_path 等）。
    - 以 /{四位作业号}/ 或 /{四位作业号}$ 开头且四位与当前 job 一致：去掉该假根后拼到 workspace。
    - 不以 / 开头的相对路径：拼到 workspace；若首段重复四位作业号则去掉首段。
    - 其它「绝对路径但不在 workspace 下」且不在白名单：拒绝（防止误删采集端任意绝对路径）。

    含 .. 的片段一律拒绝。
    """
    ws = _posix_norm_path(workspace)
    if not ws or ws == ".":
        raise CollectDeletePathError("invalid job workspace")

    seg = job_dir_segment(job_number)
    pn = _posix_norm_path(raw_path)
    if not pn:
        raise CollectDeletePathError("empty path")

    if ".." in PurePosixPath(pn).parts:
        raise CollectDeletePathError("path contains parent traversal")

    if collect_path_is_under_workspace(ws, pn):
        return pn

    allow_norm = {
        _posix_norm_path(x)
        for x in (allowed_exact_paths or [])
        if isinstance(x, str) and (x or "").strip()
    }
    allow_norm.discard("")
    if pn in allow_norm:
        return pn

    s = (raw_path or "").strip().replace("\\", "/")
    rel_parts: list[str]

    m = _WRONG_ROOT_JOB_PREFIX.match(s)
    if m:
        root_seg = m.group(1)
        if root_seg != seg:
            raise CollectDeletePathError(
                "path leading job segment does not match current job; refuse delete"
            )
        rest = (m.group(2) or "").strip("/")
        rel_parts = [p for p in rest.split("/") if p] if rest else []
    elif not s.startswith("/"):
        rel_parts = [p for p in s.split("/") if p]
    else:
        raise CollectDeletePathError(
            "absolute path is outside this job workspace and not in allowed paths; refuse delete"
        )

    if rel_parts and rel_parts[0] == seg:
        rel_parts = rel_parts[1:]

    if not rel_parts:
        raise CollectDeletePathError("nothing to delete under workspace after normalization")

    if ".." in rel_parts:
        raise CollectDeletePathError("invalid relative path")

    cand = str(PurePosixPath(ws).joinpath(*rel_parts))
    cand = _posix_norm_path(cand)

    if not collect_path_is_under_workspace(ws, cand):
        raise CollectDeletePathError("normalized path escapes job workspace")

    if cand != pn:
        logger.info(
            "collect_delete_path_normalized job_number=%s workspace=%s from=%s to=%s",
            job_number,
            ws,
            pn,
            cand,
        )
    return cand
