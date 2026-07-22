import asyncio
import os
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import select, text

from app.crud.device import get_device_by_id
from app.crud.job import get_job
from app.db.data_assets_session import get_data_assets_db
from app.db.session import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.deps import get_current_user, get_current_user_ws
from app.core.roles import is_super_admin
from app.crud.project import is_project_visible_to_user
from app.models.data_asset import CollectionTaskAsset, CollectionJobAsset
from app.models.project_asset import Project
from app.models.user import User
from app.services.script_runner import script_runner
from app.services.agent_registry import agent_registry
from app.services.agent_tunnel_manager import agent_tunnel_manager
from app.services.agent_collect_proxy import (
    start_collect_via_agent,
    stop_collect_via_agent,
    get_last_collect_context,
)
from app.services.collect_disk_reconcile import find_latest_episode_dir_for_incomplete_cleanup
from app.services.collect_storage_layout import (
    CollectDeletePathError,
    normalize_collect_delete_path,
    resolve_collect_job_workspace_path,
)
from app.services.experiment_logger import log_experiment_event

router = APIRouter()
logger = logging.getLogger(__name__)


def _episode_container_dir_for_validation(raw_path: str) -> str:
    """
    若 path 指向具体 bag 文件（如 *.mcap），应对齐到其所在 episode 目录；
    不能用 os.path.join(file.mcap, ...) 当作目录访问。
    """
    p = (raw_path or "").strip()
    if not p:
        return p
    def _pick_single_bag_file_in_dir(d: str) -> Optional[str]:
        try:
            if not os.path.isdir(d):
                return None
            hits: list[str] = []
            for name in os.listdir(d):
                low = name.lower()
                if low.endswith(".mcap") or low.endswith(".mca") or low.endswith(".db3"):
                    full = os.path.join(d, name)
                    if os.path.isfile(full):
                        hits.append(full)
            if len(hits) == 1:
                return hits[0]
        except OSError:
            return None
        return None

    def _pick_latest_episode_dir(d: str) -> Optional[str]:
        """当传入的是“上层目录”时，尝试选择最近修改的子目录（且子目录内有 bag 文件）。"""
        try:
            if not os.path.isdir(d):
                return None
            candidates: list[tuple[float, str]] = []
            for name in os.listdir(d):
                full = os.path.join(d, name)
                if not os.path.isdir(full):
                    continue
                if _pick_single_bag_file_in_dir(full) is not None:
                    try:
                        ts = float(os.path.getmtime(full))
                    except OSError:
                        ts = 0.0
                    candidates.append((ts, full))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]
        except OSError:
            return None
        return None

    try:
        if os.path.exists(p):
            if os.path.isdir(p):
                # 若目录内仅有 1 个 bag 文件（mcap/db3），直接用该目录（episode）
                single = _pick_single_bag_file_in_dir(p)
                if single is not None:
                    return p
                # 若是“上层目录”，尝试选最近的 episode 子目录
                latest = _pick_latest_episode_dir(p)
                if latest is not None:
                    return latest
                return p
            return os.path.dirname(p) or p
    except OSError:
        pass
    base = os.path.basename(p)
    ext = os.path.splitext(base)[1].lower()
    if ext in (".mcap", ".mca", ".db3", ".bag"):
        parent = os.path.dirname(p)
        return parent if parent else p
    return p


def _normalize_episode_dir_in_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """若 episode_dir 为目录且内含单个 mcap/db3，补全为文件路径供前端识别类型。"""
    ep_dir = report_data.get("episode_dir")
    if ep_dir and isinstance(ep_dir, str) and os.path.isdir(ep_dir):
        found_file = None
        try:
            for f_name in os.listdir(ep_dir):
                if f_name.endswith(".mcap"):
                    found_file = os.path.join(ep_dir, f_name)
                    break
            if not found_file:
                for f_name in os.listdir(ep_dir):
                    if f_name.endswith(".db3"):
                        found_file = os.path.join(ep_dir, f_name)
                        break
            if found_file:
                report_data["episode_dir"] = found_file
        except OSError:
            pass
    return report_data


def _local_bag_target_for_validate(candidate: str) -> Optional[str]:
    """
    若 candidate 在本机存在（文件或目录型 bag），返回可传给 validate_bag 的路径；否则 None。
    与是否部署采集端 Agent 无关：用于「数据已在平台存储」场景。
    """
    c = (candidate or "").strip()
    if not c:
        return None
    try:
        if os.path.isfile(c):
            return c
        bag_dir = _episode_container_dir_for_validation(c)
        if os.path.isdir(bag_dir):
            return bag_dir
        if os.path.isfile(bag_dir):
            return bag_dir
    except OSError:
        return None
    return None


async def _validation_path_candidates(
    *,
    raw_path: str,
    job_id: Optional[str],
) -> List[str]:
    """优先 jobs 表中的 mcap_path（常为同步后的平台路径），其次为 URL 中的 path。"""
    out: List[str] = []
    seen: set[str] = set()

    jid = (job_id or "").strip()
    if jid:
        try:
            uid = UUID(jid)
        except ValueError:
            uid = None
        if uid is not None:
            async with AsyncSessionLocal() as db:
                row = await get_job(db, uid)
            mp = getattr(row, "mcap_path", None) if row else None
            if mp:
                s = str(mp).strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)

    rp = (raw_path or "").strip()
    if rp and rp not in seen:
        out.append(rp)
    return out


def _validate_bag_via_subprocess(repo_root: Path, bag_target: str) -> Dict[str, Any]:
    """在本机调用 agent/validate_bag.py，从 stdout 读取 JSON（不落盘）。"""
    script = repo_root / "agent" / "validate_bag.py"
    if not script.is_file():
        raise RuntimeError(
            f"未找到校验脚本：{script}（本地拉取报告依赖仓库 agent/validate_bag.py）"
        )
    proc = subprocess.run(
        [sys.executable, str(script), bag_target, "30", "raw"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(repo_root / "agent"),
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or (proc.stdout or "").strip() or "validate_bag 执行失败"
        raise RuntimeError(err)
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError("validate_bag 无输出")
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"校验报告 JSON 解析失败: {e}") from e


def _collect_execution_command_id(
    *,
    run_id: Optional[str],
    task_id: Optional[str],
    job_id: Optional[str],
) -> str:
    marker = (run_id or job_id or task_id or "default").strip() or "default"
    return f"collect-execution:{marker}"


async def _resolve_agent_id_for_tunnel(
    device_id: Optional[int],
    agent_id: Optional[str],
) -> Optional[str]:
    aid = (agent_id or "").strip() or None
    if aid:
        return aid
    if device_id is not None:
        info = agent_registry.get_by_device_id_strict(int(device_id))
        if info and info.agent_id:
            return info.agent_id
        # 自动回填：按 device.hardware_uuid(agent_id) 绑定 device_id -> agent_id
        async with AsyncSessionLocal() as db:
            dev = await get_device_by_id(db, int(device_id))
        if dev is not None:
            hw = str(getattr(dev, "hardware_uuid", "") or "").strip()
            if hw:
                agent = agent_registry.get_by_id(hw)
                if agent and agent.agent_id:
                    try:
                        agent_registry.bind_device_to_agent(device_id=int(device_id), agent_id=hw)
                    except Exception:
                        pass
                    return agent.agent_id
    return None


class ScriptRequest(BaseModel):
    script_path: str = Field(description="采集脚本路径")
    args: List[str] = Field(default_factory=list, description="脚本参数")
    env: Optional[Dict[str, str]] = Field(default=None, description="额外环境变量")

    # Agent + 平台模式下的额外元信息（可选）
    task_id: Optional[str] = Field(
        default=None, description="任务 ID（前端可传入，便于 Agent 侧记录）"
    )
    job_id: Optional[str] = Field(
        default=None, description="作业 ID（前端可传入，便于 Agent 侧记录）"
    )
    device_id: Optional[int] = Field(
        default=None, description="设备 ID，用于通过设备选择采集端 Agent"
    )
    agent_id: Optional[str] = Field(
        default=None, description="显式指定采集端 Agent ID，优先级高于 device_id"
    )
    run_id: Optional[str] = Field(default=None, description="实验运行 ID")
    scenario_id: Optional[str] = Field(default=None, description="实验场景 ID")


class AgentLogRequest(BaseModel):
    """采集端 Agent 上报日志请求体。"""

    message: str = Field(description="单行日志文本")
    task_id: Optional[str] = Field(
        default=None, description="任务 ID，可选，仅用于后续扩展过滤/聚合"
    )
    job_id: Optional[str] = Field(
        default=None, description="作业 ID，可选，仅用于后续扩展过滤/聚合"
    )
    agent_id: Optional[str] = Field(
        default=None, description="Agent ID，可选"
    )
    run_id: Optional[str] = Field(default=None, description="实验运行 ID")
    scenario_id: Optional[str] = Field(default=None, description="实验场景 ID")


class StopScriptRequest(BaseModel):
    """
    显式停止请求。

    说明：
    - 旧实现完全依赖服务内存中的 last_collect_context；后端重启后会丢失，导致 stop 返回 400，
      前端就会卡在“采集中/加载中”但无法恢复控制。
    - 新实现允许前端带上 job/device 信息，后端可自行复原 project_id 做权限校验，并路由到正确 Agent。
    """

    device_id: Optional[int] = Field(default=None, description="设备 ID（用于解析采集端 Agent）")
    agent_id: Optional[str] = Field(default=None, description="显式 Agent ID（优先级高于 device_id）")
    task_id: Optional[str] = Field(default=None, description="任务 ID（用于项目权限校验）")
    job_id: Optional[str] = Field(default=None, description="作业 ID（用于项目权限校验）")
    run_id: Optional[str] = Field(default=None, description="实验运行 ID（可选）")
    scenario_id: Optional[str] = Field(default=None, description="实验场景 ID（可选）")


@router.post("/start")
async def start_script(
    request: ScriptRequest,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """Start a script execution via Agent abstraction."""
    # 最小收口：根据 task_id/job_id 复原 project_id，并做“项目可见/未归档”校验。
    task_id = (request.task_id or "").strip() or None
    job_id = (request.job_id or "").strip() or None

    project_id: Optional[str] = None
    if task_id:
        t = await db.get(CollectionTaskAsset, str(task_id))
        if not t:
            raise HTTPException(status_code=404, detail="采集任务不存在")
        project_id = (getattr(t, "project_id", None) or "").strip() if t else None
    elif job_id:
        j = await db.get(CollectionJobAsset, str(job_id))
        if not j:
            raise HTTPException(status_code=404, detail="采集作业不存在")
        project_id = (getattr(j, "project_id", None) or "").strip() if j else None

    if not project_id:
        raise HTTPException(status_code=400, detail="缺少 task_id/job_id，无法确定所属项目")

    p = await db.execute(select(Project).where(Project.id == project_id))
    proj = p.scalar_one_or_none()
    if not proj:
        raise HTTPException(status_code=404, detail="项目不存在")
    if (proj.status or "").strip() == "已归档":
        raise HTTPException(status_code=403, detail="项目已归档，禁止该操作")
    if not is_super_admin(getattr(current_user, "role", None)):
        visible = await is_project_visible_to_user(
            db,
            project_id=project_id,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            raise HTTPException(status_code=404, detail="项目不存在")

    ok, msg = await start_collect_via_agent(
        script_path=request.script_path,
        args=request.args,
        env=request.env,
        device_id=request.device_id,
        agent_id=request.agent_id,
        task_id=request.task_id,
        job_id=request.job_id,
        run_id=request.run_id,
        scenario_id=request.scenario_id,
    )
    if not ok:
        raise HTTPException(status_code=502, detail=msg or "启动采集失败")
    log_experiment_event(
        role="platform",
        event="command_sent",
        ts_ms=int(time.time() * 1000),
        command_id=_collect_execution_command_id(
            run_id=request.run_id,
            task_id=request.task_id,
            job_id=request.job_id,
        ),
        cmd="COLLECT_EXECUTION",
        task_id=request.task_id,
        job_id=request.job_id,
        run_id=request.run_id,
        scenario_id=request.scenario_id,
        device_id=request.device_id,
        agent_id=request.agent_id,
    )
    log_experiment_event(
        role="platform",
        event="task_state_transition",
        ts_ms=int(time.time() * 1000),
        task_id=request.task_id,
        job_id=request.job_id,
        run_id=request.run_id,
        scenario_id=request.scenario_id,
        device_id=request.device_id,
        agent_id=request.agent_id,
        state="RUNNING",
    )
    return {"success": True}

@router.post("/stop")
async def stop_script(
    payload: StopScriptRequest | None = None,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """Stop the currently running script (send SIGINT)."""
    # 兼容：后端重启后内存上下文会丢失；允许使用显式 payload 兜底恢复。
    device_id, agent_id, task_id, job_id, run_id, scenario_id, _ = get_last_collect_context()
    if payload is not None:
        if payload.device_id is not None:
            device_id = payload.device_id
        if (payload.agent_id or "").strip():
            agent_id = (payload.agent_id or "").strip()
        if (payload.task_id or "").strip():
            task_id = (payload.task_id or "").strip()
        if (payload.job_id or "").strip():
            job_id = (payload.job_id or "").strip()
        if (payload.run_id or "").strip():
            run_id = (payload.run_id or "").strip()
        if (payload.scenario_id or "").strip():
            scenario_id = (payload.scenario_id or "").strip()

    # 最小收口：STOP 也尝试复原项目并校验，防止跨项目终止他人采集。
    project_id: Optional[str] = None
    if job_id:
        j = await db.get(CollectionJobAsset, str(job_id))
        project_id = (getattr(j, "project_id", None) or "").strip() if j else None
    if not project_id and task_id:
        t = await db.get(CollectionTaskAsset, str(task_id))
        project_id = (getattr(t, "project_id", None) or "").strip() if t else None

    if project_id:
        p = await db.execute(select(Project).where(Project.id == project_id))
        proj = p.scalar_one_or_none()
        if not proj:
            raise HTTPException(status_code=404, detail="项目不存在")
        if (proj.status or "").strip() == "已归档":
            raise HTTPException(status_code=403, detail="项目已归档，禁止该操作")
        if not is_super_admin(getattr(current_user, "role", None)):
            visible = await is_project_visible_to_user(
                db,
                project_id=project_id,
                user_id=str(getattr(current_user, "id", "") or ""),
                include_owner_projects=True,
            )
            if not visible:
                raise HTTPException(status_code=404, detail="项目不存在")
    else:
        # 没有上下文且也没有显式 job/task 信息则无法复原项目：拒绝，避免越权 stop。
        raise HTTPException(status_code=400, detail="无法确定当前采集所属项目（请提供 job_id/task_id），拒绝停止")

    success = await stop_collect_via_agent(
        device_id=device_id,
        agent_id=agent_id,
        task_id=task_id,
        job_id=job_id,
        run_id=run_id,
        scenario_id=scenario_id,
    )
    log_experiment_event(
        role="platform",
        event="task_state_transition",
        ts_ms=int(time.time() * 1000),
        task_id=task_id,
        job_id=job_id,
        run_id=run_id,
        scenario_id=scenario_id,
        device_id=device_id,
        agent_id=agent_id,
        state="STOPPED" if success else "STOP_FAILED",
        success=success,
    )
    return {"success": success}

@router.post("/reset")
async def reset_script(
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """Reset the script state and clear logs."""
    device_id, agent_id, task_id, job_id, run_id, scenario_id, _ = get_last_collect_context()
    project_id: Optional[str] = None
    if job_id:
        j = await db.get(CollectionJobAsset, str(job_id))
        project_id = (getattr(j, "project_id", None) or "").strip() if j else None
    if not project_id and task_id:
        t = await db.get(CollectionTaskAsset, str(task_id))
        project_id = (getattr(t, "project_id", None) or "").strip() if t else None
    if project_id:
        p = await db.execute(select(Project).where(Project.id == project_id))
        proj = p.scalar_one_or_none()
        if not proj:
            raise HTTPException(status_code=404, detail="项目不存在")
        if (proj.status or "").strip() == "已归档":
            raise HTTPException(status_code=403, detail="项目已归档，禁止该操作")
        if not is_super_admin(getattr(current_user, "role", None)):
            visible = await is_project_visible_to_user(
                db,
                project_id=project_id,
                user_id=str(getattr(current_user, "id", "") or ""),
                include_owner_projects=True,
            )
            if not visible:
                raise HTTPException(status_code=404, detail="项目不存在")
    else:
        raise HTTPException(status_code=400, detail="无法确定当前采集所属项目，拒绝重置")

    script_runner.clear_history()
    log_experiment_event(
        role="platform",
        event="task_state_transition",
        ts_ms=int(time.time() * 1000),
        task_id=task_id,
        job_id=job_id,
        run_id=run_id,
        scenario_id=scenario_id,
        device_id=device_id,
        agent_id=agent_id,
        state="RESET",
    )
    return {"success": True}


@router.post("/agent-log")
async def ingest_agent_log(
    payload: AgentLogRequest,
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    采集端 Agent 上报一行日志，转发到现有 WebSocket 管道。

    - 前端仍通过 `/api/script/ws` 接收日志，无需改动。
    - 当前仅广播 `message`，task/job/agent 元信息预留给后续多任务隔离。
    """
    # 直接复用 script_runner 的广播能力
    await script_runner.broadcast(payload.message)
    clean = str(payload.message or "").strip()

    # 将关键产物落库：跨标签/刷新稳定读取（质检页不再依赖 URL path 或 sessionStorage）
    # - OUTPUT_PATH: <path>            -> jobs.mcap_path
    # - OUTPUT_SIZE_BYTES: <int>       -> jobs.mcap_size_bytes
    # - EAI_VALIDATION_REPORT_JSON:... -> jobs.validation_report_json
    # 注意：该接口由采集端 Agent 调用，不走用户鉴权；权限在“采集开始”时已校验项目可见性与归档状态。
    async def _ensure_job_validation_report_column() -> None:
        try:
            from app.services.asset_registration_service import data_assets_sync_engine

            with data_assets_sync_engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE collection_jobs ADD COLUMN IF NOT EXISTS validation_report_json TEXT"
                    )
                )
        except Exception:
            pass

    try:
        await _ensure_job_validation_report_column()
    except Exception:
        pass

    job_id = (payload.job_id or "").strip()
    if clean.startswith("EAI_VALIDATION_REPORT_JSON:") and not job_id:
        logger.warning(
            "agent_log: got validation report but missing job_id; drop persist. agent_id=%s device_id=%s msg_prefix=%s",
            (payload.agent_id or "").strip(),
            payload.device_id,
            clean[:80],
        )
    if job_id:
        try:
            j = await db.get(CollectionJobAsset, job_id)
        except Exception:
            j = None
        if j is None and clean.startswith("EAI_VALIDATION_REPORT_JSON:"):
            logger.warning(
                "agent_log: got validation report but job not found; job_id=%s agent_id=%s device_id=%s",
                job_id,
                (payload.agent_id or "").strip(),
                payload.device_id,
            )

        if j is not None:
            updated = False
            if clean.startswith("OUTPUT_PATH:"):
                p = clean[len("OUTPUT_PATH:") :].strip()
                if p:
                    try:
                        setattr(j, "mcap_path", p)
                        updated = True
                    except Exception:
                        pass
            elif clean.startswith("OUTPUT_SIZE_BYTES:"):
                raw = clean[len("OUTPUT_SIZE_BYTES:") :].strip()
                try:
                    sz = int(raw)
                    setattr(j, "mcap_size_bytes", sz if sz >= 0 else 0)
                    updated = True
                except Exception:
                    pass
            elif clean.startswith("EAI_VALIDATION_REPORT_JSON:"):
                js = clean[len("EAI_VALIDATION_REPORT_JSON:") :].strip()
                if js:
                    try:
                        json.loads(js)
                        setattr(j, "validation_report_json", js)
                        updated = True
                    except Exception:
                        # ignore malformed report
                        logger.warning(
                            "agent_log: validation_report_json invalid json; job_id=%s agent_id=%s device_id=%s len=%s",
                            job_id,
                            (payload.agent_id or "").strip(),
                            payload.device_id,
                            len(js),
                        )
                        pass
            if updated:
                try:
                    await db.commit()
                    await db.refresh(j)
                    if clean.startswith("EAI_VALIDATION_REPORT_JSON:"):
                        logger.info(
                            "agent_log: persisted validation_report_json ok; job_id=%s agent_id=%s device_id=%s",
                            job_id,
                            (payload.agent_id or "").strip(),
                            payload.device_id,
                        )
                except Exception:
                    try:
                        await db.rollback()
                    except Exception:
                        pass
                    logger.exception(
                        "agent_log: commit failed while persisting outputs; job_id=%s agent_id=%s device_id=%s",
                        job_id,
                        (payload.agent_id or "").strip(),
                        payload.device_id,
                    )

    if clean.startswith("Script finished with exit code"):
        success = "exit code 0" in clean
        log_experiment_event(
            role="platform",
            event="result_received",
            ts_ms=int(time.time() * 1000),
            command_id=_collect_execution_command_id(
                run_id=payload.run_id,
                task_id=payload.task_id,
                job_id=payload.job_id,
            ),
            cmd="COLLECT_EXECUTION",
            success=success,
            task_id=payload.task_id,
            job_id=payload.job_id,
            run_id=payload.run_id,
            scenario_id=payload.scenario_id,
            agent_id=payload.agent_id,
            message=clean,
        )
        log_experiment_event(
            role="platform",
            event="task_state_transition",
            ts_ms=int(time.time() * 1000),
            task_id=payload.task_id,
            job_id=payload.job_id,
            run_id=payload.run_id,
            scenario_id=payload.scenario_id,
            agent_id=payload.agent_id,
            state="DONE" if success else "FAILED",
            success=success,
        )
    return {"ok": True}

@router.delete("/data")
async def delete_data(
    path: str = Query("", description="bag 路径（可为采集端路径；若为空且提供 job_id，则自动使用 jobs.mcap_path）"),
    cleanup_incomplete: bool = Query(
        False,
        description="为 true 时：无 OUTPUT_PATH 场景下，按 since_ms+workspace 在采集端查找并删除最新半成品 episode 目录",
    ),
    since_ms: Optional[int] = Query(
        None, description="本轮采集开始时间（Unix 毫秒），与 cleanup_incomplete 联用"
    ),
    workspace: str = Query(
        "",
        description="与 /api/script/start 的 -o 一致的本作业输出根目录；cleanup_incomplete 时必填且须与库中任务配置一致",
    ),
    device_id: Optional[int] = Query(None, description="设备 ID（用于转发到采集端 Agent）"),
    agent_id: Optional[str] = Query(None, description="Agent ID（优先级高于 device_id）"),
    job_id: Optional[str] = Query(None, description="作业 UUID：用于在 path 缺失时从 jobs.mcap_path 推断删除目标"),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """Delete a file or directory (local or via Agent)."""
    jid = (job_id or "").strip()

    if cleanup_incomplete:
        if not jid:
            raise HTTPException(status_code=400, detail="cleanup_incomplete 需要 job_id")
        if since_ms is None:
            raise HTTPException(status_code=400, detail="cleanup_incomplete 需要 since_ms")
        ws_hint = (workspace or "").strip()
        if not ws_hint:
            raise HTTPException(status_code=400, detail="cleanup_incomplete 需要 workspace")
        try:
            db_job_ci = await db.get(CollectionJobAsset, str(jid))
        except Exception:
            db_job_ci = None
        if db_job_ci is None:
            raise HTTPException(status_code=404, detail="作业不存在")
        tid_ci = str(getattr(db_job_ci, "task_id", "") or "").strip()
        task_row_ci = await db.get(CollectionTaskAsset, tid_ci) if tid_ci else None
        desc_ci = getattr(task_row_ci, "description", None) if task_row_ci else None
        jn_ci = int(getattr(db_job_ci, "job_number", 0) or 0)
        expected_ws = resolve_collect_job_workspace_path(desc_ci, jn_ci)
        n1 = os.path.normpath(ws_hint.replace("\\", "/"))
        n2 = os.path.normpath(str(expected_ws).replace("\\", "/"))
        if n1 != n2:
            raise HTTPException(
                status_code=400,
                detail="workspace 与作业约定输出目录不一致，拒绝清理",
            )
        resolved_aid_ci = await _resolve_agent_id_for_tunnel(device_id, agent_id)
        if resolved_aid_ci is None:
            raise HTTPException(
                status_code=400,
                detail="清理未完成采集需要 device_id（或 agent_id）以连接采集端隧道",
            )
        if not await agent_tunnel_manager.has_connection(resolved_aid_ci):
            raise HTTPException(status_code=503, detail="采集端隧道未连接")
        episode_abs, ferr = await find_latest_episode_dir_for_incomplete_cleanup(
            resolved_aid_ci, expected_ws, int(since_ms)
        )
        if ferr:
            raise HTTPException(status_code=502, detail=ferr)
        if not episode_abs:
            logger.info(
                "delete_data cleanup_incomplete: no episode dir since_ms=%s workspace=%s job_id=%s",
                since_ms,
                expected_ws,
                jid,
            )
            return {"success": True, "message": "没有匹配时间条件的 episode 目录，跳过删除"}
        target_ci = _episode_container_dir_for_validation(episode_abs)
        logger.info(
            "delete_data cleanup_incomplete: deleting episode_abs=%s target=%s job_id=%s",
            episode_abs,
            target_ci,
            jid,
        )
        try:
            result = await agent_tunnel_manager.send_cmd_and_wait(
                agent_id=resolved_aid_ci,
                cmd="SCRIPT_DELETE_DATA",
                payload={
                    "path": target_ci,
                    "allow_incomplete_episode": True,
                    "workspace_root": expected_ws,
                },
                timeout_sec=30.0,
                retry_times=1,
            )
            if not bool(result.get("success", False)):
                raise HTTPException(
                    status_code=502,
                    detail=result.get("msg") or result.get("message") or "删除失败",
                )
            return {
                "success": True,
                "message": result.get("msg") or result.get("message") or "ok",
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"通过隧道删除失败: {str(e)}")

    db_job: Optional[CollectionJobAsset] = None
    mp_stored = ""

    # 兼容：前端在“重新采集”时可能拿不到 OUTPUT_PATH（刷新/重连/日志缺失），导致 path 为空。
    # 此时允许通过 job_id 反查 jobs.mcap_path 作为删除目标。
    if not (path or "").strip():
        if not jid:
            raise HTTPException(status_code=400, detail="Path is required")
        try:
            db_job = await db.get(CollectionJobAsset, str(jid))
        except Exception:
            db_job = None
        mp_stored = (getattr(db_job, "mcap_path", None) or "").strip() if db_job else ""
        if not mp_stored:
            raise HTTPException(status_code=404, detail="未找到可删除的数据路径（jobs.mcap_path 为空或作业不存在）")
        path = mp_stored
    elif jid:
        try:
            db_job = await db.get(CollectionJobAsset, str(jid))
        except Exception:
            db_job = None
        if db_job is not None:
            mp_stored = (getattr(db_job, "mcap_path", None) or "").strip()

    # 归一化到「本作业 workspace」下的绝对路径，防止 /0002/... 等假绝对路径删失败；白名单放行 jobs.mcap_path（如同步到平台的路径）
    if db_job is not None:
        tid = str(getattr(db_job, "task_id", "") or "").strip()
        task_row = await db.get(CollectionTaskAsset, tid) if tid else None
        desc = getattr(task_row, "description", None) if task_row else None
        jn = int(getattr(db_job, "job_number", 0) or 0)
        ws = resolve_collect_job_workspace_path(desc, jn)
        allow_list = [mp_stored] if mp_stored else []
        try:
            path = normalize_collect_delete_path(
                path,
                workspace=ws,
                job_number=jn,
                allowed_exact_paths=allow_list,
            )
        except CollectDeletePathError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 防误删：前端有时会把“上层目录”当作 path 传入；删除动作只允许针对 bag 文件或 episode 目录
    def _looks_like_bag_file(p: str) -> bool:
        low = (p or "").strip().lower()
        # 兼容：现场脚本可能输出 .mca（实际为 mcap 的变体/误拼后缀）
        return low.endswith(".mcap") or low.endswith(".mca") or low.endswith(".db3") or low.endswith(".bag")

    def _dir_has_bag_files(d: str) -> bool:
        try:
            if not os.path.isdir(d):
                return False
            for name in os.listdir(d):
                low = name.lower()
                if low.endswith(".mcap") or low.endswith(".mca") or low.endswith(".db3"):
                    full = os.path.join(d, name)
                    if os.path.isfile(full):
                        return True
            # 兼容：采集脚本可能只生成 validation_report.json（用于质检页展示）
            if os.path.isfile(os.path.join(d, "validation_report.json")):
                return True
        except OSError:
            return False
        return False

    raw_path = (path or "").strip()
    raw_is_bag_file = _looks_like_bag_file(raw_path)

    target = _episode_container_dir_for_validation(raw_path)
    # 若传入的是具体 bag 文件：无论该路径是否在平台本机存在，都按“episode 目录”删除（更符合“重新采集”语义）
    # 关键：采集数据通常在采集端机器上，平台本机 os.path.exists/isdir 可能为 False。
    if raw_is_bag_file:
        try:
            target = os.path.dirname(raw_path) or target
        except OSError:
            pass

    # 仅允许删除：bag 文件 或 episode 目录；拒绝删除任意普通目录（避免把脚本目录、工作目录整棵删掉）
    if os.path.isdir(target):
        if not _dir_has_bag_files(target):
            raise HTTPException(
                status_code=400,
                detail=f"拒绝删除非采集数据目录（未检测到 bag 文件/validation_report.json）：{target}",
            )
    else:
        # 若 raw_path 是 bag 文件，则 target 可能是 episode 目录（无扩展名且在平台本机不存在），仍应允许走隧道删除。
        if not raw_is_bag_file and not _looks_like_bag_file(target):
            raise HTTPException(
                status_code=400,
                detail=f"拒绝删除非 bag 文件：{target}",
            )

    resolved_aid = await _resolve_agent_id_for_tunnel(device_id, agent_id)
    if resolved_aid is not None:
        if not await agent_tunnel_manager.has_connection(resolved_aid):
            raise HTTPException(status_code=503, detail="采集端隧道未连接")
        try:
            result = await agent_tunnel_manager.send_cmd_and_wait(
                agent_id=resolved_aid,
                cmd="SCRIPT_DELETE_DATA",
                payload={"path": target},
                timeout_sec=30.0,
                retry_times=1,
            )
            if not bool(result.get("success", False)):
                raise HTTPException(
                    status_code=502,
                    detail=result.get("msg") or result.get("message") or "删除失败",
                )
            return {
                "success": True,
                "message": result.get("msg") or result.get("message") or "ok",
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"通过隧道删除失败: {str(e)}")

    if not os.path.exists(target):
        return {"success": False, "message": "Path does not exist"}
    
    try:
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

@router.websocket("/ws")
async def websocket_script_output(
    websocket: WebSocket,
    reset: bool = False,
    task_id: Optional[str] = Query(None),
    job_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user_ws),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """WebSocket for script output streaming."""
    # 优先使用前端传入的 task_id/job_id 做项目校验；若缺失则回退到最近一次 start_script 的上下文。
    project_id: Optional[str] = None
    if job_id:
        j = await db.get(CollectionJobAsset, str(job_id))
        project_id = (getattr(j, "project_id", None) or "").strip() if j else None
    if not project_id and task_id:
        t = await db.get(CollectionTaskAsset, str(task_id))
        project_id = (getattr(t, "project_id", None) or "").strip() if t else None
    if not project_id:
        _, _, last_task_id, last_job_id, _, _, _ = get_last_collect_context()
        if last_job_id:
            j = await db.get(CollectionJobAsset, str(last_job_id))
            project_id = (getattr(j, "project_id", None) or "").strip() if j else None
        if not project_id and last_task_id:
            t = await db.get(CollectionTaskAsset, str(last_task_id))
            project_id = (getattr(t, "project_id", None) or "").strip() if t else None
    if not project_id:
        await websocket.close(code=1008)
        return

    p = await db.execute(select(Project).where(Project.id == project_id))
    proj = p.scalar_one_or_none()
    if not proj:
        await websocket.close(code=1008)
        return
    if (proj.status or "").strip() == "已归档":
        await websocket.close(code=1008)
        return
    if not is_super_admin(getattr(current_user, "role", None)):
        visible = await is_project_visible_to_user(
            db,
            project_id=project_id,
            user_id=str(getattr(current_user, "id", "") or ""),
            include_owner_projects=True,
        )
        if not visible:
            await websocket.close(code=1008)
            return

    if reset:
        script_runner.clear_history()
    await script_runner.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe receive commands if needed
            await websocket.receive_text()
    except WebSocketDisconnect:
        script_runner.disconnect(websocket)

@router.get("/report")
async def get_report(
    path: str = Query("", description="bag 路径（可为采集端路径；若平台本机不存在则再尝试隧道）"),
    device_id: Optional[int] = Query(None, description="设备 ID（用于转发到采集端 Agent）"),
    agent_id: Optional[str] = Query(None, description="Agent ID（优先级高于 device_id）"),
    job_id: Optional[str] = Query(
        None,
        description="作业 UUID：优先使用 jobs.mcap_path（同步后常为平台可访问路径），可不依赖采集端",
    ),
):
    """
    校验报告获取顺序（无需依赖采集端 Agent 的前提：数据已在平台本机可访问路径上）：
    0) 若 job_id 对应 jobs.validation_report_json 已持久化，则直接返回（跨标签/刷新最稳）；
    1) 合并 job.mcap_path 与 path，依次尝试在本机对 bag 执行 validate_bag；
    2) 均失败且配置了 device/agent 且隧道在线时，转发 SCRIPT_GET_REPORT 到采集端；
    3) 否则返回明确错误（提示同步数据或连接隧道）。
    """
    repo_root = Path(__file__).resolve().parents[3]

    # 0) 优先：若作业表已持久化报告 JSON，则直接返回（不依赖 path / ROS / 隧道）
    jid = (job_id or "").strip()
    if jid:
        try:
            uid = UUID(jid)
        except ValueError:
            uid = None
        if uid is not None:
            # 防御：若数据库尚未迁移 validation_report_json 列，直接查询会 500。
            # 这里与 /agent-log 保持一致：请求路径上也做一次 best-effort 迁移。
            try:
                from app.services.asset_registration_service import data_assets_sync_engine

                with data_assets_sync_engine.begin() as conn:
                    conn.execute(
                        text(
                            "ALTER TABLE collection_jobs ADD COLUMN IF NOT EXISTS validation_report_json TEXT"
                        )
                    )
            except Exception:
                pass
            async with AsyncSessionLocal() as db:
                row = await get_job(db, uid)
            js = (getattr(row, "validation_report_json", None) or "").strip() if row else ""
            if js:
                try:
                    data = json.loads(js)
                except Exception:
                    raise HTTPException(status_code=502, detail="作业已保存 validation_report_json，但 JSON 解析失败")
                if isinstance(data, dict):
                    return _normalize_episode_dir_in_report(data)
                raise HTTPException(status_code=502, detail="作业已保存 validation_report_json，但格式不是 JSON 对象")

    candidates = await _validation_path_candidates(raw_path=path, job_id=job_id)
    if not candidates:
        raise HTTPException(status_code=400, detail="请提供 path 或有效的 job_id")

    last_local_error: Optional[str] = None
    for cand in candidates:
        bt = _local_bag_target_for_validate(cand)
        if bt is None:
            continue
        try:
            report_data = await asyncio.to_thread(
                _validate_bag_via_subprocess, repo_root, bt
            )
            return _normalize_episode_dir_in_report(report_data)
        except RuntimeError as e:
            last_local_error = str(e)
            continue

    primary = candidates[0]
    bag_for_tunnel = _episode_container_dir_for_validation(primary)
    resolved_aid = await _resolve_agent_id_for_tunnel(device_id, agent_id)
    if resolved_aid is not None and await agent_tunnel_manager.has_connection(resolved_aid):
        try:
            result = await agent_tunnel_manager.send_cmd_and_wait(
                agent_id=resolved_aid,
                cmd="SCRIPT_GET_REPORT",
                payload={"path": bag_for_tunnel},
                timeout_sec=20.0,
                retry_times=1,
            )
            if not bool(result.get("success", False)):
                raise HTTPException(
                    status_code=404,
                    detail=result.get("msg") or "Validation report not found",
                )
            data = result.get("data")
            if isinstance(data, dict):
                return _normalize_episode_dir_in_report(data)
            raise HTTPException(status_code=502, detail="Agent 返回的报告格式不正确")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"通过隧道获取报告失败: {str(e)}")

    msg_parts = [
        "平台本机无法访问该 bag（路径可能仍在采集端），且采集端隧道不可用或未返回报告。",
        "可选方案：① 将数据同步到平台存储后，确保 jobs.mcap_path 或 path 指向服务器上的文件；",
        "② 保持采集端在线并成功连接隧道。",
    ]
    if last_local_error:
        msg_parts.append(f"（平台侧校验末次错误：{last_local_error}）")
    raise HTTPException(status_code=404, detail=" ".join(msg_parts))
