from __future__ import annotations

import asyncio
import logging
import os
import shutil
from time import time as _time
import time
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

from app.crud.device import get_device_by_id
from app.db.session import AsyncSessionLocal
from .experiment_logger import log_experiment_event
from .agent_registry import agent_registry, AgentInfo
from .agent_tunnel_manager import agent_tunnel_manager
from .script_runner import script_runner


_last_collect_lock = asyncio.Lock()
_last_collect_device_id: Optional[int] = None
_last_collect_agent_id: Optional[str] = None
_last_collect_task_id: Optional[str] = None
_last_collect_job_id: Optional[str] = None
_last_collect_run_id: Optional[str] = None
_last_collect_scenario_id: Optional[str] = None
_last_collect_execution_command_id: Optional[str] = None
_last_collect_ts: float = 0.0


async def set_last_collect_context(
    *,
    device_id: Optional[int],
    agent_id: Optional[str],
    task_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
    execution_command_id: Optional[str] = None,
) -> None:
    global _last_collect_device_id, _last_collect_agent_id, _last_collect_task_id, _last_collect_job_id
    global _last_collect_run_id, _last_collect_scenario_id, _last_collect_execution_command_id, _last_collect_ts
    async with _last_collect_lock:
        _last_collect_device_id = device_id
        _last_collect_agent_id = agent_id
        _last_collect_task_id = (task_id or "").strip() or None
        _last_collect_job_id = (job_id or "").strip() or None
        _last_collect_run_id = (run_id or "").strip() or None
        _last_collect_scenario_id = (scenario_id or "").strip() or None
        _last_collect_execution_command_id = (execution_command_id or "").strip() or None
        _last_collect_ts = _time()


def get_last_collect_context() -> Tuple[
    Optional[int],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    # Best-effort in-memory state for STOP routing.
    # Intentionally no lock to keep this getter lightweight.
    return (
        _last_collect_device_id,
        _last_collect_agent_id,
        _last_collect_task_id,
        _last_collect_job_id,
        _last_collect_run_id,
        _last_collect_scenario_id,
        _last_collect_execution_command_id,
    )


def _execution_command_id(
    *,
    run_id: Optional[str],
    task_id: Optional[str],
    job_id: Optional[str],
) -> str:
    marker = (run_id or job_id or task_id or "default").strip() or "default"
    return f"collect-execution:{marker}"


def _resolve_agent_id_for_env(
    *,
    device_id: Optional[int],
    agent_id: Optional[str],
    agent: Optional[AgentInfo],
) -> Optional[str]:
    if agent and agent.agent_id:
        return agent.agent_id
    aid = (agent_id or "").strip() or None
    if aid:
        return aid
    if device_id is not None:
        info = agent_registry.get_by_device_id(device_id)
        if info and info.agent_id:
            return info.agent_id
    la = agent_registry.get_by_id("local-agent")
    return la.agent_id if la else "local-agent"


async def _resolve_agent_for_device_autobind(device_id: Optional[int]) -> Optional[AgentInfo]:
    """
    strict 映射缺失时自动回填：
    - device.hardware_uuid 约定为 agent_id
    - 若该 agent 在线，则写入 device_id -> agent_id 映射
    """
    if device_id is None:
        return None
    info = agent_registry.get_by_device_id_strict(int(device_id))
    if info is not None:
        return info
    async with AsyncSessionLocal() as db:
        dev = await get_device_by_id(db, int(device_id))
    if dev is None:
        return None
    hw = str(getattr(dev, "hardware_uuid", "") or "").strip()
    if not hw:
        return None
    agent = agent_registry.get_by_id(hw)
    if agent is None:
        return None
    try:
        agent_registry.bind_device_to_agent(device_id=int(device_id), agent_id=hw)
    except Exception:
        pass
    return agent


_collect_start_mutex_by_key: Dict[str, asyncio.Lock] = {}
_collect_start_mutex_registry_lock = asyncio.Lock()


async def _collect_start_mutex_for(
    resolved_agent_id: str,
    args: List[str],
    job_id: Optional[str],
) -> asyncio.Lock:
    """
    同一采集端 + 同一输出根目录（脚本 -o）下串行执行启动。
    采集脚本多在本地用 find 计算下一个 episode 序号；并发 COLLECT_START 会得到相同序号。
    说明：锁仅在单进程内生效；多 uvicorn worker 时需前置单 worker 或分布式锁。
    """
    raw_out = _extract_output_path(args)
    try:
        out_k = os.path.normpath(str(raw_out).strip()) if raw_out else ""
    except Exception:
        out_k = str(raw_out or "").strip()
    jid = (job_id or "").strip()
    if out_k:
        key = f"{resolved_agent_id}\x1eo:{out_k}"
    elif jid:
        key = f"{resolved_agent_id}\x1ej:{jid}"
    else:
        key = f"{resolved_agent_id}\x1esingleton"
    async with _collect_start_mutex_registry_lock:
        lk = _collect_start_mutex_by_key.get(key)
        if lk is None:
            lk = asyncio.Lock()
            _collect_start_mutex_by_key[key] = lk
        return lk


async def start_collect_via_agent(
    *,
    script_path: str,
    args: List[str],
    env: Optional[Dict[str, str]] = None,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    通过 Agent 启动采集。

    统一网络模型：
    - 非 local-agent：强制通过隧道 CMD_REQUEST 执行 COLLECT_START
    - local-agent：本地 script_runner 兜底
    """

    agent: Optional[AgentInfo] = None
    if agent_id:
        agent = agent_registry.get_by_id(agent_id)
    elif device_id is not None:
        agent = await _resolve_agent_for_device_autobind(device_id)
    else:
        agent = agent_registry.get_by_id("local-agent")

    resolved_aid = _resolve_agent_id_for_env(
        device_id=device_id, agent_id=agent_id, agent=agent
    ) or "local-agent"

    if resolved_aid != "local-agent":
        if not await agent_tunnel_manager.has_connection(resolved_aid):
            return False, "采集端隧道未连接"

    mtx = await _collect_start_mutex_for(resolved_aid, args, job_id)
    async with mtx:
        if resolved_aid != "local-agent":
            command_id = uuid4().hex
            payload: Dict[str, object] = {
                "script_path": script_path,
                "args": args,
                "task_id": task_id,
                "job_id": job_id,
                "run_id": run_id,
                "scenario_id": scenario_id,
                "duration_sec": int(args[args.index("-t") + 1]) if "-t" in args else 30,
                "storage_path": _extract_output_path(args) or "",
                "camera_data_format": (env or {}).get("CAMERA_DATA_FORMAT"),
                "env": env or {},
            }
            try:
                data = await agent_tunnel_manager.send_cmd_and_wait(
                    agent_id=resolved_aid,
                    cmd="COLLECT_START",
                    payload=payload,
                    timeout_sec=25.0,
                    command_id=command_id,
                    retry_times=1,
                )
                ok = bool(data.get("success", False))
                msg = data.get("msg")
                await set_last_collect_context(
                    device_id=device_id,
                    agent_id=resolved_aid,
                    task_id=task_id,
                    job_id=job_id,
                    run_id=run_id,
                    scenario_id=scenario_id,
                    execution_command_id=_execution_command_id(run_id=run_id, task_id=task_id, job_id=job_id),
                )
                return ok, str(msg).strip() if isinstance(msg, str) and msg.strip() else None
            except Exception as e:
                return False, str(e)

        # local-agent：本地执行
        exec_command_id = _execution_command_id(run_id=run_id, task_id=task_id, job_id=job_id)
        await set_last_collect_context(
            device_id=None,
            agent_id="local-agent",
            task_id=task_id,
            job_id=job_id,
            run_id=run_id,
            scenario_id=scenario_id,
            execution_command_id=exec_command_id,
        )
        merged_env: Dict[str, str] = {}
        if env:
            merged_env.update(env)

        if task_id:
            merged_env.setdefault("EAI_TASK_ID", str(task_id))
        if job_id:
            merged_env.setdefault("EAI_JOB_ID", str(job_id))
        aid = resolved_aid
        if aid:
            merged_env.setdefault("EAI_AGENT_ID", aid)
        if run_id:
            merged_env.setdefault("EAI_RUN_ID", str(run_id))
        if scenario_id:
            merged_env.setdefault("EAI_SCENARIO_ID", str(scenario_id))

        command_id = uuid4().hex
        sent_ts_ms = int(time.time() * 1000)
        log_experiment_event(
            role="platform",
            event="command_sent",
            ts_ms=sent_ts_ms,
            command_id=command_id,
            cmd="COLLECT_START",
            task_id=task_id,
            job_id=job_id,
            run_id=run_id,
            scenario_id=scenario_id,
            device_id=device_id,
            agent_id="local-agent",
        )
        ok = await script_runner.start_script(script_path, args, merged_env or None)
        now_ms = int(time.time() * 1000)
        log_experiment_event(
            role="platform",
            event="ack_received",
            ts_ms=now_ms,
            command_id=command_id,
            cmd="COLLECT_START",
            success=ok,
            task_id=task_id,
            job_id=job_id,
            run_id=run_id,
            scenario_id=scenario_id,
            device_id=device_id,
            agent_id="local-agent",
        )
        log_experiment_event(
            role="platform",
            event="result_received",
            ts_ms=now_ms,
            command_id=command_id,
            cmd="COLLECT_START",
            success=ok,
            task_id=task_id,
            job_id=job_id,
            run_id=run_id,
            scenario_id=scenario_id,
            device_id=device_id,
            agent_id="local-agent",
        )
        return ok, None if ok else "脚本启动失败"


async def stop_collect_via_agent(
    *,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> bool:
    """
    通过 Agent 停止采集。

    统一网络模型：
    - 非 local-agent：强制通过隧道 CMD_REQUEST 执行 COLLECT_STOP
    - local-agent：本地 script_runner 停止
    """

    agent: Optional[AgentInfo] = None
    if agent_id:
        agent = agent_registry.get_by_id(agent_id)
    elif device_id is not None:
        agent = await _resolve_agent_for_device_autobind(device_id)
    else:
        agent = agent_registry.get_by_id("local-agent")

    resolved_aid = _resolve_agent_id_for_env(
        device_id=device_id,
        agent_id=agent_id,
        agent=agent,
    ) or "local-agent"

    if resolved_aid != "local-agent":
        if not await agent_tunnel_manager.has_connection(resolved_aid):
            return False
        try:
            command_id = uuid4().hex
            result_payload = await agent_tunnel_manager.send_cmd_and_wait(
                agent_id=resolved_aid,
                cmd="COLLECT_STOP",
                payload={
                    "task_id": task_id,
                    "job_id": job_id,
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                },
                timeout_sec=20.0,
                command_id=command_id,
                retry_times=1,
            )
            return bool(result_payload.get("success", False))
        except Exception:
            return False

    # local-agent：停止本地脚本
    command_id = uuid4().hex
    sent_ts_ms = int(time.time() * 1000)
    log_experiment_event(
        role="platform",
        event="command_sent",
        ts_ms=sent_ts_ms,
        command_id=command_id,
        cmd="COLLECT_STOP",
        task_id=task_id,
        job_id=job_id,
        run_id=run_id,
        scenario_id=scenario_id,
        device_id=device_id,
        agent_id="local-agent",
    )
    ok = await script_runner.stop_script()
    now_ms = int(time.time() * 1000)
    log_experiment_event(
        role="platform",
        event="ack_received",
        ts_ms=now_ms,
        command_id=command_id,
        cmd="COLLECT_STOP",
        success=ok,
        task_id=task_id,
        job_id=job_id,
        run_id=run_id,
        scenario_id=scenario_id,
        device_id=device_id,
        agent_id="local-agent",
    )
    log_experiment_event(
        role="platform",
        event="result_received",
        ts_ms=now_ms,
        command_id=command_id,
        cmd="COLLECT_STOP",
        success=ok,
        task_id=task_id,
        job_id=job_id,
        run_id=run_id,
        scenario_id=scenario_id,
        device_id=device_id,
        agent_id="local-agent",
    )
    return ok


def _delete_collect_workspace_local(path: str) -> tuple[bool, Optional[str]]:
    """本机 local-agent：删除作业约定目录（四位数字文件夹）。"""
    p = (path or "").strip()
    if not p:
        return False, "empty path"
    if not os.path.exists(p):
        return True, None
    try:
        bn = os.path.basename(os.path.normpath(p).rstrip(os.sep))
        if os.path.isdir(p) and len(bn) == 4 and bn.isdigit():
            shutil.rmtree(p)
            return True, None
    except Exception as e:
        return False, str(e)
    return False, "refuse: not a job workspace dir"


async def delete_collect_job_workspace_remote(
    *,
    device_id_str: Optional[str],
    workspace_path: str,
) -> tuple[bool, Optional[str]]:
    """
    删除采集端或本机上的作业输出目录（任务 storagePath + 四位作业子目录）。
    失败时返回 (False, msg)；路径不存在视为成功。
    """
    path = (workspace_path or "").strip().replace("\\", "/")
    if not path:
        return False, "empty workspace path"

    device_id: Optional[int] = None
    if device_id_str and str(device_id_str).strip():
        try:
            device_id = int(str(device_id_str).strip())
        except ValueError:
            device_id = None

    agent = await _resolve_agent_for_device_autobind(device_id) if device_id is not None else None
    resolved_aid = _resolve_agent_id_for_env(device_id=device_id, agent_id=None, agent=agent) or "local-agent"

    if resolved_aid != "local-agent":
        if not await agent_tunnel_manager.has_connection(resolved_aid):
            return False, "采集端隧道未连接"
        try:
            result_payload = await agent_tunnel_manager.send_cmd_and_wait(
                agent_id=resolved_aid,
                cmd="SCRIPT_DELETE_DATA",
                payload={"path": path, "allow_job_workspace": True},
                timeout_sec=90.0,
                retry_times=1,
            )
            ok = bool(result_payload.get("success", False))
            msg = (result_payload.get("msg") or result_payload.get("message") or "") or ""
            if not ok and ("not exist" in msg.lower() or "不存在" in msg):
                return True, None
            return (True, None) if ok else (False, msg or "删除失败")
        except Exception as e:
            logger.warning("delete_collect_job_workspace_remote tunnel error: %s", e)
            return False, str(e)

    ok, err = _delete_collect_workspace_local(path)
    return (ok, err if not ok else None)


def _extract_output_path(args: List[str]) -> Optional[str]:
    """从脚本参数中解析输出路径（-o PATH）"""
    if "-o" in args:
        idx = args.index("-o")
        if idx + 1 < len(args):
            return args[idx + 1]
    return None
