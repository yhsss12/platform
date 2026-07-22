"""
采集端 Agent HTTP 根地址解析（与数据同步、WebRTC 策略一致）。

后端重启后内存 AgentRegistry 会清空，但 devices 表中的 agent_ip/agent_port 仍在，
故所有需转发到采集端 Agent 的 HTTP 调用应优先用本模块，而非仅查 agent_registry。
"""
from __future__ import annotations

import logging
from typing import Optional

from app.crud.device import get_device_by_id
from app.db.session import AsyncSessionLocal
from app.services.agent_registry import agent_registry

logger = logging.getLogger(__name__)

def _is_local_only_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return True
    if h in ("0.0.0.0", "localhost", "127.0.0.1"):
        return True
    if h.startswith("127."):
        return True
    return False


async def resolve_agent_http_base_url(
    *,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    """
    解析采集端 ``http://host:port`` 根地址。

    顺序：
    1. ``agent_id`` → AgentRegistry（register/heartbeat）
    2. ``device_id`` → Registry 内 device→agent 映射
    3. ``device_id`` → PostgreSQL ``devices`` 表 ``agent_ip`` / ``agent_port``（设备连接成功后落库）
    4. 任一在线且带 base_url 的非 ``local-agent``（兜底）
    """
    aid = (agent_id or "").strip() or None
    if aid:
        info = agent_registry.get_by_id(aid)
        if info and info.base_url and not _is_local_only_host(getattr(info, "host", "") or ""):
            return str(info.base_url).rstrip("/")

    if device_id is not None:
        try:
            did = int(device_id)
        except (TypeError, ValueError):
            did = None
        if did is not None:
            info = agent_registry.get_by_device_id(did)
            if info and info.base_url and not _is_local_only_host(getattr(info, "host", "") or ""):
                return str(info.base_url).rstrip("/")
            async with AsyncSessionLocal() as db:
                dev = await get_device_by_id(db, did)
            if dev and dev.agent_ip and dev.agent_port and not _is_local_only_host(str(dev.agent_ip)):
                u = f"http://{str(dev.agent_ip).strip()}:{int(dev.agent_port)}"
                logger.debug("agent_http: 使用 devices 表地址 device_id=%s url=%s", did, u)
                return u

    for a in agent_registry.list_agents():
        if a.agent_id != "local-agent" and a.online and a.base_url and not _is_local_only_host(getattr(a, "host", "") or ""):
            return str(a.base_url).rstrip("/")

    return None
