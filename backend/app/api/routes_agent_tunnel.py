from __future__ import annotations

import logging
from typing import Optional, Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.services.agent_tunnel_manager import agent_tunnel_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tunnel/metrics")
async def agent_tunnel_metrics(
    include_commands: bool = Query(False, description="是否附带最近命令状态"),
    command_limit: int = Query(100, ge=1, le=500, description="最近命令状态条数"),
    agent_id: Optional[str] = Query(None, description="仅查看指定 agent 的命令状态"),
):
    """MJPEG 重组、隧道基础指标与命令状态（运维/验收）。结构化遥测另见 TELEMETRY_FILE_LOG_DIR（默认 logs/telemetry/*.jsonl）。"""
    data = {"metrics": agent_tunnel_manager.get_metrics()}
    if include_commands:
        data["commands"] = agent_tunnel_manager.get_command_states(
            limit=command_limit,
            agent_id=(agent_id or "").strip() or None,
        )
    return data


class TunnelCmdRequest(BaseModel):
    """
    仅供平台本机进程间转发：
    - RQ worker 无法访问 Web 进程内存态隧道连接，因此通过 HTTP 回环请求让 Web 进程代发 CMD。
    """

    agent_id: str = Field(..., description="采集端 agent_id（推荐网卡 MAC，小写冒号）")
    cmd: str = Field(..., description="命令名，如 DATA_SYNC")
    payload: Dict[str, Any] = Field(default_factory=dict, description="命令数据（将作为 CMD_REQUEST.payload.data 下发）")
    timeout_sec: float = Field(1800.0, ge=0.5, le=3600.0, description="等待 CMD_RESULT 超时（秒）")
    retry_times: int = Field(0, ge=0, le=2, description="超时重试次数（仅对同一 command_id 重发）")
    platform_device_id: Optional[int] = Field(
        default=None,
        description="可选：平台 devices.id，用于在 agent_id 不一致时通过 registry 映射到真实隧道键",
    )


@router.post("/tunnel/cmd")
async def agent_tunnel_cmd(
    request: Request,
    body: TunnelCmdRequest = Body(...),
):
    """
    **内部接口（仅允许本机回环访问）**：
    让当前 Web 进程代发隧道命令并等待结果。

    设计背景：
    - WebSocket 隧道连接存于 Web 进程内存（agent_tunnel_manager._connections）
    - RQ worker 进程不共享该内存，因此同步任务会误报“隧道未连接”
    """
    host = (getattr(getattr(request, "client", None), "host", None) or "").strip()
    if host not in {"127.0.0.1", "::1", "localhost"} and not host.startswith("127."):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    agent_id = (body.agent_id or "").strip()
    if not agent_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="agent_id required")
    cmd = (body.cmd or "").strip()
    if not cmd:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="cmd required")

    socket_key = await agent_tunnel_manager.resolve_connected_socket_key(
        agent_id,
        platform_device_id=body.platform_device_id,
    )
    if not socket_key:
        return {"success": False, "msg": f"采集端隧道未连接（agent_id={agent_id}）"}

    payload = body.payload if isinstance(body.payload, dict) else {}
    result = await agent_tunnel_manager.send_cmd_and_wait(
        agent_id=socket_key,
        cmd=cmd,
        payload=payload,
        timeout_sec=float(body.timeout_sec or 1800.0),
        retry_times=int(body.retry_times or 0),
    )
    return result


@router.websocket("/tunnel")
async def agent_tunnel(
    websocket: WebSocket,
    agent_id: str = Query(..., description="Agent identifier; must match AGENT_ID"),
    token: Optional[str] = Query(None, description="与平台 AGENT_TUNNEL_TOKEN 一致的共享密钥"),
):
    """
    Agent -> Platform tunnel.

    Phase-1: Control (CMD_REQUEST/CMD_ACK/CMD_RESULT) + Log (LOG).
    Phase-2 (later): MJPEG_CHUNK (binary) + platform re-assembly.
    """
    if not agent_id:
        await websocket.close(code=4401, reason="Missing agent_id")
        return

    if not await agent_tunnel_manager.accept(
        agent_id=agent_id,
        websocket=websocket,
        tunnel_token=token,
    ):
        return

    # 最佳努力：若采集端通过隧道连入，且 registry 里 host 为空/回环，则用连接来源纠正一次，
    # 便于后续将“采集端 HTTP 地址”解析为真实可达的内网 IP。
    try:
        from app.services.agent_registry import agent_registry

        peer_host = None
        try:
            peer_host = websocket.client.host if websocket.client else None
        except Exception:
            peer_host = None
        peer_host = (peer_host or "").strip() or None
        if peer_host and peer_host not in {"127.0.0.1", "localhost"} and not peer_host.startswith("127."):
            info = agent_registry.get_by_id(agent_id)
            if info:
                cur = str(getattr(info, "host", "") or "").strip()
                if not cur or cur in {"127.0.0.1", "localhost"} or cur.startswith("127."):
                    info.host = peer_host
    except Exception:
        pass

    disconnect_reason: Optional[str] = None
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                code = msg.get("code", "")
                r = msg.get("reason") or ""
                disconnect_reason = f"code={code}" + (f"; reason={r}" if str(r).strip() else "")
                break
            if msg.get("type") != "websocket.receive":
                continue
            text = msg.get("text")
            if text:
                await agent_tunnel_manager.handle_text_message(agent_id=agent_id, message_text=text)
                continue

            # Phase-2: MJPEG_CHUNK (binary)
            if msg.get("bytes"):
                try:
                    await agent_tunnel_manager.handle_mjpeg_chunk_binary(
                        agent_id=agent_id,
                        message_bytes=msg.get("bytes") or b"",
                    )
                except Exception:
                    # Never break tunnel on a single malformed chunk.
                    continue
    except WebSocketDisconnect as e:
        disconnect_reason = f"code={getattr(e, 'code', '')}; reason={getattr(e, 'reason', '')}"
    except Exception as exc:  # pragma: no cover
        disconnect_reason = f"exception={exc!r}"
        logger.warning("agent_tunnel: agent_id=%s err=%s", agent_id, exc)
    finally:
        try:
            await agent_tunnel_manager.disconnect(
                agent_id=agent_id,
                websocket=websocket,
                reason=disconnect_reason or "normal_exit",
            )
        except Exception:
            pass
