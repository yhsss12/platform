from fastapi import APIRouter, Request

from app.schemas.agent import (
    AgentRegisterRequest,
    AgentHeartbeatRequest,
    AgentResponse,
)
from app.services.agent_registry import agent_registry

router = APIRouter()


@router.post("/register", response_model=AgentResponse)
async def register_agent(payload: AgentRegisterRequest, request: Request) -> AgentResponse:
    """
    采集端 Agent 注册接口。

    当前实现仅在内存中记录 Agent 信息，后续可迁移到数据库。
    """

    host = (payload.host or "").strip()
    # Agent 侧经常会把 host 填成 127.0.0.1/localhost（对采集端“自己”成立，对平台无意义）。
    # 这里统一用请求来源 IP 作为兜底/覆盖，避免平台误连到自身 127.0.0.1。
    if host in {"0.0.0.0", "", "127.0.0.1", "localhost"} or host.startswith("127."):
        host = request.client.host if request.client else host

    info = agent_registry.register_agent(
        agent_id=payload.agent_id,
        name=payload.name,
        host=host,
        port=payload.port,
        devices=payload.devices,
    )
    return AgentResponse(
        agent_id=info.agent_id,
        name=info.name,
        host=info.host,
        port=info.port,
        devices=info.devices,
        runtime_status=info.runtime_status,
        online=info.online,
    )


@router.post("/heartbeat")
async def agent_heartbeat(payload: AgentHeartbeatRequest, request: Request) -> dict:
    """
    Agent 心跳。

    平台重启后内存注册表为空时，若心跳携带 name/host/port（与 register 一致），则自动补注册，无需重启采集端。
    """
    info = agent_registry.get_by_id(payload.agent_id)
    if info is None:
        host = (payload.host or "").strip()
        if host in {"0.0.0.0", "", "127.0.0.1", "localhost"} or host.startswith("127."):
            host = request.client.host if request.client else host

        agent_registry.register_agent(
            agent_id=payload.agent_id,
            name=(payload.name or payload.agent_id),
            host=host,
            port=payload.port,
            devices=list(payload.devices or []),
        )
        info = agent_registry.get_by_id(payload.agent_id)
    else:
        agent_registry.set_online(payload.agent_id, payload.online)
        # 心跳若携带 host/port，且当前 registry 里是本地回环地址，则用请求来源纠正一次
        try:
            host = (payload.host or "").strip()
            if host in {"0.0.0.0", "", "127.0.0.1", "localhost"} or host.startswith("127."):
                host = request.client.host if request.client else host
            port = int(payload.port) if payload.port is not None else None
            if host and port:
                if (not info.host) or (str(info.host).strip() in {"127.0.0.1", "localhost"} or str(info.host).startswith("127.")):
                    info.host = host
                    info.port = port
        except Exception:
            pass

    if payload.runtime_status is not None and info:
        info.runtime_status = payload.runtime_status
    return {"ok": True}


@router.get("/", response_model=list[AgentResponse])
async def list_agents() -> list[AgentResponse]:
    """
    列出所有已注册 Agent。
    """

    items = []
    for info in agent_registry.list_agents():
        items.append(
            AgentResponse(
                agent_id=info.agent_id,
                name=info.name,
                host=info.host,
                port=info.port,
                devices=info.devices,
                runtime_status=info.runtime_status,
                online=info.online,
            )
        )
    return items

