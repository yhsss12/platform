from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field


class AgentBase(BaseModel):
    agent_id: str = Field(description="Agent 唯一 ID")
    name: str = Field(description="Agent 名称")
    host: Optional[str] = Field(default=None, description="Agent 主机地址（可选）")
    port: Optional[int] = Field(default=None, description="Agent 端口（可选）")
    devices: List[int] = Field(default_factory=list, description="绑定的设备 ID 列表")
    runtime_status: Optional[str] = Field(
        default=None,
        description="运行状态：OFFLINE/ONLINE_IDLE/LAUNCHING/READY/COLLECTING/ERROR",
    )


class AgentRegisterRequest(AgentBase):
    pass


class AgentHeartbeatRequest(BaseModel):
    agent_id: str = Field(description="Agent 唯一 ID")
    online: bool = Field(default=True, description="当前在线状态")
    runtime_status: Optional[str] = Field(
        default=None,
        description="运行状态（可选），例如 READY/COLLECTING/ERROR 等",
    )
    # 平台进程重启后内存注册表清空；若仅发旧版心跳无法补登记。携带下列字段时可在「无记录」时自动等价于 register
    name: Optional[str] = Field(default=None, description="与 register 一致，补注册时使用")
    host: Optional[str] = Field(default=None, description="Agent 可达地址，补注册时使用")
    port: Optional[int] = Field(default=None, description="Agent 端口，补注册时使用")
    devices: List[int] = Field(default_factory=list, description="绑定设备 id 列表，补注册时使用")


class AgentResponse(AgentBase):
    online: bool = Field(description="是否在线")

    class Config:
        from_attributes = True

