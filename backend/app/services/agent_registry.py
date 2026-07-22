import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class AgentInfo:
    """
    采集端 Agent 元数据。

    当前实现为内存注册表，后续可迁移到数据库。
    """

    agent_id: str
    name: str
    host: Optional[str] = None
    port: Optional[int] = None
    devices: List[int] = field(default_factory=list)
    online: bool = True
    # 运行时状态：OFFLINE / ONLINE_IDLE / LAUNCHING / READY / COLLECTING / ERROR
    runtime_status: str = "ONLINE_IDLE"

    @property
    def base_url(self) -> Optional[str]:
        if not self.host or not self.port:
            return None
        return f"http://{self.host}:{self.port}"


class AgentRegistry:
    """
    Agent 注册表。

    目标：
    - 为“采集端 Agent + 服务器平台”模式提供统一查询入口
    - 目前实现为进程内内存结构，后续可以替换为 DB 模型
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentInfo] = {}
        self._device_to_agent: Dict[int, str] = {}
        self._lock = threading.Lock()

        # 默认注册一个本地 Agent，兼容原有“平台直接采集”模式
        self.register_agent(
            agent_id="local-agent",
            name="Local Agent",
            host=None,
            port=None,
            devices=[],
        )

    def register_agent(
        self,
        agent_id: str,
        name: str,
        host: Optional[str],
        port: Optional[int],
        devices: Optional[List[int]] = None,
    ) -> AgentInfo:
        info = AgentInfo(
            agent_id=agent_id,
            name=name,
            host=host,
            port=port,
            devices=devices or [],
            online=True,
            runtime_status="ONLINE_IDLE",
        )
        with self._lock:
            self._agents[agent_id] = info
            for dev_id in info.devices:
                self._device_to_agent[dev_id] = agent_id
        return info

    def update_devices(self, agent_id: str, devices: List[int]) -> None:
        with self._lock:
            info = self._agents.get(agent_id)
            if not info:
                return
            info.devices = devices
            for dev, aid in list(self._device_to_agent.items()):
                if aid == agent_id:
                    self._device_to_agent.pop(dev, None)
            for dev_id in devices:
                self._device_to_agent[dev_id] = agent_id

    def set_online(self, agent_id: str, online: bool) -> None:
        with self._lock:
            info = self._agents.get(agent_id)
            if not info:
                return
            info.online = online
            if not online:
                info.runtime_status = "OFFLINE"
            else:
                # 若此前为 OFFLINE，则恢复到 ONLINE_IDLE；其他状态保持不变
                if info.runtime_status == "OFFLINE":
                    info.runtime_status = "ONLINE_IDLE"

    def get_by_id(self, agent_id: str) -> Optional[AgentInfo]:
        with self._lock:
            return self._agents.get(agent_id)

    def get_by_device_id(self, device_id: int) -> Optional[AgentInfo]:
        with self._lock:
            aid = self._device_to_agent.get(device_id)
            if aid:
                return self._agents.get(aid)
        # 回退到本地 Agent，兼容老方案
        return self._agents.get("local-agent")

    def get_by_device_id_strict(self, device_id: int) -> Optional[AgentInfo]:
        """
        仅当 device_id 已显式映射到某个 agent 时返回；不回退 local-agent。
        用于 WebRTC/MJPEG 等必须指向真实远端采集端的场景，避免误用 local-agent 导致「隧道未连接」。
        """
        with self._lock:
            aid = self._device_to_agent.get(int(device_id))
            if not aid:
                return None
            return self._agents.get(aid)

    def list_agents(self) -> List[AgentInfo]:
        with self._lock:
            return list(self._agents.values())

    def unregister_device(self, device_id: int) -> None:
        """设备从 DB 删除后同步摘掉内存映射，避免仍按旧 device_id 解析到 Agent。"""
        with self._lock:
            self._device_to_agent.pop(int(device_id), None)
            for info in self._agents.values():
                if info.devices and int(device_id) in info.devices:
                    info.devices = [d for d in info.devices if int(d) != int(device_id)]

    def bind_device_to_agent(self, *, device_id: int, agent_id: str) -> bool:
        """
        将 device_id 绑定到指定 agent_id（内存映射自动回填）。
        返回是否成功绑定（agent 存在才会成功）。
        """
        did = int(device_id)
        aid = (agent_id or "").strip()
        if not aid:
            return False
        with self._lock:
            info = self._agents.get(aid)
            if info is None:
                return False
            self._device_to_agent[did] = aid
            if did not in info.devices:
                info.devices.append(did)
        return True


agent_registry = AgentRegistry()

