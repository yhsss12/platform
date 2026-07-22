"""
设备管理 Schema
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union
from datetime import datetime


# ROS2 配置相关
class ROS2ConfigBase(BaseModel):
    mode: Optional[str] = Field(default="fastdds_tailscale_peer", description="连接模式")
    local_bind_ip: Optional[str] = Field(None, description="本地IP")
    domain_id: Optional[int] = Field(default=0, description="ROS2 Domain ID")
    discovery_protocol: Optional[str] = Field(default="SIMPLE", description="发现协议")
    initial_announcements_count: Optional[int] = Field(default=5, description="初始公告次数")
    initial_announcements_period_sec: Optional[int] = Field(default=1, description="初始公告周期（秒）")
    peer_ips: Optional[List[str]] = Field(default_factory=list, description="设备IP列表")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v is None:
            return v
        allowed = ["fastdds_tailscale_peer", "lan_multicast"]
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v

    @field_validator("discovery_protocol")
    @classmethod
    def validate_discovery_protocol(cls, v):
        if v is None:
            return v
        allowed = ["SIMPLE", "SUPER_CLIENT", "CLIENT"]
        if v not in allowed:
            raise ValueError(f"discovery_protocol must be one of {allowed}")
        return v


class ROS2ConfigCreate(ROS2ConfigBase):
    pass


class ROS2ConfigUpdate(ROS2ConfigBase):
    pass


class ROS2ConfigResponse(ROS2ConfigBase):
    profile_path: Optional[str] = None

    class Config:
        from_attributes = True


# 启动配置相关
class DeviceLaunchConfigBase(BaseModel):
    script_path: Optional[str] = Field(description="启动脚本路径")
    script_args: Optional[str] = Field(None, description="启动参数")
    stop_script_path: Optional[str] = Field(None, description="停止脚本路径")
    stop_script_args: Optional[str] = Field(None, description="停止参数")
    env_vars: Optional[dict] = Field(None, description="环境变量")


class DeviceLaunchConfigCreate(DeviceLaunchConfigBase):
    pass


class DeviceLaunchConfigUpdate(DeviceLaunchConfigBase):
    pass


class DeviceLaunchConfigResponse(DeviceLaunchConfigBase):
    id: int

    class Config:
        from_attributes = True


# 设备测试结果
class DeviceTestResultBase(BaseModel):
    status: str = Field(description="测试状态: untested/success/fail")
    node_count: Optional[int] = None
    nodes_sample: Optional[List[str]] = None
    topic_count: Optional[int] = None
    topics_sample: Optional[List[str]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class DeviceTestResultResponse(DeviceTestResultBase):
    tested_at: datetime

    class Config:
        from_attributes = True


# 设备相关
class DeviceBase(BaseModel):
    name: str = Field(description="设备名称")
    vendor: Optional[str] = None
    model: Optional[str] = None
    device_type: str = Field(default="ROS2", description="设备类型")

    team_id: Optional[str] = Field(
        default=None, description="归属团队 teams.id"
    )
    team_name: Optional[str] = Field(
        default=None, description="归属团队名称"
    )

    hardware_uuid: Optional[str] = Field(default=None, description="采集端硬件唯一标识（UUID）")
    hostname: Optional[str] = Field(default=None, description="采集端主机名（Hostname）")
    agent_ip: Optional[str] = Field(default=None, description="采集端 Agent 地址（IP）")
    agent_port: Optional[int] = Field(default=None, description="采集端 Agent 地址（Port）")
    agent_status: Optional[str] = Field(default=None, description="采集端自描述状态（如 READY/ERROR）")
    camera_list: Optional[List[str]] = Field(
        default=None,
        description="采集端能力扫描得到的摄像头/视频流列表（可包含 /dev/video* 与 ROS2 topic）",
    )
    collect_script_compress: Optional[str] = Field(
        default=None,
        description="压缩模式数据采集脚本路径（采集端绝对路径）",
    )
    collect_script_raw: Optional[str] = Field(
        default=None,
        description="原始图像模式数据采集脚本路径（采集端绝对路径）",
    )


class DeviceCreate(DeviceBase):
    ros2_config: Optional[ROS2ConfigCreate] = None
    launch_config: Optional[DeviceLaunchConfigCreate] = None


class DeviceUpdate(BaseModel):
    name: Optional[str] = None
    vendor: Optional[str] = None
    model: Optional[str] = None
    device_type: Optional[str] = None
    hardware_uuid: Optional[str] = None
    hostname: Optional[str] = None
    agent_ip: Optional[str] = None
    agent_port: Optional[int] = None
    agent_status: Optional[str] = None
    camera_list: Optional[List[str]] = None
    collect_script_compress: Optional[str] = None
    collect_script_raw: Optional[str] = None
    ros2_config: Optional[Union[ROS2ConfigUpdate, ROS2ConfigCreate]] = None
    launch_config: Optional[Union[DeviceLaunchConfigUpdate, DeviceLaunchConfigCreate]] = None


class DeviceResponse(DeviceBase):
    id: int
    created_at: datetime
    updated_at: datetime
    ros2_config: Optional[ROS2ConfigResponse] = None
    launch_config: Optional[DeviceLaunchConfigResponse] = None
    last_test_result: Optional[DeviceTestResultResponse] = None
    runtime_status: Optional[str] = Field(
        default=None,
        description="运行状态：OFFLINE/ONLINE_IDLE/LAUNCHING/READY/COLLECTING/ERROR",
    )
    agent_tunnel_connected: Optional[bool] = Field(
        default=None,
        description="采集端 WebSocket 隧道是否在线；有值时表示设备走 Agent 通道，false 时前端应显示未连接",
    )

    class Config:
        from_attributes = True


# 连接测试请求
class TestConnectionRequest(BaseModel):
    pass


class TestConnectionResponse(BaseModel):
    success: bool
    result: DeviceTestResultResponse
    message: Optional[str] = None


# "设备主动添加"请求
class DeviceConnectRequest(BaseModel):
    ip: str = Field(description="采集端 Agent 的 IP")
    port: int = Field(description="采集端 Agent 的端口", ge=1, le=65535)

    name: Optional[str] = Field(default=None, description="设备名称（可选）")
    vendor: Optional[str] = Field(default=None, description="厂商（可选）")
    model: Optional[str] = Field(default=None, description="型号（可选）")
    device_type: Optional[str] = Field(default="ROS2", description="设备类型（可选，默认 ROS2）")
    launch_config: Optional[DeviceLaunchConfigCreate] = Field(
        default=None,
        description='设备启动配置（可选；用于与"启动设备"脚本绑定）',
    )


class DeviceConnectByAgentRequest(BaseModel):
    agent_id: str = Field(description="在线 Agent ID")

    name: Optional[str] = Field(default=None, description="设备名称（可选）")
    vendor: Optional[str] = Field(default=None, description="厂商（可选）")
    model: Optional[str] = Field(default=None, description="型号（可选）")
    device_type: Optional[str] = Field(default="ROS2", description="设备类型（可选，默认 ROS2）")
    hostname: Optional[str] = Field(default=None, description="设备主机名（可选）")
    ros2_config: Optional[ROS2ConfigCreate] = Field(default=None, description="ROS2 配置（可选）")
    launch_config: Optional[DeviceLaunchConfigCreate] = Field(
        default=None,
        description='设备启动配置（可选；用于与"启动设备"脚本绑定）',
    )
    collect_script_compress: Optional[str] = Field(default=None, description="压缩采集脚本路径（可选）")
    collect_script_raw: Optional[str] = Field(default=None, description="原始采集脚本路径（可选）")


class ScanCollectScriptRequest(BaseModel):
    """按任务摄像头格式选择设备上对应的采集脚本并扫描频率检测话题。"""

    camera_data_format: Optional[str] = Field(
        default="压缩",
        description='与采集任务一致：「压缩」或「原始」',
    )