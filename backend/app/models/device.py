"""
设备管理模型（PostgreSQL 统一库）
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base


class Device(Base):
    """设备表"""
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, comment="设备名称")
    vendor = Column(String, nullable=True, comment="厂商")
    model = Column(String, nullable=True, comment="型号")
    device_type = Column(String, nullable=False, default="ROS2", comment="设备类型: ROS/ROS2")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # 采集端硬件/Agent 元数据（用于“设备主动添加/握手”）
    hardware_uuid = Column(String, nullable=True, comment="采集端硬件唯一标识（UUID）")
    hostname = Column(String, nullable=True, comment="采集端主机名（Hostname）")
    agent_ip = Column(String, nullable=True, comment="采集端 Agent 地址（IP）")
    agent_port = Column(Integer, nullable=True, comment="采集端 Agent 地址（Port）")
    agent_status = Column(String, nullable=True, comment="采集端自描述状态（如 READY/ERROR）")
    camera_list_json = Column(Text, nullable=True, comment="采集端能力扫描得到的摄像头/视频流列表（JSON）")
    # 数据采集脚本（采集端路径）；与任务「相机数据格式」压缩/原图对应，为空则前端使用内置默认路径
    collect_script_compress = Column(String(1024), nullable=True, comment="压缩采集脚本路径，如 collect_data_compress.sh")
    collect_script_raw = Column(String(1024), nullable=True, comment="原始图像采集脚本路径，如 collect_data.sh")

    # 设备归属团队：用于设备可见性控制（只有同团队成员/管理员可见）
    team_id = Column(String(128), nullable=True, index=True, comment="归属团队 teams.id")

    # 关系
    ros2_config = relationship("ROS2Config", back_populates="device", uselist=False, cascade="all, delete-orphan")
    launch_config = relationship("DeviceLaunchConfig", back_populates="device", uselist=False, cascade="all, delete-orphan")
    test_results = relationship("DeviceTestResult", back_populates="device", cascade="all, delete-orphan", order_by="desc(DeviceTestResult.tested_at)")

    __table_args__ = (
        UniqueConstraint("hardware_uuid", name="uq_device_hardware_uuid"),
        Index("idx_device_name", "name"),
        Index("idx_device_type", "device_type"),
        Index("idx_device_hardware_uuid", "hardware_uuid"),
        Index("idx_device_team_id", "team_id"),
        Index("idx_updated_at", "updated_at"),
    )


class ROS2Config(Base):
    """ROS2 配置表"""
    __tablename__ = "ros2_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, unique=True)
    mode = Column(String, nullable=False, default="fastdds_tailscale_peer", comment="连接模式: fastdds_tailscale_peer/lan_multicast")
    local_bind_ip = Column(String, nullable=True, comment="本地IP")
    domain_id = Column(Integer, nullable=False, default=0, comment="ROS2 Domain ID")
    discovery_protocol = Column(String, nullable=False, default="SIMPLE", comment="发现协议: SIMPLE/SUPER_CLIENT/CLIENT")
    initial_announcements_count = Column(Integer, nullable=False, default=5, comment="初始公告次数")
    initial_announcements_period_sec = Column(Integer, nullable=False, default=1, comment="初始公告周期（秒）")
    profile_path = Column(String, nullable=True, comment="FastDDS配置文件路径")
    peer_ips_json = Column(Text, nullable=True, comment="设备IP列表（JSON数组）")

    # 关系
    device = relationship("Device", back_populates="ros2_config")

    __table_args__ = (
        Index("idx_ros2_config_device_id", "device_id"),
    )


class DeviceLaunchConfig(Base):
    """设备启动配置表"""
    __tablename__ = "device_launch_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, unique=True)
    script_path = Column(String, nullable=False, comment="启动脚本路径")
    script_args = Column(String, nullable=True, comment="启动参数")
    stop_script_path = Column(String, nullable=True, comment="停止脚本路径")
    stop_script_args = Column(String, nullable=True, comment="停止参数")
    env_vars_json = Column(Text, nullable=True, comment="环境变量（JSON）")
    
    device = relationship("Device", back_populates="launch_config")

    __table_args__ = (
        Index("idx_device_launch_config_device_id", "device_id"),
    )


class DeviceTestResult(Base):
    """设备测试结果表"""
    __tablename__ = "device_test_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("devices.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False, comment="测试状态: untested/success/fail")
    node_count = Column(Integer, nullable=True, comment="发现的节点数量")
    nodes_sample_json = Column(Text, nullable=True, comment="节点示例（JSON数组，最多10个）")
    topic_count = Column(Integer, nullable=True, comment="发现的主题数量")
    topics_sample_json = Column(Text, nullable=True, comment="主题示例（JSON数组，最多10个）")
    error_type = Column(String, nullable=True, comment="错误类型")
    error_message = Column(Text, nullable=True, comment="错误消息")
    tested_at = Column(DateTime, server_default=func.now(), nullable=False)

    # 关系
    device = relationship("Device", back_populates="test_results")

    __table_args__ = (
        Index("idx_device_test_result_device_id", "device_id"),
        Index("idx_tested_at", "tested_at"),
        Index("idx_device_test_status", "status"),
    )
