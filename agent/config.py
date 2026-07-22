"""
采集端 Agent 配置示例。

使用方法：
- 将本文件复制为 config.py，并按实际环境修改常量。
"""

# 平台后端根地址（注意端口与协议）
SERVER_BASE = "http://36.133.93.68:8000"

# Agent 标识：留空则自动使用「主网卡 MAC」（小写冒号，如 aa:bb:cc:dd:ee:ff），与平台 devices.hardware_uuid、隧道 ?agent_id= 一致。
# 若填写非 MAC 字符串，须与平台「添加设备」时使用的 agent_id 完全一致；推荐留空统一走网卡 MAC。
AGENT_ID = ""
AGENT_NAME = "采集端 Agent"

# Agent 侧用于注册/绑定的本地编号列表（可选，与平台 devices 表主键无关）。
# 平台以「设备连接成功」时下发的平台设备 ID（领取作业时写入 collection_jobs.device_id）为准；
# 数据资产、同步到 MinIO 等链路请勿依赖本项，可留空 []。
DEVICES = []

# Agent HTTP 服务绑定地址与端口（uvicorn）
# 说明：平台在「设备」里连接采集端时由用户填写可访问的 IP/端口并落库；同步到 MinIO 时平台优先使用该地址，一般无需为 IP 单独再配一项。
AGENT_HOST = "0.0.0.0"
AGENT_PORT = 9000

# 可选：与平台环境变量 AGENT_TUNNEL_TOKEN 一致时，WebSocket 隧道 URL 会附带 token=（也可用环境变量 EAI_AGENT_TUNNEL_TOKEN 覆盖）
AGENT_TUNNEL_TOKEN = "dhiwhid12321312"
#
# 平台侧可选用 AGENT_TUNNEL_TOKEN_BY_AGENT_JSON（.env）为每个 agent_id 配置独立密钥（文档 §4.2）；此时 Agent URL 的 token 须与该 agent 在 map 中的值一致。
#
# 以下为环境变量示例（可写在 shell / systemd 而非本文件）：
# EAI_AGENT_ROS_STATE_USE_SUB=1          # 1=优先用 rclpy 订阅关节/末端（DDS），0=仅用 ros2 topic echo
# EAI_AGENT_ROS_STATE_EXPORT_INTERVAL_SEC=0.5  # 订阅模式下写入 heartbeat 缓存的刷新周期（秒）
# EAI_AGENT_ROS_STATE_RECONCILE_SEC=2.0   # 发现/增删订阅的周期（秒）
#
# 预览（MJPEG）：流畅优先可设 EAI_AGENT_PREVIEW_FLUID=1（约 30FPS + 低 JPEG 质量 + 长边 640）
# EAI_AGENT_PREVIEW_MAX_FPS=30            # 预览编码上限帧率；0=不限制
# EAI_AGENT_PREVIEW_JPEG_QUALITY=55      # 1–100，越低越糊、体积越小；0=使用 OpenCV 默认质量
# EAI_AGENT_PREVIEW_MAX_EDGE=640         # 长边超过则缩小再编码，0=不缩放

