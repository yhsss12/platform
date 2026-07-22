## agent/ 目录运行依赖（完整清单）

本目录包含采集端 Agent（FastAPI）与可选的 ROS2 订阅（MJPEG）/ WebRTC 推流能力。下面按“必需 / 可选”给出完整依赖。

---

## 1) 必需（运行 agent_main.py）

### 1.1 系统环境

- Linux x86_64（Ubuntu 20.04/22.04 建议）
- **Python 3.10**（与平台离线 wheel `cp310`、一键安装脚本 `python3.10` 一致；venv/conda 均可，但解释器主版本须为 3.10）

### 1.2 Python 依赖（pip）

见同目录 `requirements.txt`，安装示例：

```bash
cd agent
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

环境版本锁定清单（与当前离线包 `0.1.30` 对齐）：见同目录 `environment.lock.json`、`environment.lock.txt`；更新依赖后执行 `scripts/generate_environment_lock.sh` 重新生成。

依赖说明（与代码 import 对齐）：

- fastapi / uvicorn：Agent HTTP 服务
- aiohttp / httpx：平台注册、心跳、隧道通信（HTTP/WS 客户端）
- numpy / opencv-python-headless：MJPEG 编码、占位图生成
- aiortc：WebRTC（Offer/Answer、PeerConnection）
- minio：上传/下载 MinIO 产物（若启用相关功能）

### 1.3 运行方式

```bash
cd agent
source .venv/bin/activate
uvicorn agent_main:app --host 0.0.0.0 --port 9000
```

---

## 2) 可选（启用 ROS2 订阅相机 + 机器人状态 DDS 桥）

如果你希望 Agent 在本机直接订阅 ROS2 topic，并提供 `/api/agent/streams` / `/api/agent/stream/{camera_id}` 的 MJPEG 输出，则需要 ROS2 环境（rclpy 等）。

### 2.1 ROS2（系统级）

- ROS2 Humble（或与你的设备一致的发行版）
- rclpy（随 ROS2 安装）
- 常见消息包（随 ROS2 安装；具体依赖由 topic 类型决定）：
  - `sensor_msgs`（Image / CompressedImage / JointState）
  - `geometry_msgs`（若力矩 topic 使用 Wrench）
  - `rosidl_runtime_py`（动态解析消息类型）

### 2.2 运行前需要 source ROS 环境（示例）

```bash
source /opt/ros/humble/setup.bash
```

若同时使用 **Conda** 与 **systemd** 常驻服务，需在启动命令外包一层 `bash -lc`，使环境与交互终端里手动 `uvicorn agent_main:app` 一致；见同目录 `README.md` 小节「ROS2 + Conda」。

---

## 3) 可选（硬件相关）

Agent 本身不强依赖相机驱动，但若你的启动脚本需要 RealSense：

- `realsense2_camera`（ROS2 包，按你的 ROS2 工作区/apt 安装方式）

---

## 4) 常见安装故障与处理

### 4.1 pip 安装 aiortc 失败

优先使用官方 manylinux wheel（大多数情况下无需额外系统库）。

若遇到编译失败，建议先安装基础构建工具与常见依赖后再重试：

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config
```

### 4.2 OpenCV 依赖

Agent 代码在启动时会 import `cv2`；因此必须安装 `opencv-python-headless`（推荐）或 `opencv-python`。

