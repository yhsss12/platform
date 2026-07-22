## 采集端 Agent 服务

### 目录说明

本目录为“采集端 Agent”服务示例，用于部署在实际连接 ROS2 / 摄像头 / 采集脚本的边缘机上。

- `agent_main.py`：FastAPI 应用入口，提供 `/api/agent/collect/start|stop` 等接口，并在启动时向平台注册自身信息。
- `requirements.txt`：采集端 Agent Python 依赖（pip）。
- `PREREQUISITES.md`：采集端 Agent 完整前置依赖说明（含 ROS2 可选项）。
- `CHANGELOG.md`：采集端 Agent 版本变更记录（安装/升级对照）。


### 部署步骤（在采集端机器上）

1. 拷贝项目中的 `agent/` 目录到采集端机器，例如：

   ```bash
   scp -r agent/ rm@collector-host:~/eai-agent
   ```

2. 在采集端创建虚拟环境并安装依赖（**请使用 Python 3.10**，与平台离线包 `wheelhouse`、ROS2 Humble 常见环境一致）：

   ```bash
   cd ~/eai-agent
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. 复制并修改配置：

   ```bash
   cp config.example.py config.py
   # 编辑 config.py，填入：
   # - SERVER_BASE（平台后端地址，如 http://10.0.0.1:8000）
   # - AGENT_ID / AGENT_NAME（自定义，需与平台上 device 绑定逻辑一致）
   # - DEVICES（该 Agent 管理的 device.id 列表）
   # - AGENT_HOST / AGENT_PORT（本机监听地址与端口）
   ```

4. 启动 Agent 服务：

   ```bash
   source .venv/bin/activate
   uvicorn agent_main:app --host 0.0.0.0 --port 9100
   ```

   等价写法（与上面二选一，均启动同一 FastAPI 应用）：

   ```bash
   source .venv/bin/activate
   python agent_main.py
   ```

   启动后，Agent 会在启动阶段调用：

   - `POST {SERVER_BASE}/api/agents/register`
   - 并通过 `POST {SERVER_BASE}/api/agents/heartbeat` 周期性上报在线状态。

   同时，Agent 会建立到平台的 WebSocket 隧道（用于控制命令、日志、以及“隧道转发同步”）：

   - `WS {SERVER_BASE}/api/agent/tunnel?agent_id=...&token=...`

5. 在平台上确认：

   - 调用 `GET /api/agents` 应能看到刚注册的 Agent；
   - 绑定的 `device.id` 对应的实时采集任务，会通过 `device_id` 自动路由到该 Agent。

### 检查采集端 Client（Agent）状态

下面按“**平台侧**”与“**采集端机器侧**”分别给出检查清单，便于定位“设备已添加但提示隧道未连接 / 同步失败”等问题。

#### 平台侧（推荐优先看）

1. **看 Agent 注册是否存在**

   - `GET /api/agents`
   - 预期：能看到目标 `agent_id`（推荐统一使用网卡 MAC，小写冒号如 `aa:bb:cc:dd:ee:ff`），且 `online=true`。

2. **看设备是否绑定到该 Agent（并看隧道连接态）**

   - 在设备列表/设备详情中查看字段 `agent_tunnel_connected`（后台会基于 `agent_id` 判断 WebSocket 隧道是否已连接）。
   - 若为 `false`：表示平台当前没有该 `agent_id` 的活跃隧道连接（常见原因：Agent 未启动、网络不通、token 不匹配、或 agent_id 不一致）。

3. **看隧道指标/命令状态（排障用）**

   - `GET /api/agent/tunnel/metrics?include_commands=true&agent_id=<agent_id>`
   - 预期：`metrics.connected_agents` 大于 0；并可看到最近 `CMD_REQUEST/CMD_RESULT` 的命令状态（例如 `DATA_SYNC`）。

#### 采集端机器侧

1. **确认进程在跑**

   - 通过 `ps` / `systemctl status`（如用 systemd）确认 Agent 进程存在且无频繁重启。

2. **确认 Agent 使用的 agent_id 正确（建议 MAC）**

   - Agent 默认会自动探测“主网卡 MAC”作为 `agent_id`（小写冒号）；若你在 `config.py` 或环境变量中显式配置了 `AGENT_ID/EAI_AGENT_ID`，请确保与平台 `devices.hardware_uuid` 完全一致。

3. **确认能连到平台（HTTP + WebSocket 隧道）**

   - Agent 启动后应调用：
     - `POST {SERVER_BASE}/api/agents/register`
     - `POST {SERVER_BASE}/api/agents/heartbeat`（周期上报）
   - 并建立 WebSocket 隧道：
     - `WS {SERVER_BASE}/api/agent/tunnel?agent_id=...&token=...`
   - 若平台开启了 `AGENT_TUNNEL_TOKEN_BY_AGENT_JSON`：token 必须与该 `agent_id` 在 map 中的值一致；否则会在平台日志中出现 `invalid_or_missing_tunnel_token`。

### 数据同步（隧道转发模式，推荐）

平台侧“同步到 MinIO”已支持通过隧道转发到采集端执行（**不依赖采集端 IP**，仅依赖 `agent_id` + 隧道连接）：

- 平台发送 `CMD_REQUEST cmd=DATA_SYNC` 到采集端
- 采集端在本机读取 `source_path`，并将文件/目录上传到平台 MinIO
- 采集端通过 `CMD_RESULT` 返回 `minio_path`（形如 `minio://bucket/prefix/...`）

**版本要求**：采集端需 **`AGENT_VERSION >= 0.1.25`**（新增 `DATA_SYNC` 命令）；建议使用 **`0.1.26+`**（停止采集会终止整个脚本进程组，避免子进程残留）。

> 兼容说明：采集端仍保留 HTTP `POST /api/agent/data/sync`（可用于调试或兼容旧链路），但生产链路建议使用隧道转发，避免采集端 IP 变化/NAT/多网卡导致的平台侧连通性问题。

> 当前 Agent 示例已实现：采集脚本控制、实时预览（隧道）、以及隧道转发同步（DATA_SYNC）。平台后端的 `agent_collect_proxy.py` / `agent_stream_proxy.py` 对应这些能力。

### ROS2 + Conda（与手动 `uvicorn agent_main:app` 等价）

你在交互终端里通常是：**先进入 conda 环境 → 再 source ROS（及自己的工作空间）→ 再 `cd` 到 `agent` 目录 → `uvicorn`**。  
`systemd` 等非登录环境没有这些初始化，因此要用 **`bash -lc` 把同一串命令包进去**，效果才与手动一致。

**交互式终端（与以前手动方式一致，便于调试）：**

```bash
# 按你机器上的 conda 安装路径修改
source "${HOME}/miniconda3/etc/profile.d/conda.sh"   # 或 anaconda3、micromamba 等
conda activate 你的环境名

source /opt/ros/humble/setup.bash   # 发行版按实际修改（foxy/humble/jazzy 等）
# 若相机/驱动在工作空间内，再叠加，例如：
# source ~/your_ws/install/setup.bash

cd /path/to/eai-agent   # 含 agent_main.py、config.py 的目录
uvicorn agent_main:app --host 0.0.0.0 --port 9100
# 或: python agent_main.py
```

**顺序提示：** 若 `conda activate` 与 `setup.bash` 互相覆盖 `PATH` / `PYTHONPATH`，以你本机「能 `ros2 topic list` 且 `python -c "import rclpy"` 都成功」的顺序为准；常见是先 conda 再 ROS，再叠加工作空间 `install/setup.bash`。

**systemd 服务（与上面同一环境，常驻运行）：**

把 `ExecStart` 写成一条 `bash -lc`，内部用 `exec` 把子进程提升为主进程，便于 systemd 跟踪与重启：

```ini
[Service]
Type=simple
WorkingDirectory=/path/to/eai-agent
# 可按需增加 EAI_* 等环境变量
Environment=EAI_AGENT_DATA_ROOT=/home/你的用户
ExecStart=/bin/bash -lc 'source "${HOME}/miniconda3/etc/profile.d/conda.sh" && conda activate 你的环境名 && source /opt/ros/humble/setup.bash && exec uvicorn agent_main:app --host 0.0.0.0 --port 9100'
User=你的用户
```

若服务以 **root** 运行但数据在用户家目录，可把 `conda.sh` 与 `conda activate` 写成该用户下的绝对路径（例如 `/home/ubuntu/miniconda3/...`），避免 `root` 的 `${HOME}` 找不到 conda。

平台一键安装脚本会生成 **`/opt/eai-agent/run-agent.sh`**，由 systemd `ExecStart` 调用（见 `backend/static/scripts/installer_template.sh`），后端通过环境变量控制 **venv + ROS** 或 **Conda + ROS**（与手动 `uvicorn`/`python agent_main.py` 同序）：

- `AGENT_INSTALL_USE_CONDA`、`AGENT_INSTALL_CONDA_SH`、`AGENT_INSTALL_CONDA_ENV`
- `AGENT_INSTALL_ROS_SETUP`、`AGENT_INSTALL_ROS_WS_SETUP`（可选叠加工作空间）
- `AGENT_INSTALL_SERVICE_USER`（非 root 时安装脚本会对 `/opt/eai-agent` 做 `chown`）

配置在后端 `backend/.env`（或部署环境变量），安装向导下发的 `curl ... linux.sh` 会带上相应查询参数。亦可自行在浏览器/终端请求 `GET /api/agent/installer/linux.sh?...` 覆盖单次安装参数。
