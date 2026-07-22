# 一湃智能数据平台 - 后端服务

**版本**: v0.1 | **框架**: FastAPI

---

## 简介

本目录为一湃智能数据平台的后端服务，提供认证、任务/作业、项目、设备、数据资产、标注、转换与审计等 REST API，支持 SQLite/PostgreSQL 与 HDF5/MCAP 等数据格式，并与项目根目录的 `label_task_description.py` 配合实现标注自动描述（OpenAI 兼容 API）。

---

## 快速启动

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 配置 .env（参考项目根目录或 docs/development.md）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

默认 API 根地址：`http://localhost:8000`。  
API 文档（Swagger）：`http://localhost:8000/docs`。

---

## 环境变量

常用项（具体以项目文档或 `.env.example` 为准）：

- **认证**：`SECRET_KEY`、JWT 相关
- **数据库**：`DATABASE_URL`（可选）、SQLite 路径等
- **数据资产/存储**：`DATA_ASSETS_DB_PATH`、`HDF5_DATA_DIR`
- **自动标注**：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`
- **Agent 一键安装**：`AGENT_LINUX_X64_TARBALL_PATH`（可选）— 覆盖 `agent_packages/manifest.json` 中的 Linux x86_64 离线包路径；不配则必须在 `backend/agent_packages/` 下存在与 manifest 中 `path`、`sha256` 一致的 `.tar.gz`（用 `agent_packages/build_agent_bundle.sh` 构建，见 `docs/agent-bundling.md`）。若 `PUBLIC_BASE_URL` 指向 Next（如 `:3001`），安装脚本里的下载地址会走该主机；项目根目录 `next.config.js` 已将 `/static/bin/*` 反代到 `NEXT_PUBLIC_API_URL` 对应的后端，避免仅代理 `/api` 时出现 `curl 404`。

---

## 目录结构（概要）

```
backend/
├── app/
│   ├── main.py          # FastAPI 入口与生命周期
│   ├── api/             # API 路由（auth, tasks, jobs, projects, devices, label, ...）
│   ├── core/            # 配置、安全、依赖
│   ├── crud/            # 数据访问
│   ├── models/          # ORM 模型
│   ├── schemas/         # Pydantic 模型
│   ├── services/        # 业务服务（HDF5、MCAP、标注、转换等）
│   └── db/              # 数据库会话与初始化
├── requirements.txt
└── README.md            # 本文件
```

---

## 实时采集架构：采集端 Agent + 服务器平台

### 总体说明

实时采集接口已按照“**采集端 Agent + 服务器平台**”模式重构，目标是：

- 采集端 Agent 负责实际连接设备（ROS2、摄像头、脚本等）；
- 服务器平台负责任务/作业编排、设备/项目管理，对前端暴露统一 REST/WS 接口；
- 平台可以通过 Agent 抽象透明地切换“本机采集”或“远程采集”。

当前实现中，默认存在一个 `local-agent`，其行为与旧版“平台直接执行脚本 + ROS2 摄像头”完全兼容。

### 关键模块位置

- **Agent 注册与查询**
  - `app/services/agent_registry.py`
    - `AgentInfo`：Agent 元信息（`agent_id`、`name`、`host`、`port`、`devices`、`online`）。
    - `agent_registry`：
      - `register_agent(...)`：注册/更新 Agent。
      - `get_by_id(agent_id)`：按 Agent ID 查询。
      - `get_by_device_id(device_id)`：按设备 ID 查询（device → agent 绑定）。
      - `list_agents()`：列出所有 Agent。
      - 默认注册一个 `local-agent`，用于“本机采集”兼容模式。

- **采集（脚本）代理**
  - `app/services/agent_collect_proxy.py`
    - `start_collect_via_agent(...)`：
      - 入参：`script_path`、`args`、`env`、`device_id?`、`agent_id?`、`task_id?`、`job_id?`。
      - 当前实现：解析出 Agent（优先 `agent_id`，再按 `device_id`，否则回退 `local-agent`），然后调用本地 `script_runner.start_script(...)`。
      - 会在环境变量中自动注入：
        - `EAI_TASK_ID`、`EAI_JOB_ID`、`EAI_AGENT_ID`（如存在）。
    - `stop_collect_via_agent(...)`：
      - 当前直接调用 `script_runner.stop_script()`，为未来远程 Agent 停止采集预留扩展点。

- **实时流（视频）代理**
  - `app/services/agent_stream_proxy.py`
    - `list_streams_via_agent(device_id?, agent_id?)`：
      - 通过 Agent 维度查询摄像头列表。
      - 当前实现仍使用本机 `ros2_camera_stream.CameraStreamManager`，返回形如：
        - `[{ "id": "camera1", "name": "camera1 (/topic/...)", "url": "/api/stream/camera1" }, ...]`。
    - `get_stream_generator_via_agent(camera_id, device_id?, agent_id?)`：
      - 返回给 FastAPI 的 MJPEG generator。
      - 当前实现仍直接调用本机 `stream_manager.get_frame_generator(camera_id)`。

- **摄像头流底层实现（未改动核心逻辑，仅通过代理访问）**
  - `app/services/ros2_camera_stream.py`：
    - `CameraStreamManager`：基于 ROS2 `rclpy` 与 `cv2` 的 MJPEG 推流实现。
    - `stream_manager`：全局实例，仍由 `app.main.lifespan` 中启动。

- **Agent API Schema 与路由**
  - `app/schemas/agent.py`：
    - `AgentRegisterRequest`：Agent 注册请求体。
    - `AgentHeartbeatRequest`：心跳请求体。
    - `AgentResponse`：返回体。
  - `app/api/routes_agents.py`：
    - `POST /api/agents/register`：注册或更新 Agent（当前使用内存 registry）。
    - `POST /api/agents/heartbeat`：心跳，更新在线状态。
    - `GET /api/agents/`：列出所有 Agent。
  - `app/api/router.py`：
    - `api_router.include_router(routes_agents.router, prefix="/agents", tags=["agents"])`。

### 对外 API 行为（与前端相关）

#### 1. 实时视频流

- **列表接口**
  - 路由：`GET /api/stream/list`
  - 查询参数：
    - `device_id`（可选，int）：指定设备 ID，平台会通过 `agent_registry.get_by_device_id` 找到对应 Agent。
    - `agent_id`（可选，str）：显式指定 Agent ID，优先级高于 `device_id`。
  - 返回值示例：
    ```json
    [
      { "id": "camera1", "name": "camera1 (/camera1/color/image_raw/compressed)", "url": "/api/stream/camera1" },
      { "id": "camera2", "name": "camera2 (/camera2/color/image_raw/compressed)", "url": "/api/stream/camera2" }
    ]
    ```

- **单路流接口**
  - 路由：`GET /api/stream/{camera_id}`
  - 查询参数：
    - `device_id`（可选，同上）
    - `agent_id`（可选，同上）
  - 返回：`StreamingResponse`，`Content-Type: multipart/x-mixed-replace; boundary=frame`，供前端 `<img src="...">` 播放 MJPEG。

> 说明：当前仍由本机 ROS2 节点提供画面；一旦 Agent 改为远程进程，只需在 `agent_stream_proxy` 中改为 HTTP/WS 代理即可，前端无需调整。

#### 2. 实时采集脚本（启动/停止）

- **启动脚本**
  - 路由：`POST /api/script/start`
  - 请求体（`ScriptRequest`）：
    ```json
    {
      "script_path": "/home/rm/IDE/eai-ide/scripts/rm_robot/collect_data_compress.sh",
      "args": ["-t", "30", "-o", "/path/to/storage"],
      "env": {
        "VALIDATION_CONFIG": "{...}"
      },
      "task_id": "task-uuid",
      "job_id": "job-uuid",
      "device_id": 123,
      "agent_id": "agent-1"
    }
    ```
    - `agent_id` 可选；若不传，则根据 `device_id` → Agent 绑定选择 Agent；都不传时回退到 `local-agent`。
  - 行为：
    - 由 `routes_script.start_script` 调用 `start_collect_via_agent(...)`；
    - 当前实现仍是本地执行脚本，并通过 WebSocket `/api/script/ws` 推送终端输出给前端。

- **停止脚本**
  - 路由：`POST /api/script/stop`
  - 行为：
    - 由 `routes_script.stop_script` 调用 `stop_collect_via_agent()`；
    - 当前为本地 `SIGINT` 停止脚本，未来可代理到远程 Agent。

#### 3. 与前端实时采集页的衔接

- 前端页面：`src/features/daq-editor/pages/Realtime.tsx`
  - 加载摄像头列表：
    - 从任务对象 `task.deviceId` 读取设备 ID；
    - 请求：`GET /api/stream/list?device_id=<task.deviceId>`；
    - 保持原有返回结构不变。
  - 启动采集：
    - 请求 `POST /api/script/start`，在原有 `script_path/args/env` 基础上新增：
      - `task_id: task.id`
      - `job_id: job.id`
      - `device_id: Number(task.deviceId)`（如存在）

> 对前端来说，交互与旧版一致，只是多传了一些语义信息；对后端来说，已经通过 Agent 抽象完成了“按设备/Agent 路由采集”的能力。

### 如何接入真正的远程 Agent（示意）

1. 在采集端部署独立 Agent 服务（例如 `fastapi` + `ros2` + 脚本）并实现：
   - `GET /api/agent/streams`
   - `POST /api/agent/collect/start`
   - `POST /api/agent/collect/stop`
   - （可选）`WS /api/agent/logs` 或事件推送。
2. Agent 启动时调用平台：
   - `POST /api/agents/register` 填写 `agent_id`、`host`、`port`、`devices`。
   - 周期性 `POST /api/agents/heartbeat` 维持在线状态。
3. 在平台侧：
   - 将设备与 Agent 绑定（`devices` 字段中配置设备 ID），或在设备管理表中持久化映射；
   - 在 `agent_collect_proxy.py` / `agent_stream_proxy.py` 中：
     - 将当前本地调用替换为 `httpx` / `aiohttp` 请求远程 Agent；
     - 其余上层路由和前端调用无需修改。

---

## 更多文档

- 完整 API 说明：[../docs/backend-api.md](../docs/backend-api.md)
- 开发与环境：[../docs/development.md](../docs/development.md)
- 项目总览：[../README.md](../README.md)
