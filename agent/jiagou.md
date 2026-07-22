采集端 Agent 总体架构

- 整体形态
  - 每台采集设备/边缘机上部署一个常驻的 Agent 服务（FastAPI + uvicorn），充当“本地采集控制器”。
  - 中心服务器平台通过 HTTP 调用各个 Agent，实现 “平台编排 / 采集端执行” 的模式；在未配置远端 Agent 时，仍可回退为平台本机直接执行采集脚本。
    

---

1. 角色划分

- 服务器平台（backend）
  - 管理设备、任务、作业及元数据（DB）。
  - 暴露统一 REST/WS 接口给前端（设备管理页、实时采集页）。
  - 通过 `AgentRegistry`（`app/services/agent_registry.py`）维护 `device.id -> agent_id -> host:port` 映射，并默认注册一个本地 `local-agent` 以兼容旧模式。
  - 所有与“开始采集 / 停止采集”相关的接口统一通过 `agent_collect_proxy` 这一层进行转发或回退：
    - `/api/script/start` → `start_collect_via_agent(...)`
    - `/api/script/stop` → `stop_collect_via_agent(...)`
  - 目前“启动设备 / 停止设备”的接口还未完全接入 Agent（以实际代码为准），后续可以参考采集链路同样接到 Agent。
    
- 采集端 Agent（`agent/agent_main.py`）
  - 部署在接 ROS2 / 摄像头 / 机械臂的那台边缘机。
  - 启动时向平台注册（`POST /api/agents/register`），并周期性心跳（`POST /api/agents/heartbeat`），上报自身绑定的 `devices` 列表和在线状态。
  - 暴露一小组 HTTP 接口：
    - `GET /api/agent/streams`：列出本机可用相机流（由 `ros2_camera_stream.py` 扫描 ROS2 话题自动得出）。
    - `GET /api/agent/stream/{camera_id}`：输出某一路相机的 MJPEG 流。
    - `POST /api/agent/device/test-connection`：在本机执行 ROS2 连通性测试（`ros2 node list` / `ros2 topic list`）。
    - `POST /api/agent/device/launch`：根据平台下发的 `launch_config` 拉起设备相关脚本/ROS2 节点。
    - `POST /api/agent/device/stop`：停止设备相关进程或执行停止脚本。
    - `POST /api/agent/collect/start`：启动采集脚本（录制数据），根据 `camera_data_format` 选择压缩/原始脚本，并使用平台下发的 `storage_path` 存储。
    - `POST /api/agent/collect/stop`：停止采集脚本进程。
  - 内部只负责在本机执行脚本和管理子进程，不做业务调度。
    

---

2. 典型调用

- 启动设备（后台设备管理页按钮，**后续可对齐到 Agent**）
  1. 前端：调用对应设备启动 API（如 `POST /api/devices/{deviceId}/launch`，以实际实现为准）。
  2. 平台：
    - 查询 `device.launch_config`（脚本路径/参数/env）。
    - 通过 `deviceId -> AgentRegistry` 找到采集端 Agent（或回退到 `local-agent`）。
    - 调用该 Agent 的 `POST /api/agent/device/launch`，在边缘机上启动 ROS2 节点/驱动。
  3. Agent：在本机执行启动脚本，并做最小化的日志与错误检测。
    
- 停止设备
  1. 前端：调用设备停止 API（如 `POST /api/devices/{deviceId}/stop`，以实际实现为准）。
  2. 平台：通过 `deviceId -> AgentRegistry` 找到 Agent，调用 `POST /api/agent/device/stop`，或回退为本机停止逻辑。
  3. Agent：执行停止脚本，并尝试终止记录在本机的启动进程。
    
- 开始采集（实时采集页按钮，**已按项目实现**）
  1. 前端：在实时采集页点击“开始采集”，触发 `POST /api/script/start`，携带：
     - `task_id` / `job_id`：任务与作业标识。
     - `device_id`：当前任务绑定的设备 ID，用于路由到对应 Agent。
     - `script_path`、`args`（含 `-t` 时长、`-o` 输出路径）、`env`（含频率检测配置等）。
  2. 平台（`routes_script.py` + `agent_collect_proxy.py`）：
     - 通过 `agent_id` 或 `device_id` 在 `AgentRegistry` 中查找 Agent 信息。
     - 如 Agent 有可用 `base_url`，则构造 `CollectStartRequest` 负载：
       - `duration_sec`：解析自 `args` 中 `-t` 参数。
       - `storage_path`：解析自 `args` 中 `-o` 参数。
       - `env` / `task_id` / `job_id`：直接转发。
     - 调用远端 `POST {agent.base_url}/api/agent/collect/start`。
     - 若调用失败或未配置 Agent，则回退为本地 `script_runner.start_script(...)` 执行同一脚本。
  3. Agent：在本机执行采集脚本，根据 `camera_data_format` 选择脚本，将录制数据写入平台下发的 `storage_path`。
    
- 停止采集
  1. 前端：调用 `POST /api/script/stop`。
  2. 平台：经 `stop_collect_via_agent(...)`：
     - 若找到远端 Agent，调用 `POST {agent.base_url}/api/agent/collect/stop`。
     - 如远端不可用或未配置，则回退为本机 `script_runner.stop_script()`。
  3. Agent：终止采集脚本进程。
    

---

3. 核心设计要点

- 单向控制：平台始终是“主控端”，所有动作从平台发起；Agent 不主动控制平台，只做注册、心跳以及采集/设备执行反馈。
- 按设备路由：以 `device.id` 为主键，在平台侧通过 `AgentRegistry` 维护“设备 → Agent”的绑定，前端只传 `deviceId`，不直接感知 Agent 细节。
- 渐进式抽象层：
  - 采集链路已经统一走 `agent_collect_proxy` + `AgentRegistry`，能优先走远端 Agent，不可用时回退到本机 `script_runner`。
  - 设备启动/停止链路正在向同一模式收敛，可参考当前 Agent 的 `/api/agent/device/launch` / `/api/agent/device/stop` 实现进行接入。
- 兼容旧模式：即使没有部署任何采集端 Agent，`AgentRegistry` 中的 `local-agent` 仍然可以让 `/api/script/*` 在服务器本机直接执行现有脚本，避免破坏原有流程。