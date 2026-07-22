# 设备接入与创建开发文档

面向对象：本项目（EAI Data Platform / eai-idev2.0）后端与前端开发者  
目标：在 Linux 设备上安装/运行采集端 Agent，将其接入平台，并在平台侧完成“设备创建/绑定/可见性控制/状态观测/启动脚本调用”等闭环。

本文参考业内常见“添加设备”流程（命令行安装 / 离线安装 / 手动录入），并结合本仓库当前业务域与代码结构落地为可实现、可测试的开发规范。

---

## 1. 名词与组件

- 平台（Platform）：本仓库 `backend/` + `src/` 提供的 Web 与 API 服务。
- 采集端 Agent（Agent）：运行在设备侧的 FastAPI 服务，源码位于 `agent/`，提供：
  - HTTP API（健康检查、FS 列表、数据同步等）
  - WebSocket 隧道（Agent→Platform，命令执行与日志上报）
- 设备（Device）：平台数据库中的设备实体（团队归属、设备信息、启动脚本、采集脚本等）。

---

## 2. 前置条件

### 2.1 设备侧（Linux）

- Ubuntu/Debian 系（支持 `apt` 或 `dpkg`）
- 具备 sudo 权限（安装 systemd service 与写入 /opt、/etc）
- 建议具备到平台的网络连通性（HTTP + WebSocket）

### 2.2 平台侧（开发环境）

- Node.js 18+（用于前端）
- Python 3.10+（用于后端）
- PostgreSQL（后端业务库）
- Redis（用于安装会话与远程安装凭据缓存）

关键环境变量（后端，`.env`）：

- `DATABASE_URL`：PostgreSQL 异步连接串（示例：`postgresql+asyncpg://user:pass@host:5432/db`）
- `PUBLIC_BASE_URL`：平台对设备侧可访问的根地址（示例：`http://172.18.0.114:8000`）
- `REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`/`REDIS_PASSWORD`
- `AGENT_TUNNEL_TOKEN`：可选，Agent 隧道共享密钥（为空则不校验）
- `AGENT_INSTALL_PUBKEYS_JSON`：可选，Ed25519 公钥列表（base64）用于安装包签名校验

---

## 3. 环境配置与依赖安装

### 3.1 后端启动

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3.2 前端启动

```bash
pnpm install
pnpm run dev
```

---

## 4. 设备接入方式（3 种）

### 4.1 命令行安装（零配置，推荐）

适用场景：少量设备逐台接入，平台不需要 SSH 到设备上。

平台流程：
1. 前端打开“Agent 安装向导”→“零配置一键安装”
2. 平台创建一个安装会话 token（Redis）
3. 平台返回安装脚本 URL 与状态轮询 URL
4. 用户把命令复制到设备侧执行
5. 设备侧脚本自动：
   - 识别架构
   - 下载 deb 到用户指定目录
   - 执行 `apt install <deb>` 或 `dpkg -i <deb>`
   - 注册 systemd service、启动并健康检查
   - 上报安装进度日志到平台（Redis 会话）

关键接口：
- `POST /api/agent/installer/start`：创建零配置安装会话，返回脚本地址与命令模板
- `GET /api/agent/installer/linux.sh?token=...`：获取安装脚本（bash）
- `GET /api/agent/installer/status/{token}`：轮询安装状态与日志

### 4.2 离线安装（deb 包分发）

适用场景：出厂批量安装、内网隔离环境；你希望拿到“标准 deb”并用企业标准流程分发。

平台能力：
- 平台后端可按需生成与缓存 deb（基于 `backend/agent_packages/manifest.json` 中的离线 tar.gz bundle）。
- deb 安装脚本包含：
  - 依赖检查（Depends + postinst 内校验）
  - 权限配置（system user、/etc 权限、/opt owner）
  - 服务注册（systemd enable --now）
  - 安装日志记录（/var/log/eai-agent/install.log）
  - 安装结果验证（调用 `http://127.0.0.1:<port>/api/agent/health`）

离线安装命令示例：

```bash
sudo apt-get update
sudo apt-get -y install ./eai-agent_<version>_<arch>.deb
```

或：

```bash
sudo dpkg -i ./eai-agent_<version>_<arch>.deb
sudo apt-get -y -f install
```

### 4.3 平台远程安装（SSH）

适用场景：批量设备、你希望“一键”由平台代为上传 deb + 执行安装（需要免密 sudo）。

为什么需要 SSH：
- 平台要在设备侧创建目录、上传 deb、执行 apt/dpkg、读取 systemd 状态并做回滚，这些动作需要远程执行通道；当前项目实现使用 SSH。

关键接口：
- `POST /api/agent/install`：创建远程安装任务（RQ worker 执行）
- `GET /api/agent/install/{task_id}/status`：查询安装任务状态

---

## 5. 核心接口调用步骤（开发者版）

### 5.1 获取访问令牌（用于 curl/CLI）

```bash
export BASE_URL="http://127.0.0.1:8000"
export SID="$(python - <<'PY'
import uuid; print(uuid.uuid4())
PY
)"

curl -sS "$BASE_URL/api/auth/login" \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: $SID" \
  -d '{"username":"<account_id>","password":"<password>"}'
```

成功返回：

```json
{ "ok": true, "data": { "access_token": "...", "token_type": "bearer", "session_id": "..." } }
```

后续请求在 Header 中带：

```
Authorization: Bearer <access_token>
```

### 5.2 创建“设备实体”（平台侧）

接口：`POST /api/devices`（管理员/团队管理员）

最小请求体：

```json
{
  "name": "robot-001",
  "device_type": "ROS2"
}
```

推荐请求体（含启动/停止脚本与采集脚本）：

```json
{
  "name": "robot-001",
  "vendor": "ACME",
  "model": "X1",
  "device_type": "ROS2",
  "launch_config": {
    "script_path": "/home/rm/IDE/eai-ide/scripts/rm_robot/start_arm.sh",
    "script_args": "",
    "stop_script_path": "/home/rm/IDE/eai-ide/scripts/rm_robot/stop_arm.sh",
    "stop_script_args": "",
    "env_vars": {}
  },
  "collect_script_compress": "/path/to/collect_compress.sh",
  "collect_script_raw": "/path/to/collect_raw.sh"
}
```

返回：
- HTTP 200，`ApiResponse<DeviceResponse>`
- 若 hardware_uuid 已存在，后端可能复用已有设备并按团队范围做可见性控制（详见后端实现）。

### 5.3 让设备与在线 Agent 绑定（不需要输入 IP/Port）

接口：`POST /api/devices/connect-agent`（管理员/团队管理员）

用途：
- 将“已在线的 agent_id”（来自 WebSocket 隧道连接或 Agent registry）绑定为平台设备
- 自动回填 agent 状态、摄像头列表，并写入 devices 表的 agent_ip/agent_port 等字段

请求示例：

```json
{
  "agent_id": "<hardware_uuid 或自定义 agent_id>",
  "name": "robot-001",
  "device_type": "ROS2"
}
```

### 5.4 Agent 注册与心跳（平台侧）

接口：
- `POST /api/agents/register`：首次注册（内存注册表）
- `POST /api/agents/heartbeat`：心跳（平台重启后可用心跳补注册）
- `GET /api/agents/`：查询在线 agent 列表

心跳建议携带 `name/host/port/devices` 以支持平台重启后自愈。

### 5.5 Agent 隧道（WebSocket）

接口：`GET ws(s)://<platform>/api/agent/tunnel?agent_id=<id>&token=<AGENT_TUNNEL_TOKEN>`

用途：
- 平台向 Agent 下发控制命令（采集启动/FS 列表/流状态等）
- Agent 上报日志与执行结果

安全：
- 若平台设置了 `AGENT_TUNNEL_TOKEN`，Agent 连接时必须带同值 token。

---

## 6. 参数说明（核心字段）

### 6.1 DeviceCreate（`POST /api/devices`）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| name | string | 是 | 设备名称 |
| vendor | string | 否 | 厂商 |
| model | string | 否 | 型号 |
| device_type | string | 否 | 默认 ROS2 |
| hardware_uuid | string | 否 | 采集端硬件唯一标识（用于去重/绑定建议） |
| hostname | string | 否 | 设备主机名 |
| agent_ip | string | 否 | 采集端地址（通常由 Agent 绑定自动回填） |
| agent_port | number | 否 | 采集端端口（通常由 Agent 绑定自动回填） |
| launch_config | object | 否 | 启动/停止脚本配置 |
| collect_script_compress/raw | string | 否 | 采集脚本路径 |

### 6.2 DeviceConnectByAgentRequest（`POST /api/devices/connect-agent`）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| agent_id | string | 是 | 在线 Agent ID（建议用 hardware_uuid） |
| name/vendor/model/device_type | string | 否 | 设备展示信息 |
| launch_config | object | 否 | 启动脚本配置（可覆盖默认） |

### 6.3 AgentInstallRequest（`POST /api/agent/install`，SSH 远程安装）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| ip | string | 是 | 目标主机 |
| os | string | 是 | linux/windows/macos（当前实现 linux） |
| arch | string | 否 | x86_64/arm64 |
| version | string | 否 | 安装包版本（空=latest） |
| ssh_user/ssh_port/ssh_private_key | string/number/string | linux 必填 | 远程执行通道 |
| agent_id/agent_name/agent_port | string/string/number | 否 | Agent 自描述信息 |
| download_dir | string | 否 | deb 下载/落盘目录（默认 /tmp） |
| install_method | string | 否 | auto/apt/dpkg |

---

## 7. 返回结构与返回码对照表

### 7.1 统一响应体（ApiResponse）

```json
{ "ok": true, "data": { ... }, "error": null }
```

或：

```json
{ "ok": false, "data": null, "error": "错误原因" }
```

### 7.2 常见 HTTP 状态码

| HTTP | 场景 |
|---:|---|
| 200 | 成功（即使业务失败也可能以 ok=false 返回） |
| 400 | 参数缺失/非法（如缺 agent_id、缺 ssh_user） |
| 401 | 未登录/Token 无效 |
| 403 | 权限不足/跨团队不可见 |
| 404 | 资源不存在（或不可见时伪装为不存在） |
| 409 | 冲突（如重复绑定、状态冲突） |
| 410 | 接口已弃用 |
| 502/503 | 下游 Agent 不可达/隧道未连接 |

### 7.3 安装任务错误码（远程 SSH 安装）

远程安装任务 result 中可能包含 `errorCode`：

| errorCode | 说明 |
|---|---|
| PERMISSION_DENIED | sudo/权限问题 |
| NETWORK_ERROR | 网络或连接问题 |
| DISK_FULL | 磁盘空间不足 |
| UNKNOWN | 未分类错误 |

---

## 8. 异常处理策略

- 设备不可见：后端按“不可见等同不存在”返回 404，避免信息泄露（跨团队隔离）。
- Agent 不在线：
  - WebSocket 隧道未连接时，涉及 Agent 执行的接口应返回 503 并给出可行动提示。
- 安装失败回滚：
  - 零配置脚本：若本机之前未安装过 eai-agent，则失败时尽力卸载；否则尽力重启恢复。
  - SSH 远程安装：同理，失败时尽力 `dpkg -r eai-agent` 并恢复服务。
- 幂等性：
  - 设备创建支持按 `hardware_uuid` 去重复用；重复创建不应产生多条设备记录（取决于入参是否携带 hardware_uuid）。

---

## 9. 安全校验机制

- 认证：平台 API 默认需要登录态（Bearer token 或前端 Cookie 流程）
- 授权：
  - 创建设备/绑定设备/远程安装：管理员或团队管理员（后端强校验）
  - 设备可见性：同团队或 SUPER_ADMIN
- 隧道密钥：
  - `AGENT_TUNNEL_TOKEN` 非空时，Agent WebSocket 隧道必须携带一致 token
- 安装包校验：
  - manifest sha256 强校验
  - 若配置了签名文件与 `AGENT_INSTALL_PUBKEYS_JSON`，可启用 Ed25519 detached signature 校验

---

## 10. 日志埋点与观测要求

建议在以下关键节点埋点（后端 logger + 审计日志）：

1. 设备创建：包含 `user_id/team_id/device_id/hardware_uuid`
2. connect-agent：包含 `user_id/team_id/agent_id/device_id`
3. 安装任务：包含 `task_id/target_ip/arch/version/download_dir/install_method`
4. 隧道状态：Agent 连接/断开、心跳补注册

日志格式建议包含可检索前缀，例如：
- `[DEVICE] create ...`
- `[DEVICE] connect-agent ...`
- `[AGENT-INSTALL] start ...`
- `[AGENT-INSTALL] stage=... progress=...`

---

## 11. 单元测试与集成测试用例（建议）

### 11.1 单元测试

- deb 构建：
  - 生成 `.deb` 并验证 ar 成员存在：`debian-binary/control.tar.gz/data.tar.gz`
  - 验证 data.tar.gz 中包含 `/opt/eai-agent`、systemd unit、runner

- manifest 解析：
  - sha256 mismatch 失败
  - latest 版本解析正确

### 11.2 集成测试（推荐在 CI 或开发机执行）

在可启动 PostgreSQL + Redis 的环境中：

1. 启动平台后端
2. 调用 `POST /api/agent/installer/start` 获得脚本 URL
3. 在同机或容器内执行脚本（指定 `--dir`）
4. 轮询 `GET /api/agent/installer/status/{token}` 直到 success
5. 调用 `GET /api/agents/` 确认 Agent online
6. 调用 `POST /api/devices/connect-agent` 完成绑定
7. 调用 `POST /api/devices/{id}/launch` 验证隧道命令可用（需要 Agent 支持对应 CMD）

---

## 12. 性能基准指标（建议）

以单设备为单位的建议指标（用于回归对比）：

- 设备创建 API：P95 < 200ms（不含外部依赖）
- connect-agent：P95 < 300ms（不含隧道命令）
- 安装脚本：
  - deb 下载：取决于网络与包体积（建议记录实际吞吐）
  - 安装阶段：P95 < 3min（取决于设备性能与 wheelhouse 完整度）
- Agent 心跳：每 15s 一次，平台处理耗时应稳定 < 50ms

---

## 13. 上线 Checklist

- [ ] `PUBLIC_BASE_URL` 配置为设备侧可访问地址（不要是 127.0.0.1）
- [ ] Redis 可用（installer/start 与进度上报依赖）
- [ ] `AGENT_TUNNEL_TOKEN` 策略明确（空=不校验；非空=强校验）
- [ ] `backend/agent_packages/manifest.json` 已更新 latest 与 sha256
- [ ] 目标 OS/Arch 的离线包准备齐全（必要时包含 wheelhouse）
- [ ] 权限策略确认：设备创建/绑定仅管理员/团队管理员
- [ ] 设备可见性：跨团队不可见时返回 404（避免泄露）
- [ ] 安装日志路径与权限确认（/var/log/eai-agent/install.log）

---

## 14. 回滚方案

### 14.1 回滚安装（设备侧）

- 保留配置（不 purge）：
  - `sudo dpkg -r eai-agent`
- 完全清理（purge）：
  - `sudo dpkg -P eai-agent`

### 14.2 回滚平台变更

- 若 deb 生成/分发逻辑引发问题，可临时回退到“tar.gz + venv + pip”安装链路（历史脚本方式）。
- 保持 API 兼容：
  - `/api/agent/installer/start/status/linux.sh` 的路由路径不变
  - 前端仍可显示安装状态

