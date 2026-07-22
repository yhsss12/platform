# Agent 自动安装 API（平台侧）

## 1. 概览

用于在平台侧触发“采集端 Agent”的远程安装，并查询安装进度与日志。

> 说明：平台侧的数据同步（同步到 MinIO）推荐通过 **WebSocket 隧道转发**到采集端执行（`CMD_REQUEST cmd=DATA_SYNC`），因此安装/升级采集端时应确保版本包含该能力（见采集端 `AGENT_VERSION` 变更记录）。

本实现采用“异步任务 + 轮询”：

- `POST /api/agent/install`：创建安装任务，返回 `taskId`
- `GET /api/agent/install/{taskId}/status`：查询任务状态（包含进度与日志）
- `GET /api/agent/install/check`：检测目标是否已运行 Agent（可供首次登录时调用）

权限：仅 `SUPER_ADMIN` / `ADMIN`（团队管理员账号）可触发安装。

## 2. POST /api/agent/install

异步触发安装任务。

### 请求体

```json
{
  "ip": "10.0.0.12",
  "os": "linux",
  "arch": "x86_64",
  "version": "0.1.0",
  "ssh_user": "ubuntu",
  "ssh_port": 22,
  "ssh_private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\n...\n-----END OPENSSH PRIVATE KEY-----\n",
  "agent_id": "agent-2",
  "agent_name": "采集端 Agent 2",
  "agent_port": 9100
}
```

说明：
- `os` 当前实现优先支持 `linux`；Windows/macOS 可扩展同一接口。
- Linux/macOS 安装要求：目标机支持免交互 SSH + 免密 sudo（用于注册 systemd/launchd 服务）。

### 响应

```json
{
  "ok": true,
  "data": {
    "taskId": "..."
  }
}
```

### 常见错误码

- `400` 参数错误（缺少 ssh_user/ssh_private_key 等）
- `403` 权限不足
- `429` 队列拥塞（RQ 入队失败/队列达到上限时由内部逻辑返回）
- `500` 服务器内部错误

## 3. GET /api/agent/install/{taskId}/status

返回任务状态（底层复用 `task_jobs` 持久表）。

### 响应（示例）

```json
{
  "ok": true,
  "data": {
    "task_id": "xxx",
    "task_type": "agent_install",
    "status": "running",
    "result": {
      "taskId": "xxx",
      "status": "running",
      "progress": 45,
      "stage": "install",
      "target": { "ip": "10.0.0.12", "os": "linux", "arch": "x86_64" },
      "logs": [
        { "ts": "2026-04-10T00:00:00Z", "level": "info", "message": "开始安装（Linux/SSH）" }
      ]
    },
    "error": null
  }
}
```

`status` 取值：
- `pending` / `queued` / `running` / `success` / `failed` / `cancelled`

## 4. GET /api/agent/install/check

用于轻量判断目标机是否已运行 Agent。

### 请求

`GET /api/agent/install/check?host=10.0.0.12&port=9100`

### 响应

- 已安装且可达：

```json
{ "ok": true, "data": { "ok": true, "agent_id": "agent-2", "version": "0.1.0" } }
```

- 不可达/未安装：

```json
{ "ok": false, "error": "..." }
```

