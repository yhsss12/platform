# 后端 API 文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **认证方式**: Bearer Token (JWT)
- **返回格式**: 统一 JSON 格式

### 统一返回结构

**成功响应**：
```json
{
  "ok": true,
  "data": { ... }
}
```

**失败响应**：
```json
{
  "ok": false,
  "error": "错误信息"
}
```

---

## 认证接口

### POST /api/auth/login

用户登录，获取 JWT Token。

**请求体**：
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**响应**：
```json
{
  "ok": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
  }
}
```

**示例**：
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

---

### GET /api/auth/me

获取当前用户信息（需要认证）。

**请求头**：
```
Authorization: Bearer <token>
```

**响应**：
```json
{
  "ok": true,
  "data": {
    "id": "uuid",
    "username": "admin",
    "role": "ADMIN"
  }
}
```

**示例**：
```bash
curl -X GET http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

---

## 任务接口 (Tasks)

### GET /api/tasks

获取任务列表。

**查询参数**：
- `skip` (int, 默认 0): 跳过数量
- `limit` (int, 默认 100): 返回数量

**响应**：
```json
{
  "ok": true,
  "data": [
    {
      "id": "uuid",
      "name": "任务名称",
      "description": "任务描述",
      "status": "DRAFT",
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00"
    }
  ]
}
```

**示例**：
```bash
curl -X GET http://localhost:8000/api/tasks \
  -H "Authorization: Bearer $TOKEN"
```

---

### POST /api/tasks

创建新任务。

**请求体**：
```json
{
  "name": "任务名称",
  "description": "任务描述（可选）",
  "status": "DRAFT"
}
```

**响应**：
```json
{
  "ok": true,
  "data": {
    "id": "uuid",
    "name": "任务名称",
    "description": "任务描述",
    "status": "DRAFT",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00"
  }
}
```

**示例**：
```bash
curl -X POST http://localhost:8000/api/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "测试任务",
    "description": "这是一个测试任务",
    "status": "DRAFT"
  }'
```

---

### GET /api/tasks/{task_id}

根据 ID 获取任务详情。

**路径参数**：
- `task_id` (UUID): 任务 ID

**示例**：
```bash
curl -X GET http://localhost:8000/api/tasks/{task_id} \
  -H "Authorization: Bearer $TOKEN"
```

---

### PATCH /api/tasks/{task_id}

更新任务。

**请求体**（所有字段可选）：
```json
{
  "name": "新名称",
  "description": "新描述",
  "status": "READY"
}
```

**示例**：
```bash
curl -X PATCH http://localhost:8000/api/tasks/{task_id} \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "READY"}'
```

---

### DELETE /api/tasks/{task_id}

删除任务。

**示例**：
```bash
curl -X DELETE http://localhost:8000/api/tasks/{task_id} \
  -H "Authorization: Bearer $TOKEN"
```

---

## 作业接口 (Jobs)

### GET /api/jobs

获取作业列表。

**查询参数**：
- `taskId` (UUID, 可选): 按任务 ID 筛选
- `skip` (int, 默认 0): 跳过数量
- `limit` (int, 默认 100): 返回数量

**示例**：
```bash
curl -X GET "http://localhost:8000/api/jobs?taskId=xxx" \
  -H "Authorization: Bearer $TOKEN"
```

---

### POST /api/jobs

创建新作业。

**请求体**：
```json
{
  "task_id": "uuid",
  "operator_name": "操作员名称（可选）",
  "status": "PENDING"
}
```

**示例**：
```bash
curl -X POST http://localhost:8000/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "xxx",
    "operator_name": "张三",
    "status": "PENDING"
  }'
```

---

### GET /api/jobs/{job_id}

获取作业详情。

**示例**：
```bash
curl -X GET http://localhost:8000/api/jobs/{job_id} \
  -H "Authorization: Bearer $TOKEN"
```

---

### PATCH /api/jobs/{job_id}

更新作业。

**请求体**（所有字段可选）：
```json
{
  "operator_name": "新操作员",
  "status": "RUNNING",
  "progress": 50,
  "mcap_path": "/data/daq/outputs/xxx.mcap",
  "mcap_size_bytes": 104857600,
  "duration_sec": 30,
  "started_at": "2024-01-01T00:00:00",
  "finished_at": "2024-01-01T00:30:00"
}
```

---

### POST /api/jobs/{job_id}/start

启动作业（开始模拟采集）。

**说明**：
- 将作业状态设置为 `RUNNING`
- 启动后台模拟任务，每 500ms 进度 +2
- 通过 WebSocket 实时推送进度
- 完成后自动设置为 `SUCCEEDED`

**示例**：
```bash
curl -X POST http://localhost:8000/api/jobs/{job_id}/start \
  -H "Authorization: Bearer $TOKEN"
```

---

### POST /api/jobs/{job_id}/cancel

取消作业。

**说明**：
- 停止后台模拟任务
- 将作业状态设置为 `CANCELED`
- 通过 WebSocket 推送取消消息

**示例**：
```bash
curl -X POST http://localhost:8000/api/jobs/{job_id}/cancel \
  -H "Authorization: Bearer $TOKEN"
```

---

### POST /api/jobs/{job_id}/finish

手动完成作业。

**请求体**：
```json
{
  "status": "SUCCEEDED",
  "mcap_path": "/data/daq/outputs/xxx.mcap",
  "mcap_size_bytes": 104857600,
  "duration_sec": 30
}
```

**示例**：
```bash
curl -X POST http://localhost:8000/api/jobs/{job_id}/finish \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "SUCCEEDED",
    "mcap_path": "/data/daq/outputs/xxx.mcap",
    "mcap_size_bytes": 104857600,
    "duration_sec": 30
  }'
```

---

## 运行接口 (Runs)

### GET /api/runs

获取运行列表。

**示例**：
```bash
curl -X GET http://localhost:8000/api/runs \
  -H "Authorization: Bearer $TOKEN"
```

---

### POST /api/runs

创建新运行。

**请求体**：
```json
{
  "task_id": "uuid",
  "status": "QUEUED"
}
```

---

### GET /api/runs/{run_id}

获取运行详情。

---

### PATCH /api/runs/{run_id}

更新运行。

---

### DELETE /api/runs/{run_id}

删除运行。

---

## 数据集接口 (Datasets)

### GET /api/datasets

获取数据集列表。

**示例**：
```bash
curl -X GET http://localhost:8000/api/datasets \
  -H "Authorization: Bearer $TOKEN"
```

---

### POST /api/datasets

创建新数据集。

**请求体**：
```json
{
  "name": "数据集名称",
  "status": "ACTIVE"
}
```

---

### GET /api/datasets/{dataset_id}

获取数据集详情。

---

### PATCH /api/datasets/{dataset_id}

更新数据集。

---

### DELETE /api/datasets/{dataset_id}

删除数据集。

---

## WebSocket 接口

### WS /api/ws/jobs/{job_id}

实时作业进度推送。

**连接 URL**：
```
ws://localhost:8000/api/ws/jobs/{job_id}
```

**消息格式**：
```json
{
  "type": "progress",
  "jobId": "uuid",
  "status": "RUNNING",
  "progress": 50
}
```

**状态值**：
- `RUNNING`: 运行中
- `SUCCEEDED`: 成功完成
- `CANCELED`: 已取消

**示例（使用 wscat）**：
```bash
wscat -c ws://localhost:8000/api/ws/jobs/{job_id}
```

**示例（JavaScript）**：
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/jobs/{job_id}');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress, '%');
};
```

---

## 健康检查

### GET /health

健康检查接口（无需认证）。

**响应**：
```json
{
  "ok": true,
  "data": {
    "status": "healthy"
  }
}
```

**示例**：
```bash
curl http://localhost:8000/health
```

---

## 完整工作流示例

### 1. 登录获取 Token

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' \
  | jq -r '.data.access_token')
```

### 2. 创建任务

```bash
TASK_ID=$(curl -s -X POST http://localhost:8000/api/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "测试采集任务",
    "description": "这是一个测试任务",
    "status": "READY"
  }' | jq -r '.data.id')
```

### 3. 创建作业

```bash
JOB_ID=$(curl -s -X POST http://localhost:8000/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"task_id\": \"$TASK_ID\",
    \"operator_name\": \"张三\",
    \"status\": \"PENDING\"
  }" | jq -r '.data.id')
```

### 4. 启动作业

```bash
curl -X POST http://localhost:8000/api/jobs/$JOB_ID/start \
  -H "Authorization: Bearer $TOKEN"
```

### 5. 连接 WebSocket 查看进度

```bash
wscat -c ws://localhost:8000/api/ws/jobs/$JOB_ID
```

### 6. 查询作业状态

```bash
curl -X GET http://localhost:8000/api/jobs/$JOB_ID \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## 错误码

所有错误都通过 `ok: false` 和 `error` 字段返回，HTTP 状态码：

- `200`: 成功
- `401`: 未授权（Token 无效或过期）
- `404`: 资源不存在
- `422`: 请求参数验证失败
- `500`: 服务器内部错误

---

## 前端集成示例

### 使用 fetch API

```javascript
// 登录
const loginResponse = await fetch('http://localhost:8000/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'admin', password: 'admin123' })
});
const { data: { access_token } } = await loginResponse.json();

// 获取任务列表
const tasksResponse = await fetch('http://localhost:8000/api/tasks', {
  headers: { 'Authorization': `Bearer ${access_token}` }
});
const { data: tasks } = await tasksResponse.json();
```

### 使用 WebSocket

```javascript
const ws = new WebSocket(`ws://localhost:8000/api/ws/jobs/${jobId}`);
ws.onmessage = (event) => {
  const { type, jobId, status, progress } = JSON.parse(event.data);
  if (type === 'progress') {
    console.log(`Job ${jobId}: ${progress}% (${status})`);
  }
};
```


