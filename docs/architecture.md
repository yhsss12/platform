# 项目架构分析文档

## 一、项目概览

### 项目名称
- **中文名**: 一湃智能数据平台
- **英文名**: Yipai Intelligent Data Platform

### 项目类型
- **前端**: Next.js 15.5.3 (App Router) + React 19.1.0 + TypeScript 5
- **后端**: FastAPI 0.115.0 + Python 3.11+
- **数据库**: PostgreSQL 16 (asyncpg 驱动)
- **架构模式**: 前后端分离，RESTful API + WebSocket

### 项目定位
数据采集、管理和处理平台，支持任务管理、作业调度、实时采集、质量校验等功能。

---

## 二、前端架构

### 2.1 技术栈

#### 核心依赖
```json
{
  "next": "15.5.3",           // Next.js 框架（App Router）
  "react": "19.1.0",          // React 库
  "react-dom": "19.1.0",      // React DOM
  "lucide-react": "^0.563.0", // Icon 库
  "typescript": "^5"          // TypeScript
}
```

#### 开发工具
- **包管理器**: pnpm
- **构建工具**: Next.js 内置 Webpack
- **类型检查**: TypeScript
- **代码规范**: TypeScript strict mode（部分开启）

### 2.2 项目结构

```
src/
├── app/                          # Next.js App Router
│   ├── layout.tsx                # 根布局
│   ├── page.tsx                  # 根页面（重定向到 /collect/tasks）
│   └── (platform)/               # 平台路由组
│       ├── layout.tsx            # 平台布局（Topbar + Sidebar）
│       ├── page.tsx              # 平台首页（重定向）
│       ├── collect/              # 采集模块
│       │   ├── tasks/page.tsx    # 任务列表
│       │   ├── jobs/page.tsx     # 作业中心
│       │   ├── realtime/page.tsx # 实时采集
│       │   └── quality/page.tsx   # 质量校验
│       ├── admin/                # 管理模块
│       └── ...                   # 其他功能模块
│
├── components/                    # 全局组件
│   ├── brand/
│   │   └── BrandMark.tsx         # 品牌标识组件
│   └── layout/
│       └── sidebar/
│           └── navItems.ts       # 菜单配置
│
└── features/                      # 功能模块（Feature-based）
    ├── daq-editor/               # 数据采集编辑器模块
    │   ├── api/                  # API 层（Mock）
    │   ├── components/           # 组件
    │   ├── models/               # 数据模型
    │   ├── pages/                # 页面组件
    │   └── utils/                 # 工具函数
    └── data-platform/            # 数据平台核心模块
        ├── api/                  # API 层（FastAPI 客户端）
        ├── components/           # 通用组件
        └── models/               # 数据模型
```

### 2.3 路由架构

#### Next.js App Router 结构
- **路由组**: `(platform)` - 不影响 URL 路径，仅用于布局分组
- **文件系统路由**: 基于文件路径自动生成路由
- **动态路由**: `[projectId]` 支持动态参数

#### 路由层级
```
/                           → 重定向到 /collect/tasks
/collect/tasks              → 任务列表（默认入口）
/collect/jobs?taskId=xxx    → 作业中心
/collect/realtime?jobId=xxx&taskId=xxx → 实时采集
/collect/quality?taskId=xxx&jobId=xxx  → 质量校验
/admin/projects             → 项目列表
/admin/projects/[projectId] → 项目详情
```

### 2.4 UI 架构

#### 布局系统
1. **Topbar（顶部横栏）**
   - 左侧: Logo + "一湃智能数据平台"
   - 右侧: 搜索图标 + 用户头像 + 用户名 + 下拉箭头
   - 高度: 60px
   - 固定定位，始终可见

2. **IconRail（最左侧竖栏）**
   - 宽度: 56px
   - 位置: Topbar 下方
   - 功能: 一级导航（概览、数据、管理）
   - 样式: 图标按钮，激活状态有左侧蓝色竖条

3. **SideMenu（第二竖栏）**
   - 宽度: 220px
   - 位置: IconRail 右侧
   - 功能: 二级导航（数据、上传、采集、标注等）
   - 分组: 主功能区（可滚动）+ 系统区（贴底）
   - 样式: 图标 + 文字，激活状态有左侧蓝色竖条

4. **主内容区**
   - 位置: SideMenu 右侧
   - 背景: 极浅灰（#f6f7f9）
   - 可滚动

#### 组件架构
- **页面组件** (`pages/`): 业务逻辑页面
- **通用组件** (`components/`): 可复用 UI 组件
- **布局组件**: 在 `layout.tsx` 中实现

### 2.5 状态管理

#### 状态管理方式
- **本地状态**: 使用 React `useState` 和 `useEffect`
- **无全局状态管理库**: 未使用 Redux、Zustand 等
- **URL 状态**: 使用 Next.js `useSearchParams` 传递参数

#### 数据流
```
用户操作 
  → 组件事件处理 
    → API 调用（Mock）
      → localStorage 读写
        → 状态更新
          → UI 重新渲染
```

### 2.6 样式方案

#### 样式实现
- **内联样式**: 使用 JavaScript 对象（`style={{ ... }}`）
- **无 CSS 文件**: 不使用 CSS Modules、Tailwind 等
- **无主题系统**: 颜色值硬编码在组件中

#### 设计规范
- **背景色**: 极浅灰（#f6f7f9）
- **卡片背景**: 纯白（#ffffff）
- **主色调**: 蓝色（#2563eb）
- **文字颜色**: 深灰（#111827）、中灰（#374151）、浅灰（#6b7280）
- **边框**: 浅灰（#e5e7eb）

---

## 三、后端架构

### 3.1 后端技术栈

#### 核心框架与运行时
- **Web 框架**: FastAPI 0.115.0
- **ASGI 服务器**: Uvicorn
- **Python 版本**: 3.11+
- **异步支持**: Python asyncio + async/await

#### 数据库层
- **数据库**: PostgreSQL 16 (Alpine 镜像)
- **ORM**: SQLAlchemy 2.0 (异步模式)
- **数据库驱动**: asyncpg (PostgreSQL 异步驱动)
- **连接池**: SQLAlchemy 内置连接池管理
- **迁移工具**: Alembic (数据库版本管理)

#### 认证与安全
- **JWT 认证**: python-jose (JWT Token 生成与验证)
- **密码哈希**: passlib[bcrypt] (bcrypt 算法)
- **Token 算法**: HS256
- **Token 过期时间**: 30 分钟（可配置）

#### 数据验证
- **验证框架**: Pydantic v2
- **请求验证**: 自动验证请求体、查询参数
- **响应序列化**: 自动序列化 SQLAlchemy 模型为 JSON

#### 实时通信
- **协议**: WebSocket (ws:// 或 wss://)
- **实现**: FastAPI 内置 WebSocket 支持
- **用途**: 实时进度推送、状态同步

### 3.3 采集端联动与“隧道转发同步”

平台与采集端（Agent）之间有两类通道：

- **控制/日志/预览隧道（WebSocket）**：`/api/agent/tunnel`
  - 平台按 `agent_id` 路由下发 `CMD_REQUEST`（如 `COLLECT_START`/`DEVICE_STOP`/`DATA_SYNC`）
  - 采集端回传 `CMD_ACK`/`CMD_RESULT`，并可通过隧道上报日志、MJPEG 分片等
- **采集端本地 HTTP 服务（FastAPI）**：例如 `/api/agent/data/sync`
  - 仍保留用于调试/兼容，但生产链路建议优先使用隧道

#### 数据同步推荐链路（不依赖采集端 IP）

平台侧“同步到 MinIO”采用 **隧道转发模式**（优先）：

1. 平台确定目标 `agent_id`（来源：资产 meta.collect.hardware_uuid/agent_id 或资产 device_id → devices.hardware_uuid）
2. 平台通过 WebSocket 隧道发送 `CMD_REQUEST cmd=DATA_SYNC`，payload 含 `source_path` 与 MinIO 参数
3. 采集端读取本机 `source_path`，将文件/目录上传到平台 MinIO
4. 采集端通过 `CMD_RESULT` 返回 `minio_path`，平台落库并更新 `sync_status=synced`

该模式避免了多网卡/NAT/动态 IP 下平台侧直连采集端 IP 不稳定的问题；前提是采集端与平台的隧道已连接。

### 3.2 后端架构设计

#### 架构模式
- **分层架构**: API 路由层 → CRUD 业务层 → 数据模型层
- **依赖注入**: FastAPI Depends 机制
- **异步处理**: 全链路异步（async/await）
- **统一响应格式**: 所有 API 返回 `{ok: boolean, data?: T, error?: string}`

#### 数据库连接
- **连接字符串格式**: `postgresql+asyncpg://user:password@host:port/database`
- **默认配置**: `postgresql+asyncpg://eai:eai123@localhost:5432/eai_platform`
- **连接池**: SQLAlchemy 自动管理异步连接池
- **会话管理**: 每个请求独立的 AsyncSession，自动关闭

#### 项目结构
```
backend/
├── app/
│   ├── main.py              # FastAPI 应用入口
│   ├── core/                 # 核心配置
│   │   ├── config.py         # 环境配置
│   │   ├── security.py       # JWT 和密码
│   │   └── deps.py           # 依赖注入（鉴权）
│   ├── db/                   # 数据库
│   │   ├── session.py        # 异步会话
│   │   └── base.py           # Base 模型
│   ├── models/               # SQLAlchemy 模型
│   │   ├── user.py
│   │   ├── task.py
│   │   ├── job.py
│   │   ├── run.py
│   │   └── dataset.py
│   ├── schemas/              # Pydantic 模式
│   │   ├── common.py         # 统一返回格式
│   │   ├── auth.py
│   │   ├── task.py
│   │   ├── job.py
│   │   ├── run.py
│   │   └── dataset.py
│   ├── crud/                 # 数据库操作
│   │   ├── user.py
│   │   ├── task.py
│   │   ├── job.py
│   │   ├── run.py
│   │   └── dataset.py
│   ├── api/                  # API 路由
│   │   ├── router.py         # 路由聚合
│   │   ├── routes_auth.py    # 认证路由
│   │   ├── routes_tasks.py   # 任务路由
│   │   ├── routes_jobs.py    # 作业路由
│   │   ├── routes_runs.py    # 运行路由
│   │   └── routes_datasets.py # 数据集路由
│   └── realtime/             # 实时功能
│       ├── job_ws.py         # WebSocket 管理
│       └── simulator.py      # 模拟采集器
├── alembic/                  # 数据库迁移
│   ├── env.py
│   └── versions/
├── alembic.ini               # Alembic 配置
├── requirements.txt          # Python 依赖
├── Dockerfile                # Docker 镜像
└── README.md                 # 后端文档
```

#### 数据库模型（PostgreSQL）

**表结构设计**:
1. **users（用户表）**
   - `id`: UUID (主键)
   - `username`: VARCHAR (唯一索引)
   - `password_hash`: VARCHAR (bcrypt 哈希)
   - `role`: ENUM (ADMIN/OPERATOR/QC)
   - `created_at`: TIMESTAMP

2. **tasks（任务表）**
   - `id`: UUID (主键)
   - `name`: VARCHAR
   - `description`: TEXT
   - `status`: ENUM (DRAFT/READY/RUNNING/COMPLETED/ARCHIVED)
   - `created_at`: TIMESTAMP
   - `updated_at`: TIMESTAMP

3. **jobs（作业表）**
   - `id`: UUID (主键)
   - `task_id`: UUID (外键 → tasks.id)
   - `status`: ENUM (PENDING/RUNNING/SUCCEEDED/FAILED/CANCELED)
   - `operator_name`: VARCHAR
   - `mcap_path`: VARCHAR (文件路径)
   - `mcap_size_bytes`: BIGINT
   - `duration_sec`: INTEGER
   - `progress`: INTEGER (0-100)
   - `started_at`: TIMESTAMP
   - `finished_at`: TIMESTAMP
   - `created_at`: TIMESTAMP
   - `updated_at`: TIMESTAMP

4. **runs（运行表）**
   - `id`: UUID (主键)
   - `task_id`: UUID (外键 → tasks.id)
   - `status`: ENUM (QUEUED/RUNNING/SUCCEEDED/FAILED/CANCELED)
   - `created_at`: TIMESTAMP
   - `updated_at`: TIMESTAMP

5. **datasets（数据集表）**
   - `id`: UUID (主键)
   - `name`: VARCHAR
   - `status`: ENUM (ACTIVE/ARCHIVED)
   - `created_at`: TIMESTAMP
   - `updated_at`: TIMESTAMP

**数据库特性**:
- **字符集**: UTF-8
- **时区**: UTC
- **索引**: 主键自动索引，外键自动索引，username 唯一索引
- **约束**: 外键约束、非空约束、枚举约束
- **事务**: 支持 ACID 事务（通过 SQLAlchemy 会话管理）
- **连接管理**: 异步连接池，自动管理连接生命周期

#### 数据库配置

**连接配置**:
- **默认连接字符串**: `postgresql+asyncpg://eai:eai123@localhost:5432/eai_platform`
- **环境变量**: `DATABASE_URL` (可通过 `.env` 文件配置)
- **连接池配置**: SQLAlchemy 默认连接池（可自定义大小、超时等）

**Docker 部署配置**:
```yaml
postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_USER: eai
    POSTGRES_PASSWORD: eai123
    POSTGRES_DB: eai_platform
  ports:
    - "5432:5432"
  volumes:
    - postgres_data:/var/lib/postgresql/data
```

**数据库迁移**:
- **工具**: Alembic
- **迁移文件位置**: `backend/alembic/versions/`
- **配置文件**: `backend/alembic.ini`
- **命令**:
  - 创建迁移: `alembic revision --autogenerate -m "描述"`
  - 应用迁移: `alembic upgrade head`
  - 回滚迁移: `alembic downgrade -1`

#### API 路由
- **认证**: `/api/auth/login`, `/api/auth/me`
- **任务**: `/api/tasks` (GET, POST, GET/{id}, PATCH/{id}, DELETE/{id})
- **作业**: `/api/jobs` (GET, POST, GET/{id}, PATCH/{id}, POST/{id}/start, POST/{id}/cancel, POST/{id}/finish)
- **运行**: `/api/runs` (GET, POST, GET/{id}, PATCH/{id}, DELETE/{id})
- **数据集**: `/api/datasets` (GET, POST, GET/{id}, PATCH/{id}, DELETE/{id})
- **WebSocket**: `/api/ws/jobs/{job_id}` (实时进度推送)
- **健康检查**: `/health` (无需认证)

#### 统一返回格式
所有 REST API 返回统一格式：
```json
{
  "ok": true,
  "data": { ... }
}
```
或
```json
{
  "ok": false,
  "error": "错误信息"
}
```

#### 认证机制
- **JWT Token**: 使用 Bearer Token 认证
- **Token 获取**: 通过 `/api/auth/login` 获取
- **Token 使用**: 在请求头中添加 `Authorization: Bearer <token>`
- **保护范围**: 除 `/health` 和 `/api/auth/login` 外，所有接口都需要认证

#### 实时功能
- **WebSocket 连接**: `/api/ws/jobs/{job_id}`
- **进度推送**: 每 500ms 推送一次进度更新
- **模拟采集器**: 后台 asyncio 任务模拟采集进度（每 500ms +2%，直到 100%）
- **状态同步**: 通过 WebSocket 实时同步作业状态（RUNNING → SUCCEEDED/CANCELED）

#### 部署方式

**1. Docker Compose 部署（推荐生产环境）**
```bash
docker compose up -d
```
- **服务**: 同时启动 PostgreSQL 16 和 FastAPI Backend
- **端口映射**: 
  - PostgreSQL: `5432:5432`
  - Backend API: `8000:8000`
- **数据持久化**: PostgreSQL 数据存储在 Docker Volume (`postgres_data`)
- **健康检查**: PostgreSQL 健康检查，Backend 等待数据库就绪后启动
- **自动迁移**: 容器启动时自动运行 `alembic upgrade head`
- **默认用户**: 自动创建 admin 用户（用户名: `admin`, 密码: `admin123`）

**2. 本地开发部署**
```bash
# 1. 创建虚拟环境
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（可选，使用 .env 文件）
# DATABASE_URL=postgresql+asyncpg://eai:eai123@localhost:5432/eai_platform
# SECRET_KEY=your-secret-key
# ALGORITHM=HS256
# ACCESS_TOKEN_EXPIRE_MINUTES=30

# 4. 启动 PostgreSQL（如果未运行）
docker run -d --name eai-postgres \
  -e POSTGRES_USER=eai \
  -e POSTGRES_PASSWORD=eai123 \
  -e POSTGRES_DB=eai_platform \
  -p 5432:5432 \
  postgres:16-alpine

# 5. 运行数据库迁移
alembic upgrade head

# 6. 启动开发服务器（自动重载）
uvicorn app.main:app --reload --port 8000
```

**3. 生产环境独立部署**
- **前端**: Next.js 构建后部署到 Nginx 或 Node.js 服务器
- **后端**: FastAPI 部署到 ASGI 服务器（Uvicorn/Gunicorn + Uvicorn Workers）
- **数据库**: PostgreSQL 独立部署，配置主从复制（可选）
- **反向代理**: Nginx 作为反向代理，处理 HTTPS、负载均衡

### 3.3 前端 API 客户端

#### API 层结构
```
src/features/
├── daq-editor/api/          # 采集编辑器适配层
│   ├── taskApi.ts          # 任务 API（适配 DaqTask）
│   └── jobApi.ts           # 作业 API（适配 Job）
│
└── data-platform/api/      # 数据平台统一 API 客户端
    ├── client.ts           # HTTP 客户端（统一请求处理）
    ├── authApi.ts          # 认证 API
    ├── taskApi.ts          # 任务 API
    ├── runApi.ts           # 运行 API
    ├── jobApi.ts           # 作业 API
    ├── datasetApi.ts       # 数据集 API
    └── index.ts            # API 导出
```

#### API 客户端设计
- **基础 URL**: 通过环境变量 `NEXT_PUBLIC_API_URL` 配置（默认 `http://localhost:8000`）
- **认证**: 自动从 localStorage 读取 JWT Token，添加到请求头
- **统一返回格式**: 与后端一致
```typescript
interface ApiResponse<T> {
  ok: boolean;
  data?: T;
  error?: string;
}
```

#### Token 管理
- **存储**: localStorage (`auth_token`)
- **获取**: `getAuthToken()` - 自动从 localStorage 读取
- **设置**: `setAuthToken(token)` - 登录后保存 Token
- **清除**: `clearAuthToken()` - 登出时清除 Token

### 3.4 数据模型

#### 核心实体（前后端一致）

**1. Task（任务）**
- **用途**: 数据采集任务定义
- **状态枚举**: DRAFT / READY / RUNNING / COMPLETED / ARCHIVED
- **关键字段**: id, name, description, status, deviceId, deviceName
- **前端模型**: 
  - `src/features/data-platform/models/task.ts` (统一模型)
  - `src/features/daq-editor/models/types.ts` (DaqTask 适配模型)
- **后端模型**: 
  - `backend/app/models/task.py` (SQLAlchemy 模型)
  - `backend/app/schemas/task.py` (Pydantic 模式)

**2. Job（作业）**
- **用途**: 采集作业执行记录
- **状态枚举**: PENDING / RUNNING / SUCCEEDED / FAILED / CANCELED
- **关键字段**: id, taskId, status, operatorName, progress, deviceId, deviceName, collectionQuantity
- **关联关系**: 多对一 (Job → Task)
- **前端模型**: 
  - `src/features/data-platform/models/job.ts` (统一模型)
  - `src/features/daq-editor/models/job.ts` (采集编辑器模型)
- **后端模型**: 
  - `backend/app/models/job.py` (SQLAlchemy 模型)
  - `backend/app/schemas/job.py` (Pydantic 模式)

**3. Run（运行）**
- **用途**: 任务执行记录
- **状态枚举**: QUEUED / RUNNING / SUCCEEDED / FAILED / CANCELED
- **关键字段**: id, taskId, status, artifact (文件信息)
- **关联关系**: 多对一 (Run → Task)
- **前端模型**: `src/features/data-platform/models/run.ts`
- **后端模型**: 
  - `backend/app/models/run.py` (SQLAlchemy 模型)
  - `backend/app/schemas/run.py` (Pydantic 模式)

**4. Dataset（数据集）**
- **用途**: 数据集管理
- **状态枚举**: ACTIVE / ARCHIVED
- **关键字段**: id, name, status
- **关联关系**: 多对多 (Dataset ↔ Run)
- **前端模型**: `src/features/data-platform/models/dataset.ts`
- **后端模型**: 
  - `backend/app/models/dataset.py` (SQLAlchemy 模型)
  - `backend/app/schemas/dataset.py` (Pydantic 模式)

**5. User（用户）**
- **用途**: 用户认证和权限管理
- **角色枚举**: ADMIN / OPERATOR / QC
- **关键字段**: id, username (唯一), password_hash, role
- **前端模型**: 无（仅后端）
- **后端模型**: 
  - `backend/app/models/user.py` (SQLAlchemy 模型)
  - `backend/app/schemas/auth.py` (Pydantic 模式)

**6. RobotDevice（设备）**
- **用途**: 机器人设备管理（前端 localStorage，未来可迁移到后端）
- **驱动类型**: ROS2 / OPCUA / PLCNEXT / HTTP / MOCK
- **状态枚举**: DISCONNECTED / CONNECTING / CONNECTED / ERROR
- **前端模型**: `src/features/data-platform/models/device.ts`
- **存储**: localStorage (`eai_devices_v1`)

#### 模型定义位置
- **前端统一模型**: `src/features/data-platform/models/` - 与后端一致的模型定义
- **前端适配模型**: `src/features/daq-editor/models/` - 采集编辑器专用模型（适配层）
- **后端数据模型**: `backend/app/models/` - SQLAlchemy ORM 模型（数据库映射）
- **后端 API 模式**: `backend/app/schemas/` - Pydantic 模式（请求/响应验证）

#### 模型同步策略
- **手动维护**: 前后端模型需要手动保持同步
- **类型安全**: TypeScript + Pydantic 提供端到端类型检查
- **建议**: 使用代码生成工具或 OpenAPI 规范自动生成类型定义

---

## 四、数据流分析

### 4.1 前端数据流

#### 数据读取流程
```
页面加载
  → 组件 mount
    → useEffect 调用 API
      → fetch() 发送 HTTP 请求
        → 后端 FastAPI 处理
          → SQLAlchemy 查询数据库
            → 返回 JSON 响应
              → 设置组件 state
                → 渲染 UI
```

#### 数据写入流程
```
用户操作（创建/更新/删除）
  → 调用 API 函数
    → apiPost/apiPatch/apiDelete（统一客户端）
      → fetch() 发送 HTTP 请求（自动添加 JWT Token）
        → 后端 FastAPI 处理
          → 验证 Token
            → SQLAlchemy 操作数据库
              → 返回 JSON 响应
                → 更新组件 state
                  → UI 重新渲染
```

#### 实时数据流（WebSocket）
```
用户启动作业
  → POST /api/jobs/{id}/start
    → 后端启动模拟采集器
      → 每 500ms 更新进度
        → WebSocket 推送进度
          → 前端接收消息
            → 更新 UI（进度条、状态）
```

### 4.2 前后端通信

#### REST API 通信
- **协议**: HTTP/HTTPS
- **方法**: GET, POST, PATCH, DELETE
- **数据格式**: JSON (Content-Type: application/json)
- **认证方式**: Bearer Token (JWT)
  - 请求头: `Authorization: Bearer <token>`
  - Token 获取: `POST /api/auth/login`
  - Token 存储: localStorage (`auth_token`)
- **Base URL**: 
  - 开发环境: `http://localhost:8000`
  - 生产环境: 通过环境变量 `NEXT_PUBLIC_API_URL` 配置
- **统一响应格式**:
  ```typescript
  interface ApiResponse<T> {
    ok: boolean;
    data?: T;
    error?: string;
  }
  ```

#### WebSocket 通信
- **协议**: WebSocket (ws:// 或 wss://)
- **连接路径**: `/api/ws/jobs/{job_id}`
- **数据格式**: JSON
- **用途**: 
  - 实时进度推送（每 500ms 推送一次）
  - 作业状态同步（RUNNING → SUCCEEDED/CANCELED）
- **消息格式**:
  ```json
  {
    "type": "progress",
    "jobId": "uuid",
    "status": "RUNNING",
    "progress": 50
  }
  ```
- **连接管理**: 
  - 前端自动重连（需要实现）
  - 后端支持多客户端连接同一作业

#### 跨组件数据同步
- **WebSocket**: 实时同步作业进度（推荐）
- **轮询**: 可定期轮询 API 获取最新状态（备用方案）
- **事件驱动**: 通过 WebSocket 事件更新 UI
- **状态管理**: 
  - 当前无全局状态管理库
  - 组件间数据不同步（需要手动刷新或重新调用 API）
  - 建议使用 Context API 或 Zustand 实现全局状态

#### 改进建议
- 使用 Context API 或状态管理库实现全局状态
- 统一 API 客户端（支持 Mock 和 Backend 切换）
- 实现 WebSocket 连接管理（重连、错误处理）

---

## 五、构建与部署

### 5.1 构建配置

#### Next.js 配置
- **模式**: App Router（Next.js 15）
- **React Strict Mode**: 关闭
- **Webpack 配置**: 自定义（修复 lucide-react 兼容性）

#### TypeScript 配置
- **严格模式**: 部分开启（`strictNullChecks: true`）
- **模块解析**: `bundler`（Next.js 推荐）
- **路径别名**: `@/*` → `./src/*`

### 5.2 开发流程

#### 开发命令
```bash
pnpm dev          # 启动开发服务器（端口 3000）
pnpm build        # 构建生产版本
pnpm start        # 启动生产服务器
```

#### 脚本工具
- `scripts/restart.sh` - 重启服务
- `scripts/start.sh` - 启动服务
- `scripts/stop.sh` - 停止服务

### 5.3 质量门禁

#### Makefile 命令
```bash
make test    # 类型检查（tsc --noEmit）
make eval    # 回归测试（需要 Playwright）
make report  # 生成路由和模型报告
make clean   # 清理 artifacts
```

#### 报告生成
- **路由报告**: `artifacts/report/routes.md`
- **模型快照**: `artifacts/report/models.schema.json`

---

## 六、架构特点总结

### 6.1 优势

1. **前后端分离**
   - 清晰的架构边界
   - 独立部署和扩展
   - 前后端统一 API 接口

2. **简单直接**
   - 无复杂状态管理，代码易理解
   - 内联样式，无需管理 CSS 文件
   - 统一的 API 客户端

3. **快速迭代**
   - Next.js App Router 提供良好的开发体验
   - TypeScript 提供类型安全
   - 热重载支持快速开发
   - FastAPI 自动生成 API 文档

4. **模块化设计**
   - Feature-based 目录结构
   - 组件职责清晰
   - API 层统一接口
   - 前后端模型一致

5. **实时功能**
   - WebSocket 实时进度推送
   - 模拟采集器后台任务
   - 状态自动同步

### 6.2 局限性

1. **状态管理**
   - 无全局状态，组件间数据不同步
   - 需要手动刷新获取最新数据

2. **样式管理**
   - 内联样式难以维护
   - 无主题系统，颜色值硬编码
   - 样式重复代码多

3. **前后端集成**
   - 需要确保后端服务运行
   - 需要处理认证 Token 管理
   - 需要处理网络错误和重试

### 6.3 技术债务

1. **API 层重复**
   - `daq-editor/api` 和 `data-platform/api` 存在功能重叠
   - `daq-editor/api` 主要是适配层，可考虑优化

2. **模型定义分散**
   - 前端模型定义在多个位置
   - 建议统一到 `data-platform/models`
   - 前后端模型需要保持同步（手动维护）

3. **样式系统**
   - 建议引入 CSS-in-JS 或 Tailwind CSS
   - 建立设计系统（颜色、间距等常量）

4. **前后端集成**
   - WebSocket 连接管理需要完善（重连、错误处理）
   - 认证 Token 管理需要统一（存储、刷新、过期处理）
   - 需要添加请求重试和错误处理机制

---

## 七、未来改进方向

### 7.1 短期改进

1. **统一 API 层**
   - 优化 `daq-editor/api` 适配层
   - 统一接口规范

2. **引入样式系统**
   - 创建样式常量文件
   - 或引入 Tailwind CSS

3. **状态管理优化**
   - 使用 Context API 管理全局状态
   - 或引入轻量级状态管理库（Zustand）

4. **前后端集成增强**
   - 完善 WebSocket 连接管理（重连、错误处理）
   - 实现 Token 自动刷新
   - 添加请求重试机制
   - 统一错误处理

### 7.2 长期规划

1. **后端服务增强**
   - 实现文件上传功能
   - 支持真实视频流处理
   - 实现更复杂的业务逻辑
   - 添加缓存层（Redis）

2. **实时功能增强**
   - WebSocket 支持更多实时数据推送
   - 实时采集视频流集成
   - 多用户协作支持

3. **性能优化**
   - 代码分割和懒加载
   - 图片优化
   - 缓存策略（前端 + 后端）
   - 数据库查询优化

4. **部署和运维**
   - CI/CD 流水线
   - 容器化部署
   - 监控和日志系统
   - 自动化测试

---

## 八、关键文件清单

### 核心配置文件
- `package.json` - 前端依赖和脚本
- `next.config.js` - Next.js 配置
- `tsconfig.json` - TypeScript 配置
- `Makefile` - 构建和测试命令
- `docker-compose.yml` - Docker 编排（Redis + `app`：前后端同容器 + `worker`；见 `docs/docker-split-deploy.md`）
- `backend/requirements.txt` - Python 依赖
- `backend/alembic.ini` - Alembic 配置

### 前端核心代码文件
- `src/app/(platform)/layout.tsx` - 平台布局（Topbar + Sidebar）
- `src/features/data-platform/bootstrap/ensureSeeded.ts` - 数据初始化（Mock 模式）
- `src/features/data-platform/storage/seed.ts` - Seed 数据生成（Mock 模式）
- `src/features/data-platform/api/*.ts` - 统一 API 接口（Mock 模式）

### 后端核心代码文件
- `backend/app/main.py` - FastAPI 应用入口
- `backend/app/core/config.py` - 环境配置
- `backend/app/core/security.py` - JWT 和密码
- `backend/app/core/deps.py` - 依赖注入（鉴权）
- `backend/app/db/session.py` - 数据库会话
- `backend/app/models/*.py` - SQLAlchemy 模型
- `backend/app/schemas/*.py` - Pydantic 模式
- `backend/app/crud/*.py` - 数据库操作
- `backend/app/api/*.py` - API 路由
- `backend/app/realtime/*.py` - WebSocket 实时功能
- `backend/alembic/versions/001_initial.py` - 初始数据库迁移

### 文档文件
- `docs/architecture.md` - 本文档
- `docs/backend-api.md` - 后端 API 文档
- `docs/dev/ui-shell-ref.md` - UI Shell 组件定位
- `docs/dev/collect-ui.md` - 采集流程 UI 文档
- `docs/regression/*.md` - 回归测试用例
- `backend/README.md` - 后端使用说明

---

## 九、总结

### 架构类型
**前后端分离架构：Next.js 前端 + FastAPI 后端 + PostgreSQL 数据库**

#### 架构模式
- **前端**: Next.js SPA (Single Page Application)
- **后端**: FastAPI RESTful API + WebSocket
- **数据库**: PostgreSQL 16 (关系型数据库)
- **通信协议**: HTTP/HTTPS (REST API) + WebSocket (实时通信)
- **数据格式**: JSON

### 技术选型

#### 前端技术栈
- **框架**: Next.js 15.5.3 (App Router)
- **UI 库**: React 19.1.0
- **语言**: TypeScript 5
- **包管理器**: pnpm
- **图标库**: lucide-react 0.563.0
- **数据存储**: 
  - Mock 模式: localStorage
  - 生产模式: HTTP REST API + WebSocket

#### 后端技术栈
- **Web 框架**: FastAPI 0.115.0
- **ASGI 服务器**: Uvicorn
- **Python 版本**: 3.11+
- **ORM**: SQLAlchemy 2.0 (异步模式)
- **数据库**: PostgreSQL 16
- **数据库驱动**: asyncpg (异步 PostgreSQL 驱动)
- **认证**: JWT (python-jose)
- **密码哈希**: passlib[bcrypt]
- **数据验证**: Pydantic v2
- **实时通信**: WebSocket (FastAPI 内置)
- **数据库迁移**: Alembic

#### 数据库技术栈
- **数据库**: PostgreSQL 16 (Alpine 镜像)
- **连接方式**: 异步连接（asyncpg）
- **连接池**: SQLAlchemy 异步连接池
- **迁移工具**: Alembic
- **数据持久化**: Docker Volume 或本地文件系统

### 适用场景

#### Mock 模式
- ✅ 原型开发
- ✅ 演示和展示
- ✅ 前端开发（无需后端）
- ✅ 离线使用
- ❌ 生产环境
- ❌ 多用户协作
- ❌ 大数据量处理

#### 后端模式
- ✅ 生产环境
- ✅ 多用户协作
- ✅ 真实数据持久化
- ✅ 实时功能（WebSocket）
- ✅ 用户认证和权限
- ✅ 大数据量处理
- ❌ 需要部署和维护后端服务

### 部署方式

#### 开发环境
```bash
# 前端
pnpm dev

# 后端（可选）
cd backend
uvicorn app.main:app --reload --port 8000
```

#### 生产环境
```bash
# Docker Compose（推荐）
docker compose up -d

# 或分别部署
# - 前端：Next.js 构建后部署到静态服务器或 Node.js 服务器
# - 后端：FastAPI 部署到 ASGI 服务器（Uvicorn/Gunicorn）
# - 数据库：PostgreSQL 独立部署
```

### 架构优势
1. **灵活性**: 支持 Mock 和 Backend 两种模式，易于开发和部署
2. **一致性**: 前后端统一 API 接口和数据模型
3. **可扩展**: 模块化设计，易于添加新功能
4. **实时性**: WebSocket 支持实时数据推送
5. **类型安全**: TypeScript + Pydantic 提供端到端类型检查

