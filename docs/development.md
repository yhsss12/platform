# 一湃智能数据平台 - 开发指南

本文档面向开发人员，说明本地环境、启动方式、目录约定与常用脚本，便于参与 v0.1 及后续版本的开发与调试。

---

## 1. 环境要求

| 项目 | 版本/要求 |
|------|-----------|
| Node.js | 18+ |
| pnpm | 8+ |
| Python | 3.10+ |
| 后端依赖 | 见 `backend/requirements.txt` |

---

## 2. 前端

### 2.1 安装与启动

```bash
# 在项目根目录 eai-ide/
pnpm install
pnpm run dev        # 默认端口 3001
pnpm run dev:3000   # 使用 3000 端口
```

### 2.2 构建与生产

```bash
pnpm run build      # 生产构建（含类型检查与 Lint）
pnpm run start      # 生产模式启动（默认 3000）
```

### 2.3 环境变量

- 前端通过 `next.config.js` 或环境变量读取配置；常用：
  - `NEXT_PUBLIC_API_URL`：后端 API 根地址（如 `http://localhost:8000`），用于服务端或直连场景。
  - `NEXT_PUBLIC_EXPERIMENT_ENABLED`：是否启用浏览器侧实验模式与实验事件上报；默认关闭。
- 开发模式下 Next.js 可通过 rewrites 将 `/api/*` 代理到后端，此时可不设 `NEXT_PUBLIC_API_URL`，以相对路径请求。

### 2.4 目录约定（src/）

- **app/**：Next.js App Router；`(platform)` 下为登录后的平台布局与各业务路由。
- **components/**：全局共享组件（布局、仪表盘、任务中心、通用 UI）。
- **features/**：按业务划分的特性模块（daq-editor 采集、label-runner 标注、data-platform API/组件）。
- **lib/**：公共逻辑（i18n、项目、任务、仪表盘聚合等）。
- **store/**：Zustand 全局状态（登录、语言、采集会话等）。

---

## 3. 后端

### 3.1 安装与启动

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 配置 .env（见下方）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3.2 环境变量（.env）

- 在 `backend/` 或项目根目录放置 `.env`，后端启动时会加载（参见 `app/main.py` 顶部）。
- 常用变量示例（具体以 `backend/.env.example` 或部署文档为准）：
  - **认证**：`SECRET_KEY`、JWT 相关（如 access/refresh 过期时间）。
  - **数据库**：`DATABASE_URL`（若用 PostgreSQL）、SQLite 路径等。
  - **数据资产/存储**：`DATA_ASSETS_DB_PATH`、`HDF5_DATA_DIR` 等。
  - **自动标注**：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`（供项目根目录 `label_task_description.py` 及标注服务调用）。
  - **实验子系统**：`EXPERIMENT_ENABLED=true|false`，关闭后 `/api/experiment/*` 与后端实验埋点都会静默。

### 3.3 数据库与迁移

- 任务/作业等使用 SQLAlchemy + 异步引擎；启动时自动建表（见 `app.main` lifespan）。
- 设备、HDF5 元数据等可使用 SQLite，路径由配置指定。
- 若有 Alembic 迁移，执行：`alembic upgrade head`（在 backend 目录下）。

### 3.4 初始化管理员

- 使用 `backend/tools/database/init_user.py` 或文档说明的脚本创建初始管理员账号（如 `admin`）。

---

## 4. 标注与自动描述

- 自动描述依赖项目根目录下的 **label_task_description.py**（HDF5/MCAP 读帧 + OpenAI 兼容 API）。
- 后端 `hdf5_service` 会将项目根加入 `sys.path` 并导入 `label_task_description`；需保证：
  - 后端进程的当前工作目录或 Python 路径能访问到项目根目录的 `label_task_description.py`；
  - `.env` 中已配置 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`（或由前端 API 配置传入）。

---

## 5. 脚本与运维

- **scripts/**：各类运维、转换、测试脚本（如 `restart.sh`、`start.sh`、`stop.sh`）。
- 脚本若依赖 Python，请在 backend 虚拟环境中运行或显式指定解释器。
- 服务管理与故障排查详见 [scripts/README.md](../scripts/README.md)。

### 5.1 开发 FAQ：3001 页面 HTTP 500/502

| 项 | 说明 |
|----|------|
| **现象** | 访问 `:3001` 页面（如 `/workspace/data`）返回 500/502 |
| **典型日志** | `Cannot find module './xxxx.js'`、`reading '/_app'`、`.next/server/vendor-chunks` 缺失（见 `logs/frontend.log`） |
| **原因** | Next.js dev 的 `.next` 缓存损坏或 HMR chunk 不同步；**不是**后端 API 或 `runtime_outputs` 清理导致 |
| **处理** | `./scripts/restart.sh frontend`（清 `.next` / `node_modules/.cache` / `.turbo`，重启后自动验收 5 次 200） |
| **预防** | `next.config.js` dev 模式已关闭 webpack 持久化 cache |
| **验证** | `curl -i http://127.0.0.1:3001/workspace/data` 与 `curl -i http://127.0.0.1:8000/health` 均应 200 |
| **勿做** | 勿为排查此问题恢复 `runtime_outputs/phygen_*` 等已清理运行产物 |

---

## 6. 代码与发布约定

- 不做未达成一致的破坏性修改；对外行为变更应在发布说明中体现。
- 新增功能建议同步更新 [用户指南](./user-guide.md) 或 [产品概述](./product-overview.md)；API 变更更新 [后端 API](./backend-api.md)。
- 版本号遵循语义化版本；v0.1 发布说明见 [RELEASE_NOTES_v0.1.md](../RELEASE_NOTES_v0.1.md)。

---

## 7. 文档索引

| 文档 | 说明 |
|------|------|
| [README.md](../README.md) | 项目总览与快速开始 |
| [product-overview.md](./product-overview.md) | 产品定位与模块 |
| [user-guide.md](./user-guide.md) | 用户操作指南 |
| [architecture.md](./architecture.md) | 系统与代码架构 |
| [backend-api.md](./backend-api.md) | 后端 API 参考 |
