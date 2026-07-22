# 一湃智能数据平台 - 架构概览

本文档为 v0.1 发布配套的架构概要，便于快速理解系统分层与数据流。更细的模块与代码结构见 [architecture.md](./architecture.md)。

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        浏览器 (Browser)                            │
└──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  前端 (Next.js 15 + React 19)  ·  Port 3001 / 3000                 │
│  App Router │ 平台布局 │ 数据/采集/标注/转换/管理  │ 任务中心       │
└──────────────────────────────────────────────────────────────────┘
                                    │ HTTP/HTTPS (REST + 可选 WebSocket)
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  后端 (FastAPI)  ·  Port 8000                                     │
│  认证(JWT) │ 任务/作业/项目/设备/数据资产/标注/转换/审计            │
└──────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ SQLite/   │   │ 文件存储   │   │ 外部服务   │
            │ PostgreSQL│   │ HDF5/MCAP │   │ OpenAI API│
            └───────────┘   └───────────┘   └───────────┘
```

- **前端**：单页应用，路由与权限由 Next.js App Router 与 layout 守卫控制；API 请求走 REST，部分实时能力（如采集日志）走 WebSocket。
- **后端**：FastAPI 提供 REST API，JWT 鉴权；任务/作业/项目/用户等存关系库，数据资产与 HDF5 元数据可走 SQLite 或配置的 DB；自动标注调用项目根目录 `label_task_description.py` 与 OpenAI 兼容接口。

---

## 2. 前端分层

| 层级 | 位置 | 职责 |
|------|------|------|
| 路由与布局 | `src/app/` | 登录、平台 layout、各业务页与动态路由 |
| 全局组件 | `src/components/` | 布局、侧栏、仪表盘、任务中心、通用确认/Toast |
| 业务特性 | `src/features/` | 采集(daq-editor)、标注(label-runner)、数据平台 API/组件(data-platform) |
| 公共逻辑 | `src/lib/` | i18n、项目/任务服务、仪表盘聚合 |
| 状态 | `src/store/` | 登录(authStore)、语言(localeStore)、采集会话(collectSessionStore) |

- 数据请求统一通过 `features/data-platform/api` 或各 feature 下的 api 发往后端；认证 token 由 authStore 与 HTTP 客户端自动携带。

---

## 3. 后端分层

| 层级 | 位置 | 职责 |
|------|------|------|
| 入口与中间件 | `app/main.py` | FastAPI 应用、CORS、全局异常、生命周期（建表等） |
| 路由 | `app/api/`, `app/routes/` | 认证、任务、作业、项目、设备、数据资产、标注、转换、审计等 |
| 服务 | `app/services/` | 业务逻辑（HDF5、MCAP、标注、转换、审计等） |
| 数据访问 | `app/crud/` | 对模型层的增删改查 |
| 模型与 schema | `app/models/`, `app/schemas/` | ORM 模型与 Pydantic 请求/响应 |

- 数据库：主业务库（SQLAlchemy 异步）；HDF5 元数据/设备等可为 SQLite，路径可配置。

---

## 4. 认证与权限

- 登录：`POST /api/auth/login` 返回 access_token（及 refresh）；前端将 token 存 store 并写入请求头。
- 需认证的接口：请求头 `Authorization: Bearer <access_token>`。
- 前端根据 `me.role` 与 `navItems` 控制侧栏与入口；后端按角色限制项目/用户/审计等访问。

---

## 5. 关键数据流简述

- **采集**：前端创建任务 → 后端持久化 → 作业中心认领/启停 → 实时页或脚本产生数据 → 可登记为数据资产。
- **标注**：前端创建标注任务、选择数据集 → 执行页加载 Episode/帧 → 保存描述走标注 API；自动描述由后端调 `label_task_description.py` + OpenAI 兼容 API。
- **转换**：前端选择项目与资产、创建转换任务；当前前端 Mock 进度与日志，后端可扩展真实转换流水线。
- **数据资产**：导入/注册后写入后端；列表/筛选/导出/删除均走数据资产 API。

更细的接口与流程见 [backend-api.md](./backend-api.md) 与 [architecture.md](./architecture.md)。
