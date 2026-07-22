# 一湃智能数据平台 (Yipai Intelligent Data Platform)

企业级数据闭环平台，覆盖数据导入、采集、标注、转换、项目与用户权限管理。

## 核心能力

- 数据资产管理（HDF5/MCAP/LeRobot）
- 采集任务与设备联动
- 标注任务与审核流程
- 数据格式转换与后台任务中心
- 多角色权限与多语言界面（中文/英文/瑞典语）

## 快速开始

### 环境要求

- Node.js 18+
- pnpm 8+
- Python 3.10+
- Docker（推荐）

### 容器构建与启动（示例）

```bash
# 先修改.env里面的IP地址，需要填写服务器实际的IP地址

# 在项目根目录执行
docker compose -f docker-compose.postgres-minio.yml up -d --build
docker compose -f docker-compose.yml up -d --build
```

默认账号、初始化脚本与环境变量配置请参考 `backend/scripts/` 与 `.env`。

## 项目结构

```text
eai-idev2.1/
├── src/            # 前端 (Next.js + React + TypeScript)
├── backend/        # 后端 (FastAPI + SQLAlchemy)
├── docs/           # 精简后的长期维护文档
├── scripts/        # 运维/集成脚本
├── Dockerfile.*    # 容器镜像构建文件
└── README.md
```

## 技术栈

| 层级 | 技术 |
|---|---|
| 前端 | Next.js、React、TypeScript、Zustand |
| 后端 | FastAPI、SQLAlchemy、Pydantic |
| 基础设施 | PostgreSQL、MinIO、Redis |

## 文档入口

- 总索引：[`docs/README.md`](./docs/README.md)
- 产品说明：[`docs/product-overview.md`](./docs/product-overview.md)
- 用户指南：[`docs/user-guide.md`](./docs/user-guide.md)
- 开发文档：[`docs/development.md`](./docs/development.md)
- 后端 API：[`docs/backend-api.md`](./docs/backend-api.md)
- 架构文档：[`docs/architecture-overview.md`](./docs/architecture-overview.md)、[`docs/architecture.md`](./docs/architecture.md)

## 说明

- 本仓库已对 `docs/` 做过一次清理，移除了大量临时排障/回归类文档，仅保留长期维护文档。
- 如需新增文档，建议优先更新现有核心文档，减少碎片化文档数量。
