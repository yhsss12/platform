# 一湃智能数据平台文档索引

本目录已完成精简，仅保留当前版本仍有维护价值的核心文档。

## 推荐阅读顺序

1. [../README.md](../README.md) - 项目总览与启动方式
2. [product-overview.md](./product-overview.md) - 产品能力与角色边界
3. [architecture-overview.md](./architecture-overview.md) - 系统全景
4. [development.md](./development.md) - 开发与联调
5. [backend-api.md](./backend-api.md) - 后端接口参考
6. [user-guide.md](./user-guide.md) - 业务操作说明

## 文档分层

### 1) 产品与使用
- [product-overview.md](./product-overview.md)
- [user-guide.md](./user-guide.md)

### 2) 架构与研发
- [architecture-overview.md](./architecture-overview.md)
- [architecture.md](./architecture.md)
- [development.md](./development.md)
- [backend-api.md](./backend-api.md)

### 3) 专项与迁移
- [minio-direct-upload.md](./minio-direct-upload.md)
- [HDF5.md](./HDF5.md)
- [SQLite-to-PostgreSQL-Migration-Guide.md](./SQLite-to-PostgreSQL-Migration-Guide.md)

### 4) 历史记录
- [WORKLOG.md](./WORKLOG.md)
- `audits/`：带时间或模块范围的审计报告。
- `plans/`：迁移与修正实施方案。
- `incidents/`：历史问题分析纪要。
- `reports/`：测试、验收和修复结果报告。

## 维护约定

- 新增文档优先并入上述核心文档，避免碎片化专题文档泛滥。
- 临时排障/回归记录建议放在 issue 或 PR，不再长期沉淀到 `docs/`。
- 目录内文档默认视为“长期维护文档”，提交前请确认是否具备长期价值。
