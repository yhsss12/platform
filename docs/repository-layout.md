# 仓库目录结构

本仓库按“平台核心、任务集成、实验研究、第三方依赖、配置、运维与文档”分层。目录迁移期间优先保持现有前后端入口稳定，避免一次性改动所有启动链路。

## 目标结构

```text
.
├── apps/                  # 最终的平台应用边界（web / api，后续阶段迁移）
├── integrations/          # 平台支持的机器人任务与外部能力集成
├── experiments/           # 不参与平台生产运行的研究代码和实验记录
│   └── phygen/
│       └── runs/          # PhyGen 独立实验目录
├── third_party/           # vendored 或独立检出的第三方源码
├── configs/               # 仓库级、任务模板及运行配置
├── scripts/               # 服务、维护、验证和历史脚本
├── deploy/                # systemd、容器和现场部署配置
├── tests/                 # 跨应用测试；应用内单元测试可暂留原目录
├── docs/                  # 架构、开发、使用和审计文档
└── backups/               # 需要随仓库保留的历史备份
```

## 当前稳定入口

在 `apps/` 迁移完成前，以下入口保持不变：

- `src/`、`public/`：Next.js 前端。
- `backend/`：FastAPI 后端。
- `integrations/`：四个任务模板及相关运行集成。
- `scripts/start.sh`、`scripts/restart.sh`、`scripts/stop.sh`：本地服务管理。
- `eai_ide_backup.sql`：Docker PostgreSQL 首次初始化种子。

## 已整理的兼容与工具入口

- `scripts/legacy/phygen_model.py`：旧 PhyGen CLI 兼容入口；新调用使用 `scripts/train_phygen.py`。
- `backend/tools/annotation/annotation_page.py`：独立 PyQt 标注工具页，不属于 Web 前端入口。
- `backend/tools/diagnostics/`：环境、数据库、MCAP 与历史作业流程诊断脚本。
- `phygen/adapters/mimicgen/coffee_repair.py`：CoffeePreparation 修复与反馈生成实现。
- `label_task_description.py`：仍是后端与发布镜像共同使用的稳定入口，暂留根目录。

## 已归位但保持独立边界

- `third_party/mimicgen/`：MimicGen 嵌套仓库及官方源数据，螺母装配与 PhyGen 实验使用。
- `third_party/IsaacLab/`：Isaac Lab 运行时源码，物块堆叠生成、训练和评测使用。
- `integrations/IsaacSimFrankaPickPlace/`：尚未启用的 Isaac Sim 专家任务包，保持 `integration_pending`。

上述目录已经位于目标路径。第三方运行环境仍可能通过 editable install 记录绝对路径，物理移动
后需重新安装对应包；不要把它们再次复制回仓库根目录。

## 放置规则

- 运行数据、日志、缓存和模型产物写入仓库外的数据根目录，不进入源码目录。
- 第三方源码不与平台自研集成代码混放。
- 实验目录可以保存复现实验所需的代码和小型配置；大型输出应转移到外部数据根。
- 根目录只保留包管理、构建、容器编排和仓库级配置。
- `runtime_outputs/`、`eai-data/`、`mnt/data/`、`mimicgen_generated/` 与 `runs/` 只作为
  历史兼容或本地实验落点，均被版本控制忽略；平台新任务应写入 `EAI_DATA_ROOT`。
- 历史兼容入口必须明确标注，迁移前先修复全部引用并提供回归检查。
