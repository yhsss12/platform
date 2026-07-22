# 后端独立工具

本目录保存不由 FastAPI 服务或 worker 直接调用的独立工具。

- `annotation/`：桌面标注辅助工具。
- `diagnostics/`：环境、数据库、数据注册表和任务链路的只读检查工具。
- `verification/`：后端训练、评测和任务生成的独立冒烟工具。
- `database/`：数据库初始化、人工修复、危险维护与连接核验工具。
- `maintenance/`：索引回填、运行状态对账、远程训练同步和清理工具。
- `setup/`：后端可选运行时与依赖安装工具。
- `operations/`：队列观测与压力测试等人工运维工具。
- `experiments/`：平台 API 与网络条件的人工实验控制和指标处理工具。
- `migrate_runtime_to_minio.py`、`replay_events_from_runtime.py`：历史运行数据迁移与事件重放工具，因具有写入行为暂留工具根目录。
