# 后端诊断工具

本目录中的脚本用于检查环境、数据库、训练节点、工作空间索引和历史任务流程。
它们不是后端启动入口，也不应由生产 worker 自动调用。

从仓库根目录运行示例：

```bash
cd backend
python tools/diagnostics/check_db_health.py
```
