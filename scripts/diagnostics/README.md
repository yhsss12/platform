# 运行环境诊断脚本

这里存放只读检查和独立探针，不是平台服务或任务模板的生产入口。

- `diagnose_isaac_runtime.py`：检查 Isaac Sim / Isaac Lab 运行环境。
- `isaac_runtime_import_probe.py`：验证 Isaac Python 模块导入。
- `isaac_franka_controller_probe.py`：验证 Franka 控制器运行条件。
- `list_isaaclab_franka_tasks.py`：列出可用的 Isaac Lab Franka 任务。

物块堆叠的正式运行脚本仍位于 `backend/integrations/isaac_lab/` 和
`integrations/IsaacLabBlockStacking/`，移动这些诊断脚本不会改变任务调度路径。
