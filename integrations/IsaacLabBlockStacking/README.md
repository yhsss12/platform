# Isaac Lab Franka Stack Cube

任务名称：Isaac Lab Franka Stack Cube

平台任务ID：task_isaaclab_franka_stack_cube_v1

来源项目：Isaac Lab / Isaac Lab Mimic

GitHub 链接：https://github.com/isaac-sim/IsaacLab

论文链接：https://arxiv.org/abs/2511.04831

许可证：遵循 Isaac Lab 原项目许可证（scene 配置 BSD-3-Clause，Mimic 相关 Apache-2.0）

仿真后端：Isaac Lab / Isaac Sim

任务说明：
Franka 机械臂在 Isaac Lab 环境中完成方块抓取与堆叠任务，可通过 seed demonstration 和 Isaac Lab Mimic 生成 demonstration 数据。

场景代码：
scene/

专家策略代码：
expert_policy/

运行入口：
run/

是否需要 seed demonstration：
是

能否生成数据：
能

数据格式：
HDF5 / Zarr

是否已跑通：
平台已登记，运行依赖 Isaac Lab 环境，平台内运行待实际任务验证

备注：
该任务包依赖完整 Isaac Lab / Isaac Sim 运行环境。scene/ 与 expert_policy/ 中包含关键配置与生成链路代码，但部分 assets 和基础模块来自 Isaac Lab 原工程。平台通过 adapter 方式调用该任务，不将该 zip 视为完全独立 Python 包。
