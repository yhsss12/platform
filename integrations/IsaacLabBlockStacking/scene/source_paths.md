# Scene Source Paths

本文件夹放 Isaac Stack Cube 任务的场景相关代码。

已放入的开源文件：

- stack_joint_pos_env_cfg.py
  - 来源路径：source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_joint_pos_env_cfg.py
  - 作用：定义 Franka Stack Cube 任务中的机器人、方块、桌面、传感器、观测、事件和奖励等基础环境配置。

- stack_ik_rel_env_cfg.py
  - 来源路径：source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_env_cfg.py
  - 作用：定义相对 IK 控制版本的 Stack Cube 任务，适合演示数据录制和 Mimic 数据生成。

原始项目：

https://github.com/isaac-sim/IsaacLab

注意：
这些文件依赖完整 Isaac Lab 工程中的 isaaclab、isaaclab_assets 和 isaaclab_tasks 包。这个压缩包用于任务代码整理和快速定位，不等价于完整 Isaac Lab 安装包。

