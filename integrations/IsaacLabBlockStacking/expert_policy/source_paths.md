# Expert Policy Source Paths

本文件夹放能产生专家 demonstration 的策略/生成逻辑。

已放入的开源文件：

- franka_stack_ik_rel_mimic_env_cfg.py
  - 来源路径：source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.py
  - 作用：定义 Stack Cube 任务用于 Mimic 数据生成的子任务配置，例如 grasp、lift、place 等阶段的切分和约束。

- mimic_data_generator.py
  - 来源路径：source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py
  - 作用：Isaac Lab Mimic 的核心数据生成逻辑，根据 seed demonstration 和任务子阶段配置生成新的 demonstration。

为什么这里不放 ACT / Diffusion Policy：

ACT / Diffusion Policy 是后续训练得到的 imitation policy，不是这个 Isaac Stack Cube 任务自带的专家策略本体。按照师兄的格式，expert_policy/ 应该放“能控制机器人完成任务或生成专家数据的代码”。因此这里放 Isaac Lab Mimic 的配置和数据生成器；DP/ACT 的对接说明放在 run/dp_adapter_note.md。

