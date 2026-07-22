# DP / ACT Adapter Note

核心思路：

把 Isaac Lab Mimic / SkillGen 生成的 HDF5 专家 demonstration，整理成 Diffusion Policy 或 ACT 数据加载器能读取的格式。

## 为什么不把 DP / ACT 放进 expert_policy/

expert_policy/ 里应该放能直接控制机器人完成任务、或能生成专家 demonstration 的代码。Isaac Lab Mimic / SkillGen 属于这一层。

Diffusion Policy / ACT 是后续用专家 demonstration 训练出来的 imitation policy。它们不是这个 Isaac Stack Cube 任务当前自带的官方专家策略 checkpoint，所以更适合放在运行和数据适配说明里。

## HDF5 到 DP Zarr 的关系

Isaac Lab Mimic 生成：

```text
generated_dataset.hdf5
  data/
    demo_0/
      obs/
      actions
    demo_1/
      obs/
      actions
```

Diffusion Policy 常见训练格式：

```text
isaac_stack_cube_dp.zarr
  data/
    state
    action
  meta/
    episode_ends
```

关键字段是 episode_ends。它记录每条轨迹在拼接后大数组里的结束位置。

例如：

```text
demo_0 长度 100
demo_1 长度 120
demo_2 长度 90

episode_ends = [100, 220, 310]
```

DP/ACT 训练时不是学“整个任务脚本”，而是从这些连续轨迹中切时间窗口，学习从 observation history 到 future action sequence 的映射。

## 对师兄可以这样解释

Isaac Lab Mimic / SkillGen 负责生成专家 demonstration，原始输出是 HDF5。后续对接 Diffusion Policy 或 ACT 时，不需要修改 Isaac 的专家策略，只需要做数据适配：读取 HDF5 中每条 demo 的 obs 和 actions，拼接成连续数组，记录 episode_ends，并保存成 Zarr 或 LeRobotDataset 等 imitation learning 框架能读取的格式。

本包已放置 convert_isaac_hdf5_to_zarr.py，作为 HDF5 到 DP Zarr ReplayBuffer 的最小转换入口。

