# Cursor 任务：导入 Isaac Sim 官方 Franka Pick Place 官方 Expert 模板

请把本包接入平台，作为一个“官方 expert policy 可用”的 Isaac 原生任务模板。

## 1. 新增任务模板

读取：

- `platform_template/isaacsim_franka_pick_place.template.yaml`
- `platform_template/success_and_metrics.yaml`

新增任务：

- task_id: `isaacsim_franka_pick_place`
- task_name: `Franka 物体搬运`
- simulator: `Isaac Sim`
- robot: `Franka Panda`
- expert_source: `NVIDIA Isaac Sim FrankaPickPlace official controller`
- task_type: `pick_and_place`
- dataset_format: `episode_manifest + metrics + optional video`

## 2. 专家策略接入方式

不要自己生成 expert policy，不要写 AI 生成策略。

使用本包中的 adapter：

- `expert/official_franka_pick_place_adapter.py`

该 adapter 只负责 import 并调用 Isaac Sim 官方：

```python
from isaacsim.robot.experimental.manipulators.examples.franka import FrankaPickPlace
```

核心调用逻辑：

```python
controller = FrankaPickPlace()
controller.setup_scene()
controller.reset()
while not controller.is_done():
    controller.forward()
```

## 3. 后端检测逻辑

平台启动时检测 Isaac Sim 环境：

- 如果能 import `isaacsim` 和 `FrankaPickPlace`，显示：`官方专家策略可用`
- 如果不能 import，不显示“占位/未接入”，显示：`需要 Isaac Sim 运行环境`

## 4. 页面展示要求

任务详情页展示：

- 任务名称：Franka 物体搬运
- 仿真后端：Isaac Sim
- 机器人：Franka Panda
- 官方专家策略：FrankaPickPlace
- 成功条件：cube 到达目标放置区域，夹爪释放，任务状态机完成
- 评测指标：success_rate、completion_step、final_position_error、grasp_success、place_success、timeout_count

## 5. 数据生成逻辑

点击“生成数据”时：

1. 调用后端 Isaac Sim runner；
2. runner 使用官方 `FrankaPickPlace` 执行；
3. 输出 episode manifest、metrics、可选视频；
4. 进入数据中心和回放中心。

## 6. 禁止事项

- 不要把它标成 Isaac Lab 任务；它是 Isaac Sim 官方 Robotics Example。
- 不要声称包里复制了 NVIDIA 官方源码；本包只是 adapter 和平台模板。
- 不要显示“示例数据”“占位任务”“未接入后端”等演示不友好的文案。
