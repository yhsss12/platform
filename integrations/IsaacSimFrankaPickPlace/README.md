# Isaac Sim Franka Pick Place 官方 Expert Policy 平台模板包

本包选择 **NVIDIA Isaac Sim 官方 Franka Pick and Place Example**，而不是 Isaac Lab Stack Cube。

原因：Isaac Lab 官方很多 manipulation 任务主要提供 RL/IL 环境、奖励、Mimic 数据生成流程，不一定内置“可直接调用的 scripted expert policy”。Isaac Sim 官方 Manipulation Example 中的 `FrankaPickPlace` 则提供了完整 pick-and-place 状态机，可作为官方原生 expert/controller 入口。

## 官方 expert 来源

官方类：

```python
from isaacsim.robot.experimental.manipulators.examples.franka import FrankaPickPlace
```

官方文档说明该类：
- `setup_scene()` 会生成 Franka、地面、cube 等完整 pick-place 场景；
- `forward()` 每个 physics frame 推进一步 pick-place 状态机；
- `is_done()` 判断任务是否完成；
- 状态机包含移动到方块上方、下降、闭合夹爪、抬起、移动到目标、释放、撤离等阶段。

## 本包定位

本包不复制 NVIDIA Isaac Sim 的源码和 USD 资产，而是提供：

- 平台任务模板
- 官方 expert policy 的 import adapter
- Cursor 接入指令
- 运行脚本
- 成功条件和评测指标
- demo_data 的 manifest/metrics 结构

真实运行需要本机安装 Isaac Sim 6.0 或兼容版本。

## 推荐平台展示名称

- 任务名称：Franka 物体搬运
- 仿真后端：Isaac Sim
- 机器人：Franka Panda
- 专家策略来源：NVIDIA Isaac Sim 官方 FrankaPickPlace controller
- 任务类型：pick-and-place
- 状态：official_expert_available_when_isaacsim_installed

## 快速运行

在 Isaac Sim 安装目录下运行：

```bash
./python.sh standalone_examples/api/isaacsim.robot.experimental.manipulators/franka/pick_place.py --test
```

或运行本包脚本：

```bash
bash scripts/run_official_pick_place.sh /path/to/isaacsim
```

## 注意

这里的 expert 不是 AI 生成，不是平台自写规则，也不是伪造 demonstration；它调用的是 Isaac Sim 官方扩展中的 `FrankaPickPlace`。
