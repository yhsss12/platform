# Source Notes

官方文档对象：NVIDIA Isaac Sim Franka Pick and Place Example。

关键点：

- 官方示例用于设置 Franka + gripper、实现 linear pick-and-place sequence、控制机器人和夹爪动作。
- 官方代码结构包含 interactive example 与 standalone example，最终都指向 `isaacsim.robot.experimental.manipulators.examples/franka/pick_place.py`。
- 官方 `FrankaPickPlace` 类提供完整场景搭建和状态机执行。
- 状态机阶段包括：移动到 cube 上方、下降、闭合夹爪、抬起、移动到目标、释放、撤离。

平台侧不要把该任务标为 Isaac Lab；应标为 Isaac Sim 官方 Robotics Example。
