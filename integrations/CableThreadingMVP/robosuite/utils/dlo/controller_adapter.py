"""薄封装层：将任务级 EEF 命令转换为 robosuite 动作向量。

本模块是 dloBench 五层架构中 L2（Controller Adapter）层的核心组件。
它在任务逻辑（L3）和 robosuite 机器人控制器（L1）之间架起桥梁。

核心设计：
  - RobosuiteControllerAdapter 不自己实现控制器
  - 它通过 robot.create_action_vector() 委托给 robosuite 的控制器兼容层
  - 这保证了对所有机器人类型（delta/absolute）的兼容性

支持两种控制器模式：
  1. delta 模式：动作 = 目标位移 / position_scale，归一化到 [-1, 1]
  2. absolute 模式：动作 = 绝对目标位置（直接传入）

关键概念：
  - EEFCommand: 与机器人无关的任务级命令（目标位置 + 夹爪状态）
  - action_dict: 机器人各部件（arm、gripper）的动作字典
  - position_scale: delta 模式下的位移缩放因子
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EEFCommand:
    """任务级 EEF（End-Effector）命令。

    与机器人类型无关，只描述"想让夹爪去哪里"和"夹爪开合状态"。

    Attributes:
        target_pos: 目标位置 (x, y, z)。
        gripper:    夹爪状态（0.0 = 全开，1.0 = 全闭）。
    """

    target_pos: np.ndarray
    gripper: float = 0.0


class RobosuiteControllerAdapter:
    """将任务级 EEF 命令转换为 robosuite 动作向量。

    该类故意不实现控制器本身。它将动作向量的组装委托给
    ``robot.create_action_vector``，让 robosuite 的控制器兼容层
    成为每个机器人的唯一事实来源。

    工作流程：
      1. 外部代码调用 action_from_delta() 或 action_to_eef_position()
      2. 本适配器根据控制器类型（delta/absolute）构建 action_dict
      3. 通过 robot.create_action_vector() 转换为最终的动作向量
      4. clip_action() 确保动作在合法范围内
    """

    def __init__(self, env, robot_index: int = 0, arm: str | None = None, position_scale: float = 0.04):
        """初始化适配器。

        Args:
            env: robosuite 环境对象。
            robot_index: 机器人在 env.robots 中的索引（多机器人场景）。
            arm: 臂的名称（默认自动检测）。
            position_scale: delta 模式下的位移缩放因子（米）。
        """
        self.env = env
        self.robot_index = int(robot_index)
        self.robot = env.robots[self.robot_index]
        self.arm = arm or self._default_arm()
        self.position_scale = float(position_scale)
        if self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive")

    def current_eef_pos(self) -> np.ndarray:
        """获取当前夹爪（末端执行器）位置。

        通过环境的 _get_gripper_site_position() 方法获取。
        这是 robosuite 环境需要暴露的接口。

        Returns:
            shape (3,) 的当前位置数组。
        """
        if hasattr(self.env, "_get_gripper_site_position"):
            return np.asarray(self.env._get_gripper_site_position(), dtype=float)
        raise AttributeError("env must expose _get_gripper_site_position() for DLO EEF commands")

    def action_to_eef_position(self, target_pos, gripper: float = 0.0) -> np.ndarray:
        """将绝对目标位置转换为动作向量。

        计算当前位置到目标位置的 delta，然后委托给 action_from_delta。

        Args:
            target_pos: 绝对目标位置 (x, y, z)。
            gripper: 夹爪状态。

        Returns:
            动作向量。
        """
        command = EEFCommand(target_pos=np.asarray(target_pos, dtype=float), gripper=float(gripper))
        return self.action_from_delta(command.target_pos - self.current_eef_pos(), command.gripper, command.target_pos)

    def action_from_delta(self, delta_pos, gripper: float = 0.0, absolute_target_pos=None) -> np.ndarray:
        """将位移增量转换为动作向量。

        这是最常用的接口。delta_pos 是"想让夹爪移动多少"。

        Args:
            delta_pos: 位移增量 (dx, dy, dz)。
            gripper: 夹爪状态。
            absolute_target_pos: 绝对目标位置（absolute 模式使用，
                                 如果为 None 则由 current_eef_pos + delta_pos 计算）。

        Returns:
            clip 后的动作向量。
        """
        action_dict = self._action_dict_for_delta(
            np.asarray(delta_pos, dtype=float),
            float(gripper),
            None if absolute_target_pos is None else np.asarray(absolute_target_pos, dtype=float),
        )
        return self.clip_action(self._env_action_from_robot_action_dict(action_dict))

    def zero_action(self) -> np.ndarray:
        """生成零动作向量（不移动、不夹持）。"""
        return self.clip_action(self._env_action_from_robot_action_dict({}))

    def clip_action(self, action) -> np.ndarray:
        """将动作向量 clip 到环境允许的范围内。

        使用 env.action_spec 返回的 (low, high) 边界。
        """
        low, high = self.env.action_spec
        return np.clip(np.asarray(action, dtype=np.float32), low, high)

    def _default_arm(self) -> str:
        """自动检测唯一的活动臂名称。

        当前 DLO 适配器只支持单臂机器人。
        """
        arms = list(getattr(self.robot, "arms", []))
        if len(arms) != 1:
            raise ValueError(f"DLO controller adapter currently expects one active arm, got {arms}")
        return arms[0]

    def _action_dict_for_delta(self, delta_pos: np.ndarray, gripper: float, absolute_target_pos: np.ndarray | None):
        """根据控制器类型构建动作字典。

        delta 模式：arm_action = delta_pos / position_scale，归一化到 [-1, 1]
        absolute 模式：arm_action = 绝对目标位置

        动作字典结构：
          {arm_name: arm_action, gripper_name: gripper_action}
        """
        arm_dim = self._part_action_dim(self.arm)
        arm_action = np.zeros(arm_dim, dtype=np.float32)
        input_type = self._controller_input_type()
        if input_type == "delta":
            # delta 模式：位移除以缩放因子，clip 到 [-1, 1]
            arm_action[: min(3, arm_dim)] = np.clip(delta_pos[: min(3, arm_dim)] / self.position_scale, -1.0, 1.0)
        elif input_type == "absolute":
            # absolute 模式：直接传入目标位置
            if absolute_target_pos is None:
                absolute_target_pos = self.current_eef_pos() + delta_pos
            arm_action[: min(3, arm_dim)] = absolute_target_pos[: min(3, arm_dim)]
        else:
            raise ValueError(f"Unsupported controller input_type: {input_type}")

        action_dict = {self.arm: arm_action}
        gripper_name = self._gripper_part_name()
        if gripper_name is not None:
            action_dict[gripper_name] = np.full(self._part_action_dim(gripper_name), gripper, dtype=np.float32)
        return action_dict

    def _controller_input_type(self) -> str:
        """检测控制器的输入类型（"delta" 或 "absolute"）。

        优先从 composite_controller.joint_action_policy 获取，
        否则从 part_controllers 中获取。
        """
        composite_controller = getattr(self.robot, "composite_controller", None)
        joint_policy = getattr(composite_controller, "joint_action_policy", None)
        if joint_policy is not None and hasattr(joint_policy, "input_type"):
            return joint_policy.input_type
        return self.robot.part_controllers[self.arm].input_type

    def _gripper_part_name(self) -> str | None:
        """检测夹爪的部件名称。

        优先使用 robot.get_gripper_name(arm)，
        否则尝试 "{arm}_gripper" 作为备选名称。
        如果夹爪不存在（action_dim == 0），返回 None。
        """
        if hasattr(self.robot, "get_gripper_name"):
            name = self.robot.get_gripper_name(self.arm)
            if self._part_action_dim(name, missing_ok=True) > 0:
                return name
        fallback = f"{self.arm}_gripper"
        if self._part_action_dim(fallback, missing_ok=True) > 0:
            return fallback
        return None

    def _part_action_dim(self, part_name: str, missing_ok: bool = False) -> int:
        """获取指定部件的动作维度。

        按优先级查找：
          1. composite_controller 的 action_split_indexes
          2. part_controllers 的 control_dim
          3. gripper 的 dof

        Args:
            part_name: 部件名称（如 "right_arm"、"right_gripper"）。
            missing_ok: 如果为 True，找不到时返回 0 而非抛异常。

        Returns:
            动作维度（整数）。
        """
        split_indexes = self._action_split_indexes()
        if part_name in split_indexes:
            start, end = split_indexes[part_name]
            return int(end - start)
        controller = getattr(self.robot, "part_controllers", {}).get(part_name)
        if controller is not None and hasattr(controller, "control_dim"):
            return int(controller.control_dim)
        if part_name.endswith("_gripper"):
            arm = part_name[: -len("_gripper")]
            gripper = getattr(self.robot, "gripper", {}).get(arm)
            if gripper is not None and hasattr(gripper, "dof"):
                return int(gripper.dof)
        if missing_ok:
            return 0
        raise KeyError(f"Could not resolve action dimension for robot part {part_name!r}")

    def _action_split_indexes(self):
        """获取各部件在动作向量中的索引范围。

        返回 dict：{part_name: (start_index, end_index)}
        """
        composite_controller = getattr(self.robot, "composite_controller", None)
        if composite_controller is not None:
            split_indexes = getattr(composite_controller, "_whole_body_controller_action_split_indexes", None)
            if split_indexes is not None:
                return split_indexes
            split_indexes = getattr(composite_controller, "_action_split_indexes", None)
            if split_indexes is not None:
                return split_indexes
        return getattr(self.robot, "_action_split_indexes", {})

    def _env_action_from_robot_action_dict(self, action_dict) -> np.ndarray:
        """将机器人的动作字典转换为环境级的动作向量。

        多机器人场景下，将所有机器人的动作拼接在一起。
        当前机器人使用传入的 action_dict，其他机器人使用空字典。
        """
        robot_actions = []
        for index, robot in enumerate(self.env.robots):
            robot_action = robot.create_action_vector(action_dict if index == self.robot_index else {})
            robot_actions.append(np.asarray(robot_action, dtype=np.float32))
        return np.concatenate(robot_actions, axis=0)
