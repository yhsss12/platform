# softgym_rope_tasks.py
# SoftGym 风格的绳索操作任务集合
# SoftGym 是一个经典的可变形物体操作 benchmark，这里的任务移植了其中两个核心任务：
#   1. RopeFlatten   —— 将弯曲的绳索拉直
#   2. RopeConfiguration —— 将绳索摆成指定的字符形状（S, O, M, C, U）
#
# 两个任务都继承自 CableStraighten，并复用了 SoftGym 的 "performance" 指标体系，
# 该体系通过归一化的性能分数来衡量操作效果，便于与 SoftGym 原始结果对比。

from pathlib import Path

import numpy as np

from robosuite.environments.manipulation.cable_straighten import CableStraighten
from robosuite.utils.dlo.softgym_rope_tasks import (
    SOFTGYM_ROPE_CHARACTERS,
    generate_softgym_character_target,
    rope_flatten_metrics,
    softgym_configuration_metrics,
)
from robosuite.utils.mjcf_utils import new_site


class RopeFlatten(CableStraighten):
    """
    绳索拉直任务：将弯曲缠绕的绳索展开为直线。

    核心指标 "performance" 的计算方式：
      performance = 1 - (当前绳索端点距离 / 绳索总长度)
      - 初始时 performance_init 记录起始状态的 performance
      - 成功条件：performance >= performance_success_threshold（默认 0.9）
      - 该指标与 SoftGym 原始实现保持一致，便于横向对比

    奖励直接使用 performance 值，这是一个连续的 [0, 1] 分数，
    越接近 1 表示绳索越接近完全拉直。
    """

    source_task = "softgym-rope-flatten"

    def __init__(
        self,
        *args,
        rope_length=None,                      # 绳索目标长度（None 则自动计算）
        performance_success_threshold=0.9,     # performance 成功阈值
        **kwargs,
    ):
        self.softgym_rope_length = rope_length
        self.performance_success_threshold = float(performance_success_threshold)
        self.performance_init = None            # 初始状态的 performance 分数，在 reset 时计算
        super().__init__(*args, **kwargs)

    def _reset_internal(self):
        """
        每次环境重置时记录初始 performance。

        这是 SoftGym 指标体系的关键：performance 是相对于初始状态的改善量，
        而不是绝对值。因此需要在 reset 时记录基线。
        """
        super()._reset_internal()
        metrics = rope_flatten_metrics(
            self._get_cable_points(),
            rope_length=self._target_rope_length(),
            success_threshold=self.performance_success_threshold,
        )
        self.performance_init = metrics["performance"]

    def _target_rope_length(self):
        """
        获取绳索的目标长度（完全拉直时的长度）。

        如果用户指定了 rope_length 则直接使用；
        否则通过累加相邻采样点之间的距离来估算当前绳索总长度。
        """
        if self.softgym_rope_length is not None:
            return float(self.softgym_rope_length)
        points = self._get_cable_points()
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def reward(self, action=None):
        """奖励直接使用 performance 分数，这是一个归一化的 [0, 1] 值。"""
        return self.reward_scale * self._compute_metrics()["performance"]

    def _compute_metrics(self):
        """
        计算 SoftGym 风格的绳索拉直指标。

        rope_flatten_metrics 工具函数会计算：
          - performance: 归一化的性能分数
          - success: 是否达到成功阈值
          - 以及其他中间指标（端点距离、绳索长度等）
        """
        metrics = super()._compute_metrics()
        softgym_metrics = rope_flatten_metrics(
            self._get_cable_points(),
            rope_length=self._target_rope_length(),
            performance_init=self.performance_init,
            success_threshold=self.performance_success_threshold,
        )
        metrics.update(softgym_metrics)
        metrics["source_task"] = self.source_task
        return metrics

    def _check_success(self):
        return self._compute_metrics()["success"]


class RopeConfiguration(CableStraighten):
    """
    绳索字符配置任务：将绳索摆成指定的字符形状。

    支持的字符（SOFTGYM_ROPE_CHARACTERS）：S, O, M, C, U
    这些字符选自 SoftGym 原始 benchmark，涵盖了不同拓扑结构：
      - C/U: 开放曲线（有端点）
      - S/M: 有转折的曲线
      - O: 闭合环（无端点）

    目标字符的形状由 generate_softgym_character_target 生成，
    返回一组 3D 关键点来描述字符的骨架。

    支持两种目标来源：
      1. 自动生成：根据 goal_character 字符生成目标关键点
      2. 文件加载：从 goal_file (.npy) 读取预设目标
    """

    source_task = "softgym-rope-configuration"

    def __init__(
        self,
        *args,
        goal_character="C",                    # 目标字符（大写字母）
        goal_file=None,                        # 可选的目标文件路径
        reward_type="bigraph",                 # 奖励类型（bigraph 是 SoftGym 的二部图匹配方法）
        target_center=(0.0, 0.0, 0.808),       # 目标字符的中心位置
        target_scale=0.18,                     # 目标字符的缩放比例
        target_yaw=0.0,                        # 目标字符的旋转角度
        performance_success_threshold=0.85,    # performance 成功阈值
        **kwargs,
    ):
        # 验证目标字符是否在支持列表中
        self.goal_character = str(goal_character).upper()
        if self.goal_character not in SOFTGYM_ROPE_CHARACTERS:
            raise ValueError(f"Unsupported SoftGym rope goal_character: {goal_character}")
        self.goal_file = goal_file
        self.reward_type = str(reward_type)
        self.target_center = np.asarray(target_center, dtype=float)
        self.target_scale = float(target_scale)
        self.target_yaw = float(target_yaw)
        self.performance_success_threshold = float(performance_success_threshold)
        self.performance_init = None
        # 尝试从文件加载目标关键点
        self._goal_file_keypoints = self._load_goal_file(goal_file)
        # target_line_visible=False: 不显示父类的直线目标
        super().__init__(*args, target_line_visible=False, **kwargs)

    def _load_goal_file(self, goal_file):
        """
        从 .npy 文件加载目标关键点。

        与 CableTargetMixin._load_goal_file 类似，但额外支持 "rope_keypoints" 键名，
        以兼容 SoftGym 格式的目标文件。
        """
        if not goal_file:
            return None
        data = np.load(Path(goal_file).expanduser())
        for key in ("target_keypoints", "goal_keypoints", "rope_keypoints", "cable_points"):
            if key in data:
                return np.asarray(data[key], dtype=float)
        raise ValueError(f"Goal file must contain target_keypoints, goal_keypoints, rope_keypoints, or cable_points: {goal_file}")

    def _load_model(self):
        """
        加载模型：生成或加载字符目标，并在场景中添加可视化标记。

        如果目标文件中的关键点只有 2D（xy），会自动补上 z 坐标
        （使用 cable_centerline_z，即线缆中心线的高度）。
        """
        super()._load_model()
        if self._goal_file_keypoints is None:
            # 自动生成字符目标关键点
            self.target_keypoints = generate_softgym_character_target(
                self.goal_character,
                self.num_cable_points,
                center=(self.target_center[0], self.target_center[1], self.cable_centerline_z),
                scale=self.target_scale,
                yaw=self.target_yaw,
            )
        else:
            self.target_keypoints = np.asarray(self._goal_file_keypoints, dtype=float)
            # 如果目标点只有 2D 坐标，自动补上 z 维度
            if self.target_keypoints.shape[1] == 2:
                z = np.full((len(self.target_keypoints), 1), self.cable_centerline_z)
                self.target_keypoints = np.concatenate([self.target_keypoints, z], axis=1)

        # 在场景中用青色小球显示目标字符的关键点
        for idx, pos in enumerate(self.target_keypoints):
            self.model.worldbody.append(
                new_site(
                    name=f"softgym_rope_configuration_target_{idx:02d}",
                    pos=pos,
                    size=(0.006,),
                    rgba=(0.1, 0.7, 0.85, 0.75),  # 青色，与其他任务的绿色区分
                )
            )

    def _reset_internal(self):
        """
        每次重置时记录初始 performance，作为后续改善量的基线。
        """
        super()._reset_internal()
        metrics = softgym_configuration_metrics(
            self._get_cable_points(),
            self.target_keypoints,
            reward_type=self.reward_type,
            success_threshold=self.performance_success_threshold,
        )
        self.performance_init = metrics["performance"]

    def reward(self, action=None):
        """奖励直接使用 performance 分数。"""
        return self.reward_scale * self._compute_metrics()["performance"]

    def _compute_metrics(self):
        """
        计算 SoftGym 风格的字符配置指标。

        softgym_configuration_metrics 工具函数使用二部图匹配（bigraph）
        或其他方法计算绳索点与目标字符点之间的匹配质量，
        输出归一化的 performance 分数和成功信号。
        """
        metrics = super()._compute_metrics()
        softgym_metrics = softgym_configuration_metrics(
            self._get_cable_points(),
            self.target_keypoints,
            performance_init=self.performance_init,
            reward_type=self.reward_type,
            success_threshold=self.performance_success_threshold,
        )
        metrics.update(softgym_metrics)
        metrics["source_task"] = self.source_task
        metrics["goal_character"] = self.goal_character
        metrics["target_keypoints"] = self.target_keypoints.copy()
        return metrics

    def _check_success(self):
        return self._compute_metrics()["success"]
