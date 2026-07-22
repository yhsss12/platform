"""
CableMoveToTarget — 线缆移动到目标区域任务。

任务目标：将线缆整体移动到桌面上的指定圆形区域内。
继承 CableStraighten，复用其线缆模型、重置逻辑和机器人设置。

目标区域由 target_center（圆心）和 target_radius（半径）定义。

成功条件（全部满足）：
  - 线缆质心（centroid）到目标中心的距离 < centroid_success_threshold
  - 线缆关键点在目标区域内的覆盖率 > coverage_success_threshold

核心度量委托给 task_logic.py 中的 move_to_target_task_metrics() 计算。
"""

import numpy as np

from robosuite.environments.manipulation.cable_straighten import CableStraighten
from robosuite.utils.dlo.task_logic import DLOTaskState, MoveToTargetTaskSpec, move_to_target_task_metrics


class CableMoveToTarget(CableStraighten):
    """
    Minimal target-region cable task built on the CableStraighten infrastructure.

    This first version intentionally reuses the cable object, reset path, robot
    setup, and keypoint reading from CableStraighten. It adds centroid and goal
    coverage metrics so smoke, random-action, schema, and evaluation paths can
    stabilize before a task-specific expert is introduced.
    """

    def __init__(
        self,
        *args,
        target_center=(0.02, 0.18, 0.808),  # 目标区域圆心（x, y, z），z 通常等于桌面高度
        target_radius=0.26,                   # 目标区域半径（米）
        centroid_success_threshold=0.05,      # 质心到目标中心的距离阈值
        coverage_success_threshold=0.8,       # 关键点覆盖率阈值（80% 以上在目标区域内）
        horizon=2400,                         # 需要比 CableStraighten 更多步骤完成搬运
        **kwargs,
    ):
        self.target_center = np.asarray(target_center, dtype=float)
        self.target_radius = float(target_radius)
        self.centroid_success_threshold = float(centroid_success_threshold)
        self.coverage_success_threshold = float(coverage_success_threshold)
        super().__init__(*args, horizon=horizon, **kwargs)

    def _load_model(self):
        super()._load_model()
        from robosuite.utils.dlo.task_scene_utils import add_target_sites
        # 目标圆环：24 个绿色球标记圆周
        n_points = 24
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        circle_points = np.column_stack([
            self.target_center[0] + self.target_radius * np.cos(angles),
            self.target_center[1] + self.target_radius * np.sin(angles),
            np.full(n_points, self.target_center[2]),
        ])
        add_target_sites(self.model.worldbody, circle_points, name_prefix="target_circle", size=[0.006], rgba=[0.2, 0.8, 0.2, 0.8])
        # 中心标记：红色大球
        add_target_sites(self.model.worldbody, self.target_center.reshape(1, 3), name_prefix="target_center", size=[0.010], rgba=[1.0, 0.2, 0.2, 0.8])

    def reward(self, action=None):
        """计算移动到目标区域任务的 reward。

        reward = -centroid_distance + coverage - action_penalty + success_bonus

        - centroid_distance: 质心到目标中心的距离（越近越好）
        - coverage: 关键点在目标区域内的覆盖率（0~1，越大越好）
        - action_penalty: 动作幅度惩罚（鼓励平稳运动，权重 0.01）
        - success_bonus: 全部达标后给 +1 的奖励
        """
        metrics = self._compute_metrics()
        action_penalty = 0.0 if action is None else 0.01 * float(np.linalg.norm(action))
        reward = -metrics["centroid_distance_to_goal"] + metrics["keypoint_goal_coverage"] - action_penalty
        if metrics["success"]:
            reward += 1.0
        return self.reward_scale * reward

    def _task_goal_obs(self):
        """返回任务目标观测：[target_center(3), target_radius(1)]。"""
        return np.concatenate([self.target_center, np.array([self.target_radius], dtype=float)])

    def _compute_metrics(self):
        """计算移动到目标区域任务的完整指标。

        先调用父类 _compute_metrics() 获取线缆拉直相关指标，
        再叠加 move_to_target_task_metrics 计算的目标区域指标：
        - centroid_distance_to_goal: 质心到目标中心的距离
        - keypoint_goal_coverage: 关键点在目标区域内的覆盖率
        - centroid_success: 质心是否足够接近目标中心
        - coverage_success: 覆盖率是否达标
        """
        metrics = super()._compute_metrics()
        points = self._get_cable_points()
        metrics.update(
            move_to_target_task_metrics(
                DLOTaskState(
                    keypoints=points,
                    table_top_z=self.table_top_z,
                    centerline_z=self.cable_centerline_z,
                    initial_polyline_length=self.initial_polyline_length,
                    initial_straightness_ratio=self.initial_straightness_ratio,
                ),
                MoveToTargetTaskSpec(
                    target_center=self.target_center,
                    target_radius=self.target_radius,
                    centroid_success_threshold=self.centroid_success_threshold,
                    coverage_success_threshold=self.coverage_success_threshold,
                ),
                metrics,
            )
        )
        return metrics

    def _check_success(self):
        return self._compute_metrics()["success"]
