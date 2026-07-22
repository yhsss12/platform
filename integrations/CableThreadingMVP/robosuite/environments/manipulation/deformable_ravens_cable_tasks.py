# deformable_ravens_cable_tasks.py
# DeformableRavens 风格的线缆形状匹配任务集合
# 包含两大类任务：
#   1. CableShape —— 将线缆摆成多边形（三角形、四边形等）或直线
#   2. CableRing —— 将线缆摆成环形
#
# 共用基础设施：
#   - CableTargetMixin: 提供目标点加载、目标可视化、目标相关指标计算
#   - _resample_keypoints: 关键点重采样工具函数
#
# 通过 target_visible 参数控制目标可视化：
#   - target_visible=True（默认）：场景中显示绿色目标标记点
#   - target_visible=False：隐藏目标标记，测试策略的泛化能力
# 支持 goal_file 参数从 .npy 文件加载预设目标。

from pathlib import Path

import numpy as np

from robosuite.environments.manipulation.cable_straighten import CableStraighten
from robosuite.utils.dlo.deformable_ravens_tasks import (
    best_ring_target_mapping,
    generate_polyline_target,
    generate_ring_target,
    ring_area_metrics,
    target_keypoint_metrics,
)
from robosuite.utils.dlo.task_scene_utils import add_target_sites


def _resample_keypoints(points, count):
    """
    将一组关键点重采样为指定数量的点。

    用途：当线缆的采样点数与目标关键点数不一致时，
    通过线性插值将目标点重采样到与线缆相同的数量，以便逐点计算距离指标。

    参数：
      points: 原始关键点数组，形状 (N, 3)
      count:  目标点数
    返回：
      重采样后的关键点数组，形状 (count, 3)
    """
    points = np.asarray(points, dtype=float)
    if len(points) == int(count):
        return points
    sample = np.linspace(0, len(points) - 1, int(count))
    lower = np.floor(sample).astype(int)
    upper = np.ceil(sample).astype(int)
    alpha = sample - lower
    return (1.0 - alpha[:, None]) * points[lower] + alpha[:, None] * points[upper]


class CableTargetMixin:
    """
    线缆目标混入类：为所有需要目标形状的任务提供通用功能。

    提供三大能力：
      1. 目标文件加载 (_load_goal_file): 从 .npy 文件读取预设的目标关键点
      2. 目标可视化 (_add_target_sites): 在 MuJoCo 场景中用绿色小球标记目标位置
      3. 目标观测指标 (_target_observation_metrics): 将目标信息打包到指标字典中
    """

    source_task = None       # 任务来源标识，由子类设置
    target_visible = True    # 是否在场景中显示目标标记

    def _load_goal_file(self, goal_file):
        """
        从 .npy 文件加载目标关键点。

        支持三种键名：target_keypoints、goal_keypoints、cable_points，
        兼容不同来源的目标文件格式。
        """
        if not goal_file:
            return None
        data = np.load(Path(goal_file).expanduser())
        for key in ("target_keypoints", "goal_keypoints", "cable_points"):
            if key in data:
                return np.asarray(data[key], dtype=float)
        raise ValueError(f"Goal file must contain one of target_keypoints, goal_keypoints, cable_points: {goal_file}")

    def _add_target_sites(self, rgba=(0.1, 0.8, 0.2, 0.75)):
        """
        在 MuJoCo 场景中为目标关键点添加可视化标记点。

        每个目标点用一个半透明绿色小球表示，便于在渲染画面中直观看到目标形状。
        如果 target_visible=False，则跳过，不添加任何标记。
        """
        if not self.target_visible:
            return
        add_target_sites(
            self.model.worldbody,
            self.target_keypoints,
            name_prefix=f"{self.source_task.replace('-', '_')}_target",
            rgba=rgba,
        )

    def _target_observation_metrics(self):
        """将目标相关的元信息打包到指标字典中，供观测空间和日志使用。"""
        return {
            "source_task": self.source_task,
            "target_visible": bool(self.target_visible),
            "target_keypoints": self.target_keypoints.copy(),
        }


class CableShape(CableTargetMixin, CableStraighten):
    """
    线缆形状匹配任务：将线缆摆成指定边数的多边形。

    核心概念：
      - 目标形状是一个正多边形（如三角形、四边形），由 25 个均匀分布的关键点描述
      - 通过 "覆盖率"（target_coverage）和 "平均关键点误差"（mean_keypoint_error）衡量匹配质量
      - 覆盖率 >= coverage_success_threshold 即视为成功

    参数：
      - target_visible: 是否在场景中显示目标标记（True=显示，False=隐藏）
      - goal_file: 可选的 .npy 目标文件路径（覆盖自动生成的目标）
      - num_sides: 多边形边数（1=直线，2=V形，3=三角形，4=四边形）

    继承关系：CableTargetMixin 提供目标功能，CableStraighten 提供线缆物理模拟基础
    """

    source_task = "cable-shape"
    target_visible = True

    def __init__(
        self,
        *args,
        target_center=(0.0, 0.0, 0.808),
        target_yaw=0.0,
        num_sides=None,
        num_sides_low=2,
        num_sides_high=4,
        target_cutoff=None,
        coverage_success_threshold=0.90,
        target_distance_threshold=None,
        target_visible=None,       # 覆盖类属性：True=显示目标，False=隐藏
        goal_file=None,            # .npy 目标文件路径
        **kwargs,
    ):
        kwargs.setdefault("cable_model", "composite_cable")
        # 如果指定了 target_visible，覆盖类属性
        if target_visible is not None:
            self.target_visible = bool(target_visible)
        # 加载目标文件（如果有）
        self.goal_file = goal_file
        self._goal_file_keypoints = self._load_goal_file(goal_file)

        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(seed)
        self.num_target_points = 25
        self.target_center = np.asarray(target_center, dtype=float)
        self.target_yaw = float(target_yaw)
        self.num_sides = int(num_sides) if num_sides is not None else int(rng.integers(int(num_sides_low), int(num_sides_high) + 1))
        self.target_cutoff = None if target_cutoff is None else int(target_cutoff)
        self.coverage_success_threshold = float(coverage_success_threshold)
        self.target_distance_threshold = target_distance_threshold
        # 设置 source_task 用于目标标记命名
        if not self.target_visible:
            self.source_task = "cable-shape-notarget"
        super().__init__(*args, target_line_visible=False, **kwargs)

    def _load_model(self):
        """
        加载模型时生成多边形目标关键点，并在场景中添加可视化标记。

        如果提供了 goal_file，从文件加载目标；否则自动生成。
        如果 target_visible=False，跳过目标标记的添加。
        """
        if self._goal_file_keypoints is not None:
            self.target_keypoints = self._goal_file_keypoints
            self.num_target_points = len(self.target_keypoints)
        else:
            self.target_keypoints = generate_polyline_target(
                self.num_target_points,
                num_sides=self.num_sides,
                length=0.5,
                center=self.target_center,
                yaw=self.target_yaw,
                cutoff=self.target_cutoff,
            )
        super()._load_model()
        self._add_target_sites()

    def reward(self, action=None):
        """奖励函数：覆盖率 - 平均关键点误差 + 成功奖励。"""
        metrics = self._compute_metrics()
        reward = metrics["target_coverage"] - metrics["mean_keypoint_error"]
        if metrics["success"]:
            reward += 1.0
        return self.reward_scale * reward

    def _compute_metrics(self):
        """计算形状匹配指标。"""
        metrics = super()._compute_metrics()
        threshold = self.target_distance_threshold
        if threshold is None:
            threshold = 3.5 * float(self.cable_radius)
        target_metrics = target_keypoint_metrics(self._get_cable_points(), self.target_keypoints, distance_threshold=threshold)
        metrics.update(target_metrics)
        metrics.update(self._target_observation_metrics())
        metrics["nb_sides"] = int(self.num_sides)
        metrics["success"] = bool(metrics["target_coverage"] >= self.coverage_success_threshold)
        metrics["task_success"] = metrics["success"]
        metrics["coverage_success_threshold"] = self.coverage_success_threshold
        return metrics

    def _check_success(self):
        return self._compute_metrics()["success"]


class CableRing(CableTargetMixin, CableStraighten):
    """
    线缆环形匹配任务：将线缆摆成一个圆环。

    与 CableShape 的区别：
      - 目标是圆形而非多边形
      - 使用 "面积分数"（area_fraction）作为额外的成功指标
      - area_fraction 衡量线缆围成的面积占目标圆面积的比例

    参数：
      - target_visible: 是否在场景中显示目标标记
      - goal_file: 可选的 .npy 目标文件路径

    成功条件（双条件）：
      1. area_fraction >= area_fraction_threshold（环形面积足够大）
      2. 线缆整体靠近桌面且映射误差足够小（ring_cable_near_table）
    """

    source_task = "cable-ring"
    target_visible = True

    def __init__(
        self,
        *args,
        ring_center=(0.0, 0.0, 0.808),
        ring_radius=0.075,
        ring_yaw=0.0,
        area_fraction_threshold=None,
        target_distance_threshold=None,
        target_visible=None,       # 覆盖类属性
        goal_file=None,            # .npy 目标文件路径
        **kwargs,
    ):
        kwargs.setdefault("cable_model", "composite_cable")
        # 如果指定了 target_visible，覆盖类属性
        if target_visible is not None:
            self.target_visible = bool(target_visible)
        # 加载目标文件（如果有）
        self.goal_file = goal_file
        self._goal_file_keypoints = self._load_goal_file(goal_file)

        self.num_target_points = 25
        self.ring_center = np.asarray(ring_center, dtype=float)
        self.ring_radius = float(ring_radius)
        self.ring_yaw = float(ring_yaw)
        self._user_area_fraction_threshold = float(area_fraction_threshold) if area_fraction_threshold is not None else None
        self.area_fraction_threshold = self._user_area_fraction_threshold if self._user_area_fraction_threshold is not None else 0.85
        self._user_target_distance_threshold = float(target_distance_threshold) if target_distance_threshold is not None else None
        self.target_distance_threshold = self._user_target_distance_threshold if self._user_target_distance_threshold is not None else 0.02
        # 设置 source_task 用于目标标记命名
        if not self.target_visible:
            self.source_task = "cable-ring-notarget"
        super().__init__(*args, target_line_visible=False, **kwargs)

    @property
    def circle_area(self):
        """目标圆的面积（平方米），用于归一化面积分数指标。"""
        return float(np.pi * self.ring_radius ** 2)

    def _load_model(self):
        """加载模型：生成圆形目标、自动调整阈值、添加可视化标记。"""
        if self._goal_file_keypoints is not None:
            self.target_keypoints = self._goal_file_keypoints
            self.num_target_points = len(self.target_keypoints)
        else:
            self.target_keypoints = generate_ring_target(
                self.num_target_points,
                radius=self.ring_radius,
                center=self.ring_center,
                yaw=self.ring_yaw,
            )
        super()._load_model()

        # 自动阈值调整
        ring_circumference = 2.0 * np.pi * self.ring_radius
        cable_len = float(getattr(self, "cable_length", 0.48))

        if self._user_area_fraction_threshold is None:
            slack_ratio = cable_len / ring_circumference
            self.area_fraction_threshold = float(np.clip(slack_ratio ** 2 - 0.20, 0.60, 0.90))

        if self._user_target_distance_threshold is None:
            num_beads = max(2, int(getattr(self, "num_beads", 25)))
            bead_spacing = cable_len / (num_beads - 1)
            self.target_distance_threshold = float(np.clip(bead_spacing * 0.50, 0.008, 0.05))

        self._add_target_sites(rgba=(0.1, 0.75, 0.25, 0.7))

    def reward(self, action=None):
        """奖励函数：面积分数 + 0.5 * 覆盖率 - 平均关键点误差 + 成功奖励。"""
        metrics = self._compute_metrics()
        reward = metrics["area_fraction"] + 0.5 * metrics["target_coverage"] - metrics["mean_keypoint_error"]
        if metrics["success"]:
            reward += 1.0
        return self.reward_scale * reward

    def _compute_metrics(self):
        """计算环形匹配指标。"""
        metrics = super()._compute_metrics()
        points = self._get_cable_points()
        mapped_targets = _resample_keypoints(self.target_keypoints, len(points))
        target_metrics = target_keypoint_metrics(points, mapped_targets, distance_threshold=self.target_distance_threshold)
        ring_metrics = ring_area_metrics(points, target_radius=self.ring_radius, area_fraction_threshold=self.area_fraction_threshold)
        mapping = best_ring_target_mapping(points, mapped_targets)
        metrics.update(target_metrics)
        metrics.update(ring_metrics)
        metrics.update({f"ring_{key}": value for key, value in mapping.items()})
        metrics.update(self._target_observation_metrics())
        metrics["ring_radius"] = self.ring_radius

        mean_map_err = float(mapping["mean_mapping_error"])
        map_err_limit = max(0.05, self.target_distance_threshold * 6.0)
        height_ok = bool(metrics["ring_area_success"] and mean_map_err < map_err_limit)
        metrics["ring_cable_near_table"] = bool(height_ok)
        metrics["success"] = bool(metrics["ring_area_success"] and height_ok)
        metrics["task_success"] = metrics["success"]
        metrics["ring_asset_status"] = (
            "open_composite_ring_surrogate" if self.cable_model != "rmb" else "open_rmb_cable_surrogate_fallback"
        )
        return metrics

    def _check_success(self):
        return self._compute_metrics()["success"]
