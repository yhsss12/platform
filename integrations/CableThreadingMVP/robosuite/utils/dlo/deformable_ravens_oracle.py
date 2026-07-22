"""
Deformable Ravens 任务的 Oracle（专家）拾放策略。

本模块为 Deformable Ravens 风格的任务提供"oracle"级别的拾放决策，
即给定当前线缆状态和目标状态，选择最优的拾取点和放置点。

核心策略：
  1. shape_pick_place: 用于开放形状（直线、L 形、U 形）
     - 尝试正序和反序匹配，选择平均误差更小的方向
     - 端点优先策略：如果端点误差接近最大误差，优先修正端点

  2. ring_pick_place: 用于闭合形状（环形）
     - 使用 best_ring_target_mapping 找到最优循环匹配
     - 选择匹配后误差最大的点作为拾取目标

  3. oracle_pick_place_for_task: 根据任务名称自动选择策略

辅助功能：
  - resample_polyline: 等弧长重采样
  - planar_joint_targets_from_polyline: 从折线计算关节角目标
"""

import numpy as np

from robosuite.utils.dlo.deformable_ravens_tasks import best_ring_target_mapping


# Deformable Ravens 任务名称集合
DEFORMABLE_RAVENS_TASK_NAMES = {
    "CableShape",
    "CableRing",
}


def is_deformable_ravens_task(env_or_name):
    """判断给定的环境或任务名是否属于 Deformable Ravens 系列。

    Args:
        env_or_name: 环境对象或任务名称字符串。

    Returns:
        bool。
    """
    name = env_or_name if isinstance(env_or_name, str) else env_or_name.__class__.__name__
    return str(name) in DEFORMABLE_RAVENS_TASK_NAMES


def equal_length_targets(points, targets):
    """将目标关键点重采样到与当前关键点相同的数量。

    当目标点数与当前点数不匹配时，使用线性插值重采样。
    这是拾放策略的前提条件：一一对应才能计算距离。
    """
    targets = np.asarray(targets, dtype=float)
    points = np.asarray(points, dtype=float)
    if len(points) == len(targets):
        return targets.copy()
    sample = np.linspace(0, len(targets) - 1, len(points))
    lower = np.floor(sample).astype(int)
    upper = np.ceil(sample).astype(int)
    alpha = sample - lower
    return (1.0 - alpha[:, None]) * targets[lower] + alpha[:, None] * targets[upper]


def shape_pick_place(points, targets):
    """开放形状的拾放决策（直线、L 形、U 形等）。

    算法：
      1. 将目标重采样到与当前相同的点数
      2. 尝试正序和反序两种匹配方向
      3. 选择平均误差更小的方向
      4. 端点优先策略：如果端点误差接近最大误差（>= 80%），
         优先修正端点（因为端点对形状影响最大）
      5. 否则，选择误差最大的点作为拾取目标

    Args:
        points:  当前关键点，shape (N, 3)。
        targets: 目标关键点，shape (M, 3)。

    Returns:
        dict 包含：
          - point_idx: 拾取点的索引
          - target_idx: 对应的目标点索引
          - pick_pos: 拾取位置
          - place_pos: 放置位置
          - mapping_reverse: 是否使用反序匹配
          - mapping_error: 拾取点的匹配误差
          - mean_mapping_error: 平均匹配误差
    """
    points = np.asarray(points, dtype=float)
    targets = equal_length_targets(points, targets)
    candidates = []
    for reverse in (False, True):
        ordered = points[::-1] if reverse else points
        distances = np.linalg.norm(ordered[:, :2] - targets[:, :2], axis=1)
        candidates.append((float(np.mean(distances)), bool(reverse), int(np.argmax(distances)), distances))
    _, reverse, target_idx, distances = min(candidates, key=lambda item: item[0])
    max_point_error = float(distances[target_idx])

    # 端点优先：如果端点误差接近最大误差，优先修正端点
    ep0_err = float(np.linalg.norm(points[0, :2] - targets[0, :2]))
    ep1_err = float(np.linalg.norm(points[-1, :2] - targets[-1, :2]))
    max_ep_err = max(ep0_err, ep1_err)
    endpoint_priority_threshold = 0.8
    if max_ep_err >= max_point_error * endpoint_priority_threshold:
        if ep0_err >= ep1_err:
            point_idx, target_idx = 0, 0
        else:
            point_idx, target_idx = len(points) - 1, len(targets) - 1
        return {
            "point_idx": int(point_idx),
            "target_idx": int(target_idx),
            "pick_pos": points[point_idx].copy(),
            "place_pos": targets[target_idx].copy(),
            "mapping_reverse": bool(reverse),
            "mapping_shift": 0,
            "mapping_error": float(distances[target_idx]),
            "mean_mapping_error": float(np.mean(distances)),
        }

    # 一般情况：选择误差最大的点
    point_idx = len(points) - 1 - target_idx if reverse else target_idx
    return {
        "point_idx": int(point_idx),
        "target_idx": int(target_idx),
        "pick_pos": points[point_idx].copy(),
        "place_pos": targets[target_idx].copy(),
        "mapping_reverse": bool(reverse),
        "mapping_shift": 0,
        "mapping_error": float(distances[target_idx]),
        "mean_mapping_error": float(np.mean(distances)),
    }


def ring_pick_place(points, targets):
    """环形形状的拾放决策。

    与 shape_pick_place 的区别：
      - 环形没有固定的起点，需要考虑循环移位
      - 使用 best_ring_target_mapping 找到最优的 (reverse, shift) 组合
      - 然后选择匹配后误差最大的点作为拾取目标

    Args:
        points:  当前关键点，shape (N, 3)。
        targets: 目标关键点，shape (M, 3)。

    Returns:
        dict（同 shape_pick_place）。
    """
    points = np.asarray(points, dtype=float)
    targets = equal_length_targets(points, targets)
    mapping = best_ring_target_mapping(points, targets)
    # 根据最优匹配的 reverse 和 shift 重排索引
    ordered_indices = np.arange(len(points))
    if mapping["reverse"]:
        ordered_indices = ordered_indices[::-1]
    ordered_indices = np.roll(ordered_indices, shift=mapping["shift"])
    ordered_points = points[ordered_indices]
    distances = np.linalg.norm(ordered_points[:, :2] - targets[:, :2], axis=1)
    target_idx = int(np.argmax(distances))
    point_idx = int(ordered_indices[target_idx])
    return {
        "point_idx": point_idx,
        "target_idx": target_idx,
        "pick_pos": points[point_idx].copy(),
        "place_pos": targets[target_idx].copy(),
        "mapping_reverse": bool(mapping["reverse"]),
        "mapping_shift": int(mapping["shift"]),
        "mapping_error": float(distances[target_idx]),
        "mean_mapping_error": float(mapping["mean_mapping_error"]),
    }


def _is_closed_polyline(targets, tol=1e-6):
    """判断目标折线是否闭合（首尾点距离 < tol）。"""
    targets = np.asarray(targets, dtype=float)
    if len(targets) < 3:
        return False
    return float(np.linalg.norm(targets[0, :2] - targets[-1, :2])) < float(tol)


def oracle_pick_place_for_task(task_name, points, targets):
    """根据任务名称自动选择拾放策略。

    CableRing -> ring_pick_place（环形策略）
    其他任务 -> shape_pick_place（开放形状策略）

    Args:
        task_name: 任务名称字符串。
        points:    当前关键点。
        targets:   目标关键点。

    Returns:
        dict（拾放决策结果）。
    """
    if str(task_name) == "CableRing":
        return ring_pick_place(points, targets)
    return shape_pick_place(points, targets)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def resample_polyline(points, count):
    """将折线重采样到指定数量的点（等弧长插值）。

    使用 numpy.interp 对每个维度分别进行线性插值，
    采样距离在 [0, total_arc_length] 上均匀分布。
    """
    points = np.asarray(points, dtype=float)
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if len(points) == count:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if cumulative[-1] < 1e-9:
        return np.repeat(points[:1], count, axis=0)
    samples = np.linspace(0.0, cumulative[-1], count)
    out = np.empty((count, points.shape[1]), dtype=float)
    for dim in range(points.shape[1]):
        out[:, dim] = np.interp(samples, cumulative, points[:, dim])
    return out


def planar_joint_targets_from_polyline(points):
    """从折线计算平面关节角目标。

    将折线的每个线段方向转换为关节角（yaw 角的差分），
    用于平面机械臂的关节空间控制。

    Returns:
        dict 包含：
          - root_pos: 起点位置
          - root_quat: 起点的四元数（只含 yaw 旋转）
          - joint_yaws: 相邻线段间的角度差（关节角）
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("At least two points are required")
    deltas = np.diff(points[:, :3], axis=0)
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])
    root_yaw = float(yaws[0])
    joint_yaws = np.diff(yaws)
    # 归一化到 [-pi, pi]
    joint_yaws = (joint_yaws + np.pi) % (2.0 * np.pi) - np.pi
    return {
        "root_pos": points[0].copy(),
        "root_quat": np.array([np.cos(0.5 * root_yaw), 0.0, 0.0, np.sin(0.5 * root_yaw)], dtype=float),
        "joint_yaws": joint_yaws.astype(float),
    }
