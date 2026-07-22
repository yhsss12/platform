"""
SoftGym 风格的绳索任务指标与目标生成。

本模块实现源自 SoftGym 基准的绳索操作任务：
  - rope_flatten: 将绳索拉直（端点距离接近绳长）
  - character shaping: 将绳索摆成字母形状（S/O/M/C/U）

核心功能：
  1. 拉直指标（rope_flatten_metrics）：端点距离与绳长的误差
  2. 字母目标生成（generate_softgym_character_target）
  3. 字母配置指标（softgym_configuration_metrics）：最优二部图匹配
  4. 匈牙利算法匹配（best_bipartite_matching）

数学基础：
  - SoftGym 的性能指标 = -(端点距离误差)，归一化到 [0, 1]
  - 匈牙利算法（scipy.optimize.linear_sum_assignment）用于最优匹配
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from robosuite.utils.dlo.cable_metrics import validate_keypoints


# SoftGym 支持的字母形状集合
SOFTGYM_ROPE_CHARACTERS = ("S", "O", "M", "C", "U")


def rope_flatten_metrics(keypoints, *, rope_length, performance_init=None, success_threshold=0.9):
    """SoftGym 风格的绳索拉直指标。

    性能定义：performance = -(端点距离误差) = -|端点距离 - 绳长|
    理想情况：端点距离 == 绳长（完全拉直），performance = 0。
    最差情况：端点距离 = 0（绳索对折），performance = -rope_length。

    归一化：normalized_performance = (performance - min) / (max - min)
    其中 min = -rope_length, max = 0，所以归一化后值域 [0, 1]。

    Args:
        keypoints: 当前绳索关键点。
        rope_length: 绳索的物理长度。
        performance_init: 初始性能值（用于计算相对于初始状态的改善）。
        success_threshold: 归一化性能的判定阈值。

    Returns:
        dict 包含原始性能、归一化性能和成功判定。
    """
    points = validate_keypoints(keypoints)
    endpoint_distance = float(np.linalg.norm(points[-1] - points[0]))
    distance_to_length_error = float(abs(endpoint_distance - float(rope_length)))
    performance = -distance_to_length_error
    reward_min = -float(rope_length)
    reward_range = max(float(rope_length), 1e-8)
    normalized_performance = (performance - reward_min) / reward_range
    if performance_init is not None:
        normalized_performance_from_init = (performance - float(performance_init)) / max(0.0 - float(performance_init), 1e-8)
    else:
        normalized_performance_from_init = normalized_performance
    return {
        "endpoint_distance": endpoint_distance,
        "rope_length": float(rope_length),
        "distance_to_length_error": distance_to_length_error,
        "performance": float(performance),
        "normalized_performance": float(normalized_performance),
        "normalized_performance_from_init": float(normalized_performance_from_init),
        "success_threshold": float(success_threshold),
        "success": bool(normalized_performance >= float(success_threshold)),
    }


def generate_softgym_character_target(
    character,
    num_points,
    *,
    center=(0.0, 0.0, 0.808),
    scale=0.16,
    yaw=0.0,
):
    """生成 SoftGym 字母形状的目标关键点。

    算法：
      1. 获取字母的顶点坐标（_character_vertices）
      2. 等弧长采样为 num_points 个点
      3. 归一化到 [-scale/2, scale/2] 范围
      4. 旋转 yaw 角度
      5. 平移到 center

    Args:
        character: 字母名称（S/O/M/C/U）。
        num_points: 采样点数量。
        center: 目标中心坐标。
        scale: 字母的物理尺寸。
        yaw: 旋转角度（弧度）。

    Returns:
        shape (num_points, 3) 的目标关键点数组。
    """
    character = str(character).upper()
    if character not in SOFTGYM_ROPE_CHARACTERS:
        raise ValueError(f"Unsupported SoftGym rope character: {character}")
    vertices = _character_vertices(character)
    xy = _sample_polyline(vertices, int(num_points))
    # 居中并归一化到指定 scale
    xy = xy - np.mean(xy, axis=0, keepdims=True)
    max_extent = max(float(np.max(np.ptp(xy, axis=0))), 1e-8)
    xy = xy / max_extent * float(scale)
    # 旋转 + 平移
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    center = np.asarray(center, dtype=float)
    out = np.zeros((int(num_points), 3), dtype=float)
    out[:, :2] = xy @ rot.T + center[:2]
    out[:, 2] = center[2]
    return out


def softgym_configuration_metrics(
    keypoints,
    target_keypoints,
    *,
    performance_init=None,
    reward_type="bigraph",
    success_threshold=0.85,
):
    """SoftGym 风格的配置匹配指标。

    计算当前绳索关键点与目标形状的匹配程度，支持两种匹配策略：
      - "index": 按索引一一对应（第 i 个点对第 i 个目标）
      - "bigraph": 使用匈牙利算法找到全局最优匹配

    匈牙利匹配能处理点的顺序不确定的情况（如环形、字母形状），
    但计算复杂度 O(N^3) 比按索引匹配 O(N) 高。

    Args:
        keypoints: 当前关键点。
        target_keypoints: 目标关键点。
        performance_init: 初始性能（用于归一化）。
        reward_type: "index" 或 "bigraph"。
        success_threshold: 归一化性能的判定阈值。

    Returns:
        dict 包含两种匹配的误差和成功判定。
    """
    points = validate_keypoints(keypoints)
    targets = validate_keypoints(target_keypoints)
    if len(points) != len(targets):
        targets = resample_keypoints(targets, len(points))
    # 按索引匹配：直接计算对应点的距离
    index_distances = np.linalg.norm(points[:, :2] - targets[:, :2], axis=1)
    index_performance = -float(np.mean(index_distances))
    # 匈牙利匹配：全局最优
    assignment = best_bipartite_matching(points, targets)
    bigraph_performance = -float(assignment["mean_error"])
    performance = index_performance if reward_type == "index" else bigraph_performance
    if performance_init is None:
        normalized = 0.0
    else:
        normalized = (performance - float(performance_init)) / max(0.0 - float(performance_init), 1e-8)
    return {
        "performance": float(performance),
        "normalized_performance": float(normalized),
        "index_mean_error": float(np.mean(index_distances)),
        "index_max_error": float(np.max(index_distances)),
        "bigraph_mean_error": float(assignment["mean_error"]),
        "bigraph_max_error": float(assignment["max_error"]),
        "bigraph_assignment": assignment["assignment"],
        "reward_type": str(reward_type),
        "success_threshold": float(success_threshold),
        "success": bool(normalized >= float(success_threshold)),
    }


# ---------------------------------------------------------------------------
# 匈牙利算法匹配
# ---------------------------------------------------------------------------

def best_bipartite_matching(keypoints, target_keypoints):
    """使用匈牙利算法（Kuhn-Munkres）找到最优的一对一匹配。

    构建 N x N 的代价矩阵（XY 距离），然后调用 scipy 的
    linear_sum_assignment 求解最小代价的完美匹配。

    时间复杂度 O(N^3)，适合 N < 1000 的场景。

    Args:
        keypoints: 当前关键点，shape (N, 3)。
        target_keypoints: 目标关键点，shape (N, 3)。

    Returns:
        dict 包含：
          - assignment: 每个当前点匹配到的目标点索引
          - mean_error: 匹配后的平均距离
          - max_error: 匹配后的最大距离
          - worst_point_index: 距离最大的当前点索引
    """
    points = validate_keypoints(keypoints)
    targets = validate_keypoints(target_keypoints)
    if len(points) != len(targets):
        raise ValueError("Bipartite matching requires equal length keypoints and target_keypoints")
    # 代价矩阵：每对点的 XY 距离
    cost_matrix = np.linalg.norm(points[:, None, :2] - targets[None, :, :2], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = np.empty(len(points), dtype=int)
    assignment[row_ind] = col_ind
    assigned = cost_matrix[np.arange(len(points)), assignment]
    return {
        "assignment": [int(i) for i in assignment],
        "mean_error": float(np.mean(assigned)),
        "max_error": float(np.max(assigned)),
        "worst_point_index": int(np.argmax(assigned)),
    }


def resample_keypoints(keypoints, num_points):
    """将关键点重采样到指定数量（线性插值）。

    当源和目标的关键点数量不匹配时使用。
    在索引空间 [0, N-1] 上均匀采样，用线性插值计算新坐标。
    """
    points = validate_keypoints(keypoints)
    if len(points) == int(num_points):
        return points
    sample = np.linspace(0, len(points) - 1, int(num_points))
    lower = np.floor(sample).astype(int)
    upper = np.ceil(sample).astype(int)
    alpha = sample - lower
    return (1.0 - alpha[:, None]) * points[lower] + alpha[:, None] * points[upper]


# ---------------------------------------------------------------------------
# 字母形状顶点定义
# ---------------------------------------------------------------------------

def _character_vertices(character):
    """返回字母形状的顶点坐标（2D，归一化到约 [-0.55, 0.55] 范围）。

    每个字母由一系列顶点定义，相邻顶点之间用直线段连接。
    形状经过设计使得绳索可以物理实现（无自交叉、连续路径）。

    S: 蛇形曲线（8 个顶点）
    O: 椭圆（32 个采样点）
    M: 山形（5 个顶点）
    C: 开口圆弧（32 个采样点，270 度）
    U: 马蹄形（6 个顶点）
    """
    if character == "S":
        return [(0.5, 0.5), (0.0, 0.65), (-0.5, 0.45), (-0.35, 0.05), (0.3, -0.05), (0.5, -0.45), (0.0, -0.65), (-0.5, -0.45)]
    if character == "O":
        theta = np.linspace(0.0, 2.0 * np.pi, 33)
        return [(0.5 * np.cos(t), 0.65 * np.sin(t)) for t in theta]
    if character == "M":
        return [(-0.55, -0.6), (-0.55, 0.6), (0.0, -0.05), (0.55, 0.6), (0.55, -0.6)]
    if character == "C":
        theta = np.linspace(0.25 * np.pi, 1.75 * np.pi, 32)
        return [(0.55 * np.cos(t), 0.65 * np.sin(t)) for t in theta]
    # U 形
    return [(-0.5, 0.55), (-0.5, -0.35), (-0.25, -0.6), (0.25, -0.6), (0.5, -0.35), (0.5, 0.55)]


def _sample_polyline(vertices, num_points):
    """沿折线等弧长采样指定数量的点。

    与 deformable_ravens_tasks._sample_polyline 相同的实现。
    在 [0, total_length] 上均匀采样，通过二分搜索定位段，
    线性插值计算坐标。
    """
    vertices = np.asarray(vertices, dtype=float)
    segs = np.diff(vertices, axis=0)
    seg_lens = np.linalg.norm(segs, axis=1)
    total = float(np.sum(seg_lens))
    if total <= 0:
        raise ValueError("polyline length must be positive")
    samples = np.linspace(0.0, total, int(num_points))
    out = []
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    for dist in samples:
        idx = min(np.searchsorted(cumulative, dist, side="right") - 1, len(segs) - 1)
        local = 0.0 if seg_lens[idx] <= 0 else (dist - cumulative[idx]) / seg_lens[idx]
        out.append(vertices[idx] + local * segs[idx])
    return np.asarray(out, dtype=float)
