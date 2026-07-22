"""
Deformable Ravens 任务的目标生成与指标计算。

本模块实现 dloBench 中源自 Deformable Ravens 基准的任务：
  - CableShape: 将线缆摆放成多边形形状（通过 target_visible 控制目标可见性）
  - CableRing: 将线缆摆成环形（通过 target_visible 控制目标可见性）

核心功能：
  1. 目标形状生成（generate_polyline_target / generate_ring_target）
  2. 关键点覆盖率指标（target_keypoint_metrics）
  3. 环形面积指标（ring_area_metrics）
  4. 环形最优匹配（best_ring_target_mapping）

数学工具：
  - 等弧长采样（_sample_polyline）
  - Andrew 凸包算法（_convex_hull_xy）
  - Shoelace 面积公式（polygon_area_xy）
"""

import numpy as np

from robosuite.utils.dlo.cable_metrics import validate_keypoints


# ---------------------------------------------------------------------------
# 目标形状生成
# ---------------------------------------------------------------------------

def generate_polyline_target(
    num_points,
    *,
    num_sides,
    length,
    center=(0.0, 0.0, 0.8265),
    yaw=0.0,
    cutoff=None,
):
    """生成多边形形状的目标关键点。

    根据边数生成不同形状的目标：
      - 1 边：直线（L 形）
      - 2 边：L 形折线
      - 3 边：U 形折线
      - 4 边：矩形（闭合）

    算法：
      1. 根据 num_sides 和 length 计算多边形顶点
      2. 用等弧长采样（_sample_polyline）将顶点间插值为 num_points 个点
      3. 应用 yaw 旋转和平移到指定 center

    Args:
        num_points:  采样点数量。
        num_sides:   多边形边数（1~4）。
        length:      总长度。
        center:      目标中心坐标。
        yaw:         旋转角度（弧度）。
        cutoff:      控制多边形各边长度分配的参数。

    Returns:
        shape (num_points, 3) 的目标关键点数组。
    """
    num_points = int(num_points)
    num_sides = int(num_sides)
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if num_sides not in {1, 2, 3, 4}:
        raise ValueError("num_sides must be in {1, 2, 3, 4}")

    # 根据边数生成多边形顶点（2D，后续旋转+平移）
    if num_sides == 1:
        vertices = [(-0.5 * length, 0.0), (0.5 * length, 0.0)]
    else:
        if cutoff is None:
            cutoff = max(2, min(num_points - 2, num_points // 2))
        alpha = float(cutoff) / float(num_points)
        if num_sides == 2:
            # L 形：一条短边 + 一条长边
            lx = length * alpha
            ly = length * (1.0 - alpha)
            vertices = [(0.5 * lx, -0.5 * ly), (-0.5 * lx, -0.5 * ly), (-0.5 * lx, 0.5 * ly)]
        elif num_sides == 3:
            # U 形：三边
            lx = length * alpha
            ly = 0.5 * length * (1.0 - alpha)
            vertices = [(0.5 * lx, 0.5 * ly), (0.5 * lx, -0.5 * ly), (-0.5 * lx, -0.5 * ly), (-0.5 * lx, 0.5 * ly)]
        else:
            # 矩形：四边闭合
            lx = 0.5 * length * alpha
            ly = 0.5 * length * (1.0 - alpha)
            vertices = [
                (0.5 * lx, -0.5 * ly),
                (-0.5 * lx, -0.5 * ly),
                (-0.5 * lx, 0.5 * ly),
                (0.5 * lx, 0.5 * ly),
                (0.5 * lx, -0.5 * ly),
            ]

    # 等弧长采样 + 旋转 + 平移
    xy = _sample_polyline(vertices, num_points)
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    center = np.asarray(center, dtype=float)
    out = np.zeros((num_points, 3), dtype=float)
    out[:, :2] = xy @ rot.T + center[:2]
    out[:, 2] = center[2]
    return out


def generate_ring_target(num_points, *, radius=0.075, center=(0.0, 0.0, 0.8265), yaw=0.0):
    """生成圆形（环形）目标关键点。

    在 XY 平面上生成均匀分布在圆周上的点，所有点的 z 坐标相同。

    Args:
        num_points: 采样点数量。
        radius:     圆的半径（米）。
        center:     圆心坐标。
        yaw:        起始角度偏移（弧度）。

    Returns:
        shape (num_points, 3) 的目标关键点数组。
    """
    theta = np.linspace(0.0, 2.0 * np.pi, int(num_points), endpoint=False) + float(yaw)
    center = np.asarray(center, dtype=float)
    points = np.zeros((int(num_points), 3), dtype=float)
    points[:, 0] = center[0] + float(radius) * np.cos(theta)
    points[:, 1] = center[1] + float(radius) * np.sin(theta)
    points[:, 2] = center[2]
    return points


# ---------------------------------------------------------------------------
# 覆盖率指标
# ---------------------------------------------------------------------------

def target_keypoint_metrics(keypoints, target_keypoints, *, distance_threshold):
    """计算当前关键点对目标关键点的覆盖情况。

    对每个当前关键点，找到最近的目标关键点。如果距离 <= distance_threshold，
    则该点被认为"已覆盖"。

    算法：
      1. 计算所有 (当前点, 目标点) 对的 XY 距离矩阵，shape (N, M)
      2. 对每个当前点，取到最近目标点的距离
      3. 统计覆盖比例和误差

    Returns:
        dict 包含：
          - target_coverage: 覆盖比例（0~1）
          - target_covered_count: 已覆盖点数
          - target_total_count: 总点数
          - mean_keypoint_error: 平均最近距离
          - max_keypoint_error: 最大最近距离
          - distance_threshold: 使用的阈值
    """
    points = validate_keypoints(keypoints)
    targets = validate_keypoints(target_keypoints)
    # 距离矩阵：每个当前点到每个目标点的 XY 距离
    distances = np.linalg.norm(points[:, None, :2] - targets[None, :, :2], axis=2)
    nearest = np.min(distances, axis=1)
    covered = nearest <= float(distance_threshold)
    return {
        "target_coverage": float(np.mean(covered)),
        "target_covered_count": int(np.sum(covered)),
        "target_total_count": int(len(points)),
        "mean_keypoint_error": float(np.mean(nearest)),
        "max_keypoint_error": float(np.max(nearest)),
        "distance_threshold": float(distance_threshold),
    }


# ---------------------------------------------------------------------------
# 面积指标（环形任务专用）
# ---------------------------------------------------------------------------

def polygon_area_xy(points):
    """计算点集在 XY 平面上的凸包面积。

    使用 Shoelace 公式（鞋带公式）计算凸包面积：
      Area = 0.5 * |sum(x_i * y_{i+1} - y_i * x_{i+1})|

    如果点数不足 3，返回 0.0。
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 0.0
    xy = pts[:, :2]
    hull = _convex_hull_xy(xy)
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def ring_area_metrics(keypoints, *, target_radius, area_fraction_threshold=0.75):
    """计算环形任务的面积指标。

    思路：如果线缆正确摆成环形，其凸包面积应接近目标圆的面积。
    面积分数 = 实际凸包面积 / 目标圆面积。

    Args:
        keypoints: 当前线缆关键点。
        target_radius: 目标环的半径。
        area_fraction_threshold: 面积分数的判定阈值。

    Returns:
        dict 包含凸包面积、目标面积、面积分数和是否达标。
    """
    points = validate_keypoints(keypoints)
    area = polygon_area_xy(points)
    best_area = float(np.pi * float(target_radius) ** 2)
    fraction = 0.0 if best_area <= 0 else area / best_area
    return {
        "convex_hull_area": area,
        "best_possible_area": best_area,
        "area_fraction": float(fraction),
        "area_fraction_threshold": float(area_fraction_threshold),
        "ring_area_success": bool(fraction >= float(area_fraction_threshold)),
    }


# ---------------------------------------------------------------------------
# 环形最优匹配
# ---------------------------------------------------------------------------

def best_ring_target_mapping(keypoints, target_keypoints):
    """找到环形目标的最优循环匹配。

    环形没有固定的"起点"，所以需要尝试所有可能的循环移位和正反序，
    找到平均映射误差最小的匹配方案。

    算法：暴力枚举 2N 种可能（N 个移位 x 2 种方向），
    对每种方案计算 mean(||points_i - targets_i||)。

    Args:
        keypoints: 当前关键点，shape (N, 3)。
        target_keypoints: 目标关键点，shape (N, 3)。

    Returns:
        dict 包含最优的 reverse/shift/mean_mapping_error/max_mapping_error。
    """
    points = validate_keypoints(keypoints)
    targets = validate_keypoints(target_keypoints)
    if len(points) != len(targets):
        raise ValueError("ring matching requires equal number of keypoints and target_keypoints")
    n = len(points)
    best = None
    # 尝试正序和反序
    for reverse in (False, True):
        ordered = points[::-1] if reverse else points
        # 尝试所有循环移位
        for shift in range(n):
            rolled = np.roll(ordered, shift=shift, axis=0)
            distances = np.linalg.norm(rolled[:, :2] - targets[:, :2], axis=1)
            mean_error = float(np.mean(distances))
            if best is None or mean_error < best["mean_mapping_error"]:
                best = {
                    "reverse": bool(reverse),
                    "shift": int(shift),
                    "mean_mapping_error": mean_error,
                    "max_mapping_error": float(np.max(distances)),
                    "worst_target_index": int(np.argmax(distances)),
                }
    return best


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _sample_polyline(vertices, num_points):
    """沿折线等弧长采样指定数量的点。

    算法：
      1. 计算每段的长度和累积长度
      2. 在 [0, total_length] 上均匀生成 num_points 个采样距离
      3. 对每个采样距离，找到对应的线段（二分搜索）和段内参数
      4. 线性插值得到采样点坐标

    这保证了采样点在弧长上均匀分布，而非在参数空间均匀分布。
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
        # 找到 dist 落在哪一段
        idx = min(np.searchsorted(cumulative, dist, side="right") - 1, len(segs) - 1)
        # 段内参数（0~1）
        local = 0.0 if seg_lens[idx] <= 0 else (dist - cumulative[idx]) / seg_lens[idx]
        out.append(vertices[idx] + local * segs[idx])
    return np.asarray(out, dtype=float)


def _convex_hull_xy(points):
    """Andrew 凸包算法（单调链法），计算 XY 平面上的凸包。

    算法步骤：
      1. 按 x 坐标排序（x 相同时按 y）
      2. 从左到右构建下凸壳（lower hull）
      3. 从右到左构建上凸壳（upper hull）
      4. 合并得到完整凸包

    时间复杂度 O(n log n)，主要来自排序。

    Returns:
        shape (M, 2) 的凸包顶点数组，按逆时针排列。
    """
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float)[:, :2])))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        """计算向量 OA 和 OB 的叉积（2D）。正值表示逆时针。"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 构建下凸壳
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # 构建上凸壳
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    # 去掉首尾重复点
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)
