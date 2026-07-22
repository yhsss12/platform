"""
环境状态适配器层（鸭子类型分发）。

本模块是 dloBench 五层架构中 L2（Adapter）层的一部分。
它提供统一的接口来从不同后端的环境中提取线缆状态和指标，
而不需要知道底层是 robosuite、SoftGym 还是其他仿真器。

鸭子类型分发模式：
  每个函数通过 hasattr 检查环境对象是否具有特定方法，
  按优先级尝试不同的方法名。这种模式允许：
    1. 新后端只需实现约定方法即可接入
    2. 不同后端可以有不同的内部实现
    3. 调用方无需关心底层差异

典型用法：
  keypoints = get_env_cable_keypoints(env)  # 不关心 env 是什么类型
"""

import numpy as np


def get_env_cable_keypoints(env):
    """从环境中获取线缆关键点坐标。

    按优先级尝试两种方法名：
      1. get_cable_keypoints() — 推荐的公开 API
      2. _get_cable_points()  — 旧版兼容接口

    Returns:
        shape (N, 3) 的 float64 数组。

    Raises:
        AttributeError: 环境未暴露任何一种获取方法时。
    """
    if hasattr(env, "get_cable_keypoints"):
        return np.asarray(env.get_cable_keypoints(), dtype=float)
    if hasattr(env, "_get_cable_points"):
        return np.asarray(env._get_cable_points(), dtype=float)
    raise AttributeError(f"{type(env).__name__} does not expose _get_cable_points()")


def get_env_metrics(env):
    """从环境中获取当前的 DLO 任务指标字典。

    按优先级尝试：
      1. get_dlo_metrics() — 推荐接口
      2. _compute_metrics() — 旧版兼容接口

    Returns:
        dict，包含各项指标。如果环境不支持，返回空字典。
    """
    if hasattr(env, "get_dlo_metrics"):
        return dict(env.get_dlo_metrics())
    if hasattr(env, "_compute_metrics"):
        return dict(env._compute_metrics())
    return {}


def get_env_success(env):
    """从环境中获取任务是否成功的布尔值。

    按优先级尝试：
      1. get_task_success() — 推荐接口
      2. _check_success()  — 旧版兼容接口
      3. 从 get_env_metrics() 中读取 "success" 字段（兜底方案）

    Returns:
        bool。
    """
    if hasattr(env, "get_task_success"):
        return bool(env.get_task_success())
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    metrics = get_env_metrics(env)
    return bool(metrics.get("success", False))
