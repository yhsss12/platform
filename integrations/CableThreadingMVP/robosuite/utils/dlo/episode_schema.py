"""
Episode 数据模式定义与校验。

本模块定义 dloBench 中 episode 数据的标准格式，并提供校验函数。
Episode 是一次完整的任务执行记录，包含观测、动作、奖励等时序数据。

设计目标：
  1. 统一数据格式：不同后端（robosuite、SoftGym 等）的 episode 数据
     都遵循相同的 schema，便于跨后端分析和离线学习。
  2. 严格校验：在数据写入/读取时检查完整性，提前发现格式问题。
  3. 版本管理：通过 schema_version 字段追踪格式变化。

Episode 数据结构：
  {
    "observations": list[dict],    # N+1 个观测（初始 + 每步后）
    "actions": list[ndarray],      # N 个动作
    "rewards": list[float],        # N 个奖励
    "terminated": list[bool],      # N 个终止标志（任务成功）
    "truncated": list[bool],       # N 个截断标志（超时等非成功终止）
    "success": bool,               # 最终成功状态
    "metrics": list[dict],         # N 个指标字典
    "metadata": dict,              # 元数据（环境名、机器人、种子等）
  }

长度约定：
  - observations: N+1（初始观测 + 每步执行后的观测）
  - actions / rewards / terminated / truncated / metrics: N（每步一个）
"""

import numpy as np


# 版本号常量
DEFAULT_OBSERVATION_SCHEMA_VERSION = "dlo_obs_v1"
DEFAULT_ACTION_SCHEMA_VERSION = "robosuite_action_v1"

# Episode 字典必须包含的顶层 key
REQUIRED_EPISODE_KEYS = (
    "observations",
    "actions",
    "rewards",
    "terminated",
    "truncated",
    "success",
    "metrics",
    "metadata",
)

# Metadata 字典必须包含的 key
REQUIRED_METADATA_KEYS = (
    "env_name",
    "robot",
    "controller",
    "backend",
    "seed",
    "horizon",
    "scene_randomization",
    "policy",
    "episode_horizon",
    "observation_schema_version",
    "action_schema_version",
)


class EpisodeSchemaError(ValueError):
    """Episode 数据模式校验失败时抛出的异常。"""
    pass


def _fail(message):
    """抛出模式校验异常。"""
    raise EpisodeSchemaError(message)


def _as_sequence(value, name):
    """确保值是序列类型（list / ndarray 等）。"""
    if isinstance(value, np.ndarray):
        return value
    if not hasattr(value, "__len__"):
        _fail(f"{name} must be a sequence")
    return value


def _check_observation_keys(observations):
    """校验观测序列的一致性。

    检查：
      1. 观测不能为空
      2. 每个观测必须是 dict
      3. 所有观测的 key 集合必须相同
      4. 每个 key 对应的 shape 必须在所有时间步保持一致

    Returns:
        (keys, shapes) — 排序后的 key 元组和对应的 shape 字典
    """
    if not observations:
        _fail("observations must not be empty")
    if not isinstance(observations[0], dict):
        _fail("observations[0] must be a dict")
    keys = tuple(sorted(observations[0].keys()))
    shapes = {key: np.asarray(observations[0][key]).shape for key in keys}
    for idx, obs in enumerate(observations):
        if not isinstance(obs, dict):
            _fail(f"observations[{idx}] must be a dict")
        if tuple(sorted(obs.keys())) != keys:
            _fail(f"observation keys changed at step {idx}")
        for key in keys:
            shape = np.asarray(obs[key]).shape
            if shape != shapes[key]:
                _fail(f"observation shape for {key} changed at step {idx}: {shape} != {shapes[key]}")
    return keys, shapes


def validate_episode(episode, expected_action_dim=None, expected_metric_keys=None):
    """校验 episode 数据的完整性和一致性。

    检查项目：
      1. 必需的顶层 key 是否存在
      2. 长度约束：observations = actions + 1，其他序列 = actions
      3. 观测的 key 和 shape 一致性
      4. 动作必须是 1D 向量，shape 一致，无 NaN/inf
      5. 指标的 key 集合一致性
      6. Metadata 的必需 key 是否存在

    Args:
        episode: 要校验的 episode 字典。
        expected_action_dim: 期望的动作维度（可选）。
        expected_metric_keys: 期望的指标 key 集合（可选）。

    Returns:
        dict 包含校验结果的摘要信息（num_steps、observation_keys 等）。
    """
    if not isinstance(episode, dict):
        _fail("episode must be a dict")
    for key in REQUIRED_EPISODE_KEYS:
        if key not in episode:
            _fail(f"missing episode key: {key}")

    observations = _as_sequence(episode["observations"], "observations")
    actions = _as_sequence(episode["actions"], "actions")
    rewards = _as_sequence(episode["rewards"], "rewards")
    terminated = _as_sequence(episode["terminated"], "terminated")
    truncated = _as_sequence(episode["truncated"], "truncated")
    metrics = _as_sequence(episode["metrics"], "metrics")

    n_steps = len(actions)
    if n_steps == 0:
        _fail("actions must not be empty")
    # observations = N+1, 其他 = N
    if len(observations) != n_steps + 1:
        _fail(f"observations length must be actions length + 1, got {len(observations)} and {n_steps}")
    for name, seq in (("rewards", rewards), ("terminated", terminated), ("truncated", truncated), ("metrics", metrics)):
        if len(seq) != n_steps:
            _fail(f"{name} length must equal actions length, got {len(seq)} and {n_steps}")

    obs_keys, obs_shapes = _check_observation_keys(observations)

    # 动作校验
    first_action_shape = np.asarray(actions[0]).shape
    if len(first_action_shape) != 1:
        _fail(f"actions[0] must be a flat vector, got shape {first_action_shape}")
    if expected_action_dim is not None and first_action_shape[0] != int(expected_action_dim):
        _fail(f"action dim mismatch: {first_action_shape[0]} != {expected_action_dim}")
    for idx, action in enumerate(actions):
        arr = np.asarray(action)
        if arr.shape != first_action_shape:
            _fail(f"action shape changed at step {idx}: {arr.shape} != {first_action_shape}")
        if not np.all(np.isfinite(arr)):
            _fail(f"action contains NaN or inf at step {idx}")

    # 指标校验
    metric_keys = tuple(sorted(metrics[0].keys())) if metrics else ()
    expected = tuple(sorted(expected_metric_keys)) if expected_metric_keys is not None else metric_keys
    for idx, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            _fail(f"metrics[{idx}] must be a dict")
        if tuple(sorted(metric.keys())) != expected:
            _fail(f"metric keys changed at step {idx}")

    # Metadata 校验
    metadata = episode["metadata"]
    if not isinstance(metadata, dict):
        _fail("metadata must be a dict")
    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            _fail(f"missing metadata key: {key}")

    return {
        "num_steps": n_steps,
        "observation_keys": obs_keys,
        "observation_shapes": obs_shapes,
        "action_dim": first_action_shape[0],
        "metric_keys": expected,
    }


def validate_metadata(metadata):
    """单独校验 metadata 字典。

    除了检查必需 key 外，还校验：
      - scene_randomization 必须是 "fixed" 或 "random"
      - policy 必须非空
    """
    if not isinstance(metadata, dict):
        _fail("metadata must be a dict")
    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            _fail(f"missing metadata key: {key}")
    if str(metadata["scene_randomization"]) not in {"fixed", "random"}:
        _fail("metadata.scene_randomization must be fixed or random")
    if not str(metadata["policy"]):
        _fail("metadata.policy must be non-empty")
    return metadata


def formal_metadata(
    *,
    env_name,
    robot,
    controller,
    seed,
    horizon,
    scene_randomization,
    policy,
    backend="robosuite",
    observation_schema_version=DEFAULT_OBSERVATION_SCHEMA_VERSION,
    action_schema_version=DEFAULT_ACTION_SCHEMA_VERSION,
    **extra,
):
    """构造标准化的 metadata 字典。

    填充所有必需字段，并通过 validate_metadata 校验。
    支持通过 **extra 传入额外字段。

    Args:
        env_name: 环境名称。
        robot: 机器人型号。
        controller: 控制器名称。
        seed: 随机种子。
        horizon: 最大步数。
        scene_randomization: "fixed" 或 "random"。
        policy: 策略名称。
        backend: 后端名称（默认 "robosuite"）。
        observation_schema_version: 观测 schema 版本。
        action_schema_version: 动作 schema 版本。
        **extra: 额外的 metadata 字段。

    Returns:
        校验通过的 metadata 字典。
    """
    metadata = {
        "env_name": env_name,
        "robot": robot,
        "controller": controller or "default",
        "backend": backend,
        "seed": int(seed),
        "horizon": int(horizon),
        "episode_horizon": int(horizon),
        "scene_randomization": str(scene_randomization),
        "policy": str(policy),
        "observation_schema_version": observation_schema_version,
        "action_schema_version": action_schema_version,
    }
    metadata.update(extra)
    return validate_metadata(metadata)


def validate_transition_trajectories(trajectories, *, metadata, expected_action_dim=None):
    """校验转换轨迹（transition trajectories）数据。

    转换轨迹是 RL 中常用的 (obs, action, reward, next_obs, done) 五元组格式。
    每条轨迹是一个 step 列表，每个 step 包含上述 5 个字段。

    Args:
        trajectories: 轨迹列表，每条轨迹是 step 字典的列表。
        metadata: 元数据字典。
        expected_action_dim: 期望的动作维度。

    Returns:
        dict 包含 num_episodes、num_steps、action_dim。
    """
    validate_metadata(metadata)
    if not trajectories:
        _fail("trajectories must not be empty")

    action_dim = None
    num_steps = 0
    for episode_idx, trajectory in enumerate(trajectories):
        if not trajectory:
            _fail(f"trajectory {episode_idx} must not be empty")
        for step_idx, step in enumerate(trajectory):
            for key in ("obs", "next_obs", "action", "reward", "done"):
                if key not in step:
                    _fail(f"trajectory {episode_idx} step {step_idx} missing key: {key}")
            obs = np.asarray(step["obs"])
            next_obs = np.asarray(step["next_obs"])
            action = np.asarray(step["action"])
            if obs.shape != next_obs.shape:
                _fail(f"obs / next_obs shape mismatch at trajectory {episode_idx} step {step_idx}")
            if action.ndim != 1:
                _fail(f"action must be a flat vector at trajectory {episode_idx} step {step_idx}")
            if action_dim is None:
                action_dim = int(action.shape[0])
            elif action.shape[0] != action_dim:
                _fail(f"action dim changed at trajectory {episode_idx} step {step_idx}")
            if expected_action_dim is not None and action.shape[0] != int(expected_action_dim):
                _fail(f"action dim mismatch: {action.shape[0]} != {expected_action_dim}")
            if not np.all(np.isfinite(obs)):
                _fail(f"obs contains NaN or inf at trajectory {episode_idx} step {step_idx}")
            if not np.all(np.isfinite(next_obs)):
                _fail(f"next_obs contains NaN or inf at trajectory {episode_idx} step {step_idx}")
            if not np.all(np.isfinite(action)):
                _fail(f"action contains NaN or inf at trajectory {episode_idx} step {step_idx}")
            if not np.isfinite(float(step["reward"])):
                _fail(f"reward contains NaN or inf at trajectory {episode_idx} step {step_idx}")
            num_steps += 1

    return {
        "num_episodes": len(trajectories),
        "num_steps": num_steps,
        "action_dim": action_dim,
    }
