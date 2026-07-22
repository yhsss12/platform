"""
随机动作 rollout 工具。

本模块提供在 robosuite 环境中执行随机动作序列的功能，
主要用于：
  1. 环境冒烟测试：验证环境能正常 step
  2. 基线性能评估：随机策略的成功率作为下界参考
  3. 数据收集：收集随机轨迹用于离线分析

随机种子保证可复现性。
"""

import numpy as np

from robosuite.utils.dlo.cable_state import get_env_metrics, get_env_success


def sample_random_action(env, rng):
    """在环境的动作空间中均匀采样一个随机动作。

    使用环境的 action_spec 获取动作的上下界，
    在 [low, high] 范围内均匀采样。

    Args:
        env: robosuite 环境。
        rng: numpy 随机数生成器（numpy.random.Generator）。

    Returns:
        随机动作向量。
    """
    low, high = env.action_spec
    return rng.uniform(low, high)


def rollout_random_actions(env, steps, seed=0, collect_episode=False):
    """在环境中执行指定步数的随机动作 rollout。

    每一步：
      1. 采样一个随机动作
      2. 执行 env.step(action)
      3. 收集指标和成功状态
      4. 如果 episode 结束（done=True），提前终止

    Args:
        env: robosuite 环境。
        steps: 最大步数。
        seed: 随机种子。
        collect_episode: 是否收集完整的 episode 数据
            （observations、actions、rewards 等），用于离线学习。

    Returns:
        (rows, episode)
          - rows: list of dict，每步的指标记录
          - episode: 如果 collect_episode=True，返回完整的 episode 字典；
                     否则返回 None
    """
    rng = np.random.default_rng(seed)
    obs = env.reset()
    rows = []
    observations = [obs] if collect_episode else None
    actions = []
    rewards = []
    terminated = []
    truncated = []
    metrics_seq = []

    for step in range(int(steps)):
        action = sample_random_action(env, rng)
        obs, reward, done, info = env.step(action)
        metrics = get_env_metrics(env)
        success = get_env_success(env)
        rows.append({"step": step, "reward": float(reward), "done": bool(done), "success": success, **metrics})

        if collect_episode:
            observations.append(obs)
            actions.append(np.asarray(action, dtype=float))
            rewards.append(float(reward))
            terminated.append(bool(success))
            truncated.append(bool(done and not success))
            metrics_seq.append(metrics)
        if done:
            break

    if not collect_episode:
        return rows, None
    episode = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "success": bool(rows[-1]["success"]) if rows else False,
        "metrics": metrics_seq,
    }
    return rows, episode
