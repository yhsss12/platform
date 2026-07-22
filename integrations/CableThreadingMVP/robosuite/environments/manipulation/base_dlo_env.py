"""BaseDLOEnv — 所有 DLO 操作任务的统一基类。

本模块是 DLO Benchmark 的核心基础设施入口。
实际实现在 cable_base.py 中（保持文件名稳定以减少 git 冲突）。

用法：
    from robosuite.environments.manipulation.base_dlo_env import BaseDLOEnv

向后兼容：
    from robosuite.environments.manipulation.cable_base import CableBaseEnv
    # CableBaseEnv == BaseDLOEnv
"""

from robosuite.environments.manipulation.cable_base import BaseDLOEnv

__all__ = ["BaseDLOEnv"]
