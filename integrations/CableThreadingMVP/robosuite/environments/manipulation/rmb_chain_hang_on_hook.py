# rmb_chain_hang_on_hook.py
# RMB 链条挂钩任务：将柔性链条（当前用刚性链式线缆替代）挂到钩子上
# 该任务继承自 CableStraighten（拉直任务），在其基础上：
#   1. 在场景中构建一个钩子几何体（柱体 + 横杆 + 标记点）
#   2. 定义链条挂钩的专用奖励函数和评估指标
#   3. 通过 rmb_chain_hang_on_hook_metrics 工具函数判断是否成功挂上
#
# 当前版本使用 RMB 线缆作为链条的替代品（surrogate），
# 任务名称和指标接口已按真实链条语义设计，未来替换资产时无需修改上层代码。

import numpy as np

from robosuite.environments.manipulation.cable_straighten import CableStraighten
from robosuite.utils.dlo.rmb_chain_task import rmb_chain_hang_on_hook_metrics
from robosuite.utils.dlo.rmb_operation_presets import require_implemented_rmb_preset
from robosuite.utils.dlo.task_scene_utils import create_hook_body


class RMBChainHangOnHook(CableStraighten):
    """
    RMB IsaacUR5eChain task migrated as a robosuite task shell.

    The first runnable version uses the existing RMB rigid-chain cable asset as
    a chain surrogate. The task name and metrics follow the RMB Chain semantics
    so the real chain asset can replace the surrogate without changing L1 entry
    points.
    """

    def __init__(
        self,
        *args,
        rmb_robot_preset="ur5e",
        hook_pos=(0.18, 0.0, 0.92),            # 钩子在世界坐标系中的位置（默认在桌面右前方偏上）
        hook_radius=0.05,                        # 钩子的有效半径（用于判断链条末端是否靠近钩子）
        hook_height_tolerance=0.06,              # 链条末端相对于钩子的高度容差（米）
        min_vertical_drop=0.08,                  # 链条需要达到的最小垂直下垂量（表示成功挂在钩上）
        **kwargs,
    ):
        # 预设验证：确保指定的机器人预设已实现
        self.rmb_robot_preset = str(rmb_robot_preset).lower()
        self.rmb_preset = require_implemented_rmb_preset(self.rmb_robot_preset)

        # 保存钩子相关参数，后续在 _load_model 和 _compute_metrics 中使用
        self.hook_pos = np.asarray(hook_pos, dtype=float)
        self.hook_radius = float(hook_radius)
        self.hook_height_tolerance = float(hook_height_tolerance)
        self.min_vertical_drop = float(min_vertical_drop)

        # 线缆模型校验
        requested_cable_model = kwargs.pop("cable_model", None)
        if requested_cable_model is not None:
            normalized = str(requested_cable_model).lower()
            _VALID_CABLE_MODELS = {
                "rmb", "rmb_chain", "robomanip_baselines",
                "flex", "flex_cable", "flexcomp",
                "composite_cable", "composite", "mujoco_composite", "deformable_ravens_composite",
                "segmented", "capsule_chain",
                "mujoco_cable", "flex_reference_composite", "flex_reference_mujoco_cable",
            }
            if normalized not in _VALID_CABLE_MODELS:
                raise ValueError(f"RMBChainHangOnHook does not support cable_model='{requested_cable_model}'")

        # 调用父类构造函数，默认使用 "rmb" 线缆模型作为链条替代品
        super().__init__(*args, cable_model=requested_cable_model or "rmb", **kwargs)

    def _load_model(self):
        """构建钩子的 MuJoCo 模型，将其添加到场景中。"""
        super()._load_model()
        self.model.worldbody.append(
            create_hook_body(name="rmb_chain_hook", pos=self.hook_pos)
        )

    def reward(self, action=None):
        """
        奖励函数：综合考虑链条末端与钩子的水平距离、高度误差和垂直下垂量。

        奖励公式：
          reward = -xy_distance - 0.5 * height_error + 0.5 * min(vertical_drop, min_vertical_drop)
          如果成功挂上，额外 +1.0

        设计思路：
          - 负的 xy_distance 鼓励末端靠近钩子的正上方
          - 负的 height_error 鼓励末端在正确的高度（靠近钩子但不过高/过低）
          - 正的 vertical_drop 鼓励链条自然下垂（挂在钩上的标志）
          - 成功奖励给予稀疏的正反馈信号
        """
        metrics = self._compute_metrics()
        reward = -metrics["rmb_chain_end_xy_distance"] - 0.5 * metrics["rmb_chain_end_height_error"]
        reward += 0.5 * min(metrics["rmb_chain_vertical_drop"], self.min_vertical_drop)
        if metrics["success"]:
            reward += 1.0
        return self.reward_scale * reward

    def _compute_metrics(self):
        """
        计算任务指标：基础指标 + 链条挂钩专用指标。

        链条指标通过 rmb_chain_hang_on_hook_metrics 工具函数计算，
        包括：末端与钩子的水平距离、高度误差、垂直下垂量、稀疏成功信号等。
        """
        metrics = super()._compute_metrics()
        chain_metrics = rmb_chain_hang_on_hook_metrics(
            self._get_cable_points(),   # 链条/线缆的所有采样点坐标
            self._get_cable_end_pos(),  # 链条末端位置
            self.hook_pos,              # 钩子位置
            hook_radius=self.hook_radius,
            hook_height_tolerance=self.hook_height_tolerance,
            min_vertical_drop=self.min_vertical_drop,
        )
        metrics.update(chain_metrics)
        # 附加元信息，便于日志分析和结果追踪
        metrics["hook_pos"] = self.hook_pos.copy()
        metrics["rmb_source_task"] = "IsaacUR5eChain"          # 标记原始 RMB 任务来源
        metrics["rmb_robot_preset"] = self.rmb_robot_preset
        metrics["rmb_asset_status"] = "rmb_cable_surrogate"    # 标记当前使用线缆替代品
        metrics["success"] = chain_metrics["rmb_chain_sparse_success"]  # 稀疏成功信号
        return metrics

    def _check_success(self):
        """检查任务是否成功完成：直接使用 _compute_metrics 中计算的稀疏成功信号。"""
        return self._compute_metrics()["success"]
