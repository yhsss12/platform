"""HasPegsMixin — 柱子/固定点能力混入类。

为 DLO 任务环境提供柱子（pegs/poles）场景元素：
  - 创建柱子几何体（两个圆柱体 + site）
  - 读取柱子位置
  - 计算柱间走廊边界
  - 检测线缆与柱子的碰撞

用法：
    class MyTask(BaseDLOEnv, HasPegsMixin):
        def _load_model(self):
            super()._load_model()
            self._create_pole_pair(self.arena)  # 向场景添加柱子

        def _setup_references(self):
            super()._setup_references()
            self._setup_pole_ids()  # 解析柱子 site ID
"""

import numpy as np
from robosuite.utils.dlo.task_scene_utils import create_pole_pair


class HasPegsMixin:
    """柱子/固定点能力混入类。

    提供柱子场景元素的创建、查询和碰撞检测。
    子类在 __init__ 中设置 pole_radius, pole_height, pole_offset, pole_spacing。
    """

    # ---- 默认参数（子类可在 __init__ 中覆盖） ----
    pole_radius: float = 0.01
    pole_height: float = 0.06
    pole_offset: np.ndarray = None  # (2,) 相对于桌面中心的偏移
    pole_spacing: float = 0.05
    gap_margin: float = 0.0

    # ---- 运行时 ID（_setup_pole_ids 设置） ----
    pole1_site_id: int = -1
    pole2_site_id: int = -1

    def _init_pole_defaults(self):
        """初始化柱子默认参数（在子类 __init__ 中调用）。"""
        if self.pole_offset is None:
            self.pole_offset = np.array([-0.025, 0.0], dtype=float)

    def _create_pole_pair(self, arena):
        """向 arena 添加柱子 body（在 _load_model 中调用）。"""
        arena.worldbody.append(create_pole_pair(
            name="threading_poles",
            pos=(self.pole_offset[0], self.pole_offset[1], self.table_offset[2] - 0.005),
            pole_radius=self.pole_radius,
            pole_height=self.pole_height,
            pole_spacing=self.pole_spacing,
        ))

    def _setup_pole_ids(self):
        """解析柱子 site 的 MuJoCo ID（在 _setup_references 中调用）。"""
        self.pole1_site_id = self.sim.model.site_name2id("pole1_site")
        self.pole2_site_id = self.sim.model.site_name2id("pole2_site")

    def _get_pole1_pos(self):
        """返回第一根杆柱的 site 位置。"""
        return self.sim.data.site_xpos[self.pole1_site_id].copy()

    def _get_pole2_pos(self):
        """返回第二根杆柱的 site 位置。"""
        return self.sim.data.site_xpos[self.pole2_site_id].copy()

    def _get_pole_positions(self):
        """返回两根杆柱的位置。"""
        return self._get_pole1_pos(), self._get_pole2_pos()

    def _gap_corridor_bounds(self):
        """返回柱间走廊的 x 范围和杆中心线 y。"""
        pole1_xy = self._get_pole1_pos()[:2]
        pole2_xy = self._get_pole2_pos()[:2]
        corridor_min = min(pole1_xy[0], pole2_xy[0]) + self.pole_radius - self.gap_margin
        corridor_max = max(pole1_xy[0], pole2_xy[0]) - self.pole_radius + self.gap_margin
        pole_y = float(pole1_xy[1])
        return float(corridor_min), float(corridor_max), pole_y

    def _post_collision_count(self):
        """计算线缆与杆柱的碰撞次数。"""
        from robosuite.utils.dlo.task_logic import threading_geometric_post_collision_count
        cable_pts = self._get_cable_points()
        pole1 = self._get_pole1_pos()
        pole2 = self._get_pole2_pos()
        return threading_geometric_post_collision_count(
            cable_pts[:, :2], pole1[:2], pole2[:2],
            pole_radius=self.pole_radius,
        )
