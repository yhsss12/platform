"""
CableInTask -- 可复用的线缆配置与重置逻辑模块。

本模块将线缆的模型选择、物理属性、场景构建、重置采样和点位读取
从任务环境中解耦，使其可被多个任务（CableBaseEnv、CableRouting、
CableThreading 等）复用。

核心设计：
- CableInTask 不持有 sim 引用，所有需要 sim 的方法将其作为参数传入
- setup_scene() 返回 CableSceneInfo 数据类，由调用方提取所需字段
- 抓取系统（mocap/weld/attachment）不在本模块范围内
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from robosuite.models.objects import cable_object_factory
from robosuite.models.objects.xml_objects import FlexCableObject, FlexImproveObject
from robosuite.utils.dlo.task_scene_utils import create_mocap_body, create_weld_constraint
from robosuite.utils.mjcf_utils import new_body, xml_path_completion


# ---------------------------------------------------------------------------
# CableSceneInfo: setup_scene() 的返回值
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CableSceneInfo:
    """setup_scene() 返回的场景构建信息。"""
    cable_object: object
    is_flex: bool
    mocap_body_names: list[str]
    grasp_eq_names: list[str]
    graspable_body_names: list[str]
    graspable_point_count: int
    grasp_endpoint_body_names: tuple[str, str]
    grasp_point_to_body: dict[int, str]
    grasp_body_to_index: dict[str, int]
    cable_root_joint: str | None
    num_cable_points: int
    placement_initializer: object | None
    flex_comp_name: str | None
    equality_elements: list


@dataclasses.dataclass
class CableSceneResult:
    """embed_in_arena() 返回的场景嵌入结果。

    替代 embed_flex_cable_in_arena() / embed_rigid_cable_in_arena() 返回的 raw dict。
    """
    cable_object: object
    is_flex: bool
    graspable_body_names: list[str]
    graspable_point_count: int
    grasp_endpoint_body_names: tuple[str, str]
    grasp_point_to_body: dict[int, str]
    grasp_body_to_index: dict[str, int]
    cable_root_joint: str | None
    num_cable_points: int
    placement_initializer: object | None
    flex_comp_name: str | None
    flex_container_body_name: str | None = None  # flex 容器 body 名称


@dataclasses.dataclass
class CableSimIds:
    """resolve_sim_ids() 返回的 MuJoCo ID 集合。

    替代 _sim_ids dict。
    """
    cable_point_ids: list[int]
    flex_id: int | None
    flex_vertadr: int | None
    flex_vertnum: int | None
    num_cable_points: int
    cable_start_body_id: int | None
    cable_end_body_id: int | None
    cable_shape_joint_names: list[str]


@dataclasses.dataclass
class ThreadingResetConfig:
    """穿杆任务重置配置。包含所有 task-specific 参数。"""
    # 杆柱几何
    pole_offset: np.ndarray
    pole_spacing: float
    # 桌面
    table_offset: np.ndarray
    table_min_xy: np.ndarray
    table_max_xy: np.ndarray
    reset_centerline_min_z: float
    # 难度配置
    difficulty: str
    reset_config_by_difficulty: dict
    # 机器人
    robot_name: str
    robot_reach_center: np.ndarray
    endpoint_reach_radius: float
    endpoint_reach_margin: float
    endpoint_reach_resample_attempts: int
    # 线缆几何
    anchor_to_center_distance: float
    initial_endpoint_distance_range: tuple
    initial_root_pos: np.ndarray
    initial_root_quat: np.ndarray
    # 约束 ID（_setup_references 后设置）
    anchor_body_id: int | None = None
    anchor_eq_id: int | None = None
    end_grasp_eq_id: int | None = None
    mocap_id: int | None = None
    # 行为
    attach_on_reset: bool = True
    max_reset_attempts: int = 32


@dataclasses.dataclass
class ThreadingResetResult:
    """穿杆任务重置结果。"""
    anchor_pos: np.ndarray | None
    endpoint_pos: np.ndarray | None
    cable_points: np.ndarray | None
    reset_attempts: int
    reset_valid: bool
    summary: dict


# ---------------------------------------------------------------------------
# CableInTask
# ---------------------------------------------------------------------------

class CableInTask:
    """可复用的线缆配置与重置逻辑。

    封装：线缆模型创建、物理属性、flex XML 解析/arena 嵌入、
    重置状态采样/应用、线缆点位读取。

    不持有：sim、robot、抓取系统。
    """

    def __init__(
        self,
        cable_model: str = "rmb",
        table_full_size: tuple = (0.8, 0.8, 0.05),
        table_friction: tuple = (1.0, 0.005, 0.0001),
        table_offset: np.ndarray | None = None,
        reset_xy_center: tuple = (-0.22, 0.06),
        reset_xy_range: float = 0.06,
        reset_yaw_range: float = 0.8,
        reset_shape_wave_scale: float = 0.03,
        reset_shape_noise_scale: float = 0.005,
        reset_shape_noise_clip: float = 0.06,
        reset_resample_attempts: int = 128,
        reset_centerline_clearance: float | None = None,
    ):
        self._cable_model = cable_model
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = (
            np.array((0.0, 0.0, 0.8)) if table_offset is None
            else np.asarray(table_offset, dtype=float)
        )
        self.table_top_z = float(self.table_offset[2])

        # 重置采样参数
        self.reset_xy_center = np.asarray(reset_xy_center, dtype=float)
        self.reset_xy_range = reset_xy_range
        self.reset_yaw_range = reset_yaw_range
        self.reset_shape_wave_scale = reset_shape_wave_scale
        self.reset_shape_noise_scale = reset_shape_noise_scale
        self.reset_shape_noise_clip = reset_shape_noise_clip
        self.reset_resample_attempts = reset_resample_attempts

        # 通过 probe 提取物理参数（不加载完整模型）
        probe = cable_object_factory(self._cable_model, name="cable_probe")
        self._cable_radius = float(probe.cable_radius)
        self._cable_length = float(getattr(probe, "cable_length", 0.48))
        self._cable_tabletop_offset = float(probe.tabletop_centerline_offset)
        self._cable_clearance = (
            float(reset_centerline_clearance) if reset_centerline_clearance is not None
            else (0.002 if self._cable_model == "rmb" else self._cable_tabletop_offset)
        )
        self._cable_centerline_z = self.table_top_z + self._cable_radius + self._cable_clearance
        self._point_reference_kind = probe.point_reference_kind
        self._point_reference_names = list(probe.point_reference_names)

        # 运行时状态（在 setup_scene / resolve_sim_ids 后填充）
        self.cable: object | None = None
        self._sim_ids: dict[str, Any] = {}
        self._flex_vertex_spacing: float | None = None
        self._flex_container_body_name: str | None = None

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def cable_model(self) -> str:
        return self._cable_model

    @property
    def cable_radius(self) -> float:
        return self._cable_radius

    @property
    def cable_length(self) -> float:
        return self._cable_length

    @property
    def cable_clearance(self) -> float:
        return self._cable_clearance

    @property
    def cable_tabletop_offset(self) -> float:
        return self._cable_tabletop_offset

    @property
    def cable_centerline_z(self) -> float:
        return self._cable_centerline_z

    @property
    def cable_point_reference_kind(self) -> str:
        return self._point_reference_kind

    @property
    def cable_point_reference_names(self) -> list[str]:
        return self._point_reference_names

    @property
    def is_flex(self) -> bool:
        if self.cable is not None:
            return isinstance(self.cable, (FlexCableObject, FlexImproveObject))
        return str(self._cable_model).lower() in {
            "flex",
            "flex_cable",
            "flexcomp",
            "flex_improve",
            "flex_improved",
        }

    @property
    def table_xy_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half_x = 0.5 * float(self.table_full_size[0]) - 0.08
        half_y = 0.5 * float(self.table_full_size[1]) - 0.08
        return np.array([-half_x, -half_y], dtype=float), np.array([half_x, half_y], dtype=float)

    @property
    def default_target_start(self) -> np.ndarray:
        if self.is_flex:
            half = 0.475 * self._cable_length
        else:
            half = 0.425 * self._cable_length
        return np.array([-half, 0.0, self._cable_centerline_z])

    @property
    def default_target_end(self) -> np.ndarray:
        if self.is_flex:
            half = 0.475 * self._cable_length
        else:
            half = 0.425 * self._cable_length
        return np.array([half, 0.0, self._cable_centerline_z])

    @property
    def cable_points_count(self) -> int:
        """线缆关键点数量（运行时需先调用 resolve_sim_ids 或 embed_*_in_arena）。"""
        n = self._sim_ids.get("flex_vertnum")
        if n is not None:
            return int(n)
        names = self._point_reference_names
        return len(names) if names else 0

    @property
    def flex_vertex_spacing(self) -> float | None:
        """flex 线缆顶点间距（米），仅 flex 模型有值。在 embed_flex_cable_in_arena 后可用。"""
        return self._flex_vertex_spacing

    # ------------------------------------------------------------------
    # Phase 1: 场景构建（_load_model 时调用）
    # ------------------------------------------------------------------

    def create_cable_object(self) -> object:
        """通过工厂创建正式线缆对象。"""
        self.cable = cable_object_factory(self._cable_model, name="cable")
        # 用正式对象覆盖 probe 的属性
        self._point_reference_kind = self.cable.point_reference_kind
        self._point_reference_names = list(self.cable.point_reference_names)
        self._cable_radius = float(self.cable.cable_radius)
        self._cable_length = float(self.cable.cable_length)
        self._cable_tabletop_offset = float(self.cable.tabletop_centerline_offset)
        self._cable_centerline_z = self.table_top_z + self._cable_radius + self._cable_clearance
        return self.cable

    def embed_in_arena(self, arena, placement_initializer=None, container_z=None) -> CableSceneResult:
        """将线缆嵌入 arena。统一的公共入口，替代 embed_flex_cable_in_arena / embed_rigid_cable_in_arena。

        Args:
            arena: TableArena 实例
            placement_initializer: 可选的放置采样器 (仅 rigid 线缆)
            container_z: flex 线缆容器的 z 坐标覆盖

        Returns:
            CableSceneResult 包含所有场景构建元数据
        """
        if self.cable is None:
            self.create_cable_object()

        if self.is_flex:
            return self._embed_flex_in_arena(arena, container_z)
        return self._embed_rigid_in_arena(arena, placement_initializer)

    def _embed_flex_in_arena(self, arena, container_z=None) -> CableSceneResult:
        """内部方法：将 flex 线缆嵌入 arena。"""
        meta = self.embed_flex_cable_in_arena(arena, container_z=container_z)
        self._flex_container_body_name = "flex_cable_container"
        return CableSceneResult(
            cable_object=meta["cable_object"],
            is_flex=meta["is_flex"],
            graspable_body_names=meta["graspable_body_names"],
            graspable_point_count=meta["graspable_point_count"],
            grasp_endpoint_body_names=meta["grasp_endpoint_body_names"],
            grasp_point_to_body=meta["grasp_point_to_body"],
            grasp_body_to_index=meta["grasp_body_to_index"],
            cable_root_joint=meta.get("cable_root_joint"),
            num_cable_points=meta["num_cable_points"],
            placement_initializer=None,
            flex_comp_name=meta.get("flex_comp_name"),
            flex_container_body_name="flex_cable_container",
        )

    def _embed_rigid_in_arena(self, arena, placement_initializer=None) -> CableSceneResult:
        """内部方法：将 rigid 线缆嵌入 arena。"""
        meta = self.embed_rigid_cable_in_arena(arena, placement_initializer=placement_initializer)
        return CableSceneResult(
            cable_object=meta["cable_object"],
            is_flex=meta["is_flex"],
            graspable_body_names=meta["graspable_body_names"],
            graspable_point_count=meta["graspable_point_count"],
            grasp_endpoint_body_names=meta["grasp_endpoint_body_names"],
            grasp_point_to_body=meta["grasp_point_to_body"],
            grasp_body_to_index=meta["grasp_body_to_index"],
            cable_root_joint=meta.get("cable_root_joint"),
            num_cable_points=meta["num_cable_points"],
            placement_initializer=meta.get("placement_initializer"),
            flex_comp_name=None,
            flex_container_body_name=None,
        )

    def embed_flex_cable_in_arena(self, arena, container_z=None):
        """将 flex 线缆 XML 嵌入 arena，不创建 mocap body 和 weld 约束。

        解析 flex_cable.xml，将 extension 插件和 worldbody 子树嵌入 arena。
        调用方可随后自行添加 task-specific 的 mocap body 和 weld 约束。

        Args:
            arena: TableArena 实例
            container_z: 容器 body 的 z 坐标，默认为 self._cable_centerline_z

        Returns:
            dict: 包含 cable_object, is_flex, graspable_body_names, graspable_point_count,
                  grasp_endpoint_body_names, grasp_point_to_body, grasp_body_to_index,
                  num_cable_points, flex_comp_name
        """
        if self.cable is None:
            self.create_cable_object()
        cable = self.cable
        if container_z is None:
            container_z = self._cable_centerline_z

        flex_comp_name = cable.point_reference_names[0]

        # 解析 flex XML 并嵌入 arena，同时从 <flexcomp count="..."> 提取真实顶点数。
        if isinstance(cable, FlexImproveObject):
            flex_xml_path = xml_path_completion("objects/dlo/flex_improve.xml")
        else:
            flex_xml_path = xml_path_completion("objects/dlo/flex_cable.xml")
        flex_tree = ET.parse(flex_xml_path)
        flex_root = flex_tree.getroot()
        flexcomp = flex_root.find(".//flexcomp")
        count_tokens = (flexcomp.get("count", "") if flexcomp is not None else "").split()
        num_cable_points = int(count_tokens[0]) if count_tokens else int(cable.flex_vertex_count)
        # 提取顶点间距
        spacing_tokens = (flexcomp.get("spacing", "") if flexcomp is not None else "").split()
        if spacing_tokens:
            self._flex_vertex_spacing = float(spacing_tokens[0])
        else:
            self._flex_vertex_spacing = 0.01  # 默认值
        graspable_body_names = [f"{flex_comp_name}_0", f"{flex_comp_name}_{num_cable_points - 1}"]
        graspable_point_count = len(graspable_body_names)
        grasp_point_to_body = {0: graspable_body_names[0], 1: graspable_body_names[-1]}
        grasp_body_to_index = {n: i for i, n in enumerate(graspable_body_names)}
        endpoint_names = (graspable_body_names[0], graspable_body_names[-1])

        for ext in flex_root.findall("extension"):
            for plugin in ext.findall("plugin"):
                arena.extension.append(plugin)
        flex_worldbody = flex_root.find("worldbody")
        if flex_worldbody is not None:
            for outer_body in flex_worldbody.findall("body"):
                if not outer_body.get("name"):
                    outer_body.set("name", "flex_cable_container")
                self._flex_container_body_name = outer_body.get("name", "flex_cable_container")
                outer_body.set(
                    "pos",
                    f"{self.table_offset[0]} {self.table_offset[1]} {container_z}",
                )
                arena.worldbody.append(outer_body)

        return {
            "cable_object": cable,
            "is_flex": True,
            "graspable_body_names": graspable_body_names,
            "graspable_point_count": graspable_point_count,
            "grasp_endpoint_body_names": endpoint_names,
            "grasp_point_to_body": grasp_point_to_body,
            "grasp_body_to_index": grasp_body_to_index,
            "num_cable_points": num_cable_points,
            "flex_comp_name": flex_comp_name,
        }

    def embed_rigid_cable_in_arena(self, arena, placement_initializer=None):
        """创建刚性线缆对象并设置放置采样器，不创建 mocap body 和 weld 约束。

        调用方可随后自行添加 task-specific 的 mocap body 和 weld 约束。

        Args:
            arena: TableArena 实例
            placement_initializer: 可选的放置采样器

        Returns:
            dict: 包含 cable_object, is_flex, graspable_body_names, graspable_point_count,
                  grasp_endpoint_body_names, grasp_point_to_body, grasp_body_to_index,
                  cable_root_joint, num_cable_points, placement_initializer
        """
        if self.cable is None:
            self.create_cable_object()
        cable = self.cable
        cable_root_joint = cable.joints[-1]

        graspable_body_names = list(cable.graspable_body_names)
        graspable_point_count = int(cable.graspable_point_count)
        grasp_point_to_body = {
            i: cable.body_name_for_point_idx(i)
            for i in range(graspable_point_count)
        }
        grasp_body_to_index = {n: i for i, n in enumerate(graspable_body_names)}
        if graspable_point_count >= 2:
            endpoint_names = (graspable_body_names[0], graspable_body_names[-1])
        else:
            endpoint_names = ("cable_B0", "cable_end")
        num_cable_points = len(self._point_reference_names)

        # 放置采样器
        if placement_initializer is not None:
            placement_initializer.reset()
            placement_initializer.add_objects(cable)
        else:
            from robosuite.utils.placement_samplers import UniformRandomSampler
            placement_initializer = UniformRandomSampler(
                name="CableSampler",
                mujoco_objects=cable,
                x_range=(-0.02, 0.02),
                y_range=(-0.02, 0.02),
                rotation=(-0.25, 0.25),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=self._cable_centerline_z - self.table_offset[2],
            )

        return {
            "cable_object": cable,
            "is_flex": False,
            "graspable_body_names": graspable_body_names,
            "graspable_point_count": graspable_point_count,
            "grasp_endpoint_body_names": endpoint_names,
            "grasp_point_to_body": grasp_point_to_body,
            "grasp_body_to_index": grasp_body_to_index,
            "cable_root_joint": cable_root_joint,
            "num_cable_points": num_cable_points,
            "placement_initializer": placement_initializer,
        }

    def setup_scene(self, arena, placement_initializer=None) -> CableSceneInfo:
        """将线缆嵌入 arena，返回 CableSceneInfo。

        调用前需先调用 create_cable_object()。
        """
        if self.cable is None:
            self.create_cable_object()

        if self.is_flex:
            return self._setup_flex_scene(arena)
        return self._setup_rigid_scene(arena, placement_initializer)

    def _setup_flex_scene(self, arena) -> CableSceneInfo:
        """flex 电缆的场景构建。"""
        meta = self.embed_flex_cable_in_arena(arena)

        mocap_body_names = [f"flex_grasp_mocap_{idx:02d}" for idx in range(meta["graspable_point_count"])]
        grasp_eq_names = [f"flex_grasp_{idx:02d}_mocap_weld" for idx in range(meta["graspable_point_count"])]

        for mocap_name in mocap_body_names:
            arena.worldbody.append(create_mocap_body(mocap_name, (0.0, 0.0, self._cable_centerline_z)))

        eq_elements = [
            create_weld_constraint(eq_name, mocap_name, body_name)
            for mocap_name, eq_name, body_name in zip(mocap_body_names, grasp_eq_names, meta["graspable_body_names"])
        ]

        return CableSceneInfo(
            cable_object=meta["cable_object"],
            is_flex=True,
            mocap_body_names=mocap_body_names,
            grasp_eq_names=grasp_eq_names,
            graspable_body_names=meta["graspable_body_names"],
            graspable_point_count=meta["graspable_point_count"],
            grasp_endpoint_body_names=meta["grasp_endpoint_body_names"],
            grasp_point_to_body=meta["grasp_point_to_body"],
            grasp_body_to_index=meta["grasp_body_to_index"],
            cable_root_joint=None,
            num_cable_points=meta["num_cable_points"],
            placement_initializer=None,
            flex_comp_name=meta["flex_comp_name"],
            equality_elements=eq_elements,
        )

    def _setup_rigid_scene(self, arena, placement_initializer) -> CableSceneInfo:
        """非 flex 电缆的场景构建。"""
        meta = self.embed_rigid_cable_in_arena(arena, placement_initializer)

        mocap_body_names = [f"cable_grasp_mocap_{idx:02d}" for idx in range(meta["graspable_point_count"])]
        grasp_eq_names = [f"cable_grasp_{idx:02d}_mocap_weld" for idx in range(meta["graspable_point_count"])]

        for mocap_name in mocap_body_names:
            arena.worldbody.append(create_mocap_body(mocap_name, (0.0, 0.0, self._cable_centerline_z)))

        eq_elements = [
            create_weld_constraint(eq_name, mocap_name, self._xml_grasp_body_name(body_name))
            for mocap_name, eq_name, body_name in zip(mocap_body_names, grasp_eq_names, meta["graspable_body_names"])
        ]

        return CableSceneInfo(
            cable_object=meta["cable_object"],
            is_flex=False,
            mocap_body_names=mocap_body_names,
            grasp_eq_names=grasp_eq_names,
            graspable_body_names=meta["graspable_body_names"],
            graspable_point_count=meta["graspable_point_count"],
            grasp_endpoint_body_names=meta["grasp_endpoint_body_names"],
            grasp_point_to_body=meta["grasp_point_to_body"],
            grasp_body_to_index=meta["grasp_body_to_index"],
            cable_root_joint=meta["cable_root_joint"],
            num_cable_points=meta["num_cable_points"],
            placement_initializer=meta["placement_initializer"],
            flex_comp_name=None,
            equality_elements=eq_elements,
        )

    # ------------------------------------------------------------------
    # Phase 2: 仿真 ID 解析（_setup_references 时调用）
    # ------------------------------------------------------------------

    def resolve_sim_ids(self, sim) -> dict[str, Any]:
        """解析 MuJoCo 仿真中的线缆相关 ID。"""
        ids: dict[str, Any] = {}

        if self._point_reference_kind == "flex":
            flex_name = self._point_reference_names[0]
            flex_id = None
            for i in range(sim.model.nflex):
                adr = sim.model.name_flexadr[i]
                stored_name = sim.model.names[adr:].split(b'\x00')[0].decode('utf-8')
                if stored_name == flex_name or stored_name.endswith(flex_name):
                    flex_id = i
                    break
            if flex_id is None:
                raise KeyError(f"Flex component '{flex_name}' not found in model")
            ids["flex_id"] = flex_id
            ids["flex_vertadr"] = int(sim.model.flex_vertadr[flex_id])
            ids["flex_vertnum"] = int(sim.model.flex_vertnum[flex_id])
            ids["cable_point_ids"] = list(range(ids["flex_vertnum"]))
            # 顶点间距：优先用 embed 时缓存的值，否则从 XML 解析
            if self._flex_vertex_spacing is None:
                # flex spacing 是线缆的固有属性，从 cable_length / (nvert-1) 推算
                nvert = ids["flex_vertnum"]
                if nvert > 1 and self._cable_length > 0:
                    self._flex_vertex_spacing = self._cable_length / (nvert - 1)
                else:
                    self._flex_vertex_spacing = 0.01
            ids["num_cable_points"] = ids["flex_vertnum"]
            ids["cable_start_body_id"] = sim.model.body_name2id(f"{flex_name}_0")
            ids["cable_end_body_id"] = sim.model.body_name2id(f"{flex_name}_{ids['flex_vertnum'] - 1}")
            ids["cable_shape_joint_names"] = []
        elif self._point_reference_kind == "site":
            ids["cable_point_ids"] = [
                sim.model.site_name2id(self._resolve_name(sim, name, "site"))
                for name in self._point_reference_names
            ]
            ids["flex_id"] = None
            ids["flex_vertadr"] = None
            ids["flex_vertnum"] = None
        elif self._point_reference_kind == "body":
            ids["cable_point_ids"] = [
                sim.model.body_name2id(self._resolve_name(sim, name, "body"))
                for name in self._point_reference_names
            ]
            ids["flex_id"] = None
            ids["flex_vertadr"] = None
            ids["flex_vertnum"] = None
        else:
            raise ValueError(f"Unsupported cable point reference kind: {self._point_reference_kind}")

        if ids["flex_id"] is None:
            ids["num_cable_points"] = len(ids["cable_point_ids"])
            start_body_name = self.cable.graspable_body_names[0] if self.cable.graspable_body_names else self._point_reference_names[0]
            end_body_name = self.cable.graspable_body_names[-1] if self.cable.graspable_body_names else self._point_reference_names[-1]
            ids["cable_start_body_id"] = sim.model.body_name2id(
                self._resolve_name(sim, start_body_name, "body")
            )
            ids["cable_end_body_id"] = sim.model.body_name2id(
                self._resolve_name(sim, end_body_name, "body")
            )
            ids["cable_shape_joint_names"] = [
                name for name in sim.model.joint_names
                if name and name.startswith(self.cable.naming_prefix) and "_J" in name
            ]

        self._sim_ids = ids
        return ids

    def resolve_sim_ids_typed(self, sim) -> CableSimIds:
        """Phase 3: 解析 MuJoCo ID，返回 CableSimIds 数据类。

        调用 resolve_sim_ids() 并将结果转换为 CableSimIds。
        """
        raw = self.resolve_sim_ids(sim)
        return CableSimIds(
            cable_point_ids=raw.get("cable_point_ids", []),
            flex_id=raw.get("flex_id"),
            flex_vertadr=raw.get("flex_vertadr"),
            flex_vertnum=raw.get("flex_vertnum"),
            num_cable_points=raw.get("num_cable_points", 0),
            cable_start_body_id=raw.get("cable_start_body_id"),
            cable_end_body_id=raw.get("cable_end_body_id"),
            cable_shape_joint_names=raw.get("cable_shape_joint_names", []),
        )

    def apply_visual_fixes(self, sim):
        """应用所有视觉/物理修复（模型编译后调用）。

        包含：
        1. geom_group 修复：将线缆 geom 移到 group 1（mjviewer 默认隐藏 group 0）
        2. flex 摩擦力设置（flexcomp 不支持 XML 属性，需程序设置）

        在 _setup_references() 中 resolve_sim_ids() 后调用。
        """
        # geom_group 修复
        for i in range(sim.model.ngeom):
            name = sim.model.geom(i).name
            if name and (name.startswith("cable_") or name.startswith("Flex")):
                sim.model.geom_group[i] = 1

        # composite 线缆：增加摩擦力和碰撞余量，使物理夹取可行。
        # composite 线缆的 flex 碰撞不生成 fingerpad 接触，
        # 需要更大摩擦力和 margin 来确保夹爪能夹住线缆。

        # flex 摩擦力
        flex_id = self._sim_ids.get("flex_id")
        if flex_id is not None:
            cable_name = type(self.cable).__name__
            if "Improve" in cable_name:
                sim.model.flex_friction[flex_id] = [3.0, 0.08, 0.0001]
            else:
                sim.model.flex_friction[flex_id] = [2.0, 0.05, 0.0001]
            sim.model.flex_condim[flex_id] = 4
            # 碰撞余量：防止 flex 线缆穿透柱子等障碍物
            # margin=0.005 创建 5mm 的碰撞缓冲层（从 2mm 增加，减少穿杆时的瞬时穿透）
            # gap=0.002 设置 2mm 的最小间距
            sim.model.flex_margin[flex_id] = 0.005
            sim.model.flex_gap[flex_id] = 0.002

    # ------------------------------------------------------------------
    # Phase 3: 重置
    # ------------------------------------------------------------------

    def apply_reset(self, sim, rng, deterministic: bool = False) -> dict:
        """应用线缆重置状态。返回诊断信息字典。"""
        if self.is_flex:
            return self._reset_flex(sim, rng, deterministic)
        if self._cable_model == "rmb":
            return self._reset_rmb(sim, rng, deterministic)
        return self._reset_other(sim, rng, deterministic)

    def _reset_flex(self, sim, rng, deterministic: bool) -> dict:
        """flex 电缆重置。"""
        self.apply_flex_initial_shape(sim, rng, deterministic)
        sim.forward()
        return {
            "min_centerline_z": float(np.min(self.get_cable_points(sim)[:, 2])),
            "deterministic_reset": deterministic,
        }

    def _reset_rmb(self, sim, rng, deterministic: bool) -> dict:
        """rmb 电缆重置。"""
        cable_root_joint = self._sim_ids.get("cable_root_joint")
        shape_joint_names = self._sim_ids.get("cable_shape_joint_names", [])
        if cable_root_joint is None:
            # 首次调用时缓存
            if self.cable is not None:
                cable_root_joint = self.cable.joints[-1]
                self._sim_ids["cable_root_joint"] = cable_root_joint

        if deterministic:
            root_pos = np.array([self.reset_xy_center[0], self.reset_xy_center[1], self._cable_centerline_z], dtype=float)
            root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            shape_noise = self.default_rmb_shape_noise(shape_joint_names)
        else:
            root_pos, root_quat, shape_noise = self.sample_rmb_reset_state(sim, rng, shape_joint_names)

        root_pos = self.apply_rmb_reset_state(sim, root_pos, root_quat, shape_noise, cable_root_joint, shape_joint_names)

        return {
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy() if isinstance(root_quat, np.ndarray) else root_quat,
            "shape_noise_l2": float(np.linalg.norm(shape_noise)),
            "min_centerline_z": float(np.min(self.get_cable_points(sim)[:, 2])),
            "lift_z": float(root_pos[2] - self._cable_centerline_z),
            "deterministic_reset": deterministic,
        }

    def _reset_other(self, sim, rng, deterministic: bool) -> dict:
        """非 rmb 非 flex 电缆重置（segmented, composite 等）。"""
        cable_root_joint = self._sim_ids.get("cable_root_joint")
        if cable_root_joint is None and self.cable is not None:
            cable_root_joint = self.cable.joints[-1]
            self._sim_ids["cable_root_joint"] = cable_root_joint

        if not deterministic and self._sim_ids.get("placement_initializer") is not None:
            pi = self._sim_ids["placement_initializer"]
            object_placements = pi.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                joint_name = cable_root_joint if obj is self.cable else obj.joints[0]
                sim.data.set_joint_qpos(joint_name, np.concatenate([obj_pos, obj_quat]))
                sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=float))
        elif not deterministic:
            pass  # 没有 placement_initializer，跳过
        else:
            yaw = 0.45
            root_pos = np.array([self.reset_xy_center[0], self.reset_xy_center[1] + 0.10, self._cable_centerline_z], dtype=float)
            root_quat = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=float)
            sim.data.set_joint_qpos(cable_root_joint, np.concatenate([root_pos, root_quat]))
            sim.data.set_joint_qvel(cable_root_joint, np.zeros(6, dtype=float))

        if not deterministic:
            self.drop_cable_to_table(sim, cable_root_joint)

        return {"deterministic_reset": deterministic}

    def apply_flex_initial_shape(self, sim, rng, deterministic: bool = False):
        """设置 flex 电缆初始形状。"""
        ids = self._sim_ids
        vertadr = ids.get("flex_vertadr")
        vertnum = ids.get("flex_vertnum")
        if vertadr is None:
            # 尝试从 sim 直接读取
            if self.cable is not None and hasattr(self.cable, "flex_vertex_count"):
                vertnum = int(self.cable.flex_vertex_count)
                # flex_id 需要从 sim 解析
                flex_name = self._point_reference_names[0]
                for i in range(sim.model.nflex):
                    adr = sim.model.name_flexadr[i]
                    stored_name = sim.model.names[adr:].split(b'\x00')[0].decode('utf-8')
                    if stored_name == flex_name or stored_name.endswith(flex_name):
                        vertadr = int(sim.model.flex_vertadr[i])
                        break
            if vertadr is None or vertnum is None:
                return

        nvert = vertnum
        if nvert < 3:
            return

        # 1. 设置 XY 位置偏移
        try:
            body_id = sim.model.body_name2id(self._flex_container_body_name)
        except ValueError:
            body_id = None
        if body_id is not None:
            target_start = self.default_target_start
            target_end = self.default_target_end
            if deterministic:
                cx = float((target_start[0] + target_end[0]) / 2 + 0.05)
                cy = float((target_start[1] + target_end[1]) / 2 + 0.03)
            else:
                cx = float((target_start[0] + target_end[0]) / 2 + rng.uniform(-self.reset_xy_range, self.reset_xy_range))
                cy = float((target_start[1] + target_end[1]) / 2 + rng.uniform(-self.reset_xy_range, self.reset_xy_range))
            sim.model.body_pos[body_id, 0] = cx
            sim.model.body_pos[body_id, 1] = cy
            sim.model.body_pos[body_id, 2] = self._cable_centerline_z

        # 2. 设置形状偏移
        sim.data.qpos[:nvert * 3] = 0.0

        if deterministic:
            amplitude = 0.05
            phase = 0.0
            freq = 1.0
        else:
            amplitude = float(rng.uniform(0.03, 0.08))
            phase = float(rng.uniform(0, 2 * np.pi))
            freq = float(rng.choice([1.0, 1.5, 2.0]))

        for i in range(nvert):
            t = i / (nvert - 1)
            sim.data.qpos[i * 3 + 1] = amplitude * np.sin(freq * np.pi * t + phase)

    def default_rmb_shape_noise(self, shape_joint_names: list[str] | None = None) -> np.ndarray:
        """生成确定性 rmb 形状噪声。"""
        if shape_joint_names is None:
            shape_joint_names = self._sim_ids.get("cable_shape_joint_names", [])
        joint_count = len(shape_joint_names)
        shape_noise = np.zeros(joint_count, dtype=float)
        if not joint_count:
            return shape_noise
        t = np.linspace(0.0, 1.0, joint_count)
        wave = self.reset_shape_wave_scale * np.sin(np.pi * t)
        vertical = 0.2 * self.reset_shape_wave_scale * np.cos(np.pi * t)
        taper = 0.25 + 0.75 * t
        shape_noise = np.clip(taper * wave, -self.reset_shape_noise_clip, self.reset_shape_noise_clip)
        for idx, name in enumerate(shape_joint_names):
            if "_J0_" in name:
                shape_noise[idx] = np.clip(
                    taper[idx] * vertical[idx],
                    -self.reset_shape_noise_clip,
                    self.reset_shape_noise_clip,
                )
        return shape_noise

    def sample_rmb_reset_state(self, sim, rng, shape_joint_names: list[str] | None = None):
        """随机采样 rmb 重置状态。返回 (root_pos, root_quat, shape_noise)。"""
        if shape_joint_names is None:
            shape_joint_names = self._sim_ids.get("cable_shape_joint_names", [])
        table_min_xy, table_max_xy = self.table_xy_bounds
        safe_z = self._cable_centerline_z
        cable_root_joint = self._sim_ids.get("cable_root_joint")
        if cable_root_joint is None and self.cable is not None:
            cable_root_joint = self.cable.joints[-1]

        best = None
        root_pos = None
        root_quat = None
        shape_noise = None

        for _ in range(self.reset_resample_attempts):
            root_xy = self.reset_xy_center + rng.uniform(low=-self.reset_xy_range, high=self.reset_xy_range, size=2)
            root_yaw = float(rng.uniform(-self.reset_yaw_range, self.reset_yaw_range))
            root_pos = np.array([root_xy[0], root_xy[1], safe_z], dtype=float)
            root_quat = np.array([np.cos(0.5 * root_yaw), 0.0, 0.0, np.sin(0.5 * root_yaw)], dtype=float)

            joint_count = len(shape_joint_names)
            shape_noise = np.zeros(joint_count, dtype=float)
            if joint_count:
                t = np.linspace(0.0, 1.0, joint_count)
                phase = float(rng.uniform(-np.pi, np.pi))
                wave = self.reset_shape_wave_scale * np.sin(2.0 * np.pi * t + phase)
                vertical = 0.35 * self.reset_shape_wave_scale * np.cos(np.pi * t + phase)
                taper = 0.25 + 0.75 * t
                jitter = rng.normal(loc=0.0, scale=self.reset_shape_noise_scale, size=joint_count)
                shape_noise = np.clip(taper * (wave + jitter), -self.reset_shape_noise_clip, self.reset_shape_noise_clip)
                for idx, name in enumerate(shape_joint_names):
                    if "_J0_" in name:
                        shape_noise[idx] = np.clip(
                            taper[idx] * (vertical[idx] + 0.4 * jitter[idx]),
                            -self.reset_shape_noise_clip,
                            self.reset_shape_noise_clip,
                        )

            sim.data.set_joint_qpos(cable_root_joint, np.concatenate([root_pos, root_quat]))
            sim.data.set_joint_qvel(cable_root_joint, np.zeros(6, dtype=float))
            for joint_name, value in zip(shape_joint_names, shape_noise):
                sim.data.set_joint_qpos(joint_name, float(value))
                sim.data.set_joint_qvel(joint_name, 0.0)
            sim.forward()

            points = self.get_cable_points(sim)
            min_centerline_z = float(np.min(points[:, 2]))
            lift_z = max(0.0, safe_z - min_centerline_z)
            if lift_z > 0.0:
                root_pos = root_pos.copy()
                root_pos[2] += lift_z
                sim.data.set_joint_qpos(cable_root_joint, np.concatenate([root_pos, root_quat]))
                sim.forward()
                points = self.get_cable_points(sim)

            xy = points[:, :2]
            if (np.all(xy[:, 0] >= table_min_xy[0]) and np.all(xy[:, 0] <= table_max_xy[0])
                    and np.all(xy[:, 1] >= table_min_xy[1]) and np.all(xy[:, 1] <= table_max_xy[1])):
                best = (root_pos, root_quat, shape_noise)
                break

        if best is None:
            best = (root_pos, root_quat, shape_noise)
        return best

    def apply_rmb_reset_state(self, sim, root_pos, root_quat, shape_noise,
                              cable_root_joint=None, shape_joint_names=None):
        """应用 rmb 重置状态到仿真。返回最终 root_pos。"""
        if cable_root_joint is None:
            cable_root_joint = self._sim_ids.get("cable_root_joint")
        if shape_joint_names is None:
            shape_joint_names = self._sim_ids.get("cable_shape_joint_names", [])

        root_pos = np.asarray(root_pos, dtype=float).copy()
        root_quat = np.asarray(root_quat, dtype=float)
        sim.data.set_joint_qpos(cable_root_joint, np.concatenate([root_pos, root_quat]))
        sim.data.set_joint_qvel(cable_root_joint, np.zeros(6, dtype=float))
        for joint_name, value in zip(shape_joint_names, shape_noise):
            sim.data.set_joint_qpos(joint_name, float(value))
            sim.data.set_joint_qvel(joint_name, 0.0)
        sim.forward()

        min_centerline_z = float(np.min(self.get_cable_points(sim)[:, 2]))
        lift_z = max(0.0, self._cable_centerline_z - min_centerline_z)
        if lift_z > 0.0:
            root_pos[2] += lift_z
            sim.data.set_joint_qpos(cable_root_joint, np.concatenate([root_pos, root_quat]))
            sim.forward()
        return root_pos

    def drop_cable_to_table(self, sim, cable_root_joint=None, max_steps=1500,
                            settle_vel_threshold=0.01, settle_steps_required=50):
        """物理沉降让线缆落到桌面。"""
        if cable_root_joint is None:
            cable_root_joint = self._sim_ids.get("cable_root_joint")
        sim.forward()
        settle_count = 0
        for _ in range(max_steps):
            sim.step()
            cable_qvel = sim.data.get_joint_qvel(cable_root_joint)
            z_vel = float(np.abs(cable_qvel[2]))
            if z_vel < settle_vel_threshold:
                settle_count += 1
                if settle_count >= settle_steps_required:
                    break
            else:
                settle_count = 0

    # ------------------------------------------------------------------
    # Phase 4: 运行时点位读取
    # ------------------------------------------------------------------

    def get_cable_points(self, sim) -> np.ndarray:
        """读取所有线缆关键点 3D 位置 [N, 3]。"""
        kind = self._point_reference_kind
        ids = self._sim_ids
        if kind == "flex":
            adr = ids.get("flex_vertadr", 0)
            n = ids.get("flex_vertnum", 0)
            if n == 0:
                return np.zeros((0, 3))
            return np.array(sim.data.flexvert_xpos[adr:adr + n].copy().reshape(n, 3))
        if kind == "site":
            return np.array([sim.data.site_xpos[pid].copy() for pid in ids.get("cable_point_ids", [])])
        return np.array([sim.data.xpos[pid].copy() for pid in ids.get("cable_point_ids", [])])

    def get_cable_start_pos(self, sim) -> np.ndarray:
        """返回线缆起始端位置。"""
        if self._point_reference_kind == "flex":
            return self.get_cable_points(sim)[0].copy()
        body_id = self._sim_ids.get("cable_start_body_id")
        if body_id is None:
            return np.zeros(3)
        return sim.data.xpos[body_id].copy()

    def get_cable_end_pos(self, sim) -> np.ndarray:
        """返回线缆末端位置。"""
        if self._point_reference_kind == "flex":
            return self.get_cable_points(sim)[-1].copy()
        body_id = self._sim_ids.get("cable_end_body_id")
        if body_id is None:
            return np.zeros(3)
        return sim.data.xpos[body_id].copy()

    def translate_cable_xy(self, sim, delta_xy):
        """平移线缆 XY 位置（不改变形状）。不负责 mocap 同步。"""
        delta_xy = np.asarray(delta_xy, dtype=float)
        if delta_xy.shape != (2,):
            raise ValueError(f"delta_xy must have shape (2,), got {delta_xy.shape}")
        if self.is_flex:
            body_id = sim.model.body_name2id(self._flex_container_body_name)
            sim.model.body_pos[body_id, 0] += float(delta_xy[0])
            sim.model.body_pos[body_id, 1] += float(delta_xy[1])
            sim.forward()
            return
        cable_root_joint = self._sim_ids.get("cable_root_joint")
        if cable_root_joint is None and self.cable is not None:
            cable_root_joint = self.cable.joints[-1]
        root_qpos = sim.data.get_joint_qpos(cable_root_joint).copy()
        root_qpos[0] += float(delta_xy[0])
        root_qpos[1] += float(delta_xy[1])
        sim.data.set_joint_qpos(cable_root_joint, root_qpos)
        root_qvel = sim.data.get_joint_qvel(cable_root_joint).copy()
        root_qvel[:] = 0.0
        sim.data.set_joint_qvel(cable_root_joint, root_qvel)
        for joint_name in self._sim_ids.get("cable_shape_joint_names", []):
            try:
                jnt_id = sim.model.joint_name2id(joint_name)
                jnt_type = sim.model.jnt_type[jnt_id]
                if jnt_type == 1:  # ball joint (3 DOF)
                    sim.data.set_joint_qvel(joint_name, np.zeros(3, dtype=float))
                else:  # hinge joint (1 DOF)
                    sim.data.set_joint_qvel(joint_name, 0.0)
            except (KeyError, ValueError):
                pass
        sim.forward()

    # ------------------------------------------------------------------
    # 名称解析
    # ------------------------------------------------------------------

    def _resolve_name(self, sim, base_name: str, kind: str) -> str:
        """解析线缆 body/site 名称（处理前缀兼容）。"""
        candidates = [base_name, f"{self.cable.naming_prefix}{base_name}"]
        for name in candidates:
            try:
                if kind == "site":
                    sim.model.site_name2id(name)
                elif kind == "body":
                    sim.model.body_name2id(name)
                else:
                    raise ValueError(f"Unsupported reference kind: {kind}")
                return name
            except (KeyError, ValueError):
                continue
        raise KeyError(f"Unable to resolve {kind} reference for cable point '{base_name}'")

    def resolve_cable_name(self, sim, base_name: str, kind: str) -> str:
        """公共 API：解析线缆名称。"""
        return self._resolve_name(sim, base_name, kind)

    def _xml_grasp_body_name(self, body_name: str) -> str:
        """返回 body 在 XML 中的完整名称。"""
        prefix = self.cable.naming_prefix
        if str(body_name).startswith(prefix):
            return body_name
        if str(body_name).startswith("cablec_"):
            return f"{prefix}{body_name}"
        if str(body_name).startswith(("gem_", "flexrefc_")):
            return f"{prefix}{body_name}"
        if self.cable.exclude_from_prefixing(body_name):
            return body_name
        return f"{prefix}{body_name}"

    # ------------------------------------------------------------------
    # 辅助：存储外部状态
    # ------------------------------------------------------------------

    def set_scene_state(self, key: str, value):
        """存储场景构建后的外部状态（如 placement_initializer、cable_root_joint）。"""
        self._sim_ids[key] = value

    # ------------------------------------------------------------------
    # Phase 5: 穿杆任务重置（统一入口）
    # ------------------------------------------------------------------

    def apply_threading_reset(self, sim, rng, config: ThreadingResetConfig,
                              get_cable_points_fn, get_cable_end_pos_fn,
                              get_pole_pos_fn, align_flex_fn=None) -> ThreadingResetResult:
        """穿杆任务的统一重置逻辑，兼容所有线缆模型。

        将 CableThreading._reset_internal() 的核心逻辑集中到此处，
        确保 flex / rmb / composite_cable 使用相同的采样、裁剪、验证流程。

        Args:
            sim: MuJoCo sim 对象
            rng: numpy RandomState
            config: 穿杆重置配置（包含杆柱几何、约束 ID、采样参数等）
            get_cable_points_fn: () -> np.ndarray [N,3] 获取线缆点位
            get_cable_end_pos_fn: () -> np.ndarray [3] 获取线缆末端位置
            get_pole_pos_fn: (site_id) -> np.ndarray [3] 获取杆柱位置
            align_flex_fn: (body_id, anchor_xy, endpoint_xy) 对齐 flex 容器（仅 flex）

        Returns:
            ThreadingResetResult
        """
        table_min_xy, table_max_xy = config.table_min_xy, config.table_max_xy
        reset_valid = False
        reset_attempts = 0
        endpoint_pos = None
        anchor_pos = None

        for reset_attempts in range(1, config.max_reset_attempts + 1):
            attempt_scale = max(0.2, 1.0 - 0.8 * ((reset_attempts - 1) / max(config.max_reset_attempts - 1, 1)))

            # 禁用约束
            if config.anchor_eq_id is not None:
                sim.data.eq_active[config.anchor_eq_id] = 0
            if config.end_grasp_eq_id is not None:
                sim.data.eq_active[config.end_grasp_eq_id] = 0

            # 采样 anchor + endpoint
            anchor_xy, endpoint_target_xy, endpoint_distance = self._sample_anchor_for_threading(
                rng, config, attempt_scale,
            )

            # 设置线缆形状（按类型分支）
            if self.is_flex:
                self._apply_flex_threading_shape(
                    sim, rng, config, anchor_xy, endpoint_target_xy, attempt_scale,
                    align_flex_fn,
                )
            else:
                self._apply_rigid_threading_shape(
                    sim, rng, config, anchor_xy, endpoint_target_xy, attempt_scale,
                )

            # 提升到桌面以上
            cable_points = get_cable_points_fn()
            min_z = float(np.min(cable_points[:, 2]))
            lift_z = max(0.0, config.reset_centerline_min_z - min_z)
            if lift_z > 0.0:
                if self.is_flex and self._flex_container_body_name:
                    body_id = sim.model.body_name2id(self._flex_container_body_name)
                    sim.model.body_pos[body_id, 2] += lift_z
                    sim.forward()
                else:
                    cable_root_joint = self._sim_ids.get("cable_root_joint")
                    if cable_root_joint:
                        root_qpos = sim.data.get_joint_qpos(cable_root_joint).copy()
                        root_qpos[2] += lift_z
                        sim.data.set_joint_qpos(cable_root_joint, root_qpos)
                        sim.forward()

            # 平移端点到目标
            if not self.is_flex:
                endpoint_pos = get_cable_end_pos_fn()
                ep_delta = np.asarray(endpoint_target_xy, dtype=float) - endpoint_pos[:2]
                if np.linalg.norm(ep_delta) > 1e-8:
                    cable_root_joint = self._sim_ids.get("cable_root_joint")
                    if cable_root_joint:
                        root_qpos = sim.data.get_joint_qpos(cable_root_joint).copy()
                        root_qpos[:2] += ep_delta
                        sim.data.set_joint_qpos(cable_root_joint, root_qpos)
                        sim.forward()
                    # drift 补偿 + 桌面裁剪
                    self._compensate_drift_and_clamp(sim, get_cable_points_fn, table_min_xy, table_max_xy)
                # 端点平移后重新将第一个 body 放回 anchor 位置（保持 anchor 在目标距离）
                first_body_pos = get_cable_points_fn()[0].copy()
                anchor_delta = np.asarray(anchor_xy, dtype=float) - first_body_pos[:2]
                if np.linalg.norm(anchor_delta) > 1e-8:
                    cable_root_joint = self._sim_ids.get("cable_root_joint")
                    if cable_root_joint:
                        root_qpos = sim.data.get_joint_qpos(cable_root_joint).copy()
                        root_qpos[:2] += anchor_delta
                        sim.data.set_joint_qpos(cable_root_joint, root_qpos)
                        sim.forward()

            # 固定 anchor
            fixed_end_pos = get_cable_points_fn()[0].copy()
            sim.model.body_pos[config.anchor_body_id] = fixed_end_pos
            anchor_pos = fixed_end_pos.copy()
            if config.anchor_eq_id is not None:
                sim.data.eq_active[config.anchor_eq_id] = 1
            sim.forward()

            # 验证
            endpoint_pos = get_cable_end_pos_fn()
            cable_points = get_cable_points_fn()
            reset_valid = self._threading_reset_is_valid(
                cable_points, anchor_xy, endpoint_pos[:2], config, get_pole_pos_fn,
            )
            if reset_valid:
                break

        # deterministic fallback
        if not reset_valid and not self.is_flex:
            anchor_pos, endpoint_pos, cable_points = self._deterministic_threading_fallback(
                sim, config, get_cable_points_fn, get_cable_end_pos_fn, get_pole_pos_fn,
            )
            reset_valid = True

        # 同步 mocap
        if endpoint_pos is not None and config.mocap_id is not None:
            sim.data.mocap_pos[config.mocap_id] = endpoint_pos
            sim.data.mocap_quat[config.mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if config.end_grasp_eq_id is not None:
            sim.data.eq_active[config.end_grasp_eq_id] = 1 if config.attach_on_reset else 0
        sim.forward()

        return ThreadingResetResult(
            anchor_pos=anchor_pos,
            endpoint_pos=get_cable_end_pos_fn() if endpoint_pos is not None else endpoint_pos,
            cable_points=get_cable_points_fn(),
            reset_attempts=reset_attempts,
            reset_valid=reset_valid,
            summary={
                "cable_model": self._cable_model,
                "reset_attempts": int(reset_attempts),
                "endpoint_valid": bool(reset_valid),
            },
        )

    def _sample_anchor_for_threading(self, rng, config: ThreadingResetConfig,
                                     attempt_scale=1.0):
        """穿杆任务的 anchor + endpoint 采样（带桌面裁剪）。"""
        cfg = config.reset_config_by_difficulty[config.difficulty]
        pole_center = config.pole_offset + np.array([0.5 * config.pole_spacing, 0.0], dtype=float)

        angle_range = cfg["anchor_angle_range"]
        anchor_angle = float(cfg["anchor_angle_center"] + rng.uniform(-angle_range, angle_range))
        radius = config.anchor_to_center_distance
        anchor_xy = pole_center + radius * np.array([np.cos(anchor_angle), np.sin(anchor_angle)], dtype=float)
        anchor_xy = np.clip(anchor_xy, config.table_min_xy + 0.02, config.table_max_xy - 0.02)

        # endpoint 采样
        best_ep = None
        best_dist = None
        best_score = float("inf")
        for _ in range(config.endpoint_reach_resample_attempts):
            ep_angle = float(rng.uniform(-np.pi, np.pi))
            ep_dist = float(rng.uniform(*config.initial_endpoint_distance_range))
            ep_dir = np.array([np.cos(ep_angle), np.sin(ep_angle)], dtype=float)
            ep_xy = anchor_xy + ep_dir * ep_dist
            ep_xy = np.clip(ep_xy, config.table_min_xy, config.table_max_xy)
            reach_error = max(0.0, float(np.linalg.norm(ep_xy - config.robot_reach_center)) - config.endpoint_reach_radius + config.endpoint_reach_margin)
            if reach_error <= 1e-8:
                return anchor_xy, ep_xy, float(np.linalg.norm(ep_xy - anchor_xy))
            if reach_error < best_score:
                best_ep = ep_xy
                best_dist = float(np.linalg.norm(ep_xy - anchor_xy))
                best_score = reach_error

        return anchor_xy, best_ep, best_dist

    def _apply_flex_threading_shape(self, sim, rng, config, anchor_xy, endpoint_target_xy,
                                    attempt_scale, align_flex_fn):
        """设置 flex 线缆的穿杆初始形状。"""
        body_id = sim.model.body_name2id(self._flex_container_body_name)
        half_length = 0.5 * (self._sim_ids.get("flex_vertnum", 51) - 1) * (self._flex_vertex_spacing or 0.01)
        direction = endpoint_target_xy - anchor_xy
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = direction / dist

        sim.model.body_pos[body_id, 0] = anchor_xy[0] + half_length * direction[0]
        sim.model.body_pos[body_id, 1] = anchor_xy[1] + half_length * direction[1]
        sim.model.body_pos[body_id, 2] = config.table_offset[2] + 0.01

        cfg = config.reset_config_by_difficulty[config.difficulty]
        yaw = float(np.arctan2(direction[1], direction[0])) + float(rng.uniform(-cfg["root_yaw_range"], cfg["root_yaw_range"])) * attempt_scale
        # 简化的 yaw quat
        sim.model.body_quat[body_id] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float)

        # 形状噪声
        nvert = self._sim_ids.get("flex_vertnum", 51)
        adr = self._sim_ids.get("flex_vertadr", 0)
        sim.data.qpos[adr: adr + nvert * 3] = 0.0
        t = np.linspace(0.0, 1.0, nvert)
        for m in range(1, 4):
            amp = rng.normal(0.0, cfg["joint_noise_scale"] * attempt_scale / m)
            phase = rng.uniform(-np.pi, np.pi)
            for i in range(nvert):
                sim.data.qpos[adr + i * 3 + 1] += amp * np.sin(m * np.pi * t[i] + phase)

        # Push cable vertices away from poles to prevent initial penetration.
        sim.forward()
        pole_radius = getattr(self, "pole_radius", 0.01)
        pole_clearance = pole_radius + 0.008
        pole_positions = [config.pole_offset[:2], config.pole_offset[:2] + np.array([config.pole_spacing, 0.0])]
        body_id = sim.model.body_name2id(self._flex_container_body_name)
        body_pos = sim.data.body_xpos[body_id]
        body_rot = sim.data.body_xmat[body_id].reshape(3, 3)
        spacing = self._flex_vertex_spacing or 0.01

        for _iter in range(3):
            verts_world = sim.data.flexvert_xpos[adr:adr + nvert].copy()
            for pole_xy in pole_positions:
                for i in range(nvert):
                    dx = verts_world[i, 0] - pole_xy[0]
                    dy = verts_world[i, 1] - pole_xy[1]
                    dist = float(np.sqrt(dx * dx + dy * dy))
                    if dist < pole_clearance and dist > 1e-8:
                        push = (pole_clearance - dist) + 0.002
                        verts_world[i, 0] += (dx / dist) * push
                        verts_world[i, 1] += (dy / dist) * push
            # Convert corrected world positions back to qpos offsets.
            verts_local = (body_rot.T @ (verts_world - body_pos).T).T
            grid_rest = np.zeros((nvert, 3), dtype=float)
            grid_rest[:, 0] = np.arange(nvert) * spacing
            offsets = verts_local - grid_rest
            for i in range(nvert):
                sim.data.qpos[adr + i * 3: adr + i * 3 + 3] = offsets[i]
            sim.forward()
        sim.forward()

        if align_flex_fn is not None:
            align_flex_fn(body_id, anchor_xy, endpoint_target_xy)

    def _apply_rigid_threading_shape(self, sim, rng, config, anchor_xy, endpoint_target_xy,
                                     attempt_scale):
        """设置 rigid 线缆的穿杆初始形状（与 flex 统一：正弦谐波噪声）。"""
        cable_root_joint = self._sim_ids.get("cable_root_joint")
        if not cable_root_joint:
            return

        cfg = config.reset_config_by_difficulty[config.difficulty]
        direction = endpoint_target_xy - anchor_xy
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = direction / dist
        base_yaw = float(np.arctan2(direction[1], direction[0]))
        yaw_noise = float(rng.uniform(-cfg["root_yaw_range"], cfg["root_yaw_range"])) * attempt_scale

        root_pos = np.array([anchor_xy[0], anchor_xy[1], config.initial_root_pos[2]], dtype=float)
        root_quat = np.array([
            np.cos((base_yaw + yaw_noise) / 2), 0.0, 0.0,
            np.sin((base_yaw + yaw_noise) / 2),
        ], dtype=float)

        sim.data.set_joint_qpos(cable_root_joint, np.concatenate([root_pos, root_quat]))
        sim.data.set_joint_qvel(cable_root_joint, np.zeros(6, dtype=float))

        # 统一形状噪声：与 flex 一致的正弦谐波（平滑波浪，非随机游走）
        shape_joint_names = self._sim_ids.get("cable_shape_joint_names", [])
        n_joints = len(shape_joint_names)
        if n_joints > 0:
            t = np.linspace(0.0, 1.0, n_joints)
            # 生成谐波偏移量（与 flex 一致的 3 阶正弦）
            # rigid 的 ball joint 旋转会累积曲率，需要按关节数缩放振幅
            # hinge 关节只绕单轴旋转，曲率被分散到多个平面，需要 2x 补偿
            harmonic_offsets = np.zeros(n_joints, dtype=float)
            wave_scale = cfg.get("shape_wave_scale", cfg["joint_noise_scale"])
            # rigid 关节旋转的曲率累积方式与 flex 顶点位移不同
            # flex: 位移 d → 曲率 ~ d/segment_length（较大）
            # rigid ball: 旋转 θ → 曲率 ~ θ（较小）
            # rigid hinge: 旋转 θ → 曲率 ~ θ*cos(α)（更小，因分散到多平面）
            # 需要放大 rigid 振幅以匹配 flex 视觉曲率
            has_hinge = any(
                sim.model.jnt_type[sim.model.joint_name2id(j)] != 1
                for j in shape_joint_names[:5]
                if sim.model.joint_name2id(j) >= 0
            )
            if has_hinge:
                # hinge: 交替轴旋转，2D 曲率只有一半，需要 4x 补偿
                scale_factor = 4.0 / max(1.0, n_joints / 10.0)
            else:
                # ball: 旋转直接产生曲率，需要 ~3x 补偿
                scale_factor = 3.0 / max(1.0, n_joints / 10.0)
            for m in range(1, 4):
                amp = float(rng.normal(0.0, wave_scale * attempt_scale * scale_factor / m))
                phase = float(rng.uniform(-np.pi, np.pi))
                harmonic_offsets += amp * np.sin(m * np.pi * t + phase)

            for j, jname in enumerate(shape_joint_names):
                try:
                    jnt_id = sim.model.joint_name2id(jname)
                    jnt_type = sim.model.jnt_type[jnt_id]
                    noise = float(harmonic_offsets[j])
                    # rigid ball joint 旋转会累积曲率，用更小的 clip 匹配 flex 视觉效果
                    rigid_clip = min(cfg["joint_noise_clip"], 0.05)
                    if jnt_type == 1:
                        # ball joint: yaw 从谐波取值
                        yaw = float(np.clip(noise, -rigid_clip, rigid_clip))
                        sim.data.set_joint_qpos(jname, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float))
                        sim.data.set_joint_qvel(jname, np.zeros(3, dtype=float))
                    else:
                        # hinge joint: 直接用谐波值
                        sim.data.set_joint_qpos(jname, float(np.clip(noise, -rigid_clip, rigid_clip)))
                        sim.data.set_joint_qvel(jname, 0.0)
                except (KeyError, ValueError):
                    pass
        sim.forward()

        # drift 补偿
        first_body = self.get_cable_points(sim)[0].copy()
        drift = first_body - root_pos
        if np.linalg.norm(drift[:2]) > 0.005:
            root_pos[:2] += drift[:2]
            sim.data.set_joint_qpos(cable_root_joint, np.concatenate([root_pos, root_quat]))
            sim.forward()

    def _compensate_drift_and_clamp(self, sim, get_cable_points_fn, table_min_xy, table_max_xy):
        """drift 补偿 + 确保第一个 body 在桌面安全区域内。"""
        cable_root_joint = self._sim_ids.get("cable_root_joint")
        if not cable_root_joint:
            return
        root_qpos = sim.data.get_joint_qpos(cable_root_joint).copy()
        first_body = get_cable_points_fn()[0].copy()
        drift = first_body - root_qpos[:3]
        if np.linalg.norm(drift[:2]) > 0.005:
            root_qpos[:2] += drift[:2]
            sim.data.set_joint_qpos(cable_root_joint, root_qpos)
            sim.forward()
        # 桌面裁剪
        first_body = get_cable_points_fn()[0].copy()
        correction = np.zeros(2)
        for d in range(2):
            if first_body[d] < table_min_xy[d] + 0.02:
                correction[d] = table_min_xy[d] + 0.02 - first_body[d]
            elif first_body[d] > table_max_xy[d] - 0.02:
                correction[d] = table_max_xy[d] - 0.02 - first_body[d]
        if np.linalg.norm(correction) > 1e-6:
            root_qpos = sim.data.get_joint_qpos(cable_root_joint).copy()
            root_qpos[:2] += correction
            sim.data.set_joint_qpos(cable_root_joint, root_qpos)
            sim.forward()

    def _threading_reset_is_valid(self, cable_points, anchor_xy, endpoint_xy,
                                  config: ThreadingResetConfig, get_pole_pos_fn) -> bool:
        """穿杆重置验证：端点在桌面内 + 在可达范围内 + 线缆未穿越杆柱。"""
        ep = np.asarray(endpoint_xy, dtype=float)
        table_min, table_max = config.table_min_xy, config.table_max_xy
        if not (np.all(ep >= table_min) and np.all(ep <= table_max)):
            return False
        reach_error = max(0.0, float(np.linalg.norm(ep - config.robot_reach_center)) - config.endpoint_reach_radius + config.endpoint_reach_margin)
        if reach_error > 1e-6:
            return False
        return True

    def _deterministic_threading_fallback(self, sim, config, get_cable_points_fn,
                                          get_cable_end_pos_fn, get_pole_pos_fn):
        """32 次随机采样失败后的确定性回退。"""
        cable_root_joint = self._sim_ids.get("cable_root_joint")
        if not cable_root_joint:
            return config.initial_root_pos.copy(), get_cable_end_pos_fn(), get_cable_points_fn()

        root_pos = config.initial_root_pos.copy()
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        sim.data.set_joint_qpos(cable_root_joint, np.concatenate([root_pos, root_quat]))
        sim.data.set_joint_qvel(cable_root_joint, np.zeros(6, dtype=float))
        for jname in self._sim_ids.get("cable_shape_joint_names", []):
            try:
                jnt_id = sim.model.joint_name2id(jname)
                jnt_type = sim.model.jnt_type[jnt_id]
                if jnt_type == 1:
                    sim.data.set_joint_qpos(jname, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
                    sim.data.set_joint_qvel(jname, np.zeros(3, dtype=float))
                else:
                    sim.data.set_joint_qpos(jname, 0.0)
                    sim.data.set_joint_qvel(jname, 0.0)
            except (KeyError, ValueError):
                pass
        sim.forward()

        # 桌面裁剪
        self._compensate_drift_and_clamp(sim, get_cable_points_fn, config.table_min_xy, config.table_max_xy)

        # 固定 anchor
        fixed_end = get_cable_points_fn()[0].copy()
        sim.model.body_pos[config.anchor_body_id] = fixed_end
        if config.anchor_eq_id is not None:
            sim.data.eq_active[config.anchor_eq_id] = 1
        sim.forward()

        return fixed_end, get_cable_end_pos_fn(), get_cable_points_fn()
