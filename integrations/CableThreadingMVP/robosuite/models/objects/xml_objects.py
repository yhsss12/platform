"""
xml_objects.py -- 线缆对象模型定义

本文件为 dloBench 的所有可变形线性物体（DLO）定义 MuJoCo 物理模型。
核心设计：

1. CableModelMixin 抽象接口：所有线缆后端的统一协议，提供：
   - cable_radius：线缆半径，用于碰撞检测和夹爪校准
   - point_reference_kind：观测系统的引用类型，决定如何从 MuJoCo data 中读取线缆关键点
     * "site"：从 data.site_xpos 读取（CableObject 的 13 段 capsule chain）
     * "body"：从 data.xpos 读取（Composite/RMB/Gemini 的刚体链）
     * "flex"：从 data.flexvert_xpos 读取（FlexCableObject 的柔性体）
   - point_reference_names：关键点名称列表
   - graspable_body_names：可被抓取的 body 名称（仅 body 类型有）
   - attachment_root_offset / tabletop_centerline_offset / recommended_attach_offset：
     夹爪附件（attachment weld）的空间偏移量

2. 四种线缆后端实现：
   - CableObject（segmented/capsule_chain）：13 段 capsule 刚体链
   - CompositeCableObject（composite_cable/composite/mujoco_composite/deformable_ravens_composite/mujoco_cable/flex_reference_composite）：MuJoCo composite cable
   - RMBCableObject（rmb）：RoboManipBaselines 的 25 body 刚体链
   - FlexCableObject（flex/flexcomp）：MuJoCo 3.x 的 flexcomp 柔性体

3. cable_object_factory：工厂函数，将字符串名称映射到具体类

4. 其他物体类：瓶子、罐头、柠檬、牛奶等刚体物体（用于 PickPlace 等任务）
"""

import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, new_inertial, xml_path_completion


class CableModelMixin:
    """所有线缆模型后端的公共接口（Mixin 模式）。

    每个线缆后端必须实现以下属性，用于：
    - 碰撞检测和物理仿真（cable_radius）
    - 观测系统（point_reference_kind, point_reference_names）
    - 夹爪附件定位（attachment_root_offset, recommended_attach_offset）
    - 桌面高度校准（tabletop_centerline_offset）
    """

    @property
    def cable_radius(self):
        """线缆半径（米），用于碰撞检测和夹爪开口校准。"""
        raise NotImplementedError

    @property
    def attachment_root_offset(self):
        """附件根节点偏移：线缆初始放置时相对于世界原点的平移向量。"""
        raise NotImplementedError

    @property
    def tabletop_centerline_offset(self):
        """线缆中心线与桌面的垂直偏移量（米），用于桌面高度校准。"""
        raise NotImplementedError

    @property
    def recommended_attach_offset(self):
        """推荐的夹爪附件偏移量：mocap body 相对于抓取点的偏移。"""
        raise NotImplementedError

    @property
    def point_reference_kind(self):
        """关键点引用类型：'site'（从 site_xpos 读取）、'body'（从 xpos 读取）或 'flex'（从 flexvert_xpos 读取）。"""
        raise NotImplementedError

    @property
    def point_reference_names(self):
        """关键点名称列表，对应 MuJoCo data 中的 site/body 名称。"""
        raise NotImplementedError

    @property
    def graspable_body_names(self):
        """可被抓取的 body 名称列表。仅 body 类型有可抓取体；site/flex 类型返回空列表。"""
        return list(self.point_reference_names) if self.point_reference_kind == "body" else []

    @property
    def graspable_point_count(self):
        """可抓取点的数量。"""
        return len(self.graspable_body_names)

    def body_name_for_point_idx(self, idx):
        """根据关键点索引返回对应的 body 名称。"""
        bodies = self.graspable_body_names
        return bodies[int(idx)]


class CableObject(CableModelMixin, MujocoXMLObject):
    """分段胶囊链（capsule chain）线缆模型。

    13 段刚体通过铰链关节连接，每段是一个 capsule 碰撞体。
    这是最原始的线缆后端，参考点类型为 "site"——
    13 个 cable_site_{i} site 嵌入在各段 body 中，观测系统从 data.site_xpos 读取。
    可抓取端点只有两个：cable_B0（起点）和 cable_end（终点）。
    """

    def __init__(self, name="cable", xml_file="objects/dlo/cable_12.xml"):
        # free 关节让线缆根节点可在空间中自由移动；damping=0.0005 防止数值振荡
        super().__init__(
            xml_path_completion(xml_file),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    def exclude_from_prefixing(self, inp):
        """排除以 'cable_' 开头的元素，避免被 MujoCoXMLObject 重复加前缀。"""
        return isinstance(inp, str) and inp.startswith("cable_")

    @property
    def cable_radius(self):
        return 0.007

    @property
    def attachment_root_offset(self):
        # 线缆初始放置在 x=-0.18 处，即从桌面左侧开始
        return np.array([-0.18, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self):
        return 0.009

    @property
    def recommended_attach_offset(self):
        # 夹爪附件向下偏移 35mm，让夹爪从上方夹住线缆
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self):
        # site 类型：观测系统从 data.site_xpos 读取 13 个 cable_site_{i} 的位置
        return "site"

    @property
    def point_reference_names(self):
        # 13 个 site，均匀分布在 13 段 capsule 上
        return [f"cable_site_{i:02d}" for i in range(13)]

    @property
    def graspable_body_names(self):
        # 只有首尾两段可被抓取
        return ["cable_B0", "cable_end"]

    def body_name_for_point_idx(self, idx):
        """将关键点索引映射到可抓取 body：索引 0 -> cable_B0，其余 -> cable_end。"""
        return "cable_B0" if int(idx) <= 0 else "cable_end"


class CompositeCableObject(CableModelMixin, MujocoXMLObject):
    """MuJoCo composite cable（对齐 flex_cable 几何参数）。

    使用 MuJoCo 原生 <composite type="cable"> 的 body 关键点路径，
    段数 50、总长 0.5m、半径 0.004m，与 flex 线缆几何一致。
    """

    _SEGMENT_COUNT = 50
    _POINT_COUNT = _SEGMENT_COUNT - 1
    _CABLE_LENGTH = 0.5
    _CABLE_RADIUS = 0.004
    _PREFIX = "flexrefc_"

    def __init__(self, name="cable", xml_file="objects/dlo/flex_reference_composite_cable.xml", joints="default"):
        super().__init__(
            xml_path_completion(xml_file),
            name=name,
            joints=[dict(type="free", damping="0.0005")] if joints == "default" else joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        composite = self._obj.find("./composite")
        if composite is not None:
            composite.set("prefix", f"{self.naming_prefix}{self._PREFIX}")
        if joints is not None and self._obj.find("./inertial") is None:
            self._obj.insert(0, new_inertial(pos=(0, 0, 0), mass=1e-3, diaginertia=(1e-6, 1e-6, 1e-6)))

    def exclude_from_prefixing(self, inp):
        return isinstance(inp, str) and inp.startswith(self._PREFIX)

    @property
    def cable_radius(self):
        return self._CABLE_RADIUS

    @property
    def cable_length(self):
        return self._CABLE_LENGTH

    @property
    def attachment_root_offset(self):
        return np.array([-self._CABLE_LENGTH / 2, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self):
        return 0.008

    @property
    def recommended_attach_offset(self):
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self):
        return "body"

    @property
    def point_reference_names(self):
        return (
            [f"{self._PREFIX}B_first"]
            + [f"{self._PREFIX}B_{i}" for i in range(1, self._POINT_COUNT - 1)]
            + [f"{self._PREFIX}B_last"]
        )


class CompositeImproveObject(CableModelMixin, MujocoXMLObject):
    """优化版 composite 线缆：增大碰撞体半径、提高摩擦力，对齐 rmb 参数。"""

    _SEGMENT_COUNT = 50
    _POINT_COUNT = _SEGMENT_COUNT - 1
    _CABLE_LENGTH = 0.5
    _CABLE_RADIUS = 0.007
    _PREFIX = "flexrefc_"

    def __init__(self, name="cable", xml_file="objects/dlo/flex_reference_composite_cable_improve.xml", joints="default"):
        super().__init__(
            xml_path_completion(xml_file),
            name=name,
            joints=[dict(type="free", damping="0.0005")] if joints == "default" else joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        composite = self._obj.find("./composite")
        if composite is not None:
            composite.set("prefix", f"{self.naming_prefix}{self._PREFIX}")
        if joints is not None and self._obj.find("./inertial") is None:
            self._obj.insert(0, new_inertial(pos=(0, 0, 0), mass=1e-3, diaginertia=(1e-6, 1e-6, 1e-6)))

    def exclude_from_prefixing(self, inp):
        return isinstance(inp, str) and inp.startswith(self._PREFIX)

    @property
    def cable_radius(self):
        return self._CABLE_RADIUS

    @property
    def cable_length(self):
        return self._CABLE_LENGTH

    @property
    def attachment_root_offset(self):
        return np.array([-self._CABLE_LENGTH / 2, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self):
        return 0.080

    @property
    def recommended_attach_offset(self):
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self):
        return "body"

    @property
    def graspable_body_names(self):
        return [f"{self._PREFIX}B_first", f"{self._PREFIX}B_last"]

    @property
    def graspable_point_count(self):
        return 2

    def body_name_for_point_idx(self, idx):
        return f"{self._PREFIX}B_first" if int(idx) <= 0 else f"{self._PREFIX}B_last"

    @property
    def point_reference_names(self):
        return (
            [f"{self._PREFIX}B_first"]
            + [f"{self._PREFIX}B_{i}" for i in range(1, self._POINT_COUNT - 1)]
            + [f"{self._PREFIX}B_last"]
        )


class CompositeSoftObject(CableModelMixin, MujocoXMLObject):
    """柔软版 composite 线缆：低刚度、低阻尼、匹配 flex 接触参数。

    与标准 composite 的区别：
    - bend=5e2, twist=1e2（标准 1.2e3/3e2）
    - joint damping=0.005（标准 0.02）
    - armature=0.0001（标准 0.0005）
    - density=800（标准 1000）
    - condim=6, solref=0.006（标准 condim=4, solref=0.02）
    - 60 段（标准 50 段）
    """

    _SEGMENT_COUNT = 60
    _POINT_COUNT = _SEGMENT_COUNT - 1
    _CABLE_LENGTH = 0.5
    _CABLE_RADIUS = 0.004
    _PREFIX = "flexrefc_"

    def __init__(self, name="cable", xml_file="objects/dlo/flex_reference_composite_cable_soft.xml", joints="default"):
        super().__init__(
            xml_path_completion(xml_file),
            name=name,
            joints=[dict(type="free", damping="0.0005")] if joints == "default" else joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        composite = self._obj.find("./composite")
        if composite is not None:
            composite.set("prefix", f"{self.naming_prefix}{self._PREFIX}")
        if joints is not None and self._obj.find("./inertial") is None:
            self._obj.insert(0, new_inertial(pos=(0, 0, 0), mass=1e-3, diaginertia=(1e-6, 1e-6, 1e-6)))

    def exclude_from_prefixing(self, inp):
        return isinstance(inp, str) and inp.startswith(self._PREFIX)

    @property
    def cable_radius(self):
        return self._CABLE_RADIUS

    @property
    def cable_length(self):
        return self._CABLE_LENGTH

    @property
    def attachment_root_offset(self):
        return np.array([-self._CABLE_LENGTH / 2, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self):
        return 0.008

    @property
    def recommended_attach_offset(self):
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self):
        return "body"

    @property
    def graspable_body_names(self):
        return [f"{self._PREFIX}B_first", f"{self._PREFIX}B_last"]

    @property
    def graspable_point_count(self):
        return 2

    def body_name_for_point_idx(self, idx):
        return f"{self._PREFIX}B_first" if int(idx) <= 0 else f"{self._PREFIX}B_last"

    @property
    def point_reference_names(self):
        return (
            [f"{self._PREFIX}B_first"]
            + [f"{self._PREFIX}B_{i}" for i in range(1, self._POINT_COUNT - 1)]
            + [f"{self._PREFIX}B_last"]
        )


class CompositeThinObject(CableModelMixin, MujocoXMLObject):
    """细径高段数 composite 线缆：通过减小半径和增加段数提高柔顺性。

    相比标准 composite 的改进：
    - 半径 3mm（标准 4mm）：更细的碰撞体更容易穿过间隙
    - 80 段（标准 50）：更细粒度的弯曲，更接近连续体
    - 保持 damping=0.02, armature=0.0005：成功的关键控制参数
    - condim=6, solref=0.006：匹配 flex 的接触参数
    """

    _SEGMENT_COUNT = 80
    _POINT_COUNT = _SEGMENT_COUNT - 1
    _CABLE_LENGTH = 0.5
    _CABLE_RADIUS = 0.003
    _PREFIX = "flexrefc_"

    def __init__(self, name="cable", xml_file="objects/dlo/flex_reference_composite_cable_thin.xml", joints="default"):
        super().__init__(
            xml_path_completion(xml_file),
            name=name,
            joints=[dict(type="free", damping="0.0005")] if joints == "default" else joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        composite = self._obj.find("./composite")
        if composite is not None:
            composite.set("prefix", f"{self.naming_prefix}{self._PREFIX}")
        if joints is not None and self._obj.find("./inertial") is None:
            self._obj.insert(0, new_inertial(pos=(0, 0, 0), mass=1e-3, diaginertia=(1e-6, 1e-6, 1e-6)))

    def exclude_from_prefixing(self, inp):
        return isinstance(inp, str) and inp.startswith(self._PREFIX)

    @property
    def cable_radius(self):
        return self._CABLE_RADIUS

    @property
    def cable_length(self):
        return self._CABLE_LENGTH

    @property
    def attachment_root_offset(self):
        return np.array([-self._CABLE_LENGTH / 2, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self):
        return 0.006

    @property
    def recommended_attach_offset(self):
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self):
        return "body"

    @property
    def graspable_body_names(self):
        return [f"{self._PREFIX}B_first", f"{self._PREFIX}B_last"]

    @property
    def graspable_point_count(self):
        return 2

    def body_name_for_point_idx(self, idx):
        return f"{self._PREFIX}B_first" if int(idx) <= 0 else f"{self._PREFIX}B_last"

    @property
    def point_reference_names(self):
        return (
            [f"{self._PREFIX}B_first"]
            + [f"{self._PREFIX}B_{i}" for i in range(1, self._POINT_COUNT - 1)]
            + [f"{self._PREFIX}B_last"]
        )


class RMBCableObject(CableModelMixin, MujocoXMLObject):
    """RoboManipBaselines 的刚体链线缆模型（25 个 body）。

    参考点类型为 "body"，有 25 个可抓取点（cable_B0..cable_B24 + cable_end）。
    这是 dloBench 的默认线缆后端（default_cable_model="rmb"），
    适用于大多数端点操作任务（CableStraighten、CableMoveToTarget 等）。
    """

    def __init__(self, name="cable", xml_file="objects/dlo/cable_rmb.xml"):
        super().__init__(
            xml_path_completion(xml_file),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    def exclude_from_prefixing(self, inp):
        return isinstance(inp, str) and inp.startswith("cable_")

    @property
    def cable_radius(self):
        return 0.0075

    @property
    def cable_length(self):
        return 0.5

    @property
    def attachment_root_offset(self):
        # RMB 线缆默认放置在世界原点
        return np.array([0.0, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self):
        return 0.017

    @property
    def recommended_attach_offset(self):
        return np.array([0.0, 0.0, -0.03], dtype=float)

    @property
    def point_reference_kind(self):
        return "body"

    @property
    def point_reference_names(self):
        # 25 个关键点 body：cable_B0 到 cable_B24
        return [f"cable_B{i}" for i in range(25)]

    @property
    def graspable_body_names(self):
        # 25 个可抓取点：cable_B0..B23 + cable_end（第 25 个段的末端）
        return [f"cable_B{i}" for i in range(24)] + ["cable_end"]

    def body_name_for_point_idx(self, idx):
        """将关键点索引映射到可抓取 body：最后一个索引映射到 cable_end。"""
        idx = int(idx)
        return "cable_end" if idx >= self.graspable_point_count - 1 else f"cable_B{idx}"


class FlexCableObject(CableModelMixin):
    """基于 MuJoCo 3.x flexcomp 的柔性线缆模型。

    与前四种模型的关键区别：
    1. 不继承 MujocoXMLObject——flexcomp 的 XML 由 scene（arena）直接加载，不通过 XML object 机制
    2. 参考点类型为 "flex"——顶点位置从 data.flexvert_xpos 读取，而非 data.site_xpos 或 data.xpos
    3. 顶点数量和线缆长度从 flex_cable.xml 动态解析，point_reference_names 只返回一个名称 "flex_cable"
    4. 可抓取端点只有两个：flex_cable_0（起点）和 flex_cable_{N-1}（终点）
    """

    _FLEX_NAME = "flex_cable"

    def __init__(self, name="cable"):
        self._name = name
        # 从 flex_cable.xml 动态解析顶点数、间距和线缆长度
        self._vertex_count, self._cable_length, self._cable_radius = self._parse_flex_xml()

    @classmethod
    def _parse_flex_xml(cls):
        """解析 flex_cable.xml 提取顶点数、线缆长度和半径。"""
        import xml.etree.ElementTree as ET
        from robosuite.utils.mjcf_utils import xml_path_completion
        flex_xml_path = xml_path_completion("objects/dlo/flex_cable.xml")
        tree = ET.parse(flex_xml_path)
        flexcomp = tree.getroot().find(".//flexcomp")
        count_tokens = flexcomp.get("count", "").split()
        vertex_count = int(count_tokens[0]) if count_tokens else 51
        spacing_tokens = flexcomp.get("spacing", "").split()
        spacing = float(spacing_tokens[0]) if spacing_tokens else 0.01
        cable_length = (vertex_count - 1) * spacing
        cable_radius = float(flexcomp.get("radius", "0.004"))
        return vertex_count, cable_length, cable_radius

    @property
    def naming_prefix(self):
        return ""

    @property
    def cable_radius(self):
        return self._cable_radius

    @property
    def cable_length(self):
        return self._cable_length

    @property
    def attachment_root_offset(self):
        return np.array([-self._cable_length / 2, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self):
        return 0.01

    @property
    def recommended_attach_offset(self):
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self):
        return "flex"

    @property
    def point_reference_names(self):
        return [self._FLEX_NAME]

    @property
    def flex_vertex_count(self):
        return self._vertex_count

    @property
    def graspable_body_names(self):
        return [f"{self._FLEX_NAME}_0", f"{self._FLEX_NAME}_{self._vertex_count - 1}"]

    @property
    def graspable_point_count(self):
        return 2

    def body_name_for_point_idx(self, idx):
        return f"{self._FLEX_NAME}_0" if int(idx) <= 0 else f"{self._FLEX_NAME}_{self._vertex_count - 1}"


class FlexImproveObject(CableModelMixin):
    """优化版 flex 线缆：增大碰撞体半径，降低弯曲刚度，提高物理抓取成功率。"""

    _FLEX_NAME = "flex_improve"

    def __init__(self, name="cable"):
        self._name = name
        self._vertex_count, self._cable_length, self._cable_radius = self._parse_flex_xml()

    @classmethod
    def _parse_flex_xml(cls):
        import xml.etree.ElementTree as ET
        from robosuite.utils.mjcf_utils import xml_path_completion
        flex_xml_path = xml_path_completion("objects/dlo/flex_improve.xml")
        tree = ET.parse(flex_xml_path)
        flexcomp = tree.getroot().find(".//flexcomp")
        count_tokens = flexcomp.get("count", "").split()
        vertex_count = int(count_tokens[0]) if count_tokens else 51
        spacing_tokens = flexcomp.get("spacing", "").split()
        spacing = float(spacing_tokens[0]) if spacing_tokens else 0.01
        cable_length = (vertex_count - 1) * spacing
        cable_radius = float(flexcomp.get("radius", "0.008"))
        return vertex_count, cable_length, cable_radius

    @property
    def naming_prefix(self):
        return ""

    @property
    def cable_radius(self):
        return self._cable_radius

    @property
    def cable_length(self):
        return self._cable_length

    @property
    def attachment_root_offset(self):
        return np.array([-self._cable_length / 2, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self):
        return 0.01

    @property
    def recommended_attach_offset(self):
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self):
        return "flex"

    @property
    def point_reference_names(self):
        return [self._FLEX_NAME]

    @property
    def flex_vertex_count(self):
        return self._vertex_count

    @property
    def graspable_body_names(self):
        return [f"{self._FLEX_NAME}_0", f"{self._FLEX_NAME}_{self._vertex_count - 1}"]

    @property
    def graspable_point_count(self):
        return 2

    def body_name_for_point_idx(self, idx):
        return f"{self._FLEX_NAME}_0" if int(idx) <= 0 else f"{self._FLEX_NAME}_{self._vertex_count - 1}"


def cable_object_factory(cable_model, name="cable"):
    """线缆模型工厂函数：将字符串名称映射到具体的线缆类实例。

    委托给 CableModelRegistry 统一管理。支持所有已注册的模型名称及别名。

    Args:
        cable_model: 模型名称或别名 (不区分大小写)
        instance_name: 实例名 (传给构造函数的 name= 参数)

    Returns:
        CableModelMixin 实例
    """
    from .cable_registry import get_registry
    return get_registry().create(cable_model, instance_name=name)

class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/plate-with-hole.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = "objects/door.xml"
        if lock:
            xml_path = "objects/door_lock.xml"
        super().__init__(
            xml_path_completion(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic
