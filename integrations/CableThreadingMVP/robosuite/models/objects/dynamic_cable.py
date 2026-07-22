"""
DynamicCable -- 动态生成不同静态参数的线缆 XML 模型

支持 flex、rmb、composite 三种线缆类型的参数化生成。
每种类型的关键可调参数：

- flex: 顶点数、间距(→长度)、半径、弯曲/扭转刚度
- rmb: 段数、段长(→总长)、半径、阻尼、密度
- composite: 段数、总长、半径、弯曲/扭转刚度

使用方式：
    spec = CableSpec(cable_type="flex", cable_length=0.5, num_segments=51)
    generator = CableModelGenerator()
    xml_string = generator.generate(spec)
    # 或保存到文件
    generator.generate_to_file(spec, "my_cable.xml")
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import numpy as np

from robosuite.models.objects.xml_objects import CableModelMixin


# ---------------------------------------------------------------------------
# CableSpec: 线缆规格参数
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CableSpec:
    """线缆规格参数。

    所有参数都有合理默认值，只需指定想要修改的参数。
    """
    cable_type: Literal["flex", "rmb", "composite"] = "flex"

    # 几何参数
    cable_length: float = 0.5          # 总长度 (m)
    cable_radius: float = 0.004        # 碰撞半径 (m)
    num_segments: int = 51             # 段数/顶点数

    # 物理参数 (flex/composite)
    twist_stiffness: float = 1e2       # 扭转刚度
    bend_stiffness: float = 5e2        # 弯曲刚度
    young_modulus: float = 1e4         # 杨氏模量 (仅 flex)
    poisson_ratio: float = 0.3         # 泊松比 (仅 flex)

    # 物理参数 (rmb/composite)
    damping: float = 0.005             # 关节阻尼
    density: float = 1000              # 材料密度 (kg/m^3)
    friction: tuple = (1.2, 0.005, 0.001)  # 摩擦系数 [slide, torsion, rolling]

    # 命名前缀 (避免多实例冲突)
    prefix: str = ""

    def __post_init__(self):
        if self.cable_type not in ("flex", "rmb", "composite"):
            raise ValueError(f"Unsupported cable_type: {self.cable_type}")
        if self.cable_length <= 0:
            raise ValueError(f"cable_length must be > 0, got {self.cable_length}")
        if self.num_segments < 2:
            raise ValueError(f"num_segments must be >= 2, got {self.num_segments}")

    @property
    def segment_length(self) -> float:
        """单段长度 (m)。"""
        if self.cable_type == "flex":
            return self.cable_length / (self.num_segments - 1)
        return self.cable_length / self.num_segments

    @property
    def body_prefix(self) -> str:
        """body 名称前缀。"""
        if self.cable_type == "flex":
            return f"{self.prefix}flex_cable" if self.prefix else "flex_cable"
        if self.cable_type == "rmb":
            return f"{self.prefix}cable" if self.prefix else "cable"
        if self.cable_type == "composite":
            return f"{self.prefix}cablec" if self.prefix else "cablec"
        return "cable"


# ---------------------------------------------------------------------------
# CableModelGenerator: XML 生成器
# ---------------------------------------------------------------------------

class CableModelGenerator:
    """线缆 XML 模型生成器。"""

    def generate(self, spec: CableSpec) -> str:
        """根据规格生成 XML 字符串。"""
        if spec.cable_type == "flex":
            return self._generate_flex_xml(spec)
        if spec.cable_type == "rmb":
            return self._generate_rmb_xml(spec)
        if spec.cable_type == "composite":
            return self._generate_composite_xml(spec)
        raise ValueError(f"Unsupported cable_type: {spec.cable_type}")

    def generate_to_file(self, spec: CableSpec, filepath: str | Path) -> Path:
        """生成 XML 并保存到文件。"""
        xml_string = self.generate(spec)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml_string, encoding="utf-8")
        return path

    # ---- flex ----

    def _generate_flex_xml(self, spec: CableSpec) -> str:
        """生成 flexcomp 线缆 XML。

        flexcomp grid 在 body 原点居中，顶点间距 = cable_length / (num_segments - 1)。
        """
        spacing = spec.segment_length
        prefix = spec.prefix

        return f'''<mujoco model="dynamic_flex_cable">
  <!--
    动态生成的 flexcomp 线缆模型
    参数: length={spec.cable_length}m, segments={spec.num_segments}, radius={spec.cable_radius}m
    spacing={spacing:.6f}m, twist={spec.twist_stiffness}, bend={spec.bend_stiffness}
  -->

  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <option timestep="0.005" solver="Newton" tolerance="1e-6"/>

  <default>
    <geom condim="4" friction="{spec.friction[0]} {spec.friction[1]} {spec.friction[2]}"
          solref="0.02 1" solimp="0.9 0.95 0.001" margin="0.001"/>
  </default>

  <worldbody>
    <body>
      <body name="object">
        <flexcomp
          name="{prefix}flex_cable"
          type="grid"
          count="{spec.num_segments} 1 1"
          spacing="{spacing:.6f} {spacing:.6f} {spacing:.6f}"
          radius="{spec.cable_radius}"
          dim="1"
          rgba="0.8 0.2 0.2 1"
        >
          <edge equality="true" stiffness="0" damping="{spec.damping}"
                 solref="0.01 1" solimp="0.95 0.99 0.001"/>
          <elasticity young="{spec.young_modulus}" poisson="{spec.poisson_ratio}"
                      damping="0.01" thickness="0.001"/>
          <plugin plugin="mujoco.elasticity.cable">
            <config key="twist" value="{spec.twist_stiffness}"/>
            <config key="bend" value="{spec.bend_stiffness}"/>
            <config key="vmax" value="0"/>
          </plugin>
        </flexcomp>
      </body>
      <site name="bottom_site" pos="0 0 -0.008" size="0.005" rgba="0 0 0 0"/>
      <site name="top_site" pos="0 0 0.008" size="0.005" rgba="0 0 0 0"/>
      <site name="horizontal_radius_site" pos="0.15 0 0" size="0.005" rgba="0 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
'''

    # ---- rmb ----

    def _generate_rmb_xml(self, spec: CableSpec) -> str:
        """生成 RMB 刚体链线缆 XML。

        每段是一个嵌套 body，包含 2 个 hinge joints (Y/Z 轴) 和 1 个 capsule geom。
        """
        seg_len = spec.segment_length
        half_len = seg_len / 2
        prefix = spec.prefix
        body_prefix = f"{prefix}cable" if prefix else "cable"
        joint_prefix = f"{prefix}cable" if prefix else "cable"
        geom_prefix = f"{prefix}cable" if prefix else "cable"

        # 生成嵌套 body 链
        bodies_xml = self._generate_rmb_body_chain(
            spec.num_segments, seg_len, half_len,
            body_prefix, joint_prefix, geom_prefix, spec
        )

        friction_str = f"{spec.friction[0]} {spec.friction[1]} {spec.friction[2]}"

        return f'''<mujoco model="dynamic_rmb_cable">
  <!--
    动态生成的 RMB 刚体链线缆模型
    参数: length={spec.cable_length}m, segments={spec.num_segments}, radius={spec.cable_radius}m
    segment_length={seg_len:.6f}m, damping={spec.damping}, density={spec.density}
  -->

  <asset>
    <material name="{body_prefix}_mat" rgba="0.92 0.92 0.92 1" specular="0.2" shininess="0.3"/>
    <material name="{body_prefix}_box_mat" rgba="1 1 0.2 1" specular="0.2" shininess="0.3"/>
  </asset>

  <default>
    <geom friction="{friction_str}" condim="4" solref="0.01 0.5"
          solimp="0.9 0.95 0.001" density="{spec.density}"/>
  </default>

  <worldbody>
    <body>
      <body name="object">
        {bodies_xml}
      </body>
    </body>
  </worldbody>
</mujoco>
'''

    def _generate_rmb_body_chain(
        self, num_segments: int, seg_len: float, half_len: float,
        body_prefix: str, joint_prefix: str, geom_prefix: str,
        spec: CableSpec
    ) -> str:
        """生成 RMB 嵌套 body 链。"""
        lines = []

        for i in range(num_segments):
            indent = "        " + "  " * i  # 缩进

            if i == 0:
                # 第一个 body: cable_B0 (无 joint)
                lines.append(f'{indent}<body name="{body_prefix}_B{i}">')
                lines.append(f'{indent}  <geom name="{geom_prefix}_endpoint_start" '
                             f'size="0.02 0.015 0.015" type="box" '
                             f'material="{body_prefix}_box_mat"/>')
                lines.append(f'{indent}  <geom name="{geom_prefix}_G{i}" '
                             f'size="{spec.cable_radius} {half_len:.6f}" '
                             f'quat="0.707107 0 0.707107 0" type="capsule" '
                             f'material="{body_prefix}_mat"/>')
            else:
                # 后续 body: 带 joints
                lines.append(f'{indent}<body name="{body_prefix}_B{i}" pos="{seg_len:.6f} 0 0">')
                lines.append(f'{indent}  <joint name="{joint_prefix}_J0_{i}" '
                             f'pos="{-half_len:.6f} 0 0" axis="0 1 0" '
                             f'group="3" damping="{spec.damping}"/>')
                lines.append(f'{indent}  <joint name="{joint_prefix}_J1_{i}" '
                             f'pos="{-half_len:.6f} 0 0" axis="0 0 1" '
                             f'group="3" damping="{spec.damping}"/>')
                lines.append(f'{indent}  <geom name="{geom_prefix}_G{i}" '
                             f'size="{spec.cable_radius} {half_len:.6f}" '
                             f'quat="0.707107 0 0.707107 0" type="capsule" '
                             f'material="{body_prefix}_mat"/>')

        # 闭合所有 body 标签
        for i in range(num_segments - 1, -1, -1):
            indent = "        " + "  " * i
            lines.append(f'{indent}</body>')

        return "\n".join(lines)

    # ---- composite ----

    def _generate_composite_xml(self, spec: CableSpec) -> str:
        """生成 MuJoCo composite cable XML。"""
        prefix = spec.prefix
        composite_prefix = f"{prefix}cablec" if prefix else "cablec"
        offset = -spec.cable_length / 2
        friction_str = f"{spec.friction[0]} {spec.friction[1]} {spec.friction[2]}"

        return f'''<mujoco model="dynamic_composite_cable">
  <!--
    动态生成的 composite cable 模型
    参数: length={spec.cable_length}m, segments={spec.num_segments}, radius={spec.cable_radius}m
    twist={spec.twist_stiffness}, bend={spec.bend_stiffness}
  -->

  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <worldbody>
    <body>
      <body name="object">
        <composite
          prefix="{composite_prefix}_"
          type="cable"
          curve="s"
          count="{spec.num_segments} 1 1"
          size="{spec.cable_length}"
          offset="{offset:.6f} 0 0"
          initial="none"
        >
          <plugin plugin="mujoco.elasticity.cable">
            <config key="twist" value="{spec.twist_stiffness}"/>
            <config key="bend" value="{spec.bend_stiffness}"/>
            <config key="vmax" value="0"/>
          </plugin>
          <joint kind="main" damping="{spec.damping}" armature="0.002"/>
          <geom
            type="capsule"
            size="{spec.cable_radius}"
            group="0"
            rgba="0.85 0.2 0.1 1"
            friction="{friction_str}"
            density="{spec.density}"
            condim="4"
          />
        </composite>
      </body>
      <site name="bottom_site" pos="0 0 -0.006" size="0.005" rgba="0 0 0 0"/>
      <site name="top_site" pos="0 0 0.006" size="0.005" rgba="0 0 0 0"/>
      <site name="horizontal_radius_site" pos="0.18 0 0" size="0.005" rgba="0 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
'''


# ---------------------------------------------------------------------------
# DynamicCableObject: 包装生成的 XML，实现 CableModelMixin
# ---------------------------------------------------------------------------

class DynamicFlexCableObject(CableModelMixin):
    """动态生成的 flex 线缆对象。"""

    def __init__(self, spec: CableSpec):
        if spec.cable_type != "flex":
            raise ValueError(f"Expected flex, got {spec.cable_type}")
        self._spec = spec
        self._name = "cable"

    @property
    def naming_prefix(self) -> str:
        return self._spec.prefix

    @property
    def cable_radius(self) -> float:
        return self._spec.cable_radius

    @property
    def cable_length(self) -> float:
        return self._spec.cable_length

    @property
    def attachment_root_offset(self) -> np.ndarray:
        return np.array([-self._spec.cable_length / 2, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self) -> float:
        return 0.01

    @property
    def recommended_attach_offset(self) -> np.ndarray:
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self) -> str:
        return "flex"

    @property
    def point_reference_names(self) -> list[str]:
        return [self._spec.body_prefix]

    @property
    def flex_vertex_count(self) -> int:
        return self._spec.num_segments

    @property
    def graspable_body_names(self) -> list[str]:
        prefix = self._spec.body_prefix
        return [f"{prefix}_0", f"{prefix}_{self._spec.num_segments - 1}"]

    @property
    def graspable_point_count(self) -> int:
        return 2

    def body_name_for_point_idx(self, idx: int) -> str:
        prefix = self._spec.body_prefix
        return f"{prefix}_0" if int(idx) <= 0 else f"{prefix}_{self._spec.num_segments - 1}"


class DynamicRMBCableObject(CableModelMixin):
    """动态生成的 RMB 线缆对象。"""

    def __init__(self, spec: CableSpec):
        if spec.cable_type != "rmb":
            raise ValueError(f"Expected rmb, got {spec.cable_type}")
        self._spec = spec
        self._name = "cable"

    @property
    def naming_prefix(self) -> str:
        return self._spec.prefix

    @property
    def cable_radius(self) -> float:
        return self._spec.cable_radius

    @property
    def cable_length(self) -> float:
        return self._spec.cable_length

    @property
    def attachment_root_offset(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self) -> float:
        return 0.017

    @property
    def recommended_attach_offset(self) -> np.ndarray:
        return np.array([0.0, 0.0, -0.03], dtype=float)

    @property
    def point_reference_kind(self) -> str:
        return "body"

    @property
    def point_reference_names(self) -> list[str]:
        prefix = self._spec.body_prefix
        return [f"{prefix}_B{i}" for i in range(self._spec.num_segments)]

    @property
    def graspable_body_names(self) -> list[str]:
        prefix = self._spec.body_prefix
        return [f"{prefix}_B{i}" for i in range(self._spec.num_segments - 1)] + \
               [f"{prefix}_end"]

    @property
    def graspable_point_count(self) -> int:
        return self._spec.num_segments

    def body_name_for_point_idx(self, idx: int) -> str:
        prefix = self._spec.body_prefix
        idx = int(idx)
        if idx >= self.graspable_point_count - 1:
            return f"{prefix}_end"
        return f"{prefix}_B{idx}"


class DynamicCompositeCableObject(CableModelMixin):
    """动态生成的 composite cable 对象。"""

    def __init__(self, spec: CableSpec):
        if spec.cable_type != "composite":
            raise ValueError(f"Expected composite, got {spec.cable_type}")
        self._spec = spec
        self._name = "cable"

    @property
    def naming_prefix(self) -> str:
        return self._spec.prefix

    @property
    def cable_radius(self) -> float:
        return self._spec.cable_radius

    @property
    def cable_length(self) -> float:
        return self._spec.cable_length

    @property
    def attachment_root_offset(self) -> np.ndarray:
        return np.array([-self._spec.cable_length / 2, 0.0, 0.0], dtype=float)

    @property
    def tabletop_centerline_offset(self) -> float:
        return 0.008

    @property
    def recommended_attach_offset(self) -> np.ndarray:
        return np.array([0.0, 0.0, -0.035], dtype=float)

    @property
    def point_reference_kind(self) -> str:
        return "body"

    @property
    def point_reference_names(self) -> list[str]:
        prefix = self._spec.body_prefix
        n = self._spec.num_segments
        return [f"{prefix}_B_first"] + \
               [f"{prefix}_B_{i}" for i in range(1, n - 1)] + \
               [f"{prefix}_B_last"]

    @property
    def graspable_body_names(self) -> list[str]:
        return self.point_reference_names

    @property
    def graspable_point_count(self) -> int:
        return self._spec.num_segments

    def body_name_for_point_idx(self, idx: int) -> str:
        names = self.point_reference_names
        idx = int(idx)
        if idx <= 0:
            return names[0]
        if idx >= len(names) - 1:
            return names[-1]
        return names[idx]


def dynamic_cable_object_factory(spec: CableSpec, name: str = "cable") -> CableModelMixin:
    """根据 CableSpec 创建动态线缆对象。"""
    if spec.cable_type == "flex":
        return DynamicFlexCableObject(spec)
    if spec.cable_type == "rmb":
        return DynamicRMBCableObject(spec)
    if spec.cable_type == "composite":
        return DynamicCompositeCableObject(spec)
    raise ValueError(f"Unsupported cable_type: {spec.cable_type}")
