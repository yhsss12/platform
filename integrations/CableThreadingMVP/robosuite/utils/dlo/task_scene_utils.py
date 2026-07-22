"""
task_scene_utils — 任务场景辅助物体工厂函数集。

提供创建目标可视化标记、weld 约束、mocap body、柱子、钩子等常用 MuJoCo 场景元素的
工厂函数，供各任务在 _load_model 中组合使用。

设计原则：
  - 每个函数返回 MuJoCo XML 元素（ET.Element 或 new_body/new_site 结果）
  - 调用方负责将返回元素 append 到 arena.worldbody 或 model.equality
  - 所有参数均有合理默认值，仅必需参数为位置参数
"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.utils.mjcf_utils import new_body, new_geom, new_site


# ---------------------------------------------------------------------------
# 目标可视化标记
# ---------------------------------------------------------------------------

def add_target_sites(worldbody, keypoints, name_prefix, size=(0.006,), rgba=(0.1, 0.8, 0.2, 0.75)):
    """在 worldbody 中为每个目标关键点添加可视化 site。

    Args:
        worldbody: MuJoCo worldbody XML 元素（或任何支持 append 的容器）
        keypoints: 目标关键点坐标序列，每个为 (x, y, z) 或 np.ndarray
        name_prefix: site 名称前缀，最终名称为 "{name_prefix}_{idx:02d}"
        size: site 尺寸，默认 (0.006,) 即半径 6mm 的球
        rgba: site 颜色，默认半透明绿色
    """
    for idx, pos in enumerate(keypoints):
        worldbody.append(
            new_site(
                name=f"{name_prefix}_{idx:02d}",
                pos=pos,
                size=size,
                rgba=rgba,
            )
        )


# ---------------------------------------------------------------------------
# Weld 约束
# ---------------------------------------------------------------------------

def create_weld_constraint(name, body1, body2, relpose="0 0 0 1 0 0 0",
                           solref="0.02 1", solimp="0.95 0.99 0.001"):
    """创建 MuJoCo weld 约束 XML 元素。

    Args:
        name: 约束名称
        body1: 约束主体 1
        body2: 约束主体 2
        relpose: 相对位姿（四元数 + 位移），默认单位位姿
        solref: 求解器参考参数
        solimp: 求解器阻抗参数

    Returns:
        ET.Element: weld 约束元素，需 append 到 model.equality
    """
    return ET.Element("weld", attrib={
        "name": name,
        "body1": body1,
        "body2": body2,
        "relpose": relpose,
        "solref": solref,
        "solimp": solimp,
    })


# ---------------------------------------------------------------------------
# Mocap body
# ---------------------------------------------------------------------------

def create_mocap_body(name, pos):
    """创建 MuJoCo mocap body XML 元素。

    Args:
        name: body 名称
        pos: 位置 (x, y, z)

    Returns:
        ET.Element: mocap body 元素，需 append 到 arena.worldbody
    """
    return new_body(name=name, mocap="true", pos=pos)


# ---------------------------------------------------------------------------
# 柱子对（穿杆任务）
# ---------------------------------------------------------------------------

def create_pole_pair(name, pos, pole_radius, pole_height, pole_spacing,
                     rgba=(0.75, 0.1, 0.1, 1.0), collision_group=1):
    """创建包含两根圆柱体和两个 site 的柱子 body。

    两根柱子沿 x 轴排列，第一根在 (0,0,half_h)，第二根在 (spacing,0,half_h)。
    每根柱子附带一个透明 site 用于位置查询。

    Args:
        name: body 名称
        pos: body 世界坐标位置
        pole_radius: 圆柱体半径
        pole_height: 圆柱体高度
        pole_spacing: 两根柱子之间的间距
        rgba: 柱子颜色
        collision_group: 碰撞组（0=启用碰撞，1=仅可视化）

    Returns:
        new_body: 柱子 body 元素，需 append 到 arena.worldbody
    """
    body = new_body(name=name, pos=pos)
    half_h = pole_height / 2.0
    for i, x_offset in enumerate([0.0, pole_spacing]):
        body.append(
            new_geom(
                name=f"pole{i+1}",
                type="cylinder",
                size=(pole_radius, half_h),
                pos=(x_offset, 0.0, half_h),
                group=collision_group,
                rgba=rgba,
            )
        )
        body.append(
            new_site(
                name=f"pole{i+1}_site",
                pos=(x_offset, 0.0, half_h),
                size=(0.005,),
                rgba=(0, 0, 0, 0),
            )
        )
    return body


# ---------------------------------------------------------------------------
# 钩子（链条挂载任务）
# ---------------------------------------------------------------------------

def create_hook_body(name, pos, post_height=0.07, post_radius=0.01,
                     bar_radius=0.008, bar_length=0.1,
                     friction=(2.0, 0.005, 0.0001),
                     rgba=(0.78, 0.12, 0.45, 1.0)):
    """创建钩子 body（立柱 + 横杆 + 标记点）。

    钩子由三部分组成：
      - 立柱（圆柱体）：从 pos 向下延伸 post_height，group=1（不参与碰撞）
      - 横杆（胶囊体）：沿 y 轴方向，group=0（启用碰撞），高摩擦
      - 标记点（site）：位于钩子中心，用于调试和可视化

    Args:
        name: body 名称
        pos: 钩子世界坐标位置
        post_height: 立柱高度
        post_radius: 立柱半径
        bar_radius: 横杆半径
        bar_length: 横杆总长度
        friction: 横杆摩擦参数 (sliding, torsional, rolling)
        rgba: 钩子颜色

    Returns:
        new_body: 钩子 body 元素，需 append 到 model.worldbody
    """
    body = new_body(name=name, pos=pos)
    half_bar = bar_length / 2.0

    # 立柱
    body.append(
        new_geom(
            name=f"{name}_post",
            type="cylinder",
            size=(post_radius, post_height),
            pos=(0.0, 0.0, -post_height),
            group=1,
            rgba=rgba,
        )
    )

    # 横杆
    body.append(
        new_geom(
            name=f"{name}_bar",
            type="capsule",
            size=(bar_radius, half_bar),
            fromto=(0.0, -half_bar, 0.0, 0.0, half_bar, 0.0),
            group=0,
            rgba=rgba,
            friction=friction,
        )
    )

    # 标记点
    body.append(
        new_site(
            name=f"{name}_site",
            pos=(0.0, 0.0, 0.0),
            size=(0.008,),
            rgba=(0.1, 0.4, 1.0, 0.9),
        )
    )

    return body
