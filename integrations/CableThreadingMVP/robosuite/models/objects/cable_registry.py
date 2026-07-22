"""CableModelRegistry -- 集中管理所有线缆模型的注册与创建。

使用方式:
    from robosuite.models.objects import get_registry

    registry = get_registry()
    cable = registry.create("composite_cable", name="cable")

    # 注册自定义参数变体
    from robosuite.models.objects import CableSpec
    registry.register_spec("rmb_stiff", CableSpec(cable_type="rmb", damping=0.01))

    # 列出所有模型
    print(registry.list_models())
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _RegistryEntry:
    """一条注册记录：指向线缆类或 CableSpec 工厂。"""
    canonical_name: str
    cls: type | None               # 静态类 (CableObject, RMBCableObject, …)
    spec: Any | None               # CableSpec 实例 (动态线缆)
    kwargs_factory: Callable | None  # kwargs_factory(instance_name) -> dict
    metadata: dict                  # 预提取的物理参数 (radius, length, …)


# ---------------------------------------------------------------------------
# CableModelRegistry
# ---------------------------------------------------------------------------

class CableModelRegistry:
    """集中式线缆模型注册表。

    - register(): 注册静态线缆类
    - register_spec(): 注册动态线缆 (CableSpec)
    - create(): 按名称创建线缆实例
    - list_models(): 列出所有已注册模型
    - get_metadata(): 获取模型元数据 (无需实例化)
    """

    def __init__(self):
        self._entries: dict[str, _RegistryEntry] = {}
        self._aliases: dict[str, str] = {}  # alias -> canonical_name
        self._register_builtins()

    # ------------------------------------------------------------------
    # 注册 API
    # ------------------------------------------------------------------

    def register(self, name: str, cls, aliases: tuple[str, ...] = (),
                 kwargs_factory: Callable | None = None):
        """注册一个静态线缆类。

        Args:
            name: 规范名称 (如 "rmb", "composite_cable")
            cls: 线缆类 (如 RMBCableObject)
            aliases: 别名元组 (如 ("rmb_chain", "robomanip_baselines"))
            kwargs_factory: 可选的构造参数工厂 kwargs_factory(instance_name) -> dict
        """
        metadata = self._extract_metadata_from_class(cls)
        entry = _RegistryEntry(
            canonical_name=name, cls=cls, spec=None,
            kwargs_factory=kwargs_factory, metadata=metadata,
        )
        self._entries[name] = entry
        for alias in aliases:
            self._aliases[alias] = name

    def register_spec(self, name: str, spec, aliases: tuple[str, ...] = ()):
        """注册一个动态线缆模型 (从 CableSpec 生成)。

        Args:
            name: 规范名称 (如 "rmb_stiff", "flex_long")
            spec: CableSpec 实例
            aliases: 别名元组
        """
        from .dynamic_cable import (
            DynamicFlexCableObject, DynamicRMBCableObject,
            DynamicCompositeCableObject, CableSpec,
        )
        if not isinstance(spec, CableSpec):
            raise TypeError(f"Expected CableSpec, got {type(spec)}")

        # 根据 cable_type 选择动态类
        _dynamic_cls_map = {
            "flex": DynamicFlexCableObject,
            "rmb": DynamicRMBCableObject,
            "composite": DynamicCompositeCableObject,
        }
        dyn_cls = _dynamic_cls_map.get(spec.cable_type)
        if dyn_cls is None:
            raise ValueError(f"Unknown cable_type: {spec.cable_type}")

        metadata = {
            "cable_radius": spec.cable_radius,
            "cable_length": spec.cable_length,
            "point_reference_kind": "flex" if spec.cable_type == "flex" else "body",
            "num_segments": spec.num_segments,
        }
        entry = _RegistryEntry(
            canonical_name=name, cls=None, spec=spec,
            kwargs_factory=None, metadata=metadata,
        )
        self._entries[name] = entry
        for alias in aliases:
            self._aliases[alias] = name

    # ------------------------------------------------------------------
    # 创建 API
    # ------------------------------------------------------------------

    def create(self, name: str, instance_name: str = "cable"):
        """按名称或别名创建线缆实例。

        Args:
            name: 模型名称或别名 (不区分大小写)
            instance_name: 实例名 (传给构造函数的 name= 参数)

        Returns:
            CableModelMixin 实例
        """
        canonical = self._resolve(name)
        entry = self._entries.get(canonical)
        if entry is None:
            raise ValueError(
                f"Unknown cable model: '{name}'. "
                f"Available: {self.list_models()}"
            )

        # 动态线缆
        if entry.spec is not None:
            from .dynamic_cable import dynamic_cable_object_factory
            return dynamic_cable_object_factory(entry.spec, name=instance_name)

        # 静态线缆
        if entry.cls is not None:
            kwargs = {}
            if entry.kwargs_factory is not None:
                kwargs = entry.kwargs_factory(instance_name)
            else:
                kwargs = {"name": instance_name}
            return entry.cls(**kwargs)

        raise RuntimeError(f"Registry entry '{canonical}' has neither cls nor spec")

    # ------------------------------------------------------------------
    # 查询 API
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """列出所有已注册的规范名称。"""
        return list(self._entries.keys())

    def list_aliases(self) -> dict[str, str]:
        """返回 {别名: 规范名称} 映射。"""
        return dict(self._aliases)

    def get_metadata(self, name: str) -> dict:
        """获取模型元数据 (无需完整实例化)。

        Returns:
            dict with keys: cable_radius, cable_length, point_reference_kind, etc.
        """
        canonical = self._resolve(name)
        entry = self._entries.get(canonical)
        if entry is None:
            raise ValueError(f"Unknown cable model: '{name}'")
        return dict(entry.metadata)

    def has(self, name: str) -> bool:
        """检查模型是否已注册。"""
        try:
            self._resolve(name)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> str:
        """将名称或别名解析为规范名称。"""
        key = name.lower().strip()
        if key in self._entries:
            return key
        if key in self._aliases:
            return self._aliases[key]
        raise ValueError(
            f"Unknown cable model: '{name}'. "
            f"Available: {self.list_models()}"
        )

    @staticmethod
    def _extract_metadata_from_class(cls) -> dict:
        """从线缆类中提取元数据 (尝试无参实例化)。"""
        try:
            probe = cls(name="__probe__")
            return {
                "cable_radius": float(probe.cable_radius),
                "cable_length": float(getattr(probe, "cable_length", 0.0)),
                "point_reference_kind": str(probe.point_reference_kind),
            }
        except Exception:
            return {}

    def _register_builtins(self):
        """注册所有内置线缆模型。"""
        from .xml_objects import (
            CableObject, CompositeCableObject, CompositeImproveObject, CompositeSoftObject, CompositeThinObject,
            RMBCableObject, FlexCableObject, FlexImproveObject,
        )

        # --- 静态线缆 ---
        self.register("segmented", CableObject,
                       aliases=("capsule_chain",))

        self.register("composite_cable", CompositeCableObject,
                       aliases=("composite", "mujoco_composite",
                                "deformable_ravens_composite", "mujoco_cable",
                                "flex_reference_composite", "flex_reference_mujoco_cable"))

        self.register("composite_improve", CompositeImproveObject,
                       aliases=("composite_improved",))

        self.register("composite_soft", CompositeSoftObject,
                       aliases=("composite_softened",))

        self.register("composite_thin", CompositeThinObject,
                       aliases=("composite_thinned",))

        self.register("rmb", RMBCableObject,
                       aliases=("rmb_chain", "robomanip_baselines"))

        self.register("flex", FlexCableObject,
                       aliases=("flex_cable", "flexcomp"))

        self.register("flex_improve", FlexImproveObject,
                       aliases=("flex_improved",))


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_registry: CableModelRegistry | None = None
_lock = threading.Lock()


def get_registry() -> CableModelRegistry:
    """获取全局 CableModelRegistry 单例。"""
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = CableModelRegistry()
    return _registry


def reset_registry():
    """重置全局注册表 (用于测试)。"""
    global _registry
    with _lock:
        _registry = None
