"""
采集端机器人状态桥接：在既有 rclpy 节点上通过订阅获取关节 / 末端力矩数据，
替代周期性 `ros2 topic echo` 子进程。

设计原则：
- 与 ros2_camera_stream.CameraStreamManager 共用同一 Node + spin 线程（DDS 原生路径）。
- 周期性 reconcile 订阅列表（基于 node.get_topic_names_and_types），发现即订阅、无则销毁。
- 无法导入 rclpy / 无节点时由 agent_main 回退到 echo 方案。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from rosidl_runtime_py.utilities import get_message

    RCLPY_STATE_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    logger.warning("ros2_robot_state_bridge: ROS2 依赖不可用，将回退 topic echo: %s", _e)
    RCLPY_STATE_AVAILABLE = False
    Node = object  # type: ignore


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if x != x:
            return None
        return x
    except Exception:
        return None


class RobotStateBridge:
    """
    挂在已有 rclpy Node 上；由 create_timer 驱动订阅表刷新。
    """

    def __init__(self, node: Node) -> None:
        if not RCLPY_STATE_AVAILABLE:
            raise RuntimeError("rclpy not available")
        self.node = node
        self._lock = threading.Lock()
        self._timer = None
        self._started = False

        self._joint_cache: Dict[str, Dict[str, Any]] = {}
        self._ft_cache: Dict[str, Dict[str, Any]] = {}
        self._last_all_topics: List[str] = []

        self._joint_subs: Dict[str, Any] = {}
        self._joint_msg_class_cache: Dict[str, Any] = {}
        self._ft_subs: Dict[str, Any] = {}

    def start(self) -> None:
        if self._started:
            return
        period = float(os.environ.get("EAI_AGENT_ROS_STATE_RECONCILE_SEC", "2.0") or "2.0")
        period = max(0.5, period)
        try:
            self._timer = self.node.create_timer(period, self._reconcile_subscriptions)
            self._started = True
            self._reconcile_subscriptions()
            logger.info("RobotStateBridge started (reconcile every %.1fs)", period)
        except Exception as e:
            logger.error("RobotStateBridge start failed: %s", e)
            self._started = False

    def is_active(self) -> bool:
        return self._started

    def get_all_topic_names(self) -> List[str]:
        with self._lock:
            return list(self._last_all_topics)

    # --- callbacks ---

    def _on_joint_state(self, topic: str, msg: JointState) -> None:
        try:
            names = [str(x) for x in (msg.name or [])]
            pos = [_safe_float(x) for x in (msg.position or [])]
            vel = [_safe_float(x) for x in (msg.velocity or [])]
            eff = [_safe_float(x) for x in (msg.effort or [])]
            count = max(len(names), len(pos), len(vel), len(eff))
            joints: List[Dict[str, Any]] = []
            for idx in range(count):
                n = str(names[idx]) if idx < len(names) and names[idx] else f"joint{idx + 1}"
                p = pos[idx] if idx < len(pos) else None
                vv = vel[idx] if idx < len(vel) else None
                e = eff[idx] if idx < len(eff) else None
                item: Dict[str, Any] = {"name": n, "position": p, "velocity": vv, "temperature": None}
                if e is not None:
                    item["effort"] = e
                joints.append(item)
            payload = {
                "joints": joints,
                "joint_positions": pos,
                "joint_velocities": vel,
                "joint_efforts": eff,
                "joint_temperatures": [],
            }
            with self._lock:
                self._joint_cache[topic] = payload
        except Exception as e:
            logger.debug("joint cb error %s: %s", topic, e)

    def _as_list(self, v: Any) -> List[Any]:
        """Best-effort: ROS2 message array fields should be list-like."""
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, tuple):
            return list(v)
        if isinstance(v, (str, bytes, int, float, bool)):
            return [v]
        try:
            return list(v)
        except Exception:
            return []

    def _on_joint_msg(self, topic: str, msg: Any) -> None:
        """
        通用关节消息解析：兼容
        - sensor_msgs/msg/JointState（name/position/velocity/effort）
        - 自定义消息（可能用 torque/torques 代替 effort）
        """
        try:
            names = []
            for attr in ("name", "names", "joint_names"):
                if hasattr(msg, attr):
                    names = self._as_list(getattr(msg, attr))
                    break

            pos_raw = []
            for attr in ("position", "positions", "pos", "q"):
                if hasattr(msg, attr):
                    pos_raw = self._as_list(getattr(msg, attr))
                    break

            vel_raw = []
            for attr in ("velocity", "velocities", "vel", "dq"):
                if hasattr(msg, attr):
                    vel_raw = self._as_list(getattr(msg, attr))
                    break

            eff_raw = []
            for attr in ("effort", "efforts", "torque", "torques"):
                if hasattr(msg, attr):
                    eff_raw = self._as_list(getattr(msg, attr))
                    break

            temp_raw = []
            for attr in ("temperature", "temperatures", "temp"):
                if hasattr(msg, attr):
                    temp_raw = self._as_list(getattr(msg, attr))
                    break

            pos = [_safe_float(x) for x in pos_raw]
            vel = [_safe_float(x) for x in vel_raw]
            eff = [_safe_float(x) for x in eff_raw]
            temps = [_safe_float(x) for x in temp_raw]

            count = max(len(names), len(pos), len(vel), len(eff), len(temps))
            if count <= 0:
                return

            joints: List[Dict[str, Any]] = []
            for idx in range(count):
                n = str(names[idx]) if idx < len(names) and names[idx] else f"joint{idx + 1}"
                p = pos[idx] if idx < len(pos) else None
                vv = vel[idx] if idx < len(vel) else None
                e = eff[idx] if idx < len(eff) else None
                t = temps[idx] if idx < len(temps) else None

                item: Dict[str, Any] = {"name": n, "position": p, "velocity": vv, "temperature": t}
                if e is not None:
                    item["effort"] = e
                joints.append(item)

            payload = {
                "joints": joints,
                "joint_positions": pos,
                "joint_velocities": vel,
                "joint_efforts": eff,
                "joint_temperatures": temps,
            }
            with self._lock:
                self._joint_cache[topic] = payload
        except Exception as e:
            logger.debug("joint cb error %s: %s", topic, e)

    def _parse_ft_vectors(self, msg: Any) -> Optional[Tuple[List[Optional[float]], List[Optional[float]]]]:
        # RealMan 扁平字段
        if hasattr(msg, "force_fx") and hasattr(msg, "force_fy") and hasattr(msg, "force_fz"):
            force = [
                _safe_float(getattr(msg, "force_fx", None)),
                _safe_float(getattr(msg, "force_fy", None)),
                _safe_float(getattr(msg, "force_fz", None)),
            ]
            torque = [
                _safe_float(getattr(msg, "force_mx", None)),
                _safe_float(getattr(msg, "force_my", None)),
                _safe_float(getattr(msg, "force_mz", None)),
            ]
            if any(v is not None for v in force + torque):
                return force, torque

        # geometry_msgs/Wrench
        if hasattr(msg, "force") and hasattr(msg, "torque"):
            try:
                f = msg.force
                t = msg.torque
                return (
                    [_safe_float(f.x), _safe_float(f.y), _safe_float(f.z)],
                    [_safe_float(t.x), _safe_float(t.y), _safe_float(t.z)],
                )
            except Exception:
                pass

        # geometry_msgs/WrenchStamped
        if hasattr(msg, "wrench"):
            try:
                w = msg.wrench
                return (
                    [_safe_float(w.force.x), _safe_float(w.force.y), _safe_float(w.force.z)],
                    [_safe_float(w.torque.x), _safe_float(w.torque.y), _safe_float(w.torque.z)],
                )
            except Exception:
                pass

        return None

    def _on_ft(self, topic: str, msg: Any) -> None:
        try:
            parsed = self._parse_ft_vectors(msg)
            if parsed is None:
                return
            force, torque = parsed
            if not any(v is not None for v in force + torque):
                return
            with self._lock:
                self._ft_cache[topic] = {"force": force, "torque": torque}
        except Exception as e:
            logger.debug("ft cb error %s: %s", topic, e)

    # --- subscription management ---

    def _qos_sensor(self) -> QoSProfile:
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

    def _ft_name_match(self, topic: str) -> bool:
        tl = topic.lower()
        return any(
            k in tl
            for k in (
                "wrench",
                "force_torque",
                "force-torque",
                "_ft",
                "/ft",
                "get_force",
                "force_data",
                "force_result",
                "six_axis",
                "sixaxis",
                "ft_sensor",
                "ftsensor",
            )
        )

    def _joint_priority(self, topic: str) -> Tuple[int, str]:
        t = topic.lower()
        if t.endswith("/joint_states"):
            if "/left" in t or "left_" in t:
                return (0, t)
            if "/right" in t or "right_" in t:
                return (1, t)
            return (2, t)
        if "joint_states" in t:
            return (3, t)
        return (4, t)

    def _ft_priority(self, topic: str) -> Tuple[int, str]:
        tl = topic.lower()
        if "wrench" in tl:
            return (0, tl)
        if "get_force_data" in tl or "force_data_result" in tl:
            return (1, tl)
        if "force_torque" in tl or "force-torque" in tl:
            return (2, tl)
        if "/ft" in tl or "_ft" in tl:
            return (3, tl)
        if "get_force" in tl or "force_data" in tl:
            return (4, tl)
        return (5, tl)

    def _reconcile_subscriptions(self) -> None:
        if not RCLPY_STATE_AVAILABLE or self.node is None:
            return
        try:
            names_and_types = self.node.get_topic_names_and_types()
        except Exception as e:
            logger.debug("get_topic_names_and_types failed: %s", e)
            return

        name_to_type: Dict[str, str] = {}
        all_names: List[str] = []
        for name, types in names_and_types:
            all_names.append(name)
            if types:
                name_to_type[name] = types[0]
        all_names = sorted(set(all_names))
        with self._lock:
            self._last_all_topics = all_names

        skip_type_prefix = (
            "sensor_msgs/msg/Image",
            "sensor_msgs/msg/CompressedImage",
        )

        # --- joint: flexible (JointState / custom) ---
        joint_candidates: List[Tuple[str, str]] = []
        for n, t in name_to_type.items():
            if "joint" not in n.lower():
                continue
            if not t:
                continue
            if any(t.startswith(p) for p in skip_type_prefix):
                continue
            joint_candidates.append((n, t))
        max_j = int(_safe_float(os.environ.get("EAI_AGENT_JOINT_TOPIC_SCAN_MAX")) or 12)
        max_j = max(1, min(max_j, 80))
        joint_sorted = sorted(joint_candidates, key=lambda x: self._joint_priority(x[0]))[:max_j]
        joint_topic_keys = [x[0] for x in joint_sorted]

        # --- ft ---
        ft_candidates: List[Tuple[str, str]] = []
        for n, t in name_to_type.items():
            if not self._ft_name_match(n):
                continue
            if any(t.startswith(p) for p in skip_type_prefix):
                continue
            if t == "sensor_msgs/msg/JointState":
                continue
            ft_candidates.append((n, t))
        max_f = int(_safe_float(os.environ.get("EAI_AGENT_FT_TOPIC_SCAN_MAX")) or 8)
        max_f = max(1, min(max_f, 60))
        ft_sorted = sorted(ft_candidates, key=lambda x: self._ft_priority(x[0]))[:max_f]

        qos = self._qos_sensor()

        # destroy removed joint subs
        for t in list(self._joint_subs.keys()):
            if t not in joint_topic_keys:
                try:
                    self.node.destroy_subscription(self._joint_subs.pop(t))
                except Exception:
                    self._joint_subs.pop(t, None)

        for topic, type_str in joint_sorted:
            if topic in self._joint_subs:
                continue
            try:
                MsgClass = self._joint_msg_class_cache.get(type_str)
                if MsgClass is None:
                    MsgClass = get_message(type_str)
                    self._joint_msg_class_cache[type_str] = MsgClass

                sub = self.node.create_subscription(
                    MsgClass,
                    topic,
                    lambda m, tp=topic: self._on_joint_msg(tp, m),
                    qos,
                )
                self._joint_subs[topic] = sub
                logger.info("RobotStateBridge subscribed joint: %s [%s]", topic, type_str)
            except Exception as e:
                logger.warning("subscribe joint %s [%s] failed: %s", topic, type_str, e)

        # ft subs
        ft_key_set = {x[0] for x in ft_sorted}
        for t in list(self._ft_subs.keys()):
            if t not in ft_key_set:
                try:
                    self.node.destroy_subscription(self._ft_subs.pop(t))
                except Exception:
                    self._ft_subs.pop(t, None)

        for topic, type_str in ft_sorted:
            if topic in self._ft_subs:
                continue
            try:
                MsgClass = get_message(type_str)
            except Exception as e:
                logger.debug("get_message %s for %s: %s", type_str, topic, e)
                continue
            try:
                sub = self.node.create_subscription(
                    MsgClass,
                    topic,
                    lambda m, tp=topic: self._on_ft(tp, m),
                    qos,
                )
                self._ft_subs[topic] = sub
                logger.info("RobotStateBridge subscribed FT %s [%s]", topic, type_str)
            except Exception as e:
                logger.warning("subscribe FT %s failed: %s", topic, e)

    # --- export for agent_main / 与 echo 路径 payload 对齐 ---

    def export_joint_payload(self) -> Dict[str, Any]:
        prefer = (os.environ.get("EAI_AGENT_JOINT_STATE_TOPIC") or "").strip()
        with self._lock:
            states = {k: v for k, v in self._joint_cache.items() if v}
            joint_topics = sorted(self._joint_subs.keys())

        out: Dict[str, Any] = {"joint_topics": joint_topics}
        if states:
            out["joint_states_by_topic"] = states
            active = prefer if prefer in states else sorted(states.keys(), key=self._joint_priority)[0]
            out["joint_active_topic"] = active
        return out

    def export_ft_payload(self) -> Dict[str, Any]:
        prefer = (os.environ.get("EAI_AGENT_FT_STATE_TOPIC") or "").strip()
        with self._lock:
            states = {k: v for k, v in self._ft_cache.items() if v}
            ft_sub_keys = list(self._ft_subs.keys())
        ft_topics = sorted(set(ft_sub_keys) | set(states.keys()))
        out: Dict[str, Any] = {"ft_topics": ft_topics}
        if states:
            out["ft_states_by_topic"] = states
            active = prefer if prefer in states else sorted(states.keys(), key=self._ft_priority)[0]
            out["ft_active_topic"] = active
            ap = states.get(active) or {}
            if isinstance(ap, dict):
                out["ft_force"] = ap.get("force")
                out["ft_torque"] = ap.get("torque")
        return out
