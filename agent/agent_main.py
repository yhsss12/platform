from __future__ import annotations

"""
采集端 Agent 示例服务。

部署位置建议：运行在实际连接 ROS2 / 摄像头 / 采集脚本的边缘机上。
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import signal
import time
import tempfile
import re
import shlex
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import json
import socket
import uuid as uuid_lib
import struct
import itertools
from urllib.parse import urlencode

import aiohttp
import httpx
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame
from minio import Minio
from minio.error import S3Error


try:
    # 支持 config.py 位于当前包内
    from . import config as _cfg
except ImportError:
    # 允许用户直接复制 config.example.py 为 config.py
    import config as _cfg  # type: ignore[no-redef]

SERVER_BASE = _cfg.SERVER_BASE
AGENT_ID_CONFIG = _cfg.AGENT_ID
AGENT_NAME = _cfg.AGENT_NAME
DEVICES = getattr(_cfg, "DEVICES", [])
AGENT_HOST = _cfg.AGENT_HOST
AGENT_PORT = _cfg.AGENT_PORT

def _default_agent_data_root() -> str:
    v = os.environ.get("EAI_AGENT_DATA_ROOT", "").strip()
    if v:
        return v
    home = os.path.expanduser("~")
    try:
        if os.geteuid() == 0 and os.path.realpath(home) == "/root":
            cand = "/home/ubuntu"
            if os.path.isdir(cand):
                return cand
            base = "/home"
            if os.path.isdir(base):
                for name in sorted(os.listdir(base)):
                    p = os.path.join(base, name)
                    if os.path.isdir(p):
                        return p
    except Exception:
        pass
    return home


AGENT_DATA_ROOT = _default_agent_data_root()

# 隧道共享密钥（与平台 AGENT_TUNNEL_TOKEN / EAI_AGENT_TUNNEL_TOKEN 一致；未设置则不校验）
AGENT_TUNNEL_TOKEN: str = (
    os.environ.get("EAI_AGENT_TUNNEL_TOKEN", "").strip()
    or str(getattr(_cfg, "AGENT_TUNNEL_TOKEN", "") or "").strip()
)

_AGENT_PROCESS_START_TS = time.time()
_HEARTBEAT_INTERVAL_SEC = float(os.environ.get("EAI_AGENT_HEARTBEAT_INTERVAL_SEC", "15") or "15")
# 出站队列积压超过该阈值时跳过本帧 MJPEG，缓解 HoL（文档 §4.1 / §7.6C）
_TUNNEL_OUTQ_BACKPRESS_CHUNKS = int(os.environ.get("EAI_AGENT_TUNNEL_OUTQ_BACKPRESS", "120") or "120")


AGENT_VERSION = "0.1.29"
app = FastAPI(title="EAI Collector Agent", version=AGENT_VERSION)

logger = logging.getLogger(__name__)


def _collect_execution_command_id(
    *,
    run_id: Optional[str],
    task_id: Optional[str],
    job_id: Optional[str],
) -> str:
    marker = (run_id or job_id or task_id or "default").strip() or "default"
    return f"collect-execution:{marker}"


def _experiment_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if AGENT_TUNNEL_TOKEN:
        headers["X-Agent-Tunnel-Token"] = AGENT_TUNNEL_TOKEN
    return headers


async def _post_experiment_event(
    client: httpx.AsyncClient,
    event: str,
    *,
    run_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
    task_id: Optional[str] = None,
    job_id: Optional[str] = None,
    command_id: Optional[str] = None,
    success: Optional[bool] = None,
    path: Optional[str] = None,
    **extra: Any,
) -> None:
    body: Dict[str, Any] = {
        "role": "agent",
        "event": event,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "task_id": task_id,
        "job_id": job_id,
        "agent_id": AGENT_ID,
        "command_id": command_id,
        "success": success,
        "path": path,
        "ts_ms": int(time.time() * 1000),
    }
    for key, value in extra.items():
        if value is not None:
            body[key] = value
    try:
        await client.post(
            f"{SERVER_BASE}/api/experiment/event",
            json=body,
            headers=_experiment_headers(),
        )
    except Exception as exc:  # pragma: no cover
        print(f"[agent] failed to post experiment event {event}: {exc}")


def _normalize_minio_endpoint(endpoint: str) -> str:
    """MinIO SDK 需要 host:port，不接受带 http(s):// 前缀。"""
    s = (endpoint or "").strip()
    if not s:
        return ""
    s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
    return s.split("/")[0].strip()


def _safe_float(v: Any) -> Optional[float]:
    try:
        n = float(v)
        if n != n:  # NaN
            return None
        return n
    except Exception:
        return None


def _run_cmd_text_sync(cmd: list[str], *, timeout_sec: float) -> str:
    """运行命令并返回文本输出；失败返回空字符串。"""
    try:
        out = subprocess.check_output(
            cmd,
            timeout=timeout_sec,
            stderr=subprocess.STDOUT,
        ).decode(errors="replace")
        return out
    except Exception:
        return ""


def _parse_ros2_topic_echo_block(text: str, key: str) -> list[Any]:
    """
    从 `ros2 topic echo --once` 输出中提取块列表，如：
    key:
    - v1
    - v2
    """
    lines = text.splitlines()
    out: list[Any] = []
    in_block = False
    key_prefix = f"{key}:"
    key_lower = key.lower()
    is_name_key = key_lower in ("name", "names", "joint_names")

    def _parse_inline_values(after_colon: str) -> list[Any]:
        s = after_colon.strip().rstrip(',')
        # 可选：支持 key: [a, b] 或 key: a
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1].strip()
        if is_name_key:
            if not s:
                return []
            tokens = [t.strip() for t in s.split(',') if t.strip()]
            return [t.strip("'").strip('"') for t in tokens]

        import re
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
        parsed: list[Any] = []
        for n in nums:
            v = _safe_float(n)
            if v is not None:
                parsed.append(v)
        return parsed

    for raw in lines:
        line = raw.rstrip()
        s = line.strip()
        if not in_block:
            # 支持：
            #   key:
            #   - v1
            #   - v2
            # 以及：
            #   key: [v1, v2]
            #   key: v1
            if s == key_prefix:
                in_block = True
                continue
            if s.startswith(key_prefix):
                rest = s[len(key_prefix):]
                return _parse_inline_values(rest)
            continue

        # 新字段开始，结束当前 block
        if s.endswith(":") and not s.startswith("- "):
            break
        if not s.startswith("- "):
            continue
        token = s[2:].strip().strip("'").strip('"').rstrip(',')
        if is_name_key:
            out.append(token)
        else:
            out.append(_safe_float(token))
    return out


def _parse_ros2_topic_echo_block_multi(text: str, keys: List[str], *, numeric: bool) -> list[Any]:
    for k in keys:
        vals = _parse_ros2_topic_echo_block(text, k)
        if vals:
            if numeric:
                return [_safe_float(x) for x in vals]
            return vals
    return []


def _collect_joint_state_from_ros_topic(topic: Optional[str] = None) -> Dict[str, Any]:
    """
    直接读取 ROS2 话题 joint_states，并转换为前端可消费格式。
    """
    topic = (topic or os.environ.get("EAI_AGENT_JOINT_STATE_TOPIC") or "/joint_states").strip() or "/joint_states"
    timeout_sec = _safe_float(os.environ.get("EAI_AGENT_JOINT_STATE_TIMEOUT_SEC")) or 2.0
    out = ""
    # 先默认 QoS；若拿不到，再用 sensor_data（常见于相机/传感器 best-effort 发布）
    for cmd in (
        ["ros2", "topic", "echo", "--once", topic],
        ["ros2", "topic", "echo", "--once", "--qos-profile", "sensor_data", topic],
    ):
        out = _run_cmd_text_sync(cmd, timeout_sec=float(timeout_sec))
        if out.strip():
            break
    if not out.strip():
        return {}

    names = _parse_ros2_topic_echo_block_multi(out, ["name", "names", "joint_names"], numeric=False)
    positions = _parse_ros2_topic_echo_block_multi(out, ["position", "positions", "pos", "q"], numeric=True)
    velocities = _parse_ros2_topic_echo_block_multi(out, ["velocity", "velocities", "vel", "dq"], numeric=True)
    efforts = _parse_ros2_topic_echo_block_multi(out, ["effort", "efforts", "torque", "torques"], numeric=True)
    temperatures = _parse_ros2_topic_echo_block_multi(out, ["temperature", "temperatures", "temp"], numeric=True)
    count = max(len(names), len(positions), len(velocities), len(efforts), len(temperatures))
    if count <= 0:
        return {}

    joints: List[Dict[str, Any]] = []
    for idx in range(count):
        n = str(names[idx]) if idx < len(names) and names[idx] else f"J{idx + 1}"
        p = positions[idx] if idx < len(positions) else None
        v = velocities[idx] if idx < len(velocities) else None
        e = efforts[idx] if idx < len(efforts) else None
        t = temperatures[idx] if idx < len(temperatures) else None
        item: Dict[str, Any] = {"name": n, "position": p, "velocity": v, "temperature": t}
        if e is not None:
            item["effort"] = e
        joints.append(item)

    return {
        "joints": joints,
        "joint_positions": positions,
        "joint_velocities": velocities,
        "joint_efforts": efforts,
        "joint_temperatures": temperatures,
    }


def _collect_joint_topics_payload(ros_topic_names: List[str]) -> Dict[str, Any]:
    """
    从已扫描的 ROS 话题中匹配所有包含 joint 的话题，批量采集关节状态。
    """
    joint_topics = sorted(set([t for t in ros_topic_names if "joint" in str(t).lower()]))
    out: Dict[str, Any] = {"joint_topics": joint_topics}
    if not joint_topics:
        return out

    max_topics = int(_safe_float(os.environ.get("EAI_AGENT_JOINT_TOPIC_SCAN_MAX")) or 12)
    max_topics = max(1, min(max_topics, 80))
    # 采样优先级：优先典型 joint_states 话题（含 left/right），其次其它包含 joint 的话题。
    def _priority(topic: str) -> tuple[int, str]:
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
    ordered_topics = sorted(joint_topics, key=_priority)
    states_by_topic: Dict[str, Any] = {}
    for topic in ordered_topics[:max_topics]:
        payload = _collect_joint_state_from_ros_topic(topic)
        if payload:
            states_by_topic[topic] = payload

    if states_by_topic:
        out["joint_states_by_topic"] = states_by_topic
        prefer_topic = (os.environ.get("EAI_AGENT_JOINT_STATE_TOPIC") or "").strip()
        active_topic = prefer_topic if prefer_topic in states_by_topic else next(iter(states_by_topic.keys()))
        out["joint_active_topic"] = active_topic
    return out


def _collect_joint_state_payload() -> Dict[str, Any]:
    """
    采集关节状态（轻量版）：
    - 回退方案：读取 EAI_AGENT_JOINT_STATE_JSON / EAI_AGENT_JOINT_STATE_FILE
    允许外部进程（如 ROS2 桥接脚本）写入后由 Agent 心跳转发。
    """
    raw = (os.environ.get("EAI_AGENT_JOINT_STATE_JSON") or "").strip()
    if not raw:
        p = (os.environ.get("EAI_AGENT_JOINT_STATE_FILE") or "").strip()
        if p:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    raw = (f.read() or "").strip()
            except Exception:
                raw = ""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}

    out: Dict[str, Any] = {}
    if isinstance(parsed, dict):
        joints = parsed.get("joints")
        if isinstance(joints, list):
            normalized: List[Dict[str, Any]] = []
            for idx, j in enumerate(joints):
                if not isinstance(j, dict):
                    continue
                position = _safe_float(j.get("position"))
                velocity = _safe_float(j.get("velocity"))
                temperature = _safe_float(j.get("temperature"))
                item: Dict[str, Any] = {
                    "name": str(j.get("name") or f"J{idx + 1}"),
                    "position": position,
                    "velocity": velocity,
                    "temperature": temperature,
                }
                if j.get("status") is not None:
                    item["status"] = str(j.get("status"))
                normalized.append(item)
            if normalized:
                out["joints"] = normalized

        # 兼容数组格式：joint_positions / joint_velocities / joint_temperatures
        for k in ("joint_positions", "joint_velocities", "joint_temperatures"):
            val = parsed.get(k)
            if isinstance(val, list):
                arr: List[Optional[float]] = [_safe_float(x) for x in val]
                out[k] = arr
    return out


def _parse_ros2_topic_echo_xyz_from_named_block(text: str, block_key: str) -> list[Optional[float]]:
    """
    从 `ros2 topic echo --once` 的缩进结构中，解析形如：
      force:
        x: ...
        y: ...
        z: ...
      torque:
        x: ...
        y: ...
        z: ...

    该 parser 只关注 x/y/z 标量键，且以 `block_key:` 的缩进作为作用域边界。
    """
    import re

    num_re = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    lines = text.splitlines()

    block_key = (block_key or "").strip().lower()
    if not block_key:
        return [None, None, None]

    # 找第一个匹配的 `block_key:`
    block_start_idx: Optional[int] = None
    block_indent: Optional[int] = None
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip():
            continue
        # 仅匹配 `block_key:` 独占一行；缩进用于确定作用域。
        if line.strip().lower() == f"{block_key}:":  # 仅匹配 `block_key:` 独占一行
            # count indentation
            indent = len(raw) - len(raw.lstrip(' '))
            block_start_idx = i
            block_indent = indent
            break

    if block_start_idx is None or block_indent is None:
        return [None, None, None]

    out_map: Dict[str, Optional[float]] = {"x": None, "y": None, "z": None}

    # 从 block_start_idx 下一行开始，遇到缩进回落到 block_indent 及以内就结束
    for j in range(block_start_idx + 1, len(lines)):
        raw = lines[j].rstrip()
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        if indent <= block_indent:
            break

        s = raw.strip()
        # 只解析 `x: 1.23` / `y: -0.4` / `z: 0`
        for axis in ("x", "y", "z"):
            if not s.lower().startswith(f"{axis}:"):
                continue
            # 抽取后面的第一个数字
            m = num_re.search(s)
            if not m:
                continue
            v = _safe_float(m.group(1))
            out_map[axis] = v
            break

    return [out_map["x"], out_map["y"], out_map["z"]]


def _parse_ros2_topic_echo_flat_force_fm_xyz(text: str) -> Optional[Tuple[List[Optional[float]], List[Optional[float]]]]:
    """
    RealMan 等驱动常见扁平格式（ros2 topic echo）：
      force_fx: ...
      force_fy: ...
      force_fz: ...
      force_mx: ...   # 力矩分量（名称仍为 force_m*）
      force_my: ...
      force_mz: ...
    返回 (force_xyz, torque_xyz)，与标准 wrench 的 force/torque 对齐。
    """
    import re

    pat = re.compile(
        r"^\s*(force_fx|force_fy|force_fz|force_mx|force_my|force_mz)\s*:\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
        re.IGNORECASE,
    )
    vals: Dict[str, float] = {}
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("---"):
            continue
        m = pat.match(s)
        if not m:
            continue
        key = m.group(1).lower()
        v = _safe_float(m.group(2))
        if v is not None:
            vals[key] = v

    force: List[Optional[float]] = [
        vals.get("force_fx"),
        vals.get("force_fy"),
        vals.get("force_fz"),
    ]
    torque: List[Optional[float]] = [
        vals.get("force_mx"),
        vals.get("force_my"),
        vals.get("force_mz"),
    ]
    if not any(v is not None for v in (force + torque)):
        return None
    return force, torque


def _collect_ft_state_from_ros_topic(topic: Optional[str] = None) -> Dict[str, Any]:
    """
    从 ROS2 话题解析端到端的“六维力/力矩”（wrench / force-torque）：
    - force: [Fx, Fy, Fz]
    - torque: [Tx, Ty, Tz]

    支持两种 echo 格式：
    1) geometry_msgs/Wrench 风格：force:/torque: 嵌套 x/y/z
    2) RealMan 等：force_fx..fz、force_mx..mz 扁平字段
    """
    topic = (topic or os.environ.get("EAI_AGENT_FT_STATE_TOPIC") or "").strip() or None
    if not topic:
        return {}

    timeout_sec = _safe_float(os.environ.get("EAI_AGENT_FT_STATE_TIMEOUT_SEC")) or 2.0
    out = ""
    for cmd in (
        ["ros2", "topic", "echo", "--once", topic],
        ["ros2", "topic", "echo", "--once", "--qos-profile", "sensor_data", topic],
    ):
        out = _run_cmd_text_sync(cmd, timeout_sec=float(timeout_sec))
        if out.strip():
            break
    if not out.strip():
        return {}

    force = _parse_ros2_topic_echo_xyz_from_named_block(out, "force")
    torque = _parse_ros2_topic_echo_xyz_from_named_block(out, "torque")
    has_any = any(v is not None for v in (force + torque))

    if not has_any:
        flat = _parse_ros2_topic_echo_flat_force_fm_xyz(out)
        if flat is not None:
            force, torque = flat
            has_any = True

    if not has_any:
        return {}

    return {
        "force": force,
        "torque": torque,
    }


def _collect_ft_topics_payload(ros_topic_names: List[str]) -> Dict[str, Any]:
    """
    从扫描得到的 ROS 话题中，匹配所有可能的末端力/力矩（wrench / ft）话题并采样。
    """
    ft_topics: List[str] = []
    for t in ros_topic_names:
        tl = str(t).lower()
        if any(
            k in tl
            for k in (
                "wrench",
                "force_torque",
                "force-torque",
                "_ft",
                "/ft",
                # RealMan / rm_driver 等：get_force_data_result、flat force_fx 消息
                "get_force",
                "force_data",
                "force_result",
                "six_axis",
                "sixaxis",
                "ft_sensor",
                "ftsensor",
            )
        ):
            ft_topics.append(str(t))

    ft_topics = sorted(set([x.strip() for x in ft_topics if x and str(x).strip()]))
    out: Dict[str, Any] = {"ft_topics": ft_topics}
    if not ft_topics:
        return out

    max_topics = int(_safe_float(os.environ.get("EAI_AGENT_FT_TOPIC_SCAN_MAX")) or 8)
    max_topics = max(1, min(max_topics, 60))

    # 简单优先级：偏向更短、更“标准”的命名
    def _priority(topic: str) -> tuple[int, int, str]:
        tl = topic.lower()
        score = 10
        if "wrench" in tl:
            score = 0
        elif "get_force_data" in tl or "force_data_result" in tl:
            score = 1
        elif "force_torque" in tl or "force-torque" in tl:
            score = 2
        elif "/ft" in tl or "_ft" in tl:
            score = 3
        elif "get_force" in tl or "force_data" in tl:
            score = 4
        return (score, len(topic), tl)

    ordered_topics = sorted(ft_topics, key=_priority)
    states_by_topic: Dict[str, Any] = {}
    for topic in ordered_topics[:max_topics]:
        payload = _collect_ft_state_from_ros_topic(topic)
        if payload:
            states_by_topic[topic] = payload

    if states_by_topic:
        out["ft_states_by_topic"] = states_by_topic
        prefer_topic = (os.environ.get("EAI_AGENT_FT_STATE_TOPIC") or "").strip()
        active_topic = prefer_topic if prefer_topic in states_by_topic else next(iter(states_by_topic.keys()))
        out["ft_active_topic"] = active_topic
        active_payload = states_by_topic.get(active_topic) or {}
        if isinstance(active_payload, dict):
            out["ft_force"] = active_payload.get("force")
            out["ft_torque"] = active_payload.get("torque")
    return out

# 采集端相机流（ROS2 订阅 -> MJPEG）
try:
    from ros2_camera_stream import stream_manager
except Exception:
    stream_manager = None


class AgentInfoResponse(BaseModel):
    """
    采集端自描述接口：用于中心平台探测/握手。

    返回字段：
    - uuid: 硬件唯一标识
    - status: ROS2 节点/硬件驱动运行状态（含简要计数与采样）
    - capabilities: 扫描得到的视频流/图像源（/dev/video* 与 ROS2 Image Topic）
    - version: Agent 软件版本号
    """

    uuid: str
    hostname: str
    status: Dict[str, Any]
    capabilities: Dict[str, Any]
    version: str


def _get_hardware_uuid_sync() -> str:
    """尽可能获取稳定的硬件级 UUID（优先级：环境变量 > machine-id > dmi product_uuid）。"""
    override = os.environ.get("EAI_AGENT_UUID") or os.environ.get("HOST_UUID")
    if override and override.strip():
        return override.strip()

    for path in ("/etc/machine-id", "/sys/class/dmi/id/product_uuid"):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    v = (f.read() or "").strip()
                if v:
                    return v
        except Exception:
            pass

    # fallback：使用主机名生成稳定 UUID（避免每次重启变化）
    host = socket.gethostname() or "unknown-host"
    return str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, host))


def _is_virtual_iface_name(iface: str) -> bool:
    iface = (iface or "").strip()
    if not iface:
        return True
    if iface == "lo":
        return True
    virtual_prefixes = (
        "docker",
        "br-",
        "veth",
        "virbr",
        "vmnet",
        "tun",
        "tap",
        "wg",
        "zt",
        "tailscale",
        "sit",
        "ip6tnl",
        "gre",
        "gretap",
        "erspan",
        "ifb",
        "dummy",
        "bond",
        "team",
        "macvlan",
        "ipvlan",
    )
    return iface.startswith(virtual_prefixes)


def _is_locally_administered_mac(mac: str) -> bool:
    try:
        first = int((mac or "").split(":")[0], 16)
        # bit1=1 means locally administered
        return bool(first & 0b00000010)
    except Exception:
        return False


def _read_iface_mac(iface: str) -> Optional[str]:
    p = os.path.join("/sys/class/net", iface, "address")
    try:
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            mac = (f.read() or "").strip().lower()
        if not mac or mac == "00:00:00:00:00:00":
            return None
        return mac
    except Exception:
        return None


def _get_primary_iface_sync() -> Optional[str]:
    """
    尽量取“主网卡”（默认路由对应网卡），避免误选 docker0/veth 导致 agent_id 不稳定。
    """
    try:
        out = subprocess.check_output(
            ["ip", "route", "get", "1.1.1.1"],
            stderr=subprocess.STDOUT,
            timeout=1.5,
        ).decode(errors="replace")
        parts = out.split()
        for i, tok in enumerate(parts):
            if tok == "dev" and i + 1 < len(parts):
                dev = parts[i + 1].strip()
                if dev:
                    return dev
    except Exception:
        return None
    return None


def _get_mac_address_sync() -> Optional[str]:
    """
    获取本机 MAC（最佳努力，偏稳定）。
    优先：默认路由网卡（若非虚拟网卡）；
    兜底：遍历非虚拟网卡，优先返回“全球唯一/GAA”MAC，其次返回任意非 00 MAC。
    """
    base = "/sys/class/net"
    if not os.path.isdir(base):
        return None

    primary = _get_primary_iface_sync()
    if primary and not _is_virtual_iface_name(primary):
        mac = _read_iface_mac(primary)
        if mac:
            return mac

    gaa_candidate: Optional[str] = None
    any_candidate: Optional[str] = None
    try:
        for iface in sorted(os.listdir(base)):
            if _is_virtual_iface_name(iface):
                continue
            mac = _read_iface_mac(iface)
            if not mac:
                continue
            any_candidate = any_candidate or mac
            if not _is_locally_administered_mac(mac):
                gaa_candidate = mac
                break
    except Exception:
        pass
    return gaa_candidate or any_candidate


_MAC_6_GROUPS = re.compile(
    r"^([0-9a-fA-F]{2})[:-]([0-9a-fA-F]{2})[:-]([0-9a-fA-F]{2})[:-]"
    r"([0-9a-fA-F]{2})[:-]([0-9a-fA-F]{2})[:-]([0-9a-fA-F]{2})$"
)


def _normalize_mac_like_agent_id(agent_id: str) -> str:
    """与平台 agent_tunnel_manager 一致：MAC 统一为小写冒号；非 MAC 原样返回。"""
    s = (agent_id or "").strip()
    if not s:
        return s
    if re.fullmatch(r"[0-9a-fA-F]{12}", s):
        h = s.lower()
        return ":".join(h[i : i + 2] for i in range(0, 12, 2))
    m = _MAC_6_GROUPS.match(s)
    if m:
        return ":".join(x.lower() for x in m.groups())
    return s


def _is_colon_mac_normalized(s: str) -> bool:
    return bool(re.fullmatch(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", (s or "").strip()))


# 平台侧 device.hardware_uuid 约定为 agent_id；默认用网卡 MAC，与隧道 / 同步 / 数据资产 meta 一致。
# 优先级：环境变量 EAI_AGENT_ID > config.AGENT_ID > 自动探测 MAC
AGENT_ID = (os.environ.get("EAI_AGENT_ID", "") or "").strip() or str(AGENT_ID_CONFIG or "").strip()
if not AGENT_ID:
    _mac = _get_mac_address_sync()
    AGENT_ID = str((_mac or "")).strip()
if not AGENT_ID:
    AGENT_ID = "unknown-agent"
else:
    _nid = _normalize_mac_like_agent_id(AGENT_ID)
    if _is_colon_mac_normalized(_nid):
        AGENT_ID = _nid


def _run_cmd_list_sync(cmd: list[str], *, timeout_sec: float) -> list[str]:
    """运行命令并按行解析输出；异常时返回空列表。"""
    try:
        out = subprocess.check_output(
            cmd,
            timeout=timeout_sec,
            stderr=subprocess.STDOUT,
        ).decode(errors="replace")
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []
    except subprocess.CalledProcessError as e:
        out = (e.output or b"").decode(errors="replace") if hasattr(e, "output") else ""
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


_ros_topic_cache: Dict[str, Any] = {"ts": 0.0, "topics": []}
_joint_payload_cache: Dict[str, Any] = {}
_ft_payload_cache: Dict[str, Any] = {}
_cpu_stat_prev: Optional[tuple[float, float]] = None
_joint_sampler_task: Optional[asyncio.Task] = None
_ft_sampler_task: Optional[asyncio.Task] = None


def _read_cpu_percent_linux() -> Optional[float]:
    """
    读取 /proc/stat 计算 CPU 使用率（两次采样差分）。
    首次调用仅建立基线，返回 None。
    """
    global _cpu_stat_prev
    try:
        with open("/proc/stat", "r", encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        if not first.startswith("cpu "):
            return None
        parts = first.split()[1:]
        if len(parts) < 4:
            return None
        vals = [float(x) for x in parts[:8]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
        total = sum(vals)
        if _cpu_stat_prev is None:
            _cpu_stat_prev = (total, idle)
            return None
        prev_total, prev_idle = _cpu_stat_prev
        dt = max(0.0, total - prev_total)
        di = max(0.0, idle - prev_idle)
        _cpu_stat_prev = (total, idle)
        if dt <= 1e-6:
            return None
        busy = max(0.0, min(1.0, (dt - di) / dt))
        return round(busy * 100.0, 2)
    except Exception:
        return None


def _collect_ros_topic_names() -> List[str]:
    """
    获取 ROS 话题名（用于设备详情页展示），带轻量缓存避免每次心跳都执行外部命令。
    """
    now = time.time()
    ttl = float(os.environ.get("EAI_AGENT_ROS_TOPIC_CACHE_SEC", "15") or "15")
    cached_ts = float(_ros_topic_cache.get("ts") or 0.0)
    cached_topics = _ros_topic_cache.get("topics") or []
    if (now - cached_ts) < max(3.0, ttl) and isinstance(cached_topics, list):
        return [str(x) for x in cached_topics if str(x).strip()]

    topics = _run_cmd_list_sync(["ros2", "topic", "list"], timeout_sec=2.5)
    if not topics:
        # ros2 命令不可用时，回退到 stream_manager 的 topic_mapping 值（至少保证相机 topic 可见）
        try:
            if stream_manager is not None and hasattr(stream_manager, "topic_mapping"):
                mapping = getattr(stream_manager, "topic_mapping", {}) or {}
                topics = sorted(set([str(v) for v in mapping.values() if str(v).strip()]))
        except Exception:
            topics = []

    topics = sorted(set([t.strip() for t in topics if t and t.strip()]))
    _ros_topic_cache["ts"] = now
    _ros_topic_cache["topics"] = topics
    return topics


def _robot_state_subscription_active() -> bool:
    """
    是否使用 rclpy 订阅（与相机共用 Node）填充关节/末端状态，而非 ros2 topic echo。
    关闭：EAI_AGENT_ROS_STATE_USE_SUB=0
    """
    if (os.environ.get("EAI_AGENT_ROS_STATE_USE_SUB", "1") or "1").strip() == "0":
        return False
    try:
        if stream_manager is None:
            return False
        br = getattr(stream_manager, "robot_state_bridge", None)
        return br is not None and bool(getattr(br, "is_active", lambda: False)())
    except Exception:
        return False


async def _joint_sampler_loop() -> None:
    """
    后台关节采样循环（与 HEARTBEAT 解耦）：
    - 在线程池里执行 ros2 命令，避免阻塞事件循环
    - HEARTBEAT 仅读缓存，保证时延稳定
    """
    interval_echo = float(os.environ.get("EAI_AGENT_JOINT_SAMPLE_INTERVAL_SEC", "2.0") or "2.0")
    interval_echo = max(0.5, interval_echo)
    interval_sub = float(os.environ.get("EAI_AGENT_ROS_STATE_EXPORT_INTERVAL_SEC", "0.5") or "0.5")
    interval_sub = max(0.2, interval_sub)

    while True:
        sub_on = _robot_state_subscription_active()
        tick = interval_sub if sub_on else interval_echo
        try:
            if sub_on:
                br = getattr(stream_manager, "robot_state_bridge", None)
                if br is None:
                    await asyncio.sleep(tick)
                    continue
                joint_payload = await asyncio.to_thread(br.export_joint_payload)
                if not joint_payload.get("joints") and not joint_payload.get("joint_states_by_topic"):
                    fallback_joint_payload = await asyncio.to_thread(_collect_joint_state_payload)
                    if fallback_joint_payload:
                        joint_payload.update(fallback_joint_payload)
                has_joint_data = bool(joint_payload.get("joints") or joint_payload.get("joint_states_by_topic"))
                if has_joint_data:
                    _joint_payload_cache.clear()
                    _joint_payload_cache.update(joint_payload)
                elif _joint_payload_cache:
                    merged = dict(_joint_payload_cache)
                    if joint_payload.get("joint_topics"):
                        merged["joint_topics"] = joint_payload.get("joint_topics")
                    _joint_payload_cache.clear()
                    _joint_payload_cache.update(merged)
                elif joint_payload.get("joint_topics"):
                    _joint_payload_cache.clear()
                    _joint_payload_cache.update({"joint_topics": joint_payload.get("joint_topics")})
                all_t = await asyncio.to_thread(br.get_all_topic_names)
                if all_t:
                    _ros_topic_cache["topics"] = all_t
                    _ros_topic_cache["ts"] = time.time()
            else:
                ros_topic_names = await asyncio.to_thread(_collect_ros_topic_names)
                joint_payload = await asyncio.to_thread(_collect_joint_topics_payload, ros_topic_names)
                if not joint_payload.get("joints") and not joint_payload.get("joint_states_by_topic"):
                    fallback_joint_payload = await asyncio.to_thread(_collect_joint_state_payload)
                    if fallback_joint_payload:
                        joint_payload.update(fallback_joint_payload)
                has_joint_data = bool(joint_payload.get("joints") or joint_payload.get("joint_states_by_topic"))
                if has_joint_data:
                    _joint_payload_cache.clear()
                    _joint_payload_cache.update(joint_payload)
                elif _joint_payload_cache:
                    # 仅发现话题但未采到值时，不覆盖旧数据；只更新话题列表供前端切换。
                    merged = dict(_joint_payload_cache)
                    if joint_payload.get("joint_topics"):
                        merged["joint_topics"] = joint_payload.get("joint_topics")
                    _joint_payload_cache.clear()
                    _joint_payload_cache.update(merged)
                elif joint_payload.get("joint_topics"):
                    # 首次启动时先让前端看到可用话题，等待下一轮采样补齐数据。
                    _joint_payload_cache.clear()
                    _joint_payload_cache.update({"joint_topics": joint_payload.get("joint_topics")})
        except Exception:
            # 采样失败保持旧缓存，避免前端抖动
            pass
        await asyncio.sleep(tick)


async def _ft_sampler_loop() -> None:
    """
    后台末端力/力矩（wrench / ft）采样循环（与 HEARTBEAT 解耦）：
    - 在线程池里执行 ros2 命令，避免阻塞事件循环
    - HEARTBEAT 仅读缓存，保证时延稳定
    """
    interval_echo = float(os.environ.get("EAI_AGENT_FT_SAMPLE_INTERVAL_SEC", "2.0") or "2.0")
    interval_echo = max(0.5, interval_echo)
    interval_sub = float(os.environ.get("EAI_AGENT_ROS_STATE_EXPORT_INTERVAL_SEC", "0.5") or "0.5")
    interval_sub = max(0.2, interval_sub)

    while True:
        sub_on = _robot_state_subscription_active()
        tick = interval_sub if sub_on else interval_echo
        try:
            if sub_on:
                br = getattr(stream_manager, "robot_state_bridge", None)
                if br is None:
                    await asyncio.sleep(tick)
                    continue
                ft_payload = await asyncio.to_thread(br.export_ft_payload)
                has_ft_data = bool(ft_payload.get("ft_states_by_topic") or ft_payload.get("ft_force") or ft_payload.get("ft_torque"))
                if has_ft_data:
                    _ft_payload_cache.clear()
                    _ft_payload_cache.update(ft_payload)
                elif _ft_payload_cache:
                    merged = dict(_ft_payload_cache)
                    if ft_payload.get("ft_topics"):
                        merged["ft_topics"] = ft_payload.get("ft_topics")
                    _ft_payload_cache.clear()
                    _ft_payload_cache.update(merged)
                elif ft_payload.get("ft_topics"):
                    _ft_payload_cache.clear()
                    _ft_payload_cache.update({"ft_topics": ft_payload.get("ft_topics")})
            else:
                ros_topic_names = await asyncio.to_thread(_collect_ros_topic_names)
                ft_payload = await asyncio.to_thread(_collect_ft_topics_payload, ros_topic_names)
                has_ft_data = bool(ft_payload.get("ft_states_by_topic") or ft_payload.get("ft_force") or ft_payload.get("ft_torque"))
                if has_ft_data:
                    _ft_payload_cache.clear()
                    _ft_payload_cache.update(ft_payload)
                elif _ft_payload_cache:
                    # 仅发现话题列表但未采到值时，不覆盖旧数据；只更新 topics 供前端切换。
                    merged = dict(_ft_payload_cache)
                    if ft_payload.get("ft_topics"):
                        merged["ft_topics"] = ft_payload.get("ft_topics")
                    _ft_payload_cache.clear()
                    _ft_payload_cache.update(merged)
                elif ft_payload.get("ft_topics"):
                    _ft_payload_cache.clear()
                    _ft_payload_cache.update({"ft_topics": ft_payload.get("ft_topics")})
        except Exception:
            # 采样失败保持旧缓存，避免前端抖动
            pass
        await asyncio.sleep(tick)


def _list_video_devices_sync() -> list[str]:
    """扫描本机 /dev/video* 设备（通过 subprocess 检查）。"""
    try:
        # 用 /bin/sh 避免依赖特定 shell；无设备时不抛错
        cp = subprocess.run(
            ["sh", "-c", "ls -1 /dev/video* 2>/dev/null || true"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        lines = cp.stdout.splitlines() if cp.stdout else []
        return [ln.strip() for ln in lines if ln.strip().startswith("/dev/video")]
    except Exception:
        return []


def _gather_agent_info_sync() -> AgentInfoResponse:
    agent_uuid = _get_hardware_uuid_sync()
    hostname = socket.gethostname() or agent_uuid

    # ROS2 运行状态采样（使用 subprocess）
    nodes = _run_cmd_list_sync(["ros2", "node", "list"], timeout_sec=8.0)
    topics = _run_cmd_list_sync(["ros2", "topic", "list"], timeout_sec=8.0)

    # 硬件驱动/视频设备采样（使用 subprocess）
    video_devices = _list_video_devices_sync()

    # 识别“疑似图像/相机”话题
    img_topic_patterns = re.compile(r"(image_raw|image|/image|/compressed|compressed|camera)", re.IGNORECASE)
    image_topics = [t for t in topics if img_topic_patterns.search(t)]
    image_topics = sorted(set(image_topics))

    camera_list = sorted(set((video_devices or []) + (image_topics or [])))

    ros2_ok = (len(nodes) > 0) or (len(topics) > 0)
    drivers_ok = len(video_devices) > 0

    overall = "READY" if (ros2_ok and drivers_ok) else "ERROR"

    status: Dict[str, Any] = {
        "overall": overall,
        "ros2": {
            "node_count": len(nodes),
            "topic_count": len(topics),
            "nodes_sample": nodes[:10],
            "topics_sample": topics[:10],
        },
        "hardware": {
            "video_devices": video_devices,
        },
    }

    capabilities: Dict[str, Any] = {
        "camera_list": camera_list,
        "video_devices": video_devices,
        "image_topics": image_topics,
    }

    return AgentInfoResponse(
        uuid=agent_uuid,
        hostname=hostname,
        status=status,
        capabilities=capabilities,
        version=AGENT_VERSION,
    )


class CollectStartRequest(BaseModel):
    task_id: Optional[str] = None
    job_id: Optional[str] = None
    run_id: Optional[str] = None
    scenario_id: Optional[str] = None
    duration_sec: int
    storage_path: str
    camera_data_format: Optional[str] = None
    env: Optional[dict] = None
    # 平台下发的采集脚本路径与参数（优先级高于 Agent 侧默认选择）
    script_path: Optional[str] = None
    args: Optional[List[str]] = None


class CollectSimpleResponse(BaseModel):
    ok: bool
    msg: Optional[str] = None


class DeviceTestRequest(BaseModel):
    device_id: Optional[int] = None


class DeviceTestResponse(BaseModel):
    ok: bool
    msg: Optional[str] = None
    node_count: int = 0
    topic_count: int = 0


class DeviceLaunchRequest(BaseModel):
    script_path: str
    script_args: Optional[str] = None
    env: Optional[dict] = None


class DeviceControlResponse(BaseModel):
    ok: bool
    msg: Optional[str] = None


class AgentListDirsResponse(BaseModel):
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class DataSyncRequest(BaseModel):
    asset_id: int
    source_path: str
    bucket_name: str
    object_prefix: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool = False


class DataSyncResponse(BaseModel):
    ok: bool
    minio_path: Optional[str] = None
    message: Optional[str] = None


async def _data_sync_via_minio_payload(data_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    隧道命令版数据同步：
    - 复用 HTTP /api/agent/data/sync 的 MinIO 上传逻辑
    - 返回 shape 与 CMD_RESULT 兼容：success/msg/minio_path
    """
    try:
        req = DataSyncRequest(**(data_payload or {}))
        minio_path = await asyncio.to_thread(_upload_path_to_minio, req)
        return {"success": True, "msg": "同步成功", "minio_path": minio_path}
    except S3Error as e:
        logger.error(
            "data_sync agent(tunnel): MinIO S3 错误 asset_id=%s code=%s message=%s resource=%s request_id=%s",
            getattr(data_payload, "asset_id", None) or data_payload.get("asset_id"),
            getattr(e, "code", None),
            getattr(e, "message", None),
            getattr(e, "resource", None),
            getattr(e, "request_id", None),
        )
        return {"success": False, "msg": f"MinIO 错误: {e.code} {e.message}"}
    except RuntimeError as e:
        logger.error(
            "data_sync agent(tunnel): 上传前置检查/路径错误 asset_id=%s err=%s source_path=%r",
            data_payload.get("asset_id"),
            e,
            (str(data_payload.get("source_path") or "")[:500]),
        )
        return {"success": False, "msg": str(e)}
    except Exception as e:
        logger.exception(
            "data_sync agent(tunnel): 未预期异常 asset_id=%s source_path=%r",
            data_payload.get("asset_id"),
            (str(data_payload.get("source_path") or "")[:500]),
        )
        return {"success": False, "msg": str(e)}


class AgentFsListResponse(BaseModel):
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None


def _resolve_agent_fs_path(path: Optional[str]) -> str:
    raw = (path or "").strip()
    base = os.path.realpath(AGENT_DATA_ROOT)
    if not raw:
        return base
    if raw.startswith("/"):
        resolved = os.path.realpath(raw)
    else:
        resolved = os.path.realpath(os.path.join(base, raw))
    if resolved != base and not resolved.startswith(base + os.sep):
        raise RuntimeError("路径不在允许访问的采集端根目录下")
    return resolved


def _upload_path_to_minio(payload: DataSyncRequest) -> str:
    raw_ep = (payload.minio_endpoint or "").strip()
    endpoint = _normalize_minio_endpoint(raw_ep)
    if endpoint != raw_ep and raw_ep:
        logger.info(
            "data_sync agent: MinIO endpoint 已规范化 asset_id=%s raw=%r -> %r",
            payload.asset_id,
            raw_ep[:120],
            endpoint,
        )
    logger.info(
        "data_sync agent: 开始上传 asset_id=%s endpoint=%s secure=%s bucket=%s prefix=%s source_path=%r",
        payload.asset_id,
        endpoint,
        bool(payload.minio_secure),
        (payload.bucket_name or "").strip(),
        (payload.object_prefix or "").strip(),
        (payload.source_path or "").strip(),
    )
    client = Minio(
        endpoint,
        access_key=payload.minio_access_key,
        secret_key=payload.minio_secret_key,
        secure=bool(payload.minio_secure),
    )
    src = os.path.normpath((payload.source_path or "").strip())
    if not src:
        raise RuntimeError("source_path 为空")
    if not os.path.exists(src):
        raise RuntimeError(f"source_path 不存在: {src}")

    bucket = (payload.bucket_name or "").strip()
    prefix = (payload.object_prefix or "").strip().strip("/")
    if not bucket or not prefix:
        raise RuntimeError("bucket_name/object_prefix 不能为空")

    if os.path.isfile(src):
        name = os.path.basename(src)
        object_name = f"{prefix}/{name}"
        client.fput_object(bucket, object_name, src)
        out = f"minio://{bucket}/{object_name}"
        logger.info("data_sync agent: 单文件上传完成 asset_id=%s minio_path=%s", payload.asset_id, out)
        return out

    if os.path.isdir(src):
        n = 0
        for root, _, files in os.walk(src):
            for fn in files:
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, src).replace(os.sep, "/")
                object_name = f"{prefix}/{rel}"
                client.fput_object(bucket, object_name, fp)
                n += 1
        out = f"minio://{bucket}/{prefix}/"
        logger.info("data_sync agent: 目录上传完成 asset_id=%s files=%s minio_path=%s", payload.asset_id, n, out)
        return out

    raise RuntimeError(f"不支持的 source_path 类型: {src}")


@app.post("/api/agent/data/sync", response_model=DataSyncResponse)
async def sync_data_to_minio(payload: DataSyncRequest) -> DataSyncResponse:
    """
    将采集端本地路径（文件或目录）同步到平台 MinIO。
    """
    try:
        minio_path = await asyncio.to_thread(_upload_path_to_minio, payload)
        return DataSyncResponse(ok=True, minio_path=minio_path, message="同步成功")
    except S3Error as e:
        logger.error(
            "data_sync agent: MinIO S3 错误 asset_id=%s code=%s message=%s resource=%s request_id=%s",
            payload.asset_id,
            getattr(e, "code", None),
            getattr(e, "message", None),
            getattr(e, "resource", None),
            getattr(e, "request_id", None),
        )
        return DataSyncResponse(ok=False, message=f"MinIO 错误: {e.code} {e.message}")
    except RuntimeError as e:
        logger.error(
            "data_sync agent: 上传前置检查/路径错误 asset_id=%s err=%s source_path=%r",
            payload.asset_id,
            e,
            (payload.source_path or "")[:500],
        )
        return DataSyncResponse(ok=False, message=str(e))
    except Exception as e:
        logger.exception(
            "data_sync agent: 未预期异常 asset_id=%s source_path=%r",
            payload.asset_id,
            (payload.source_path or "")[:500],
        )
        return DataSyncResponse(ok=False, message=str(e))


@app.get("/api/agent/script/report")
async def agent_get_report(path: str):
    """在采集端读取 validation_report.json 并返回 JSON。"""
    report_path = os.path.join(path, "validation_report.json")
    if not os.path.exists(report_path):
        return {"ok": False, "error": "Validation report not found"}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        # 保持与服务器版本一致的 episode_dir 修正逻辑
        ep_dir = report_data.get("episode_dir")
        if ep_dir and os.path.isdir(ep_dir):
            found_file = None
            try:
                for f_name in os.listdir(ep_dir):
                    if f_name.endswith(".mcap"):
                        found_file = os.path.join(ep_dir, f_name)
                        break
                if not found_file:
                    for f_name in os.listdir(ep_dir):
                        if f_name.endswith(".db3"):
                            found_file = os.path.join(ep_dir, f_name)
                            break
                if found_file:
                    report_data["episode_dir"] = found_file
            except OSError:
                pass
        return {"ok": True, "data": report_data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _is_job_workspace_segment_dir(path: str) -> bool:
    """与平台作业编号子目录一致：纯四位数字（如 0001）。"""
    try:
        bn = os.path.basename(os.path.normpath(path).rstrip(os.sep))
    except Exception:
        return False
    return len(bn) == 4 and bn.isdigit()


def agent_delete_data_impl(
    path: str,
    *,
    allow_job_workspace: bool = False,
    allow_incomplete_episode: bool = False,
    workspace_root: Optional[str] = None,
):
    """同步删除逻辑；供 HTTP 与隧道共用。返回 dict success/message。"""
    if not path:
        return {"success": False, "message": "Path is required"}
    if not os.path.exists(path):
        return {"success": False, "message": "Path does not exist"}
    try:
        rp = os.path.realpath(path)
        if rp == "/opt/eai-agent" or rp.startswith("/opt/eai-agent/"):
            return {"success": False, "message": f"Refuse to delete protected path: {path}"}
    except Exception:
        pass

    def _looks_like_bag_file(p: str) -> bool:
        low = (p or "").strip().lower()
        return low.endswith(".mcap") or low.endswith(".mca") or low.endswith(".db3") or low.endswith(".bag")

    def _dir_has_bag_files(d: str) -> bool:
        try:
            for name in os.listdir(d):
                low = name.lower()
                if low.endswith(".mcap") or low.endswith(".mca") or low.endswith(".db3"):
                    full = os.path.join(d, name)
                    if os.path.isfile(full):
                        return True
            if os.path.isfile(os.path.join(d, "validation_report.json")):
                return True
        except OSError:
            return False
        return False

    def _is_under_workspace_root(root: Optional[str], target: str) -> bool:
        if not (root or "").strip():
            return False
        try:
            rr = os.path.realpath(str(root).strip())
            rt = os.path.realpath(target)
            return rt == rr or rt.startswith(rr + os.sep)
        except OSError:
            return False

    try:
        if os.path.isdir(path):
            if allow_job_workspace and _is_job_workspace_segment_dir(path):
                pass
            elif (
                allow_incomplete_episode
                and workspace_root
                and _is_under_workspace_root(workspace_root, path)
            ):
                try:
                    bn = os.path.basename(os.path.normpath(path).rstrip(os.sep))
                except Exception:
                    bn = ""
                if not (bn.lower().startswith("episode_")):
                    return {"success": False, "message": f"Refuse incomplete delete: not episode dir: {path}"}
            elif not _dir_has_bag_files(path):
                return {"success": False, "message": f"Refuse to delete non-bag dir: {path}"}
        else:
            if not _looks_like_bag_file(path):
                return {"success": False, "message": f"Refuse to delete non-bag file: {path}"}
    except Exception:
        return {"success": False, "message": f"Refuse to delete unsafe path: {path}"}
    try:
        if os.path.isdir(path):
            import shutil

            shutil.rmtree(path)
        else:
            os.remove(path)
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": f"Failed to delete: {str(e)}"}


@app.delete("/api/agent/script/data")
async def agent_delete_data(
    path: str,
    allow_job_workspace: bool = False,
):
    """在采集端删除文件或目录（用于重新采集）。allow_job_workspace=True 时允许删除四位作业子目录（可与平台删除作业同步）。"""
    return agent_delete_data_impl(path, allow_job_workspace=bool(allow_job_workspace))


class WebRtcOffer(BaseModel):
    """平台/浏览器发送到 Agent 的 WebRTC Offer。"""

    sdp: str
    type: str
    camera_id: Optional[str] = None


@app.post("/api/agent/webrtc/offer")
async def agent_webrtc_offer(offer: WebRtcOffer):
    """
    使用 aiortc 在采集端建立 WebRTC 会话，将 ROS2 摄像头视频推送给浏览器。
    """
    global _webrtc_pcs

    if stream_manager is None:
        # 透传给平台：这不是“answer 格式不对”，而是采集端能力缺失
        return {"ok": False, "error": "ROS2 camera stream manager not available", "code": "NO_STREAM_MANAGER"}

    if not stream_manager.topic_mapping:
        stream_manager.refresh_topics()

    available_ids = sorted(stream_manager.topic_mapping.keys())
    if not available_ids:
        return {"ok": False, "error": "No camera topics available from ROS2", "code": "NO_CAMERA_TOPICS"}

    # 根据前端传入的 camera_id 选择具体相机；如果不存在则回退到第一个
    if offer.camera_id and offer.camera_id in stream_manager.topic_mapping:
        camera_id = offer.camera_id
    else:
        camera_id = available_ids[0]

    # 每个 camera_id 维护一个独立的 PeerConnection；新建前关闭旧连接
    key = camera_id or "__default__"
    lock = _get_webrtc_pc_lock(key)
    async with lock:
        old_pc = _webrtc_pcs.get(key)
        if old_pc is not None:
            try:
                await old_pc.close()
            except Exception:
                pass
            _webrtc_pcs.pop(key, None)

        pc = RTCPeerConnection(
            RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                ]
            )
        )
        _webrtc_pcs[key] = pc

        @pc.on("connectionstatechange")
        async def _on_connection_state_change() -> None:
            try:
                state = str(getattr(pc, "connectionState", ""))
                if state in ("failed", "closed", "disconnected"):
                    if _webrtc_pcs.get(key) is pc:
                        _webrtc_pcs.pop(key, None)
                    if state != "closed":
                        await pc.close()
            except Exception:
                pass

        class CameraVideoTrack(VideoStreamTrack):
            """从 CameraStreamManager 读取最新 JPEG 帧并转为 WebRTC 视频轨。"""

            def __init__(self, cam_id: str):
                super().__init__()
                self.cam_id = cam_id

            async def recv(self) -> VideoFrame:
                pts, time_base = await self.next_timestamp()

                frame_bytes = None
                # 尝试多次读取最新帧，避免立即返回空帧
                for _ in range(5):
                    with stream_manager.lock:
                        topic = stream_manager.active_topics.get(self.cam_id)
                        if topic:
                            frame_bytes = topic.get("latest_frame")
                    if frame_bytes:
                        break
                    await asyncio.sleep(0.04)

                if frame_bytes:
                    arr = np.frombuffer(frame_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        img = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)

                frame = VideoFrame.from_ndarray(img, format="bgr24")
                frame.pts = pts
                frame.time_base = time_base
                return frame

        pc.addTrack(CameraVideoTrack(camera_id))

        # 处理浏览器 Offer
        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            if getattr(pc, "iceGatheringState", "") != "complete":
                done = asyncio.Event()

                @pc.on("icegatheringstatechange")
                async def _on_ice_gathering_state_change() -> None:
                    if getattr(pc, "iceGatheringState", "") == "complete":
                        done.set()

                try:
                    await asyncio.wait_for(done.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            try:
                await pc.close()
            except Exception:
                pass
            if _webrtc_pcs.get(key) is pc:
                _webrtc_pcs.pop(key, None)
            raise

        return {
            "ok": True,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }


_process: Optional[subprocess.Popen[str]] = None
_device_process: Optional[subprocess.Popen[str]] = None
_webrtc_pcs: Dict[str, RTCPeerConnection] = {}
_webrtc_pc_locks: Dict[str, asyncio.Lock] = {}


def _get_webrtc_pc_lock(key: str) -> asyncio.Lock:
    lock = _webrtc_pc_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _webrtc_pc_locks[key] = lock
    return lock


def _install_asyncio_webrtc_exception_filter() -> None:
    """
    过滤 aiortc 在连接主动关闭时产生的噪声日志：
    Task exception was never retrieved ... InvalidStateError('RTCIceTransport is closed')
    """
    loop = asyncio.get_running_loop()
    prev_handler = loop.get_exception_handler()

    def _handler(loop_obj: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
        exc = context.get("exception")
        msg = str(context.get("message") or "")
        if (
            msg == "Task exception was never retrieved"
            and exc is not None
            and exc.__class__.__name__ == "InvalidStateError"
            and "RTCIceTransport is closed" in str(exc)
        ):
            logger.debug("ignore aiortc closed ice transport noise: %s", exc)
            return
        if prev_handler is not None:
            prev_handler(loop_obj, context)
        else:
            loop_obj.default_exception_handler(context)

    loop.set_exception_handler(_handler)


# -------------------------
# WebSocket Tunnel (Phase-1)
# -------------------------
_tunnel_ws: Optional[object] = None  # aiohttp ClientWebSocketResponse (typed loosely)
_tunnel_connected_event = asyncio.Event()
# 文本控制/日志优先于 MJPEG 二进制（文档 §4.1 单隧道下的发送优先级）
_tunnel_out_seq = itertools.count()
_tunnel_outq: asyncio.PriorityQueue = asyncio.PriorityQueue()
_tunnel_out_worker_task: Optional[asyncio.Task] = None
_tunnel_heartbeat_task: Optional[asyncio.Task] = None

# command_id -> CMD_RESULT.payload (for idempotency)
_tunnel_command_results: Dict[str, Dict[str, Any]] = {}
_tunnel_command_lock = asyncio.Lock()
# 同一 command_id 并发/重发时串行执行，避免 COLLECT_START 等被重复跑（文档 §9）
_tunnel_cmd_id_exec_locks: Dict[str, asyncio.Lock] = {}

# MJPEG tunnel state (phase-2)
_tunnel_mjpeg_camera_id_to_cam_idx: Dict[str, int] = {}
_tunnel_mjpeg_sender_tasks: Dict[str, asyncio.Task] = {}
_tunnel_mjpeg_frame_id: Dict[str, int] = {}
_tunnel_mjpeg_last_update_ts: Dict[str, float] = {}

CHUNK_SIZE = 32 * 1024  # 32768
MJPEG_CHUNK_MAGIC = b"EAI1"
MJPEG_CHUNK_VER = 1
MJPEG_CHUNK_KIND = 1


async def _get_tunnel_cmd_exec_lock(command_id: str) -> asyncio.Lock:
    async with _tunnel_command_lock:
        lk = _tunnel_cmd_id_exec_locks.get(command_id)
        if lk is None:
            lk = asyncio.Lock()
            _tunnel_cmd_id_exec_locks[command_id] = lk
        return lk


def _server_base_to_ws_base(server_base: str) -> str:
    s = (server_base or "").strip().rstrip("/")
    if s.startswith("https://"):
        return "wss://" + s.removeprefix("https://")
    if s.startswith("http://"):
        return "ws://" + s.removeprefix("http://")
    # Fallback: treat as already ws(s) base.
    return s


async def _tunnel_send_envelope(envelope: Dict[str, Any]) -> None:
    """
    Best-effort send over tunnel.
    If tunnel is disconnected, caller should handle fallback externally.
    """
    if _tunnel_ws is None:
        raise RuntimeError("tunnel websocket not connected")
    s = json.dumps(envelope, ensure_ascii=False)
    # priority 0 = high (text)
    await _tunnel_outq.put((0, next(_tunnel_out_seq), "text", s))


async def _tunnel_send_bytes(message_bytes: bytes) -> None:
    if _tunnel_ws is None:
        raise RuntimeError("tunnel websocket not connected")
    # priority 1 = low (MJPEG binary)
    await _tunnel_outq.put((1, next(_tunnel_out_seq), "bytes", message_bytes))


def _tunnel_drain_outq() -> None:
    """断连时丢弃积压的出站消息，避免重连后乱序发往新连接。"""
    while True:
        try:
            _tunnel_outq.get_nowait()
        except asyncio.QueueEmpty:
            break


async def _tunnel_outbound_worker() -> None:
    """Drain outbound priority queue; text always ahead of binary chunks."""
    while True:
        try:
            _prio, _seq, kind, payload = await asyncio.wait_for(_tunnel_outq.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise
        except Exception:
            continue
        if _tunnel_ws is None:
            continue
        try:
            if kind == "text":
                await _tunnel_ws.send_str(payload)  # type: ignore[attr-defined]
            else:
                await _tunnel_ws.send_bytes(payload)  # type: ignore[attr-defined]
        except Exception:
            pass


async def _tunnel_heartbeat_loop() -> None:
    while True:
        try:
            await _tunnel_connected_event.wait()
            collect_running = bool(_process and _process.poll() is None)
            device_running = bool(_device_process and _device_process.poll() is None)
            cpu_percent = _read_cpu_percent_linux()
            mem_total_mb = None
            mem_used_mb = None
            disk_total_gb = None
            disk_used_gb = None
            disk_free_gb = None
            try:
                # 优先读取 Linux /proc/meminfo，避免额外依赖
                meminfo: Dict[str, int] = {}
                with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        parts = line.split(":", 1)
                        if len(parts) != 2:
                            continue
                        key = parts[0].strip()
                        raw = parts[1].strip().split()
                        if not raw:
                            continue
                        try:
                            meminfo[key] = int(raw[0])  # kB
                        except Exception:
                            continue
                total_kb = int(meminfo.get("MemTotal", 0))
                avail_kb = int(meminfo.get("MemAvailable", 0))
                if total_kb > 0:
                    mem_total_mb = round(total_kb / 1024.0, 2)
                    used_kb = max(0, total_kb - max(0, avail_kb))
                    mem_used_mb = round(used_kb / 1024.0, 2)
            except Exception:
                pass
            try:
                du = shutil.disk_usage(AGENT_DATA_ROOT or "/")
                total = float(du.total)
                free = float(du.free)
                used = max(0.0, total - free)
                if total > 0:
                    disk_total_gb = round(total / (1024.0 ** 3), 2)
                    disk_used_gb = round(used / (1024.0 ** 3), 2)
                    disk_free_gb = round(free / (1024.0 ** 3), 2)
            except Exception:
                pass
            ros_topic_names = [str(x) for x in (_ros_topic_cache.get("topics") or []) if str(x).strip()]
            joint_payload = dict(_joint_payload_cache)
            ft_payload = dict(_ft_payload_cache)
            await _tunnel_send_envelope(
                {
                    "ver": 1,
                    "type": "HEARTBEAT",
                    "agent_id": AGENT_ID,
                    "ts_ms": int(time.time() * 1000),
                    "payload": {
                        "online": True,
                        "uptime_sec": int(max(0.0, time.time() - _AGENT_PROCESS_START_TS)),
                        "collect_running": collect_running,
                        "device_running": device_running,
                        "cpu_percent": cpu_percent,
                        "outq_size": int(_tunnel_outq.qsize()),
                        "mem_total_mb": mem_total_mb,
                        "mem_used_mb": mem_used_mb,
                        "disk_total_gb": disk_total_gb,
                        "disk_used_gb": disk_used_gb,
                        "disk_free_gb": disk_free_gb,
                        "ros_topic_names": ros_topic_names,
                        **ft_payload,
                        **joint_payload,
                    },
                }
            )
            # 定期重发 STREAM_CAPS：避免隧道首次连接时 topic_mapping 尚未就绪导致平台相机列表长期为空。
            await _send_stream_caps()
        except Exception:
            pass
        await asyncio.sleep(max(5.0, _HEARTBEAT_INTERVAL_SEC))


def _get_available_camera_ids() -> List[str]:
    if stream_manager is None:
        return []
    try:
        if hasattr(stream_manager, "refresh_topics"):
            stream_manager.refresh_topics()
    except Exception:
        pass
    try:
        mapping = getattr(stream_manager, "topic_mapping", {}) or {}
        return sorted([str(k) for k in mapping.keys() if str(k).strip()])
    except Exception:
        return []


async def _send_stream_caps() -> None:
    # STREAM_CAPS tells platform which camera_ids are available; platform assigns cam_idx.
    camera_ids = _get_available_camera_ids()
    if not camera_ids:
        return
    try:
        await _tunnel_send_envelope(
            {
                "ver": 1,
                "type": "STREAM_CAPS",
                "agent_id": AGENT_ID,
                "ts_ms": int(time.time() * 1000),
                "payload": {"camera_ids": camera_ids},
            }
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            await _post_experiment_event(
                client,
                "preview_capability_report",
                camera_ids=camera_ids,
                camera_count=len(camera_ids),
                device_open_ok=bool(camera_ids),
            )
    except Exception:
        pass


async def _stop_mjpeg_senders() -> None:
    # Cancel existing sender tasks.
    for t in list(_tunnel_mjpeg_sender_tasks.values()):
        try:
            t.cancel()
        except Exception:
            pass
    _tunnel_mjpeg_sender_tasks.clear()
    _tunnel_mjpeg_frame_id.clear()
    _tunnel_mjpeg_last_update_ts.clear()


async def _apply_mjpeg_mapping(camera_id_to_cam_idx: Dict[str, int]) -> None:
    global _tunnel_mjpeg_camera_id_to_cam_idx
    _tunnel_mjpeg_camera_id_to_cam_idx = {str(k): int(v) for k, v in camera_id_to_cam_idx.items()}
    await _stop_mjpeg_senders()
    # Start new sender tasks.
    # Ensure ROS2 stream subscriptions exist for each mapped camera_id.
    try:
        if stream_manager is not None and hasattr(stream_manager, "refresh_topics"):
            stream_manager.refresh_topics()
    except Exception:
        pass
    try:
        if stream_manager is not None and hasattr(stream_manager, "subscribe"):
            active = getattr(stream_manager, "active_topics", {}) or {}
            topic_mapping = getattr(stream_manager, "topic_mapping", {}) or {}
            for camera_id in _tunnel_mjpeg_camera_id_to_cam_idx.keys():
                if camera_id in active:
                    continue
                topic = topic_mapping.get(camera_id)
                if topic:
                    stream_manager.subscribe(camera_id, topic)
    except Exception:
        # If subscription fails, sender loop will keep producing placeholders on platform side.
        pass

    for camera_id, cam_idx in _tunnel_mjpeg_camera_id_to_cam_idx.items():
        _tunnel_mjpeg_frame_id[camera_id] = 0
        _tunnel_mjpeg_last_update_ts[camera_id] = 0.0
        _tunnel_mjpeg_sender_tasks[camera_id] = asyncio.create_task(
            _mjpeg_sender_loop(camera_id=camera_id, cam_idx=cam_idx)
        )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await _post_experiment_event(
                client,
                "fallback_preview_up",
                camera_ids=list(_tunnel_mjpeg_camera_id_to_cam_idx.keys()),
                camera_count=len(_tunnel_mjpeg_camera_id_to_cam_idx),
            )
    except Exception:
        pass


async def _mjpeg_sender_loop(*, camera_id: str, cam_idx: int) -> None:
    """
    Continuously send latest JPEG bytes as MJPEG_CHUNK over tunnel.
    """
    global _tunnel_ws

    if stream_manager is None:
        return

    last_update_ts = _tunnel_mjpeg_last_update_ts.get(camera_id, 0.0)
    frame_id = _tunnel_mjpeg_frame_id.get(camera_id, 0)

    try:
        while True:
            # Exit quickly on tunnel disconnect.
            if _tunnel_ws is None or not _tunnel_connected_event.is_set():
                return

            jpeg_bytes: bytes | None = None
            update_ts: float = 0.0
            try:
                with stream_manager.lock:
                    topic = getattr(stream_manager, "active_topics", {}).get(camera_id) if hasattr(stream_manager, "active_topics") else None
                    if topic:
                        jpeg_bytes = topic.get("latest_frame")
                        update_ts = float(topic.get("last_update") or 0.0)
            except Exception:
                jpeg_bytes = None

            if jpeg_bytes and update_ts and update_ts != last_update_ts:
                # 出站队列积压时跳过本帧，优先让控制面/日志 drain（文档 §7.6C）
                if _tunnel_outq.qsize() > _TUNNEL_OUTQ_BACKPRESS_CHUNKS:
                    await asyncio.sleep(0.12)
                    continue
                last_update_ts = update_ts
                frame_id += 1
                _tunnel_mjpeg_frame_id[camera_id] = frame_id
                _tunnel_mjpeg_last_update_ts[camera_id] = last_update_ts

                # Send this frame in 32KB chunks.
                try:
                    await _tunnel_send_mjpeg_frame(cam_idx=cam_idx, frame_id=frame_id, jpeg_bytes=jpeg_bytes)
                except Exception:
                    # If send fails, tunnel likely unstable; stop sender to avoid spam.
                    return

            # 按队列深度降档帧间隔（文档 §7.6C 动态抽帧）
            qsz = _tunnel_outq.qsize()
            if qsz > int(_TUNNEL_OUTQ_BACKPRESS_CHUNKS * 1.5):
                await asyncio.sleep(0.22)
            elif qsz > _TUNNEL_OUTQ_BACKPRESS_CHUNKS:
                await asyncio.sleep(0.12)
            else:
                await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        return


async def _tunnel_send_mjpeg_frame(*, cam_idx: int, frame_id: int, jpeg_bytes: bytes) -> None:
    # chunk header: 32 bytes Big-endian
    # magic='EAI1', ver=1, kind=1, reserved=0, frame_id(u64), cam_idx(u32),
    # chunk_index(u32), chunk_count(u32), chunk_len(u32)
    total_len = len(jpeg_bytes)
    if total_len <= 0:
        return
    chunk_count = (total_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_index in range(int(chunk_count)):
        start = chunk_index * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_len)
        chunk = jpeg_bytes[start:end]
        chunk_len = len(chunk)

        header = struct.pack(
            ">4sBBH Q I I I I",
            MJPEG_CHUNK_MAGIC,
            MJPEG_CHUNK_VER,
            MJPEG_CHUNK_KIND,
            0,
            int(frame_id),
            int(cam_idx),
            int(chunk_index),
            int(chunk_count),
            int(chunk_len),
        )
        assert len(header) == 32
        await _tunnel_send_bytes(header + chunk)


async def _stop_collect_idempotent() -> tuple[bool, str]:
    """
    Stop collect process, with idempotent semantics:
    - no running process => success ("already stopped")
    - terminate in progress => success ("stopping")
    """
    global _process
    if not _process or _process.poll() is not None:
        _process = None
        return True, "already stopped"
    try:
        # Best effort: terminate whole process group/session.
        # start_collect uses start_new_session=True so pid is also pgid.
        pid = int(getattr(_process, "pid", 0) or 0)
        if pid > 0:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                _process = None
                return True, "already stopped"
            except Exception:
                # Fallback: terminate just the parent process.
                try:
                    _process.terminate()
                except Exception:
                    pass
        else:
            try:
                _process.terminate()
            except Exception:
                pass

        # Wait a bit; if still alive, force kill the whole group.
        for _ in range(20):  # ~2s
            if _process.poll() is not None:
                _process = None
                return True, "stopped"
            await asyncio.sleep(0.1)
        if pid > 0:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    _process.kill()
                except Exception:
                    pass
        else:
            try:
                _process.kill()
            except Exception:
                pass
        _process = None
        return True, "killed"
    except Exception as exc:
        return False, str(exc)


async def _handle_tunnel_text_message(message_text: str) -> None:
    """
    Handle incoming tunnel messages.
    Supports CMD_REQUEST:
    - COLLECT_START / COLLECT_STOP
    - DEVICE_LAUNCH / DEVICE_STOP
    - WEBRTC_OFFER

    Phase-2 supports:
    - STREAM_MAPPING (start MJPEG_CHUNK sending)
    """
    try:
        data = json.loads(message_text)
    except Exception:
        return

    msg_type = data.get("type")
    if msg_type == "STREAM_MAPPING":
        payload = data.get("payload") or {}
        if not isinstance(payload, dict):
            return
        cid_to_idx = payload.get("camera_id_to_cam_idx") or {}
        if not isinstance(cid_to_idx, dict):
            return
        # Start/refresh mjpeg senders
        try:
            await _apply_mjpeg_mapping({str(k): int(v) for k, v in cid_to_idx.items()})
            print(f"[agent] tunnel STREAM_MAPPING applied: cameras={len(cid_to_idx)}")
        except Exception:
            pass
        return

    if msg_type != "CMD_REQUEST":
        return

    command_id = data.get("command_id")
    payload = data.get("payload") or {}
    if not command_id or not isinstance(payload, dict):
        return

    cmd = payload.get("cmd") or ""
    ts_ms = int(time.time() * 1000)
    data_payload = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    run_id = str(data_payload.get("run_id") or "").strip() or None
    scenario_id = str(data_payload.get("scenario_id") or "").strip() or None
    task_id = str(data_payload.get("task_id") or "").strip() or None
    job_id = str(data_payload.get("job_id") or "").strip() or None

    if cmd == "COLLECT_STOP":
        print(f"[agent] tunnel CMD_REQUEST COLLECT_STOP command_id={command_id}")
    if cmd == "COLLECT_START":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await _post_experiment_event(
                    client,
                    "task_received",
                    run_id=run_id,
                    scenario_id=scenario_id,
                    task_id=task_id,
                    job_id=job_id,
                    command_id=str(command_id),
                    cmd=cmd,
                )
        except Exception:
            pass

    # Always ACK immediately so platform isn't blocked.
    try:
        await _tunnel_send_envelope(
            {
                "ver": 1,
                "type": "CMD_ACK",
                "agent_id": AGENT_ID,
                "command_id": str(command_id),
                "ts_ms": ts_ms,
                "payload": {"cmd": cmd},
            }
        )
    except Exception:
        return

    exec_lock = await _get_tunnel_cmd_exec_lock(str(command_id))
    async with exec_lock:
        async with _tunnel_command_lock:
            cached = _tunnel_command_results.get(str(command_id))
        if cached is not None:
            replay = dict(cached)
            replay["duplicate"] = True
            try:
                await _tunnel_send_envelope(
                    {
                        "ver": 1,
                        "type": "CMD_RESULT",
                        "agent_id": AGENT_ID,
                        "command_id": str(command_id),
                        "ts_ms": int(time.time() * 1000),
                        "payload": replay,
                    }
                )
            except Exception:
                pass
            return

        if cmd == "COLLECT_STOP":
            ok, msg = await _stop_collect_idempotent()
            result_payload = {"success": bool(ok), "msg": msg, "cmd": cmd}
        elif cmd == "COLLECT_START":
            try:
                req = CollectStartRequest(
                    task_id=data_payload.get("task_id"),
                    job_id=data_payload.get("job_id"),
                    run_id=data_payload.get("run_id"),
                    scenario_id=data_payload.get("scenario_id"),
                    duration_sec=int(data_payload.get("duration_sec") or 30),
                    storage_path=str(data_payload.get("storage_path") or ""),
                    camera_data_format=data_payload.get("camera_data_format"),
                    env=data_payload.get("env") if isinstance(data_payload.get("env"), dict) else None,
                    script_path=str(data_payload.get("script_path") or "").strip() or None,
                    args=data_payload.get("args") if isinstance(data_payload.get("args"), list) else None,
                )
                resp = await start_collect(req)
                result_payload = {"success": bool(resp.ok), "msg": resp.msg, "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "DEVICE_LAUNCH":
            try:
                req = DeviceLaunchRequest(
                    script_path=str(data_payload.get("script_path") or ""),
                    script_args=data_payload.get("script_args"),
                    env=data_payload.get("env") if isinstance(data_payload.get("env"), dict) else None,
                )
                resp = await device_launch(req)
                result_payload = {"success": bool(resp.ok), "msg": resp.msg, "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "DEVICE_STOP":
            try:
                stop_req = None
                if data_payload.get("script_path"):
                    stop_req = DeviceLaunchRequest(
                        script_path=str(data_payload.get("script_path")),
                        script_args=data_payload.get("script_args"),
                        env=data_payload.get("env") if isinstance(data_payload.get("env"), dict) else None,
                    )
                resp = await device_stop(stop_req)
                result_payload = {"success": bool(resp.ok), "msg": resp.msg, "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "DEVICE_TEST_CONNECTION":
            try:
                req = DeviceTestRequest(device_id=data_payload.get("device_id"))
                resp = await device_test_connection(req)
                result_payload = {
                    "success": bool(resp.ok),
                    "msg": resp.msg,
                    "cmd": cmd,
                    "node_count": int(resp.node_count or 0),
                    "topic_count": int(resp.topic_count or 0),
                }
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "SCRIPT_DELETE_DATA":
            try:
                p = str(data_payload.get("path") or "")
                allow_jw = bool(data_payload.get("allow_job_workspace"))
                allow_inc = bool(data_payload.get("allow_incomplete_episode"))
                ws_root = str(data_payload.get("workspace_root") or "").strip()
                rep = agent_delete_data_impl(
                    p,
                    allow_job_workspace=allow_jw,
                    allow_incomplete_episode=allow_inc,
                    workspace_root=ws_root or None,
                )
                if isinstance(rep, dict):
                    result_payload = {
                        "success": bool(rep.get("success")),
                        "msg": rep.get("message"),
                        "cmd": cmd,
                    }
                else:
                    result_payload = {"success": False, "msg": "invalid delete response", "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "SCRIPT_GET_REPORT":
            try:
                p = str(data_payload.get("path") or "")
                rep = await agent_get_report(p)
                if isinstance(rep, dict) and rep.get("ok") and "data" in rep:
                    result_payload = {"success": True, "data": rep["data"], "cmd": cmd}
                else:
                    err = rep.get("error") if isinstance(rep, dict) else "failed"
                    result_payload = {"success": False, "msg": str(err), "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "FS_LIST_DIRS":
            try:
                b = data_payload.get("base")
                resp = await agent_list_dirs(base=b)
                if hasattr(resp, "model_dump"):
                    d = resp.model_dump()
                else:
                    d = resp.dict()
                result_payload = {"success": True, "data": d, "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "FS_LIST":
            try:
                p = data_payload.get("path")
                resp = await agent_fs_list(path=p)
                if hasattr(resp, "model_dump"):
                    d = resp.model_dump()
                else:
                    d = resp.dict()
                result_payload = {"success": True, "data": d, "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "SCAN_COLLECT_SCRIPT":
            try:
                from collect_script_scanner import scan_collect_script_file

                script_path = str(data_payload.get("script_path") or "").strip()
                if not script_path:
                    result_payload = {"success": False, "msg": "script_path required", "cmd": cmd}
                elif not os.path.isfile(script_path):
                    result_payload = {
                        "success": False,
                        "msg": f"脚本不存在或不可读: {script_path}",
                        "cmd": cmd,
                    }
                else:
                    scanned = scan_collect_script_file(script_path)
                    result_payload = {"success": True, "data": scanned, "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "DATA_SYNC":
            try:
                rep = await _data_sync_via_minio_payload(data_payload)
                # 统一补齐 cmd 字段，便于平台侧排查
                result_payload = {**rep, "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "WEBRTC_OFFER":
            try:
                offer = WebRtcOffer(
                    sdp=str(data_payload.get("sdp") or ""),
                    type=str(data_payload.get("type") or "offer"),
                    camera_id=data_payload.get("camera_id"),
                )
                answer = await agent_webrtc_offer(offer)
                if isinstance(answer, dict) and answer.get("sdp") and answer.get("type"):
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            await _post_experiment_event(
                                client,
                                "primary_preview_up",
                                run_id=run_id,
                                scenario_id=scenario_id,
                                task_id=task_id,
                                job_id=job_id,
                                path="primary",
                                camera_id=offer.camera_id,
                            )
                    except Exception:
                        pass
                    result_payload = {
                        "success": True,
                        "msg": "ok",
                        "cmd": cmd,
                        "sdp": answer.get("sdp"),
                        "type": answer.get("type"),
                    }
                else:
                    # 透传更具体的原因，避免统一报 “invalid answer”
                    err = None
                    if isinstance(answer, dict):
                        err = answer.get("error") or answer.get("detail") or answer.get("message")
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            await _post_experiment_event(
                                client,
                                "primary_preview_fail",
                                run_id=run_id,
                                scenario_id=scenario_id,
                                task_id=task_id,
                                job_id=job_id,
                                path="primary",
                                camera_id=offer.camera_id,
                                success=False,
                            )
                    except Exception:
                        pass
                    result_payload = {"success": False, "msg": str(err or "invalid answer"), "cmd": cmd}
            except Exception as exc:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        await _post_experiment_event(
                            client,
                            "primary_preview_fail",
                            run_id=run_id,
                            scenario_id=scenario_id,
                            task_id=task_id,
                            job_id=job_id,
                            path="primary",
                            camera_id=data_payload.get("camera_id"),
                            success=False,
                            message=str(exc),
                        )
                except Exception:
                    pass
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        elif cmd == "STREAM_STATUS":
            try:
                if stream_manager is None:
                    result_payload = {"success": False, "msg": "stream_manager is None", "cmd": cmd}
                else:
                    result_payload = {"success": True, "data": stream_manager.get_status(), "cmd": cmd}
            except Exception as exc:
                result_payload = {"success": False, "msg": str(exc), "cmd": cmd}
        else:
            result_payload = {"success": False, "msg": f"Unsupported cmd: {cmd}", "cmd": cmd}

        async with _tunnel_command_lock:
            to_store = dict(result_payload)
            to_store.pop("duplicate", None)
            _tunnel_command_results[str(command_id)] = to_store
            if len(_tunnel_command_results) > 200:
                old_key = next(iter(_tunnel_command_results.keys()))
                _tunnel_command_results.pop(old_key, None)
                _tunnel_cmd_id_exec_locks.pop(old_key, None)

        try:
            await _tunnel_send_envelope(
                {
                    "ver": 1,
                    "type": "CMD_RESULT",
                    "agent_id": AGENT_ID,
                    "command_id": str(command_id),
                    "ts_ms": int(time.time() * 1000),
                    "payload": result_payload,
                }
            )
        except Exception:
            pass


async def _tunnel_client_loop() -> None:
    """
    Agent establishes outbound WS connection to platform:
      ws(s)://{platform}/api/agent/tunnel?agent_id={AGENT_ID}[&token=...]
    """
    global _tunnel_ws
    ws_base = _server_base_to_ws_base(SERVER_BASE)
    qparams = {"agent_id": AGENT_ID}
    if AGENT_TUNNEL_TOKEN:
        qparams["token"] = AGENT_TUNNEL_TOKEN
    ws_url = f"{ws_base}/api/agent/tunnel?{urlencode(qparams)}"
    safe_log = (
        ws_url.replace(AGENT_TUNNEL_TOKEN, "***") if AGENT_TUNNEL_TOKEN else ws_url
    )

    backoff_sec = 2.0
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, heartbeat=25, timeout=10) as ws:
                    _tunnel_ws = ws
                    _tunnel_connected_event.set()
                    print(f"[agent] tunnel connected: {safe_log}")
                    # Phase-2: advertise available camera_ids to platform.
                    await _send_stream_caps()
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await _handle_tunnel_text_message(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
        except Exception as exc:  # pragma: no cover
            _tunnel_connected_event.clear()
            try:
                _tunnel_ws = None
            except Exception:
                pass
            print(f"[agent] tunnel connect failed: {exc}; retrying in {backoff_sec:.0f}s")
            await asyncio.sleep(backoff_sec)
            backoff_sec = min(30.0, backoff_sec * 1.5)
        finally:
            _tunnel_ws = None
            _tunnel_connected_event.clear()
            _tunnel_drain_outq()


async def _post_log_to_server(
    client: httpx.AsyncClient,
    message: str,
    *,
    task_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> None:
    """将日志行上报到平台后端；失败不抛异常。"""
    # Phase-1: Prefer tunnel for logs to avoid HTTP HoL and keep ordering.
    if _tunnel_ws is not None and _tunnel_connected_event.is_set():
        try:
            await _tunnel_send_envelope(
                {
                    "ver": 1,
                    "type": "LOG",
                    "agent_id": AGENT_ID,
                    "command_id": None,
                    "ts_ms": int(time.time() * 1000),
                    "payload": {
                        "message": message,
                        "task_id": task_id,
                        "job_id": job_id,
                        "run_id": run_id,
                        "scenario_id": scenario_id,
                    },
                }
            )
            return
        except Exception:
            # Tunnel may be unstable; fallback to HTTP.
            pass
    try:
        await client.post(
            f"{SERVER_BASE}/api/script/agent-log",
            json={
                "message": message,
                "agent_id": AGENT_ID,
                "task_id": task_id,
                "job_id": job_id,
                "run_id": run_id,
                "scenario_id": scenario_id,
            },
        )
    except Exception as exc:  # pragma: no cover
        print(f"[agent] failed to forward log to server: {exc}")


async def _register_agent() -> None:
    """启动时向服务器平台注册 Agent 信息（可选）。设备「添加/连接」时平台会记录采集端 IP/端口，同步等能力主要依赖该项。"""
    payload = {
        "agent_id": AGENT_ID,
        "name": AGENT_NAME,
        "host": AGENT_HOST,
        "port": AGENT_PORT,
        "devices": DEVICES,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{SERVER_BASE}/api/agents/register", json=payload)
    except Exception as exc:  # pragma: no cover - 仅打印告警
        print(f"[agent] register failed: {exc}")


def _heartbeat_payload() -> dict:
    """与 register 一致，便于平台重启后仅靠心跳补登记。"""
    devs = DEVICES if isinstance(DEVICES, list) else []
    return {
        "agent_id": AGENT_ID,
        "online": True,
        "name": AGENT_NAME,
        "host": AGENT_HOST,
        "port": AGENT_PORT,
        "devices": devs,
    }


async def _heartbeat_loop() -> None:
    """心跳循环：标记在线；平台断连恢复后由后端用心跳自动补注册（无需重启 Agent）。"""
    while True:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{SERVER_BASE}/api/agents/heartbeat",
                    json=_heartbeat_payload(),
                )
                if r.status_code >= 400:
                    print(f"[agent] heartbeat HTTP {r.status_code}: {(r.text or '')[:200]}")
        except Exception as exc:  # pragma: no cover
            print(f"[agent] heartbeat failed: {exc}")
        await asyncio.sleep(15)


@app.on_event("startup")
async def on_startup() -> None:
    global _tunnel_out_worker_task, _tunnel_heartbeat_task, _joint_sampler_task, _ft_sampler_task
    _install_asyncio_webrtc_exception_filter()
    await _register_agent()
    # 先启动 ROS2 相机节点（内含 RobotStateBridge），再跑关节/ft 采样，避免首轮回退 echo
    try:
        if stream_manager is not None:
            stream_manager.start()
    except Exception as exc:  # pragma: no cover
        print(f"[agent] stream_manager start failed: {exc}")
    asyncio.create_task(_heartbeat_loop())
    _joint_sampler_task = asyncio.create_task(_joint_sampler_loop())
    _ft_sampler_task = asyncio.create_task(_ft_sampler_loop())
    _tunnel_out_worker_task = asyncio.create_task(_tunnel_outbound_worker())
    _tunnel_heartbeat_task = asyncio.create_task(_tunnel_heartbeat_loop())
    asyncio.create_task(_tunnel_client_loop())


@app.get("/api/agent/info", response_model=AgentInfoResponse)
async def agent_get_info() -> AgentInfoResponse:
    """采集端自描述接口：用于中心平台连接探测/握手。"""
    info = await asyncio.to_thread(_gather_agent_info_sync)
    return info


@app.get("/api/agent/health")
async def agent_health() -> dict:
    return {
        "ok": True,
        "agent_id": AGENT_ID,
        "agent_name": AGENT_NAME,
        "version": AGENT_VERSION,
    }


@app.get("/api/agent/fs/list-dirs", response_model=AgentListDirsResponse)
async def agent_list_dirs(base: Optional[str] = None) -> AgentListDirsResponse:
    """
    在采集端本机列出子目录，返回结构与平台 /api/fs/list-dirs 兼容：
    {
      "ok": true,
      "data": { "base": "...", "dirs": ["sub1", "sub2"] }
    }
    """
    try:
        target_path = _resolve_agent_fs_path(base)

        if not os.path.exists(target_path):
            return AgentListDirsResponse(ok=False, error=f"路径不存在: {target_path}")
        if not os.path.isdir(target_path):
            return AgentListDirsResponse(ok=False, error=f"路径不是目录: {target_path}")

        dirs: list[str] = []
        try:
            with os.scandir(target_path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        name = entry.name
                        safe_name = name.encode("utf-8", "replace").decode("utf-8")
                        dirs.append(safe_name)
        except PermissionError as e:
            return AgentListDirsResponse(ok=False, error=f"没有权限访问该目录: {e}")
        except Exception as e:
            return AgentListDirsResponse(ok=False, error=f"读取目录失败: {e}")

        dirs.sort()
        return AgentListDirsResponse(ok=True, data={"base": target_path, "dirs": dirs})
    except Exception as e:  # pragma: no cover
        return AgentListDirsResponse(ok=False, error=f"服务器错误: {e}")


def _format_mtime_utc_iso(st_mtime: float) -> str:
    """FS_LIST mtime：Unix 秒 → UTC ISO8601（Z），避免采集端本地时区与平台 since_ms(UTC) 比较错位。"""
    return datetime.fromtimestamp(st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.get("/api/agent/fs/list", response_model=AgentFsListResponse)
async def agent_fs_list(path: Optional[str] = None) -> AgentFsListResponse:
    try:
        target = _resolve_agent_fs_path(path)
        if not os.path.isdir(target):
            return AgentFsListResponse(ok=False, error="路径不是目录")
        items: list[dict] = []
        with os.scandir(target) as entries:
            for entry in entries:
                try:
                    name = entry.name.encode("utf-8", "replace").decode("utf-8")
                    stat = entry.stat()
                    mtime = _format_mtime_utc_iso(stat.st_mtime) if stat else None
                    if entry.is_dir():
                        items.append({"name": name, "type": "dir", "mtime": mtime})
                    else:
                        items.append({"name": name, "type": "file", "size": stat.st_size, "mtime": mtime})
                except (OSError, PermissionError):
                    continue
        items.sort(key=lambda x: (x["type"] != "dir", x["name"].lower()))
        return AgentFsListResponse(ok=True, data={"path": target, "items": items})
    except Exception as e:
        return AgentFsListResponse(ok=False, error=str(e))


@app.get("/api/agent/streams")
async def list_streams() -> list[dict]:
    """
    列出采集端可用相机流（由采集端本机 ROS2 发现并订阅）。
    返回结构与平台 /api/stream/list 兼容。
    """
    if stream_manager is None:
        print("[agent] list_streams: stream_manager is None")
        return []
    try:
        stream_manager.refresh_topics()
        try:
            print("[agent] list_streams: topic_mapping =", getattr(stream_manager, "topic_mapping", {}))
        except Exception:
            pass
        cameras = []
        sorted_topics = sorted(stream_manager.topic_mapping.items(), key=lambda x: x[0])
        for cam_id, topic in sorted_topics:
            cameras.append(
                {
                    "id": cam_id,
                    "name": f"{cam_id} ({topic})",
                    # 注意：此 url 为 Agent 内部路径，平台会转发为 /api/stream/{id}?device_id=...
                    "url": f"/api/agent/stream/{cam_id}",
                }
            )
        return cameras
    except Exception as exc:
        print(f"[agent] list_streams failed: {exc}")
        return []


@app.get("/api/agent/stream/{camera_id}")
async def agent_camera_stream(camera_id: str):
    """
    采集端相机 MJPEG 流。
    """
    if stream_manager is None:
        return StreamingResponse(
            iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"]),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    return StreamingResponse(
        stream_manager.get_frame_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/agent/stream-status")
async def agent_stream_status() -> dict:
    if stream_manager is None:
        return {"ok": False, "error": "stream_manager is None"}
    try:
        return stream_manager.get_status()
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}




@app.post("/api/agent/device/test-connection", response_model=DeviceTestResponse)
async def device_test_connection(_: DeviceTestRequest) -> DeviceTestResponse:
    """
    在采集端本机执行一次 ROS2 连接测试。

    当前实现：
    - 执行 `ros2 node list` 和 `ros2 topic list`
    - 返回节点数与话题数，错误信息写在 msg 中
    """
    try:
        env = os.environ.copy()
        nodes_out = subprocess.check_output(
            ["ros2", "node", "list"], env=env, timeout=10
        ).decode(errors="replace")
        topics_out = subprocess.check_output(
            ["ros2", "topic", "list"], env=env, timeout=10
        ).decode(errors="replace")

        nodes = [ln.strip() for ln in nodes_out.splitlines() if ln.strip()]
        topics = [ln.strip() for ln in topics_out.splitlines() if ln.strip()]

        return DeviceTestResponse(
            ok=True,
            node_count=len(nodes),
            topic_count=len(topics),
        )
    except subprocess.CalledProcessError as exc:
        return DeviceTestResponse(
            ok=False,
            msg=exc.stderr.decode(errors="replace") if exc.stderr else str(exc),
        )
    except Exception as exc:  # pragma: no cover - 仅打印告警
        return DeviceTestResponse(ok=False, msg=str(exc))


@app.post("/api/agent/collect/start", response_model=CollectSimpleResponse)
async def start_collect(payload: CollectStartRequest) -> CollectSimpleResponse:
    """
    在采集端本地启动采集脚本。

    注意：
    - 脚本路径按需修改为当前设备上的实际路径。
    - 推荐将业务逻辑封装到 shell 或 python 脚本中，Agent 只负责拉起。
    """

    global _process
    if _process and _process.poll() is None:
        return CollectSimpleResponse(ok=False, msg="collect already running")

    # 平台可下发脚本路径：优先使用（用于设备侧自定义脚本）
    script_path = (payload.script_path or "").strip()
    if not script_path:
        # 根据 camera_data_format 选择采集脚本：
        # - 默认为压缩脚本（节省带宽与存储）
        # - 当显式指定为“原始”时，使用原始脚本
        if (payload.camera_data_format or "").strip() == "原始":
            script_path = "/home/rm/IDE/eai-ide/scripts/rm_robot/collect_data.sh"
        else:
            script_path = "/home/rm/IDE/eai-ide/scripts/rm_robot/collect_data_compress.sh"

    args = payload.args if isinstance(payload.args, list) and payload.args else ["-t", str(payload.duration_sec), "-o", payload.storage_path]

    env = os.environ.copy()
    if not payload.env or "VALIDATION_CONFIG" not in payload.env:
        env.pop("VALIDATION_CONFIG", None)
    if payload.env:
        env.update(payload.env)
    if payload.task_id:
        env.setdefault("EAI_TASK_ID", payload.task_id)
    if payload.job_id:
        env.setdefault("EAI_JOB_ID", payload.job_id)
    if payload.run_id:
        env.setdefault("EAI_RUN_ID", payload.run_id)
    if payload.scenario_id:
        env.setdefault("EAI_SCENARIO_ID", payload.scenario_id)
    env.setdefault("EAI_AGENT_ID", AGENT_ID)
    # 供质检/报告兜底使用：即使采集脚本没传/没打印，也可由 Agent 侧生成报告
    env.setdefault("EAI_COLLECT_DURATION_SEC", str(int(payload.duration_sec or 30)))
    cam_fmt = (payload.camera_data_format or "").strip()
    env.setdefault("EAI_COLLECT_CAMERA_MODE", "compressed" if ("压缩" in cam_fmt or cam_fmt == "" ) else "raw")

    _process = subprocess.Popen(
        [script_path] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await _post_experiment_event(
                client,
                "local_start",
                run_id=payload.run_id,
                scenario_id=payload.scenario_id,
                task_id=payload.task_id,
                job_id=payload.job_id,
                command_id=_collect_execution_command_id(
                    run_id=payload.run_id,
                    task_id=payload.task_id,
                    job_id=payload.job_id,
                ),
                success=True,
                device_open_ok=True,
                script_path=script_path,
            )
    except Exception:
        pass

    # 后台读取日志并转发到平台
    if _process.stdout is not None:
        # 保存 OUTPUT_PATH（如果脚本打印了），用于采集完成时兜底补发
        shared: dict[str, Optional[str]] = {
            "output_path": None,
            "duration_sec": str(int(payload.duration_sec or 30)),
            "mode": ("compressed" if ("压缩" in (payload.camera_data_format or "")) else "raw"),
            "validation_config": env.get("VALIDATION_CONFIG"),
        }
        asyncio.create_task(
            _consume_stdout(
                _process,
                payload.task_id,
                payload.job_id,
                payload.run_id,
                payload.scenario_id,
                shared,
            )
        )
        asyncio.create_task(
            _watch_collect_process(
                _process,
                payload.task_id,
                payload.job_id,
                payload.run_id,
                payload.scenario_id,
                payload.storage_path,
                shared,
            )
        )

    return CollectSimpleResponse(ok=True)


async def _watch_collect_process(
    proc: subprocess.Popen[str],
    task_id: Optional[str],
    job_id: Optional[str],
    run_id: Optional[str],
    scenario_id: Optional[str],
    storage_path: str,
    shared: dict[str, Optional[str]],
) -> None:
    """等待采集脚本退出，并将完成状态转发到平台，供前端实时页切换状态。"""
    try:
        rc = await asyncio.to_thread(proc.wait)
    except Exception as exc:  # pragma: no cover
        print(f"[agent] wait collect process failed: {exc}")
        return

    def _pick_latest_bag_target(base: str) -> str:
        """
        从 storage_path（可能是上层目录）中定位本次采集产物：
        - 若 base 是文件：直接返回
        - 若 base 是目录：优先找最近修改的 *.mcap/*.db3 文件；找不到则尝试找最近修改的“子目录内含 bag 文件”的目录
        """
        b = (base or "").strip()
        if not b:
            return b
        try:
            if os.path.isfile(b):
                return b
            if not os.path.isdir(b):
                return b
        except OSError:
            return b

        best_file: tuple[float, str] | None = None
        try:
            for root, _, files in os.walk(b):
                for name in files:
                    low = name.lower()
                    if not (low.endswith(".mcap") or low.endswith(".db3")):
                        continue
                    full = os.path.join(root, name)
                    try:
                        ts = float(os.path.getmtime(full))
                    except OSError:
                        ts = 0.0
                    if best_file is None or ts > best_file[0]:
                        best_file = (ts, full)
        except Exception:
            best_file = None
        if best_file is not None:
            return best_file[1]

        # 目录型 episode：选最近修改且包含 bag 文件的子目录
        best_dir: tuple[float, str] | None = None
        try:
            for name in os.listdir(b):
                full = os.path.join(b, name)
                if not os.path.isdir(full):
                    continue
                has_bag = False
                try:
                    for fn in os.listdir(full):
                        low = fn.lower()
                        if low.endswith(".mcap") or low.endswith(".db3"):
                            if os.path.isfile(os.path.join(full, fn)):
                                has_bag = True
                                break
                except OSError:
                    has_bag = False
                if not has_bag:
                    continue
                try:
                    ts = float(os.path.getmtime(full))
                except OSError:
                    ts = 0.0
                if best_dir is None or ts > best_dir[0]:
                    best_dir = (ts, full)
        except Exception:
            best_dir = None
        return best_dir[1] if best_dir is not None else b

    def _try_build_validation_report_json(
        *,
        bag_target: str,
        duration_sec: str,
        mode: str,
        validation_config: Optional[str] = None,
    ) -> Optional[str]:
        here = os.path.dirname(os.path.abspath(__file__))
        vb = os.path.join(here, "validate_bag.py")
        if not os.path.isfile(vb):
            return None
        dur = (duration_sec or "").strip() or "30"
        m = (mode or "").strip().lower() or "raw"
        run_env = os.environ.copy()
        vc = (validation_config or "").strip()
        if vc:
            run_env["VALIDATION_CONFIG"] = vc
        else:
            run_env.pop("VALIDATION_CONFIG", None)
        try:
            r = subprocess.run(
                [sys.executable, vb, bag_target, dur, m],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=here,
                env=run_env,
            )
            if r.returncode != 0:
                return None
            out = (r.stdout or "").strip()
            if not out:
                return None
            last = out.splitlines()[-1].strip()
            if not last.startswith("{"):
                return None
            json.loads(last)
            return last
        except Exception:
            return None

    # 统一发出“脚本结束”标志，前端依赖该文本切换到 DONE/FAILED
    async with httpx.AsyncClient(timeout=5.0) as client:
        # 采集脚本可能只输出“上层目录”；这里兜底扫描 storage_path，定位最新 episode/bag（成熟做法：以文件系统为准）
        raw_out = (shared.get("output_path") or "").strip() or ""
        output_path = raw_out or storage_path
        final_target = _pick_latest_bag_target(output_path or storage_path)
        # 始终补发/覆盖 OUTPUT_PATH（让平台能稳定得到精确路径）
        await _post_log_to_server(
            client,
            f"OUTPUT_PATH: {final_target}",
            task_id=task_id,
            job_id=job_id,
            run_id=run_id,
            scenario_id=scenario_id,
        )

        # 同步补发文件大小，便于平台在“采集端落盘、平台不可见路径”场景稳定写入 file_size_bytes
        size_bytes = 0
        try:
            if final_target and os.path.isfile(final_target):
                size_bytes = int(os.path.getsize(final_target))
        except Exception:
            size_bytes = 0
        await _post_log_to_server(
            client,
            f"OUTPUT_SIZE_BYTES: {size_bytes}",
            task_id=task_id,
            job_id=job_id,
            run_id=run_id,
            scenario_id=scenario_id,
        )

        # 若采集成功：生成并上报质检报告 JSON（让质量页不依赖接口兜底/跨标签缓存）
        if rc == 0:
            js = _try_build_validation_report_json(
                bag_target=final_target,
                duration_sec=(shared.get("duration_sec") or "").strip() or "30",
                mode=(shared.get("mode") or "").strip() or "raw",
                validation_config=(shared.get("validation_config") or "").strip() or None,
            )
            if js:
                await _post_log_to_server(
                    client,
                    f"EAI_VALIDATION_REPORT_JSON:{js}",
                    task_id=task_id,
                    job_id=job_id,
                    run_id=run_id,
                    scenario_id=scenario_id,
                )

        await _post_log_to_server(
            client,
            f"Script finished with exit code {rc}",
            task_id=task_id,
            job_id=job_id,
            run_id=run_id,
            scenario_id=scenario_id,
        )
        await _post_experiment_event(
            client,
            "local_stop",
            run_id=run_id,
            scenario_id=scenario_id,
            task_id=task_id,
            job_id=job_id,
            command_id=_collect_execution_command_id(run_id=run_id, task_id=task_id, job_id=job_id),
            success=rc == 0,
            output_path=output_path,
            exit_code=rc,
        )


@app.post("/api/agent/collect/stop", response_model=CollectSimpleResponse)
async def stop_collect() -> CollectSimpleResponse:
    """停止当前采集脚本。"""
    ok, msg = await _stop_collect_idempotent()
    return CollectSimpleResponse(ok=bool(ok), msg=msg if msg else None)


@app.post("/api/agent/device/launch", response_model=DeviceControlResponse)
async def device_launch(payload: DeviceLaunchRequest) -> DeviceControlResponse:
    """
    启动设备相关的 ROS2 节点/进程。

    平台会将设备的 launch_config（脚本路径/参数/环境变量）转发到此接口。
    """
    #print("[agent] device_launch DEBUG: using local modified agent_main.py")
    global _device_process
    if _device_process and _device_process.poll() is None:
        return DeviceControlResponse(ok=False, msg="device already launched")

    script_path = payload.script_path
    args: list[str] = [script_path]
    if payload.script_args:
        # 按空格拆分参数，保持与手动输入一致
        args.extend(shlex.split(payload.script_args))

    # 调试信息：打印收到的启动参数与最终命令
    print(f"[agent] device_launch payload: script_path={payload.script_path}, script_args={payload.script_args}")
    print(f"[agent] device_launch command: {' '.join(args)}")

    env = os.environ.copy()
    # 统一确保 /opt/ros/humble/bin 在 PATH 中（基础环境）
    ros2_bin = "/opt/ros/humble/bin"
    ros2_lib = "/opt/ros/humble/lib"
    base_path = env.get("PATH", "")
    if ros2_bin not in base_path.split(":"):
        env["PATH"] = f"{ros2_bin}:{base_path}" if base_path else ros2_bin

    # 同时补充 LD_LIBRARY_PATH，避免 librcl_action.so 之类找不到
    base_ld = env.get("LD_LIBRARY_PATH", "")
    if ros2_lib not in base_ld.split(":"):
        env["LD_LIBRARY_PATH"] = f"{ros2_lib}:{base_ld}" if base_ld else ros2_lib

    # 为避免 /home/ubuntu/.ros 无权限问题，默认将 ROS 日志目录指向 /tmp 下的可写路径
    env.setdefault("ROS_LOG_DIR", "/tmp/ros2_logs")

    # 合并平台下发的 env，特别处理 PATH，避免覆盖掉 ros2 路径
    if payload.env:
        for k, v in payload.env.items():
            kk = str(k)
            vv = "" if v is None else str(v)
            if kk == "PATH":
                merged = vv
                # 如果平台的 PATH 里不含 ros2_bin，就加到最前面
                if ros2_bin not in merged.split(":"):
                    merged = f"{ros2_bin}:{merged}" if merged else ros2_bin
                env["PATH"] = merged
            elif kk == "LD_LIBRARY_PATH":
                merged_ld = vv
                if ros2_lib not in merged_ld.split(":"):
                    merged_ld = f"{ros2_lib}:{merged_ld}" if merged_ld else ros2_lib
                env["LD_LIBRARY_PATH"] = merged_ld
            else:
                env[kk] = vv

    # 调试信息：关键环境变量（此 env 将传给子进程）
    print(f"[agent] device_launch env ROS_DOMAIN_ID={env.get('ROS_DOMAIN_ID')}, PATH={env.get('PATH')}")

    # 与平台原有 launch 逻辑保持一致：启动脚本后短暂观察日志/退出码，给出成功或失败反馈
    error_patterns = [
        re.compile(r"\bTraceback\b"),
        re.compile(r"\bSerialException\b"),
        re.compile(r"\[ERROR\]"),
        re.compile(r"No such file or directory", re.IGNORECASE),
        re.compile(r"Failed to open serial", re.IGNORECASE),
    ]

    def _looks_like_failure(output: str) -> tuple[bool, str | None]:
        for pattern in error_patterns:
            if pattern.search(output):
                return True, f"检测到异常输出: {pattern.pattern}"
        return False, None

    def _read_tail_bytes(file_path: str, max_bytes: int) -> str:
        try:
            with open(file_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
                data = f.read()
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _wait_and_read_tail(proc: subprocess.Popen[str], file_path: str, timeout_sec: float, max_bytes: int) -> str:
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        return _read_tail_bytes(file_path, max_bytes)

    log_path = os.path.join(
        tempfile.gettempdir(),
        f"agent_device_launch_{int(time.time())}.log",
    )

    try:
        log_file = open(log_path, "w", encoding="utf-8", errors="replace")
        print(f"[agent] device_launch logging to {log_path}")
        proc = subprocess.Popen(
            args,
            env=env,
            cwd=os.path.dirname(script_path) or None,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log_file.close()
    except Exception as exc:
        print(f"[agent] device_launch Popen failed: {exc}")
        return DeviceControlResponse(ok=False, msg=str(exc))

    # 短暂等待脚本启动，并读取一点日志判断是否明显失败
    startup_output = await asyncio.to_thread(_wait_and_read_tail, proc, log_path, 3.0, 64_000)
    print(f"[agent] device_launch startup_output (truncated): {startup_output[-200:]}")
    failed_by_output, failed_reason = _looks_like_failure(startup_output)

    if proc.poll() is not None:
        rc = proc.returncode
        ok = (rc == 0) and (not failed_by_output)
        if ok:
            _device_process = proc
            print(f"[agent] device_launch finished quickly with rc=0, treated as success")
            return DeviceControlResponse(ok=True, msg="启动成功")
        print(f"[agent] device_launch finished quickly with rc={rc}, failed")
        return DeviceControlResponse(
            ok=False,
            msg=f"启动脚本退出 (代码 {rc})；日志片段: {startup_output[-2000:]}",
        )

    if failed_by_output:
        print(f"[agent] device_launch detected failure by output: {failed_reason}")
        return DeviceControlResponse(
            ok=False,
            msg=f"{failed_reason or '检测到异常输出'}；日志片段: {startup_output[-2000:]}",
        )

    # 进程仍在运行且未检测到明显错误，认为启动成功
    _device_process = proc
    return DeviceControlResponse(ok=True, msg="连接成功：启动脚本已运行")


@app.post("/api/agent/device/stop", response_model=DeviceControlResponse)
async def device_stop(payload: DeviceLaunchRequest | None = None) -> DeviceControlResponse:
    """
    停止设备相关进程。

    可以通过：
    - 直接终止在本 Agent 内记录的 _device_process，或
    - 由平台下发专门的停止脚本路径（payload.script_path），在此执行。
    """

    global _device_process

    # 优先使用专门的停止脚本（如果提供）
    if payload and payload.script_path:
        script_path = payload.script_path
        args: list[str] = [script_path]
        if payload.script_args:
            args.extend(payload.script_args.split())
        env = os.environ.copy()
        ros2_bin = "/opt/ros/humble/bin"
        ros2_lib = "/opt/ros/humble/lib"
        base_path = env.get("PATH", "")
        if ros2_bin not in base_path.split(":"):
            env["PATH"] = f"{ros2_bin}:{base_path}" if base_path else ros2_bin
        base_ld = env.get("LD_LIBRARY_PATH", "")
        if ros2_lib not in base_ld.split(":"):
            env["LD_LIBRARY_PATH"] = f"{ros2_lib}:{base_ld}" if base_ld else ros2_lib
        env.setdefault("ROS_LOG_DIR", "/tmp/ros2_logs")

        if payload.env:
            for k, v in payload.env.items():
                kk = str(k)
                vv = "" if v is None else str(v)
                if kk == "PATH":
                    merged = vv
                    if ros2_bin not in merged.split(":"):
                        merged = f"{ros2_bin}:{merged}" if merged else ros2_bin
                    env["PATH"] = merged
                elif kk == "LD_LIBRARY_PATH":
                    merged_ld = vv
                    if ros2_lib not in merged_ld.split(":"):
                        merged_ld = f"{ros2_lib}:{merged_ld}" if merged_ld else ros2_lib
                    env["LD_LIBRARY_PATH"] = merged_ld
                else:
                    env[kk] = vv
        try:
            subprocess.Popen(
                args,
                env=env,
                cwd=os.path.dirname(script_path) or None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            return DeviceControlResponse(ok=False, msg=str(exc))

    # 同时尝试终止已记录的启动进程
    if _device_process and _device_process.poll() is None:
        try:
            _device_process.terminate()
        except Exception:
            pass

    return DeviceControlResponse(ok=True)


async def _consume_stdout(
    proc: subprocess.Popen[str],
    task_id: Optional[str],
    job_id: Optional[str],
    run_id: Optional[str],
    scenario_id: Optional[str],
    shared: dict[str, Optional[str]],
) -> None:
    """读取子进程 stdout，并转发到平台后端。"""
    assert proc.stdout is not None
    loop = asyncio.get_event_loop()
    # 复用一个 HTTP 客户端，避免每行都新建连接
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                break
            text = line.rstrip()
            if not text:
                continue

            # 尝试捕获 OUTPUT_PATH（供完成时兜底）
            if "OUTPUT_PATH:" in text:
                try:
                    shared["output_path"] = text.split("OUTPUT_PATH:", 1)[1].strip() or None
                except Exception:
                    pass

            # 本地输出方便调试
            print(f"[agent-script] {text}")
            # 发送到平台后端，让 /api/script/ws 的 websocket 也能看到
            await _post_log_to_server(
                client,
                text,
                task_id=task_id,
                job_id=job_id,
                run_id=run_id,
                scenario_id=scenario_id,
            )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("agent_main:app", host=AGENT_HOST, port=AGENT_PORT, reload=False)
