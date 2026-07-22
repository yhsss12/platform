"""
从采集 bash 脚本静态解析：默认频率阈值 + 频率检查话题清单。
与 agent/collect_script_scanner.py 保持逻辑一致。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_FREQ_ARRAY_NAMES = (
    "CAMERA_COLOR_TOPICS",
    "DEPTH_TOPICS",
    "CAMERA_TOPICS",
    "GRIPPER_TOPICS",
    "JOINT_TOPICS",
    "FORCE_TOPICS",
)

_MIN_VAR_KEYS = (
    "MIN_CAMERA_FREQ",
    "MIN_GRIPPER_FREQ",
    "MIN_JOINT_FREQ",
    "MIN_FORCE_FREQ",
)


def _strip_bash_comments(text: str) -> str:
    out: List[str] = []
    for raw in text.splitlines():
        line = raw
        if re.match(r"^\s*#", line):
            continue
        if "#" in line:
            in_quote: Optional[str] = None
            buf: List[str] = []
            for ch in line:
                if ch in "\"'" and in_quote is None:
                    in_quote = ch
                elif ch == in_quote:
                    in_quote = None
                elif ch == "#" and in_quote is None:
                    break
                buf.append(ch)
            line = "".join(buf)
        out.append(line)
    return "\n".join(out)


def _parse_min_defaults(text: str) -> Dict[str, float]:
    defaults: Dict[str, float] = {
        "MIN_CAMERA_FREQ": 25.0,
        "MIN_GRIPPER_FREQ": 10.0,
        "MIN_JOINT_FREQ": 10.0,
        "MIN_FORCE_FREQ": 10.0,
    }
    for key in _MIN_VAR_KEYS:
        m = re.search(rf"^{re.escape(key)}=([0-9.]+)\s*$", text, re.M)
        if m:
            try:
                defaults[key] = float(m.group(1))
            except ValueError:
                pass
    return defaults


def _parse_bash_array_body(text: str, name: str) -> List[str]:
    m = re.search(rf"{re.escape(name)}\s*=\s*\((.*?)\)", text, re.S)
    if not m:
        return []
    body = m.group(1)
    entries: List[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sm in re.finditer(r'"([^"]*)"|\'([^\']*)\'', line):
            val = (sm.group(1) or sm.group(2) or "").strip()
            if val:
                entries.append(val)
    return entries


def _infer_min_for_array(array_name: str, defaults: Dict[str, float], topic: str) -> float:
    an = array_name.upper()
    tl = topic.lower()
    if "FORCE" in an or "six_force" in tl:
        return defaults["MIN_FORCE_FREQ"]
    if "GRIPPER" in an or "gripper" in tl:
        return defaults["MIN_GRIPPER_FREQ"]
    if "JOINT" in an or "joint_states" in tl or "/gello/" in tl:
        return defaults["MIN_JOINT_FREQ"]
    if "DEPTH" in an or "depth" in tl or "compresseddepth" in tl:
        return defaults["MIN_CAMERA_FREQ"]
    if "CAMERA" in an or "COLOR" in an or "camera" in tl:
        return defaults["MIN_CAMERA_FREQ"]
    return defaults["MIN_CAMERA_FREQ"]


def _infer_group(array_name: str, topic: str) -> str:
    an = array_name.upper()
    tl = topic.lower()
    if "FORCE" in an or "six_force" in tl:
        return "force"
    if "GRIPPER" in an or "gripper" in tl:
        return "gripper"
    if "JOINT" in an or "joint" in tl:
        return "joint"
    if "DEPTH" in an or "depth" in tl:
        return "depth"
    if "color" in tl or "COLOR" in an:
        return "camera_color"
    return "other"


def _split_topic_label(entry: str) -> Tuple[Optional[str], str]:
    s = entry.strip()
    if not s.startswith("/"):
        return None, s
    if ":" in s:
        topic, label = s.split(":", 1)
        return topic.strip(), label.strip()
    parts = [p for p in s.split("/") if p]
    label = parts[-1] if parts else s
    return s, label


def scan_collect_script_content(content: str, *, script_path: str = "") -> Dict[str, Any]:
    text = _strip_bash_comments(content or "")
    defaults = _parse_min_defaults(text)

    by_topic: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def add_topic(topic: str, label: str, *, array_name: str) -> None:
        if not topic.startswith("/"):
            return
        min_hz = _infer_min_for_array(array_name, defaults, topic)
        group = _infer_group(array_name, topic)
        if topic not in by_topic:
            by_topic[topic] = {
                "topic": topic,
                "label": label,
                "group": group,
                "default_min_hz": min_hz,
            }
            order.append(topic)
        else:
            existing = by_topic[topic]
            if not existing.get("label") and label:
                existing["label"] = label

    found_arrays: List[str] = []
    for name in _FREQ_ARRAY_NAMES:
        if _parse_bash_array_body(text, name):
            found_arrays.append(name)

    if not found_arrays:
        for m in re.finditer(r"(\w+_TOPICS)\s*=\s*\(", text):
            name = m.group(1)
            if name in ("TOPICS", "OPTIONAL_TOPICS", "FINAL_TOPICS"):
                continue
            if name not in found_arrays:
                found_arrays.append(name)

    for array_name in found_arrays:
        for entry in _parse_bash_array_body(text, array_name):
            topic, label = _split_topic_label(entry)
            if topic:
                add_topic(topic, label, array_name=array_name)

    topics = [by_topic[t] for t in order]
    return {
        "script_path": script_path,
        "defaults": {
            "camera_hz": defaults["MIN_CAMERA_FREQ"],
            "gripper_hz": defaults["MIN_GRIPPER_FREQ"],
            "joint_hz": defaults["MIN_JOINT_FREQ"],
            "force_hz": defaults["MIN_FORCE_FREQ"],
        },
        "topics": topics,
    }


def scan_collect_script_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return scan_collect_script_content(content, script_path=path)
