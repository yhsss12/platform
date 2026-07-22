import os
import sys
import json
import subprocess
import re
import time
from typing import Any, Dict, Optional


def bag_container_dir(bag_path: str) -> str:
    """episode 目录：path 为具体 bag 文件时取其父目录（用于定位目录型 bag）。"""
    try:
        p = os.path.abspath(bag_path)
        if os.path.isdir(p):
            return p
        parent = os.path.dirname(p)
        return parent if parent else "."
    except OSError:
        parent = os.path.dirname(os.path.abspath(bag_path))
        return parent if parent else "."


def parse_ros2_bag_info(bag_path):
    try:
        # Use -y for yaml output if available? Humble doesn't support -y properly sometimes.
        # But let's stick to text parsing which works on Humble.
        result = subprocess.run(
            ["ros2", "bag", "info", bag_path],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running ros2 bag info: {e}", file=sys.stderr)
        return None

    info = {"duration": 0.0, "size_bytes": 0, "topics": {}}

    # Parse Duration: 12.3s
    duration_match = re.search(r"Duration:\s+([\d\.]+s)", output)
    if duration_match:
        try:
            info["duration"] = float(duration_match.group(1).replace("s", ""))
        except Exception:
            pass

    # Parse Size: calculate directory size
    total_size = 0
    if os.path.isdir(bag_path):
        for dirpath, _, filenames in os.walk(bag_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    elif os.path.exists(bag_path):
        total_size = os.path.getsize(bag_path)
    info["size_bytes"] = total_size

    # Parse Topics
    # Topic: /topic_name | Type: type/name | Count: 123 | Serialization Format: cdr
    topic_lines = re.findall(
        r"Topic:\s+([^\s]+)\s+\|\s+Type:\s+[^\s]+\s+\|\s+Count:\s+(\d+)", output
    )
    for topic, count in topic_lines:
        info["topics"][topic] = int(count)

    return info


def validate(bag_path, expected_duration, mode="raw") -> Optional[Dict[str, Any]]:
    """
    在内存中生成校验报告并返回 dict；不落盘 json。
    expected_duration 预留与 CLI 兼容，当前校验逻辑未使用。
    """
    info = parse_ros2_bag_info(bag_path)
    if not info:
        print("Failed to get bag info", file=sys.stderr)
        return None

    report = {
        "episode_dir": str(bag_path),
        "topics_info": info["topics"],
        "duration_seconds": info["duration"],
        "file_size_bytes": info["size_bytes"],
        "file_size_mb": info["size_bytes"] / (1024 * 1024),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "check_result": {"missing_topics": [], "empty_topics": [], "is_valid": True},
        "frequency_analysis": {
            "total_topics": 0,
            "compliant_topics": 0,
            "critical_topics": [],
            "warning_topics": [],
            "frequency_details": {},
            "summary": "",
        },
    }

    def _default_expected_topics(m: str) -> list[str]:
        # 兼容旧脚本：当采集脚本未显式传入「本次录制的话题列表」时，回退到历史默认值
        base = [
            "/left_gripper_cmd",
            "/left_gripper_state",
            "/left/rm_driver/movej_canfd_custom_cmd",
            "/left/joint_states",
            "/right_gripper_cmd",
            "/right_gripper_state",
            "/right/joint_states",
            "/right/rm_driver/movej_canfd_custom_cmd",
            "/camera1/camera1/depth/image_rect_raw",
            "/camera2/camera2/depth/image_rect_raw",
            "/camera3/camera3/depth/image_rect_raw",
        ]
        if (m or "").strip().lower() == "compressed":
            base.extend(
                [
                    "/camera1/camera1/color/image_raw/compressed",
                    "/camera2/camera2/color/image_raw/compressed",
                    "/camera3/camera3/color/image_raw/compressed",
                ]
            )
        else:
            base.extend(
                [
                    "/camera1/camera1/color/image_raw",
                    "/camera2/camera2/color/image_raw",
                    "/camera3/camera3/color/image_raw",
                ]
            )
        return base

    def _load_expected_topics_from_env(m: str) -> Optional[list[str]]:
        """
        让质检脚本与采集脚本对齐：
        - EAI_EXPECTED_TOPICS_JSON:
            1) JSON 数组：["/a","/b",...]
            2) JSON 对象：{"raw":[...], "compressed":[...]} 或 {"topics":[...]}
        - EAI_EXPECTED_TOPICS:
            逗号/空格/换行分隔的 topic 列表
        """
        raw_json = (os.environ.get("EAI_EXPECTED_TOPICS_JSON") or "").strip()
        if raw_json:
            try:
                obj = json.loads(raw_json)
                if isinstance(obj, list):
                    out = [str(x).strip() for x in obj if str(x).strip()]
                    return out or None
                if isinstance(obj, dict):
                    mm = (m or "").strip().lower()
                    cand = obj.get(mm) if mm else None
                    if cand is None:
                        cand = obj.get("topics")
                    if isinstance(cand, list):
                        out = [str(x).strip() for x in cand if str(x).strip()]
                        return out or None
            except Exception:
                print("Failed to parse EAI_EXPECTED_TOPICS_JSON", file=sys.stderr)

        raw_list = (os.environ.get("EAI_EXPECTED_TOPICS") or "").strip()
        if raw_list:
            parts = re.split(r"[,\s]+", raw_list.strip())
            out = [p.strip() for p in parts if p.strip()]
            return out or None
        return None

    expected_topics = _load_expected_topics_from_env(mode) or _default_expected_topics(mode)

    # Check for custom frequency standards from environment variable
    custom_standards = {}
    validation_config_env = os.environ.get("VALIDATION_CONFIG")
    if validation_config_env:
        try:
            custom_standards = json.loads(validation_config_env)
            print(
                f"Loaded custom validation standards: {custom_standards}", file=sys.stderr
            )
        except json.JSONDecodeError:
            print(
                "Failed to parse VALIDATION_CONFIG environment variable", file=sys.stderr
            )

    should_check_frequency = False
    topic_min_map: dict[str, float] = {}
    if custom_standards and isinstance(custom_standards, dict):
        if custom_standards.get("enabled", False):
            should_check_frequency = True
        for item in custom_standards.get("topics") or []:
            if not isinstance(item, dict):
                continue
            tname = str(item.get("topic") or "").strip()
            if not tname:
                continue
            try:
                topic_min_map[tname] = float(item.get("min_hz"))
            except (TypeError, ValueError):
                pass
        if topic_min_map:
            should_check_frequency = True

    # Analyze
    total_topics = 0
    compliant_topics = 0

    for topic in expected_topics:
        total_topics += 1
        count = info["topics"].get(topic, 0)

        actual_freq = 0
        if info["duration"] > 0.1:  # Avoid division by zero
            actual_freq = count / info["duration"]

        # Determine target frequency
        target_freq = 0
        if should_check_frequency:
            if topic in topic_min_map:
                target_freq = topic_min_map[topic]
            elif "camera" in topic:
                target_freq = float(custom_standards.get("camera_freq", 25.0))
            else:
                target_freq = float(custom_standards.get("other_freq", 50.0))

        is_compliant = True
        is_critical = False
        is_warning = False

        # Check presence
        if topic not in info["topics"]:
            if should_check_frequency:
                report["check_result"]["missing_topics"].append(topic)
                report["check_result"]["is_valid"] = False
                is_compliant = False
                is_critical = True
            else:
                compliant_topics += 1
        elif count == 0:
            if should_check_frequency:
                report["check_result"]["empty_topics"].append(topic)
                report["check_result"]["is_valid"] = False
                is_compliant = False
                is_critical = True
            else:
                compliant_topics += 1
        else:
            # Check frequency ONLY if enabled and target > 0
            if should_check_frequency and target_freq > 0:
                # Allow 20% deviation
                if actual_freq < target_freq * 0.8:
                    is_compliant = False
                    is_warning = True
                    report["frequency_analysis"]["warning_topics"].append(
                        {
                            "topic": topic,
                            "actual_freq": actual_freq,
                            "standard_freq": target_freq,
                        }
                    )
                else:
                    compliant_topics += 1
            else:
                compliant_topics += 1

        if is_critical:
            report["frequency_analysis"]["critical_topics"].append(
                {
                    "topic": topic,
                    "actual_freq": actual_freq,
                    "standard_freq": target_freq,
                }
            )

        report["frequency_analysis"]["frequency_details"][topic] = {
            "actual_frequency": actual_freq,
            "standard_frequency": target_freq if should_check_frequency else 0,
            "deviation_percent": (
                (actual_freq - target_freq) / target_freq * 100
                if (should_check_frequency and target_freq > 0)
                else 0
            ),
            "is_compliant": is_compliant,
            "is_critical": is_critical,
            "is_warning": is_warning,
            "message_count": count,
        }

    report["frequency_analysis"]["total_topics"] = total_topics
    report["frequency_analysis"]["compliant_topics"] = compliant_topics

    if not should_check_frequency:
        report["frequency_analysis"]["summary"] = "✅ 未启用频率异常检测"
    elif compliant_topics == total_topics:
        report["frequency_analysis"]["summary"] = "✅ 所有话题频率检查通过"
    elif not report["check_result"]["is_valid"]:
        report["frequency_analysis"]["summary"] = "❌ 关键话题缺失或为空"
    else:
        report["frequency_analysis"]["summary"] = (
            f"⚠️ {total_topics - compliant_topics} 个话题频率异常"
        )

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python3 validate_bag.py <bag_path> [expected_duration] [mode]",
            file=sys.stderr,
        )
        sys.exit(1)

    bag_path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    mode = sys.argv[3] if len(sys.argv) > 3 else "raw"

    rep = validate(bag_path, duration, mode)
    if rep:
        print(json.dumps(rep, ensure_ascii=False))
        sys.exit(0)
    sys.exit(2)

