"""
ROS2 连接测试服务
"""
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict
from app.models.device import ROS2Config
from app.services.ros2_env import build_ros2_env


async def test_ros2_connection(ros2_config: ROS2Config) -> Dict:
    """
    测试 ROS2 连接

    Args:
        ros2_config: ROS2 配置对象

    Returns:
        测试结果字典，包含 status, node_count, nodes_sample, topic_count, topics_sample, error_type, error_message
    """
    if not ros2_config.profile_path or not Path(ros2_config.profile_path).exists():
        return {
            "status": "fail",
            "node_count": None,
            "nodes_sample": None,
            "topic_count": None,
            "topics_sample": None,
            "error_type": "CONFIG_ERROR",
            "error_message": f"配置文件不存在: {ros2_config.profile_path}"
        }

    # 准备环境变量（使用公共函数）
    env = build_ros2_env(ros2_config)

    result = {
        "status": "fail",
        "node_count": 0,
        "nodes_sample": [],
        "topic_count": 0,
        "topics_sample": [],
        "error_type": None,
        "error_message": None
    }

    try:
        # 测试节点列表
        try:
            node_list_cmd = ["ros2", "node", "list"]
            node_result = subprocess.run(
                node_list_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )

            if node_result.returncode == 0:
                nodes = [line.strip() for line in node_result.stdout.strip().split('\n') if line.strip()]
                result["node_count"] = len(nodes)
                result["nodes_sample"] = nodes[:10]  # 最多10个
            else:
                result["error_type"] = "NODE_LIST_ERROR"
                result["error_message"] = node_result.stderr or "无法获取节点列表"
                return result

        except subprocess.TimeoutExpired:
            result["error_type"] = "TIMEOUT"
            result["error_message"] = "获取节点列表超时"
            return result
        except Exception as e:
            result["error_type"] = "NODE_LIST_EXCEPTION"
            result["error_message"] = str(e)
            return result

        # 测试主题列表
        try:
            topic_list_cmd = ["ros2", "topic", "list"]
            topic_result = subprocess.run(
                topic_list_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )

            if topic_result.returncode == 0:
                topics = [line.strip() for line in topic_result.stdout.strip().split('\n') if line.strip()]
                result["topic_count"] = len(topics)
                result["topics_sample"] = topics[:10]  # 最多10个
            else:
                result["error_type"] = "TOPIC_LIST_ERROR"
                result["error_message"] = topic_result.stderr or "无法获取主题列表"
                return result

        except subprocess.TimeoutExpired:
            result["error_type"] = "TIMEOUT"
            result["error_message"] = "获取主题列表超时"
            return result
        except Exception as e:
            result["error_type"] = "TOPIC_LIST_EXCEPTION"
            result["error_message"] = str(e)
            return result

        # 如果成功获取了节点或主题，认为连接成功
        if result["node_count"] > 0 or result["topic_count"] > 0:
            result["status"] = "success"
        else:
            result["status"] = "fail"
            result["error_type"] = "NO_NODES_OR_TOPICS"
            result["error_message"] = "未发现任何节点或主题"

    except Exception as e:
        result["error_type"] = "UNKNOWN_ERROR"
        result["error_message"] = str(e)

    return result


