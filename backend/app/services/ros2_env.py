"""
ROS2 环境变量构建工具
"""
import os
from typing import Dict
from app.models.device import ROS2Config


def build_ros2_env(ros2_config: ROS2Config) -> Dict[str, str]:
    """
    构建 ROS2 环境变量字典
    
    Args:
        ros2_config: ROS2 配置对象
        
    Returns:
        环境变量字典，包含：
        - RMW_IMPLEMENTATION=rmw_fastrtps_cpp
        - FASTRTPS_DEFAULT_PROFILES_FILE={profile_path}
        - ROS2_DISABLE_DAEMON=1
        - ROS_DOMAIN_ID={domain_id}
    """
    env = os.environ.copy()
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    env["FASTRTPS_DEFAULT_PROFILES_FILE"] = ros2_config.profile_path
    env["ROS2_DISABLE_DAEMON"] = "1"
    env["ROS_DOMAIN_ID"] = str(ros2_config.domain_id)
    return env























