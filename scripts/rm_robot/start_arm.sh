#!/bin/bash

echo "========================================"
echo "  双机械臂控制系统启动脚本"
echo "========================================"

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查是否已有ROS2节点在运行
if pgrep -f "gello_arm1.py\|gripper_control.py\|realman_left_arm.py\|realman_right_arm.py\|force_listener.py\|realsense2_camera" > /dev/null; then
    echo "警告: 检测到已有相关程序在运行"
    
    # 检查是否有强制重启的环境变量
    if [ "$FORCE_RESTART" = "true" ]; then
        echo "检测到 FORCE_RESTART=true，自动重启..."
        answer="y"
    else
        echo "是否要关闭所有现有进程？(y/n)"
        read -r answer
    fi

    if [ "$answer" = "y" ]; then
        # 调用停止脚本
        ./stop_arm.sh
        sleep 2
    else
        echo "退出启动"
        exit 1
    fi
fi

# 尝试加载 ROS2 环境（无论外部是否已 source，都再确保一次）
if [ -z "${ROS_DISTRO:-}" ]; then
    echo "正在加载 ROS2 环境..."
    if [ -d /opt/ros/humble ]; then
        export PATH="/opt/ros/humble/bin:$PATH"
        # shellcheck source=/opt/ros/humble/setup.bash
        [ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
    fi
fi

# 再次确认 ros2 是否可用，避免后续命令直接失败
if ! command -v ros2 >/dev/null 2>&1; then
    echo "❌ 未找到 ros2 命令。原因通常是：ROS2 未安装，或当前进程环境未 source /opt/ros/humble/setup.bash（systemd/服务进程常见）。"
    echo "   - 你可以手动执行：source /opt/ros/humble/setup.bash && ros2 -h"
    echo "   - 或改用 scripts/rm_robot/start_arm_with_compressed.sh（已内置 PATH+检测）"
    exit 1
fi

# 叠加睿尔曼 install（提供 rm_ros_interfaces），realman_* 才能发布 Sixforce 六维力话题
_RM_DONE=0
if [ -n "${RM_ROS_WS:-}" ] && [ -f "${RM_ROS_WS}/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "${RM_ROS_WS}/setup.bash"
    echo "已叠加 RM_ROS_WS: ${RM_ROS_WS}"
    _RM_DONE=1
elif [ -n "${RM_ROS_WS:-}" ] && [ -f "${RM_ROS_WS}/install/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "${RM_ROS_WS}/install/setup.bash"
    echo "已叠加 RM_ROS_WS/install: ${RM_ROS_WS}/install"
    _RM_DONE=1
fi
if [ "$_RM_DONE" -eq 0 ] && [ -f "$HOME/Workspace/RM/ros2_rm_robot-humble/install/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "$HOME/Workspace/RM/ros2_rm_robot-humble/install/setup.bash"
    echo "已叠加默认睿尔曼工作空间: $HOME/Workspace/RM/ros2_rm_robot-humble/install"
fi

# 启动相机
echo "正在启动Realsense相机..."
echo "========================================"

# 在当前目录下查找realsense工作空间
REALSENSE_WS_PATH=""
POSSIBLE_PATHS=(
    "$SCRIPT_DIR/realsense_ws"                    # 当前目录下的realsense_ws
    "$SCRIPT_DIR/../realsense_ws"                  # 上级目录下的realsense_ws
    "$SCRIPT_DIR/src/realsense_ws"                  # 当前目录/src下的realsense_ws
    "$(pwd)/realsense_ws"                           # 当前执行目录下的realsense_ws
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/install/setup.bash" ]; then
        REALSENSE_WS_PATH="$path"
        echo "找到realsense工作空间: $REALSENSE_WS_PATH"
        break
    fi
done

if [ -n "$REALSENSE_WS_PATH" ]; then
    # 找到工作空间，直接使用
    source "$REALSENSE_WS_PATH/install/setup.bash"
    ros2 launch realsense2_camera rs_triple_camera_launch.py \
        camera_name1:=camera1 camera_namespace1:=camera1 serial_no1:=_216322073550 \
        camera_name2:=camera2 camera_namespace2:=camera2 serial_no2:=_419522071631 \
        camera_name3:=camera3 camera_namespace3:=camera3 serial_no3:=_243722075193 &
elif [ -f "./rs_triple_camera_launch.py" ]; then
    # 在当前目录找到相机启动脚本
    echo "在当前目录找到相机启动脚本，尝试直接启动..."
    ros2 launch realsense2_camera rs_triple_camera_launch.py \
        camera_name1:=camera1 camera_namespace1:=camera1 serial_no1:=_216322073550 \
        camera_name2:=camera2 camera_namespace2:=camera2 serial_no2:=_419522071631 \
        camera_name3:=camera3 camera_namespace3:=camera3 serial_no3:=_243722075193 &
else
    # 尝试在当前目录的install中查找
    if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
        echo "在当前目录的install中找到setup.bash"
        source "$SCRIPT_DIR/install/setup.bash"
        ros2 launch realsense2_camera rs_triple_camera_launch.py \
            camera_name1:=camera1 camera_namespace1:=camera1 serial_no1:=_216322073550 \
            camera_name2:=camera2 camera_namespace2:=camera2 serial_no2:=_419522071631 \
            camera_name3:=camera3 camera_namespace3:=camera3 serial_no3:=_243722075193 &
    else
        echo "错误: 找不到realsense工作空间"
        echo "请在以下位置查找:"
        for path in "${POSSIBLE_PATHS[@]}"; do
            echo "  - $path"
        done
        echo "或者手动启动相机:"
        echo "source [realsense_ws_path]/install/setup.bash"
        echo "ros2 launch realsense2_camera rs_triple_camera_launch.py ..."
        exit 1
    fi
fi

CAMERA_PID=$!
echo "相机已启动 (PID: $CAMERA_PID)"

# 等待相机初始化
echo "等待相机初始化 (5秒)..."
sleep 5

# 检查相机进程是否仍在运行
if ! kill -0 $CAMERA_PID 2>/dev/null; then
    echo "❌ 相机启动失败"
    exit 1
fi

echo "✅ 相机启动成功"
echo "========================================"

# 可选：睿尔曼 rm_driver（双臂）+ 六维力监听 — 在独立终端执行:
#   ./start_rm_driver_with_force.sh
# rm_driver 与 realman_left_arm.py / realman_right_arm.py 会争用同一臂 TCP，勿对同一 IP 同时启用。

# 启动主控制程序
echo "启动主控制程序..."
python3 dual_arm_launcher.py &
LAUNCHER_PID=$!

# 保存PID到文件，便于停止脚本使用
echo "$CAMERA_PID" > /tmp/arm_camera.pid
echo "$LAUNCHER_PID" > /tmp/arm_launcher.pid

# 捕获Ctrl+C
trap 'echo "正在关闭..."; ./stop_arm.sh; exit' INT

# 等待主程序结束
wait $LAUNCHER_PID

# 主程序结束后自动关闭相机
./stop_arm.sh
