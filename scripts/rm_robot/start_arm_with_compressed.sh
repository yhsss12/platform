#!/bin/bash

echo "========================================"
echo "  双机械臂控制系统启动脚本 (启用彩色压缩)"
echo "========================================"

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查是否已有ROS2节点在运行
if pgrep -f "gello_arm1.py\|gripper_control.py\|realman_left_arm.py\|realman_right_arm.py\|force_listener.py\|realsense2_camera" > /dev/null; then
    echo "警告: 检测到已有相关程序在运行"
    echo "是否要关闭所有现有进程？(y/n)"
    read -r answer
    if [ "$answer" = "y" ]; then
        ./stop_arm.sh
        sleep 2
    else
        echo "退出启动"
        exit 1
    fi
fi

# 尝试加载 ROS2 环境（无论外部是否已 source，都再确保一次）
if [ -z "$ROS_DISTRO" ]; then
    echo "正在加载 ROS2 环境..."
    # 你的 ros2 在 /opt/ros/humble/bin/ros2，这里强制加入 PATH 并 source 一次
    if [ -d /opt/ros/humble ]; then
        export PATH="/opt/ros/humble/bin:$PATH"
        # shellcheck source=/opt/ros/humble/setup.bash
        [ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
    fi
fi

# 再次确认 ros2 是否可用，避免后续命令直接失败
if ! command -v ros2 >/dev/null 2>&1; then
    echo "❌ 未找到 ros2 命令，请确认已安装 ROS2 并在本脚本中配置正确的 setup.bash 路径（当前假定为 /opt/ros/humble）。"
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

# 启动相机 - 启用彩色压缩，禁用深度压缩
echo "正在启动三个 Realsense 相机 (启用彩色压缩)..."
echo "========================================"

# 通用相机参数 - 关键参数说明：
# - enable_compressed=true: 启用图像压缩（彩色和深度都会尝试压缩）
# - depth_module.enable_compressed=false: 专门禁用深度压缩
# - rgb_camera.enable_compressed=true: 专门启用彩色压缩
COMMON_PARAMS="\
    -p serial_no:=_SERIAL_NO \
    -p enable_depth:=true \
    -p enable_color:=true \
    -p depth_module.profile:=640x480x30 \
    -p rgb_camera.profile:=640x480x30 \
    -p pointcloud.enable:=false \
    -p align_depth.enable:=false \
    -p enable_compressed:=true \
    -p depth_module.enable_compressed:=false \
    -p rgb_camera.enable_compressed:=true"

# 启动相机1 (serial: 216322073550)
PARAMS1=${COMMON_PARAMS/_SERIAL_NO/_216322073550}
ros2 run realsense2_camera realsense2_camera_node \
    --ros-args \
    -r __ns:=/camera1 \
    -r __node:=camera1 \
    $PARAMS1 &
CAMERA1_PID=$!
echo "相机1 (216322073550) PID: $CAMERA1_PID"

# 启动相机2 (serial: 419522071631)
PARAMS2=${COMMON_PARAMS/_SERIAL_NO/_419522071631}
ros2 run realsense2_camera realsense2_camera_node \
    --ros-args \
    -r __ns:=/camera2 \
    -r __node:=camera2 \
    $PARAMS2 &
CAMERA2_PID=$!
echo "相机2 (419522071631) PID: $CAMERA2_PID"

# 启动相机3 (serial: 243722075193)
PARAMS3=${COMMON_PARAMS/_SERIAL_NO/_243722075193}
ros2 run realsense2_camera realsense2_camera_node \
    --ros-args \
    -r __ns:=/camera3 \
    -r __node:=camera3 \
    $PARAMS3 &
CAMERA3_PID=$!
echo "相机3 (243722075193) PID: $CAMERA3_PID"

# 等待相机初始化
echo "等待相机初始化 (5秒)..."
sleep 5

# 检查相机进程
if kill -0 $CAMERA1_PID 2>/dev/null && kill -0 $CAMERA2_PID 2>/dev/null && kill -0 $CAMERA3_PID 2>/dev/null; then
    echo "✅ 所有相机启动成功"
    
    echo ""
    echo "可用话题:"
    echo "----------------------------------------"
    echo "原始彩色图像:"
    ros2 topic list | grep -E "/camera[123]/camera[123]/color/image_raw$" | head -3
    echo ""
    echo "压缩彩色图像:"
    ros2 topic list | grep -E "/camera[123]/camera[123]/color/image_raw/compressed$" | head -3
    echo ""
    echo "深度图像 (原始):"
    ros2 topic list | grep -E "/camera[123]/camera[123]/depth/image_raw$" | head -3
else
    echo "❌ 有相机启动失败，请检查"
    kill $CAMERA1_PID $CAMERA2_PID $CAMERA3_PID 2>/dev/null
    exit 1
fi

echo "========================================"

# 末端六维力由 realman_left_arm.py / realman_right_arm.py 经 UDP 发布（需上方已叠加 rm_ros_interfaces）。
# 勿与 rm_driver 同时连同一机械臂；若仅需独立驱动可另用 ./start_rm_driver_with_force.sh

# 启动主控制程序
echo "启动主控制程序..."
python3 dual_arm_launcher.py &
LAUNCHER_PID=$!

# 保存PID
echo "$CAMERA1_PID $CAMERA2_PID $CAMERA3_PID" > /tmp/arm_camera.pids
echo "$LAUNCHER_PID" > /tmp/arm_launcher.pid

# 捕获Ctrl+C
trap 'echo "正在关闭..."; ./stop_arm.sh; exit' INT

# 等待主程序结束
wait $LAUNCHER_PID

# 主程序结束后自动关闭相机
./stop_arm.sh
