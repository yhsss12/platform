#!/bin/bash

echo "正在关闭所有机械臂控制程序和相机..."

# 从PID文件读取进程ID
if [ -f /tmp/arm_launcher.pid ]; then
    LAUNCHER_PID=$(cat /tmp/arm_launcher.pid)
    echo "找到启动器进程: $LAUNCHER_PID"
    # 终止启动器进程组
    pkill -P $LAUNCHER_PID 2>/dev/null
    kill $LAUNCHER_PID 2>/dev/null
    rm -f /tmp/arm_launcher.pid
fi

# 终止所有机械臂相关Python进程
echo "终止机械臂控制程序..."
pkill -f "gello_arm1.py"
pkill -f "gripper_control.py"
pkill -f "realman_left_arm.py"
pkill -f "realman_right_arm.py"
pkill -f "dual_arm_launcher.py"

# 终止相机进程
echo "终止相机进程..."
if [ -f /tmp/arm_camera.pid ]; then
    CAMERA_PID=$(cat /tmp/arm_camera.pid)
    echo "找到相机进程: $CAMERA_PID"
    kill $CAMERA_PID 2>/dev/null
    rm -f /tmp/arm_camera.pid
fi

# 查找并终止所有realsense相机相关进程
pkill -f "realsense2_camera"
pkill -f "rs_triple_camera_launch.py"

# 查找并终止可能残留的ROS2节点
if command -v ros2 &> /dev/null; then
    echo "检查ROS2节点..."
    # 列出当前运行的节点
    NODES=$(ros2 node list 2>/dev/null | grep -E "camera|realsense" || true)
    if [ ! -z "$NODES" ]; then
        echo "找到相机相关节点，正在关闭..."
        # 尝试优雅关闭相机节点
        ros2 lifecycle set /camera1/camera configure 2>/dev/null
        ros2 lifecycle set /camera2/camera configure 2>/dev/null
        ros2 lifecycle set /camera3/camera configure 2>/dev/null
        sleep 1
    fi
fi

# 等待进程结束
sleep 2

# 检查是否还有残留进程
REMAINING=$(pgrep -f "gello_arm1.py\|gripper_control.py\|realman_left_arm.py\|realman_right_arm.py\|dual_arm_launcher.py\|realsense2_camera\|rs_triple_camera_launch.py")
if [ ! -z "$REMAINING" ]; then
    echo "强制终止残留进程..."
    pkill -9 -f "gello_arm1.py\|gripper_control.py\|realman_left_arm.py\|realman_right_arm.py\|dual_arm_launcher.py\|realsense2_camera\|rs_triple_camera_launch.py"
fi

# 清理临时文件
rm -f /tmp/arm_camera.pid /tmp/arm_launcher.pid 2>/dev/null

echo "========================================"
echo "✅ 所有程序已关闭"
echo "========================================"
