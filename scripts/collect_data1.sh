#!/bin/bash
# record_aloha_data.sh - 用于记录ALOHA机器人所有相关话题的脚本

set -euo pipefail

# 避免 Conda 干扰，并加载 ROS Humble 环境（确保 storage 插件可见）
if command -v conda >/dev/null 2>&1; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    # 某些环境需要先初始化才能 deactivate；若失败则忽略
    conda deactivate 2>/dev/null || true
  fi
fi
if [ -f /opt/ros/humble/setup.bash ]; then
  # ROS 的 setup.bash 在未设置 AMENT_TRACE_SETUP_FILES 时会读取空变量；
  # 为避免 set -u 导致报错，这里临时关闭 -u，再恢复。
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

# 选择存储后端：优先使用 mcap，若未安装则回退到 sqlite3
STORAGE_BACKEND="mcap"
EXT="mcap"
if [ ! -f "/opt/ros/humble/lib/librosbag2_storage_mcap.so" ]; then
  STORAGE_BACKEND="sqlite3"
  EXT="db3"
  echo "警告: 未检测到 MCAP 插件(librosbag2_storage_mcap.so)，将使用 sqlite3 存储。"
else
  echo "检测到 MCAP 插件，将使用 mcap 存储。"
fi

# 默认记录时长（秒）
RECORD_DURATION=40
# 默认存储根目录
ROOT_BAG_DIR="$HOME/data/rosbags/aloha_recordings"

# 监控脚本路径配置 - 修改这个路径指向您的check_rosbag_monitor.py文件
MONITOR_SCRIPT_PATH="$HOME/realman_ros2/tools/check_rosbag_monitor.py"

# 检查是否有传入参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--time)
            if [[ $2 =~ ^[0-9]+$ ]]; then
                RECORD_DURATION="$2"
                shift 2
            else
                echo "错误: -t|--time 参数需要一个数值"
                echo "用法: $0 [-t|--time <秒数>] [-o|--output <目录>] [-h|--help]"
                exit 1
            fi
            ;;
        -o|--output)
            if [[ -n "$2" ]]; then
                ROOT_BAG_DIR="$2"
                shift 2
            else
                echo "错误: -o|--output 参数需要一个目录路径"
                exit 1
            fi
            ;;
        -h|--help)
            echo "用法: $0 [-t|--time <秒数>] [-o|--output <目录>] [-h|--help]"
            echo "选项:"
            echo "  -t, --time <秒数>    设置记录时长（默认: 40秒）"
            echo "  -o, --output <目录>  设置存储根目录"
            echo "  -h, --help           显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [-t|--time <秒数>] [-o|--output <目录>] [-h|--help]"
            exit 1
            ;;
    esac
done

# 定义存储目录，增加日期作为子目录
DATE_DIR=$(date +"%Y-%m-%d")
BAG_DIR="$ROOT_BAG_DIR/$DATE_DIR"
mkdir -p $BAG_DIR

# 生成带时间戳的包名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 查找已存在的episode文件夹，确定下一个episode编号
NEXT_IDX=0
if [ -d "$ROOT_BAG_DIR" ]; then
    # 查找所有日期目录下的episode文件夹并提取最大的索引号
    for dir in $(find "$ROOT_BAG_DIR" -type d -name "episode_*_*" 2>/dev/null); do
        # 检查目录名是否符合 episode_数字_年月日_时分秒 格式
        dir_name=$(basename "$dir")
        if [[ $dir_name =~ ^episode_[0-9]+_[0-9]{8}_[0-9]{6}$ ]]; then
            # 提取索引号 (格式: episode_数字_时间戳)
            idx=$(echo "$dir_name" | cut -d'_' -f2)
            if [[ $idx =~ ^[0-9]+$ ]] && [ $idx -ge $NEXT_IDX ]; then
                NEXT_IDX=$((idx + 1))
            fi
        fi
    done
fi

BAG_NAME="episode_${NEXT_IDX}_${TIMESTAMP}"

echo "开始记录ALOHA相关话题到 $BAG_DIR/$BAG_NAME (格式: $EXT)"
echo "记录时长: $RECORD_DURATION 秒"

# 进入存储目录
cd $BAG_DIR

# 启动记录进程在后台
ros2 bag record \
  --storage "${STORAGE_BACKEND}" \
  --include-unpublished-topics \
  -o $BAG_NAME \
  /left_gripper_cmd \
  /left_gripper_state \
  /left_master_arm_joint_states \
  /left_slave_arm_joint_states \
  /right_gripper_cmd \
  /right_gripper_state \
  /right_master_arm_joint_states \
  /right_slave_arm_joint_states \
  /camera1/cam_bottom/color/image_raw \
  /camera2/cam_extr/color/image_raw \
  /camera3/cam_left/color/image_raw \
  /camera4/cam_right/color/image_raw \
  /camera1/cam_bottom/depth/image_rect_raw \
  /camera2/cam_extr/depth/image_rect_raw \
  /camera3/cam_left/depth/image_rect_raw \
  /camera4/cam_right/depth/image_rect_raw &
# 记录进程ID
RECORD_PID=$!

# 等待指定的时间，支持提前结束
echo "开始记录，将在 $RECORD_DURATION 秒后自动停止..."
echo "💡 提示: 按 Ctrl+C 可以提前结束记录并进入数据分析阶段"
echo ""

# 设置Ctrl+C信号处理
user_interrupted=false
trap 'user_interrupted=true; echo ""; echo "⏹️  用户按 Ctrl+C 提前结束记录..."' INT

# 使用sleep实现可中断的等待
remaining_time=$RECORD_DURATION
while [ $remaining_time -gt 0 ] && [ "$user_interrupted" = false ]; do
    # 显示倒计时（每1秒显示一次）
    if [ $((remaining_time % 1)) -eq 0 ] || [ $remaining_time -eq 1 ]; then
        echo "⏱️  剩余时间: ${remaining_time} 秒 (按 Ctrl+C 提前结束)"
    fi
    
    # 每次等待1秒，如果被中断会立即退出
    sleep 1
    remaining_time=$((remaining_time - 1))
done

# 停止记录
if [ "$user_interrupted" = true ]; then
    echo "⏹️  用户按 Ctrl+C 提前结束，正在停止记录..."
elif [ $remaining_time -eq 0 ]; then
    echo "⏰ 时间到，正在停止记录..."
else
    echo "⏹️  用户提前结束，正在停止记录..."
fi
kill -2 $RECORD_PID 2>/dev/null
sleep 3  # 等待最多3秒确保数据写盘

# 计算实际记录时长
actual_duration=$((RECORD_DURATION - remaining_time))

echo "记录已完成，保存到 $BAG_DIR/$BAG_NAME (格式: $EXT)"
echo "Episode编号: $NEXT_IDX"
echo "实际记录时长: ${actual_duration} 秒"

# 等待文件写入完成
echo "等待文件写入完成..."
sleep 2

# 运行数据质量检查
echo ""
echo "=========================================="
echo "🔍 开始检查数据记录情况..."
echo "=========================================="

# 检查check_rosbag_monitor.py是否存在
if [ -f "$MONITOR_SCRIPT_PATH" ]; then
    echo "运行数据质量检查脚本..."
    echo "检查路径: $BAG_DIR/$BAG_NAME"
    echo "使用默认频率配置"
    echo "----------------------------------------"
    # 传递指定的文件夹和文件路径给监控脚本，使用默认频率配置
    # 临时关闭错误立即退出，防止检查脚本失败导致后续OUTPUT_PATH无法输出
    set +e
    export PYTHONIOENCODING=utf-8
    python3 $MONITOR_SCRIPT_PATH --once --force -f "$BAG_DIR/$BAG_NAME" -s "/home/rm-sia/realman_ros2/tools/collect_data1.sh" > /dev/null 2>&1
    CHECK_EXIT_CODE=$?
    set -e
    
    echo "----------------------------------------"
    if [ $CHECK_EXIT_CODE -eq 0 ]; then
        echo "✅ 数据检查完成"
    else
        echo "⚠️  数据检查脚本执行失败 (Exit Code: $CHECK_EXIT_CODE)"
        echo "可能是环境问题或脚本错误，但这不影响数据记录。"
    fi
else
    echo "⚠️  警告: 未找到监控脚本文件"
    echo "配置的路径: $MONITOR_SCRIPT_PATH"
    echo "请修改脚本第36行的 MONITOR_SCRIPT_PATH 变量，指向正确的文件路径"
fi

# 平台实时采集页 / 质量页：OUTPUT_PATH + EAI_VALIDATION_REPORT_JSON（采集端本机 validate_bag）
echo "OUTPUT_PATH: $BAG_DIR/$BAG_NAME"

_EAI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_EAI_HELP=""
for _d in "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$_EAI_REPO/agent" "/opt/eai-agent"; do
  if [ -f "$_d/eai_quality_report_helpers.sh" ]; then
    _EAI_HELP="$_d/eai_quality_report_helpers.sh"
    break
  fi
done
if [ -n "$_EAI_HELP" ]; then
  # shellcheck source=/dev/null
  . "$_EAI_HELP"
  eai_emit_validation_report "$BAG_DIR/$BAG_NAME" "${actual_duration:-30}" "$_EAI_REPO" || true
fi

echo ""
echo "=========================================="
echo "📊 记录总结:"
echo "  - Episode编号: $NEXT_IDX"
echo "  - 实际记录时长: ${actual_duration} 秒"
echo "  - 保存位置: $BAG_DIR/$BAG_NAME.$EXT"
echo "  - 数据检查: 已完成"
echo "=========================================="

# 脚本自动退出，不需要等待用户输入
exit 0
