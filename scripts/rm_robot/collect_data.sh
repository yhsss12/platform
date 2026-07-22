#!/bin/bash
# record_aloha_data.sh - 用于记录ALOHA机器人所有相关话题的脚本
# 优化版：添加进度显示和话题频率检查

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
WHITE='\033[0;37m'
NC='\033[0m' # No Color

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
  echo -e "${YELLOW}警告: 未检测到 MCAP 插件(librosbag2_storage_mcap.so)，将使用 sqlite3 存储。${NC}"
else
  echo -e "${GREEN}检测到 MCAP 插件，将使用 mcap 存储。${NC}"
fi

# 默认记录时长（秒）
RECORD_DURATION=30

# 检查是否有传入参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--time)
            if [[ $2 =~ ^[0-9]+$ ]]; then
                RECORD_DURATION="$2"
                shift 2
            else
                echo -e "${RED}错误: -t|--time 参数需要一个数值${NC}"
                echo "用法: $0 [-t|--time <秒数>] [-o|--output <目录>] [-h|--help]"
                exit 1
            fi
            ;;
        -o|--output)
            if [[ -n "$2" ]]; then
                ROOT_BAG_DIR="$2"
                shift 2
            else
                echo -e "${RED}错误: -o|--output 参数需要一个目录路径${NC}"
                exit 1
            fi
            ;;
        -h|--help)
            echo "用法: $0 [-t|--time <秒数>] [-o|--output <目录>] [-h|--help]"
            echo "选项:"
            echo "  -t, --time <秒数>    设置记录时长（默认: 30秒）"
            echo "  -o, --output <目录>  设置存储根目录"
            echo "  -h, --help           显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "用法: $0 [-t|--time <秒数>] [-o|--output <目录>] [-h|--help]"
            exit 1
            ;;
    esac
done

# 定义存储根目录
ROOT_BAG_DIR="/home/rm/Workspace/data"

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
BAG_PATH="$BAG_DIR/$BAG_NAME"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}开始记录ALOHA相关话题${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "存储目录: ${YELLOW}$BAG_PATH${NC}"
echo -e "存储格式: ${YELLOW}$EXT${NC}"
echo -e "记录时长: ${YELLOW}$RECORD_DURATION 秒${NC}"
echo -e "${BLUE}========================================${NC}"

# 进入存储目录
cd $BAG_DIR

# 启动记录进程在后台
ros2 bag record \
  --storage "${STORAGE_BACKEND}" \
  --include-unpublished-topics \
  -o $BAG_NAME \
  /left_gripper_cmd \
  /left_gripper_state \
  /left/joint_states \
  /left/rm_driver/get_force_data_result \
  /right_gripper_cmd \
  /right_gripper_state \
  /right/joint_states \
  /right/rm_driver/get_force_data_result \
  /camera1/camera1/color/image_raw \
  /camera2/camera2/color/image_raw \
  /camera3/camera3/color/image_raw \
  /camera1/camera1/depth/image_rect_raw \
  /camera2/camera2/depth/image_rect_raw \
  /camera3/camera3/depth/image_rect_raw \
  > /dev/null 2>&1 &

# 记录进程ID
RECORD_PID=$!

# 等待 ros2 bag record 启动并开始写入 (2秒缓冲)
echo -e "${YELLOW}等待记录进程初始化 (2秒)...${NC}"
sleep 2

# 进度显示函数 - 修复版
show_progress() {
    local current=$1
    local total=$2
    local width=40
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    
    # 清除当前行
    printf "\r\033[K"
    
    # 开始记录文字 - 白色
    printf "${WHITE}开始记录:${NC} "
    
    # 进度条 - 蓝色使用 - 符号
    printf "${BLUE}[${NC}"
    for ((i=0; i<width; i++)); do
        if [ $i -lt $completed ]; then
            printf "${BLUE}-${NC}"
        else
            printf " "
        fi
    done
    printf "${BLUE}]${NC} "
    
    # 百分比和时间 - 白色
    printf "${WHITE}%3d%%${NC} " $percentage
    
    # 已用时间/总时间 - 白色
    printf "${WHITE}%ds/${total}s${NC}" $current $total
    
    # 剩余时间 - 白色
    local remaining=$((total - current))
    if [ $remaining -gt 0 ]; then
        printf " ${WHITE}(剩余: %02d:%02d)${NC}" $((remaining / 60)) $((remaining % 60))
    fi
}

# 等待并显示进度
echo -e "${WHITE}开始记录...${NC}"
START_TIME=$(date +%s)
END_TIME=$((START_TIME + RECORD_DURATION))

while [ $(date +%s) -lt $END_TIME ]; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    # 每秒更新一次进度
    show_progress $ELAPSED $RECORD_DURATION
    
    sleep 1
done

# 完成进度显示
show_progress $RECORD_DURATION $RECORD_DURATION
echo -e "\n${GREEN}记录完成！${NC}"

# 停止记录
echo -e "${YELLOW}正在停止记录进程...${NC}"
kill -2 $RECORD_PID 2>/dev/null
wait $RECORD_PID 2>/dev/null

echo -e "${GREEN}记录已保存到 $BAG_PATH${NC}"
echo -e "Episode编号: ${YELLOW}$NEXT_IDX${NC}"
echo -e "记录时长: ${YELLOW}$RECORD_DURATION 秒${NC}"

# 频率检查函数
check_topic_frequency() {
    local topic=$1
    local min_freq=$2
    local topic_name=$3
    
    echo -e "\n${BLUE}检查话题: ${YELLOW}$topic${NC}"
    
    # 使用ros2 bag info获取包信息并保存到临时文件
    BAG_INFO_FILE=$(mktemp)
    ros2 bag info "$BAG_PATH" > "$BAG_INFO_FILE" 2>&1
    
    # 查找包含该话题的行
    TOPIC_LINE=$(grep -F "$topic" "$BAG_INFO_FILE" || true)
    
    if [ -z "$TOPIC_LINE" ]; then
        echo -e "${RED}❌ $topic_name: 未找到话题${NC}"
        rm -f "$BAG_INFO_FILE"
        return 1
    fi
    
    # 提取消息数量
    MSG_COUNT=$(echo "$TOPIC_LINE" | grep -o 'Count: [0-9]\+' | grep -o '[0-9]\+' || echo "0")
    
    # 如果没有找到Count，尝试其他格式
    if [ "$MSG_COUNT" = "0" ]; then
        MSG_COUNT=$(echo "$TOPIC_LINE" | grep -o '[0-9]\+' | head -1 || echo "0")
    fi
    
    # 清理临时文件
    rm -f "$BAG_INFO_FILE"
    
    # 检查是否成功提取数字
    if ! [[ "$MSG_COUNT" =~ ^[0-9]+$ ]] || [ "$MSG_COUNT" -eq 0 ]; then
        echo -e "${RED}❌ $topic_name: 无法获取消息数量${NC}"
        return 1
    fi
    
    # 计算频率
    if [ "$RECORD_DURATION" -gt 0 ]; then
        FREQ=$(echo "scale=2; $MSG_COUNT / $RECORD_DURATION" | bc)
        
        # 比较是否达到最低频率要求
        if (( $(echo "$FREQ >= $min_freq" | bc -l) )); then
            echo -e "${GREEN}✅ $topic_name: ${FREQ}Hz (${MSG_COUNT}条消息, ≥ ${min_freq}Hz)${NC}"
            return 0
        else
            echo -e "${RED}❌ $topic_name: ${FREQ}Hz (${MSG_COUNT}条消息, < ${min_freq}Hz)${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ $topic_name: 记录时长为0${NC}"
        return 1
    fi
}

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}开始检查话题频率...${NC}"
echo -e "${BLUE}========================================${NC}"

# 等待一下确保bag写入完成
sleep 2

# 检查相机话题 (要求 ≥ 25Hz)
echo -e "\n${YELLOW}--- 相机话题检查 (要求 ≥ 25Hz) ---${NC}"
CAMERA_TOPICS=(
    "/camera1/camera1/color/image_raw:camera1_color"
    "/camera2/camera2/color/image_raw:camera2_color"
    "/camera3/camera3/color/image_raw:camera3_color"
    "/camera1/camera1/depth/image_rect_raw:camera1_depth"
    "/camera2/camera2/depth/image_rect_raw:camera2_depth"
    "/camera3/camera3/depth/image_rect_raw:camera3_depth"
)

CAMERA_FAILED=0
CAMERA_TOTAL=0
for topic_info in "${CAMERA_TOPICS[@]}"; do
    IFS=':' read -r topic name <<< "$topic_info"
    CAMERA_TOTAL=$((CAMERA_TOTAL + 1))
    if ! check_topic_frequency "$topic" 25 "$name"; then
        CAMERA_FAILED=$((CAMERA_FAILED + 1))
    fi
done

# 检查夹爪话题 (要求 ≥ 50Hz)
echo -e "\n${YELLOW}--- 夹爪话题检查 (要求 ≥ 50Hz) ---${NC}"
GRIPPER_TOPICS=(
    "/left_gripper_cmd:left_gripper_cmd"
    "/right_gripper_cmd:right_gripper_cmd"
    # 注意: /left_gripper_state 和 /right_gripper_state 不检查
)

GRIPPER_FAILED=0
GRIPPER_TOTAL=0
for topic_info in "${GRIPPER_TOPICS[@]}"; do
    IFS=':' read -r topic name <<< "$topic_info"
    GRIPPER_TOTAL=$((GRIPPER_TOTAL + 1))
    if ! check_topic_frequency "$topic" 50 "$name"; then
        GRIPPER_FAILED=$((GRIPPER_FAILED + 1))
    fi
done

# 检查关节状态话题 (要求 ≥ 50Hz)
echo -e "\n${YELLOW}--- 关节状态话题检查 (要求 ≥ 50Hz) ---${NC}"
JOINT_TOPICS=(
    "/left/joint_states:left_joint_states"
    "/right/joint_states:right_joint_states"
)

JOINT_FAILED=0
JOINT_TOTAL=0
for topic_info in "${JOINT_TOPICS[@]}"; do
    IFS=':' read -r topic name <<< "$topic_info"
    JOINT_TOTAL=$((JOINT_TOTAL + 1))
    if ! check_topic_frequency "$topic" 50 "$name"; then
        JOINT_FAILED=$((JOINT_FAILED + 1))
    fi
done

# 输出总结
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}频率检查总结${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "相机话题: ${CAMERA_FAILED}/${CAMERA_TOTAL} 失败"
echo -e "夹爪话题: ${GRIPPER_FAILED}/${GRIPPER_TOTAL} 失败"
echo -e "关节状态: ${JOINT_FAILED}/${JOINT_TOTAL} 失败"

if [ $CAMERA_FAILED -eq 0 ] && [ $GRIPPER_FAILED -eq 0 ] && [ $JOINT_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✅ 所有话题频率检查通过！${NC}"
else
    echo -e "\n${RED}❌ 部分话题频率未达到要求！${NC}"
    echo -e "${YELLOW}建议检查:${NC}"
    echo "  1. 相机是否正常工作"
    echo "  2. 网络带宽是否足够"
    echo "  3. 系统负载是否过高"
    echo "  4. 发布者是否正常发布数据"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}记录完成！${NC}"

# 显示bag基本信息
echo -e "\n${YELLOW}Bag文件信息:${NC}"
ros2 bag info "$BAG_PATH" | head -10

# 与平台质量页对齐：OUTPUT_PATH + EAI_VALIDATION_REPORT_JSON
echo "OUTPUT_PATH: $BAG_PATH"
_EAI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_EAI_HELP=""
for _d in "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" "$_EAI_REPO/agent" "/opt/eai-agent"; do
  if [ -f "$_d/eai_quality_report_helpers.sh" ]; then
    _EAI_HELP="$_d/eai_quality_report_helpers.sh"
    break
  fi
done
if [ -n "$_EAI_HELP" ]; then
  # shellcheck source=/dev/null
  . "$_EAI_HELP"
  eai_emit_validation_report "$BAG_PATH" "${RECORD_DURATION:-30}" "$_EAI_REPO" || true
fi
