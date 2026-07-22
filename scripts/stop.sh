#!/bin/bash

# 停止项目前端和后端脚本
# 使用方法: ./scripts/stop.sh [frontend|backend|all]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置（与 package.json 中 next dev -p 3001 保持一致）
FRONTEND_PORT=3001
BACKEND_PORT=${BACKEND_PORT:-8000}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$PROJECT_ROOT/.pids"

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 根据端口查找进程（优先 lsof 得到纯 PID）
find_process_by_port() {
    local port=$1
    if command -v lsof > /dev/null 2>&1; then
        lsof -ti:$port 2>/dev/null | head -1 || true
    elif command -v ss > /dev/null 2>&1; then
        ss -tlnp 2>/dev/null | grep ":$port " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true
    elif command -v netstat > /dev/null 2>&1; then
        netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | head -1 || true
    else
        echo ""
    fi
}

# 停止进程
stop_process() {
    local name=$1
    local port=$2
    local pid_file="$PID_DIR/${name}.pid"
    
    log_info "停止 $name..."
    
    # 从 PID 文件读取
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_info "从 PID 文件找到进程 $pid，正在停止..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            if ps -p "$pid" > /dev/null 2>&1; then
                log_warn "进程仍在运行，强制停止..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$pid_file"
    fi
    
    # 通过端口查找并停止
    local port_pid=$(find_process_by_port "$port")
    if [ -n "$port_pid" ]; then
        log_info "在端口 $port 找到进程 $port_pid，正在停止..."
        kill "$port_pid" 2>/dev/null || true
        sleep 1
        if ps -p "$port_pid" > /dev/null 2>&1; then
            kill -9 "$port_pid" 2>/dev/null || true
        fi
    fi
    
    # 通过进程名查找
    if [ "$name" = "frontend" ]; then
        pkill -f "next dev" 2>/dev/null || true
        pkill -f "next-server" 2>/dev/null || true
        pkill -f "next start" 2>/dev/null || true
        if command -v lsof >/dev/null 2>&1; then
            lsof -ti:3000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        fi
    elif [ "$name" = "backend" ]; then
        pkill -f "uvicorn app.main:app" 2>/dev/null || true
        pkill -f "uvicorn.*8000" 2>/dev/null || true
    fi
    
    log_info "$name 已停止"
}

# 主函数
main() {
    local target=${1:-all}
    
    log_info "=========================================="
    log_info "停止项目服务"
    log_info "目标: $target"
    log_info "=========================================="
    
    case "$target" in
        frontend)
            stop_process "frontend" "$FRONTEND_PORT"
            ;;
        backend)
            stop_process "backend" "$BACKEND_PORT"
            ;;
        all)
            stop_process "frontend" "$FRONTEND_PORT"
            stop_process "backend" "$BACKEND_PORT"
            ;;
        *)
            log_error "无效的参数: $target"
            echo "使用方法: $0 [frontend|backend|all]"
            exit 1
            ;;
    esac
    
    log_info "=========================================="
    log_info "停止完成"
    log_info "=========================================="
}

# 执行主函数
main "$@"

