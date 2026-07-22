#!/bin/bash

# 重启项目前端和后端脚本
# 使用方法: ./scripts/restart.sh [frontend|backend|all]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置（与 package.json 中 next dev -p 3001 保持一致）
FRONTEND_PORT=3001
BACKEND_PORT=${BACKEND_PORT:-8000}  # 如果后端存在，可配置端口
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"

# 创建必要的目录
mkdir -p "$LOG_DIR" "$PID_DIR"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/lib/frontend_dev.sh"

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

# 停止 RQ workers
stop_workers() {
    log_info "停止 RQ workers..."
    pkill -f "worker/start_worker.py --queues gpu_queue" 2>/dev/null || true
    pkill -f "worker/start_worker.py --queues cpu_queue" 2>/dev/null || true
    pkill -f "worker/start_worker.py --queues io_queue" 2>/dev/null || true
    pkill -f "worker/start_worker.py --queues collect_queue" 2>/dev/null || true
}

# 某队列是否已有 RQ worker 进程（与 backend app.main 中 pgrep 逻辑一致）
_rq_worker_running() {
    local q="$1"
    pgrep -f "start_worker.py --queues ${q}" >/dev/null 2>&1
}

# 启动 RQ workers（仅 USE_QUEUE=true 时）；已运行的队列不会重复启动
start_workers() {
    if [ "${USE_QUEUE:-false}" != "true" ] && [ "${USE_QUEUE:-false}" != "1" ]; then
        log_warn "USE_QUEUE 未开启，跳过 RQ workers 启动"
        return
    fi
    log_info "检查并启动 RQ workers（缺则启）..."
    local py_bin="$1"
    local Q_GPU="${QUEUE_ANNOTATION:-gpu_queue}"
    local Q_CPU="${QUEUE_CONVERSION:-cpu_queue}"
    local Q_BATCH="${QUEUE_BATCH:-io_queue}"
    local Q_COLLECT="${QUEUE_COLLECT:-collect_queue}"

    local q logf
    for q in "$Q_GPU" "$Q_CPU" "$Q_BATCH" "$Q_COLLECT"; do
        if _rq_worker_running "$q"; then
            log_info "RQ worker 已在运行（queue=$q），跳过"
            continue
        fi
        logf="rq-worker-${q}.log"
        nohup "$py_bin" worker/start_worker.py --queues "$q" >> "$LOG_DIR/$logf" 2>&1 &
        log_info "已启动 RQ worker queue=$q pid=$! 日志=$LOG_DIR/$logf"
    done
}

is_auto_start_workers_enabled() {
    local v="${AUTO_START_WORKERS:-true}"
    case "${v,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# 根据端口查找进程（优先 lsof 得到纯 PID，ss 输出格式因系统而异）
find_process_by_port() {
    local port=$1
    if command -v lsof > /dev/null 2>&1; then
        lsof -n -P -t -i:$port 2>/dev/null | head -1 || true
    elif command -v ss > /dev/null 2>&1; then
        ss -tlnp 2>/dev/null | grep ":$port " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true
    elif command -v netstat > /dev/null 2>&1; then
        netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | head -1 || true
    else
        log_warn "无法找到端口 $port 的进程（需要安装 lsof、ss 或 netstat）"
        echo ""
    fi
}

# 清理与验收逻辑见 scripts/lib/frontend_dev.sh

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
            # 如果还在运行，强制杀死
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
        stop_all_next_frontend
    elif [ "$name" = "backend" ]; then
        pkill -f "uvicorn app.main:app" 2>/dev/null || true
        pkill -f "uvicorn.*8000" 2>/dev/null || true
    fi
    
    log_info "$name 已停止"
}

# 启动前端（见 scripts/lib/frontend_dev.sh）
start_frontend() {
    log_info "启动前端 (Next.js dev server on port $FRONTEND_PORT)..."
    launch_frontend_dev_server
}

# 启动后端
start_backend() {
    log_info "启动后端 (FastAPI uvicorn on port $BACKEND_PORT)..."
    
    cd "$PROJECT_ROOT/backend"
    
    # 与执行本脚本时当前 Shell 已激活的环境一致（不硬编码 conda/venv 路径）
    # 可选：export EAI_PYTHON=/path/to/python 强制指定解释器
    local PY_BIN=""
    if [ -n "${EAI_PYTHON:-}" ] && [ -x "${EAI_PYTHON}" ]; then
        PY_BIN="${EAI_PYTHON}"
        log_info "使用 EAI_PYTHON 指定的 Python"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PY_BIN="${VIRTUAL_ENV}/bin/python"
        log_info "使用当前 venv (VIRTUAL_ENV)"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PY_BIN="${CONDA_PREFIX}/bin/python"
        log_info "使用当前 Conda 环境 (CONDA_PREFIX)"
    elif command -v python > /dev/null 2>&1; then
        PY_BIN="$(command -v python)"
        log_info "使用 PATH 中的 python（请先 conda activate / source venv，再运行本脚本）"
    elif command -v python3 > /dev/null 2>&1; then
        PY_BIN="$(command -v python3)"
        log_info "使用 PATH 中的 python3"
    fi

    if [ -z "$PY_BIN" ]; then
        log_error "未找到可用的 Python 解释器"
        return 1
    fi

    log_info "使用 Python: $PY_BIN"
    
    # 检查是否安装了依赖
    if ! "$PY_BIN" -c "import uvicorn" 2>/dev/null; then
        log_warn "后端依赖未安装，正在安装..."
        "$PY_BIN" -m pip install -q -r requirements.txt
    fi

    # 主库 Alembic 迁移（与 app 启动时 audit_logs 缺列补齐互为补充）
    ALEMBIC_BIN="$(dirname "$PY_BIN")/alembic"
    if [ -x "$ALEMBIC_BIN" ]; then
        log_info "执行数据库迁移: $ALEMBIC_BIN upgrade head"
        (cd "$PROJECT_ROOT/backend" && "$ALEMBIC_BIN" upgrade head) || log_warn "Alembic 未完全成功，应用启动时仍会尝试补齐 audit_logs 等结构"
    else
        log_info "未找到与当前 Python 同目录的 alembic 可执行文件；依赖应用启动时自动补齐 audit_logs 缺列"
    fi

    # 队列模式：若后端已配置 AUTO_START_WORKERS=true，则由后端启动时自动拉起 worker，避免重复启动。
    if is_auto_start_workers_enabled; then
        log_info "AUTO_START_WORKERS=true，交由后端启动时自动拉起 workers（跳过脚本手动启动）"
        stop_workers
    else
        stop_workers
        start_workers "$PY_BIN"
    fi

    # 启动后端服务（后台）
    nohup "$PY_BIN" -m uvicorn app.main:app \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --reload \
        >> "$LOG_DIR/backend.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/backend.pid"
    log_info "后端已启动 (PID: $pid)"
    log_info "日志文件: $LOG_DIR/backend.log"
    log_info "API 地址: http://127.0.0.1:$BACKEND_PORT"

    sleep 2
    if ps -p "$pid" > /dev/null 2>&1; then
        log_info "✅ 后端启动成功"
    else
        log_error "❌ 后端启动失败，请查看日志: $LOG_DIR/backend.log"
        exit 1
    fi
}

# 主函数
main() {
    local target=${1:-all}
    
    log_info "=========================================="
    log_info "重启项目服务"
    log_info "项目根目录: $PROJECT_ROOT"
    log_info "目标: $target"
    # 加载 .env，确保后端能读到 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
    if [ -f "$PROJECT_ROOT/.env" ]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
        log_info "已加载 .env"
    fi
    log_info "=========================================="
    
    case "$target" in
        frontend)
            stop_process "frontend" "$FRONTEND_PORT"
            sleep 1
            start_frontend
            ;;
        backend)
            stop_process "backend" "$BACKEND_PORT"
            stop_workers
            sleep 1
            start_backend
            ;;
        all)
            stop_process "frontend" "$FRONTEND_PORT"
            stop_process "backend" "$BACKEND_PORT"
            stop_workers
            sleep 2
            start_frontend
            start_backend
            ;;
        *)
            log_error "无效的参数: $target"
            echo "使用方法: $0 [frontend|backend|all]"
            exit 1
            ;;
    esac
    
    log_info "=========================================="
    log_info "重启完成"
    log_info "查看日志: tail -f $LOG_DIR/*.log"
    log_info "停止服务: ./scripts/stop.sh"
    log_info "=========================================="
}

# 执行主函数
main "$@"

