#!/bin/bash

# 启动项目前端和后端脚本（不停止现有进程）
# 使用方法: ./scripts/start.sh [frontend|backend|all]

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
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"

# 创建必要的目录
mkdir -p "$LOG_DIR" "$PID_DIR"

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

# 检查端口是否被占用
check_port() {
    local port=$1
    if command -v lsof > /dev/null 2>&1; then
        if lsof -ti:$port > /dev/null 2>&1; then
            return 0
        fi
    elif command -v netstat > /dev/null 2>&1; then
        if netstat -tln 2>/dev/null | grep -q ":$port "; then
            return 0
        fi
    elif command -v ss > /dev/null 2>&1; then
        if ss -tln 2>/dev/null | grep -q ":$port "; then
            return 0
        fi
    fi
    return 1
}

# 启动前端（与 restart.sh 共用 scripts/lib/frontend_dev.sh）
start_frontend() {
    if check_port "$FRONTEND_PORT"; then
        log_warn "端口 $FRONTEND_PORT 已被占用，前端可能已在运行"
        return
    fi

    log_info "启动前端 (Next.js dev server on port $FRONTEND_PORT)..."
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/scripts/lib/frontend_dev.sh"
    launch_frontend_dev_server
}

_python_has_backend_deps() {
    local py="$1"
    [ -x "$py" ] && "$py" -c "import uvicorn, torch" 2>/dev/null
}

resolve_python() {
    local candidate
    for candidate in \
        "${EAI_PYTHON:-}" \
        "$HOME/miniconda3/envs/IDE/bin/python" \
        "$HOME/miniconda3/envs/aloha/bin/python" \
        "$PROJECT_ROOT/backend/.venv/bin/python"; do
        [ -n "$candidate" ] || continue
        if _python_has_backend_deps "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# 启动后端
start_backend() {
    if check_port "$BACKEND_PORT"; then
        log_warn "端口 $BACKEND_PORT 已被占用，后端可能已在运行"
        return
    fi

    local py_bin
    if ! py_bin="$(resolve_python)"; then
        log_error "未找到可用的 Python（需 uvicorn + torch）"
        log_error "请设置: export EAI_PYTHON=\$HOME/miniconda3/envs/IDE/bin/python"
        exit 1
    fi
    log_info "使用 Python: $py_bin"

    if ! "$py_bin" -c "import uvicorn" 2>/dev/null; then
        log_warn "安装后端依赖..."
        "$py_bin" -m pip install -q -r "$PROJECT_ROOT/backend/requirements.txt"
    fi

    cd "$PROJECT_ROOT/backend"
    log_info "启动后端 (uvicorn :$BACKEND_PORT)..."
    nohup "$py_bin" -m uvicorn app.main:app \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --reload \
        >> "$LOG_DIR/backend.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/backend.pid"

    log_info "后端已启动 (PID: $pid)"
    log_info "日志文件: $LOG_DIR/backend.log"
    log_info "API: http://127.0.0.1:$BACKEND_PORT"

    sleep 3
    if ps -p "$pid" > /dev/null 2>&1; then
        log_info "✅ 后端启动成功"
    else
        log_error "❌ 后端启动失败，请查看日志: $LOG_DIR/backend.log"
        exit 1
    fi
}

# 加载 .env（EAI_PYTHON、DATABASE_URL 等）
load_env() {
    if [ -f "$PROJECT_ROOT/.env" ]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.env"
        set +a
        log_info "已加载 .env"
    fi
}

# 主函数
main() {
    local target=${1:-all}
    
    log_info "=========================================="
    log_info "启动项目服务"
    log_info "项目根目录: $PROJECT_ROOT"
    log_info "目标: $target"
    log_info "=========================================="

    load_env
    
    case "$target" in
        frontend)
            start_frontend
            ;;
        backend)
            start_backend
            ;;
        all)
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
    log_info "启动完成"
    log_info "查看日志: tail -f $LOG_DIR/*.log"
    log_info "停止服务: ./scripts/stop.sh"
    log_info "=========================================="
}

# 执行主函数
main "$@"


