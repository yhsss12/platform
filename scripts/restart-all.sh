#!/usr/bin/env bash
#
# 重启整个项目（本机开发环境）
#   1. 停止前端、后端、RQ workers
#   2. 拉起 Docker 基础设施（PostgreSQL + MinIO）
#   3. 后台启动 FastAPI 后端
#   4. 后台启动 Next.js 前端
#
# 用法:
#   ./scripts/restart-all.sh
#   SKIP_DOCKER=1 ./scripts/restart-all.sh    # 不重启 Docker，仅重启前后端
#
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FRONTEND_PORT="${FRONTEND_PORT:-3001}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"
COMPOSE_FILE="docker-compose.postgres-minio.yml"
DOCKER_WAIT_SEC="${DOCKER_WAIT_SEC:-90}"

mkdir -p "$LOG_DIR" "$PID_DIR"

log_info()  { echo -e "${GREEN}[INFO]${NC} $*" >&2; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

load_env() {
  if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
    log_info "已加载 .env"
  else
    log_warn "未找到 .env，将使用默认环境变量"
  fi
  if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/backend/.env"
    set +a
    log_info "已加载 backend/.env"
  fi
}

stop_workers() {
  log_info "停止 RQ workers..."
  pkill -f "worker/start_worker.py --queues gpu_queue" 2>/dev/null || true
  pkill -f "worker/start_worker.py --queues cpu_queue" 2>/dev/null || true
  pkill -f "worker/start_worker.py --queues io_queue" 2>/dev/null || true
  pkill -f "worker/start_worker.py --queues collect_queue" 2>/dev/null || true
}

stop_app_services() {
  log_info "停止前端与后端..."
  if [ -x "$PROJECT_ROOT/scripts/stop.sh" ]; then
    bash "$PROJECT_ROOT/scripts/stop.sh" all || true
  else
    log_warn "scripts/stop.sh 不可用，尝试按端口结束进程"
    pkill -f "next dev" 2>/dev/null || true
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
  fi
  stop_workers
  sleep 1
}

restart_docker_infra() {
  if [ "${SKIP_DOCKER:-0}" = "1" ]; then
    log_warn "SKIP_DOCKER=1，跳过 Docker 基础设施"
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    log_warn "未安装 docker，跳过 PostgreSQL / MinIO"
    return 0
  fi

  cd "$PROJECT_ROOT"
  log_info "启动 Docker 基础设施 ($COMPOSE_FILE)..."
  docker compose -f "$COMPOSE_FILE" up -d

  log_info "等待 PostgreSQL 就绪（最多 ${DOCKER_WAIT_SEC}s）..."
  local i=0
  while [ "$i" -lt "$DOCKER_WAIT_SEC" ]; do
    local status
    status="$(docker inspect eai-postgres --format '{{.State.Health.Status}}' 2>/dev/null || echo "")"
    if [ "$status" = "healthy" ]; then
      log_info "PostgreSQL 已就绪 (healthy)"
      return 0
    fi
    if [ "$status" = "unhealthy" ]; then
      log_warn "PostgreSQL 健康检查为 unhealthy，继续等待..."
    fi
    sleep 2
    i=$((i + 2))
  done
  log_warn "PostgreSQL 健康检查超时，仍将继续启动应用（请检查: docker compose -f $COMPOSE_FILE ps)"
}

_python_has_backend_deps() {
  local py="$1"
  [ -x "$py" ] && "$py" -c "import uvicorn, torch" 2>/dev/null
}

resolve_python() {
  local py candidate
  # 仅向 stdout 输出解释器路径（供 py_bin="$(resolve_python)" 捕获）
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

run_alembic() {
  local py_bin="$1"
  local alembic_bin
  alembic_bin="$(dirname "$py_bin")/alembic"
  if [ -x "$alembic_bin" ]; then
    log_info "执行数据库迁移: alembic upgrade head"
    (cd "$PROJECT_ROOT/backend" && "$alembic_bin" upgrade head) \
      || log_warn "Alembic 未完全成功，应用启动时可能继续尝试补齐结构"
  else
    log_warn "未找到 alembic，跳过迁移"
  fi
}

start_backend() {
  local py_bin
  if ! py_bin="$(resolve_python)"; then
    log_error "未找到可用的 Python（需 uvicorn + torch）"
    log_error "请执行: export EAI_PYTHON=$HOME/miniconda3/envs/IDE/bin/python"
    log_error "或先: conda activate IDE，再运行本脚本"
    return 1
  fi
  log_info "使用 Python: $py_bin"

  if ! "$py_bin" -c "import uvicorn" 2>/dev/null; then
    log_warn "安装后端依赖..."
    "$py_bin" -m pip install -q -r "$PROJECT_ROOT/backend/requirements.txt"
  fi

  run_alembic "$py_bin"

  cd "$PROJECT_ROOT/backend"
  log_info "启动后端 (uvicorn :$BACKEND_PORT)..."
  nohup "$py_bin" -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --reload \
    >> "$LOG_DIR/backend.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_DIR/backend.pid"
  log_info "后端已启动 PID=$pid，日志: $LOG_DIR/backend.log"

  sleep 2
  if ! ps -p "$pid" >/dev/null 2>&1; then
    log_error "后端启动失败，请查看: tail -n 50 $LOG_DIR/backend.log"
    return 1
  fi
  log_info "API: http://127.0.0.1:$BACKEND_PORT"
}

detect_package_manager() {
  cd "$PROJECT_ROOT"
  if [ -f yarn.lock ] && command -v yarn >/dev/null 2>&1; then
    echo "yarn"
  elif command -v pnpm >/dev/null 2>&1; then
    echo "pnpm"
  elif command -v npm >/dev/null 2>&1; then
    echo "npm"
  else
    echo ""
  fi
}

start_frontend() {
  log_info "启动前端 (yarn dev -H 0.0.0.0 -p $FRONTEND_PORT)..."
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/lib/frontend_dev.sh"
  launch_frontend_dev_server
}

print_summary() {
  log_info "=========================================="
  log_info "项目重启完成"
  log_info "  前端: http://localhost:$FRONTEND_PORT"
  log_info "  后端: http://127.0.0.1:$BACKEND_PORT"
  log_info "  日志: tail -f $LOG_DIR/*.log"
  log_info "  停止: ./scripts/stop.sh all"
  log_info "=========================================="
}

main() {
  log_info "=========================================="
  log_info "重启整个项目"
  log_info "项目根目录: $PROJECT_ROOT"
  log_info "=========================================="

  load_env
  stop_app_services
  restart_docker_infra
  start_backend
  start_frontend
  print_summary
}

main "$@"
