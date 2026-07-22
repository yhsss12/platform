# Shared Next.js dev frontend launch — always 0.0.0.0:3001, never default port 3000.
# Sourced by scripts/restart.sh, scripts/start.sh, scripts/restart-all.sh

: "${FRONTEND_PORT:=3001}"
: "${BACKEND_PORT:=8000}"
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
: "${LOG_DIR:=$PROJECT_ROOT/logs}"
: "${PID_DIR:=$PROJECT_ROOT/.pids}"

# 停止所有 Next dev / next-server，并清理误启动的 3000
stop_all_next_frontend() {
    log_info "停止所有 next dev / next-server 进程..."
    pkill -f "next dev" 2>/dev/null || true
    pkill -f "next-server" 2>/dev/null || true
    pkill -f "next start" 2>/dev/null || true
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:3000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        lsof -ti:"$FRONTEND_PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    fi
    sleep 1
}

# 清理 dev 缓存
cleanup_frontend_dev_cache() {
    log_info "清理 Next.js dev 缓存 (.next / node_modules/.cache / .turbo)..."
    rm -rf "$PROJECT_ROOT/.next"
    rm -rf "$PROJECT_ROOT/node_modules/.cache"
    rm -rf "$PROJECT_ROOT/.turbo"
    rm -rf "$PROJECT_ROOT/.vercel"
    rm -rf "$PROJECT_ROOT/frontend/.next" \
           "$PROJECT_ROOT/frontend/node_modules/.cache" \
           "$PROJECT_ROOT/frontend/.turbo" 2>/dev/null || true
}

# 检测本机 LAN IP
detect_lan_host() {
    if [ -n "${FRONTEND_LAN_HOST:-}" ]; then
        echo "$FRONTEND_LAN_HOST"
        return
    fi
    local ip=""
    if command -v ip >/dev/null 2>&1; then
        ip=$(ip -4 route get 1.8.8.8 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')
    fi
    echo "${ip:-172.18.0.101}"
}

# 启动前端：固定 INTERNAL_API_URL + yarn dev -H 0.0.0.0 -p 3001
launch_frontend_dev_server() {
    cd "$PROJECT_ROOT"
    mkdir -p "$LOG_DIR" "$PID_DIR"

    stop_all_next_frontend
    cleanup_frontend_dev_cache

    export INTERNAL_API_URL="${INTERNAL_API_URL:-http://127.0.0.1:${BACKEND_PORT}}"
    export FRONTEND_LAN_HOST="$(detect_lan_host)"
    export HOSTNAME=0.0.0.0
    export NEXT_PUBLIC_WORKSPACE_DEMO_REAL_ONLY="${NEXT_PUBLIC_WORKSPACE_DEMO_REAL_ONLY:-true}"

    if ! command -v yarn >/dev/null 2>&1; then
        log_error "未找到 yarn，无法启动前端"
        return 1
    fi
    if [ ! -d "node_modules" ]; then
        log_warn "node_modules 不存在，正在安装依赖..."
        yarn install
    fi

    # 覆盖写入，避免 >> 追加导致日志混入历史 next dev (3000) 记录
    : > "$LOG_DIR/frontend.log"

    log_info "前端监听: 0.0.0.0:${FRONTEND_PORT}  LAN: http://${FRONTEND_LAN_HOST}:${FRONTEND_PORT}"
    log_info "启动命令: INTERNAL_API_URL=${INTERNAL_API_URL} yarn dev -H 0.0.0.0 -p ${FRONTEND_PORT}"

    INTERNAL_API_URL="${INTERNAL_API_URL}" \
        nohup yarn dev -H 0.0.0.0 -p "$FRONTEND_PORT" > "$LOG_DIR/frontend.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/frontend.pid"

    log_info "前端已启动 (PID: $pid)，日志: $LOG_DIR/frontend.log"

    sleep 3
    if ! ps -p "$pid" >/dev/null 2>&1; then
        log_error "❌ 前端进程已退出"
        tail -120 "$LOG_DIR/frontend.log" >&2 || true
        return 1
    fi

    verify_frontend_listening
}

# 验收：3001 监听 + curl -I 本机与 LAN
verify_frontend_listening() {
    local lan_host max_tries=12 i code lan_code

    lan_host="$(detect_lan_host)"
    log_info "等待 Next.js 就绪并验证 3001 ..."

    for i in $(seq 1 "$max_tries"); do
        if ss -ltnp 2>/dev/null | grep -q ":${FRONTEND_PORT} "; then
            code=$(curl -s -o /dev/null -w '%{http_code}' -I "http://127.0.0.1:${FRONTEND_PORT}/login" 2>/dev/null || echo "000")
            lan_code=$(curl -s -o /dev/null -w '%{http_code}' -I "http://${lan_host}:${FRONTEND_PORT}/login" 2>/dev/null || echo "000")
            if [ "$code" = "200" ] && [ "$lan_code" = "200" ]; then
                log_info "✅ 前端验收通过 (127.0.0.1/login=$code LAN/login=$lan_code, try=$i)"
                log_info "Local access: OK  (http://127.0.0.1:${FRONTEND_PORT})"
                log_info "LAN access: OK   (http://${lan_host}:${FRONTEND_PORT})"
                ss -ltnp 2>/dev/null | grep ":${FRONTEND_PORT} " || true
                if ss -ltnp 2>/dev/null | grep -q ':3000 '; then
                    log_warn "⚠ 检测到 3000 端口仍有监听，请检查是否有残留 next dev"
                else
                    log_info "✅ 3000 端口无残留"
                fi
                return 0
            fi
            log_warn "验收 try=$i: 127.0.0.1=$code LAN=$lan_code，等待重试..."
        else
            log_warn "验收 try=$i: 3001 尚未监听，等待重试..."
        fi
        sleep 2
    done

    log_error "❌ 前端验收失败：3001 未就绪或 /login 非 200"
    log_error "ss -ltnp | grep 3001:"
    ss -ltnp 2>/dev/null | grep ':3001' >&2 || echo "(无 3001 监听)" >&2
    log_error "最近 frontend.log (120 行):"
    tail -120 "$LOG_DIR/frontend.log" >&2 || true
    return 1
}
