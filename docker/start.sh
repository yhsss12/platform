#!/bin/bash
#
# 启动模式（默认 all，与历史单容器行为一致）：
#   all    — 前台 Next.js 生产模式 + 后台 uvicorn
#   api    — 仅 FastAPI（请配 USE_QUEUE=true、AUTO_START_WORKERS=false）
#   worker — 仅 RQ Worker（请配 USE_QUEUE=true）
#
set -e

MODE="${START_MODE:-all}"
FRONTEND_PORT="${FRONTEND_PORT:-3001}"

# host 网络：检测宿主机端口是否已被占用（含 IPv6 :::PORT）
port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    if ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
      return 0
    fi
    if ss -ltn 2>/dev/null | grep -qE "[:.]${port}([^0-9]|$)"; then
      return 0
    fi
    return 1
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | grep -qE ":${port}([^0-9]|$)"
    return $?
  fi
  return 1
}

print_port_conflict_help() {
  local port="$1"
  echo "ERROR: 端口 ${port} 已被占用（compose 使用 network_mode: host，占用的是宿主机端口）"
  echo "  在宿主机执行："
  echo "    ss -ltnp | grep :${port}"
  echo "    ./scripts/free_frontend_port.sh"
  echo "  或先停开发前端： pkill -f 'next dev' ; pkill -f 'next start'"
  echo "  然后： docker compose -f docker-compose.yml restart app"
}

if [ "$MODE" = "api" ]; then
  echo "START_MODE=api：仅启动后端 uvicorn（端口 8000）"
  cd /app/backend
  exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
fi

if [ "$MODE" = "worker" ]; then
  echo "START_MODE=worker：启动 RQ Worker（队列: ${RQ_WORKER_QUEUES:-gpu_queue,cpu_queue,io_queue,collect_queue}）"
  cd /app/backend
  exec python3 worker/start_worker.py \
    --queues "${RQ_WORKER_QUEUES:-gpu_queue,cpu_queue,io_queue,collect_queue}"
fi

if port_in_use "${FRONTEND_PORT}"; then
  print_port_conflict_help "${FRONTEND_PORT}"
  exit 1
fi

echo "START_MODE=all：启动后端（后台）+ 前端生产模式（端口 ${FRONTEND_PORT}）..."
cd /app/backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
UV_PID=$!

echo "等待 uvicorn 监听 127.0.0.1:8000（PID ${UV_PID}）..."
READY=0
for i in $(seq 1 180); do
  if curl -sf -o /dev/null "http://127.0.0.1:8000/health"; then
    echo "uvicorn 已就绪（探活 ${i} 次）"
    READY=1
    break
  fi
  if ! kill -0 "${UV_PID}" 2>/dev/null; then
    echo "ERROR: uvicorn 进程已退出，请查看上方 Python 报错"
    wait "${UV_PID}" 2>/dev/null || true
    exit 1
  fi
  sleep 0.5
done
if [ "$READY" != "1" ]; then
  echo "ERROR: 超时仍未连上 http://127.0.0.1:8000/health"
  kill "${UV_PID}" 2>/dev/null || true
  exit 1
fi

# 启动前再次确认（避免探活期间其它进程抢占端口）
if port_in_use "${FRONTEND_PORT}"; then
  print_port_conflict_help "${FRONTEND_PORT}"
  kill "${UV_PID}" 2>/dev/null || true
  exit 1
fi

echo "启动前端（next start -p ${FRONTEND_PORT}）..."
cd /app
exec npx next start -p "${FRONTEND_PORT}"
