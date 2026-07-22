#!/usr/bin/env bash
# 释放 FRONTEND_PORT（默认 3001），供 docker compose host 网络下的 eai-app 使用。
set -euo pipefail

PORT="${FRONTEND_PORT:-3001}"

echo "检查端口 ${PORT} ..."
if command -v ss >/dev/null 2>&1; then
  ss -ltnp "sport = :${PORT}" 2>/dev/null || ss -ltnp 2>/dev/null | grep ":${PORT}" || true
fi

echo "停止常见 Next 开发/生产进程 ..."
pkill -f "next dev -p ${PORT}" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
pkill -f "next start -p ${PORT}" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true
if command -v lsof >/dev/null 2>&1; then
  lsof -ti:3000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

sleep 1
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  echo "WARN: 端口 ${PORT} 仍被占用，请手动处理： ss -ltnp | grep :${PORT}"
  exit 1
fi

echo "端口 ${PORT} 已释放。可执行： docker compose -f docker-compose.yml up -d app"
