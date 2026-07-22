#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_URL="${PUBLIC_BASE_URL:-http://127.0.0.1:8000}"
FRONT_URL="${FRONT_URL:-http://127.0.0.1:3001}"

echo "=== 容器状态 ==="
docker ps --filter name=eai- --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

echo ""
echo "=== API 健康检查 ==="
curl -sf "${API_URL%/}/health" | head -c 200
echo ""

echo ""
echo "=== 前端端口 ==="
if curl -sf -o /dev/null -w "%{http_code}" "$FRONT_URL" | grep -qE '200|307|308'; then
  echo "前端可达: $FRONT_URL"
else
  echo "警告: 无法访问 $FRONT_URL（检查防火墙或 FRONT_URL）"
fi

echo ""
echo "=== 交付镜像内容抽检（eai-app）==="
if docker ps --format '{{.Names}}' | grep -q '^eai-app$'; then
  for p in /app/.env /app/src /app/backend/tests; do
    if docker exec eai-app test -e "$p" 2>/dev/null; then
      echo "[FAIL] 不应存在: $p"
      exit 1
    else
      echo "[OK] 不存在 $p"
    fi
  done
  if docker exec eai-app test -d /app/.next; then
    echo "[OK] 存在 /app/.next"
  else
    echo "[FAIL] 缺少 /app/.next"
    exit 1
  fi
else
  echo "跳过容器内检查（eai-app 未运行）"
fi

echo ""
echo "验证完成。"
