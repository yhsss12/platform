#!/usr/bin/env bash
# 停止全套服务（不删数据卷）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "停止 App / Worker / Redis ..."
docker compose down

echo "停止 PostgreSQL / MinIO ..."
docker compose -f docker-compose.postgres-minio.yml down

docker ps --filter name=eai- --format 'table {{.Names}}\t{{.Status}}' || true
echo "已停止（数据卷保留）。"
