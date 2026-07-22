#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
  ROOT="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../docker-compose.yml" ]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "错误: 未找到 docker-compose.yml，请在部署包根目录或其 scripts/ 下执行。"
  exit 1
fi
cd "$ROOT"

if [ ! -f .env ] && [ -f scripts/.env ]; then
  echo "提示: 发现 scripts/.env，已复制到 $ROOT/.env"
  cp scripts/.env .env
fi

if [ ! -f .env ]; then
  echo "请先在本目录（与 docker-compose.yml 同级）准备 .env："
  echo "  cp app.env.example .env && 编辑 .env"
  exit 1
fi

echo "使用环境文件: $ROOT/.env"
echo "工作目录: $ROOT"

if [ ! -f eai_ide_backup.sql ]; then
  echo "警告: 未找到 eai_ide_backup.sql，Postgres 首次初始化可能无种子数据。"
  echo "可从仓库根目录复制: cp /path/to/eai-idev2.0-main/eai_ide_backup.sql ."
fi

echo "启动 PostgreSQL + MinIO ..."
docker compose -f docker-compose.postgres-minio.yml up -d

echo "等待 Postgres 健康 ..."
for i in $(seq 1 60); do
  if docker inspect -f '{{.State.Health.Status}}' eai-postgres 2>/dev/null | grep -q healthy; then
    echo "Postgres 已 healthy"
    break
  fi
  sleep 2
done

echo "启动 Redis + App + Worker ..."
docker compose up -d

docker compose ps
echo "完成。请运行: ./scripts/verify.sh"
