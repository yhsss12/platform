#!/usr/bin/env bash
# 现场一键：检查 .env → 按需 load 镜像 → 启动全套服务
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Compose 的 env_file 相对于本目录（与 docker-compose.yml 同级），不是 scripts/
if [ ! -f .env ] && [ -f scripts/.env ]; then
  echo "提示: 发现 scripts/.env，已复制到部署包根目录 .env（Compose 只读根目录 .env）"
  cp scripts/.env .env
fi

if [ ! -f .env ]; then
  if [ -f app.env.example ]; then
    echo "未找到 .env，已从 app.env.example 复制模板。"
    cp app.env.example .env
    echo "请将 .env 放在本目录（与 docker-compose.yml 同级），勿放在 scripts/ 下。"
    echo "编辑 .env（IP、数据库/JWT/MinIO 密码等）后，再重新执行本脚本。"
    exit 1
  fi
  echo "错误: 缺少 .env 与 app.env.example"
  exit 1
fi

need_load=0
for img in eai-ide:local eai-postgres:local eai-minio:local; do
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    need_load=1
    break
  fi
done

if [ "$need_load" = "1" ]; then
  echo "本地尚无交付镜像，正在从 images/*.tar 加载 ..."
  "$ROOT/scripts/load-images.sh"
else
  echo "交付镜像已存在，跳过 docker load。"
fi

exec "$ROOT/start-all.sh"
