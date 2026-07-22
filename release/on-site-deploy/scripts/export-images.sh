#!/usr/bin/env bash
# 在开发机执行：将已构建的 :local 镜像导出到现场包 images/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG_DIR="$ROOT/images"
mkdir -p "$IMG_DIR"

REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine}"

need_image() {
  if ! docker image inspect "$1" >/dev/null 2>&1; then
    echo "缺少镜像: $1 — 请先在项目根目录 docker compose build"
    exit 1
  fi
}

need_image eai-ide:local
need_image eai-postgres:local
need_image eai-minio:local
need_image "$REDIS_IMAGE"

echo "导出到 $IMG_DIR ..."
docker save eai-ide:local -o "$IMG_DIR/eai-ide-local.tar"
docker save eai-postgres:local -o "$IMG_DIR/eai-postgres-local.tar"
docker save eai-minio:local -o "$IMG_DIR/eai-minio-local.tar"
docker save "$REDIS_IMAGE" -o "$IMG_DIR/redis-7-alpine.tar"

ls -lh "$IMG_DIR"/*.tar
echo "完成。请将整个 on-site-deploy 目录（含 images/）拷贝至客户现场。"
