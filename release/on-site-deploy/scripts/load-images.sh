#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG_DIR="$ROOT/images"

for tar in eai-ide-local.tar eai-postgres-local.tar eai-minio-local.tar redis-7-alpine.tar; do
  path="$IMG_DIR/$tar"
  if [ ! -f "$path" ]; then
    echo "缺少: $path"
    exit 1
  fi
  echo "加载 $tar ..."
  docker load -i "$path"
done

echo "已加载镜像："
docker images | grep -E 'eai-|redis' || true
