#!/usr/bin/env bash
# 在项目根目录构建全套 :local 交付镜像（发布工具）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  echo "提示: 根目录存在 .env（已被 .dockerignore 排除，不会 COPY 进镜像）。"
fi

export NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://127.0.0.1:8000}"
export PYARMOR_ENABLED="${PYARMOR_ENABLED:-true}"
export PYARMOR_SKIP_FILES="${PYARMOR_SKIP_FILES:-mcap_converter.py,flexible_mcap_to_hdf5.py}"
export PYARMOR_OBF_CODE="${PYARMOR_OBF_CODE:-1}"
export CYTHON_ENABLED="${CYTHON_ENABLED:-true}"
echo "NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL"
echo "CYTHON_ENABLED=$CYTHON_ENABLED"
echo "PYARMOR_ENABLED=$PYARMOR_ENABLED PYARMOR_SKIP_FILES=$PYARMOR_SKIP_FILES"

docker compose -f docker-compose.postgres-minio.yml build
docker compose build

echo "构建完成。验证:"
"$REPO_ROOT/scripts/release/verify_release_image.sh" eai-ide:local
