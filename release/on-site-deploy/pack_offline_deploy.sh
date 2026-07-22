#!/usr/bin/env bash
# 打包「文件夹 + Docker 镜像 tar」离线部署包（在项目根目录执行）
#
# 用法：
#   export NEXT_PUBLIC_API_URL=http://192.168.1.100:8000   # 客户浏览器能访问的 API 地址
#   ./scripts/release/pack_offline_deploy.sh
#   ./scripts/release/pack_offline_deploy.sh /tmp/eai-ide-offline
#
# 产物目录结构：
#   eai-ide-offline/
#   ├── DEPLOY.md              # 现场操作说明
#   ├── app.env.example
#   ├── docker-compose.yml
#   ├── docker-compose.postgres-minio.yml
#   ├── eai_ide_backup.sql     # 若仓库存在则复制
#   ├── images/
#   │   ├── eai-ide-local.tar
#   │   ├── eai-postgres-local.tar
#   │   ├── eai-minio-local.tar
#   │   └── redis-7-alpine.tar
#   ├── install-and-start.sh   # 现场一键入口（根目录）
#   └── scripts/
#       ├── install-and-start.sh
#       ├── load-images.sh
#       ├── start-all.sh
#       ├── stop-all.sh
#       └── verify.sh
#
# 将整个 eai-ide-offline/ 目录拷贝到 U 盘或目标机即可。

set -euo pipefail

# 支持两种调用位置：
# 1. 项目根的 scripts/release/pack_offline_deploy.sh（REPO_ROOT = dirname "$0"/../..）
# 2. release/on-site-deploy/pack_offline_deploy.sh（REPO_ROOT = dirname "$0"/../..）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../../package.json" ]; then
  # 在 release/on-site-deploy/ 下调用
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [ -f "$SCRIPT_DIR/../package.json" ]; then
  # 在 scripts/ 下调用
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "错误: 无法定位项目根目录（未找到 package.json）"
  exit 1
fi
cd "$REPO_ROOT"

DEST="${1:-$REPO_ROOT/dist/eai-ide-offline}"
TEMPLATE="$REPO_ROOT/release/on-site-deploy"

if [ -z "${NEXT_PUBLIC_API_URL:-}" ] && [ -f .env ]; then
  NEXT_PUBLIC_API_URL="$(grep -E '^[[:space:]]*NEXT_PUBLIC_API_URL=' .env | head -1 | cut -d= -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')"
  export NEXT_PUBLIC_API_URL
fi
if [ -z "${NEXT_PUBLIC_API_URL:-}" ]; then
  echo "错误: 请设置客户现场 API 地址，例如："
  echo "  export NEXT_PUBLIC_API_URL=http://192.168.1.100:8000"
  exit 1
fi

SKIP_BUILD="${SKIP_BUILD:-0}"

echo "=========================================="
echo " EAI IDE 离线部署包打包"
echo " NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL"
echo " 输出目录: $DEST"
echo "=========================================="

export PYARMOR_ENABLED=true
export PYARMOR_SKIP_FILES="${PYARMOR_SKIP_FILES:-mcap_converter.py,flexible_mcap_to_hdf5.py,routes_data_assets.py}"
export PYARMOR_MAX_LINES="${PYARMOR_MAX_LINES:-2000}"
export PYARMOR_OBF_CODE="${PYARMOR_OBF_CODE:-1}"
export CYTHON_ENABLED=true

echo "NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL"
echo "CYTHON_ENABLED=$CYTHON_ENABLED"
echo "PYARMOR_ENABLED=$PYARMOR_ENABLED PYARMOR_SKIP_FILES=$PYARMOR_SKIP_FILES PYARMOR_MAX_LINES=$PYARMOR_MAX_LINES PYARMOR_OBF_CODE=$PYARMOR_OBF_CODE"

if [ "$SKIP_BUILD" = "1" ]; then
  echo "[1/5] 跳过构建（SKIP_BUILD=1），使用本地已有镜像 ..."
  for img in eai-ide:local eai-postgres:local eai-minio:local; do
    docker image inspect "$img" >/dev/null
  done
else
  echo "[1/5] 构建镜像 ..."
  docker compose -f docker-compose.postgres-minio.yml build
  docker compose -f docker-compose.yml build
fi

echo "[2/5] 验证 eai-ide:local ..."
chmod +x scripts/release/verify_release_image.sh
./scripts/release/verify_release_image.sh eai-ide:local

echo "[3/5] 组装目录 $DEST ..."
rm -rf "$DEST"
mkdir -p "$DEST/images" "$DEST/scripts"

for f in app.env.example docker-compose.yml docker-compose.postgres-minio.yml README.md; do
  if [ -f "$TEMPLATE/$f" ]; then
    cp "$TEMPLATE/$f" "$DEST/"
  else
    echo "  警告: 缺少 $TEMPLATE/$f，跳过复制"
  fi
done
cp "$TEMPLATE/scripts/"*.sh "$DEST/scripts/"
chmod +x "$DEST/scripts/"*.sh
for s in install-and-start.sh start-all.sh stop-all.sh; do
  if [ -f "$TEMPLATE/$s" ]; then
    cp "$TEMPLATE/$s" "$DEST/$s"
    chmod +x "$DEST/$s"
  elif [ -f "$TEMPLATE/scripts/$s" ]; then
    cp "$TEMPLATE/scripts/$s" "$DEST/$s"
    chmod +x "$DEST/$s"
  fi
done
# 根目录入口脚本已由上面的 for 循环复制，无需重复处理

if [ -f "$REPO_ROOT/eai_ide_backup.sql" ]; then
  cp "$REPO_ROOT/eai_ide_backup.sql" "$DEST/"
  echo "  已复制 eai_ide_backup.sql"
else
  echo "  提示: 未找到 eai_ide_backup.sql，现场首次 PG 为空库"
fi

echo "[4/5] 导出 Docker 镜像 tar ..."
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine}"
for img in eai-ide:local eai-postgres:local eai-minio:local "$REDIS_IMAGE"; do
  docker image inspect "$img" >/dev/null
done
docker save eai-ide:local -o "$DEST/images/eai-ide-local.tar"
docker save eai-postgres:local -o "$DEST/images/eai-postgres-local.tar"
docker save eai-minio:local -o "$DEST/images/eai-minio-local.tar"
docker save "$REDIS_IMAGE" -o "$DEST/images/redis-7-alpine.tar"

API_URL="$NEXT_PUBLIC_API_URL"
# 从 URL 提取 host 用于模板提示
SERVER_HINT="${API_URL#http://}"
SERVER_HINT="${SERVER_HINT#https://}"
SERVER_HINT="${SERVER_HINT%%/*}"

cat > "$DEST/DEPLOY.md" <<EOF
# EAI IDE 离线部署（文件夹 + 镜像）

本目录为完整现场包：**无需源码、无需 build**，仅需 Docker（Linux + host 网络）。

构建时前端 API 地址已写入镜像：\`${API_URL}\`

## 目标机要求

- Linux，已安装 Docker 与 Docker Compose v2
- 端口未被占用：5432、9000、9001、6379、8000、3001
- 勿在宿主机同时运行 \`npm run dev\`（会占用 3001）

## 部署步骤

\`\`\`bash
# 1. 拷贝整个目录到目标机，例如：
sudo mkdir -p /opt/eai-ide
sudo cp -a ./eai-ide-offline/* /opt/eai-ide/
cd /opt/eai-ide

# 2. 载入镜像
./scripts/load-images.sh

# 3. 配置环境（必改密码与 IP）
cp app.env.example .env
# 编辑 .env：将 CHANGE_ME_SERVER_IP 改为 ${SERVER_HINT}，并修改数据库/JWT/MinIO 密码

# 4. 一键启动（自动 load 镜像 + compose up）
./install-and-start.sh

# 5. 验证
./scripts/verify.sh
curl http://127.0.0.1:8000/health
\`\`\`

浏览器访问：\`http://${SERVER_HINT}:3001\`

## 停止

\`\`\`bash
docker compose down
docker compose -f docker-compose.postgres-minio.yml down
\`\`\`

## 镜像列表

| 文件 | 镜像名 |
|------|--------|
| images/eai-ide-local.tar | eai-ide:local |
| images/eai-postgres-local.tar | eai-postgres:local |
| images/eai-minio-local.tar | eai-minio:local |
| images/redis-7-alpine.tar | redis:7-alpine |

详细说明见 README.md。
EOF

ARCHIVE="${DEST}.tar.gz"
echo "[5/5] 压缩归档（可选）..."
tar -czf "$ARCHIVE" -C "$(dirname "$DEST")" "$(basename "$DEST")"

echo ""
echo "=========================================="
echo " 打包完成"
echo " 目录: $DEST"
echo " 压缩包: $ARCHIVE"
echo " 体积:"
du -sh "$DEST" "$ARCHIVE"
ls -lh "$DEST/images"/*.tar
echo ""
echo "请将整个目录或 tar.gz 拷贝到目标机后，按 DEPLOY.md 操作。"
echo "=========================================="
