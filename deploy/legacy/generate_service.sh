#!/bin/bash

# Legacy generator: uses the historical backend/.venv and dev-server layout.

set -e

APP_SERVICE_NAME="eai-app"
BACKEND_SERVICE_NAME="eai-backend"

APP_SERVICE_FILE="${APP_SERVICE_NAME}.service"
BACKEND_SERVICE_FILE="${BACKEND_SERVICE_NAME}.service"

APP_SERVICE_PATH="/etc/systemd/system/${APP_SERVICE_FILE}"
BACKEND_SERVICE_PATH="/etc/systemd/system/${BACKEND_SERVICE_FILE}"

echo "👉 获取当前用户..."
CURRENT_USER=$(logname 2>/dev/null || echo $SUDO_USER || whoami)
echo "👉 用户: $CURRENT_USER"

echo "👉 获取当前目录..."
BASE_DIR=$(pwd)
echo "👉 项目路径: $BASE_DIR"

# =========================
# 前端部分
# =========================

echo "👉 检测 pnpm..."
PNPM_PATH=$(which pnpm || true)
[ -z "$PNPM_PATH" ] && echo "❌ pnpm 未找到" && exit 1
echo "👉 pnpm: $PNPM_PATH"

echo "👉 生成前端 service..."

cat <<EOF > ${APP_SERVICE_FILE}
[Unit]
Description=EAI iDev 2.0 App Service (pnpm)
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
Group=${CURRENT_USER}

WorkingDirectory=${BASE_DIR}

ExecStart=/bin/bash -lc 'cd ${BASE_DIR} && ${PNPM_PATH} run dev'

Restart=always
RestartSec=5

Environment=NODE_ENV=development

[Install]
WantedBy=multi-user.target
EOF

# =========================
# 后端部分
# =========================

BACKEND_DIR="${BASE_DIR}/backend"
UVICORN_PATH="${BACKEND_DIR}/.venv/bin/uvicorn"

echo "👉 检测 backend..."

if [ ! -d "$BACKEND_DIR" ]; then
  echo "❌ backend 目录不存在"
  exit 1
fi

if [ ! -f "$UVICORN_PATH" ]; then
  echo "❌ 未找到 uvicorn: $UVICORN_PATH"
  exit 1
fi

echo "👉 生成后端 service..."

cat <<EOF > ${BACKEND_SERVICE_FILE}
[Unit]
Description=EAI Backend Service (FastAPI + Uvicorn)
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
Group=${CURRENT_USER}

WorkingDirectory=${BACKEND_DIR}

ExecStart=${UVICORN_PATH} app.main:app --host 0.0.0.0 --port 8000

Restart=always
RestartSec=5

Environment=PATH=${BACKEND_DIR}/.venv/bin:/usr/bin:/usr/local/bin:/bin

[Install]
WantedBy=multi-user.target
EOF

# =========================
# 安装 & 启动
# =========================

echo "👉 安装 services..."
sudo mv ${APP_SERVICE_FILE} ${APP_SERVICE_PATH}
sudo mv ${BACKEND_SERVICE_FILE} ${BACKEND_SERVICE_PATH}

echo "👉 重载 systemd..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

echo "👉 设置开机自启..."
sudo systemctl enable ${APP_SERVICE_NAME}
sudo systemctl enable ${BACKEND_SERVICE_NAME}

echo "👉 启动服务..."
sudo systemctl restart ${APP_SERVICE_NAME}
sudo systemctl restart ${BACKEND_SERVICE_NAME}

echo "👉 服务状态："
echo "================ APP ================"
sudo systemctl status ${APP_SERVICE_NAME} --no-pager

echo "================ BACKEND ================"
sudo systemctl status ${BACKEND_SERVICE_NAME} --no-pager
