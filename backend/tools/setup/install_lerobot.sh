#!/usr/bin/env bash
# 在已激活的虚拟环境 / conda 环境中安装 lerobot，避免 Linux 上编译 evdev。
# 用法：cd backend && ./tools/setup/install_lerobot.sh
# 或指定解释器：EAI_PYTHON=/path/to/python ./tools/setup/install_lerobot.sh
set -euo pipefail
PY="${EAI_PYTHON:-python3}"
exec "$PY" "$(dirname "$0")/install_lerobot.py"
