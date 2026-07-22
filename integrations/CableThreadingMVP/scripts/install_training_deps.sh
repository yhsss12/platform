#!/usr/bin/env bash
# 将 robomimic BC 训练所需依赖安装到 cable-threading-mvp conda 环境。
# 训练子进程设置 PYTHONNOUSERSITE=1，依赖必须位于 env site-packages，而非 ~/.local。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MVP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${CABLE_THREADING_ENV:-cable-threading-mvp}"

if command -v conda >/dev/null 2>&1; then
  PY="$(conda run -n "$ENV_NAME" which python 2>/dev/null || true)"
fi
if [[ -z "${PY:-}" ]]; then
  PY="${CABLE_THREADING_PYTHON:-/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python}"
fi
if [[ ! -x "$PY" ]]; then
  echo "错误: 未找到 Python: $PY" >&2
  echo "请先创建环境: conda env create -f $MVP_ROOT/environment.yml" >&2
  exit 1
fi

echo "使用 Python: $PY"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$MVP_ROOT/requirements.txt"
"$PY" -m pip install -e "$MVP_ROOT" 2>/dev/null || true

echo "验证训练依赖（PYTHONNOUSERSITE=1）..."
PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import psutil, tqdm, sympy, matplotlib
import robomimic
from robomimic.scripts.train import train  # noqa: F401
print("ok: psutil", psutil.__version__)
print("ok: robomimic", getattr(robomimic, "__version__", "imported"))
PY

echo "训练依赖安装完成。"
