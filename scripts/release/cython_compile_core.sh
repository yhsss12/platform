#!/usr/bin/env bash
# 将超大核心模块编译为 Cython 扩展（.so），并从交付树中删除对应 .py 明文。
#
# 环境变量：
#   CYTHON_ENABLED  默认 true；false 时保留 .py
#
# 用法（Docker builder 内）：
#   ./scripts/release/cython_compile_core.sh /build

set -euo pipefail

BUILD_ROOT="$(cd "${1:?BUILD_ROOT required}" && pwd)"
export BUILD_ROOT
CYTHON_ENABLED="${CYTHON_ENABLED:-true}"

MCAP_PY="${BUILD_ROOT}/backend/app/services/mcap_converter.py"
RELMAN_PY="${BUILD_ROOT}/scripts/relman/flexible_mcap_to_hdf5.py"

if [[ "${CYTHON_ENABLED}" != "true" && "${CYTHON_ENABLED}" != "1" ]]; then
  echo "[cython] CYTHON_ENABLED=${CYTHON_ENABLED}，跳过 .so 编译"
  exit 0
fi

for f in "${MCAP_PY}" "${RELMAN_PY}"; do
  if [[ ! -f "$f" ]]; then
    echo "[cython] 缺少源文件: $f"
    exit 1
  fi
done

echo "[cython] BUILD_ROOT=${BUILD_ROOT}"
python3 -m pip install -q "cython>=3.0,<4"

compile_and_strip() {
  local py_file="$1"
  local mod_dir mod_base
  mod_dir="$(dirname "$py_file")"
  mod_base="$(basename "$py_file" .py)"

  echo "[cython] 编译 ${py_file} ..."
  (cd "${mod_dir}" && cythonize -3 -i "$(basename "$py_file")")

  if ! compgen -G "${mod_dir}/${mod_base}"*.so >/dev/null; then
    echo "[cython] 未找到 ${mod_dir}/${mod_base}*.so"
    exit 1
  fi

  rm -f "${mod_dir}/${mod_base}.py" "${mod_dir}/${mod_base}.c"
  echo "[cython] 已生成 .so 并删除明文: ${mod_base}.py"
  ls -la "${mod_dir}/${mod_base}"*.so
}

compile_and_strip "${MCAP_PY}"
compile_and_strip "${RELMAN_PY}"

cd "${BUILD_ROOT}/backend"
python3 <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["BUILD_ROOT"])
backend = root / "backend"
relman = root / "scripts" / "relman"
sys.path.insert(0, str(backend))
sys.path.insert(0, str(relman))

from app.services import mcap_converter  # noqa: F401
import flexible_mcap_to_hdf5  # noqa: F401

if (backend / "app/services/mcap_converter.py").exists():
    raise SystemExit("mcap_converter.py 应已删除")
if (relman / "flexible_mcap_to_hdf5.py").exists():
    raise SystemExit("flexible_mcap_to_hdf5.py 应已删除")
print("[cython] import smoke ok")
PY

echo "[cython] 完成"
