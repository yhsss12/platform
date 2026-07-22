#!/usr/bin/env bash
# 在 Docker builder 内对核心 Python 做 PyArmor 混淆（原地写回 BUILD_ROOT）。
#
# 环境变量：
#   PYARMOR_SKIP_FILES   逗号分隔文件名（basename）
#   PYARMOR_MAX_LINES    超过此行数的 .py 保留明文（Trial 常见限制），默认 2000
#   PYARMOR_OBF_CODE     0|1|2，默认 1；trial 可设 0
#   PYARMOR_ENABLED      默认 true；设 false 跳过
#
# 用法（builder 内）：
#   ./scripts/release/pyarmor_obfuscate.sh /build

set -euo pipefail

BUILD_ROOT="$(cd "${1:?BUILD_ROOT required}" && pwd)"

PYARMOR_ENABLED="${PYARMOR_ENABLED:-true}"
PYARMOR_SKIP_FILES="${PYARMOR_SKIP_FILES:-mcap_converter.py,flexible_mcap_to_hdf5.py,routes_data_assets.py,rq_executor.py}"
PYARMOR_MAX_LINES="${PYARMOR_MAX_LINES:-2000}"
PYARMOR_OBF_CODE="${PYARMOR_OBF_CODE:-1}"

if [[ "${PYARMOR_ENABLED}" != "true" && "${PYARMOR_ENABLED}" != "1" ]]; then
  echo "[pyarmor] PYARMOR_ENABLED=${PYARMOR_ENABLED}，跳过混淆"
  exit 0
fi

IFS=',' read -ra SKIP_LIST <<< "${PYARMOR_SKIP_FILES}"

should_skip_name() {
  local base="$1"
  for s in "${SKIP_LIST[@]}"; do
    s="${s// /}"
    [[ -z "$s" ]] && continue
    [[ "$base" == "$s" ]] && return 0
  done
  return 1
}

too_many_lines() {
  local f="$1"
  local n
  n="$(wc -l < "$f" | tr -d ' ')"
  [[ "${n}" -gt "${PYARMOR_MAX_LINES}" ]]
}

OBF_ROOT="${BUILD_ROOT}/.pyarmor-build"
STAGE="${OBF_ROOT}/stage"
RUNTIME_DIR="${OBF_ROOT}/runtime"
PLAIN="${OBF_ROOT}/plain"

SVC_SRC="${BUILD_ROOT}/backend/app/services"
SVC_STAGE="${STAGE}/app/services"
RELMAN_SRC="${BUILD_ROOT}/scripts/relman"
RELMAN_STAGE="${STAGE}/scripts/relman"
API_SRC="${BUILD_ROOT}/backend/app/api"
CRUD_SRC="${BUILD_ROOT}/backend/app/crud"
CRUD_STAGE="${STAGE}/app/crud"
WORKER_SRC="${BUILD_ROOT}/backend/worker.py"
LABEL_SRC="${BUILD_ROOT}/label_task_description.py"

echo "[pyarmor] BUILD_ROOT=${BUILD_ROOT}"
echo "[pyarmor] SKIP=${PYARMOR_SKIP_FILES}"
echo "[pyarmor] MAX_LINES=${PYARMOR_MAX_LINES}"
echo "[pyarmor] OBF_CODE=${PYARMOR_OBF_CODE}"

rm -rf "${OBF_ROOT}"
mkdir -p "${SVC_STAGE}" "${RELMAN_STAGE}" "${PLAIN}/services" "${PLAIN}/relman"

python3 -m pip install -q "pyarmor>=9.0,<10"

pyarmor gen runtime -O "${RUNTIME_DIR}"

RT_ARG=(--use-runtime "${RUNTIME_DIR}")
GEN_OPTS=(--obf-code "${PYARMOR_OBF_CODE}")

stage_tree() {
  local src_dir="$1"
  local stage_dir="$2"
  local plain_dir="$3"
  local f base
  shopt -s nullglob
  for f in "${src_dir}"/*.py; do
    base="$(basename "$f")"
    if should_skip_name "${base}" || too_many_lines "${f}"; then
      if should_skip_name "${base}"; then
        echo "[pyarmor] 保留明文(SKIP): ${base}"
      else
        echo "[pyarmor] 保留明文(>${PYARMOR_MAX_LINES}行): ${base}"
      fi
      cp -a "$f" "${plain_dir}/${base}"
    else
      cp -a "$f" "${stage_dir}/${base}"
    fi
  done
  shopt -u nullglob
}

obfuscate_flat_dir() {
  local src_dir="$1"
  local stage_dir="$2"
  local plain_dir="$3"
  local out_name="$4"
  local out_subdir="$5"

  mkdir -p "${stage_dir}" "${plain_dir}"
  stage_tree "${src_dir}" "${stage_dir}" "${plain_dir}"

  if compgen -G "${stage_dir}/*.py" >/dev/null; then
    pyarmor gen -O "${OBF_ROOT}/out-${out_name}" -r "${GEN_OPTS[@]}" "${RT_ARG[@]}" "${stage_dir}"
    cp -a "${OBF_ROOT}/out-${out_name}/${out_subdir}/." "${src_dir}/"
  else
    echo "[pyarmor] 无 ${out_name} 待混淆文件"
  fi

  if compgen -G "${plain_dir}/*.py" >/dev/null; then
    cp -a "${plain_dir}/"*.py "${src_dir}/"
  fi
}

# API：逐文件混淆，避免 Trial 在超大 routes_data_assets 上整包失败
obfuscate_api_per_file() {
  local src_dir="$1"
  local f base one_stage one_out

  if [[ ! -d "${src_dir}" ]]; then
    return 0
  fi

  shopt -s nullglob
  for f in "${src_dir}"/*.py; do
    base="$(basename "$f")"
    if should_skip_name "${base}"; then
      echo "[pyarmor] api 保留明文(SKIP): ${base}"
      continue
    fi
    if too_many_lines "${f}"; then
      echo "[pyarmor] api 保留明文(>${PYARMOR_MAX_LINES}行): ${base}"
      continue
    fi

    one_stage="${OBF_ROOT}/stage-api-one"
    one_out="${OBF_ROOT}/out-api-one"
    rm -rf "${one_stage}" "${one_out}"
    mkdir -p "${one_stage}"
    cp -a "${f}" "${one_stage}/${base}"

    echo "[pyarmor] api 单文件混淆: ${base}"
    if pyarmor gen -O "${one_out}" "${GEN_OPTS[@]}" "${RT_ARG[@]}" "${one_stage}/${base}"; then
      cp -a "${one_out}/${base}" "${src_dir}/${base}"
    else
      echo "[pyarmor] WARN api 混淆失败，保留明文: ${base}"
    fi
  done
  shopt -u nullglob
  echo "[pyarmor] 已处理 api 目录: ${src_dir}"
}

obfuscate_flat_dir "${SVC_SRC}" "${SVC_STAGE}" "${PLAIN}/services" "services" "services"
obfuscate_flat_dir "${RELMAN_SRC}" "${RELMAN_STAGE}" "${PLAIN}/relman" "relman" "relman"
obfuscate_api_per_file "${API_SRC}"
obfuscate_flat_dir "${CRUD_SRC}" "${CRUD_STAGE}" "${PLAIN}/crud" "crud" "crud"

if should_skip_name "worker.py"; then
  echo "[pyarmor] 保留明文: worker.py"
else
  pyarmor gen -O "${OBF_ROOT}/out-worker" "${GEN_OPTS[@]}" "${RT_ARG[@]}" "${WORKER_SRC}"
  cp -a "${OBF_ROOT}/out-worker/worker.py" "${WORKER_SRC}"
fi

if [[ -f "${LABEL_SRC}" ]]; then
  if should_skip_name "label_task_description.py" || too_many_lines "${LABEL_SRC}"; then
    echo "[pyarmor] 保留明文: label_task_description.py"
  else
    if pyarmor gen -O "${OBF_ROOT}/out-label" "${GEN_OPTS[@]}" "${RT_ARG[@]}" "${LABEL_SRC}"; then
      cp -a "${OBF_ROOT}/out-label/label_task_description.py" "${LABEL_SRC}"
      echo "[pyarmor] 已混淆: label_task_description.py"
    else
      echo "[pyarmor] WARN label_task_description 混淆失败，保留明文"
    fi
  fi
else
  echo "[pyarmor] 未找到 label_task_description.py（自动标注将不可用）"
fi

RUNTIME_PKG="${RUNTIME_DIR}/pyarmor_runtime_000000"
if [[ -d "${RUNTIME_PKG}" ]]; then
  rm -rf "${BUILD_ROOT}/backend/pyarmor_runtime_000000"
  cp -a "${RUNTIME_PKG}" "${BUILD_ROOT}/backend/pyarmor_runtime_000000"
  echo "[pyarmor] runtime -> backend/pyarmor_runtime_000000"
fi

cd "${BUILD_ROOT}/backend"
if python3 -c "import h5py" 2>/dev/null; then
  python3 -c "
import sys
from pathlib import Path
build = Path('${BUILD_ROOT}')
sys.path.insert(0, str(build))
sys.path.insert(0, str(build / 'backend'))
from label_task_description import find_image_group
from app.services import annotation_service
from app.api import router as api_router
import worker
print('[pyarmor] import smoke ok')
"
else
  echo "[pyarmor] 跳过 import 冒烟（当前环境未安装 backend 依赖，Docker build 内会执行）"
fi

rm -rf "${OBF_ROOT}"
echo "[pyarmor] 完成"
