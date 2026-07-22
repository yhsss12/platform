#!/usr/bin/env bash
# 校验 eai-ide 交付镜像：无 .env、无前端 src、无 tests；核心模块已保护。
set -euo pipefail

IMAGE="${1:-eai-ide:local}"
FAIL=0

say_ok() { echo "[OK] $*"; }
say_fail() { echo "[FAIL] $*"; FAIL=1; }

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "镜像不存在: $IMAGE"
  exit 1
fi

check_absent() {
  local path="$1"
  local label="$2"
  if docker run --rm "$IMAGE" test -e "$path" 2>/dev/null; then
    say_fail "$label 仍存在: $path"
  else
    say_ok "不存在 $path ($label)"
  fi
}

check_present() {
  local path="$1"
  local label="$2"
  if docker run --rm "$IMAGE" test -e "$path" 2>/dev/null; then
    say_ok "存在 $path ($label)"
  else
    say_fail "缺少 $path ($label)"
  fi
}

check_absent /app/.env "根目录 .env"
check_absent /app/src "前端源码 src"
check_absent /app/backend/tests "后端测试目录"
check_present /app/.next "Next 生产构建 .next"
check_present /app/backend/app/main.py "FastAPI 入口"
check_present /app/label_task_description.py "自动标注 label_task_description"

# 核心大模块：Cython 交付应为 .so，无 .py 明文
check_absent /app/backend/app/services/mcap_converter.py "mcap_converter 明文"
check_absent /app/scripts/relman/flexible_mcap_to_hdf5.py "flexible_mcap_to_hdf5 明文"
if docker run --rm "$IMAGE" sh -c 'ls /app/backend/app/services/mcap_converter*.so >/dev/null 2>&1'; then
  say_ok "mcap_converter Cython .so 已安装"
else
  say_fail "缺少 mcap_converter*.so（CYTHON_ENABLED=false 或未执行 cython 步骤）"
fi
if docker run --rm "$IMAGE" sh -c 'ls /app/scripts/relman/flexible_mcap_to_hdf5*.so >/dev/null 2>&1'; then
  say_ok "flexible_mcap_to_hdf5 Cython .so 已安装"
else
  say_fail "缺少 flexible_mcap_to_hdf5*.so"
fi

if docker run --rm "$IMAGE" test -d /app/backend/pyarmor_runtime_000000 2>/dev/null; then
  say_ok "PyArmor runtime 已安装"
  if docker run --rm "$IMAGE" head -n 3 /app/backend/app/services/annotation_service.py 2>/dev/null | grep -q pyarmor_runtime; then
    say_ok "services 已为 PyArmor 混淆格式"
  else
    say_fail "annotation_service 未呈现 PyArmor 头部（可能 PYARMOR_ENABLED=false）"
  fi
  if docker run --rm "$IMAGE" head -n 3 /app/backend/app/api/routes_auth.py 2>/dev/null | grep -q pyarmor_runtime; then
    say_ok "api 已为 PyArmor 混淆格式"
  else
    say_fail "routes_auth 未呈现 PyArmor 头部"
  fi
  if docker run --rm "$IMAGE" head -n 3 /app/label_task_description.py 2>/dev/null | grep -q pyarmor_runtime; then
    say_ok "label_task_description 已为 PyArmor 混淆格式"
  else
    say_fail "label_task_description 未呈现 PyArmor 头部"
  fi
else
  echo "[WARN] 未检测到 pyarmor_runtime（构建时可能 PYARMOR_ENABLED=false）"
fi

# 容器内导入冒烟（DB/Redis 不在此步校验，仅 Python 模块链）
if docker run --rm "$IMAGE" python3 -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/scripts/relman')
from label_task_description import find_image_group
from app.services import mcap_converter, annotation_service
import flexible_mcap_to_hdf5
from app.api import router
import worker
print('import-smoke-ok')
" 2>/dev/null | grep -q import-smoke-ok; then
  say_ok "Python 模块导入冒烟通过"
else
  say_fail "Python 模块导入冒烟失败（请查看 docker run 输出）"
fi

if docker run --rm "$IMAGE" grep -r "jinlian1234" /app/backend/app/crud/user.py >/dev/null 2>&1; then
  echo "[WARN] crud/user.py 仍为明文（交付后请强制改默认超管密码）"
fi

if [ "$FAIL" -ne 0 ]; then
  echo "验证未通过"
  exit 1
fi
echo "验证通过: $IMAGE"
