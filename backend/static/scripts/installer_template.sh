#!/usr/bin/env bash
set -euo pipefail

# 环境变量占位符 (由后端在生成时替换)
TOKEN="{{TOKEN}}"
SERVER_IP="{{SERVER_IP}}"
SERVER_PORT="{{SERVER_PORT}}"
WS_URL="ws://${SERVER_IP}:${SERVER_PORT}/api/agent/ws"
HTTP_URL="http://${SERVER_IP}:${SERVER_PORT}"

# 配置项
AGENT_DIR="/opt/eai-agent"
# 平台注入的 systemd User=：__SUDO_USER__ 表示与「sudo 安装者」同账户（默认，不 root 跑服务）
_RAW_SERVICE_USER="{{SERVICE_USER}}"
if [ "${_RAW_SERVICE_USER}" = "__SUDO_USER__" ]; then
  if [ -n "${SUDO_USER:-}" ]; then
    SERVICE_USER="${SUDO_USER}"
  else
    SERVICE_USER="root"
  fi
else
  SERVICE_USER="${_RAW_SERVICE_USER}"
fi
AGENT_USE_CONDA="{{AGENT_USE_CONDA}}"
CONFIG_FILE="${AGENT_DIR}/config.py"
SERVICE_FILE="/etc/systemd/system/eai-client.service"
DOWNLOAD_URL="${HTTP_URL}/static/bin/agent_linux_x64.tar.gz"
# 数据根：与运行服务用户 home 一致（EAI_AGENT_DATA_ROOT）
RUN_USER="${SERVICE_USER}"
DATA_ROOT="/home/${RUN_USER}"
if [ ! -d "$DATA_ROOT" ]; then
  if [ "$SERVICE_USER" = "root" ] && [ -d "/root" ]; then
    DATA_ROOT="/root"
  elif [ -n "${SUDO_USER:-}" ] && [ -d "/home/${SUDO_USER}" ]; then
    DATA_ROOT="/home/${SUDO_USER}"
  else
    DATA_ROOT="/home/ubuntu"
  fi
fi

echo ">>> 启动 EAI Client 自动化安装"
echo ">>> 服务运行用户: ${SERVICE_USER}（EAI_AGENT_DATA_ROOT=${DATA_ROOT}）"

progress_report() {
  local stage="$1"
  local prog="$2"
  local status="${3:-running}"
  local level="${4:-info}"
  local msg="${5:-}"
  local enc
  enc="$("${EAI_PY310:-/usr/bin/python3.10}" -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$msg" 2>/dev/null || echo "")"
  curl -fsSL --retry 2 --retry-delay 1 \
    "${HTTP_URL}/api/agent/installer/progress?token=${TOKEN}&stage=${stage}&progress=${prog}&status=${status}&level=${level}&message=${enc}" \
    >/dev/null 2>&1 || true
}

# 1. 权限检查
echo ">>> [1/4] 环境检测..."
if [ "$EUID" -ne 0 ]; then
  echo "❌ 错误: 请使用 root 权限或 sudo 运行此脚本。" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "❌ 错误: 未找到 curl 命令，请先安装 curl。" >&2
  exit 1
fi
if ! command -v tar >/dev/null 2>&1; then
  echo "❌ 错误: 未找到 tar 命令，请先安装 tar。" >&2
  exit 1
fi
# 与 offline wheelhouse（cp310）一致，固定使用 Python 3.10
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "❌ 需要 Python 3.10（与平台 Agent 离线包、ROS2 Humble 常用环境一致）。" >&2
  echo "   Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv python3-pip" >&2
  exit 1
fi
EAI_PY310="$(command -v python3.10)"
export EAI_PY310
if [ ! -d "$DATA_ROOT" ]; then
  DATA_ROOT="/home/ubuntu"
fi

progress_report "detect" 5 "running" "info" "环境检测通过"

# 错误清理钩子
cleanup() {
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "❌ 安装失败！正在清理临时文件..."
    progress_report "failed" 100 "failed" "error" "安装失败（请查看终端输出）"
    rm -rf /tmp/eai-agent-download.*
  fi
  exit $exit_code
}
trap cleanup EXIT

# 2. 下载并解压 Agent 二进制包
echo ">>> [2/4] 下载 Client 程序包..."
progress_report "download" 10 "running" "info" "开始下载"
TMP_DIR=$(mktemp -d /tmp/eai-agent-download.XXXXXX)
cd "$TMP_DIR"

echo "正在从 ${DOWNLOAD_URL} 下载..."
curl -fsSL --retry 3 --retry-delay 2 -o agent.tar.gz "$DOWNLOAD_URL"

echo "正在解压到 ${AGENT_DIR}..."
mkdir -p "$AGENT_DIR"
tar -xzf agent.tar.gz -C "$AGENT_DIR"
# build_agent_bundle.sh 打出的 tar 顶层目录通常为「agent/」。
# 为避免出现 /opt/eai-agent 与 /opt/eai-agent/agent 两套文件并存导致混乱，这里只要检测到 agent/，
# 就无条件将其内容整体上移到 ${AGENT_DIR} 根目录，并删除 agent/ 目录。
# （含 config.py：重复安装/升级时允许包内版本覆盖旧文件；后续仍会由本脚本用平台参数重写 config.py。）
if [ -d "${AGENT_DIR}/agent" ]; then
  echo ">>> 扁平化安装目录：将包内 agent/ 上移到 ${AGENT_DIR} 根目录..."
  shopt -s dotglob
  for _f in "${AGENT_DIR}/agent"/*; do
    [ -e "$_f" ] || continue
    _base="$(basename "$_f")"
    # 兼容“重复安装/升级”：目标已存在目录时，合并内容而非直接 mv（否则会报 Directory not empty）
    if [ -d "$_f" ] && [ -d "${AGENT_DIR}/${_base}" ]; then
      if command -v rsync >/dev/null 2>&1; then
        rsync -a "$_f"/ "${AGENT_DIR}/${_base}/"
        rm -rf "$_f" 2>/dev/null || true
      else
        # 无 rsync 时用 cp 兜底（尽量保留权限/时间戳）
        cp -a "$_f"/. "${AGENT_DIR}/${_base}/" 2>/dev/null || cp -r "$_f"/. "${AGENT_DIR}/${_base}/"
        rm -rf "$_f" 2>/dev/null || true
      fi
      continue
    fi
    mv -f "$_f" "$AGENT_DIR"/
  done
  shopt -u dotglob
  rmdir "${AGENT_DIR}/agent" 2>/dev/null || true
fi
progress_report "extract" 35 "running" "info" "下载并解压完成"

# 写入启动包装脚本（与交互终端 conda+ROS+python 顺序一致；systemd 仅 ExecStart 该脚本）
printf '%s' '{{RUN_AGENT_SH_B64}}' | base64 -d > "${AGENT_DIR}/run-agent.sh"
chmod 0755 "${AGENT_DIR}/run-agent.sh" || true

if [ "${AGENT_USE_CONDA}" = "1" ] && [ -f "${AGENT_DIR}/requirements.txt" ]; then
  echo ">>> Conda 模式：在指定环境中安装/更新 Python 依赖..."
  progress_report "venv" 45 "running" "info" "Conda 环境 pip 安装依赖"
  CONDA_SH_PATH="$(printf '%s' '{{CONDA_SH_B64}}' | base64 -d)"
  CONDA_ENV_NAME="$(printf '%s' '{{CONDA_ENV_B64}}' | base64 -d)"
  set +u
  if [ -d "${AGENT_DIR}/wheelhouse" ] && [ -n "$(ls -A "${AGENT_DIR}/wheelhouse" 2>/dev/null)" ]; then
    if ! bash -lc "source \"${CONDA_SH_PATH}\" && conda activate \"${CONDA_ENV_NAME}\" && pip install -U pip && pip install --no-index --find-links \"${AGENT_DIR}/wheelhouse\" -r \"${AGENT_DIR}/requirements.txt\""; then
      echo "❌ Conda 环境下离线 pip install 失败，请检查 wheelhouse 与包版本。" >&2
      exit 1
    fi
  else
    if ! bash -lc "source \"${CONDA_SH_PATH}\" && conda activate \"${CONDA_ENV_NAME}\" && pip install -r \"${AGENT_DIR}/requirements.txt\""; then
      echo "❌ Conda 环境下 pip install 失败，请检查 conda.sh 路径、环境名与网络。" >&2
      exit 1
    fi
  fi
  set -u
  progress_report "venv" 60 "running" "info" "Conda 依赖安装完成"
elif [ "${AGENT_USE_CONDA}" != "1" ]; then
  # 与 run-agent.sh 中 venv/bin/python 一致：无论是否有 requirements 都创建 venv，避免服务秒退
  # ensurepip 需系统包 python3.X-venv；可选将 .deb 放入包内 debs/ 以离线安装（见 agent/debs/README.md）
  eai_ensure_venv_apt() {
    eai_venv_creates() {
      local t="$1"
      if "${EAI_PY310}" -m venv "$t"; then
        rm -rf "$t" 2>/dev/null || true
        return 0
      fi
      return 1
    }
    local t_probe
    t_probe="$(mktemp -d)" || {
      echo "❌ 无法创建临时目录（mktemp）" >&2
      return 1
    }
    if eai_venv_creates "$t_probe"; then
      return 0
    fi
    rm -rf "$t_probe" 2>/dev/null || true
    echo ">>> 上为「试建 venv」失败（常因未装 ensurepip 对应的 python3.10-venv 包，切勿只信 -h 检测）" >&2
    if ! command -v apt-get > /dev/null 2>&1; then
      echo "❌ 本机无 apt-get。Debian/Ubuntu 上请先: sudo apt install -y python3.10-venv" >&2
      return 1
    fi
    local pkg="python3.10-venv"
    echo ">>> 将尝试: apt 安装 ${pkg}、python3-pip 或 /opt/eai-agent/debs/*.deb" >&2
    apt-get update -qq 2>&1 | tail -n 1 || true
    shopt -s nullglob
    local dbs=( "${AGENT_DIR}"/debs/*.deb )
    shopt -u nullglob
    if ((${#dbs[@]} > 0)); then
      echo ">>> 使用安装包内 debs/ 安装（共 ${#dbs[@]} 个 deb）..." >&2
      apt-get install -y -qq --no-install-recommends "${dbs[@]}" || {
        dpkg -i "${dbs[@]}" || true
        apt-get install -y -f -qq || true
      }
    fi
    t_probe="$(mktemp -d)" || {
      return 1
    }
    if eai_venv_creates "$t_probe"; then
      return 0
    fi
    rm -rf "$t_probe" 2>/dev/null || true
    echo ">>> 正在: apt-get install -y --no-install-recommends ${pkg} python3-pip" >&2
    if apt-get install -y -qq --no-install-recommends "$pkg" python3-pip; then
      true
    elif apt-get install -y -qq --no-install-recommends python3-venv python3-pip; then
      true
    else
      echo "❌ apt 未成功安装 venv(ensurepip)。" >&2
      echo "   有网: sudo apt-get update && sudo apt-get install -y ${pkg} python3-pip" >&2
      echo "   离线: 在同类 Ubuntu 上见 agent/debs/ 准备 .deb 后重新打 tar" >&2
      return 1
    fi
    t_probe="$(mktemp -d)" || {
      return 1
    }
    if eai_venv_creates "$t_probe"; then
      return 0
    fi
    rm -rf "$t_probe" 2>/dev/null || true
    echo "❌ 安装包后试建 venv 仍失败，请本机检查: ${EAI_PY310} -m venv /tmp/venv-test" >&2
    return 1
  }
  echo ">>> 准备 Python 虚拟环境 (venv)..."
  progress_report "venv" 45 "running" "info" "创建虚拟环境"
  if ! eai_ensure_venv_apt; then
    exit 1
  fi
  rm -rf "${AGENT_DIR}/venv"
  if ! "${EAI_PY310}" -m venv "${AGENT_DIR}/venv"; then
    echo "❌ 在 ${AGENT_DIR}/venv 创建 venv 失败（上见 Python 错误）。已确认 eai_ensure_venv_apt 通过时仍失败请反馈。" >&2
    exit 1
  fi
  "${AGENT_DIR}/venv/bin/pip" install -U pip >/dev/null 2>&1 || true
  if [ -f "${AGENT_DIR}/requirements.txt" ]; then
    if [ -d "${AGENT_DIR}/wheelhouse" ] && [ -n "$(ls -A "${AGENT_DIR}/wheelhouse" 2>/dev/null)" ]; then
      echo ">>> 从离线 wheelhouse 安装 Python 依赖（推荐）..."
      # 先单独安装 exceptiongroup：在 --no-index 下，若与 fastapi/anyio 等一同解析，pip 可能因 anyio
      # 元数据里「python_version < 3.11」对 exceptiongroup 的约束而报 versions: none（见 issue：先装可消）
      PIPV="${AGENT_DIR}/venv/bin/pip"
      WHL="${AGENT_DIR}/wheelhouse"
      if compgen -G "${WHL}"/exceptiongroup-*.whl >/dev/null; then
        if ! "${PIPV}" install --no-index --find-links "${WHL}" "exceptiongroup>=1.0.2"; then
          echo "❌ 从 wheelhouse 预装 exceptiongroup 失败。请确认 venv 为 Python 3.10。" >&2
          exit 1
        fi
      else
        echo ">>> 当前离线包内无 exceptiongroup-*.whl（多为平台仍提供 0.1.5 及更早包）。正在尝试经 PyPI 安装 exceptiongroup（需本机可访问 pypi.org）…" >&2
        if ! "${PIPV}" install "exceptiongroup>=1.0.2"; then
          echo "❌ 无法从 PyPI 安装 exceptiongroup（可能无网）。请二选一：" >&2
          echo "   1) 在部署机用当前仓库打 **0.1.6+**（或含 async-timeout 的 **0.1.7+**）包，将 manifest 中 agent-linux 的 sha256 更新到平台并重启后端，使 /static/bin/agent_linux_x64.tar.gz 指向新包；" >&2
          echo "   2) 本机联网后重跑本脚本，或先执行: sudo ${PIPV} install 'exceptiongroup>=1.0.2'" >&2
          exit 1
        fi
      fi
      # aiohttp 在 Py3.10 上需 async-timeout；wheel 文件名为 async_timeout-*.whl
      if compgen -G "${WHL}"/async_timeout-*.whl >/dev/null; then
        if ! "${PIPV}" install --no-index --find-links "${WHL}" "async-timeout>=4.0,<6.0"; then
          echo "❌ 从 wheelhouse 预装 async-timeout 失败。请确认 venv 为 Python 3.10。" >&2
          exit 1
        fi
      else
        echo ">>> 当前离线包内无 async_timeout-*.whl（包可能早于 0.1.7）。正在尝试经 PyPI 安装 async-timeout…" >&2
        if ! "${PIPV}" install "async-timeout>=4.0,<6.0"; then
          echo "❌ 无法从 PyPI 安装 async-timeout（可能无网）。请换用 **0.1.7+** 全量包或本机联网后重试。" >&2
          exit 1
        fi
      fi
      if ! "${PIPV}" install --no-index --find-links "${WHL}" -r "${AGENT_DIR}/requirements.txt"; then
        echo "❌ 离线安装 -r 失败。若包过旧，请按上一步换 0.1.7+ 全量包；或本机联网后重试。亦可: ${PIPV} install -U pip" >&2
        exit 1
      fi
    else
      if ! "${AGENT_DIR}/venv/bin/pip" install -r "${AGENT_DIR}/requirements.txt"; then
        echo "❌ pip install 失败，请检查网络，或在联网环境下由平台重新打包带 wheelhouse 的 Agent 包。" >&2
        exit 1
      fi
    fi
  else
    echo "❌ 安装包中缺少 requirements.txt，无法安装 Agent 依赖，通常表示下载的 tar 不完整或 URL 指错了文件。" >&2
    echo "   请确认: (1) 平台 /static/bin/agent_linux_x64.tar.gz 能拿到含 agent/requirements.txt 的离线包" >&2
    echo "   (2) 在 backend 执行 build_agent_bundle.sh 重新生成包并更新 manifest" >&2
    exit 1
  fi
  progress_report "venv" 60 "running" "info" "依赖安装完成"
fi

# 3. 生成配置文件
echo ">>> [3/4] 生成配置文件..."

# 3a. 释放默认端口：重装/升级时旧 eai-agent 常仍占用 9100，先停服务再选端口
echo ">>> 如本机已安装 eai-client/eai-agent 服务，将先停止以释放可能的端口占用（安全重装/升级）…"
systemctl stop eai-client.service 2>/dev/null || true
systemctl stop eai-agent.service 2>/dev/null || true
sleep 1

# 检测本机某 TCP 端口是否已在监听（0.0.0.0 或 ANY）
eai_is_port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -qE ":${port}[[:space:]]"
  elif command -v netstat >/dev/null 2>&1; then
    netstat -tln 2>/dev/null | grep -qE ":${port}[[:space:]]"
  else
    if "${EAI_PY310}" -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', int('${port}'))); s.close()" 2>/dev/null; then
      return 1
    else
      return 0
    fi
  fi
}

# 在 9100-9200 上选第一个空闲端口（若 9100 仍被非 eai 进程占用则顺延）
EAI_PORT_MIN=9100
EAI_PORT_MAX=9200
AGENT_CHOSEN_PORT=
for (( _p = EAI_PORT_MIN; _p <= EAI_PORT_MAX; _p++ )); do
  if ! eai_is_port_listening "$_p"; then
    AGENT_CHOSEN_PORT="$_p"
    break
  fi
done
if [ -z "${AGENT_CHOSEN_PORT}" ]; then
  echo "❌ 在 ${EAI_PORT_MIN}-${EAI_PORT_MAX} 范围内未找到可用 TCP 端口，eai-client 无法绑定。请结束占用后重试。" >&2
  exit 1
fi
if [ "${AGENT_CHOSEN_PORT}" != "9100" ]; then
  echo ">>> 本机 9100 等端口被占用，已自动选用 :${AGENT_CHOSEN_PORT}。"
  echo "    在平台「添加/编辑设备」时，请将**采集端端口**填为: ${AGENT_CHOSEN_PORT}"
fi

progress_report "config" 70 "running" "info" "写入配置(端口${AGENT_CHOSEN_PORT})"
eai_pick_primary_mac() {
  # 1) 优先用默认路由对应网卡
  local dev=""
  dev="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}' || true)"
  if [ -n "${dev}" ] && [ -e "/sys/class/net/${dev}/address" ]; then
    cat "/sys/class/net/${dev}/address" 2>/dev/null | tr 'A-Z' 'a-z' || true
    return 0
  fi

  # 2) 兜底：遍历常见物理网卡/无线网卡，跳过虚拟接口
  local iface=""
  for iface in /sys/class/net/*; do
    iface="$(basename "$iface")"
    case "$iface" in
      lo|docker*|br-*|veth*|virbr*|vmnet*|tun*|tap*|wg*|zt*|tailscale*|sit*|ip6tnl*|gre*|gretap*|erspan*|ifb*|dummy*|bond*|team*|macvlan*|ipvlan*)
        continue
        ;;
    esac
    if [ -e "/sys/class/net/${iface}/address" ]; then
      local mac=""
      mac="$(cat "/sys/class/net/${iface}/address" 2>/dev/null | tr 'A-Z' 'a-z' || true)"
      if [ -n "${mac}" ] && [ "${mac}" != "00:00:00:00:00:00" ]; then
        echo "${mac}"
        return 0
      fi
    fi
  done
  return 1
}

# Client ID（设备唯一标识，用于平台绑定）：
# - 优先取环境变量 EAI_CLIENT_ID（便于手动指定）
# - 否则自动取本机主网卡 MAC（推荐）
EAI_CLIENT_ID="${EAI_CLIENT_ID:-}"
if [ -z "${EAI_CLIENT_ID}" ]; then
  EAI_CLIENT_ID="$(eai_pick_primary_mac || true)"
fi
if [ -z "${EAI_CLIENT_ID}" ]; then
  # 极端兜底：使用安装 token，避免空值导致运行时异常（同时会在安装输出中提醒）
  EAI_CLIENT_ID="${TOKEN}"
  echo "⚠️ 未能自动获取 MAC 地址，已临时使用安装 token 作为 Client ID（建议设置 EAI_CLIENT_ID 或检查网络接口）。" >&2
fi

cat > "$CONFIG_FILE" <<EOF
SERVER_BASE = "${HTTP_URL}"
AGENT_ID = "${EAI_CLIENT_ID}"
AGENT_NAME = "eai-client"
DEVICES = []
AGENT_HOST = "0.0.0.0"
AGENT_PORT = ${AGENT_CHOSEN_PORT}
AGENT_TUNNEL_TOKEN = "{{AGENT_TUNNEL_TOKEN}}"
EOF

# 3b. 依赖与端口自检（尽早暴露导入错误，避免服务秒退成 status=1）
echo ">>> 校验关键依赖与即将监听的端口 ${AGENT_CHOSEN_PORT}..."
if [ "${AGENT_USE_CONDA}" = "1" ]; then
  CONDA_SH_PATH="$(printf '%s' '{{CONDA_SH_B64}}' | base64 -d)"
  CONDA_ENV_NAME="$(printf '%s' '{{CONDA_ENV_B64}}' | base64 -d)"
  set +u
  if ! (cd "$AGENT_DIR" && EAI_AGENT_DATA_ROOT="$DATA_ROOT" \
      bash -lc "source \"${CONDA_SH_PATH}\" && conda activate \"${CONDA_ENV_NAME}\" && python -c \"import fastapi,uvicorn,cv2,httpx\""); then
    echo "❌ 依赖自检失败：在 Conda 环境中无法 import 关键包。" >&2
    exit 1
  fi
  set -u
else
  if ! (cd "$AGENT_DIR" && EAI_AGENT_DATA_ROOT="$DATA_ROOT" \
      "${AGENT_DIR}/venv/bin/python" -c "import fastapi,uvicorn,cv2,httpx"); then
    echo "❌ 依赖自检失败：venv 中 import fastapi/… 不通过。常见原因: pip 未装齐（网络/wheelhouse）、或上一步已提示缺 requirements.txt。" >&2
    echo "   仅当 OpenCV 报缺 .so 时再试: sudo apt-get install -y libgl1 libglib2.0-0" >&2
    exit 1
  fi
fi

# 3c. 非 root 服务用户：chown 必须成功（原 || true 会藏掉失败，导致 systemd 以该用户起不来）
#     并在此后以目标用户复测，与 systemctl 实际执行身份一致（仅 root 自检会漏权限问题）
if [ "${SERVICE_USER}" != "root" ]; then
  if ! getent passwd "${SERVICE_USER}" >/dev/null 2>&1; then
    echo "❌ 系统无用户: ${SERVICE_USER}（/etc/passwd 中不存在）。请创建该用户或设 AGENT_INSTALL_SERVICE_USER=root 重装。" >&2
    exit 1
  fi
  if ! chown -R "${SERVICE_USER}:${SERVICE_USER}" "${AGENT_DIR}"; then
  echo "❌ chown -R ${SERVICE_USER} ${AGENT_DIR} 失败。服务无法以 ${SERVICE_USER} 访问 /opt/eai-agent。" >&2
    exit 1
  fi
  echo ">>> 以系统用户 ${SERVICE_USER} 复测（与 systemd User= 一致）..."
  _EAI_HOME="$(getent passwd "${SERVICE_USER}" | cut -d: -f6)"
  eai_as_service_user() {
    if command -v runuser >/dev/null 2>&1; then
      runuser -u "$SERVICE_USER" -- env "HOME=${_EAI_HOME}" "EAI_AGENT_DATA_ROOT=${DATA_ROOT}" "$@"
    else
      sudo -u "$SERVICE_USER" -H env "HOME=${_EAI_HOME}" "EAI_AGENT_DATA_ROOT=${DATA_ROOT}" "$@"
    fi
  }
  if [ "${AGENT_USE_CONDA}" = "1" ]; then
    CONDA_SH_PATH="$(printf '%s' '{{CONDA_SH_B64}}' | base64 -d)"
    CONDA_ENV_NAME="$(printf '%s' '{{CONDA_ENV_B64}}' | base64 -d)"
    if ! eai_as_service_user bash -lc "cd \"${AGENT_DIR}\" && source \"${CONDA_SH_PATH}\" && conda activate \"${CONDA_ENV_NAME}\" && python -c \"import fastapi,uvicorn,cv2,httpx\""; then
      echo "❌ 以用户 ${SERVICE_USER} 执行 Conda+import 失败（chown/权限/conda 环境）。见上错误。" >&2
      exit 1
    fi
  else
    if ! eai_as_service_user /bin/sh -c "cd \"${AGENT_DIR}\" && exec \"${AGENT_DIR}/venv/bin/python\" -c \"import fastapi,uvicorn,cv2,httpx\""; then
      echo "❌ 以用户 ${SERVICE_USER} 在 ${AGENT_DIR} 下无法执行 venv 或 import（chown/权限/SELinux？）。可试: ls -la ${AGENT_DIR}/venv" >&2
      exit 1
    fi
  fi
fi
# 再次确认选用端口未在监听（与系统ctl stop 后的时间窗竞态很少见）
if eai_is_port_listening "${AGENT_CHOSEN_PORT}"; then
  echo "❌ 端口 ${AGENT_CHOSEN_PORT} 在配置写入后仍被占用，eai-agent 无法绑定。请检查是否有其他进程抢占后重试。" >&2
  exit 1
fi

# 4. 创建并启动 Systemd 服务
echo ">>> [4/4] 配置系统服务..."
progress_report "service" 85 "running" "info" "配置 systemd 服务"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=EAI Client Service
After=network.target

[Service]
Type=simple
#
# 实际启动逻辑在 ${AGENT_DIR}/run-agent.sh（venv 或 Conda+ROS 由平台生成该脚本）。
# 仍设置 PATH，便于子进程或诊断命令找到 ros2。
Environment="PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment=PYTHONUNBUFFERED=1
ExecStart=${AGENT_DIR}/run-agent.sh
WorkingDirectory=${AGENT_DIR}
Restart=no
RestartSec=5
# 避免反复崩溃刷满 journal、掩盖首条关键错误；修好后: systemctl reset-failed eai-client && systemctl start eai-client
StartLimitIntervalSec=2min
StartLimitBurst=8
User=${SERVICE_USER}
Environment=EAI_AGENT_DATA_ROOT=${DATA_ROOT}
# 使用 journal 便于通过 journalctl -u eai-client 查看应用日志（含 Python 堆栈）
StandardOutput=journal
StandardError=journal
SyslogIdentifier=eai-client

[Install]
WantedBy=multi-user.target
EOF

# 重新加载 systemd 并启动服务
systemctl daemon-reload
systemctl enable eai-client.service
systemctl restart eai-client.service

echo ">>> 等待服务启动并检查状态..."
progress_report "verify" 95 "running" "info" "等待服务启动"
sleep 2
if systemctl is-active --quiet eai-client.service; then
  echo "✅ 安装成功！Client 服务已启动并配置为开机自启。"
  echo "   本机 HTTP 监听: 0.0.0.0:${AGENT_CHOSEN_PORT}（请与平台「设备」中填写的端口一致）"
  if [ -f "${CONFIG_FILE}" ]; then
    _CID="$(awk -F '\"' '/^AGENT_ID[[:space:]]*=/{print $2; exit}' "${CONFIG_FILE}" 2>/dev/null || true)"
    if [ -n "${_CID:-}" ]; then
      echo "   Client ID（设备唯一标识，用于平台绑定）: ${_CID}"
    fi
  fi
  echo "你可以使用以下命令查看日志: journalctl -u eai-client -f"
  progress_report "done" 100 "success" "info" "安装成功"
else
  echo "⚠️ 服务已安装但启动状态异常，请使用 'systemctl status eai-client' 检查原因。"
  progress_report "failed" 100 "failed" "error" "服务启动失败"
  exit 1
fi
