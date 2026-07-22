#!/usr/bin/env bash
# 根据 agent/requirements.txt 解析并刷新 environment.lock.json / environment.lock.txt
# 与 build_agent_bundle.sh 使用相同的 wheel 目标（cp310 / manylinux2014_x86_64）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$ROOT/requirements.txt"
LOCK_JSON="$ROOT/environment.lock.json"
LOCK_TXT="$ROOT/environment.lock.txt"
MANIFEST="$ROOT/../backend/agent_packages/manifest.json"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ -n "${PY_BIN:-}" ] && [ -x "${PY_BIN}" ] && "${PY_BIN}" -m pip --version >/dev/null 2>&1; then
  :
elif command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
else
  echo "需要带 pip 的 python3" >&2
  exit 1
fi

WHEEL_PYVER="${WHEEL_PYVER:-310}"
WHEEL_PLATFORM="${WHEEL_PLATFORM:-manylinux2014_x86_64}"
WHEEL_ABI="${WHEEL_ABI:-cp310}"

"$PY_BIN" -m pip download \
  --only-binary=:all: \
  --python-version "$WHEEL_PYVER" \
  --implementation cp \
  --abi "$WHEEL_ABI" \
  --platform "$WHEEL_PLATFORM" \
  -r "$REQ" \
  -d "$TMP"

AGENT_VER="0.0.0"
AGENT_SHA=""
if [ -f "$MANIFEST" ]; then
  AGENT_VER="$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m.get('latest','0.0.0'))")"
  AGENT_SHA="$(python3 -c "import json; m=json.load(open('$MANIFEST')); pk=[p for p in m.get('packages',[]) if p.get('version')==m.get('latest')]; print(pk[0].get('sha256','') if pk else '')")"
fi

export TMP REQ LOCK_JSON LOCK_TXT AGENT_VER AGENT_SHA WHEEL_ABI WHEEL_PLATFORM
python3 << 'PY'
import json, os, re
from datetime import datetime, timezone

tmp = os.environ["TMP"]
req_path = os.environ["REQ"]
lock_json = os.environ["LOCK_JSON"]
lock_txt = os.environ["LOCK_TXT"]
agent_ver = os.environ["AGENT_VER"]
agent_sha = os.environ["AGENT_SHA"]

def parse_wheel(name: str):
    base = name.replace(".whl", "")
    m = re.match(r"^(.+?)-(\d+(?:\.\d+)*)(?:-(.+))?$", base)
    if not m:
        return None, None
    pkg = m.group(1).replace("_", "-").lower()
    return pkg, m.group(2)

locked = {}
for fn in sorted(os.listdir(tmp)):
    if not fn.endswith(".whl"):
        continue
    p, v = parse_wheel(fn)
    if p:
        locked[p] = v

direct = {}
for line in open(req_path, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if " #" in line:
        line = line.split(" #", 1)[0].strip()
    if ";" in line:
        line = line.split(";", 1)[0].strip()
    if "==" in line:
        name, ver = line.split("==", 1)
        direct[name.strip()] = f"=={ver.strip()}"
    elif ">=" in line:
        name, ver = line.split(">=", 1)
        direct[name.strip()] = f">={ver.strip()}"
    elif "<" in line:
        name, ver = line.split("<", 1)
        direct[name.strip()] = f"<{ver.strip()}"
    else:
        direct[line] = ""

manifest = {
    "schema": "eai-agent-environment-lock/v1",
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "agent_bundle": {
        "version": agent_ver,
        "os": "linux",
        "arch": "x86_64",
        "artifact": f"backend/agent_packages/agent-linux-x86_64-{agent_ver}.tar.gz",
        "manifest_sha256": agent_sha,
    },
    "runtime": {
        "python": "3.10",
        "python_abi": os.environ.get("WHEEL_ABI", "cp310"),
        "wheel_platform": os.environ.get("WHEEL_PLATFORM", "manylinux2014_x86_64"),
        "recommended_os": ["Ubuntu 20.04", "Ubuntu 22.04"],
        "install_path_default": "/opt/eai-agent",
    },
    "ros2_optional": {
        "distribution": "humble",
        "setup_bash": "/opt/ros/humble/setup.bash",
        "notes": "rclpy 与 sensor_msgs 等随 ROS 安装；用于 MJPEG 预览、关节/力矩心跳及 ros2 topic 探测",
        "suggested_apt_packages": [
            "ros-humble-ros-base",
            "ros-humble-sensor-msgs",
            "ros-humble-rosbag2-storage-mcap",
        ],
    },
    "collect_script_optional": {
        "description": "独立采集脚本（如 IDE/ceshi/collect.sh），不经 Agent wheel 安装",
        "ros_setup": "/opt/ros/humble/setup.bash",
        "storage_plugin": "mcap",
        "system_tools": ["ros2", "bc", "flock"],
    },
    "direct_dependencies": direct,
    "locked_python_packages": locked,
}

with open(lock_json, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
    f.write("\n")

lines = [
    "# EAI 采集端 Client 环境锁定清单（Python 3.10 / cp310 / manylinux2014_x86_64）",
    f"# 生成时间: {manifest['generated_at']}",
    f"# 对应离线包: agent-linux-x86_64-{agent_ver}.tar.gz",
    "#",
]
for pkg, ver in sorted(locked.items()):
    lines.append(f"{pkg}=={ver}")
with open(lock_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {lock_json}")
print(f"Wrote {lock_txt} ({len(locked)} packages)")
PY

echo "Done."
