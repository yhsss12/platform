#!/usr/bin/env bash
# 在「与目标采集机相同 Ubuntu/Debian、相同 python3 大版本小版本」上执行，下载
# pythonX.Y-venv 的 .deb 到当前目录，便于随 Agent 包一同分发。
# 需要 root: sudo ./fetch-ubuntu-venv-debs.sh [3.10]
set -euo pipefail

if [ "${EUID:-0}" -ne 0 ]; then
  echo "需要 root: sudo $0" >&2
  exit 1
fi

PYV="${1:-$(python3 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')}"
PKG="python${PYV}-venv"
HERE="$(cd "$(dirname "$0")" && pwd)"
PART="$HERE/partial_apt_$$"
mkdir -p "$PART" "$HERE"

apt-get update -qq
# 将 apt 的「仅下载」缓存指到 debs/，便于全部依赖落盘
# shellcheck disable=SC2016
if ! apt-get install -y -qq \
  -o "Dir::Cache::archives=$HERE" \
  -o "Dir::Cache::archives::Partial=$PART" \
  -o "Dir::Cache=$HERE/.apttmp" \
  --download-only \
  "$PKG" \
  ; then
  echo "注意: 完全下载 $PKG 失败，尝试只下载主包: apt-get download" >&2
  (cd "$HERE" && apt-get download "$PKG" 2>/dev/null) || true
fi
rm -rf "$PART" "$HERE/.apttmp" 2>/dev/null || true

if ! compgen -G "$HERE"/*.deb >/dev/null; then
  echo "未生成 .deb。请在 Ubuntu 上手动: apt-get -y -o Dir::Cache::archives=\"$HERE\" install --download-only $PKG" >&2
  echo "或见 agent/debs/README.md" >&2
  exit 1
fi

ls -la "$HERE"/*.deb
echo ">>> 下一步: cd backend/agent_packages && ./build_agent_bundle.sh <ver> linux x86_64"
