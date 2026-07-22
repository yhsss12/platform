from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class DebBuildResult:
    file_path: str
    sha256: str
    version: str
    arch: str


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _deb_arch_from_agent_arch(arch: str) -> str:
    a = (arch or "").strip().lower()
    if a in {"x86_64", "amd64"}:
        return "amd64"
    if a in {"arm64", "aarch64"}:
        return "arm64"
    return a or "all"


def _ar_write_member(buf: io.BufferedWriter, *, name: str, data: bytes, mode: int = 0o100644) -> None:
    n = (name or "").encode("utf-8")
    if len(n) > 15:
        raise ValueError("ar member name too long")
    name_field = (n + b"/").ljust(16, b" ")
    ts_field = str(int(time.time())).encode("ascii").ljust(12, b" ")
    uid_field = b"0".ljust(6, b" ")
    gid_field = b"0".ljust(6, b" ")
    mode_field = oct(int(mode) & 0o777777)[2:].encode("ascii").ljust(8, b" ")
    size_field = str(len(data)).encode("ascii").ljust(10, b" ")
    header = name_field + ts_field + uid_field + gid_field + mode_field + size_field + b"`\n"
    if len(header) != 60:
        raise RuntimeError("invalid ar header length")
    buf.write(header)
    buf.write(data)
    if len(data) % 2 == 1:
        buf.write(b"\n")


def _tar_gz_from_dir(root_dir: Path, *, prefix: str = "") -> bytes:
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for p in sorted(root_dir.rglob("*")):
            rel = p.relative_to(root_dir).as_posix()
            arcname = f"{prefix}{rel}" if prefix else rel
            info = tf.gettarinfo(str(p), arcname=arcname)
            if info.isdir():
                tf.addfile(info)
                continue
            if info.isreg():
                with open(p, "rb") as f:
                    tf.addfile(info, fileobj=f)
                continue
            if info.issym():
                tf.addfile(info)
                continue
    return bio.getvalue()


def _safe_extract_strip_first_component(tgz_path: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, mode="r:gz") as tf:
        for m in tf.getmembers():
            name = (m.name or "").lstrip("/")
            if not name or name.startswith("..") or "/../" in name:
                continue
            parts = name.split("/", 1)
            if len(parts) < 2:
                continue
            stripped = parts[1].lstrip("/")
            if not stripped:
                continue
            out_path = dest_dir / stripped
            out_path_parent = out_path.parent.resolve()
            if not str(out_path_parent).startswith(str(dest_dir.resolve())):
                continue
            if m.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            if m.issym():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    out_path.symlink_to(m.linkname)
                except Exception:
                    pass
                continue
            if not m.isreg():
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            with open(out_path, "wb") as wf:
                wf.write(f.read())
            try:
                os.chmod(out_path, m.mode & 0o777)
            except Exception:
                pass


def _write_file(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.chmod(path, mode)


def build_deb_from_agent_tarball(
    *,
    tar_gz_path: str,
    output_deb_path: str,
    version: str,
    arch: str,
) -> DebBuildResult:
    out_path = Path(output_deb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deb_arch = _deb_arch_from_agent_arch(arch)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rootfs = tmp / "rootfs"
        agent_root = rootfs / "opt" / "eai-agent"
        _safe_extract_strip_first_component(tar_gz_path, agent_root)

        _write_file(
            rootfs / "usr" / "bin" / "eai-agent-run",
            """#!/usr/bin/env bash
set -euo pipefail
ENV_FILE="/etc/eai-agent/env"
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
fi
PORT="${EAI_AGENT_PORT:-9100}"
cd /opt/eai-agent
exec /opt/eai-agent/.venv/bin/python -m uvicorn agent_main:app --host 0.0.0.0 --port "$PORT"
""",
            0o755,
        )

        _write_file(
            rootfs / "lib" / "systemd" / "system" / "eai-agent.service",
            """[Unit]
Description=EAI Agent
After=network.target

[Service]
Type=simple
User=eai-agent
Group=eai-agent
EnvironmentFile=-/etc/eai-agent/env
WorkingDirectory=/opt/eai-agent
ExecStart=/usr/bin/eai-agent-run
Restart=no
RestartSec=2

[Install]
WantedBy=multi-user.target
""",
            0o644,
        )

        _write_file(rootfs / "etc" / "eai-agent" / "env", "", 0o600)

        control_dir = tmp / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        _write_file(
            control_dir / "control",
            "\n".join(
                [
                    "Package: eai-agent",
                    f"Version: {version}",
                    f"Architecture: {deb_arch}",
                    "Priority: optional",
                    "Section: utils",
                    "Maintainer: eai-platform",
                    "Depends: python3, python3-venv, python3-pip, systemd, curl",
                    "Description: EAI collector agent (offline deb)",
                    "",
                ]
            ),
            0o644,
        )
        _write_file(
            control_dir / "conffiles",
            "/etc/eai-agent/env\n",
            0o644,
        )
        _write_file(
            control_dir / "postinst",
            """#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="/var/log/eai-agent"
LOG_FILE="$LOG_DIR/install.log"
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"
chmod 600 "$LOG_FILE" || true

log() { echo "$(date -u +%FT%TZ) $*" >>"$LOG_FILE"; }

if ! id -u eai-agent >/dev/null 2>&1; then
  if command -v adduser >/dev/null 2>&1; then
    adduser --system --group --home /opt/eai-agent --no-create-home eai-agent >/dev/null 2>&1 || true
  else
    useradd -r -s /usr/sbin/nologin -d /opt/eai-agent eai-agent >/dev/null 2>&1 || true
    groupadd -f eai-agent >/dev/null 2>&1 || true
  fi
fi

mkdir -p /etc/eai-agent
chmod 700 /etc/eai-agent || true
touch /etc/eai-agent/env
chmod 600 /etc/eai-agent/env || true

kv_set_if_missing() {
  local k="$1"; local v="$2"
  if grep -qE "^${k}=" /etc/eai-agent/env; then
    return 0
  fi
  echo "${k}=${v}" >>/etc/eai-agent/env
}

if [ -n "${EAI_SERVER_BASE:-}" ]; then kv_set_if_missing "EAI_SERVER_BASE" "${EAI_SERVER_BASE}"; fi
if [ -n "${EAI_AGENT_ID:-}" ]; then kv_set_if_missing "EAI_AGENT_ID" "${EAI_AGENT_ID}"; fi
if [ -n "${EAI_AGENT_NAME:-}" ]; then kv_set_if_missing "EAI_AGENT_NAME" "${EAI_AGENT_NAME}"; fi
if [ -n "${EAI_AGENT_PORT:-}" ]; then kv_set_if_missing "EAI_AGENT_PORT" "${EAI_AGENT_PORT}"; fi
if [ -n "${EAI_AGENT_TUNNEL_TOKEN:-}" ]; then kv_set_if_missing "EAI_AGENT_TUNNEL_TOKEN" "${EAI_AGENT_TUNNEL_TOKEN}"; fi

chown -R eai-agent:eai-agent /opt/eai-agent >/dev/null 2>&1 || true

if [ ! -d /opt/eai-agent/.venv ]; then
  python3 -m venv /opt/eai-agent/.venv >>"$LOG_FILE" 2>&1 || { log "venv create failed"; exit 1; }
fi
/opt/eai-agent/.venv/bin/pip install --upgrade pip >>"$LOG_FILE" 2>&1 || true

if [ -d /opt/eai-agent/wheelhouse ]; then
  /opt/eai-agent/.venv/bin/pip install --no-index --find-links /opt/eai-agent/wheelhouse -r /opt/eai-agent/requirements.txt >>"$LOG_FILE" 2>&1 || { log "pip offline install failed"; exit 1; }
else
  /opt/eai-agent/.venv/bin/pip install -r /opt/eai-agent/requirements.txt >>"$LOG_FILE" 2>&1 || { log "pip install failed"; exit 1; }
fi

systemctl daemon-reload >>"$LOG_FILE" 2>&1 || true
systemctl enable --now eai-agent.service >>"$LOG_FILE" 2>&1 || { log "systemd enable failed"; exit 1; }

PORT="$(grep -E '^EAI_AGENT_PORT=' /etc/eai-agent/env | tail -n1 | cut -d= -f2 | tr -d '\r' || true)"
if [ -z "$PORT" ]; then PORT="9100"; fi
for i in 1 2 3 4 5; do
  sleep 1
  if curl -fsS "http://127.0.0.1:${PORT}/api/agent/health" >/dev/null 2>&1; then
    log "health ok"
    exit 0
  fi
done
log "health check failed"
exit 1
""",
            0o755,
        )
        _write_file(
            control_dir / "prerm",
            """#!/usr/bin/env bash
set -euo pipefail
if command -v systemctl >/dev/null 2>&1; then
  systemctl stop eai-agent.service >/dev/null 2>&1 || true
  systemctl disable eai-agent.service >/dev/null 2>&1 || true
fi
exit 0
""",
            0o755,
        )
        _write_file(
            control_dir / "postrm",
            """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "purge" ]; then
  rm -rf /opt/eai-agent >/dev/null 2>&1 || true
  rm -rf /etc/eai-agent >/dev/null 2>&1 || true
fi
exit 0
""",
            0o755,
        )

        control_tgz = _tar_gz_from_dir(control_dir)
        data_tgz = _tar_gz_from_dir(rootfs, prefix="./")
        with open(out_path, "wb") as f:
            f.write(b"!<arch>\n")
            _ar_write_member(f, name="debian-binary", data=b"2.0\n", mode=0o100644)
            _ar_write_member(f, name="control.tar.gz", data=control_tgz, mode=0o100644)
            _ar_write_member(f, name="data.tar.gz", data=data_tgz, mode=0o100644)

    sha = _sha256_file(str(out_path))
    return DebBuildResult(file_path=str(out_path), sha256=sha, version=version, arch=deb_arch)
