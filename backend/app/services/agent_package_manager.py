from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AgentPackageRef:
    version: str
    os: str
    arch: str
    file_path: str
    sha256: str
    sig_path: Optional[str] = None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_ed25519_signature_bytes(*, message: bytes, signature: bytes, public_key: bytes) -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except Exception as e:
        raise RuntimeError("缺少 cryptography，无法进行安装包签名校验") from e
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        return True
    except Exception:
        return False


def verify_package_signature(
    *,
    file_path: str,
    sig_path: str,
    public_keys_b64: List[str],
) -> None:
    """
    Verify detached Ed25519 signature.
    - signature file: raw bytes (base64 allowed if content is ascii)
    - public keys: base64-encoded 32-byte Ed25519 public keys
    """
    raw = Path(file_path).read_bytes()
    sig_raw = Path(sig_path).read_bytes()
    sig = sig_raw
    try:
        txt = sig_raw.decode("ascii").strip()
        if txt and all(c.isalnum() or c in "+/=\n\r\t " for c in txt):
            sig = base64.b64decode(txt)
    except Exception:
        sig = sig_raw

    ok = False
    for pk_b64 in public_keys_b64:
        pk_b64 = (pk_b64 or "").strip()
        if not pk_b64:
            continue
        try:
            pk = base64.b64decode(pk_b64)
        except Exception:
            continue
        if len(pk) != 32:
            continue
        if _verify_ed25519_signature_bytes(message=raw, signature=sig, public_key=pk):
            ok = True
            break
    if not ok:
        raise ValueError("安装包签名校验失败")


class AgentPackageManager:
    """
    Load and validate agent install packages from a manifest.
    Manifest schema (minimal):
    {
      "latest": "1.2.3",
      "packages": [
        {"version":"1.2.3","os":"linux","arch":"x86_64","path":"agent_packages/agent-linux-x86_64-1.2.3.tar.gz","sha256":"...","sig":"..."}
      ]
    }
    """

    def __init__(self, *, manifest_path: str) -> None:
        self.manifest_path = manifest_path

    def _load_manifest(self) -> Dict[str, Any]:
        p = Path(self.manifest_path)
        if not p.exists():
            return {"latest": "", "packages": []}
        return json.loads(p.read_text(encoding="utf-8"))

    def resolve(
        self,
        *,
        os_name: str,
        arch: str,
        version: Optional[str] = None,
    ) -> AgentPackageRef:
        m = self._load_manifest()
        latest = str(m.get("latest") or "").strip()
        ver = (version or "").strip() or latest
        if not ver:
            raise ValueError("未配置可用的 agent 安装包版本")
        pkgs = m.get("packages") or []
        if not isinstance(pkgs, list):
            pkgs = []

        os_norm = str(os_name or "").strip().lower()
        arch_norm = str(arch or "").strip().lower()
        hit = None
        for item in pkgs:
            if not isinstance(item, dict):
                continue
            if str(item.get("version") or "").strip() != ver:
                continue
            if str(item.get("os") or "").strip().lower() != os_norm:
                continue
            if str(item.get("arch") or "").strip().lower() != arch_norm:
                continue
            hit = item
            break
        if not hit:
            raise ValueError(f"未找到匹配的安装包：version={ver}, os={os_norm}, arch={arch_norm}")

        base = Path(self.manifest_path).resolve().parent
        rel = str(hit.get("path") or "").strip()
        if not rel:
            raise ValueError("安装包 manifest 缺少 path")
        file_path = str((base / rel).resolve())
        if not Path(file_path).exists():
            raise FileNotFoundError(f"安装包不存在：{file_path}")

        sha256 = str(hit.get("sha256") or "").strip().lower()
        if not sha256 or len(sha256) != 64:
            raise ValueError("安装包 manifest 缺少 sha256")
        got = _sha256_file(file_path)
        if got.lower() != sha256:
            raise ValueError("安装包 sha256 校验失败")

        sig_rel = str(hit.get("sig") or "").strip()
        sig_path = str((base / sig_rel).resolve()) if sig_rel else None
        if sig_path and not Path(sig_path).exists():
            raise FileNotFoundError(f"签名文件不存在：{sig_path}")

        return AgentPackageRef(
            version=ver,
            os=os_norm,
            arch=arch_norm,
            file_path=file_path,
            sha256=sha256,
            sig_path=sig_path,
        )


def default_package_manager() -> AgentPackageManager:
    backend_dir = Path(__file__).resolve().parents[2]
    manifest = os.path.join(str(backend_dir), "agent_packages", "manifest.json")
    return AgentPackageManager(manifest_path=manifest)


def resolve_linux_x86_64_tarball_path(*, override_path: Optional[str] = None) -> Path:
    """
    供 HTTP /static/bin/agent_linux_x64.tar.gz 使用。
    - 若 override_path 非空且指向现有文件：直接返回（不做 manifest sha256 校验，由部署方保证一致性）。
    - 否则按 agent_packages/manifest.json 解析并校验 sha256。
    """
    raw = (override_path or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path(__file__).resolve().parents[2] / p).resolve()
        else:
            p = p.resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"AGENT_LINUX_X64_TARBALL_PATH 无效或文件不存在：{p}")

    mgr = default_package_manager()
    ref = mgr.resolve(os_name="linux", arch="x86_64")
    return Path(ref.file_path)
