"""统一加载项目环境变量（根目录 .env + backend/.env，后者可覆盖前者）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_LOADED = False
_LOADED_PATHS: list[Path] = []


def backend_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def project_root() -> Path:
    return backend_root().parent


def ensure_dotenv_loaded(*, verbose: bool = False) -> list[Path]:
    """加载 .env；已加载则跳过。返回实际加载的文件列表。"""
    global _LOADED, _LOADED_PATHS
    if _LOADED:
        return list(_LOADED_PATHS)

    loaded: list[Path] = []
    try:
        from dotenv import load_dotenv
    except ImportError:
        _LOADED = True
        if verbose:
            print("⚠ python-dotenv 未安装，跳过 .env 加载")
        return loaded

    root = project_root()
    backend = backend_root()
    for path in (root / ".env", backend / ".env"):
        if path.is_file():
            load_dotenv(path, override=bool(loaded))
            loaded.append(path)
            if verbose:
                print(f"✓ 已加载环境变量: {path}")

    _LOADED = True
    _LOADED_PATHS = loaded
    return list(loaded)


def reset_dotenv_loaded_for_tests() -> None:
    global _LOADED, _LOADED_PATHS
    _LOADED = False
    _LOADED_PATHS = []


def train_node_password_configured() -> bool:
    ensure_dotenv_loaded()
    password = (os.environ.get("TRAIN_NODE_L20_PASSWORD") or "").strip()
    if password:
        return True
    key_path = (os.environ.get("TRAIN_NODE_L20_SSH_KEY") or "").strip()
    return bool(key_path and os.path.isfile(os.path.expanduser(key_path)))
