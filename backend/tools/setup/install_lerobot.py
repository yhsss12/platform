#!/usr/bin/env python3
"""安装 LeRobot，且在 Linux 上避免 pip 拉取需本地编译的 evdev。

pynput 声明依赖发行版名「evdev」；evdev-binary 可提供同名模块，但 pip 解析器不认。
因此：先装 evdev-binary，再以 --no-deps 安装 pynput，最后 --no-deps 安装 lerobot 并补全其余依赖（跳过 pynput）。
"""
from __future__ import annotations

import subprocess
import sys

try:
    from packaging.requirements import Requirement
    from packaging.markers import default_environment
except ImportError:
    print("需要 packaging（一般随 pip 可用）: pip install packaging", file=sys.stderr)
    sys.exit(1)


def _pip_install(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def main() -> None:
    marker_env = dict(default_environment())
    marker_env.setdefault("extra", "")

    _pip_install("evdev-binary>=1.9.3")
    _pip_install("pynput>=1.7.7,<1.9.0", "--no-deps")
    _pip_install("python-xlib>=0.17", "six")
    _pip_install("lerobot", "--no-deps")

    from importlib import metadata

    dist = metadata.distribution("lerobot")
    lines = dist.requires or []
    # Dockerfile 已装 CPU 版 torch；lerobot 元数据若再装 torch/torchvision 会拉取 GB 级 nvidia-* CUDA 包
    _skip_names = {
        "pynput",
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
    }

    to_install: list[str] = []
    for raw in lines:
        req = Requirement(raw)
        name = req.name.lower()
        if name in _skip_names or name.startswith("nvidia-"):
            continue
        if req.marker and not req.marker.evaluate(marker_env):
            continue
        to_install.append(str(req))

    if to_install:
        _pip_install(*to_install)

    # 确认仍为 CPU torch（避免被后续依赖升级成 CUDA 版）
    _pip_install(
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
        "torch",
        "torchvision",
        "torchaudio",
    )


if __name__ == "__main__":
    main()
