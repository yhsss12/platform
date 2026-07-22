"""通过 isaaclab.sh 子进程执行 Isaac Lab CLI（不在 FastAPI 进程内 import isaaclab）。"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.services.isaac_lab.paths import resolve_isaaclab_root, resolve_isaaclab_sh, PROJECT_ROOT


@dataclass
class IsaacLabCliRunResult:
    returncode: int
    command: list[str]
    stdout_path: Path
    stderr_path: Path
    timed_out: bool = False


class IsaacLabCliRunner:
    """封装 ./isaaclab.sh -p <script> 调用。"""

    def __init__(
        self,
        *,
        root: Path | None = None,
        sh_path: Path | None = None,
        python_path: str | None = None,
    ) -> None:
        self.root = root or resolve_isaaclab_root()
        self.sh_path = sh_path or resolve_isaaclab_sh(self.root)
        self.python_path = (python_path or "").strip() or None

    def is_ready(self) -> bool:
        return self.root is not None and self.root.is_dir() and self.sh_path is not None

    def build_command(self, script_relative: str, *args: str) -> list[str]:
        if self.sh_path is None:
            raise RuntimeError("isaaclab.sh is not configured")
        script = script_relative.strip()
        if script.startswith("/"):
            raise ValueError("script path must be relative to ISAACLAB_ROOT")
        return [str(self.sh_path), "-p", script, *args]

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["TERM"] = "xterm"
        if self.python_path:
            env["ISAACLAB_PYTHON"] = self.python_path
            # isaaclab.sh 通过 CONDA_PREFIX / _isaac_sim/python.sh 选 Python，不读取 ISAACLAB_PYTHON。
            # 当配置指向 conda 环境时，显式设置 CONDA_PREFIX，避免继承 backend 进程的 base conda。
            python = Path(self.python_path)
            if python.name == "python" and python.is_file():
                conda_prefix = python.parent.parent
                if (conda_prefix / "conda-meta").is_dir():
                    env["CONDA_PREFIX"] = str(conda_prefix)
                    env["CONDA_DEFAULT_ENV"] = conda_prefix.name
            elif python.name == "python.sh" and python.is_file():
                # 二进制 Isaac Sim：清除 conda 前缀，让 isaaclab.sh 走 _isaac_sim/python.sh
                env.pop("CONDA_PREFIX", None)
                env.pop("CONDA_DEFAULT_ENV", None)
        backend_root = (PROJECT_ROOT / "backend").resolve()
        if backend_root.is_dir():
            existing_pp = env.get("PYTHONPATH", "")
            prefix = str(backend_root)
            env["PYTHONPATH"] = f"{prefix}:{existing_pp}" if existing_pp else prefix
        return env

    def run_to_files(
        self,
        script_relative: str,
        *args: str,
        stdout_path: Path,
        stderr_path: Path,
        timeout: int,
    ) -> IsaacLabCliRunResult:
        if not self.is_ready():
            raise RuntimeError("Isaac Lab CLI runner is not ready")
        assert self.root is not None
        assert self.sh_path is not None

        cmd = self.build_command(script_relative, *args)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        timed_out = False
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_f:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.root),
                    env=self._build_env(),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    timeout=timeout,
                    check=False,
                )
                returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
            stderr_path.write_text(f"Command timed out after {timeout}s\n", encoding="utf-8")

        return IsaacLabCliRunResult(
            returncode=returncode,
            command=cmd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timed_out=timed_out,
        )

    def popen_to_log(
        self,
        script_relative: str,
        *args: str,
        log_path: Path,
    ) -> subprocess.Popen:
        """启动子进程并将 stdout/stderr 合并写入 log_path（非阻塞）。"""
        if not self.is_ready():
            raise RuntimeError("Isaac Lab CLI runner is not ready")
        assert self.root is not None

        cmd = self.build_command(script_relative, *args)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write("command: " + " ".join(cmd) + "\n\n")
        log_file.flush()
        return subprocess.Popen(
            cmd,
            cwd=str(self.root),
            env=self._build_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    @classmethod
    def from_settings(cls) -> "IsaacLabCliRunner":
        from app.core.config import settings

        return cls(python_path=settings.ISAACLAB_PYTHON)
