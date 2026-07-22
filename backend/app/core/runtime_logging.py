"""
平台后端运行日志落盘：全量写入 root logger，按天轮转，默认保留约 2 天。

环境变量：
- RUNTIME_LOG_DIR：日志目录，默认 <项目根>/logs/runtime
- RUNTIME_LOG_DISABLE：设为 1/true 时关闭文件日志（仍输出控制台）
"""

from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_MARK_ATTR = "_eai_runtime_timed_rotating"


def configure_runtime_file_logging() -> None:
    """幂等：向 root 挂载按天轮转的文件 Handler；失败静默跳过（打印一行提示）。"""
    raw = os.getenv("RUNTIME_LOG_DISABLE", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return

    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, _MARK_ATTR, False):
            return

    # __file__ = backend/app/core/runtime_logging.py → parents[2] = backend/
    backend_dir = Path(__file__).resolve().parents[2]
    project_root = backend_dir.parent
    default_dir = project_root / "logs" / "runtime"
    log_dir = Path(os.getenv("RUNTIME_LOG_DIR", str(default_dir))).expanduser()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"⚠ 运行日志目录不可用，跳过落盘: {log_dir} ({e})")
        return

    log_file = log_dir / "platform-backend.log"
    try:
        handler = TimedRotatingFileHandler(
            str(log_file),
            when="midnight",
            interval=1,
            backupCount=2,
            encoding="utf-8",
            utc=False,
        )
    except OSError as e:
        print(f"⚠ 运行日志文件不可用，跳过落盘: {log_file} ({e})")
        return

    setattr(handler, _MARK_ATTR, True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)

    # 确保 uvicorn / FastAPI 体系日志向上汇总到 root（文件 Handler 才能收到）
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).propagate = True

    print(f"✓ 运行日志已落盘: {log_file}（按天轮转，保留约 2 个历史文件）")
