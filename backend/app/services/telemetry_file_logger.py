"""
控制隧道 / 预览 / 平台资源的结构化遥测，写入独立 JSONL 日志（默认保留 2 天）。

与 uvicorn access 日志分离，便于验收与论文级指标聚合。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_root() -> Path:
    return _PROJECT_ROOT


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


class TelemetryFileLogger:
    """
    按日落盘：logs/telemetry/telemetry-YYYY-MM-DD.jsonl
    每次写入前清理超过 TELEMETRY_FILE_RETENTION_DAYS 的同目录 jsonl。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_prune_monotonic: float = 0.0
        self._net_last: Optional[tuple[float, int, int]] = None  # (t, bytes_sent, bytes_recv)

    def _log_dir(self) -> Path:
        from app.core.config import settings

        rel = (getattr(settings, "TELEMETRY_FILE_LOG_DIR", None) or "logs/telemetry").strip()
        p = Path(rel)
        if p.is_absolute():
            return p
        return _project_root() / p

    def _retention_days(self) -> int:
        from app.core.config import settings

        try:
            d = int(getattr(settings, "TELEMETRY_FILE_RETENTION_DAYS", 2) or 2)
        except Exception:
            d = 2
        return max(1, min(d, 30))

    def _enabled(self) -> bool:
        from app.core.config import settings

        return bool(getattr(settings, "TELEMETRY_FILE_LOG_ENABLED", True))

    def _prune_old_files(self) -> None:
        now = time.time()
        if now - self._last_prune_monotonic < 60.0:
            return
        self._last_prune_monotonic = now
        root = self._log_dir()
        try:
            if not root.exists():
                return
            cutoff = now - float(self._retention_days()) * 86400.0
            for fp in root.iterdir():
                if not fp.is_file():
                    continue
                if not fp.name.endswith(".jsonl"):
                    continue
                try:
                    if fp.stat().st_mtime < cutoff:
                        fp.unlink(missing_ok=True)
                except OSError:
                    continue
        except OSError as e:
            logger.debug("telemetry prune skipped: %s", e)

    def log_event(self, *, category: str, event: str, **fields: Any) -> None:
        if not self._enabled():
            return
        self._prune_old_files()
        now = datetime.now(timezone.utc)
        body: Dict[str, Any] = {
            "ts": now.isoformat().replace("+00:00", "Z"),
            "category": category,
            "event": event,
        }
        for k, v in fields.items():
            if v is None:
                continue
            body[k] = _json_safe(v)
        line = json.dumps(body, ensure_ascii=False, sort_keys=True)
        path = self._log_dir() / f"telemetry-{now.strftime('%Y-%m-%d')}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with path.open("a", encoding="utf-8") as fp:
                    fp.write(line)
                    fp.write("\n")
        except Exception as e:
            logger.warning("telemetry write failed: %s", e)

    def log_system_resources(self) -> None:
        if not self._enabled():
            return
        try:
            import psutil  # type: ignore

            proc = psutil.Process()
            with proc.oneshot():
                rss = int(proc.memory_info().rss)
                vms = int(getattr(proc.memory_info(), "vms", 0) or 0)
                cpu_pct = float(proc.cpu_percent(interval=None))
            sys_cpu = float(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            now = time.time()
            net_sent = net_recv = 0
            net_rate_sent_bps: Optional[float] = None
            net_rate_recv_bps: Optional[float] = None
            try:
                n = psutil.net_io_counters()
                net_sent = int(n.bytes_sent)
                net_recv = int(n.bytes_recv)
                if self._net_last is not None:
                    t0, s0, r0 = self._net_last
                    dt = max(1e-3, now - t0)
                    net_rate_sent_bps = (net_sent - s0) / dt
                    net_rate_recv_bps = (net_recv - r0) / dt
                self._net_last = (now, net_sent, net_recv)
            except Exception:
                pass
            self.log_event(
                category="system",
                event="platform_resource_sample",
                platform_cpu_percent=round(cpu_pct, 3),
                platform_rss_bytes=rss,
                platform_vms_bytes=vms,
                system_cpu_percent=round(sys_cpu, 3),
                system_mem_percent=round(float(mem.percent), 3) if mem else None,
                system_mem_available_bytes=int(mem.available) if mem else None,
                net_bytes_sent_total=net_sent,
                net_bytes_recv_total=net_recv,
                net_rate_sent_bps=round(net_rate_sent_bps, 1) if net_rate_sent_bps is not None else None,
                net_rate_recv_bps=round(net_rate_recv_bps, 1) if net_rate_recv_bps is not None else None,
            )
        except Exception as e:
            self.log_event(category="system", event="platform_resource_sample_failed", error=str(e)[:300])

    def log_preview_counters(self, metrics: Dict[str, Any]) -> None:
        if not self._enabled():
            return
        self.log_event(category="preview", event="tunnel_mjpeg_counters", **metrics)


telemetry_file_logger = TelemetryFileLogger()


def log_telemetry_event(*, category: str, event: str, **fields: Any) -> None:
    telemetry_file_logger.log_event(category=category, event=event, **fields)


async def telemetry_periodic_sampler_loop(stop_event: asyncio.Event) -> None:
    """后台：周期性写入系统资源 + 隧道 MJPEG/WebRTC 计数器快照。"""
    from app.core.config import settings

    try:
        interval = float(getattr(settings, "TELEMETRY_SAMPLE_INTERVAL_SEC", 30) or 30)
    except Exception:
        interval = 30.0
    interval = max(5.0, min(interval, 600.0))
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass
        if not bool(getattr(settings, "TELEMETRY_FILE_LOG_ENABLED", True)):
            continue
        try:
            from app.services.agent_tunnel_manager import agent_tunnel_manager

            telemetry_file_logger.log_system_resources()
            telemetry_file_logger.log_preview_counters(agent_tunnel_manager.get_metrics())
        except Exception as e:
            logger.debug("telemetry periodic sample failed: %s", e)
