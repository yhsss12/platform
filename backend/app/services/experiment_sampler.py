from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Dict, Optional

import psutil

from app.core.config import settings
from app.services.experiment_logger import log_experiment_event


@dataclass
class SamplerContext:
    run_id: str
    scenario_id: Optional[str]
    task_id: Optional[str]
    job_id: Optional[str]
    device_id: Optional[str]
    agent_id: Optional[str]
    method: Optional[str]
    interval_sec: float
    relay_pid: Optional[int]


class ExperimentSamplerService:
    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}
        self._contexts: Dict[str, SamplerContext] = {}
        self._lock = asyncio.Lock()

    async def start(self, ctx: SamplerContext) -> None:
        if not bool(getattr(settings, "EXPERIMENT_ENABLED", False)):
            return
        async with self._lock:
            await self._stop_locked(ctx.run_id)
            task = asyncio.create_task(self._run(ctx), name=f"experiment-sampler:{ctx.run_id}")
            self._tasks[ctx.run_id] = task
            self._contexts[ctx.run_id] = ctx

    async def stop(self, run_id: str) -> bool:
        if not bool(getattr(settings, "EXPERIMENT_ENABLED", False)):
            return False
        async with self._lock:
            return await self._stop_locked(run_id)

    async def stop_all(self) -> None:
        async with self._lock:
            for run_id in list(self._tasks.keys()):
                await self._stop_locked(run_id)

    async def is_active(self, run_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(run_id)
            return bool(task and not task.done())

    async def _stop_locked(self, run_id: str) -> bool:
        task = self._tasks.pop(run_id, None)
        self._contexts.pop(run_id, None)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def _run(self, ctx: SamplerContext) -> None:
        process = psutil.Process(os.getpid())
        relay_process = process
        if ctx.relay_pid and ctx.relay_pid > 0:
            try:
                relay_process = psutil.Process(int(ctx.relay_pid))
            except Exception:
                relay_process = process

        process.cpu_percent(interval=None)
        relay_process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        try:
            while True:
                await asyncio.sleep(ctx.interval_sec)
                net = psutil.net_io_counters()
                process_mem = process.memory_info()
                relay_mem = relay_process.memory_info()
                log_experiment_event(
                    role="platform",
                    event="platform_resource_sample",
                    run_id=ctx.run_id,
                    scenario_id=ctx.scenario_id,
                    task_id=ctx.task_id,
                    job_id=ctx.job_id,
                    device_id=ctx.device_id,
                    agent_id=ctx.agent_id,
                    method=ctx.method,
                    interval_sec=ctx.interval_sec,
                    process_pid=process.pid,
                    relay_pid=relay_process.pid,
                    system_cpu_percent=psutil.cpu_percent(interval=None),
                    platform_cpu_percent=process.cpu_percent(interval=None),
                    relay_cpu_percent=relay_process.cpu_percent(interval=None),
                    platform_mem_bytes=process_mem.rss,
                    platform_rss_bytes=process_mem.rss,
                    relay_mem_bytes=relay_mem.rss,
                    relay_rss_bytes=relay_mem.rss,
                    platform_tx_bytes=net.bytes_sent,
                    platform_rx_bytes=net.bytes_recv,
                )
        except asyncio.CancelledError:
            raise


_sampler_service = ExperimentSamplerService()


def get_experiment_sampler_service() -> ExperimentSamplerService:
    return _sampler_service
