"""
RQ worker execution entrypoint.

RQ 通过字符串 "worker.execute_task" 调用该模块函数。
本文件只负责复用现有执行逻辑，不改业务实现。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
import threading
from app.services.task_job_store import update_task_status, is_cancelled
from app.services.task_profiles import get_task_profile


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _params(task: Dict[str, Any]) -> Dict[str, Any]:
    params = task.get("params")
    if isinstance(params, dict):
        return params
    return task


_task_gates_lock = threading.Lock()
_task_gates: dict[str, threading.BoundedSemaphore] = {}


def _gate_for(task_type: str) -> threading.BoundedSemaphore:
    t = (task_type or "").strip().lower()
    with _task_gates_lock:
        gate = _task_gates.get(t)
        if gate is not None:
            return gate
        limit = int(get_task_profile(t).concurrency_limit)
        gate = threading.BoundedSemaphore(value=max(1, limit))
        _task_gates[t] = gate
        return gate


def run_annotation(task: Dict[str, Any]) -> Any:
    from app.services.annotation_service import AnnotationService, resolve_annotation_executor_file_path
    from app.services.hdf5_service import HDF5Service

    p = _params(task)
    service = AnnotationService(HDF5Service(data_dir="/tmp/hdf5_data"))
    file_path = resolve_annotation_executor_file_path(p)
    return asyncio.run(
        service._run_annotation(
            job_id=str(p.get("job_id") or task.get("task_id") or ""),
            file_path=file_path,
            camera_name=p.get("camera_name"),
            model=p.get("model"),
            openai_api_key=p.get("openai_api_key"),
            openai_base_url=p.get("openai_base_url"),
        )
    )


def run_conversion(task: Dict[str, Any]) -> Any:
    from app.api import routes_conversion as conv

    p = _params(task)
    job_payload = p.get("job")
    if isinstance(job_payload, dict):
        try:
            conv.jobs_store[str(job_payload.get("jobId") or p.get("job_id") or task.get("task_id") or "")] = conv.ConversionJob(**job_payload)
        except Exception:
            pass

    mode = str(p.get("mode") or "").strip().lower()
    job_id = str(p.get("job_id") or task.get("task_id") or "")
    mcap_path = conv.resolve_executor_conversion_mcap_path(p)
    config = p.get("config") or {}

    if mode == "hdf5":
        return conv.run_conversion_task(
            job_id=job_id,
            mcap_path=mcap_path,
            output_path=str(p.get("output_path") or ""),
            config=config,
        )
    if mode == "lerobot_with_limit":
        return asyncio.run(
            conv.run_lerobot_with_limit(
                job_id=job_id,
                mcap_path=mcap_path,
                output_repo_id=str(p.get("output_repo_id") or ""),
                config=config,
                jobs_store=conv.jobs_store,
            )
        )
    if mode == "lerobot_direct":
        return asyncio.run(
            conv.convert_mcap_to_lerobot_task(
                job_id=job_id,
                mcap_path=mcap_path,
                output_repo_id=str(p.get("output_repo_id") or ""),
                config=config,
                jobs_store=conv.jobs_store,
            )
        )
    raise ValueError(f"Unknown conversion mode: {mode}")


def run_collect(task: Dict[str, Any]) -> Any:
    from app.api.routes_jobs import run_collection_service
    from app.realtime.job_ws import manager as job_ws_manager

    p = _params(task)
    return asyncio.run(
        run_collection_service(
            job_id=str(p.get("job_id") or task.get("task_id") or ""),
            duration=int(p.get("duration") or 0),
            job_ws=job_ws_manager,
        )
    )


def run_batch(task: Dict[str, Any]) -> Any:
    from app.services.sync_batch_service import run_batch_job

    p = _params(task)
    return asyncio.run(run_batch_job(str(p.get("job_id") or task.get("task_id") or "")))


def run_stress(task: Dict[str, Any]) -> Any:
    """队列压测：sleep + 可选短 CPU 忙等，不产生 MinIO/业务副作用。"""
    import time

    p = _params(task)
    sleep_ms = float(p.get("sleep_ms", 50) or 0)
    cpu_ms = float(p.get("cpu_ms", 0) or 0)
    if sleep_ms > 0:
        time.sleep(min(sleep_ms, 60_000) / 1000.0)
    iterations = 0
    if cpu_ms > 0:
        cpu_ms = min(cpu_ms, 60_000)
        deadline = time.monotonic() + cpu_ms / 1000.0
        while time.monotonic() < deadline:
            iterations += 1
    return {"ok": True, "sleep_ms": sleep_ms, "cpu_ms": cpu_ms, "iterations": iterations}


def execute_task(task: dict):
    task_id = str((task or {}).get("task_id") or "").strip()
    print(f"[Worker] Start task {task_id}")
    task_type = (task or {}).get("type")
    if task_id and is_cancelled(task_id):
        print(f"[Cancel] Task {task_id} already cancelled")
        return None
    if task_id:
        update_task_status(task_id, "running", started_at=_now())
    result = None

    gate = _gate_for(str(task_type or ""))
    gate.acquire()
    try:
        if task_type == "annotation":
            result = run_annotation(task)
            if task_id and result is not None:
                try:
                    from app.crud.data_asset import sync_persist_instruction_text_after_annotation

                    if not sync_persist_instruction_text_after_annotation(_params(task), str(result)):
                        print(
                            f"[Worker] WARN annotation 未写回 data_assets.instruction_text task_id={task_id} "
                            f"(路径未匹配资产，左侧「已标注」可能仍为空)"
                        )
                except Exception as ex:
                    print(f"[Worker] WARN annotation 回写 data_assets 失败 task_id={task_id}: {ex}")
        elif task_type == "conversion":
            result = run_conversion(task)
        elif task_type == "export":
            raise ValueError(
                "导出任务已改为在 API 进程事件循环中调度，不再由 RQ worker 执行；请使用 POST /export/jobs"
            )
        elif task_type == "collect":
            result = run_collect(task)
        elif task_type == "batch":
            result = run_batch(task)
        elif task_type == "stress":
            result = run_stress(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        if task_id and not is_cancelled(task_id):
            update_task_status(task_id, "success", result={"value": result} if result is not None else {}, finished_at=_now())
        elif task_id and is_cancelled(task_id):
            print(f"[Cancel] Task {task_id} cancelled")
        return result
    except Exception as e:
        if task_id:
            if is_cancelled(task_id):
                print(f"[Cancel] Task {task_id} raised after cancel: {e}")
                return None
            update_task_status(task_id, "failed", error=str(e), finished_at=_now())
        raise
    finally:
        try:
            gate.release()
        except Exception:
            pass
