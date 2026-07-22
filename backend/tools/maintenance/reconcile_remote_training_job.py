#!/usr/bin/env python3
"""对已存在的 remote_ssh 训练任务执行日志/checkpoint 同步与 DB 索引写入。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded

ensure_dotenv_loaded()


def _log(verbose: bool, message: str, *, t0: Optional[list[float]] = None) -> None:
    if t0 is not None:
        elapsed = round((time.perf_counter() - t0[0]) * 1000, 1)
        print(f"[{elapsed}ms] {message}")
    elif verbose:
        print(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile remote_ssh training job runtime state")
    parser.add_argument("train_job_id", help="train job id, e.g. train_20260626_125605_11be")
    parser.add_argument("--local-only", action="store_true", help="skip SSH pull; only sync local runtime to DB")
    parser.add_argument("--db-timeout", type=float, default=5.0, help="DB operation timeout seconds")
    parser.add_argument("--verbose", action="store_true", help="print step timings")
    args = parser.parse_args()

    verbose = bool(args.verbose)
    t0 = [time.perf_counter()]
    job_id = args.train_job_id.strip()
    result: dict[str, Any] = {
        "ok": False,
        "trainJobId": job_id,
        "localOnly": bool(args.local_only),
        "steps": {},
        "dbAvailable": False,
        "dbLevel": None,
    }

    def step(name: str, ok: bool, **extra: Any) -> None:
        payload = {"ok": ok, "ms": round((time.perf_counter() - t0[0]) * 1000, 1), **extra}
        result["steps"][name] = payload
        if verbose:
            status = "OK" if ok else "FAIL"
            detail = " ".join(f"{k}={v}" for k, v in extra.items())
            print(f"[{payload['ms']}ms] {name}: {status} {detail}".strip())

    from app.core.db_health import check_db_health
    from app.services.training_metrics import normalized_training_metrics, sync_metrics_from_logs
    from app.services.training_service import _read_json, read_training_job_log
    from app.services.workspace_runtime_paths import resolve_training_job_root

    _log(verbose, "connecting database", t0=t0)
    health = check_db_health(connect_timeout=args.db_timeout, statement_timeout_ms=int(args.db_timeout * 1000))
    result["dbLevel"] = health.level
    result["dbAvailable"] = health.ok and health.level != "DB_HEALTH_FAIL"
    step(
        "db_health",
        result["dbAvailable"],
        level=health.level,
        connectMs=health.connect_ms,
        select1Ms=health.select1_ms,
        idleInTransaction=health.idle_in_transaction,
        blockingLocks=health.blocking_locks,
    )
    if not result["dbAvailable"]:
        result["error"] = "DB_UNAVAILABLE"
        result["dbErrors"] = health.errors
        result["dbWarnings"] = health.warnings
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    _log(verbose, "loading job runtime", t0=t0)
    train_job_dir = resolve_training_job_root(job_id)
    if train_job_dir is None:
        result["error"] = "RUNTIME_NOT_FOUND"
        step("load_runtime", False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    step("load_runtime", True, path=str(train_job_dir))

    status_data = _read_json(train_job_dir / "status.json")
    step("load_status", bool(status_data))

    if args.local_only:
        _log(verbose, "syncing metrics from local logs", t0=t0)
        sync_metrics_from_logs(train_job_dir, status_data)
        step("sync_metrics_runtime", True)
        if str(status_data.get("executionMode") or "").lower() == "remote_ssh":
            step("skip_ssh", True, message="local-only mode")
    else:
        from app.services.training_remote_runner import reconcile_remote_training_job_runtime

        _log(verbose, "reconciling remote runtime via SSH", t0=t0)
        remote_result = reconcile_remote_training_job_runtime(job_id)
        reconcile_ok = bool(remote_result.get("ok"))
        reconcile_extra = {k: v for k, v in remote_result.items() if k != "ok"}
        step("remote_reconcile", reconcile_ok, **reconcile_extra)
        status_data = _read_json(train_job_dir / "status.json")

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    from app.services.training_job_sync_service import (
        get_training_job_summary_from_db,
        sync_training_job_from_runtime,
    )

    _log(verbose, "syncing status/metrics/assets to database", t0=t0)

    sync_payload: dict[str, Any] = {}
    sync_err: Optional[str] = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(sync_training_job_from_runtime, job_id)
            sync_payload = future.result(timeout=max(5.0, args.db_timeout * 3)) or {}
    except FuturesTimeoutError:
        sync_err = "DB_TIMEOUT"
    except Exception as exc:
        sync_err = f"{type(exc).__name__}: {exc}"

    if sync_err:
        step("db_sync", False, error=sync_err)
        result["error"] = sync_err
    else:
        sync_ok = bool(sync_payload.get("ok", True))
        sync_extra = {k: v for k, v in sync_payload.items() if k != "ok"}
        step("db_sync", sync_ok, **sync_extra)

    _log(verbose, "reading database summary", t0=t0)
    db_summary: Optional[dict[str, Any]] = None
    summary_err: Optional[str] = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_training_job_summary_from_db, job_id)
            db_summary = future.result(timeout=max(1.0, args.db_timeout))
    except FuturesTimeoutError:
        summary_err = "DB_TIMEOUT"
    except Exception as exc:
        summary_err = f"{type(exc).__name__}: {exc}"
    if summary_err:
        step("db_read_summary", False, error=summary_err)
    else:
        step("db_read_summary", db_summary is not None)

    metrics = normalized_training_metrics(train_job_dir, _read_json(train_job_dir / "status.json"))

    final_asset: dict[str, Any] = {}
    assets_warning: Optional[str] = None
    try:
        from app.services.workspace_model_asset_service import list_training_job_model_assets_detail

        assets_payload = list_training_job_model_assets_detail(job_id, sync_db=False)
        assets_warning = assets_payload.get("warning")
        final_asset = next(
            (
                row
                for row in assets_payload.get("modelAssets") or []
                if str(row.get("checkpointKind") or "").lower() == "final"
            ),
            {},
        )
        step("db_read_assets", True, count=len(assets_payload.get("modelAssets") or []))
    except Exception as exc:
        step("db_read_assets", False, error=str(exc))
        assets_warning = str(exc)

    runtime_status = _read_json(train_job_dir / "status.json")
    sync_ok = not sync_err and bool((sync_payload or {}).get("ok", True))
    if not sync_ok and (db_summary or {}).get("status") == runtime_status.get("status") == "completed":
        sync_ok = True
        result.setdefault("warnings", []).append("db_sync_lock_timeout_but_state_consistent")

    result.update(
        {
            "ok": sync_ok,
            "runtimeStatus": runtime_status.get("status"),
            "dbStatus": (db_summary or {}).get("status"),
            "progress": (db_summary or {}).get("progress", runtime_status.get("progress")),
            "epoch": (db_summary or {}).get("epoch", runtime_status.get("epoch")),
            "totalEpochs": (db_summary or {}).get("totalEpochs", runtime_status.get("totalEpochs")),
            "lossSeriesLen": len(metrics.get("lossSeries") or []),
            "finalAssetDisplayStatus": final_asset.get("displayStatus"),
            "finalAssetStatus": final_asset.get("status"),
            "logHasEpoch": "Epoch" in read_training_job_log(job_id),
            "assetsWarning": assets_warning,
            "elapsedMs": round((time.perf_counter() - t0[0]) * 1000, 1),
        }
    )
    if sync_err:
        result["error"] = sync_err
    elif assets_warning and result["ok"]:
        result["warning"] = assets_warning

    try:
        from app.core.database import engine

        engine.dispose()
        step("dispose_engine", True)
    except Exception as exc:
        step("dispose_engine", False, error=str(exc))

    _log(verbose, "done", t0=t0)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
