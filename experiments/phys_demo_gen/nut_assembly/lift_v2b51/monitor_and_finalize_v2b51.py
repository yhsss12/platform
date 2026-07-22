#!/usr/bin/env python3
"""监控 PID 1007916 搜索进程，完成后生成报告并执行分支。"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_V2B51_DIR = Path(__file__).resolve().parent
for path in (_EXPERIMENT_DIR, _V2B51_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUTPUT_DIR = _EXPERIMENT_DIR / "outputs" / "lift_v2b51"
SEARCH_LOG = OUTPUT_DIR / "search.log"
JSONL = OUTPUT_DIR / "lift_v2b51_rollout_samples.jsonl"
PID = 1007916
POLL_SEC = 90
STALL_SEC = 1800
WEAK_THRESH = 0.002


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _log_size() -> int:
    return SEARCH_LOG.stat().st_size if SEARCH_LOG.exists() else 0


def _parse_progress() -> dict:
    info = {"eval_progress": None, "partial": None, "best_nut_lift": None}
    if not SEARCH_LOG.exists():
        return info
    for line in reversed(SEARCH_LOG.read_text(encoding="utf-8", errors="replace").splitlines()):
        m = re.search(r"demo_3 v2b51:\s*(\d+)/(\d+)\s+partial=(\d+)\s+best_nut_lift=([\d.\-]+)", line)
        if m:
            info["eval_progress"] = f"{m.group(1)}/{m.group(2)}"
            info["partial"] = int(m.group(3))
            info["best_nut_lift"] = float(m.group(4))
            break
    return info


def _quick_stats_from_jsonl() -> dict | None:
    if not JSONL.exists():
        return None
    partial = weak = 0
    max_lift = -999.0
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        nz = float(r.get("nut_z_lift_delta", r.get("nut_lift_phase_delta", 0.0)))
        if r.get("partial_lift_success"):
            partial += 1
        if nz >= WEAK_THRESH:
            weak += 1
        max_lift = max(max_lift, nz)
    return {
        "partial_lift_success_count": partial,
        "weak_lift_positive_count": weak,
        "max_nut_lift_delta": max_lift,
        "jsonl_lines": sum(1 for _ in JSONL.read_text().splitlines() if _.strip()),
    }


def _status_line() -> str:
    prog = _parse_progress()
    stats = _quick_stats_from_jsonl()
    parts = [f"pid={PID} alive={_pid_alive(PID)}", f"log_bytes={_log_size()}"]
    if prog.get("eval_progress"):
        parts.append(f"progress={prog['eval_progress']}")
        parts.append(f"log_partial={prog['partial']}")
        parts.append(f"log_best_nut_lift={prog['best_nut_lift']}")
    if stats:
        parts.append(f"jsonl_partial={stats['partial_lift_success_count']}")
        parts.append(f"jsonl_weak={stats['weak_lift_positive_count']}")
        parts.append(f"jsonl_max_lift={stats['max_nut_lift_delta']:.4f}")
    return " | ".join(parts)


def _run_branch(report: dict) -> None:
    partial = int(report.get("partial_lift_success_count", 0))
    weak = int(report.get("weak_lift_positive_count", 0))
    if partial > 0:
        subprocess.run(
            [sys.executable, str(_V2B51_DIR / "build_v1g_contact_aware_dataset_draft.py")],
            check=False,
        )
    elif weak > 0:
        subprocess.run(
            [sys.executable, str(_V2B51_DIR / "run_lift_v2b52_cem_search.py")],
            check=False,
        )
    else:
        subprocess.run(
            [sys.executable, str(_V2B51_DIR / "generate_contact_failure_diagnosis.py")],
            check=False,
        )


def main() -> int:
    print(f"[monitor] watching PID {PID}", flush=True)
    last_size = _log_size()
    last_growth = time.time()

    while _pid_alive(PID):
        size = _log_size()
        if size > last_size:
            last_size = size
            last_growth = time.time()
        elif time.time() - last_growth > STALL_SEC:
            print(f"[monitor] STALL > {STALL_SEC}s without log growth; not killing per user policy unless confirmed", flush=True)
        print(f"[monitor] {_status_line()}", flush=True)
        time.sleep(POLL_SEC)

    print(f"[monitor] PID {PID} exited", flush=True)
    for _ in range(30):
        if JSONL.exists():
            break
        time.sleep(2)

    if not JSONL.exists():
        print(f"[monitor] ERROR: missing {JSONL}", file=sys.stderr)
        return 2

    from generate_lift_v2b51_final_report import generate_reports

    report = generate_reports(jsonl_path=JSONL, output_dir=OUTPUT_DIR, seed=42, max_evals=1200)
    print(f"[monitor] report written branch={report['branch_recommendation']}", flush=True)
    _run_branch(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
