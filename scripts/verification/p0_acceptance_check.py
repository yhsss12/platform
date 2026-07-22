#!/usr/bin/env python3
"""P0 performance fix acceptance — real environment checks."""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests

API = "http://127.0.0.1:8000/api"
LOGIN_USER = "Pibot0001"
LOGIN_PASS = "jinlian1234"
PROJECT = Path(__file__).resolve().parents[2]


def login(session: requests.Session) -> None:
    session.headers["X-Session-Id"] = str(uuid.uuid4())
    resp = session.post(
        f"{API}/auth/login",
        json={"username": LOGIN_USER, "password": LOGIN_PASS},
        timeout=30,
    )
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"login failed: {data}")
    session.headers["Authorization"] = f"Bearer {data['data']['access_token']}"


def poll_isaac_job(session: requests.Session, job_id: str, timeout_sec: int = 900) -> dict:
    phases: list[dict] = []
    started = time.time()
    last_phase = None
    while time.time() - started < timeout_sec:
        resp = session.get(f"{API}/workspace/isaaclab-franka-stack-cube/jobs/{job_id}/status", timeout=30)
        payload = resp.json()
        phase = payload.get("phase")
        if phase != last_phase:
            phases.append(
                {
                    "t": round(time.time() - started, 1),
                    "phase": phase,
                    "phaseLabel": payload.get("phaseLabel"),
                    "progressMessage": payload.get("progressMessage"),
                    "resolvedDevice": payload.get("resolvedDevice"),
                    "status": payload.get("status"),
                }
            )
            last_phase = phase
            print(f"  [{phases[-1]['t']}s] phase={phase} msg={payload.get('progressMessage')!r}", flush=True)
        if payload.get("status") in ("completed", "failed"):
            return {"final": payload, "phases": phases, "elapsed": time.time() - started}
        time.sleep(5)
    return {"final": payload, "phases": phases, "elapsed": time.time() - started, "timeout": True}


def grep_file(path: Path, patterns: list[str], max_lines: int = 8) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {p: [] for p in patterns}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        low = line.lower()
        for p in patterns:
            if p.lower() in low and len(out[p]) < max_lines:
                out[p].append(line.strip()[:200])
    return out


def main() -> int:
    session = requests.Session()
    login(session)
    report: dict = {"timestamp": datetime.now().isoformat()}

    # --- Expert policy job (save_video=false for faster device check) ---
    print("\n=== Starting expert_policy job ===", flush=True)
    t0 = time.time()
    resp = session.post(
        f"{API}/workspace/isaaclab-franka-stack-cube/generate-async",
        json={
            "episodes": 1,
            "seed": 0,
            "saveVideo": False,
            "saveTrajectory": True,
            "headless": True,
            "generationMode": "expert_policy",
        },
        timeout=30,
    )
    resp.raise_for_status()
    expert_job = resp.json()["jobId"]
    print(f"expert jobId={expert_job}", flush=True)
    expert_poll = poll_isaac_job(session, expert_job, timeout_sec=600)
    expert_dir = PROJECT / "runs/data_generation/jobs" / expert_job

    job_config = json.loads((expert_dir / "metadata/job_config.json").read_text()) if (expert_dir / "metadata/job_config.json").is_file() else {}
    status_json = json.loads((expert_dir / "status.json").read_text()) if (expert_dir / "status.json").is_file() else {}
    stdout_log = expert_dir / "artifacts/expert_policy.stdout.log"

    report["expert"] = {
        "jobId": expert_job,
        "elapsed_sec": round(expert_poll["elapsed"], 1),
        "job_config_device": {k: job_config.get(k) for k in ("requestedDevice", "resolvedDevice", "isGpuRequested", "cudaVisibleDevices", "torchCudaAvailable")},
        "status_device": {k: status_json.get(k) for k in ("resolvedDevice", "torchCudaAvailable", "phase", "phaseLabel", "progressMessage", "phaseTimings")},
        "api_final": {k: expert_poll["final"].get(k) for k in ("resolvedDevice", "phaseLabel", "progressMessage", "phase", "status")},
        "phase_transitions": expert_poll["phases"],
        "stdout_grep": grep_file(stdout_log, ["Environment device", "device", "cuda", "cpu"]),
        "run_log_tail": (expert_dir / "logs/run.log").read_text(encoding="utf-8", errors="replace")[-800:] if (expert_dir / "logs/run.log").is_file() else "",
    }

    # --- Mimic job ---
    print("\n=== Starting mimic_auto job ===", flush=True)
    resp = session.post(
        f"{API}/workspace/isaaclab-franka-stack-cube/generate-async",
        json={
            "episodes": 1,
            "seed": 0,
            "saveVideo": False,
            "saveTrajectory": True,
            "headless": True,
            "generationMode": "mimic_auto",
        },
        timeout=30,
    )
    resp.raise_for_status()
    mimic_job = resp.json()["jobId"]
    print(f"mimic jobId={mimic_job}", flush=True)
    mimic_poll = poll_isaac_job(session, mimic_job, timeout_sec=900)
    mimic_dir = PROJECT / "runs/data_generation/jobs" / mimic_job
    mimic_status = json.loads((mimic_dir / "status.json").read_text()) if (mimic_dir / "status.json").is_file() else {}
    live_status = json.loads((mimic_dir / "live/status.json").read_text()) if (mimic_dir / "live/status.json").is_file() else {}

    report["mimic"] = {
        "jobId": mimic_job,
        "elapsed_sec": round(mimic_poll["elapsed"], 1),
        "status": {k: mimic_status.get(k) for k in ("phase", "phaseLabel", "progressMessage", "phaseTimings", "resolvedDevice", "status")},
        "live_status": {k: live_status.get(k) for k in ("phase", "phaseLabel", "progressMessage", "phaseTimings")},
        "phase_transitions": mimic_poll["phases"],
        "platform_run_grep": grep_file(mimic_dir / "logs/platform_run.log", ["Environment device", "device", "cuda", "cpu", "annotating", "generating"]),
    }

    # --- Cable threading polling simulation ---
    print("\n=== Starting cable_threading job + polling sim ===", flush=True)
    resp = session.post(
        f"{API}/workspace/cable-threading/generate-async",
        json={
            "episodes": 1,
            "seed": 0,
            "horizon": 150,
            "robot": "Panda",
            "cableModel": "default",
            "difficulty": "easy",
            "saveHdf5": False,
            "outputFormat": "npz",
            "saveProcessVideo": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    ct_job = resp.json()["jobId"]
    print(f"ct jobId={ct_job}", flush=True)

    status_times: list[float] = []
    frame_times: list[float] = []
    log_times: list[float] = []
    post_complete_status: list[float] = []
    t_start = time.time()
    last_status = last_frame = last_log = 0.0
    terminal_at: float | None = None

    while time.time() - t_start < 300:
        now = time.time() - t_start
        st = session.get(f"{API}/workspace/cable-threading/jobs/{ct_job}/status", timeout=15).json()
        job_status = st.get("status", "running")

        if now - last_status >= 1.0:
            status_times.append(now)
            last_status = now

        if now - last_frame >= 1.2:
            session.get(f"{API}/workspace/cable-threading/jobs/{ct_job}/frame?t={int(now*1000)}", timeout=15)
            frame_times.append(now)
            last_frame = now

        if now - last_log >= 3.0:
            session.get(f"{API}/workspace/cable-threading/jobs/{ct_job}/log", timeout=15)
            log_times.append(now)
            last_log = now

        if job_status in ("completed", "failed", "canceled", "cancelled", "timeout"):
            if terminal_at is None:
                terminal_at = now
                print(f"  terminal at {now:.1f}s status={job_status}", flush=True)
            elif now - terminal_at >= 5.0:
                break
        time.sleep(0.1)

    # After terminal, simulate 5s more WITHOUT terminal check (old bug) vs WITH (new code stops client-side)
    if terminal_at is not None:
        for _ in range(5):
            time.sleep(1.0)
            post_complete_status.append(time.time() - t_start)

    def intervals(times: list[float]) -> list[float]:
        return [round(times[i] - times[i - 1], 3) for i in range(1, min(len(times), 12))]

    report["cable_threading"] = {
        "jobId": ct_job,
        "status_intervals_sec": intervals(status_times),
        "frame_intervals_sec": intervals(frame_times),
        "log_intervals_sec": intervals(log_times),
        "status_count_30s": len([t for t in status_times if t < 30]),
        "frame_count_30s": len([t for t in frame_times if t < 30]),
        "terminal_at_sec": terminal_at,
        "post_terminal_poll_sim_sec": post_complete_status,
    }

    out_path = PROJECT / "runs/p0_acceptance_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
