#!/usr/bin/env python3
"""Run an explicit network-condition experiment matrix."""
from __future__ import annotations

import argparse
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List

import requests


NETEM_PROFILES: Dict[str, List[str]] = {
    "N0": [],
    "N1": ["delay", "50ms", "5ms"],
    "N2": ["delay", "80ms", "20ms", "loss", "1%"],
    "N3": ["delay", "120ms", "30ms", "loss", "2%", "rate", "8mbit"],
    "N4_10": ["delay", "120ms", "30ms", "loss", "10%"],
    "N4_20": ["delay", "120ms", "30ms", "loss", "20%"],
    "N5": ["delay", "80ms", "20ms", "loss", "1%", "rate", "8mbit"],
}


def api_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def apply_netem(iface: str, profile_name: str) -> None:
    profile = NETEM_PROFILES.get(profile_name, [])
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", iface, "root", "netem"], check=False)
    if profile:
        subprocess.run(["sudo", "tc", "qdisc", "add", "dev", iface, "root", "netem", *profile], check=True)


def clear_netem(iface: str) -> None:
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", iface, "root", "netem"], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment matrix controller")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--duration-sec", type=int, default=120)
    parser.add_argument("--iface", default="")
    parser.add_argument("--netem", default="N0")
    parser.add_argument("--sampler-interval-sec", type=float, default=1.0)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--device-id", default="")
    parser.add_argument("--frontend-url", default="")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    session = requests.Session()

    for method_name in args.method:
        for repeat in range(1, max(1, args.repeats) + 1):
            run_id = uuid.uuid4().hex
            print(f"\n[experiment] scenario={args.scenario} method={method_name} repeat={repeat}/{args.repeats} run_id={run_id}")

            session.put(
                f"{base_url}/api/experiment/method",
                headers=api_headers(args.token),
                json={"name": method_name},
                timeout=20,
            ).raise_for_status()

            session.post(
                f"{base_url}/api/experiment/event",
                headers=api_headers(args.token),
                json={
                    "role": "platform",
                    "event": "scenario_start",
                    "run_id": run_id,
                    "scenario_id": args.scenario,
                    "task_id": args.task_id or None,
                    "job_id": args.job_id or None,
                    "device_id": args.device_id or None,
                    "method": method_name,
                    "repeat_index": repeat,
                },
                timeout=20,
            ).raise_for_status()

            session.post(
                f"{base_url}/api/experiment/sample/start",
                headers=api_headers(args.token),
                json={
                    "run_id": run_id,
                    "scenario_id": args.scenario,
                    "task_id": args.task_id or None,
                    "job_id": args.job_id or None,
                    "device_id": args.device_id or None,
                    "method": method_name,
                    "interval_sec": args.sampler_interval_sec,
                },
                timeout=20,
            ).raise_for_status()

            if args.iface:
                apply_netem(args.iface, args.netem)
                session.post(
                    f"{base_url}/api/experiment/event",
                    headers=api_headers(args.token),
                    json={
                        "role": "platform",
                        "event": "netem_applied",
                        "run_id": run_id,
                        "scenario_id": args.scenario,
                        "method": method_name,
                        "iface": args.iface,
                        "netem_profile": args.netem,
                    },
                    timeout=20,
                ).raise_for_status()

            try:
                frontend_url = args.frontend_url.rstrip("/")
                if frontend_url:
                    print(
                        "[experiment] open realtime page:",
                        f"{frontend_url}/collect/realtime?taskId={args.task_id}&jobId={args.job_id}&scenario={args.scenario}&run={run_id}",
                    )
                time.sleep(max(1, args.duration_sec))
            finally:
                session.post(
                    f"{base_url}/api/experiment/sample/stop",
                    headers=api_headers(args.token),
                    json={"run_id": run_id},
                    timeout=20,
                ).raise_for_status()
                if args.iface:
                    clear_netem(args.iface)
                    session.post(
                        f"{base_url}/api/experiment/event",
                        headers=api_headers(args.token),
                        json={
                            "role": "platform",
                            "event": "netem_cleared",
                            "run_id": run_id,
                            "scenario_id": args.scenario,
                            "method": method_name,
                            "iface": args.iface,
                        },
                        timeout=20,
                    ).raise_for_status()
                session.post(
                    f"{base_url}/api/experiment/event",
                    headers=api_headers(args.token),
                    json={
                        "role": "platform",
                        "event": "scenario_end",
                        "run_id": run_id,
                        "scenario_id": args.scenario,
                        "task_id": args.task_id or None,
                        "job_id": args.job_id or None,
                        "device_id": args.device_id or None,
                        "method": method_name,
                        "repeat_index": repeat,
                    },
                    timeout=20,
                ).raise_for_status()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
