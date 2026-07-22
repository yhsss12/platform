#!/usr/bin/env python3
"""Run scripted-expert generation jobs and collect quality artifacts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from app.services.isaac_lab import generate_service as gen_svc
from app.services.isaac_lab.isaac_job_utils import read_json
from app.services.isaac_lab.job_paths import (
    isaac_job_artifacts_dir,
    isaac_job_dataset_path,
    isaac_job_generation_manifest_path,
    isaac_job_preview_video_path,
    isaac_job_status_path,
)


def wait_job(job_id: str, *, poll_sec: float = 5.0, timeout_sec: int = 7200) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_phase = None
    while time.time() < deadline:
        status = read_json(isaac_job_status_path(job_id)) or {"jobId": job_id, "status": "unknown"}
        phase = status.get("phase")
        if phase != last_phase:
            print(f"[{job_id}] status={status.get('status')} phase={phase} msg={status.get('message')}", flush=True)
            last_phase = phase
        if status.get("status") in {"completed", "failed"}:
            return status
        time.sleep(poll_sec)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_sec}s")


def collect_job_summary(job_id: str) -> dict[str, Any]:
    artifacts = isaac_job_artifacts_dir(job_id)
    metrics_path = artifacts / "scripted_expert_metrics.json"
    quality_path = artifacts / "trajectory_quality_report.json"
    manifest_path = isaac_job_generation_manifest_path(job_id)
    status = read_json(isaac_job_status_path(job_id)) or {}

    summary: dict[str, Any] = {
        "jobId": job_id,
        "datasetId": status.get("datasetId"),
        "generationMode": "scripted_expert",
        "status": status.get("status"),
        "message": status.get("message"),
        "datasetHdf5": str(isaac_job_dataset_path(job_id)),
        "previewMp4": str(isaac_job_preview_video_path(job_id)),
        "scriptedExpertMetrics": str(metrics_path),
        "trajectoryQualityReport": str(quality_path),
        "generationManifest": str(manifest_path),
    }

    if metrics_path.is_file():
        summary["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if quality_path.is_file():
        summary["trajectoryQuality"] = json.loads(quality_path.read_text(encoding="utf-8"))
    if manifest_path.is_file():
        summary["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return summary


def run_one(num_demos: int, *, seed: int = 0) -> dict[str, Any]:
    started = gen_svc.start_generate_dataset(
        dataset_name=f"scripted_expert_smoke_{num_demos}",
        num_demos=num_demos,
        seed=seed,
        generation_mode="scripted_expert",
        headless=True,
        enable_cameras=True,
        video=True,
        max_attempts=0,
    )
    job_id = str(started["jobId"])
    print(f"started job {job_id} num_demos={num_demos}", flush=True)
    final = wait_job(job_id)
    summary = collect_job_summary(job_id)
    summary["finalStatus"] = final
    return summary


def main() -> int:
    demos_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 3]
    results = []
    for num_demos in demos_list:
        results.append(run_one(num_demos))
        time.sleep(2.0)
    out = Path(__file__).resolve().parents[3] / "runs" / "isaac_lab" / "scripted_expert_smoke_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    for item in results:
        print(json.dumps({k: item.get(k) for k in ("jobId", "datasetId", "status", "datasetHdf5", "previewMp4")}, ensure_ascii=False))
    failed = [r for r in results if r.get("status") != "completed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
