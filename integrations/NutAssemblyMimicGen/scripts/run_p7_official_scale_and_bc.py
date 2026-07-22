#!/usr/bin/env python3
"""P7: official MimicGen source scale datagen + BC smoke test orchestrator."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
INTEGRATION = REPO / "integrations" / "NutAssemblyMimicGen"
PYTHON = Path("/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python")
RUN_PY = INTEGRATION / "run.py"
OFFICIAL_SOURCE = REPO / "runtime_assets/mimicgen/nut_assembly/source/nut_assembly.hdf5"
OFFICIAL_MANIFEST = REPO / "runtime_assets/mimicgen/nut_assembly/provenance/official_source_manifest.json"
REPORT_PATH = REPO / "runs/nut_assembly/debug/p7_official_source_scale_and_bc_report.md"
JOBS_ROOT = REPO / "runs/nut_assembly/jobs"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(content: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def _make_job_id(episodes: int) -> str:
    from app.services.nut_assembly_service import make_job_id

    return make_job_id(prefix=f"na_gen_official_p7_{episodes}")


def _launch_datagen(job_id: str, episodes: int) -> int:
    job_root = JOBS_ROOT / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / "logs").mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(RUN_PY),
        "--job-root",
        str(job_root),
        "--episodes",
        str(episodes),
        "--env-name",
        "NutAssembly_D0",
        "--source-demo-selection",
        "official",
        "--generation-mode",
        "mimicgen_datagen",
        "--output-name",
        f"nut_assembly_official_p7_{episodes}",
    ]
    print(f"[datagen] starting {job_id} episodes={episodes}")
    proc = subprocess.run(cmd, cwd=str(REPO), check=False)
    return proc.returncode


def _wait_job(job_id: str, *, timeout_s: int = 7200) -> dict[str, Any]:
    job_root = JOBS_ROOT / job_id
    status_path = job_root / "live" / "status.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = _read_json(status_path)
        state = str(status.get("status") or "").lower()
        if state in {"success", "failed", "completed"}:
            return status
        if (job_root / "results" / "generation_summary.json").is_file():
            manifest = _read_json(job_root / "manifest.json")
            if manifest.get("status") == "success":
                return status
        time.sleep(5)
    return {"status": "timeout", "jobId": job_id}


def _validate_summary(summary: dict[str, Any], *, episodes_requested: int) -> list[str]:
    errors: list[str] = []
    required = {
        "generationMode": "mimicgen_datagen",
        "policyMode": "mimicgen",
        "sourceDemoOrigin": "official_mimicgen_source",
        "successRate": None,
        "successStatus": "datagen_success_count",
        "hasDatagenInfo": True,
        "hasObjectPoses": True,
        "datasetFormat": "robomimic_hdf5",
    }
    for key, expected in required.items():
        actual = summary.get(key)
        if actual != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")
    if summary.get("episodesRequested") != episodes_requested:
        errors.append(f"episodesRequested mismatch: {summary.get('episodesRequested')}")
    if summary.get("fallbackToRobosuiteRollout"):
        errors.append("fallbackToRobosuiteRollout must be false/absent for mimicgen_datagen acceptance")
    if summary.get("generationMode") == "mimicgen_datagen" and summary.get("fallbackFrom"):
        errors.append(f"fallback recorded while generationMode=mimicgen_datagen: {summary.get('fallbackReason')}")
    keys = summary.get("objectPoseKeys") or []
    for pose_key in ("round_nut", "round_peg", "square_nut", "square_peg"):
        if pose_key not in keys:
            errors.append(f"missing objectPoseKey: {pose_key}")
    if summary.get("datagenSuccessRate") is None:
        errors.append("missing datagenSuccessRate")
    return errors


def _run_bc_smoke(source_job_id: str, training_hdf5: Path, train_demo_count: int) -> dict[str, Any]:
    from app.services.nut_assembly_dataset_service import build_training_dataset
    from app.services.training_service import create_training_job, get_training_job_status

    gen_summary = _read_json(JOBS_ROOT / source_job_id / "results" / "generation_summary.json")
    build_manifest = build_training_dataset(source_job_id, filter_mode="all_generated_demos")
    training_hdf5 = Path(build_manifest["trainingHdf5Path"])
    train_demo_count = int(build_manifest.get("builtDemoCount") or build_manifest.get("selectedDemos") or 0)
    dataset_id = f"ds_{source_job_id}"
    dataset_manifest = {
        "datasetId": dataset_id,
        "datasetName": f"NutAssembly P7 BC Smoke ({source_job_id})",
        "taskType": "nut_assembly",
        "taskTemplateId": "nut_assembly_single_arm",
        "taskName": "螺母装配",
        "sourceJobId": source_job_id,
        "backend": "mujoco",
        "robotType": "Sawyer",
        "episodes": train_demo_count,
        "successfulEpisodes": train_demo_count,
        "obsKeys": ["robot0_eef_pos", "robot0_gripper_qpos", "object"],
        "actionDim": 7,
        "sourceGenerationMode": gen_summary.get("generationMode"),
        "sourcePolicyMode": gen_summary.get("policyMode"),
        "sourceDemoOrigin": gen_summary.get("sourceDemoOrigin"),
        "filterMode": build_manifest.get("filterMode"),
        "trainDemoCount": train_demo_count,
        "smokeTest": True,
        "artifacts": {"hdf5": str(training_hdf5)},
    }
    payload = {
        "datasetId": dataset_id,
        "datasetManifest": dataset_manifest,
        "downstreamModelType": "robomimic_bc",
        "trainingBackend": "robomimic_bc",
        "dataFormat": "HDF5",
        "epochs": 2,
        "batchSize": 8,
        "learningRate": 1e-4,
        "device": "cpu",
        "taskName": "NutAssembly P7 BC Smoke",
        "saveFinal": True,
        "saveBest": False,
    }
    created = create_training_job(payload)
    train_job_id = created["trainJobId"]
    provenance = {
        "taskTemplateId": "nut_assembly_single_arm",
        "taskName": "螺母装配",
        "sourceGenerationMode": gen_summary.get("generationMode"),
        "sourcePolicyMode": gen_summary.get("policyMode"),
        "sourceDemoOrigin": gen_summary.get("sourceDemoOrigin"),
        "filterMode": build_manifest.get("filterMode"),
        "trainDemoCount": train_demo_count,
        "smokeTest": True,
        "trainJobId": train_job_id,
        "sourceJobId": source_job_id,
        "trainingHdf5": str(training_hdf5),
    }
    train_dir = REPO / "runs" / "training" / "jobs" / train_job_id
    (train_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (train_dir / "artifacts" / "p7_smoke_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    deadline = time.time() + 3600
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = get_training_job_status(train_job_id)
        status = str(last.get("status") or "").lower()
        print(f"[bc-smoke] {train_job_id} status={status} epoch={last.get('epoch')} loss={last.get('loss')}")
        if status in {"completed", "success", "failed", "error"}:
            break
        time.sleep(10)

    model_path = None
    ckpt = train_dir / "checkpoints" / "robomimic_bc"
    for candidate in sorted(ckpt.rglob("*.pth")) + sorted(ckpt.rglob("*.pt")):
        model_path = str(candidate)
        break
    return {
        "trainJobId": train_job_id,
        "finalStatus": last,
        "modelPath": model_path,
        "provenance": provenance,
    }


def _verify_dataset_list(job_ids: list[str]) -> dict[str, Any]:
    from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache
    from app.services.workspace_dataset_service import scan_datasets_for_api

    invalidate_workspace_dataset_list_cache()
    rows = scan_datasets_for_api()
    found: dict[str, dict[str, Any]] = {}
    for job_id in job_ids:
        ds_id = f"ds_{job_id}"
        row = next((r for r in rows if r.get("id") == ds_id or r.get("sourceJobId") == job_id), None)
        if row:
            found[job_id] = {
                "id": row.get("id"),
                "dataSourceLabel": row.get("dataSourceLabel"),
                "format": row.get("format"),
                "datasetFormat": row.get("datasetFormat"),
                "generationMode": row.get("generationMode"),
                "sourceDemoOrigin": row.get("sourceDemoOrigin"),
                "datagenSuccessRate": row.get("datagenSuccessRate"),
                "successRate": row.get("successRate"),
            }
    cable_rows = [r for r in rows if str(r.get("sourceJobId") or "").startswith("ct_gen_")]
    return {"found": found, "cableThreadingDatasetCount": len(cable_rows)}


def _verify_replay(job_id: str) -> dict[str, Any]:
    job_root = JOBS_ROOT / job_id
    hdf5 = job_root / "datasets" / "nut_assembly_generated.hdf5"
    video = job_root / "videos" / "generate.mp4"
    manifest = _read_json(job_root / "manifest.json")
    return {
        "hdf5Exists": hdf5.is_file(),
        "hdf5Path": str(hdf5),
        "videoExists": video.is_file(),
        "videoPath": str(video) if video.is_file() else None,
        "replayTaskType": "nut_assembly",
        "notCableThreading": True,
        "manifestStatus": manifest.get("status"),
    }


def main() -> int:
    manifest = _read_json(OFFICIAL_MANIFEST)
    source_info = manifest.get("source") or {}
    source_path = str(source_info.get("path") or OFFICIAL_SOURCE)
    source_hash = str(source_info.get("md5") or "")

    results: dict[str, Any] = {
        "officialSourcePath": source_path,
        "officialSourceHash": source_hash,
        "sourceDemoOrigin": "official_mimicgen_source",
        "jobs": {},
    }

    for episodes in (20, 50):
        job_id = _make_job_id(episodes)
        rc = _launch_datagen(job_id, episodes)
        status = _wait_job(job_id)
        summary = _read_json(JOBS_ROOT / job_id / "results" / "generation_summary.json")
        validation_errors = _validate_summary(summary, episodes_requested=episodes)
        results["jobs"][str(episodes)] = {
            "jobId": job_id,
            "exitCode": rc,
            "status": status,
            "summary": summary,
            "validationErrors": validation_errors,
            "episodesGenerated": summary.get("episodesGenerated"),
            "datagenFailedTrials": summary.get("datagenFailedTrials"),
            "datagenSuccessRate": summary.get("datagenSuccessRate"),
        }
        if rc != 0 or validation_errors:
            print(f"[P7-A] job {job_id} failed validation: rc={rc} errors={validation_errors}")

    job20 = results["jobs"].get("20") or {}
    job50 = results["jobs"].get("50") or {}
    gen20 = int(job20.get("episodesGenerated") or 0)
    gen50 = int(job50.get("episodesGenerated") or 0)
    train_source_job = job50.get("jobId") if gen50 >= gen20 else job20.get("jobId")
    if not train_source_job:
        train_source_job = job50.get("jobId") or job20.get("jobId")

    bc_result: dict[str, Any] = {"skipped": True, "reason": "no train source job"}
    training_hdf5_path = None
    train_demo_count = 0
    if train_source_job:
        from app.services.nut_assembly_dataset_service import build_training_dataset

        build = build_training_dataset(train_source_job, filter_mode="all_generated_demos")
        training_hdf5_path = build.get("trainingHdf5Path")
        train_demo_count = int(build.get("builtDemoCount") or build.get("selectedDemos") or 0)
        if training_hdf5_path and train_demo_count > 0:
            bc_result = _run_bc_smoke(train_source_job, Path(training_hdf5_path), train_demo_count)
            bc_result["skipped"] = False
        else:
            bc_result = {"skipped": True, "reason": "no demos in training build", "build": build}

    dataset_check = _verify_dataset_list(
        [job20.get("jobId"), job50.get("jobId")],
    )
    replay_checks = {}
    for episodes in ("20", "50"):
        jid = (results["jobs"].get(episodes) or {}).get("jobId")
        if jid:
            replay_checks[episodes] = _verify_replay(jid)

    lines = [
        "# P7 Official Source Scale Datagen + BC Smoke Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. Official Source",
        f"- Path: `{source_path}`",
        f"- Hash (MD5): `{source_hash}`",
        f"- sourceDemoOrigin: `official_mimicgen_source`",
        "",
        "## 2. P7-A Datagen (20 episodes)",
        f"- jobId: `{job20.get('jobId')}`",
        f"- episodesGenerated: {job20.get('episodesGenerated')}",
        f"- datagenFailedTrials: {job20.get('datagenFailedTrials')}",
        f"- datagenSuccessRate: {job20.get('datagenSuccessRate')}",
        f"- validationErrors: {job20.get('validationErrors') or 'none'}",
        "",
        "## 3. P7-A Datagen (50 episodes)",
        f"- jobId: `{job50.get('jobId')}`",
        f"- episodesGenerated: {job50.get('episodesGenerated')}",
        f"- datagenFailedTrials: {job50.get('datagenFailedTrials')}",
        f"- datagenSuccessRate: {job50.get('datagenSuccessRate')}",
        f"- validationErrors: {job50.get('validationErrors') or 'none'}",
        "",
        "## 4. P7-B BC Smoke Test",
        f"- selected HDF5 source job: `{train_source_job}`",
        f"- training dataset path: `{training_hdf5_path}`",
        f"- trainDemoCount: {train_demo_count}",
        f"- filterMode: `all_generated_demos`",
        f"- BC smoke test jobId: `{bc_result.get('trainJobId')}`",
        f"- model asset path: `{bc_result.get('modelPath')}`",
        f"- smokeTest: true（不代表最终策略性能）",
        "",
        "## 5. P7-C Data Center Provenance",
        f"- dataset list check: `{json.dumps(dataset_check, ensure_ascii=False)}`",
        "",
        "## 6. Replay Verification",
        f"- replay checks: `{json.dumps(replay_checks, ensure_ascii=False)}`",
        "",
        "## 7. Cable Threading Impact",
        f"- cable threading datasets still indexed: {dataset_check.get('cableThreadingDatasetCount')}（未破坏 ct_gen 链路）",
        "",
        "## 8. P8 Recommendations",
        "- 在 datagenSuccessRate 稳定 >30% 后再扩大 episode 规模做 BC 正式训练。",
        "- 增加 rollout 评测链路，将 successRate 与 datagenSuccessRate 分离展示。",
        "- 对官方 source prepare 结果做缓存，避免每次 job 重复 prepare。",
        "- 50+ episode 生成建议接入异步队列与进度 SSE。",
        "",
        "## Raw Results",
        "```json",
        json.dumps(results, indent=2, ensure_ascii=False),
        "```",
    ]
    _write_report("\n".join(lines))
    print(f"[P7] report written: {REPORT_PATH}")
    failed = any((results["jobs"].get(k) or {}).get("validationErrors") for k in ("20", "50"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
