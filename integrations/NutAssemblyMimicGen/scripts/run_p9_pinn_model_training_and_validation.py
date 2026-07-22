#!/usr/bin/env python3
"""P9: PINN training, torch_model validation, baseline comparison, and report."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py

REPO = Path(__file__).resolve().parents[3]
INTEGRATION = REPO / "integrations" / "NutAssemblyMimicGen"
DATA_ROOT = Path(os.environ.get("EAI_DATA_ROOT") or (REPO / "eai-data")).expanduser()
PYTHON = Path("/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python")
RUN_PY = INTEGRATION / "run.py"
JOBS_ROOT = DATA_ROOT / "runs/nut_assembly/jobs"
BUILD_SCRIPT = INTEGRATION / "scripts/build_pinn_repair_training_dataset.py"
TRAIN_SCRIPT = INTEGRATION / "scripts/train_pinn_repair_model.py"
TRAINING_DIR = DATA_ROOT / "runs/nut_assembly/pinn_training"
MODEL_DIR = DATA_ROOT / "assets/models/pinn/nut_assembly_pinn_v1"
REPORT_PATH = DATA_ROOT / "runs/nut_assembly/debug/p9_pinn_model_training_and_validation_report.md"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _make_job_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{secrets.token_hex(2)}"


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print(f"[p9] run: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd or REPO), check=False).returncode


def _build_training_data() -> dict[str, Any]:
    rc = _run([str(PYTHON), str(BUILD_SCRIPT)])
    manifest = _read_json(TRAINING_DIR / "repair_training_manifest.json")
    manifest["buildRc"] = rc
    return manifest


def _train_model() -> dict[str, Any]:
    rc = _run([str(PYTHON), str(TRAIN_SCRIPT), "--epochs", "150"])
    train_log = _read_json(MODEL_DIR / "train_log.json")
    eval_report = _read_json(MODEL_DIR / "eval_report.json")
    metadata = _read_json(MODEL_DIR / "metadata.json")
    return {
        "trainRc": rc,
        "trainLog": train_log,
        "evalReport": eval_report,
        "metadata": metadata,
        "modelPath": str(MODEL_DIR / "model.pt"),
        "modelExists": (MODEL_DIR / "model.pt").is_file(),
    }


def _launch_generation_job(
    *,
    job_id: str,
    physics_cfg: dict[str, Any] | None,
    prefix: str,
) -> str:
    job_root = JOBS_ROOT / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    cfg_path = job_root / "configs" / "physics_enhancement.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(physics_cfg or {"enabled": False}, indent=2, ensure_ascii=False), encoding="utf-8")

    cmd = [
        str(PYTHON),
        str(RUN_PY),
        "--job-root",
        str(job_root),
        "--episodes",
        "20",
        "--env-name",
        "NutAssembly_D0",
        "--source-demo-selection",
        "official",
        "--generation-mode",
        "mimicgen_datagen",
        "--output-name",
        prefix,
        "--physics-enhancement-config",
        str(cfg_path),
    ]
    print(f"[p9] starting {job_id} mode={prefix}")
    proc = subprocess.run(cmd, cwd=str(REPO), check=False)
    if proc.returncode != 0:
        print(f"[p9] worker exited {proc.returncode} for {job_id}")
    return job_id


def _wait_job(job_id: str, *, timeout_s: int = 7200) -> dict[str, Any]:
    job_root = JOBS_ROOT / job_id
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        summary = _read_json(job_root / "results" / "generation_summary.json")
        status = _read_json(job_root / "live" / "status.json")
        state = str(status.get("status") or summary.get("status") or "").lower()
        if summary and state in {"success", "failed", "completed", "partial_success"}:
            return summary
        if summary and summary.get("completedAt"):
            return summary
        time.sleep(5)
    return _read_json(job_root / "results" / "generation_summary.json")


def _count_pinn_repaired_demos(hdf5_path: Path) -> int:
    if not hdf5_path.is_file():
        return 0
    count = 0
    with h5py.File(hdf5_path, "r") as f:
        data = f.get("data")
        if data is None:
            return 0
        for key in data.keys():
            if not str(key).startswith("demo_"):
                continue
            if str(data[key].attrs.get("demo_source", "") or "") == "pinn_repaired":
                count += 1
    return count


def _collect_job_metrics(job_id: str) -> dict[str, Any]:
    job_root = JOBS_ROOT / job_id
    summary = _read_json(job_root / "results" / "generation_summary.json")
    pinn = _read_json(job_root / "repair" / "pinn_repair_summary.json")
    merged = {**pinn, **summary}
    hdf5 = job_root / "datasets" / "nut_assembly_generated.hdf5"
    return {
        "jobId": job_id,
        "pinnCandidateCount": merged.get("pinnCandidateCount") or merged.get("candidateCount"),
        "pinnRepairAttempted": merged.get("pinnRepairAttempted") or merged.get("repairAttempted"),
        "pinnValidationSucceeded": merged.get("pinnValidationSucceeded") or merged.get("validationSucceeded"),
        "repairedDemoCount": merged.get("repairedDemoCount"),
        "rawDemoCount": merged.get("rawDemoCount"),
        "finalDemoCount": merged.get("finalDemoCount") or merged.get("demoCount"),
        "pinnBackend": merged.get("pinnBackend"),
        "modelLoaded": merged.get("modelLoaded"),
        "modelPath": merged.get("modelPath"),
        "enhancementStatus": merged.get("enhancementStatus"),
        "averageFinalXYError": merged.get("averageFinalXYError"),
        "averageFinalHeightError": merged.get("averageFinalHeightError"),
        "hdf5PinnRepaired": _count_pinn_repaired_demos(hdf5),
        "runtimeSeconds": merged.get("runtimeSeconds"),
    }


def _baseline_physics(mode: str) -> dict[str, Any]:
    base = {
        "enabled": True,
        "method": "pinn_repair",
        "modelId": "nut_assembly_pinn_v1",
        "repairStages": ["align_over_peg", "descend_insert"],
        "candidateSource": ["high_error_generated_demos", "synthetic_perturbation_candidates"],
        "maxCandidates": 10,
        "maxRepairAttemptsPerCandidate": 5,
        "xyErrorThreshold": 0.025,
        "heightErrorThreshold": 0.02,
        "validationMode": "mujoco_rollout",
        "appendRepairedDemos": True,
    }
    if mode == "heuristic":
        base["forcePinnBackend"] = "heuristic"
    elif mode == "torch_model":
        base["forcePinnBackend"] = "torch_model"
    return base


def _count_cable_threading_datasets() -> int:
    ct_root = REPO / "runs/cable_threading/datasets"
    if not ct_root.is_dir():
        return 0
    return len(list(ct_root.glob("**/*.hdf5")))


def _write_report(
    *,
    training_manifest: dict[str, Any],
    model_info: dict[str, Any],
    acceptance: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    cable_count: int,
) -> None:
    lines = [
        "# P9 PINN 模型训练与修复效果验证报告",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        "> PINN 修复通过率指 MuJoCo 复核后写入 HDF5 的 repaired demo 比例，**不是**任务 rollout 评测成功率。",
        "",
        "## 1. 训练数据",
        "",
        f"- 样本数: {training_manifest.get('sampleCount', '—')}",
        f"- 来源: {json.dumps(training_manifest.get('sources', []), ensure_ascii=False)}",
        "",
        "## 2. 模型",
        "",
        f"- 路径: `{model_info.get('modelPath')}`",
        f"- modelLoaded: {acceptance.get('modelLoaded')}",
        f"- metadata.backend: {model_info.get('metadata', {}).get('backend', '—')}",
        f"- valLoss: {model_info.get('evalReport', {}).get('valLoss', '—')}",
        "",
        "## 3. P9 验收 job (torch_model)",
        "",
        "| 字段 | 值 |",
        "|------|-----|",
        f"| jobId | `{acceptance.get('jobId', '—')}` |",
        f"| rawDemoCount | {acceptance.get('rawDemoCount', '—')} |",
        f"| pinnBackend | {acceptance.get('pinnBackend', '—')} |",
        f"| modelLoaded | {acceptance.get('modelLoaded', '—')} |",
        f"| pinnCandidateCount | {acceptance.get('pinnCandidateCount', '—')} |",
        f"| pinnRepairAttempted | {acceptance.get('pinnRepairAttempted', '—')} |",
        f"| pinnValidationSucceeded | {acceptance.get('pinnValidationSucceeded', '—')} |",
        f"| repairedDemoCount | {acceptance.get('repairedDemoCount', '—')} |",
        f"| finalDemoCount | {acceptance.get('finalDemoCount', '—')} |",
        f"| enhancementStatus | {acceptance.get('enhancementStatus', '—')} |",
        f"| HDF5 demo_source=pinn_repaired | {acceptance.get('hdf5PinnRepaired', '—')} |",
        "",
        "## 4. Baseline 对比（PINN 修复通过率，非任务成功率）",
        "",
        "| 模式 | candidates | attempted | validationSucceeded | repaired | final | backend | runtime(s) |",
        "|------|------------|-----------|---------------------|----------|-------|---------|------------|",
    ]

    for mode, metrics in baselines.items():
        lines.append(
            f"| {mode} | {metrics.get('pinnCandidateCount')} | {metrics.get('pinnRepairAttempted')} | "
            f"{metrics.get('pinnValidationSucceeded')} | {metrics.get('repairedDemoCount')} | "
            f"{metrics.get('finalDemoCount')} | {metrics.get('pinnBackend')} | {metrics.get('runtimeSeconds')} |"
        )

    lines.extend(
        [
            "",
            "## 5. HDF5 pinn_repaired",
            "",
            f"- 存在 demo_source=pinn_repaired: {bool(acceptance.get('hdf5PinnRepaired', 0))}",
            "",
            "## 6. 线缆穿杆",
            "",
            f"- ct_gen 数据集数量: {cable_count}",
            "- 未修改 CableThreadingMVP 链路",
            "",
            "## 7. P10 建议",
            "",
            "- 扩大训练集：纳入更多 MimicGen datagen 成功 job",
            "- 引入真实 robosuite 闭环 replay 微调 action delta",
            "- 分离 align / insert 两阶段独立 head",
            "- datagen 成功率提升后复验 torch_model 修复通过率",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[P9] report written: {REPORT_PATH}")


def main() -> int:
    t0 = time.time()
    training_manifest = _build_training_data()
    model_info = _train_model()

    baselines: dict[str, dict[str, Any]] = {}

    no_repair_id = _make_job_id(prefix="na_gen_p9_norepair")
    t_no = time.time()
    _launch_generation_job(job_id=no_repair_id, physics_cfg={"enabled": False}, prefix="nut_assembly_p9_norepair")
    _wait_job(no_repair_id)
    baselines["no_repair"] = _collect_job_metrics(no_repair_id)
    baselines["no_repair"]["runtimeSeconds"] = round(time.time() - t_no, 1)

    heuristic_id = _make_job_id(prefix="na_gen_p9_heur")
    t_heur = time.time()
    _launch_generation_job(
        job_id=heuristic_id,
        physics_cfg=_baseline_physics("heuristic"),
        prefix="nut_assembly_p9_heuristic",
    )
    _wait_job(heuristic_id)
    baselines["heuristic_repair"] = _collect_job_metrics(heuristic_id)
    baselines["heuristic_repair"]["runtimeSeconds"] = round(time.time() - t_heur, 1)

    torch_id = _make_job_id(prefix="na_gen_p9_torch")
    t_torch = time.time()
    _launch_generation_job(
        job_id=torch_id,
        physics_cfg=_baseline_physics("torch_model"),
        prefix="nut_assembly_p9_torch",
    )
    _wait_job(torch_id)
    acceptance = _collect_job_metrics(torch_id)
    acceptance["runtimeSeconds"] = round(time.time() - t_torch, 1)
    baselines["torch_model_pinn_repair"] = acceptance

    cable_count = _count_cable_threading_datasets()
    _write_report(
        training_manifest=training_manifest,
        model_info=model_info,
        acceptance=acceptance,
        baselines=baselines,
        cable_count=cable_count,
    )
    print(f"[P9] total runtime={time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
