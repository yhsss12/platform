#!/usr/bin/env python3
"""P8: PINN candidate construction + repair acceptance orchestrator."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py

REPO = Path(__file__).resolve().parents[3]
INTEGRATION = REPO / "integrations" / "NutAssemblyMimicGen"
PYTHON = Path("/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python")
RUN_PY = INTEGRATION / "run.py"
JOBS_ROOT = REPO / "runs/nut_assembly/jobs"
REPORT_PATH = REPO / "runs/nut_assembly/debug/p8_pinn_candidate_and_repair_report.md"

if str(REPO / "backend") not in sys.path:
    sys.path.insert(0, str(REPO / "backend"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _make_job_id() -> str:
    from app.services.nut_assembly_service import make_job_id

    return make_job_id(prefix="na_gen_p8_pinn")


def _launch(job_id: str) -> int:
    job_root = JOBS_ROOT / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    physics_cfg = {
        "enabled": True,
        "method": "pinn_repair",
        "modelId": "nut_assembly_pinn_v1",
        "repairStages": ["align_over_peg", "descend_insert"],
        "candidateSource": [
            "high_error_generated_demos",
            "synthetic_perturbation_candidates",
        ],
        "maxCandidates": 5,
        "maxRepairAttemptsPerCandidate": 2,
        "xyErrorThreshold": 0.025,
        "heightErrorThreshold": 0.02,
        "validationMode": "mujoco_rollout",
        "appendRepairedDemos": True,
    }
    cfg_path = job_root / "configs" / "physics_enhancement.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(physics_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

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
        "nut_assembly_p8_pinn",
        "--physics-enhancement-config",
        str(cfg_path),
    ]
    print(f"[p8] starting {job_id}")
    proc = subprocess.run(cmd, cwd=str(REPO), check=False)
    return proc.returncode


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
            src = str(data[key].attrs.get("demo_source", "") or "")
            if src == "pinn_repaired":
                count += 1
    return count


def _list_repair_artifacts(job_root: Path) -> dict[str, Any]:
    repair = job_root / "repair"
    candidates = list((repair / "candidates").glob("*")) if (repair / "candidates").is_dir() else []
    repaired = list((repair / "repaired").glob("*")) if (repair / "repaired").is_dir() else []
    validation = list((repair / "validation").glob("*")) if (repair / "validation").is_dir() else []
    return {
        "candidates": len(candidates),
        "repaired": len(repaired),
        "validation": len(validation),
        "pinn_repair_summary": (repair / "pinn_repair_summary.json").is_file(),
        "pinn_repair_log": (job_root / "logs" / "pinn_repair.log").is_file(),
    }


def _write_report(job_id: str, summary: dict[str, Any], artifacts: dict[str, Any], repaired_in_hdf5: int) -> None:
    pinn = _read_json(JOBS_ROOT / job_id / "repair" / "pinn_repair_summary.json")
    merged = {**pinn, **summary}
    no_repair_reason = []
    if int(merged.get("pinnCandidateCount") or 0) <= 0:
        no_repair_reason.append("未筛选到候选轨迹（enhancementStatus=completed_no_candidates）")
    elif int(merged.get("pinnRepairAttempted") or 0) <= 0:
        no_repair_reason.append("有候选但未发起修复尝试")
    elif int(merged.get("pinnValidationSucceeded") or 0) <= 0:
        no_repair_reason.append("修复尝试均未通过 MuJoCo 复核（enhancementStatus=completed_no_repair_success）")
    if repaired_in_hdf5 <= 0 and not no_repair_reason:
        no_repair_reason.append("summary 有 validation 但 HDF5 未写入 pinn_repaired demo")

    lines = [
        "# P8 PINN 候选构造与修复验收报告",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        "## 任务摘要",
        "",
        f"| 字段 | 值 |",
        f"|------|-----|",
        f"| jobId | `{job_id}` |",
        f"| rawDemoCount | {merged.get('rawDemoCount', '—')} |",
        f"| pinnBackend | {merged.get('pinnBackend', '—')} |",
        f"| modelLoaded | {merged.get('modelLoaded', '—')} |",
        f"| modelPath | {merged.get('modelPath') or 'null'} |",
        f"| candidateMode | {merged.get('candidateMode', '—')} |",
        f"| pinnCandidateCount | {merged.get('pinnCandidateCount', '—')} |",
        f"| pinnRepairAttempted | {merged.get('pinnRepairAttempted', '—')} |",
        f"| pinnRepairSucceeded | {merged.get('pinnRepairSucceeded', '—')} |",
        f"| pinnValidationSucceeded | {merged.get('pinnValidationSucceeded', '—')} |",
        f"| repairedDemoCount | {merged.get('repairedDemoCount', '—')} |",
        f"| finalDemoCount | {merged.get('finalDemoCount', '—')} |",
        f"| enhancementStatus | {merged.get('enhancementStatus', '—')} |",
        f"| HDF5 demo_source=pinn_repaired | {repaired_in_hdf5} |",
        "",
        "## repair/ 产物",
        "",
        f"- candidates 文件数: {artifacts.get('candidates')}",
        f"- repaired 文件数: {artifacts.get('repaired')}",
        f"- validation 文件数: {artifacts.get('validation')}",
        f"- pinn_repair_summary.json: {'✅' if artifacts.get('pinn_repair_summary') else '❌'}",
        f"- pinn_repair.log: {'✅' if artifacts.get('pinn_repair_log') else '❌'}",
        "",
        "## 验收结论",
        "",
    ]

    checks = [
        ("MimicGen 正常生成", int(merged.get("rawDemoCount") or 0) > 0),
        ("PINN 阶段执行", artifacts.get("pinn_repair_log")),
        ("pinnCandidateCount > 0", int(merged.get("pinnCandidateCount") or 0) > 0),
        ("pinnRepairAttempted > 0", int(merged.get("pinnRepairAttempted") or 0) > 0),
        ("repair/candidates 存在", artifacts.get("candidates", 0) > 0),
        ("repair/validation 存在", artifacts.get("validation", 0) > 0),
        ("generation_summary 字段完整", bool(summary.get("physicsEnhancementEnabled"))),
        ("pinnBackend 诚实标注", merged.get("pinnBackend") in {"heuristic", "torch_model"}),
    ]
    if int(merged.get("pinnValidationSucceeded") or 0) > 0:
        checks.append(("HDF5 含 pinn_repaired demo", repaired_in_hdf5 > 0))

    for label, ok in checks:
        lines.append(f"- {'✅' if ok else '❌'} {label}")

    lines.extend(
        [
            "",
            "## 未产生 repaired demo 的原因（如适用）",
            "",
            "; ".join(no_repair_reason) if no_repair_reason else "已产生 repaired demo 或尚未终态。",
            "",
            "## 下一步",
            "",
            "- 部署真实 `<EAI_DATA_ROOT>/assets/models/pinn/nut_assembly_pinn_v1/model.pt` 后可切换 pinnBackend=torch_model",
            "- 提升 MimicGen demo 质量评分覆盖率，增加 high_error 自然候选比例",
            "- 线缆穿杆链路未修改，预期无影响",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[P8] report written: {REPORT_PATH}")


def main() -> int:
    job_id = _make_job_id()
    rc = _launch(job_id)
    if rc != 0:
        print(f"[P8] worker exited {rc}, still waiting for artifacts...")
    summary = _wait_job(job_id)
    job_root = JOBS_ROOT / job_id
    artifacts = _list_repair_artifacts(job_root)
    repaired_in_hdf5 = _count_pinn_repaired_demos(job_root / "datasets" / "nut_assembly_generated.hdf5")
    _write_report(job_id, summary, artifacts, repaired_in_hdf5)
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
