#!/usr/bin/env python3
"""V1-F-100Base end-to-end pipeline launcher。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_V1F_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1F_DIR.parent.parent
_SCRIPT = sys.executable


def _run(label: str, script: str, *extra: str, skip: bool = False) -> None:
    if skip:
        print(f"[skip] {label}", flush=True)
        return
    cmd = [_SCRIPT, str(_V1F_DIR / script), *extra]
    print(f"[run] {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(_EXPERIMENT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch V1-F-100Base pipeline")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-contexts", action="store_true")
    parser.add_argument("--skip-success-ref", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-sanity", action="store_true", help="跳过 pre-train gate（不推荐）")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    py_path = ":".join(
        str(p)
        for p in (
            _EXPERIMENT_DIR,
            _EXPERIMENT_DIR / "v1_residual_model",
            _V1F_DIR,
            _EXPERIMENT_DIR / "offline_mimicgen_repair_test",
        )
    )
    os.environ["PYTHONPATH"] = py_path + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

    out = _EXPERIMENT_DIR / "outputs" / "v1f_100base"
    out.mkdir(parents=True, exist_ok=True)

    _run(
        "audit Square_D0 100 demos",
        "audit_new_square_d0_demos.py",
        "--output-dir",
        str(out / "audit"),
        skip=args.skip_audit,
    )
    _run(
        "build failed contexts (classified failure_type)",
        "build_v1f_plus_failed_contexts.py",
        "--audit-report",
        str(out / "audit" / "new_demo_audit_report.json"),
        "--output",
        str(out / "failed_contexts.jsonl"),
        skip=args.skip_contexts,
    )
    _run("success reference (77 demos)", "run_v1f_100base_success_reference.py", skip=args.skip_success_ref)
    _run("failed rollout phase1+2", "run_v1f_100base_rollout_sampling.py", skip=args.skip_rollout)
    _run("build 100base dataset", "build_v1f_100base_dataset.py", skip=args.skip_dataset)
    if not args.skip_sanity:
        _run("pre-train sanity gate", "run_v1f_100base_pretrain_sanity_gate.py", skip=False)
    else:
        print("[skip] pre-train sanity gate (--skip-sanity)", flush=True)
    _run(
        "train 100base model",
        "train_pinn_v1f_100base_model.py",
        "--epochs",
        str(args.epochs),
        skip=args.skip_train,
    )
    _run("quick validation", "run_v1f_100base_quick_validation.py", "--resume", skip=args.skip_eval)

    status = {
        "pipeline": "V1-F-100Base",
        "output_dir": str(out),
        "init_checkpoint": "outputs/v1f_aligned_repair_parameter_model/original_failed/trained_model/model_v1f_aligned_original.pt",
        "expected_outputs": {
            "dataset": str(out / "repair_parameter_dataset_v1f_100base.npz"),
            "sanity_json": str(out / "pretrain_sanity_report.json"),
            "sanity_md": str(out / "pretrain_sanity_summary.md"),
            "model": str(out / "trained_model" / "model_v1f_100base.pt"),
            "eval": str(out / "evaluation" / "quick_validation_report.json"),
        },
    }
    (out / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
