#!/usr/bin/env python3
"""V1-F-100Base-R1 pipeline：结构修复（build dataset + loss gate），默认跳过训练。"""
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
    parser = argparse.ArgumentParser(description="Launch V1-F-100Base-R1 structural pipeline")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-loss-gate", action="store_true")
    parser.add_argument("--run-train", action="store_true", help="显式启用训练（默认跳过）")
    parser.add_argument("--run-eval", action="store_true", help="显式启用 quick validation（需已训练模型）")
    parser.add_argument("--manifest-only", action="store_true", help="只生成 validation candidate manifest")
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

    out = _EXPERIMENT_DIR / "outputs" / "v1f_100base_r1"
    out.mkdir(parents=True, exist_ok=True)

    _run("build R1 dataset (demo_uid namespace)", "build_v1f_100base_r1_dataset.py", skip=args.skip_dataset)
    _run("pre-train loss contribution gate", "run_v1f_100base_r1_pretrain_loss_gate.py", skip=args.skip_loss_gate)

    if args.run_train:
        _run(
            "train 100Base-R1 model",
            "train_pinn_v1f_100base_r1_model.py",
            "--epochs",
            str(args.epochs),
        )
    else:
        _run(
            "train dry-run (structure check only)",
            "train_pinn_v1f_100base_r1_model.py",
            "--dry-run",
        )

    if args.run_eval:
        _run("quick validation", "run_v1f_100base_r1_quick_validation.py", "--resume", skip=False)
    elif args.manifest_only:
        _run("validation candidate manifest", "run_v1f_100base_r1_quick_validation.py", "--manifest-only", skip=False)
    else:
        print("[skip] quick validation (use --run-eval or --manifest-only)", flush=True)

    status = {
        "pipeline": "V1-F-100Base-R1",
        "output_dir": str(out),
        "training_skipped_by_default": not args.run_train,
        "init_checkpoint": "outputs/v1f_aligned_repair_parameter_model/original_failed/trained_model/model_v1f_aligned_original.pt",
        "expected_outputs": {
            "dataset": str(out / "repair_parameter_dataset_v1f_100base_r1.npz"),
            "loss_gate": str(out / "pretrain_loss_contribution_report.json"),
            "model": str(out / "trained_model" / "model_v1f_100base_r1.pt"),
            "candidate_manifest": str(out / "evaluation" / "validation_candidate_manifest.json"),
            "eval": str(out / "evaluation" / "quick_validation_report.json"),
        },
        "r1_fixes": [
            "demo_uid namespace isolation",
            "retention legacy_old demo_0-4 only",
            "success_reference excluded from focal/ranking",
            "pairwise ranking grouped by demo_uid",
            "pretrain loss contribution gate",
            "early stopping val_loss + old-demo ranking gate",
            "fixed validation candidate pool manifest",
        ],
    }
    (out / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
