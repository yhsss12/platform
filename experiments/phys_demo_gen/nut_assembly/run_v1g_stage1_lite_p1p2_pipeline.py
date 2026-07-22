#!/usr/bin/env python3
"""V1-G-stage1-lite-p1p2 全流程：checkpoint 完整性 → fine-tune → 双模型对比验收。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_V1F_DIR = _EXPERIMENT_DIR / "v1_residual_model" / "repair_parameter_model_v1f"
_SCRIPT = sys.executable

DEFAULT_INIT = (
    _EXPERIMENT_DIR
    / "outputs"
    / "v1f_aligned_repair_parameter_model"
    / "original_failed"
    / "trained_model"
    / "model_v1f_aligned_original.pt"
)
DEFAULT_OUT_DIR = _EXPERIMENT_DIR / "outputs" / "v1g_stage1_lite_p1p2"
DEFAULT_CKPT = DEFAULT_OUT_DIR / "trained_model" / "model_v1g_stage1_lite_p1p2.pt"


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-G-stage1-lite-p1p2 pipeline")
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output-checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--demos", nargs="+", default=["demo_2", "demo_4"])
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-residual-val", action="store_true")
    parser.add_argument("--skip-rollout-val", action="store_true")
    parser.add_argument("--enable-physics-residual-repair", action="store_true")
    args = parser.parse_args()

    if not args.init_checkpoint.exists():
        raise SystemExit(f"Init checkpoint missing: {args.init_checkpoint}")

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

    if args.enable_physics_residual_repair:
        os.environ["enable_physics_residual_repair"] = "true"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    before_fp = _fingerprint(args.init_checkpoint)
    integrity: dict[str, Any] = {
        "aligned_original_checkpoint": str(args.init_checkpoint),
        "before_train": before_fp,
        "after_train": None,
        "aligned_original_unchanged": None,
        "v1g_lite_checkpoint": str(args.output_checkpoint),
    }

    if not args.skip_train:
        _run(
            [
                _SCRIPT,
                str(_V1F_DIR / "train_pinn_v1g_stage1_lite_p1p2_model.py"),
                "--init-checkpoint",
                str(args.init_checkpoint),
                "--output-checkpoint",
                str(args.output_checkpoint),
                "--output-dir",
                str(args.output_checkpoint.parent),
                "--epochs",
                str(args.epochs),
            ],
            cwd=_EXPERIMENT_DIR,
        )

    after_fp = _fingerprint(args.init_checkpoint)
    integrity["after_train"] = after_fp
    integrity["aligned_original_unchanged"] = (
        before_fp["sha256"] == after_fp["sha256"] and before_fp["mtime"] == after_fp["mtime"]
    )
    integrity_path = args.output_dir / "checkpoint_integrity.json"
    integrity_path.write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    if not args.output_checkpoint.exists() and not args.skip_train:
        raise SystemExit(f"Expected output checkpoint missing: {args.output_checkpoint}")

    repair_flag = ["--enable-physics-residual-repair"] if args.enable_physics_residual_repair else []
    if not repair_flag:
        os.environ["enable_physics_residual_repair"] = "true"
        repair_flag = ["--enable-physics-residual-repair"]

    if not args.skip_residual_val:
        _run(
            [
                _SCRIPT,
                str(_EXPERIMENT_DIR / "run_physics_residual_repair_validation.py"),
                *repair_flag,
                "--demos",
                *args.demos,
                "--v1f-model",
                str(args.init_checkpoint),
                "--output-json",
                str(args.output_dir / "residual_validation_aligned_original.json"),
                "--report-md",
                str(args.output_dir / "residual_validation_aligned_original.md"),
                "--num-samples",
                str(args.num_samples),
                "--top-k",
                str(args.top_k),
            ],
            cwd=_EXPERIMENT_DIR,
        )
        _run(
            [
                _SCRIPT,
                str(_EXPERIMENT_DIR / "run_physics_residual_repair_validation.py"),
                *repair_flag,
                "--demos",
                *args.demos,
                "--v1f-model",
                str(args.output_checkpoint),
                "--output-json",
                str(args.output_dir / "residual_validation_v1g_lite.json"),
                "--report-md",
                str(args.output_dir / "residual_validation_v1g_lite.md"),
                "--num-samples",
                str(args.num_samples),
                "--top-k",
                str(args.top_k),
            ],
            cwd=_EXPERIMENT_DIR,
        )

    if not args.skip_rollout_val:
        _run(
            [
                _SCRIPT,
                str(_EXPERIMENT_DIR / "run_v1g_lite_model_comparison.py"),
                *repair_flag,
                "--aligned-model",
                str(args.init_checkpoint),
                "--v1g-model",
                str(args.output_checkpoint),
                "--demos",
                *args.demos,
                "--num-samples",
                str(args.num_samples),
                "--top-k",
                str(args.top_k),
                "--seeds",
                *[str(s) for s in args.seeds],
                "--output-dir",
                str(args.output_dir),
                "--checkpoint-integrity",
                str(integrity_path),
            ],
            cwd=_EXPERIMENT_DIR,
        )

    comparison_path = args.output_dir / "model_comparison_report.json"
    acceptance: dict[str, Any] = {}
    if comparison_path.exists():
        acceptance = json.loads(comparison_path.read_text(encoding="utf-8")).get("acceptance", {})

    status = {
        "pipeline": "V1-G-stage1-lite-p1p2",
        "init_checkpoint": str(args.init_checkpoint),
        "output_checkpoint": str(args.output_checkpoint),
        "aligned_original_unchanged": integrity["aligned_original_unchanged"],
        "physics_residual_repair_opt_in": True,
        "validation_demos": args.demos,
        "excluded_demo_3": True,
        "acceptance": acceptance,
        "outputs": {
            "training_log": str(args.output_dir / "training_log.json"),
            "checkpoint_integrity": str(integrity_path),
            "residual_validation_aligned_original": str(args.output_dir / "residual_validation_aligned_original.json"),
            "residual_validation_v1g_lite": str(args.output_dir / "residual_validation_v1g_lite.json"),
            "rollout_validation_aligned_original": str(args.output_dir / "rollout_validation_aligned_original.json"),
            "rollout_validation_v1g_lite": str(args.output_dir / "rollout_validation_v1g_lite.json"),
            "model_comparison_report_json": str(comparison_path),
            "model_comparison_report_md": str(args.output_dir / "model_comparison_report.md"),
        },
    }
    (args.output_dir / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
