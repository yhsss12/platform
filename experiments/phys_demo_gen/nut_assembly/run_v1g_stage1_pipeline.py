#!/usr/bin/env python3
"""V1-G-stage1 全流程：训练 + demo_2/demo_4 residual/rollout 验证。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_V1F_DIR = _EXPERIMENT_DIR / "v1_residual_model" / "repair_parameter_model_v1f"
_SCRIPT = sys.executable


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-G-stage1 pipeline")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-residual-val", action="store_true")
    parser.add_argument("--skip-rollout-val", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
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

    out = _EXPERIMENT_DIR / "outputs" / "v1g_stage1_p1xy"
    model_path = out / "trained_model" / "model_v1g_stage1_p1xy.pt"

    if not args.skip_train:
        _run(
            [_SCRIPT, str(_V1F_DIR / "train_pinn_v1g_stage1_model.py"), "--epochs", str(args.epochs)],
            cwd=_EXPERIMENT_DIR,
        )

    os.environ["enable_physics_residual_repair"] = "true"

    if not args.skip_residual_val:
        _run(
            [
                _SCRIPT,
                str(_EXPERIMENT_DIR / "run_physics_residual_repair_validation.py"),
                "--enable-physics-residual-repair",
                "--demos",
                "demo_2",
                "demo_4",
                "--v1f-model",
                str(model_path),
                "--output-json",
                str(out / "residual_breakdown_v1g.json"),
                "--report-md",
                str(out / "residual_validation_v1g_report.md"),
            ],
            cwd=_EXPERIMENT_DIR,
        )

    if not args.skip_rollout_val:
        _run(
            [
                _SCRIPT,
                str(_EXPERIMENT_DIR / "run_physics_residual_rollout_validation.py"),
                "--enable-physics-residual-repair",
                "--aligned-original-model",
                str(model_path),
                "--output-json",
                str(out / "rollout_validation_report.json"),
                "--report-md",
                str(out / "rollout_validation_report.md"),
            ],
            cwd=_EXPERIMENT_DIR,
        )

    status = {
        "pipeline": "V1-G-stage1-p1xy",
        "model": str(model_path),
        "aligned_original_unmodified": True,
        "outputs": {
            "config": str(out / "trained_model" / "v1g_stage1_config.json"),
            "residual_breakdown": str(out / "residual_breakdown_v1g.json"),
            "rollout_validation": str(out / "rollout_validation_report.json"),
        },
    }
    (out / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
