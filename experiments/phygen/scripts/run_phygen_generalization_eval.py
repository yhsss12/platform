#!/usr/bin/env python3
"""Batch train/eval PhyGen for coffee_preparation generalization splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default="runs/phygen_coffee_theta_sweep_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--budget", type=int, default=5)
    args = parser.parse_args()

    root = (ROOT / args.sweep_root).resolve()
    splits = [
        ("by_sweep", root / "train_by_sweep_feedback.jsonl", root / "test_by_sweep_feedback.jsonl"),
        ("by_source", root / "train_by_source_feedback.jsonl", root / "test_by_source_feedback.jsonl"),
    ]

    reports: dict[str, Any] = {"splits": {}}
    for name, train_path, test_path in splits:
        out_dir = root / f"train_{name}"
        _run(
            [
                sys.executable,
                "scripts/train_phygen.py",
                "--task",
                "coffee_preparation",
                "--feedback-jsonl",
                str(train_path.relative_to(ROOT)),
                "--output-dir",
                str(out_dir.relative_to(ROOT)),
                "--epochs",
                str(args.epochs),
                "--standard-pinn",
                "--use-component-loss",
                "--pool-size",
                str(args.pool_size),
                "--budget",
                str(args.budget),
                "--include-repaired",
            ]
        )
        eval_out = root / f"eval_report_{name}.json"
        _run(
            [
                sys.executable,
                "experiments/phygen/scripts/evaluate_phygen_selector_on_feedback.py",
                "--task",
                "coffee_preparation",
                "--checkpoint",
                str((out_dir / "coffee_preparation_failed_conditioned_pinn.pt").relative_to(ROOT)),
                "--feedback-jsonl",
                str(test_path.relative_to(ROOT)),
                "--output",
                str(eval_out.relative_to(ROOT)),
                "--budgets",
                "1,3,5",
            ]
        )
        with eval_out.open("r", encoding="utf-8") as f:
            reports["splits"][name] = json.load(f)

    summary_path = root / "generalization_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
