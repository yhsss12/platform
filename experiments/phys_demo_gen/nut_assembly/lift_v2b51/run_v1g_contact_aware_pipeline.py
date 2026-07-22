#!/usr/bin/env python3
"""Task 5：仅当 V2-B5.1 产生 partial_lift_success 时，才启动 V1-G contact-aware PINN 训练。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_V2B51_DIR = Path(__file__).resolve().parent
for path in (_EXPERIMENT_DIR, _V2B51_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_V2B51_REPORT = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_report.json"
DEFAULT_V2B51_JSONL = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_rollout_samples.jsonl"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1g_contact_aware"


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-G contact-aware PINN gate")
    parser.add_argument("--v2b51-report", type=Path, default=DEFAULT_V2B51_REPORT)
    parser.add_argument("--v2b51-jsonl", type=Path, default=DEFAULT_V2B51_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Skip partial_lift_success gate (debug only)")
    args = parser.parse_args()

    if not args.v2b51_report.exists():
        print(f"SKIP V1-G: missing V2-B5.1 report at {args.v2b51_report}")
        return 2

    report = json.loads(args.v2b51_report.read_text(encoding="utf-8"))
    acceptance = report.get("acceptance", {})
    has_partial = bool(acceptance.get("has_partial_lift_success"))

    gate_path = args.output_dir / "v1g_training_gate.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not has_partial and not args.force:
        gate = {
            "status": "blocked",
            "reason": "no partial_lift_success from V2-B5.1 targeted lift search",
            "v2b51_report": str(args.v2b51_report),
            "next_step": "Improve contact-aware lift refiner / expand demo_3 search before PINN training",
        }
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print(json.dumps(gate, indent=2))
        return 1

    gate = {
        "status": "ready",
        "reason": "partial_lift_success found" if has_partial else "forced by --force",
        "v2b51_report": str(args.v2b51_report),
        "v2b51_jsonl": str(args.v2b51_jsonl),
        "planned_steps": [
            "build_v1g_contact_aware_dataset.py (partial + hard-negative lift samples)",
            "train_pinn_v1g_contact_aware_model.py (contact residual heads + uncertainty)",
        ],
        "note": "V1-G training scripts not yet implemented; gate passed.",
    }
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
