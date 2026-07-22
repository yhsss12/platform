#!/usr/bin/env python3
"""V1-G contact-aware dataset draft（仅 partial + weak positive + hard negative）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_V2B51_DIR = Path(__file__).resolve().parent
for path in (_EXPERIMENT_DIR, _V2B51_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_JSONL = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_rollout_samples.jsonl"
DEFAULT_REPORT = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_report.json"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1g_contact_aware_draft"

PARTIAL_THRESH = 0.005
WEAK_THRESH = 0.002
HARD_NEG_BILATERAL_MIN = 5


def _nut_z(rec: dict[str, Any]) -> float:
    return float(rec.get("nut_z_lift_delta", rec.get("nut_lift_phase_delta", 0.0)))


def _bucket(rec: dict[str, Any]) -> str | None:
    if rec.get("partial_lift_success") or _nut_z(rec) >= PARTIAL_THRESH:
        return "partial_lift_success"
    if _nut_z(rec) >= WEAK_THRESH:
        return "weak_lift_positive"
    if int(rec.get("bilateral_contact_steps", 0)) >= HARD_NEG_BILATERAL_MIN and _nut_z(rec) < WEAK_THRESH:
        return "hard_negative_bilateral_no_lift"
    if _nut_z(rec) < 0 and int(rec.get("left_finger_contact_count", 0)) > 50:
        return "hard_negative_contact_no_lift"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if int(report.get("partial_lift_success_count", 0)) <= 0:
            print("SKIP: partial_lift_success_count=0, V1-G dataset draft blocked")
            return 1
    else:
        print(f"WARN: missing report {args.report}")

    records = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    buckets: dict[str, list[dict[str, Any]]] = {
        "partial_lift_success": [],
        "weak_lift_positive": [],
        "hard_negative_bilateral_no_lift": [],
        "hard_negative_contact_no_lift": [],
    }
    for rec in records:
        b = _bucket(rec)
        if b:
            slim = {k: v for k, v in rec.items() if not str(k).startswith("per_step_")}
            buckets[b].append(slim)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draft_jsonl = args.output_dir / "v1g_contact_aware_draft_samples.jsonl"
    with draft_jsonl.open("w", encoding="utf-8") as handle:
        for bucket, items in buckets.items():
            for item in items:
                item["v1g_bucket"] = bucket
                handle.write(json.dumps(item, default=str) + "\n")

    build_report = {
        "status": "draft_only_no_training",
        "gate": "partial_lift_success_count > 0",
        "await_user_confirmation_before_training": True,
        "counts": {k: len(v) for k, v in buckets.items()},
        "total_draft_samples": sum(len(v) for v in buckets.values()),
        "thresholds": {
            "partial_lift_delta_m": PARTIAL_THRESH,
            "weak_lift_delta_m": WEAK_THRESH,
            "hard_neg_bilateral_min_steps": HARD_NEG_BILATERAL_MIN,
        },
        "outputs": {
            "draft_jsonl": str(draft_jsonl),
            "build_report": str(args.output_dir / "v1g_dataset_build_report.json"),
        },
        "note": "Do NOT train V1-G until user confirms.",
    }
    (args.output_dir / "v1g_dataset_build_report.json").write_text(json.dumps(build_report, indent=2), encoding="utf-8")
    print(json.dumps(build_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
