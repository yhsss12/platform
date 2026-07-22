#!/usr/bin/env python3
"""Contact failure diagnosis + V2-B5.2 asymmetric grasp refinement 方案。"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_V2B51_DIR = Path(__file__).resolve().parent
for path in (_EXPERIMENT_DIR, _V2B51_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_lift_v2b51_final_report import _load_records, _nut_z  # noqa: E402

DEFAULT_JSONL = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_rollout_samples.jsonl"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b51"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = _load_records(args.jsonl)
    partial = sum(1 for r in records if r.get("partial_lift_success"))
    weak = sum(1 for r in records if _nut_z(r) >= 0.002)

    bilateral = [r for r in records if int(r.get("bilateral_contact_steps", 0)) > 0]
    right_zero = sum(1 for r in records if int(r.get("right_finger_contact_count", 0)) == 0)
    left_only = sum(1 for r in records if int(r.get("left_finger_contact_count", 0)) > 0 and int(r.get("right_finger_contact_count", 0)) == 0)
    bilateral_no_lift = [r for r in bilateral if _nut_z(r) <= 0]

    right_counts = [int(r.get("right_finger_contact_count", 0)) for r in records]
    left_counts = [int(r.get("left_finger_contact_count", 0)) for r in records]

    diagnosis = {
        "status": "transport_failed_primary_lift_underdeveloped_secondary",
        "primary_failure_mode": "transport_failed",
        "secondary_failure_mode": "lift_underdeveloped",
        "legacy_label": "lift_failed",
        "success_lift_audit_note": "76/77 success demos nut_z_lift_delta>=0.005m; partial threshold retained.",
        "partial_lift_success_count": partial,
        "weak_lift_positive_count": weak,
        "total_rollouts": len(records),
        "right_finger_never_contact_rate": float(right_zero / max(len(records), 1)),
        "left_only_contact_rate": float(left_only / max(len(records), 1)),
        "bilateral_contact_rollouts": len(bilateral),
        "bilateral_but_no_positive_lift": len(bilateral_no_lift),
        "bilateral_no_lift_rate": float(len(bilateral_no_lift) / max(len(bilateral), 1)),
        "right_finger_contact_stats": {
            "mean": float(statistics.mean(right_counts)) if right_counts else 0,
            "max": max(right_counts) if right_counts else 0,
            "zero_fraction": float(right_zero / max(len(records), 1)),
        },
        "left_finger_contact_stats": {
            "mean": float(statistics.mean(left_counts)) if left_counts else 0,
            "max": max(left_counts) if left_counts else 0,
        },
        "outcome_distribution": dict(Counter(str(r.get("outcome_label", "unknown")) for r in records)),
        "root_cause_hypotheses": [
            "Right finger pad rarely reaches nut geometry: grasp_xy / lateral_correction insufficient for asymmetric nut pose.",
            "Left finger dominates contact (high left_finger_contact_count, right=0) -> nut pivots instead of lifting.",
            "Bilateral contact sometimes achieved but nut_z_lift_delta stays <= 0: premature lift or slip during micro_lift.",
            "Gripper close is symmetric; demo_3 may need gripper_asym_close_offset to bias right pad inward.",
        ],
        "v2b52_asymmetric_grasp_plan": {
            "new_params": [
                "grasp_lateral_bias_x",
                "grasp_lateral_bias_y",
                "approach_yaw_bias",
                "gripper_asym_close_offset",
                "right_finger_contact_bias",
            ],
            "template_sequence": [
                "asymmetric_grasp_correction",
                "lower_approach",
                "squeeze_close (stronger)",
                "longer contact_settle_steps",
                "micro_lift",
                "reclose_after_micro_lift",
                "second_micro_lift",
                "slow_lift",
            ],
            "constraints": [
                "No object_poses / states modification",
                "All labels from MuJoCo rollout only",
                "V1-G PINN remains gated until partial_lift_success",
            ],
        },
        "recommended_next_actions": [
            "Implement lift_v2b52_refiner with asymmetric grasp params",
            "Seed CEM from top bilateral candidates with right_finger_contact_count=0 filtered out",
            "Add per-step right_finger_tip_distance logging at grasp_idx",
        ],
    }

    out_path = args.output_dir / "contact_failure_diagnosis.json"
    out_path.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    plan_path = args.output_dir / "v2b52_asymmetric_grasp_plan.json"
    plan_path.write_text(json.dumps(diagnosis["v2b52_asymmetric_grasp_plan"], indent=2), encoding="utf-8")
    print(json.dumps({"diagnosis": str(out_path), "plan": str(plan_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
