#!/usr/bin/env python3
"""Task 1：New demo rollout repairability audit。"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_ROLLOUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "v1f_aligned_plus"
    / "new_rollout_samples.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "v1f_aligned_plus"
    / "repairability_audit"
)

IMPROVEMENT_REL_THRESH = 0.15
IMPROVEMENT_ABS_THRESH = 5.0


def _classify_repairability(
    *,
    success_count: int,
    avg_e_before: float,
    best_e_after: float,
) -> str:
    if success_count > 0:
        return "repairable"
    drop = avg_e_before - best_e_after
    rel = drop / max(avg_e_before, 1e-6)
    if drop >= IMPROVEMENT_ABS_THRESH or rel >= IMPROVEMENT_REL_THRESH:
        return "hard_but_improvable"
    return "no_positive_candidate"


def audit_repairability(rollout_jsonl: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_demo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with rollout_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_demo[str(rec["source_demo"])].append(rec)

    per_demo: list[dict[str, Any]] = []
    for demo_key in sorted(by_demo.keys(), key=lambda k: int(k.split("_")[-1])):
        rows = by_demo[demo_key]
        e_totals = [float(r["E_total"]) for r in rows]
        e_before_vals = [float(r.get("context", {}).get("original_E_total_norm", r["E_total"])) for r in rows]
        avg_e_before = float(statistics.mean(e_before_vals))
        best_idx = min(range(len(rows)), key=lambda i: e_totals[i])
        best_row = rows[best_idx]
        success_rows = [r for r in rows if r.get("success_flag")]
        success_count = len(success_rows)
        best_e_after = float(min(e_totals))
        failure_type = str(rows[0].get("original_failure_type", "unknown"))
        whether = _classify_repairability(
            success_count=success_count,
            avg_e_before=avg_e_before,
            best_e_after=best_e_after,
        )
        per_demo.append(
            {
                "source_demo": demo_key,
                "candidate_count": len(rows),
                "success_count": success_count,
                "success_rate": success_count / max(len(rows), 1),
                "best_E_total": best_e_after,
                "best_theta": best_row.get("repair_theta", {}),
                "best_active": best_row.get("active"),
                "failure_type": failure_type,
                "avg_E_before": avg_e_before,
                "best_E_after": best_e_after,
                "energy_drop": avg_e_before - best_e_after,
                "energy_drop_ratio": (avg_e_before - best_e_after) / max(avg_e_before, 1e-6),
                "whether_repairable": whether,
                "ranking_supervision_eligible": whether != "no_positive_candidate",
            }
        )

    counts = defaultdict(int)
    for row in per_demo:
        counts[row["whether_repairable"]] += 1
    ft_counts = defaultdict(int)
    for row in per_demo:
        ft_counts[row["failure_type"]] += 1

    report = {
        "task": "new_demo_repairability_audit",
        "input": str(rollout_jsonl),
        "thresholds": {
            "improvement_rel": IMPROVEMENT_REL_THRESH,
            "improvement_abs": IMPROVEMENT_ABS_THRESH,
        },
        "num_demos": len(per_demo),
        "whether_repairable_counts": dict(counts),
        "failure_type_counts": dict(ft_counts),
        "per_demo": per_demo,
        "repairable_demo_keys": [r["source_demo"] for r in per_demo if r["whether_repairable"] == "repairable"],
        "hard_but_improvable_demo_keys": [
            r["source_demo"] for r in per_demo if r["whether_repairable"] == "hard_but_improvable"
        ],
        "no_positive_candidate_demo_keys": [
            r["source_demo"] for r in per_demo if r["whether_repairable"] == "no_positive_candidate"
        ],
    }
    return report, per_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit new demo repairability from rollout samples")
    parser.add_argument("--rollout-jsonl", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report, rows = audit_repairability(args.rollout_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "new_demo_repairability_report.json"
    csv_path = args.output_dir / "new_demo_repairability_summary.csv"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_fields = [
        "source_demo",
        "candidate_count",
        "success_count",
        "success_rate",
        "best_E_total",
        "failure_type",
        "avg_E_before",
        "best_E_after",
        "energy_drop",
        "energy_drop_ratio",
        "whether_repairable",
        "ranking_supervision_eligible",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"report": str(report_path), "csv": str(csv_path), "counts": report["whether_repairable_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
