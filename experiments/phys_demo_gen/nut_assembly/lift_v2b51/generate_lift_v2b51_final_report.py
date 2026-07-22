#!/usr/bin/env python3
"""从 lift_v2b51_rollout_samples.jsonl 生成最终诊断报告。"""
from __future__ import annotations

import argparse
import csv
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

DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b51"
PARTIAL_THRESH = 0.005
WEAK_THRESH = 0.002

DIAG_FIELDS = (
    "gripper_qpos_before_close",
    "gripper_qpos_after_close",
    "left_finger_contact_count",
    "right_finger_contact_count",
    "bilateral_contact_steps",
    "contact_duration",
    "eef_nut_xy_at_close",
    "eef_nut_z_at_close",
    "eef_z_lift_delta",
    "nut_z_lift_delta",
    "nut_eef_coupling_ratio",
    "nut_xy_slip",
    "partial_lift_success",
    "outcome_label",
    "search_index",
    "rollout_kind",
)


def _nut_z(rec: dict[str, Any]) -> float:
    return float(rec.get("nut_z_lift_delta", rec.get("nut_lift_phase_delta", 0.0)))


def _load_records(jsonl_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _distribution(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None}
    nums = [float(v) for v in values]
    nums_sorted = sorted(nums)
    p50 = nums_sorted[len(nums_sorted) // 2]
    p90 = nums_sorted[int(len(nums_sorted) * 0.9)]
    return {
        "count": len(nums),
        "min": min(nums),
        "max": max(nums),
        "mean": float(statistics.mean(nums)),
        "p50": p50,
        "p90": p90,
    }


def _candidate_summary(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "search_index": rec.get("search_index"),
        "rollout_kind": rec.get("rollout_kind"),
        "outcome_label": rec.get("outcome_label"),
        "partial_lift_success": rec.get("partial_lift_success"),
        "nut_z_lift_delta": _nut_z(rec),
        "bilateral_contact_steps": rec.get("bilateral_contact_steps"),
        "right_finger_contact_count": rec.get("right_finger_contact_count"),
        "left_finger_contact_count": rec.get("left_finger_contact_count"),
        "contact_duration": rec.get("contact_duration"),
        "nut_eef_coupling_ratio": rec.get("nut_eef_coupling_ratio"),
        "nut_xy_slip": rec.get("nut_xy_slip"),
        "lift_v2b51_params": rec.get("lift_v2b51_params"),
        **{k: rec.get(k) for k in DIAG_FIELDS if k in rec},
    }


def _top_candidates(records: list[dict[str, Any]], *, key_fn, reverse: bool = True, k: int = 20) -> list[dict[str, Any]]:
    ranked = sorted(records, key=key_fn, reverse=reverse)
    return [_candidate_summary(r) for r in ranked[:k]]


def generate_reports(
    *,
    jsonl_path: Path,
    output_dir: Path,
    seed: int = 42,
    max_evals: int = 1200,
) -> dict[str, Any]:
    records = _load_records(jsonl_path)
    search_records = [r for r in records if r.get("rollout_kind") in ("lift_v2b51_search", "contact_aware_seed", "baseline")]

    partial_count = sum(1 for r in records if r.get("partial_lift_success"))
    weak_count = sum(1 for r in records if _nut_z(r) >= WEAK_THRESH)
    max_nut_lift = max((_nut_z(r) for r in records), default=0.0)

    best_nut = max(records, key=_nut_z, default={})
    best_bilateral = max(records, key=lambda r: int(r.get("bilateral_contact_steps", 0)), default={})
    best_coupling = max(records, key=lambda r: float(r.get("nut_eef_coupling_ratio", -999)), default={})
    best_low_slip = min(
        [r for r in records if _nut_z(r) > 0] or records,
        key=lambda r: float(r.get("nut_xy_slip", 999)),
        default={},
    )

    outcome_dist = dict(Counter(str(r.get("outcome_label", "unknown")) for r in records))

    distributions = {
        "right_finger_contact_count": _distribution([int(r.get("right_finger_contact_count", 0)) for r in records]),
        "left_finger_contact_count": _distribution([int(r.get("left_finger_contact_count", 0)) for r in records]),
        "bilateral_contact_steps": _distribution([int(r.get("bilateral_contact_steps", 0)) for r in records]),
        "contact_duration": _distribution([int(r.get("contact_duration", 0)) for r in records]),
        "nut_xy_slip": _distribution([float(r.get("nut_xy_slip", 0.0)) for r in records]),
        "nut_z_lift_delta": _distribution([_nut_z(r) for r in records]),
    }

    top_csv_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()

    def _add_top(candidates: list[dict[str, Any]], rank_source: str) -> None:
        for rank, cand in enumerate(candidates, start=1):
            key = (cand.get("search_index"), cand.get("rollout_kind"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            row = dict(cand)
            row["rank_source"] = rank_source
            row["rank"] = rank
            top_csv_rows.append(row)

    _add_top(_top_candidates(records, key_fn=_nut_z), "nut_z_lift_delta")
    _add_top(_top_candidates(records, key_fn=lambda r: int(r.get("bilateral_contact_steps", 0))), "bilateral_contact_steps")
    _add_top(_top_candidates(records, key_fn=lambda r: float(r.get("nut_eef_coupling_ratio", -999))), "nut_eef_coupling_ratio")
    _add_top(
        _top_candidates(records, key_fn=lambda r: float(r.get("nut_xy_slip", 999)), reverse=False),
        "low_nut_xy_slip",
    )

    csv_path = output_dir / "top_lift_candidates.csv"
    if top_csv_rows:
        fieldnames: list[str] = []
        for row in top_csv_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in top_csv_rows:
                flat = dict(row)
                if isinstance(flat.get("lift_v2b51_params"), dict):
                    flat["lift_v2b51_params_json"] = json.dumps(flat.pop("lift_v2b51_params"))
                writer.writerow(flat)

    diag_summary = {
        "total_evals": len(search_records),
        "total_rollouts_in_jsonl": len(records),
        "partial_lift_success_count": partial_count,
        "weak_lift_positive_count": weak_count,
        "weak_lift_delta_threshold_m": WEAK_THRESH,
        "partial_lift_delta_threshold_m": PARTIAL_THRESH,
        "max_nut_lift_delta": max_nut_lift,
        "best_nut_lift_delta_candidate": _candidate_summary(best_nut),
        "best_bilateral_contact_candidate": _candidate_summary(best_bilateral),
        "best_nut_eef_coupling_candidate": _candidate_summary(best_coupling),
        "best_low_slip_candidate": _candidate_summary(best_low_slip),
        "distributions": distributions,
        "outcome_distribution": outcome_dist,
        "top_candidates_csv": str(csv_path),
    }

    report = {
        "task": "lift_v2b51_final_diagnostics",
        "demo_key": "demo_3",
        "seed": seed,
        "max_evals": max_evals,
        "total_evals": len(search_records),
        "partial_lift_success_count": partial_count,
        "weak_lift_positive_count": weak_count,
        "max_nut_lift_delta": max_nut_lift,
        "partial_lift_delta_goal_m": PARTIAL_THRESH,
        "weak_lift_delta_threshold_m": WEAK_THRESH,
        "best_nut_lift_delta_candidate": _candidate_summary(best_nut),
        "best_bilateral_contact_candidate": _candidate_summary(best_bilateral),
        "best_nut_eef_coupling_candidate": _candidate_summary(best_coupling),
        "best_low_slip_candidate": _candidate_summary(best_low_slip),
        "distributions": distributions,
        "outcome_distribution": outcome_dist,
        "branch_recommendation": (
            "v1g_dataset_draft" if partial_count > 0 else ("v2b52_cem" if weak_count > 0 else "v2b52_asymmetric_grasp")
        ),
        "outputs": {
            "rollout_samples_jsonl": str(jsonl_path),
            "diagnostics_summary_json": str(output_dir / "lift_v2b51_diagnostics_summary.json"),
            "report_json": str(output_dir / "lift_v2b51_report.json"),
            "top_lift_candidates_csv": str(csv_path),
        },
    }

    (output_dir / "lift_v2b51_diagnostics_summary.json").write_text(json.dumps(diag_summary, indent=2), encoding="utf-8")
    (output_dir / "lift_v2b51_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_OUTPUT / "lift_v2b51_rollout_samples.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-evals", type=int, default=1200)
    args = parser.parse_args()
    if not args.jsonl.exists():
        print(f"missing jsonl: {args.jsonl}", file=sys.stderr)
        return 2
    report = generate_reports(
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        seed=args.seed,
        max_evals=args.max_evals,
    )
    print(json.dumps({"branch": report["branch_recommendation"], "partial": report["partial_lift_success_count"], "weak": report["weak_lift_positive_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
