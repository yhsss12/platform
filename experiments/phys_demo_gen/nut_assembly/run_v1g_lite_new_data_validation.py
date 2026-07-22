#!/usr/bin/env python3
"""V1-G-lite 在新 data/ demo 集上的性能验证（vs aligned-original reference）。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_V1F_DIR = _EXPERIMENT_DIR / "v1_residual_model" / "repair_parameter_model_v1f"
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
for path in (_EXPERIMENT_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DEFAULT_CEM_REPORT,
    DEFAULT_PINN_MODEL,
    V1G_STAGE1_LITE_P1P2_MODEL,
)
from run_v1f_plus_evaluation import _repair_cfg_for_new_demo, run_offline_repair  # noqa: E402
from run_v1f_quick_evaluation import job_key, load_partial  # noqa: E402
from v1f_plus_utils import DEFAULT_ALIGNED_MODEL, list_demo_keys, load_failure_map  # noqa: E402

DEFAULT_DATA_DIR = _EXPERIMENT_DIR / "data"
DEFAULT_FAILED = DEFAULT_DATA_DIR / "demo_failed.hdf5"
DEFAULT_SUCCESS = DEFAULT_DATA_DIR / "demo.hdf5"
DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"
DEFAULT_ALIGNED_REF_EVAL = (
    _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus" / "evaluation" / "v1f_plus_evaluation_report.json"
)
DEFAULT_OUT_DIR = _EXPERIMENT_DIR / "outputs" / "v1g_stage1_lite_p1p2" / "new_data_validation"


def _load_aligned_reference(
    report_path: Path,
    *,
    demo_keys: list[str],
    method: str = "v1f_plain_top_k",
) -> dict[str, dict[str, Any]]:
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in report.get("results", []):
        if (
            row.get("demo_group") == "new"
            and row.get("demo_key") in demo_keys
            and row.get("model_label") == "aligned-original"
            and row.get("selection_method") == method
        ):
            m = row["metrics"]
            out[row["demo_key"]] = {
                "repair_rate_at_20": float(m.get("repair_rate_at_20", 0.0)),
                "success_at_20": m.get("success_at_k", {}).get("at_20"),
                "num_successes": m.get("num_successes_written"),
                "failure_type": row.get("failure_type"),
                "coarse_failure_type": row.get("coarse_failure_type"),
            }
    return out


def _summarize(rows: list[dict[str, Any]], aligned_ref: dict[str, dict[str, Any]]) -> dict[str, Any]:
    v1g_rates = [float(r["repair_rate_at_20"]) for r in rows]
    ref_rates = [float(aligned_ref[r["demo_key"]]["repair_rate_at_20"]) for r in rows if r["demo_key"] in aligned_ref]
    deltas = []
    improved = 0
    regressed = 0
    for r in rows:
        dk = r["demo_key"]
        if dk not in aligned_ref:
            continue
        d = float(r["repair_rate_at_20"]) - float(aligned_ref[dk]["repair_rate_at_20"])
        deltas.append(d)
        if d > 0.01:
            improved += 1
        elif d < -0.05:
            regressed += 1

    by_ft: dict[str, list[float]] = defaultdict(list)
    by_ft_ref: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        ft = str(r.get("coarse_failure_type") or r.get("failure_type") or "unknown")
        by_ft[ft].append(float(r["repair_rate_at_20"]))
        if r["demo_key"] in aligned_ref:
            by_ft_ref[ft].append(float(aligned_ref[r["demo_key"]]["repair_rate_at_20"]))

    return {
        "demo_count": len(rows),
        "v1g_avg_repair_rate_at_20": float(statistics.mean(v1g_rates)) if v1g_rates else None,
        "aligned_ref_avg_repair_rate_at_20": float(statistics.mean(ref_rates)) if ref_rates else None,
        "avg_delta_repair_rate_at_20": float(statistics.mean(deltas)) if deltas else None,
        "demos_improved": improved,
        "demos_regressed_gt_5pp": regressed,
        "by_failure_type": {
            ft: {
                "v1g_avg": float(statistics.mean(vals)),
                "aligned_ref_avg": float(statistics.mean(by_ft_ref.get(ft, []))) if by_ft_ref.get(ft) else None,
                "count": len(vals),
            }
            for ft, vals in sorted(by_ft.items())
        },
    }


def write_summary_md(
    *,
    payload: dict[str, Any],
    path: Path,
) -> None:
    summary = payload["summary"]
    lines = [
        "# V1-G-stage1-lite-p1p2 新 demo 验证报告",
        "",
        f"> 生成时间：{payload['generated_at']}",
        f"> 数据：`data/demo_failed.hdf5`（{payload['demo_count']} 条 failed demo）",
        f"> 方法：`v1f_plain_top_k`，num_samples={payload['num_samples']}，top_k={payload['top_k']}，seed={payload['seed']}",
        "",
        "## 模型",
        "",
        f"- **PINN 模型**：V1-G-stage1-lite-p1p2 (`{payload['v1g_model']}`)",
        f"- **对照 reference**：aligned-original (`{payload['aligned_reference_model']}`)",
        "",
        "## 汇总",
        "",
        f"| 指标 | V1-G-lite | aligned-original ref | Δ |",
        f"|------|-----------|----------------------|---|",
        f"| 平均 repair_rate@20 | {summary['v1g_avg_repair_rate_at_20']:.1%} | "
        f"{summary['aligned_ref_avg_repair_rate_at_20']:.1%} | "
        f"{summary['avg_delta_repair_rate_at_20']:+.1%} |",
        f"| demo 改善数（Δ>1pp） | {summary['demos_improved']} | — | — |",
        f"| demo regression（Δ<-5pp） | {summary['demos_regressed_gt_5pp']} | — | — |",
        "",
        "## 按 failure type",
        "",
        "| failure_type | count | V1-G-lite avg | aligned-original avg |",
        "|--------------|-------|---------------|----------------------|",
    ]
    for ft, block in summary.get("by_failure_type", {}).items():
        ref = block.get("aligned_ref_avg")
        ref_s = f"{ref:.1%}" if ref is not None else "N/A"
        lines.append(f"| {ft} | {block['count']} | {block['v1g_avg']:.1%} | {ref_s} |")

    lines.extend(["", "## 逐 demo 对比", "", "| demo | failure_type | aligned ref | V1-G-lite | Δ |", "|------|--------------|-------------|-----------|---|"])
    for row in sorted(payload["per_demo"], key=lambda r: int(r["demo_key"].split("_")[-1])):
        ref = row.get("aligned_ref_repair_rate_at_20")
        ref_s = f"{ref:.1%}" if ref is not None else "N/A"
        delta = row.get("delta_repair_rate_at_20")
        delta_s = f"{delta:+.1%}" if delta is not None else "N/A"
        lines.append(
            f"| {row['demo_key']} | {row.get('failure_type', 'unknown')} | {ref_s} | "
            f"{row['repair_rate_at_20']:.1%} | {delta_s} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate V1-G-lite on new data/ demos")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--aligned-ref-eval", type=Path, default=DEFAULT_ALIGNED_REF_EVAL)
    parser.add_argument("--v1g-model", type=Path, default=V1G_STAGE1_LITE_P1P2_MODEL)
    parser.add_argument("--aligned-reference-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-limit", type=int, default=0, help="0 = all demos")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    if not args.failed_hdf5.exists():
        raise SystemExit(f"Failed HDF5 missing: {args.failed_hdf5}")
    if not args.v1g_model.exists():
        raise SystemExit(f"V1-G-lite checkpoint missing: {args.v1g_model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "validation_partial.jsonl"
    done = load_partial(partial_path) if args.resume else {}
    rows: list[dict[str, Any]] = list(done.values())

    demo_keys = list_demo_keys(args.failed_hdf5)
    if args.demo_limit > 0:
        demo_keys = demo_keys[: args.demo_limit]

    failure_map = load_failure_map(args.audit_report)
    aligned_ref = _load_aligned_reference(args.aligned_ref_eval, demo_keys=demo_keys)

    for demo_key in demo_keys:
        key = job_key("new", demo_key, "v1g-lite")
        if key in done:
            continue
        cfg = _repair_cfg_for_new_demo(demo_key, failure_map)
        print(f"[new-data-val] {demo_key} ({cfg.get('failure_type')})", flush=True)
        result = run_offline_repair(
            demo_key=demo_key,
            cfg=cfg,
            v1f_model=args.v1g_model,
            failed_hdf5=args.failed_hdf5,
            cem_report=args.cem_report,
            selection_method="v1f_plain_top_k",
            num_samples=args.num_samples,
            top_k=args.top_k,
            seed=args.seed,
            v1e_model=args.v1e_model,
            model_label="v1g-lite",
            demo_group="new",
        )
        ref = aligned_ref.get(demo_key, {})
        row = {
            "job_key": key,
            "demo_key": demo_key,
            "failure_type": result.get("failure_type", cfg.get("failure_type")),
            "coarse_failure_type": result.get("coarse_failure_type", cfg.get("failure_type")),
            "search_kind": cfg.get("search_kind"),
            "repair_rate_at_20": float(result["metrics"]["repair_rate_at_20"]),
            "success_at_20": result["metrics"].get("success_at_k", {}).get("at_20"),
            "num_successes": result["metrics"].get("num_successes_written"),
            "best_E_total": result["metrics"].get("best_E_total"),
            "aligned_ref_repair_rate_at_20": ref.get("repair_rate_at_20"),
            "delta_repair_rate_at_20": (
                float(result["metrics"]["repair_rate_at_20"]) - float(ref["repair_rate_at_20"])
                if "repair_rate_at_20" in ref
                else None
            ),
        }
        rows.append(row)
        done[key] = row
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summarize(rows, aligned_ref)
    payload = {
        "schema": "v1g_lite_new_data_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "failed_hdf5": str(args.failed_hdf5),
            "success_hdf5": str(DEFAULT_SUCCESS),
            "demo_count": len(demo_keys),
        },
        "models": {
            "v1g_lite": str(args.v1g_model),
            "aligned_reference": str(args.aligned_reference_model),
            "aligned_ref_eval_source": str(args.aligned_ref_eval),
        },
        "validation": {
            "method": "v1f_plain_top_k",
            "num_samples": args.num_samples,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "seed": args.seed,
        "v1g_model": str(args.v1g_model),
        "aligned_reference_model": str(args.aligned_reference_model),
        "demo_count": len(demo_keys),
        "demo_keys": demo_keys,
        "summary": summary,
        "per_demo": rows,
    }

    json_path = args.output_dir / "new_data_validation_report.json"
    md_path = args.output_dir / "new_data_validation_report.md"
    csv_path = args.output_dir / "new_data_validation_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(payload=payload, path=md_path)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({"json": str(json_path), "md": str(md_path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
