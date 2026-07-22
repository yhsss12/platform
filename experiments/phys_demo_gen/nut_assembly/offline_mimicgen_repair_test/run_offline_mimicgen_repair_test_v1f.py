#!/usr/bin/env python3
"""Offline MimicGen Repair Test V1-F：V1-E vs V1-F PINN 对比。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_OFFLINE_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _OFFLINE_DIR.parent
_V1_DIR = _EXPERIMENT_DIR / "v1_residual_model"
_V1E_DIR = _V1_DIR / "repair_parameter_model"
_V1F_DIR = _V1_DIR / "repair_parameter_model_v1f"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    CONTEXT_SOURCES,
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_PINN_MODEL,
    DEFAULT_V1F_MODEL,
    DEFAULT_V1F_OUTPUT_DIR,
    DEMO_REPAIR_CONFIGS,
    V1F_SELECTION_METHODS,
)
from pinn_repair_inference import clear_repair_model_cache  # noqa: E402
from pinn_v1f_inference import clear_v1f_model_cache  # noqa: E402
from repair_common_v1f import (  # noqa: E402
    extract_repair_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
    summarize_method_results_v1f,
)
from repair_rollout import run_repair_rollout  # noqa: E402
from repaired_hdf5_writer import append_successful_repair_demo, init_repaired_dataset  # noqa: E402
from robosuite_env_loader import check_environment  # noqa: E402


def _failure_record(
    *,
    source_demo: str,
    method: str,
    candidate: dict[str, Any],
    rollout: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_demo": source_demo,
        "method": method,
        "candidate_index": candidate.get("index"),
        "v1e_E_total": candidate.get("v1e_E_total"),
        "v1f_E_total": candidate.get("v1f_E_total"),
        "explicit_E_total": candidate.get("explicit_E_total"),
        "v1e_success_prob": candidate.get("v1e_success_prob"),
        "v1f_success_prob": candidate.get("v1f_success_prob"),
        "v1f_uncertainty": candidate.get("v1f_uncertainty"),
        "repair_theta": candidate.get("insertion") or candidate.get("grasp_lift"),
        "repair_lift_extra": candidate.get("lift_extra"),
        "success_flag": bool(rollout.get("success_flag")),
        "E_total_norm": float(rollout.get("E_total_norm", 0.0)),
        "nut_lift_delta": float(rollout.get("nut_lift_delta", 0.0)),
        "grasp_success_proxy": bool(rollout.get("grasp_success_proxy", False)),
        "lift_success_proxy": bool(rollout.get("lift_success_proxy", False)),
        "final_nut_peg_xy": float(rollout.get("final_nut_peg_xy", 0.0)),
        "final_z_diff": rollout.get("final_z_diff"),
        "failure_guess": rollout.get("failure_guess"),
        "object_poses_modified": bool(rollout.get("object_poses_modified", False)),
    }


def _run_method_for_demo(
    *,
    demo_key: str,
    cfg: dict[str, Any],
    method: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    failed_hdf5: Path,
    cem_report: Path | None,
    repaired_all: Path,
    repaired_v1f_only: Path,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    indices = select_candidate_indices_v1f(candidates, method=method, top_k=top_k, rng=rng)
    rollout_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    written_keys: list[str] = []

    for rank, idx in enumerate(indices, start=1):
        cand = candidates[idx]
        rollout = run_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            search_kind=cfg["search_kind"],
            cem_report=cem_report,
            candidate=cand,
        )
        rollout_results.append(rollout)

        if rollout.get("success_flag"):
            repaired_key = f"repaired_{demo_key}_{method}_{rank:02d}"
            meta = {
                "source_demo": demo_key,
                "source_failure_type": cfg["failure_type"],
                "selection_method": method,
                "candidate_index": cand.get("index"),
                "v1e_E_total": cand.get("v1e_E_total"),
                "v1f_E_total": cand.get("v1f_E_total"),
                "explicit_E_total": cand.get("explicit_E_total"),
                "v1f_uncertainty": cand.get("v1f_uncertainty"),
                "repair_theta": cand.get("insertion") or cand.get("grasp_lift"),
                "repair_lift_extra": cand.get("lift_extra"),
                "rollout_kind": "offline_mimicgen_repair_v1f",
                "object_poses_modified": False,
            }
            append_successful_repair_demo(
                output_hdf5=repaired_all,
                source_hdf5=failed_hdf5,
                source_demo_key=demo_key,
                repaired_demo_key=repaired_key,
                rollout=rollout,
                meta=meta,
            )
            written_keys.append(repaired_key)
            if method in ("v1f_pinn_top_k", "v1f_plain_top_k", "v1f_diverse_top_k"):
                append_successful_repair_demo(
                    output_hdf5=repaired_v1f_only,
                    source_hdf5=failed_hdf5,
                    source_demo_key=demo_key,
                    repaired_demo_key=repaired_key,
                    rollout=rollout,
                    meta=meta,
                )
        else:
            failures.append(_failure_record(source_demo=demo_key, method=method, candidate=cand, rollout=rollout))

    summary = summarize_method_results_v1f(rollout_results, method=method, rollout_budget=top_k)
    summary["repaired_demo_keys"] = written_keys
    return summary, failures, rollout_results


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline MimicGen Repair Test V1-F")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--v1f-model", type=Path, default=DEFAULT_V1F_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_V1F_OUTPUT_DIR)
    parser.add_argument("--demo-keys", default="demo_4,demo_2,demo_3")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--context-source",
        choices=CONTEXT_SOURCES,
        default="original_failed_context",
        help="PINN scoring context: original failed baseline vs CEM-refined baseline",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repaired_all = args.output_dir / "repaired_dataset_v1f.hdf5"
    repaired_v1f_only = args.output_dir / "repaired_pinn_v1f_only.hdf5"
    failures_path = args.output_dir / "failures_v1f.json"
    report_path = args.output_dir / "offline_mimicgen_repair_report_v1f.json"
    csv_path = args.output_dir / "v1e_vs_v1f_comparison.csv"

    demo_keys = [k.strip() for k in args.demo_keys.split(",") if k.strip()]
    env_check = check_environment()

    report: dict[str, Any] = {
        "task": "offline_mimicgen_repair_test_v1f",
        "description": "Offline MimicGen Repair Test — V1-E vs V1-F PINN repair layer",
        "failed_hdf5": str(args.failed_hdf5),
        "v1e_model": str(args.v1e_model),
        "v1f_model": str(args.v1f_model),
        "num_samples": args.num_samples,
        "rollout_budget": args.top_k,
        "context_source": args.context_source,
        "demo_keys": demo_keys,
        "selection_methods": list(V1F_SELECTION_METHODS),
        "environment_check": env_check,
    }

    if not env_check["available"]:
        report["blocked"] = True
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1
    if not args.v1e_model.exists() or not args.v1f_model.exists():
        report["blocked"] = True
        report["block_reason"] = ["missing v1e or v1f model"]
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    for p in (repaired_all, repaired_v1f_only):
        if p.exists():
            p.unlink()
    init_repaired_dataset(repaired_all, args.failed_hdf5)
    init_repaired_dataset(repaired_v1f_only, args.failed_hdf5)
    clear_repair_model_cache()
    clear_v1f_model_cache()
    rng = random.Random(args.seed)

    all_failures: list[dict[str, Any]] = []
    per_demo: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    failure_type_demos: dict[str, list[str]] = {}

    for demo_key in demo_keys:
        cfg = DEMO_REPAIR_CONFIGS[demo_key]
        print(f"offline repair v1f: {demo_key} ({cfg['failure_type']})")

        context = extract_repair_context_v1f(
            context_source=args.context_source,
            failed_hdf5=args.failed_hdf5,
            demo_key=demo_key,
            failure_type=cfg["failure_type"],
            search_kind=cfg["search_kind"],
            cem_report=args.cem_report,
        )
        candidates = sample_repair_candidates_v1f(
            search_kind=cfg["search_kind"],
            n_samples=args.num_samples,
            seed=args.seed + hash(demo_key) % 10000,
        )
        score_repair_candidates_v1f(
            context=context,
            candidates=candidates,
            active=cfg["active"],
            v1e_model_path=args.v1e_model,
            v1f_model_path=args.v1f_model,
        )

        cem = args.cem_report if cfg["search_kind"] == "insertion" else None
        method_summaries: dict[str, Any] = {}
        for method in V1F_SELECTION_METHODS:
            print(f"  rollout method={method}")
            summary, failures, _ = _run_method_for_demo(
                demo_key=demo_key,
                cfg=cfg,
                method=method,
                candidates=candidates,
                top_k=args.top_k,
                failed_hdf5=args.failed_hdf5,
                cem_report=cem,
                repaired_all=repaired_all,
                repaired_v1f_only=repaired_v1f_only,
                rng=rng,
            )
            method_summaries[method] = summary
            all_failures.extend(failures)
            row = {
                "demo_key": demo_key,
                "failure_type": cfg["failure_type"],
                "method": method,
                "repair_success_rate": summary["repair_success_rate"],
                "repair_rate_at_20": summary["repair_rate_at_20"],
                "rollouts_per_success": summary["rollouts_per_success"],
                "best_E_total": summary["best_E_total"],
                "success_at_1": summary["success_at_k"]["at_1"],
                "success_at_3": summary["success_at_k"]["at_3"],
                "success_at_5": summary["success_at_k"]["at_5"],
                "success_at_10": summary["success_at_k"]["at_10"],
                "success_at_20": summary["success_at_k"]["at_20"],
                "num_successes": summary["num_successes_written"],
            }
            comparison_rows.append(row)

        per_demo[demo_key] = {
            "failure_type": cfg["failure_type"],
            "failed_context": context,
            "context_source": args.context_source,
            "num_candidates_sampled": len(candidates),
            "methods": method_summaries,
        }
        ft = cfg["failure_type"]
        failure_type_demos.setdefault(ft, []).append(demo_key)

    report["per_demo"] = per_demo
    report["by_failure_type"] = {
        ft: {
            method: {
                "repair_success_rate_mean": float(
                    np.mean([per_demo[d]["methods"][method]["repair_success_rate"] for d in demos])
                ),
                "success_at_20_mean": float(
                    np.mean([per_demo[d]["methods"][method]["success_at_k"]["at_20"] for d in demos])
                ),
                "rollouts_per_success_mean": float(
                    np.mean([
                        per_demo[d]["methods"][method]["rollouts_per_success"]
                        for d in demos
                        if per_demo[d]["methods"][method]["rollouts_per_success"] < 1e8
                    ])
                    if any(per_demo[d]["methods"][method]["rollouts_per_success"] < 1e8 for d in demos)
                    else float("inf")
                ),
                "best_E_total_min": float(
                    min(per_demo[d]["methods"][method]["best_E_total"] for d in demos)
                ),
            }
            for method in V1F_SELECTION_METHODS
        }
        for ft, demos in failure_type_demos.items()
    }
    report["v1e_vs_v1f"] = {
        demo: {
            "v1e_success_at_20": per_demo[demo]["methods"]["v1e_pinn_top_k"]["success_at_k"]["at_20"],
            "v1f_success_at_20": per_demo[demo]["methods"]["v1f_pinn_top_k"]["success_at_k"]["at_20"],
            "v1e_rollouts_per_success": per_demo[demo]["methods"]["v1e_pinn_top_k"]["rollouts_per_success"],
            "v1f_rollouts_per_success": per_demo[demo]["methods"]["v1f_pinn_top_k"]["rollouts_per_success"],
            "v1e_repair_rate_at_20": per_demo[demo]["methods"]["v1e_pinn_top_k"]["repair_rate_at_20"],
            "v1f_repair_rate_at_20": per_demo[demo]["methods"]["v1f_pinn_top_k"]["repair_rate_at_20"],
        }
        for demo in demo_keys
    }

    failures_path.write_text(json.dumps({"failures": all_failures}, indent=2), encoding="utf-8")
    _write_comparison_csv(comparison_rows, csv_path)
    report["outputs"] = {
        "report_json": str(report_path),
        "repaired_dataset_v1f_hdf5": str(repaired_all),
        "repaired_pinn_v1f_only_hdf5": str(repaired_v1f_only),
        "failures_v1f_json": str(failures_path),
        "v1e_vs_v1f_comparison_csv": str(csv_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
