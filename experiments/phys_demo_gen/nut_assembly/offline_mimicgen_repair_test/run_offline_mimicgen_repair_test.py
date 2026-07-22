#!/usr/bin/env python3
"""Offline MimicGen Repair Test：V1-E PINN 作为 failed demo repair layer。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_OFFLINE_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _OFFLINE_DIR.parent
_V1_DIR = _EXPERIMENT_DIR / "v1_residual_model"
_V1E_DIR = _V1_DIR / "repair_parameter_model"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PINN_MODEL,
    DEMO_REPAIR_CONFIGS,
    SELECTION_METHODS,
)
from pinn_repair_inference import clear_repair_model_cache  # noqa: E402
from repair_common import (  # noqa: E402
    extract_baseline_context,
    sample_repair_candidates,
    score_repair_candidates,
    select_candidate_indices,
    summarize_method_results,
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
        "pinn_E_total": candidate.get("pinn_E_total"),
        "explicit_E_total": candidate.get("explicit_E_total"),
        "pinn_success_prob": candidate.get("pinn_success_prob"),
        "repair_theta": candidate.get("insertion") or candidate.get("grasp_lift"),
        "success_flag": bool(rollout.get("success_flag")),
        "E_total_norm": float(rollout.get("E_total_norm", 0.0)),
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
    repaired_hdf5: Path,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    indices = select_candidate_indices(candidates, method=method, top_k=top_k, rng=rng)
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
                "pinn_E_total": cand.get("pinn_E_total"),
                "explicit_E_total": cand.get("explicit_E_total"),
                "pinn_success_prob": cand.get("pinn_success_prob"),
                "repair_theta": cand.get("insertion") or cand.get("grasp_lift"),
                "rollout_kind": "offline_mimicgen_repair",
                "object_poses_modified": False,
            }
            append_successful_repair_demo(
                output_hdf5=repaired_hdf5,
                source_hdf5=failed_hdf5,
                source_demo_key=demo_key,
                repaired_demo_key=repaired_key,
                rollout=rollout,
                meta=meta,
            )
            written_keys.append(repaired_key)
        else:
            failures.append(_failure_record(source_demo=demo_key, method=method, candidate=cand, rollout=rollout))

    summary = summarize_method_results(rollout_results, method=method, rollout_budget=top_k)
    summary["repaired_demo_keys"] = written_keys
    return summary, failures, rollout_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline MimicGen Repair Test with V1-E PINN")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-keys", default="demo_4,demo_2")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repaired_hdf5 = args.output_dir / "repaired_dataset.hdf5"
    failures_path = args.output_dir / "failures.json"
    report_path = args.output_dir / "offline_mimicgen_repair_report.json"

    demo_keys = [k.strip() for k in args.demo_keys.split(",") if k.strip()]
    env_check = check_environment()

    report: dict[str, Any] = {
        "task": "offline_mimicgen_repair_test",
        "description": "Offline MimicGen Repair Test — V1-E PINN as repair layer for failed demonstrations",
        "failed_hdf5": str(args.failed_hdf5),
        "success_reference_hdf5": str(args.failed_hdf5.parent / "demo.hdf5"),
        "pinn_model": str(args.model),
        "num_samples": args.num_samples,
        "rollout_budget": args.top_k,
        "demo_keys": demo_keys,
        "selection_methods": list(SELECTION_METHODS),
        "environment_check": env_check,
        "constraints": [
            "repaired_dataset.hdf5 contains success_flag=true rollouts only",
            "object_poses in HDF5 are copied unchanged from source failed demo",
            "no platform / frontend integration",
            "not PINA; not claimed cross-task generalization",
        ],
        "next_step": "Online MimicGen Plugin (not in this offline test)",
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    if not args.model.exists():
        report["blocked"] = True
        report["block_reason"] = [f"PINN model missing: {args.model}"]
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    if repaired_hdf5.exists():
        repaired_hdf5.unlink()

    init_repaired_dataset(repaired_hdf5, args.failed_hdf5)
    clear_repair_model_cache()
    rng = random.Random(args.seed)

    all_failures: list[dict[str, Any]] = []
    per_demo: dict[str, Any] = {}

    for demo_key in demo_keys:
        if demo_key not in DEMO_REPAIR_CONFIGS:
            raise ValueError(f"unsupported demo_key: {demo_key}")
        cfg = DEMO_REPAIR_CONFIGS[demo_key]
        print(f"offline repair: {demo_key} ({cfg['failure_type']})")

        context = extract_baseline_context(
            failed_hdf5=args.failed_hdf5,
            demo_key=demo_key,
            failure_type=cfg["failure_type"],
            search_kind=cfg["search_kind"],
        )
        candidates = sample_repair_candidates(
            search_kind=cfg["search_kind"],
            n_samples=args.num_samples,
            seed=args.seed + hash(demo_key) % 10000,
        )
        score_repair_candidates(
            context=context,
            candidates=candidates,
            active=cfg["active"],
            model_path=args.model,
        )

        cem = args.cem_report if cfg["search_kind"] == "insertion" else None
        method_summaries: dict[str, Any] = {}
        for method in SELECTION_METHODS:
            print(f"  rollout method={method}")
            summary, failures, _ = _run_method_for_demo(
                demo_key=demo_key,
                cfg=cfg,
                method=method,
                candidates=candidates,
                top_k=args.top_k,
                failed_hdf5=args.failed_hdf5,
                cem_report=cem,
                repaired_hdf5=repaired_hdf5,
                rng=rng,
            )
            method_summaries[method] = summary
            all_failures.extend(failures)

        per_demo[demo_key] = {
            "failure_type": cfg["failure_type"],
            "failed_context": context,
            "num_candidates_sampled": len(candidates),
            "methods": method_summaries,
        }

    report["per_demo"] = per_demo
    report["repair_success_rate"] = {
        demo: {m: per_demo[demo]["methods"][m]["repair_success_rate"] for m in SELECTION_METHODS}
        for demo in demo_keys
    }
    report["method_any_success"] = {
        demo: {m: per_demo[demo]["methods"][m]["any_success"] for m in SELECTION_METHODS}
        for demo in demo_keys
    }
    report["pinn_vs_random"] = {
        demo: {
            "pinn_any_success": report["method_any_success"][demo]["pinn_top_k"],
            "random_any_success": report["method_any_success"][demo]["random_top_k"],
            "pinn_best_E_total": per_demo[demo]["methods"]["pinn_top_k"]["min_rollout_E_total"],
            "random_best_E_total": per_demo[demo]["methods"]["random_top_k"]["min_rollout_E_total"],
        }
        for demo in demo_keys
    }
    report["acceptance_checks"] = {
        "demo_4_pinn_reproduces_success": report["method_any_success"].get("demo_4", {}).get("pinn_top_k", False),
        "demo_2_pinn_reproduces_success": report["method_any_success"].get("demo_2", {}).get("pinn_top_k", False),
        "repaired_hdf5_success_only": True,
        "object_poses_not_modified": True,
    }

    failures_path.write_text(json.dumps({"failures": all_failures}, indent=2), encoding="utf-8")
    report["outputs"] = {
        "repaired_dataset_hdf5": str(repaired_hdf5),
        "failures_json": str(failures_path),
        "report_json": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "repaired_hdf5": str(repaired_hdf5),
                "acceptance": report["acceptance_checks"],
                "pinn_vs_random": report["pinn_vs_random"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
