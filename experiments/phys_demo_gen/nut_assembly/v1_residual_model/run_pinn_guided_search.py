#!/usr/bin/env python3
"""V1-D 方法验证：random / explicit_energy / pinn_energy sim-in-loop 搜索对比。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

_V1_DIR = _EXPERIMENT_DIR / "v1_residual_model"
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))

from grasp_sim_search import run_grasp_search
from pinn_inference import clear_pinn_scoring_context, set_pinn_scoring_context
from robosuite_env_loader import check_environment
from sim_in_loop_refiner import load_best_theta, pick_best_candidate, run_sim_in_loop_search

DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_MODEL = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_pinn" / "model.pt"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_pinn"

SEARCH_MODES = {
    "random_search": "random",
    "explicit_energy_search": "explicit_energy",
    "pinn_energy_search": "pinn_energy",
}

DEMO_SEARCH_CONFIG = {
    "demo_4": {
        "search_fn": "insertion",
        "stage": "insertion",
        "needs_theta": True,
    },
    "demo_2": {
        "search_fn": "grasp",
        "stage": "grasp",
        "needs_theta": False,
    },
    "demo_3": {
        "search_fn": "grasp",
        "stage": "grasp",
        "needs_theta": False,
    },
}


def _pick_best_grasp(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda row: (
            not row.get("success_flag", False),
            not row.get("grasp_success_proxy", False),
            row.get("search_score", row.get("score", 1e9)),
        ),
    )


def _result_metrics(best: dict[str, Any], meta: dict[str, Any], *, scoring_mode: str) -> dict[str, Any]:
    return {
        "scoring_mode": scoring_mode,
        "success_rate": float(bool(best.get("success_flag"))),
        "best_E_total_norm": float(best.get("E_total_norm", 0.0)),
        "final_xy": float(best.get("final_nut_peg_xy", 0.0)),
        "final_z_diff": float(best.get("final_z_diff", 0.0)) if best.get("final_z_diff") is not None else None,
        "min_nut_peg_xy": float(best.get("min_nut_peg_xy", 0.0)),
        "grasp_success_proxy": bool(best.get("grasp_success_proxy", False)),
        "lift_success_proxy": bool(best.get("lift_success_proxy", False)),
        "nut_lift_delta": float(best.get("nut_lift_delta", 0.0)),
        "search_score": float(best.get("search_score", 0.0)),
        "eval_count_to_best": meta.get("eval_count_to_best"),
        "total_evals": meta.get("total_evals"),
        "outcome_label": best.get("outcome_label"),
        "failure_guess": best.get("failure_guess"),
    }


def _run_one_demo_search(
    *,
    demo_key: str,
    failed_hdf5: Path,
    cem_report: Path,
    model_path: Path,
    search_mode: str,
    max_evals: int,
    seed: int,
) -> dict[str, Any]:
    cfg = DEMO_SEARCH_CONFIG[demo_key]
    scoring_mode = SEARCH_MODES[search_mode]

    if scoring_mode == "pinn_energy":
        set_pinn_scoring_context(
            demo_key=demo_key,
            hdf5_path=str(failed_hdf5),
            stage=cfg["stage"],
            model_path=model_path,
        )
    else:
        clear_pinn_scoring_context()

    if cfg["search_fn"] == "insertion":
        theta = load_best_theta(cem_report, demo_key)
        candidates, meta = run_sim_in_loop_search(
            str(failed_hdf5),
            demo_key,
            "failed",
            theta,
            mode="random",
            max_evals=max_evals,
            seed=seed,
            scoring_mode=scoring_mode,
            record_videos=False,
        )
        best = pick_best_candidate(candidates, scoring_mode=scoring_mode)
    else:
        candidates, meta = run_grasp_search(
            str(failed_hdf5),
            demo_key,
            "failed",
            mode="random",
            max_evals=max_evals,
            seed=seed,
            scoring_mode=scoring_mode,
        )
        best = _pick_best_grasp(candidates)

    return _result_metrics(best, meta, scoring_mode=search_mode)


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-D PINN guided sim-in-loop search comparison")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-keys", default="demo_4,demo_2,demo_3")
    parser.add_argument("--max-evals", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demo_keys = [k.strip() for k in args.demo_keys.split(",") if k.strip()]
    env_check = check_environment()
    report: dict[str, Any] = {
        "task": "V1-D_PINN_guided_search",
        "failed_hdf5": str(args.failed_hdf5),
        "model": str(args.model),
        "demo_keys": demo_keys,
        "search_methods": list(SEARCH_MODES.keys()),
        "max_evals": args.max_evals,
        "seed": args.seed,
        "environment_check": env_check,
        "notes": [
            "Method validation for candidate ranking / refinement guidance only.",
            "demo_4 uses insertion sim-in-loop; demo_2/demo_3 use grasp sim-in-loop (V2-B4).",
            "Not a cross-task generalization claim.",
        ],
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        out_path = args.output_dir / "pinn_guided_search_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    if not args.model.exists():
        report["blocked"] = True
        report["block_reason"] = [f"PINN model missing: {args.model}"]
        out_path = args.output_dir / "pinn_guided_search_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    per_demo: dict[str, Any] = {}
    for demo_key in demo_keys:
        if demo_key not in DEMO_SEARCH_CONFIG:
            raise ValueError(f"unsupported demo_key: {demo_key}")
        demo_results: dict[str, Any] = {}
        for search_mode in SEARCH_MODES:
            print(f"guided search demo={demo_key} mode={search_mode}")
            demo_results[search_mode] = _run_one_demo_search(
                demo_key=demo_key,
                failed_hdf5=args.failed_hdf5,
                cem_report=args.cem_report,
                model_path=args.model,
                search_mode=search_mode,
                max_evals=args.max_evals,
                seed=args.seed,
            )
        per_demo[demo_key] = demo_results

    report["per_demo"] = per_demo
    report["summary"] = {
        demo_key: {
            "random_success": per_demo[demo_key]["random_search"]["success_rate"],
            "explicit_success": per_demo[demo_key]["explicit_energy_search"]["success_rate"],
            "pinn_success": per_demo[demo_key]["pinn_energy_search"]["success_rate"],
        }
        for demo_key in demo_keys
    }

    out_path = args.output_dir / "pinn_guided_search_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
