#!/usr/bin/env python3
"""V2-A：对 failed demos 运行 Residual-Guided CEM proxy refinement。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from cem_refiner import refine_trajectory_cem
from energy_model import score_candidate_trajectory
from trajectory_parameterization import load_all_proxies

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUCCESS = PROJECT_ROOT / "mnt/data/demo.hdf5"
DEFAULT_FAILED = PROJECT_ROOT / "mnt/data/demo_failed.hdf5"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "cem_refinement"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summary(result: dict[str, Any]) -> dict[str, Any]:
    before = result["components_before"]
    after = result["components_after"]
    residual_before = result["residual_before"]
    residual_after = result["residual_after"]
    return {
        "demo_key": result["demo_key"],
        "label": result["label"],
        "energy_before": result["energy_before"],
        "energy_after": result["energy_after"],
        "energy_drop_ratio": result["energy_drop_ratio"],
        "E_xy_norm_before": before["xy"],
        "E_xy_norm_after": after["xy"],
        "E_transport_norm_before": before["transport"],
        "E_transport_norm_after": after["transport"],
        "E_yaw_norm_before": before["yaw"],
        "E_yaw_norm_after": after["yaw"],
        "E_z_norm_before": before["z"],
        "E_z_norm_after": after["z"],
        "E_smooth_norm_before": before["smooth"],
        "E_smooth_norm_after": after["smooth"],
        "final_xy_before": residual_before["final_xy"],
        "final_xy_after": residual_after["final_xy"],
        "min_xy_before": residual_before["min_xy"],
        "min_xy_after": residual_after["min_xy"],
        "final_z_before": residual_before["final_z"],
        "final_z_after": residual_after["final_z"],
        "min_yaw_before": residual_before["min_yaw"],
        "min_yaw_after": residual_after["min_yaw"],
        "failure_type_before": result["failure_type_before"],
        "failure_type_after": result["failure_type_after"],
        "optimization_targets_before": ",".join(result["optimization_targets_before"]),
        "optimization_targets_after": ",".join(result["optimization_targets_after"]),
        "transport_xy_offset_x": result["best_theta"]["transport_xy_offset"][0],
        "transport_xy_offset_y": result["best_theta"]["transport_xy_offset"][1],
        "insert_z_offset": result["best_theta"]["insert_z_offset"],
        "align_yaw_offset": result["best_theta"]["align_yaw_offset"],
        "speed_scale": result["best_theta"]["speed_scale"],
    }


def run_cem_refinement(
    failed_path: str,
    output_dir: str,
    success_path: str | None = None,
    n_samples: int = 128,
    elite_frac: float = 0.1,
    num_iters: int = 5,
    seed: int = 0,
    run_success_sanity: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    failed_proxies = load_all_proxies(failed_path, "failed")
    results: list[dict[str, Any]] = []
    per_demo_histories: dict[str, Any] = {}

    for index, proxy in enumerate(failed_proxies):
        result = refine_trajectory_cem(
            proxy,
            score_fn=score_candidate_trajectory,
            n_samples=n_samples,
            elite_frac=elite_frac,
            num_iters=num_iters,
            seed=seed + index,
        )
        results.append(result)
        per_demo_histories[proxy.demo_key] = result["iteration_history"]

    sanity_results: list[dict[str, Any]] = []
    if run_success_sanity and success_path:
        success_proxies = load_all_proxies(success_path, "success")
        for index, proxy in enumerate(success_proxies[:2]):
            result = refine_trajectory_cem(
                proxy,
                score_fn=score_candidate_trajectory,
                n_samples=64,
                elite_frac=0.15,
                num_iters=3,
                seed=seed + 100 + index,
            )
            sanity_results.append(result)

    transport_results = [row for row in results if row["demo_key"] in {f"demo_{i}" for i in range(4)}]
    insertion_result = next(row for row in results if row["demo_key"] == "demo_4")

    def _mono_converge(histories: list[list[dict[str, Any]]]) -> bool:
        ok = True
        for history in histories:
            if len(history) < 2:
                continue
            best_series = [item["best_energy"] for item in history]
            ok = ok and best_series[-1] <= best_series[0] * 1.05
        return ok

    report = {
        "task": "Square_D0 / NutAssembly",
        "model_version": "V2-A_residual_guided_cem_proxy",
        "inputs": {"failed_hdf5": failed_path, "success_hdf5": success_path},
        "cem_config": {
            "n_samples": n_samples,
            "elite_frac": elite_frac,
            "num_iters": num_iters,
            "seed": seed,
        },
        "results": results,
        "sanity_check_success": sanity_results,
        "acceptance_checks": {
            "transport_demo_0_to_3_drop_50pct": bool(
                all(row["energy_drop_ratio"] >= 0.50 for row in transport_results)
            ),
            "transport_xy_transport_norm_drop": bool(
                all(
                    row["components_after"]["xy"] < row["components_before"]["xy"]
                    and row["components_after"]["transport"] < row["components_before"]["transport"]
                    for row in transport_results
                )
            ),
            "transport_targets_include_xy": bool(
                all(
                    "transport_xy_offset" in row["optimization_targets_before"]
                    or "pre_align_pose" in row["optimization_targets_before"]
                    for row in transport_results
                )
            ),
            "demo_4_drop_30pct": bool(insertion_result["energy_drop_ratio"] >= 0.30),
            "demo_4_z_norm_drop": bool(
                insertion_result["components_after"]["z"] < insertion_result["components_before"]["z"]
            ),
            "demo_4_outcome_improved": insertion_result["failure_type_after"]
            in {"candidate_ready", "lower_energy_candidate"},
            "cem_converging": bool(_mono_converge([row["iteration_history"] for row in results])),
            "success_sanity_no_large_regression": bool(
                all(row["energy_after"] <= row["energy_before"] * 1.15 for row in sanity_results)
            )
            if sanity_results
            else True,
        },
    }

    summary_rows = [_flatten_summary(row) for row in results]

    report_path = output / "cem_refinement_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=_json_default)

    summary_path = output / "cem_refinement_summary.csv"
    _write_csv(summary_path, summary_rows)

    history_path = output / "per_demo_iteration_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(per_demo_histories, handle, indent=2, ensure_ascii=False, default=_json_default)

    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {history_path}")
    print()
    print("Acceptance checks:")
    for name, passed in report["acceptance_checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print()
    for row in summary_rows:
        print(
            f"{row['demo_key']}: E_norm {row['energy_before']:.2f} -> {row['energy_after']:.2f} "
            f"({row['energy_drop_ratio']*100:.1f}% drop), {row['failure_type_before']} -> {row['failure_type_after']}"
        )

    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nut Assembly V2-A CEM proxy refinement")
    parser.add_argument("--failed", default=str(DEFAULT_FAILED))
    parser.add_argument("--success", default=str(DEFAULT_SUCCESS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-success-sanity", action="store_true")
    args = parser.parse_args()

    run_cem_refinement(
        failed_path=args.failed,
        output_dir=args.output_dir,
        success_path=args.success,
        n_samples=args.n_samples,
        elite_frac=args.elite_frac,
        num_iters=args.num_iters,
        seed=args.seed,
        run_success_sanity=args.run_success_sanity,
    )


if __name__ == "__main__":
    main()
