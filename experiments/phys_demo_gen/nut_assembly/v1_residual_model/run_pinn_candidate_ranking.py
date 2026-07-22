#!/usr/bin/env python3
"""V1-D 方法验证：PINN vs explicit energy 候选排序（V2-B2.6 / B3 / B4 样本）。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from group_split_utils import top_k_refined_success_rate
from pinn_inference import (
    clear_pinn_model_cache,
    explicit_energy_score,
    predict_pinn_outputs,
    rollout_row_to_features,
    set_pinn_scoring_context,
)
from residual_dataset import normalize_failure_type, normalize_outcome

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_pinn"
DEFAULT_MODEL = DEFAULT_OUTPUT / "model.pt"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _candidate_id(pool_id: str, row: dict[str, Any], index: int) -> str:
    for key in ("search_index", "rank", "method_id", "seed"):
        if row.get(key) is not None and row.get(key) != "":
            return f"{pool_id}:{key}={row.get(key)}"
    return f"{pool_id}:idx={index}"


def _refined_success_flag(row: dict[str, Any]) -> bool:
    outcome = row.get("outcome") or row.get("outcome_label") or ""
    if str(outcome) == "refined_success":
        return True
    return _as_bool(row.get("success_flag"))


def _failure_type(row: dict[str, Any]) -> str:
    for key in ("failure_type", "failure_reason", "failure_guess"):
        if row.get(key):
            return normalize_failure_type(str(row[key]))
    return normalize_failure_type("unknown_failed")


def _outcome(row: dict[str, Any]) -> str:
    outcome = row.get("outcome") or row.get("outcome_label")
    if outcome:
        return normalize_outcome(str(outcome), success_flag=_as_bool(row.get("success_flag")))
    return normalize_outcome("unknown_outcome", success_flag=_as_bool(row.get("success_flag")))


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_insertion_pools(outputs: Path, failed_hdf5: Path) -> list[dict[str, Any]]:
    pools: list[dict[str, Any]] = []

    sim_report_path = outputs / "sim_in_loop_refinement" / "sim_in_loop_refinement_report.json"
    if sim_report_path.exists():
        report = json.loads(sim_report_path.read_text(encoding="utf-8"))
        demo_key = report.get("demo_key", "demo_4")
        pools.append(
            {
                "pool_id": f"v2b25_insertion_{demo_key}",
                "source_task": "V2-B2.5",
                "demo_key": demo_key,
                "stage": "insertion",
                "hdf5_path": str(failed_hdf5),
                "candidates": list(report.get("top_10_candidates", [])),
            }
        )

    repeat_path = outputs / "sim_in_loop_repeatability" / "repeatability_report.json"
    if repeat_path.exists():
        report = json.loads(repeat_path.read_text(encoding="utf-8"))
        demo_key = report.get("demo_key", "demo_4")
        runs = []
        for row in report.get("runs", []):
            payload = dict(row)
            if payload.get("best_params") and not payload.get("sim_params"):
                payload["sim_params"] = json.loads(payload["best_params"])
            runs.append(payload)
        if runs:
            pools.append(
                {
                    "pool_id": f"v2b26_repeatability_{demo_key}",
                    "source_task": "V2-B2.6_repeatability",
                    "demo_key": demo_key,
                    "stage": "insertion",
                    "hdf5_path": str(failed_hdf5),
                    "candidates": runs,
                }
            )

    ablation_path = outputs / "sim_in_loop_ablation" / "ablation_report.json"
    if ablation_path.exists():
        report = json.loads(ablation_path.read_text(encoding="utf-8"))
        demo_key = report.get("demo_key", "demo_4")
        search_rows = []
        for method_id, row in (report.get("method_results") or {}).items():
            cfg = (report.get("methods") or {}).get(method_id, {})
            if cfg.get("needs_search"):
                payload = dict(row)
                payload["method_id"] = method_id
                search_rows.append(payload)
        if search_rows:
            pools.append(
                {
                    "pool_id": f"v2b26_ablation_search_{demo_key}",
                    "source_task": "V2-B2.6_ablation",
                    "demo_key": demo_key,
                    "stage": "insertion",
                    "hdf5_path": str(failed_hdf5),
                    "candidates": search_rows,
                }
            )

    return pools


def _load_transport_pools(outputs: Path, failed_hdf5: Path) -> list[dict[str, Any]]:
    pools: list[dict[str, Any]] = []
    report_path = outputs / "transport_refinement" / "transport_refinement_report.json"
    if not report_path.exists():
        return pools

    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_demo = report.get("per_demo", {})
    for demo_key, demo_result in per_demo.items():
        candidates = list(demo_result.get("top_10_candidates", []))
        if candidates:
            pools.append(
                {
                    "pool_id": f"v2b3_transport_{demo_key}",
                    "source_task": "V2-B3",
                    "demo_key": demo_key,
                    "stage": "transport",
                    "hdf5_path": str(failed_hdf5),
                    "candidates": candidates,
                }
            )

    for row in _load_csv_rows(outputs / "transport_refinement" / "top_candidates.csv"):
        demo_key = row["demo_key"]
        pool_id = f"v2b3_transport_csv_{demo_key}"
        target = next((p for p in pools if p["demo_key"] == demo_key and p["source_task"] == "V2-B3"), None)
        if target is None:
            target = {
                "pool_id": pool_id,
                "source_task": "V2-B3",
                "demo_key": demo_key,
                "stage": "transport",
                "hdf5_path": str(failed_hdf5),
                "candidates": [],
            }
            pools.append(target)
        target["candidates"].append(dict(row))

    return pools


def _load_grasp_pools(outputs: Path, failed_hdf5: Path) -> list[dict[str, Any]]:
    pools: list[dict[str, Any]] = []
    report_path = outputs / "grasp_refinement" / "grasp_refinement_report.json"
    if not report_path.exists():
        return pools

    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_demo = report.get("per_demo", {})
    for demo_key, demo_result in per_demo.items():
        candidates = list(demo_result.get("top_10_candidates", []))
        if candidates:
            pools.append(
                {
                    "pool_id": f"v2b4_grasp_{demo_key}",
                    "source_task": "V2-B4",
                    "demo_key": demo_key,
                    "stage": "grasp",
                    "hdf5_path": str(failed_hdf5),
                    "candidates": candidates,
                }
            )

    for row in _load_csv_rows(outputs / "grasp_refinement" / "top_candidates.csv"):
        demo_key = row["demo_key"]
        target = next((p for p in pools if p["demo_key"] == demo_key and p["source_task"] == "V2-B4"), None)
        if target is None:
            target = {
                "pool_id": f"v2b4_grasp_csv_{demo_key}",
                "source_task": "V2-B4",
                "demo_key": demo_key,
                "stage": "grasp",
                "hdf5_path": str(failed_hdf5),
                "candidates": [],
            }
            pools.append(target)
        target["candidates"].append(dict(row))

    return pools


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        key_parts = [
            row.get("rollout_kind"),
            row.get("search_index"),
            row.get("rank"),
            row.get("seed"),
            row.get("method_id"),
            row.get("E_total_norm"),
            row.get("final_nut_peg_xy"),
            row.get("nut_displacement_after_grasp"),
        ]
        key = json.dumps(key_parts, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _score_pool(
    pool: dict[str, Any],
    *,
    model_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    demo_key = pool["demo_key"]
    stage = pool["stage"]
    hdf5_path = pool["hdf5_path"]
    set_pinn_scoring_context(demo_key=demo_key, hdf5_path=hdf5_path, stage=stage, model_path=model_path)

    rows: list[dict[str, Any]] = []
    explicit_scores: list[float] = []
    pinn_scores: list[float] = []
    refined_flags: list[float] = []

    for index, candidate in enumerate(pool["candidates"]):
        row = dict(candidate)
        row["demo_key"] = demo_key
        row["pool_id"] = pool["pool_id"]
        row["source_task"] = pool["source_task"]
        row["candidate_id"] = _candidate_id(pool["pool_id"], row, index)

        explicit = float(row.get("E_total_norm", explicit_energy_score(row)))
        features = rollout_row_to_features(row, demo_key=demo_key, hdf5_path=hdf5_path, stage=stage)
        pinn = predict_pinn_outputs(features, model_path=model_path)["pinn_E_total"]

        refined = _refined_success_flag(row)
        row["explicit_E_total_norm"] = explicit
        row["pinn_predicted_E_total"] = pinn
        row["success_flag"] = _as_bool(row.get("success_flag"))
        row["outcome"] = _outcome(row)
        row["failure_type"] = _failure_type(row)
        row["refined_success_flag"] = refined
        row["grasp_success_proxy"] = _as_bool(row.get("grasp_success_proxy"))
        row["lift_success_proxy"] = _as_bool(row.get("lift_success_proxy"))

        rows.append(row)
        explicit_scores.append(explicit)
        pinn_scores.append(pinn)
        refined_flags.append(float(refined))

    explicit_arr = np.array(explicit_scores, dtype=float)
    pinn_arr = np.array(pinn_scores, dtype=float)
    refined_arr = np.array(refined_flags, dtype=float)

    spearman = None
    if len(explicit_arr) >= 2 and np.std(explicit_arr) > 1e-8 and np.std(pinn_arr) > 1e-8:
        spearman = float(spearmanr(explicit_arr, pinn_arr).correlation)

    pool_summary = {
        "pool_id": pool["pool_id"],
        "source_task": pool["source_task"],
        "demo_key": demo_key,
        "stage": stage,
        "num_candidates": len(rows),
        "explicit_ranking": {
            "top_1_refined_success_hit_rate": top_k_refined_success_rate(explicit_arr, refined_arr, 1),
            "top_3_refined_success_hit_rate": top_k_refined_success_rate(explicit_arr, refined_arr, 3),
            "top_5_refined_success_hit_rate": top_k_refined_success_rate(explicit_arr, refined_arr, 5),
        },
        "pinn_ranking": {
            "top_1_refined_success_hit_rate": top_k_refined_success_rate(pinn_arr, refined_arr, 1),
            "top_3_refined_success_hit_rate": top_k_refined_success_rate(pinn_arr, refined_arr, 3),
            "top_5_refined_success_hit_rate": top_k_refined_success_rate(pinn_arr, refined_arr, 5),
        },
        "ranking_spearman_explicit_vs_pinn": spearman,
        "num_refined_success_in_pool": int(np.sum(refined_arr > 0.5)),
    }
    return rows, pool_summary


def _macro_average(pools: list[dict[str, Any]], prefix: str, metric: str) -> float | None:
    values = [p[prefix][metric] for p in pools if p.get(prefix, {}).get(metric) is not None]
    return float(np.mean(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-D PINN candidate ranking validation")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"PINN model not found: {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _EXPERIMENT_DIR / "outputs"

    pools = _load_insertion_pools(outputs, args.failed_hdf5)
    pools.extend(_load_transport_pools(outputs, args.failed_hdf5))
    pools.extend(_load_grasp_pools(outputs, args.failed_hdf5))
    for pool in pools:
        pool["candidates"] = _dedupe_candidates(pool["candidates"])

    clear_pinn_model_cache()
    all_rows: list[dict[str, Any]] = []
    pool_summaries: list[dict[str, Any]] = []
    for pool in pools:
        if not pool["candidates"]:
            continue
        rows, summary = _score_pool(pool, model_path=args.model)
        all_rows.extend(rows)
        pool_summaries.append(summary)

    report = {
        "task": "V1-D_PINN_candidate_ranking",
        "model": str(args.model),
        "failed_hdf5": str(args.failed_hdf5),
        "candidate_sources": ["V2-B2.5", "V2-B2.6", "V2-B3", "V2-B4"],
        "num_pools": len(pool_summaries),
        "num_candidates_scored": len(all_rows),
        "pools": pool_summaries,
        "macro_average": {
            "explicit_top_1_refined_success_hit_rate": _macro_average(
                pool_summaries, "explicit_ranking", "top_1_refined_success_hit_rate"
            ),
            "explicit_top_3_refined_success_hit_rate": _macro_average(
                pool_summaries, "explicit_ranking", "top_3_refined_success_hit_rate"
            ),
            "explicit_top_5_refined_success_hit_rate": _macro_average(
                pool_summaries, "explicit_ranking", "top_5_refined_success_hit_rate"
            ),
            "pinn_top_1_refined_success_hit_rate": _macro_average(
                pool_summaries, "pinn_ranking", "top_1_refined_success_hit_rate"
            ),
            "pinn_top_3_refined_success_hit_rate": _macro_average(
                pool_summaries, "pinn_ranking", "top_3_refined_success_hit_rate"
            ),
            "pinn_top_5_refined_success_hit_rate": _macro_average(
                pool_summaries, "pinn_ranking", "top_5_refined_success_hit_rate"
            ),
        },
        "notes": [
            "Ranking validation on existing V2-B2.6/B3/B4 candidate pools only.",
            "Not a cross-task generalization claim; task-specific PINN-style residual model.",
            "Lower predicted / explicit energy is treated as better for top-k hit rate.",
        ],
    }

    report_path = args.output_dir / "ranking_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "ranking_predictions.csv"
    csv_fields = [
        "candidate_id",
        "pool_id",
        "source_task",
        "demo_key",
        "stage",
        "explicit_E_total_norm",
        "pinn_predicted_E_total",
        "success_flag",
        "outcome",
        "failure_type",
        "refined_success_flag",
        "grasp_success_proxy",
        "lift_success_proxy",
        "rollout_kind",
        "search_index",
        "rank",
        "method_id",
        "seed",
        "final_nut_peg_xy",
        "final_z_diff",
        "min_nut_peg_xy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps({"ranking_report": str(report_path), "macro_average": report["macro_average"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
