#!/usr/bin/env python3
"""V1-F demo_4 context alignment ablation：original vs CEM-refined context × plain vs diverse top-k。"""
from __future__ import annotations

import argparse
import csv
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
_V1F_DIR = _V1_DIR / "repair_parameter_model_v1f"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    CONTEXT_SOURCES,
    DEFAULT_CEM_REPORT,
    DEFAULT_CONTEXT_ALIGNMENT_OUTPUT_DIR,
    DEFAULT_FAILED_HDF5,
    DEFAULT_V1F_MODEL,
    DEMO_REPAIR_CONFIGS,
)
from osc_action_converter import SEARCH_SPACE  # noqa: E402
from pinn_v1f_inference import build_v1f_features_from_repair_spec, score_v1f_repair_candidate  # noqa: E402
from repair_common_v1f import (  # noqa: E402
    extract_repair_context_v1f,
    rank_theta_by_score,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
    summarize_method_results_v1f,
    top_k_diversity_stats,
)
from repair_rollout import run_repair_rollout  # noqa: E402

DEFAULT_ROLLOUT_JSONL = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "rollout_samples.jsonl"

ABLATION_ARMS = (
    ("original_failed_context", "v1f_plain_top_k"),
    ("original_failed_context", "v1f_diverse_top_k"),
    ("cem_refined_context", "v1f_plain_top_k"),
    ("cem_refined_context", "v1f_diverse_top_k"),
)


def _normalize_insertion_params(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in SEARCH_SPACE:
        val = raw[key]
        if key in ("insertion_steps", "hold_steps", "pre_insert_pause", "release_shift"):
            out[key] = float(int(float(val)))
        else:
            out[key] = float(val)
    return out


def load_known_good_thetas(jsonl_path: Path, demo_key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("demo_key") != demo_key:
                continue
            rollout = rec.get("rollout", {})
            if not rollout.get("success_flag"):
                continue
            sim_params = rollout.get("sim_params") or {}
            if not sim_params:
                continue
            records.append(
                {
                    "insertion": _normalize_insertion_params(sim_params),
                    "rollout_E_total_norm": float(rollout.get("E_total_norm", 0.0)),
                    "sampling_index": rollout.get("sampling_index"),
                }
            )
    return records


def _score_known_good_theta(
    *,
    context: dict[str, Any],
    insertion: dict[str, float],
    active: str,
    model_path: Path,
) -> float:
    features = build_v1f_features_from_repair_spec(
        context=context,
        insertion=insertion,
        transport=None,
        grasp_lift=None,
        lift_extra=None,
        active=active,
    )
    return float(score_v1f_repair_candidate(features, model_path=model_path)["v1f_E_total"])


def _known_good_rank_stats(
    *,
    candidates: list[dict[str, Any]],
    known_goods: list[dict[str, Any]],
    context: dict[str, Any],
    active: str,
    model_path: Path,
) -> dict[str, Any]:
    ranks: list[int] = []
    per_theta: list[dict[str, Any]] = []
    for i, item in enumerate(known_goods):
        score = _score_known_good_theta(
            context=context,
            insertion=item["insertion"],
            active=active,
            model_path=model_path,
        )
        rank = rank_theta_by_score(candidates, score_key="v1f_E_total", target_score=score)
        ranks.append(rank)
        per_theta.append(
            {
                "known_good_index": i,
                "sampling_index": item.get("sampling_index"),
                "predicted_E_total": score,
                "rank": rank,
                "in_top_20": rank <= 20,
            }
        )
    best = min(per_theta, key=lambda x: x["rank"])
    return {
        "num_known_good": len(known_goods),
        "best_known_good_rank": int(best["rank"]),
        "mean_known_good_rank": float(sum(ranks) / max(len(ranks), 1)),
        "num_known_good_in_top_20": int(sum(1 for r in ranks if r <= 20)),
        "per_theta": per_theta,
    }


def _run_ablation_arm(
    *,
    context_source: str,
    selection_method: str,
    demo_key: str,
    cfg: dict[str, Any],
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    known_goods: list[dict[str, Any]],
    failed_hdf5: Path,
    cem_report: Path,
    v1f_model: Path,
    top_k: int,
    rng: random.Random,
) -> dict[str, Any]:
    rank_stats = _known_good_rank_stats(
        candidates=candidates,
        known_goods=known_goods,
        context=context,
        active=cfg["active"],
        model_path=v1f_model,
    )
    indices = select_candidate_indices_v1f(
        candidates, method=selection_method, top_k=top_k, rng=rng
    )
    diversity = top_k_diversity_stats(candidates, indices)

    rollout_results: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices, start=1):
        rollout = run_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            search_kind=cfg["search_kind"],
            cem_report=cem_report,
            candidate=candidates[idx],
        )
        rollout_results.append(rollout)

    metrics = summarize_method_results_v1f(
        rollout_results, method=selection_method, rollout_budget=top_k
    )
    return {
        "context_source": context_source,
        "selection_method": selection_method,
        "context": context,
        "known_good_rank": rank_stats,
        "top20_diversity": diversity,
        "rollout_metrics": {
            "repair_rate_at_20": metrics["repair_rate_at_20"],
            "success_at_k": metrics["success_at_k"],
            "best_E_total": metrics["best_E_total"],
            "rollouts_per_success": metrics["rollouts_per_success"],
            "repair_success_rate": metrics["repair_success_rate"],
            "num_successes": metrics["num_successes_written"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F demo_4 context alignment ablation")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1f-model", type=Path, default=DEFAULT_V1F_MODEL)
    parser.add_argument("--rollout-jsonl", type=Path, default=DEFAULT_ROLLOUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONTEXT_ALIGNMENT_OUTPUT_DIR)
    parser.add_argument("--demo-key", default="demo_4")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demo_key = args.demo_key
    cfg = DEMO_REPAIR_CONFIGS[demo_key]
    pool_seed = args.seed + hash(demo_key) % 10000
    rng = random.Random(args.seed)

    known_goods = load_known_good_thetas(args.rollout_jsonl, demo_key)
    if not known_goods:
        raise SystemExit(f"No known-good theta for {demo_key} in {args.rollout_jsonl}")

    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"],
        n_samples=args.num_samples,
        seed=pool_seed,
    )

    arms: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for context_source, selection_method in ABLATION_ARMS:
        print(f"ablation: {context_source} + {selection_method}", flush=True)
        context = extract_repair_context_v1f(
            context_source=context_source,
            failed_hdf5=args.failed_hdf5,
            demo_key=demo_key,
            failure_type=cfg["failure_type"],
            search_kind=cfg["search_kind"],
            cem_report=args.cem_report,
        )
        score_repair_candidates_v1f(
            context=context,
            candidates=candidates,
            active=cfg["active"],
            v1e_model_path=_EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "model.pt",
            v1f_model_path=args.v1f_model,
        )
        arm = _run_ablation_arm(
            context_source=context_source,
            selection_method=selection_method,
            demo_key=demo_key,
            cfg=cfg,
            candidates=candidates,
            context=context,
            known_goods=known_goods,
            failed_hdf5=args.failed_hdf5,
            cem_report=args.cem_report,
            v1f_model=args.v1f_model,
            top_k=args.top_k,
            rng=rng,
        )
        arms.append(arm)
        m = arm["rollout_metrics"]
        sk = m["success_at_k"]
        summary_rows.append(
            {
                "demo_key": demo_key,
                "context_source": context_source,
                "selection_method": selection_method,
                "best_known_good_rank": arm["known_good_rank"]["best_known_good_rank"],
                "mean_known_good_rank": arm["known_good_rank"]["mean_known_good_rank"],
                "num_known_good_in_top_20": arm["known_good_rank"]["num_known_good_in_top_20"],
                "repair_rate_at_20": m["repair_rate_at_20"],
                "success_at_1": sk["at_1"],
                "success_at_3": sk["at_3"],
                "success_at_5": sk["at_5"],
                "success_at_10": sk["at_10"],
                "success_at_20": sk["at_20"],
                "best_E_total": m["best_E_total"],
                "rollouts_per_success": m["rollouts_per_success"],
                "top20_mean_pairwise_distance": arm["top20_diversity"]["mean_pairwise_param_distance"],
                "top20_min_pairwise_distance": arm["top20_diversity"]["min_pairwise_param_distance"],
                "num_successes": m["num_successes"],
            }
        )

    baseline = next(a for a in arms if a["context_source"] == "original_failed_context" and a["selection_method"] == "v1f_plain_top_k")
    best_rank_arm = min(arms, key=lambda a: a["known_good_rank"]["best_known_good_rank"])
    best_repair_arm = max(arms, key=lambda a: a["rollout_metrics"]["repair_rate_at_20"])

    cem_arms = [a for a in arms if a["context_source"] == "cem_refined_context"]
    orig_arms = [a for a in arms if a["context_source"] == "original_failed_context"]
    cem_best_rank = min(a["known_good_rank"]["best_known_good_rank"] for a in cem_arms)
    orig_best_rank = min(a["known_good_rank"]["best_known_good_rank"] for a in orig_arms)
    cem_best_repair = max(a["rollout_metrics"]["repair_rate_at_20"] for a in cem_arms)
    orig_best_repair = max(a["rollout_metrics"]["repair_rate_at_20"] for a in orig_arms)

    context_mismatch_confirmed = (
        cem_best_rank < orig_best_rank * 0.5
        or cem_best_repair > orig_best_repair + 0.05
        or cem_best_rank <= 20 <= orig_best_rank
    )

    report = {
        "task": "context_alignment_ablation",
        "demo_key": demo_key,
        "failure_type": cfg["failure_type"],
        "num_samples": args.num_samples,
        "rollout_budget": args.top_k,
        "seed": args.seed,
        "pool_seed": pool_seed,
        "context_sources": list(CONTEXT_SOURCES),
        "ablation_arms": arms,
        "acceptance": {
            "context_mismatch_confirmed": context_mismatch_confirmed,
            "baseline_original_plain": {
                "best_known_good_rank": baseline["known_good_rank"]["best_known_good_rank"],
                "repair_rate_at_20": baseline["rollout_metrics"]["repair_rate_at_20"],
            },
            "best_rank_arm": {
                "context_source": best_rank_arm["context_source"],
                "selection_method": best_rank_arm["selection_method"],
                "best_known_good_rank": best_rank_arm["known_good_rank"]["best_known_good_rank"],
            },
            "best_repair_arm": {
                "context_source": best_repair_arm["context_source"],
                "selection_method": best_repair_arm["selection_method"],
                "repair_rate_at_20": best_repair_arm["rollout_metrics"]["repair_rate_at_20"],
            },
            "cem_vs_original": {
                "cem_best_rank": cem_best_rank,
                "original_best_rank": orig_best_rank,
                "cem_best_repair_rate_at_20": cem_best_repair,
                "original_best_repair_rate_at_20": orig_best_repair,
            },
        },
        "outputs": {
            "report_json": str(args.output_dir / "context_alignment_ablation_report.json"),
            "summary_csv": str(args.output_dir / "context_alignment_ablation_summary.csv"),
        },
    }

    report_path = args.output_dir / "context_alignment_ablation_report.json"
    csv_path = args.output_dir / "context_alignment_ablation_summary.csv"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps({"report": str(report_path), "acceptance": report["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
