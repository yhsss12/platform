#!/usr/bin/env python3
"""V1-F-aligned-original 训练后验证：injected ranking + offline repair + 对比报告。"""
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

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
_V1_DIR = _EXPERIMENT_DIR / "v1_residual_model"
_V1E_DIR = _V1_DIR / "repair_parameter_model"
_V1F_DIR = _V1_DIR / "repair_parameter_model_v1f"
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_PINN_MODEL,
    DEFAULT_V1F_MODEL,
    DEMO_REPAIR_CONFIGS,
)
from osc_action_converter import SEARCH_SPACE  # noqa: E402
from pinn_v1f_inference import (  # noqa: E402
    build_v1f_features_from_repair_spec,
    clear_v1f_model_cache,
    load_v1f_repair_model,
    score_v1f_repair_candidate,
)
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

DEFAULT_ALIGNED_MODEL = (
    _EXPERIMENT_DIR
    / "outputs"
    / "v1f_aligned_repair_parameter_model"
    / "original_failed"
    / "trained_model"
    / "model_v1f_aligned_original.pt"
)
DEFAULT_ROLLOUT_JSONL = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "rollout_samples.jsonl"
DEFAULT_OUTPUT_DIR = (
    _EXPERIMENT_DIR / "outputs" / "v1f_aligned_repair_parameter_model" / "original_failed" / "validation"
)
OLD_OFFLINE_REPORT = _EXPERIMENT_DIR / "outputs" / "offline_mimicgen_repair_test_v1f" / "offline_mimicgen_repair_report_v1f.json"
CEM_ABLATION_CSV = _EXPERIMENT_DIR / "outputs" / "context_alignment_ablation" / "context_alignment_ablation_summary.csv"
DEMO_KEYS = ("demo_4", "demo_2", "demo_3")


def _normalize_insertion(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in SEARCH_SPACE:
        val = raw[key]
        if key in ("insertion_steps", "hold_steps", "pre_insert_pause", "release_shift"):
            out[key] = float(int(float(val)))
        else:
            out[key] = float(val)
    return out


def load_known_good_thetas(jsonl_path: Path, demo_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            sim = rollout.get("sim_params") or {}
            if not sim:
                continue
            rows.append({"insertion": _normalize_insertion(sim), "sampling_index": rollout.get("sampling_index")})
    return rows


def run_injected_ranking(
    *,
    demo_key: str,
    v1f_model: Path,
    context_source: str,
    failed_hdf5: Path,
    cem_report: Path,
    rollout_jsonl: Path,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    cfg = DEMO_REPAIR_CONFIGS[demo_key]
    pool_seed = seed + hash(demo_key) % 10000
    known_goods = load_known_good_thetas(rollout_jsonl, demo_key)
    if not known_goods:
        return {"demo_key": demo_key, "error": "no_known_good_theta", "known_goods": []}

    best_insertion = known_goods[0]["insertion"]

    rng = random.Random(pool_seed + 17)
    injected_random = num_samples - 1
    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"], n_samples=injected_random, seed=pool_seed + 101
    )
    target_index = rng.randrange(len(candidates) + 1)
    injected = {
        "index": -1,
        "insertion": best_insertion,
        "transport": None,
        "grasp_lift": None,
        "lift_extra": None,
    }
    candidates.insert(target_index, injected)
    for i, c in enumerate(candidates):
        c["index"] = i

    context = extract_repair_context_v1f(
        context_source=context_source,
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
        cem_report=cem_report,
    )
    clear_v1f_model_cache()
    score_repair_candidates_v1f(
        context=context,
        candidates=candidates,
        active=cfg["active"],
        v1e_model_path=DEFAULT_PINN_MODEL,
        v1f_model_path=v1f_model,
    )
    rank = rank_theta_by_score(candidates, score_key="v1f_E_total", target_score=candidates[target_index]["v1f_E_total"])
    return {
        "demo_key": demo_key,
        "context_source": context_source,
        "known_good_index": target_index,
        "known_good_rank": rank,
        "in_top_5": rank <= 5,
        "in_top_20": rank <= 20,
        "known_good_v1f_E_total": candidates[target_index]["v1f_E_total"],
        "num_known_good_in_pool": len(known_goods),
    }


def run_offline_repair_demo(
    *,
    demo_key: str,
    v1f_model: Path,
    context_source: str,
    selection_method: str,
    failed_hdf5: Path,
    cem_report: Path,
    num_samples: int,
    top_k: int,
    seed: int,
    v1e_model: Path,
) -> dict[str, Any]:
    cfg = DEMO_REPAIR_CONFIGS[demo_key]
    pool_seed = seed + hash(demo_key) % 10000
    rng = random.Random(seed)

    context = extract_repair_context_v1f(
        context_source=context_source,
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
        cem_report=cem_report,
    )
    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"], n_samples=num_samples, seed=pool_seed
    )
    clear_v1f_model_cache()
    score_repair_candidates_v1f(
        context=context,
        candidates=candidates,
        active=cfg["active"],
        v1e_model_path=v1e_model,
        v1f_model_path=v1f_model,
    )
    indices = select_candidate_indices_v1f(
        candidates, method=selection_method, top_k=top_k, rng=rng
    )
    diversity = top_k_diversity_stats(candidates, indices)
    rollout_results = []
    for idx in indices:
        rollout_results.append(
            run_repair_rollout(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                search_kind=cfg["search_kind"],
                cem_report=cem_report if cfg["search_kind"] == "insertion" else None,
                candidate=candidates[idx],
            )
        )
    metrics = summarize_method_results_v1f(rollout_results, method=selection_method, rollout_budget=top_k)
    note = ""
    if demo_key == "demo_3" and metrics["num_successes_written"] == 0:
        note = "no-positive-lift-candidate"
    return {
        "demo_key": demo_key,
        "failure_type": cfg["failure_type"],
        "context_source": context_source,
        "selection_method": selection_method,
        "model": str(v1f_model),
        "metrics": metrics,
        "top20_diversity": diversity,
        "note": note,
    }


def _load_old_offline_row(demo_key: str, method: str) -> dict[str, Any] | None:
    if not OLD_OFFLINE_REPORT.exists():
        return None
    report = json.loads(OLD_OFFLINE_REPORT.read_text(encoding="utf-8"))
    methods = report.get("per_demo", {}).get(demo_key, {}).get("methods", {})
    return methods.get(method)


def _load_cem_ablation_demo4() -> dict[str, Any] | None:
    if not CEM_ABLATION_CSV.exists():
        return None
    with CEM_ABLATION_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("context_source") == "cem_refined_context" and row.get("selection_method") == "v1f_plain_top_k":
                return row
    return None


def build_comparison_rows(
    *,
    aligned_offline: list[dict[str, Any]],
    injected_demo4: dict[str, Any],
    old_injected_rank: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cem4 = _load_cem_ablation_demo4()

    for demo_key in DEMO_KEYS:
        v1e = _load_old_offline_row(demo_key, "v1e_pinn_top_k")
        v1f_old = _load_old_offline_row(demo_key, "v1f_pinn_top_k")
        aligned = next((r for r in aligned_offline if r["demo_key"] == demo_key), None)

        def _pack(label: str, src: dict[str, Any] | None, extra: dict[str, Any] | None = None) -> None:
            if src is None:
                return
            m = src.get("metrics", src)
            sk = m.get("success_at_k", {})
            rows.append(
                {
                    "demo_key": demo_key,
                    "variant": label,
                    "repair_rate_at_20": m.get("repair_rate_at_20", m.get("repair_success_rate")),
                    "success_at_20": sk.get("at_20") if sk else m.get("success_at_20"),
                    "rollouts_per_success": m.get("rollouts_per_success"),
                    "best_E_total": m.get("best_E_total", m.get("best_E_total_norm")),
                    "num_successes": m.get("num_successes_written", m.get("num_successes")),
                    **(extra or {}),
                }
            )

        if v1e:
            rows.append(
                {
                    "demo_key": demo_key,
                    "variant": "V1-E",
                    "repair_rate_at_20": v1e.get("repair_rate_at_20"),
                    "success_at_20": v1e.get("success_at_k", {}).get("at_20"),
                    "rollouts_per_success": v1e.get("rollouts_per_success"),
                    "best_E_total": v1e.get("best_E_total"),
                    "num_successes": v1e.get("num_successes_written"),
                }
            )
        if v1f_old:
            rows.append(
                {
                    "demo_key": demo_key,
                    "variant": "V1-F_old_original_context",
                    "repair_rate_at_20": v1f_old.get("repair_rate_at_20"),
                    "success_at_20": v1f_old.get("success_at_k", {}).get("at_20"),
                    "rollouts_per_success": v1f_old.get("rollouts_per_success"),
                    "best_E_total": v1f_old.get("best_E_total"),
                    "num_successes": v1f_old.get("num_successes_written"),
                }
            )
        if demo_key == "demo_4" and cem4:
            rows.append(
                {
                    "demo_key": demo_key,
                    "variant": "V1-F_old_cem_refined_context",
                    "repair_rate_at_20": float(cem4.get("repair_rate_at_20", 0)),
                    "success_at_20": float(cem4.get("success_at_20", 0)),
                    "rollouts_per_success": float(cem4.get("rollouts_per_success", 0)),
                    "best_E_total": float(cem4.get("best_E_total", 0)),
                    "num_successes": float(cem4.get("num_successes", 0)),
                    "known_good_rank": float(cem4.get("best_known_good_rank", 0)),
                }
            )
        if aligned:
            m = aligned["metrics"]
            extra = {}
            if demo_key == "demo_4":
                extra["known_good_rank"] = injected_demo4.get("known_good_rank")
            if demo_key == "demo_3":
                extra["note"] = aligned.get("note", "")
            rows.append(
                {
                    "demo_key": demo_key,
                    "variant": "V1-F-aligned-original_original_context",
                    "repair_rate_at_20": m["repair_rate_at_20"],
                    "success_at_20": m["success_at_k"]["at_20"],
                    "rollouts_per_success": m["rollouts_per_success"],
                    "best_E_total": m["best_E_total"],
                    "num_successes": m["num_successes_written"],
                    **extra,
                }
            )

    if old_injected_rank is not None:
        rows.append(
            {
                "demo_key": "demo_4",
                "variant": "V1-F_old_injected_rank_reference",
                "known_good_rank": old_injected_rank,
            }
        )
    return rows


def evaluate_acceptance(
    *,
    injected_demo4: dict[str, Any],
    aligned_offline: list[dict[str, Any]],
    old_injected_rank: int,
) -> dict[str, Any]:
    demo4 = next(r for r in aligned_offline if r["demo_key"] == "demo_4")
    demo2 = next(r for r in aligned_offline if r["demo_key"] == "demo_2")
    demo3 = next(r for r in aligned_offline if r["demo_key"] == "demo_3")
    v1f_old_d4 = _load_old_offline_row("demo_4", "v1f_pinn_top_k") or {}
    v1f_old_d2 = _load_old_offline_row("demo_2", "v1f_pinn_top_k") or {}

    d4_rate = float(demo4["metrics"]["repair_rate_at_20"])
    d4_rank = int(injected_demo4.get("known_good_rank", 9999))
    d2_rate = float(demo2["metrics"]["repair_rate_at_20"])
    old_d2_rate = float(v1f_old_d2.get("repair_rate_at_20", 0))

    return {
        "demo_4_no_zero_success": d4_rate > 0.0,
        "demo_4_known_good_rank_better_than_old": d4_rank < old_injected_rank,
        "demo_4_known_good_in_top_20": d4_rank <= 20,
        "demo_4_known_good_in_top_5": d4_rank <= 5,
        "demo_2_not_worse_than_old_v1f": d2_rate >= old_d2_rate - 1e-9,
        "demo_3_marked_no_positive_lift": demo3.get("note") == "no-positive-lift-candidate"
        or demo3["metrics"]["num_successes_written"] == 0,
        "all_pass": (
            d4_rate > 0.0
            and d4_rank < old_injected_rank
            and d2_rate >= old_d2_rate - 1e-9
        ),
        "details": {
            "demo_4_repair_rate_at_20": d4_rate,
            "demo_4_known_good_rank": d4_rank,
            "old_injected_rank_reference": old_injected_rank,
            "old_v1f_demo_4_repair_rate_at_20": v1f_old_d4.get("repair_rate_at_20"),
            "demo_2_repair_rate_at_20": d2_rate,
            "old_v1f_demo_2_repair_rate_at_20": old_d2_rate,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-aligned-original post-train validation")
    parser.add_argument("--aligned-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--rollout-jsonl", type=Path, default=DEFAULT_ROLLOUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--old-injected-rank", type=int, default=112)
    args = parser.parse_args()

    if not args.aligned_model.exists():
        raise SystemExit(f"Aligned model not found: {args.aligned_model}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_v1f_repair_model(args.aligned_model)

    injected_demo4 = run_injected_ranking(
        demo_key="demo_4",
        v1f_model=args.aligned_model,
        context_source="original_failed_context",
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        rollout_jsonl=args.rollout_jsonl,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    aligned_offline: list[dict[str, Any]] = []
    for demo_key in DEMO_KEYS:
        print(f"offline repair aligned: {demo_key}", flush=True)
        aligned_offline.append(
            run_offline_repair_demo(
                demo_key=demo_key,
                v1f_model=args.aligned_model,
                context_source="original_failed_context",
                selection_method="v1f_plain_top_k",
                failed_hdf5=args.failed_hdf5,
                cem_report=args.cem_report,
                num_samples=args.num_samples,
                top_k=args.top_k,
                seed=args.seed,
                v1e_model=args.v1e_model,
            )
        )

    comparison_rows = build_comparison_rows(
        aligned_offline=aligned_offline,
        injected_demo4=injected_demo4,
        old_injected_rank=args.old_injected_rank,
    )
    acceptance = evaluate_acceptance(
        injected_demo4=injected_demo4,
        aligned_offline=aligned_offline,
        old_injected_rank=args.old_injected_rank,
    )

    report = {
        "task": "v1f_aligned_original_validation",
        "aligned_model": str(args.aligned_model),
        "context_source": "original_failed_context",
        "selection_method": "v1f_plain_top_k",
        "injected_ranking_demo_4": injected_demo4,
        "offline_repair": aligned_offline,
        "acceptance": acceptance,
    }
    report_path = args.output_dir / "v1f_aligned_validation_report.json"
    csv_path = args.output_dir / "v1f_aligned_comparison.csv"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if comparison_rows:
        fieldnames = sorted({k for row in comparison_rows for k in row})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(comparison_rows)

    print(json.dumps({"report": str(report_path), "acceptance": acceptance}, indent=2))
    return 0 if acceptance["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
