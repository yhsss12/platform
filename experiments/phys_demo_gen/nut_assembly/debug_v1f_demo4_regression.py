#!/usr/bin/env python3
"""V1-F demo_4 regression audit：定位 offline ranking vs repair rollout 差异。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parent
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
    DEFAULT_V1F_MODEL,
    DEMO_REPAIR_CONFIGS,
)
from osc_action_converter import SEARCH_SPACE, SimLoopParams  # noqa: E402
from pinn_v1f_inference import (  # noqa: E402
    build_v1f_features_from_repair_spec,
    load_v1f_repair_model,
    score_v1f_repair_candidate,
)
from repair_common_v1f import (  # noqa: E402
    compute_ranking_metrics,
    extract_baseline_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
)
from repair_rollout import run_repair_rollout  # noqa: E402
from sim_in_loop_refiner import load_best_theta, run_refined_waypoint_rollout  # noqa: E402
from v1f_repair_dataset import extract_failed_context, infer_coarse_failure_mode  # noqa: E402

DEFAULT_ROLLOUT_JSONL = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "rollout_samples.jsonl"
DEFAULT_OUTPUT_DIR = _EXPERIMENT_DIR / "outputs" / "debug_v1f_demo4_regression"

INSERTION_KEYS = list(SEARCH_SPACE.keys())
DEMO_KEY = "demo_4"
SCORING_METHODS = (
    "predicted_E_total",
    "neg_success_prob",
    "E_minus_alpha_success",
    "E_plus_beta_uncertainty",
    "E_minus_alpha_success_plus_beta_uncertainty",
)


def _normalize_insertion_params(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in INSERTION_KEYS:
        val = raw[key]
        if key in ("insertion_steps", "hold_steps", "pre_insert_pause", "release_shift"):
            out[key] = float(int(float(val)))
        else:
            out[key] = float(val)
    return out


def _candidate_from_insertion(index: int, insertion: dict[str, float]) -> dict[str, Any]:
    return {
        "index": index,
        "insertion": insertion,
        "transport": None,
        "grasp_lift": None,
        "lift_extra": None,
    }


def load_known_good_thetas(jsonl_path: Path, *, demo_key: str = DEMO_KEY) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("demo_key") != demo_key:
                continue
            rollout = rec.get("rollout", {})
            if not rollout.get("success_flag"):
                continue
            sim_params = rollout.get("sim_params") or rollout.get("repair_insertion_params") or {}
            if not sim_params:
                continue
            insertion = _normalize_insertion_params(sim_params)
            records.append(
                {
                    "line_no": line_no,
                    "sampling_index": rollout.get("sampling_index"),
                    "insertion": insertion,
                    "training_context": rec.get("context", {}),
                    "rollout_E_total_norm": float(rollout.get("E_total_norm", 0.0)),
                    "rollout_final_xy": float(rollout.get("final_nut_peg_xy", 0.0)),
                }
            )
    return records


def build_training_context(failed_hdf5: Path, cem_report: Path, demo_key: str) -> dict[str, Any]:
    theta = load_best_theta(cem_report, demo_key)
    baseline = run_refined_waypoint_rollout(
        str(failed_hdf5),
        demo_key,
        "failed",
        theta,
        sim_params=SimLoopParams(),
    )
    return extract_failed_context(
        baseline,
        demo_key=demo_key,
        failure_type=infer_coarse_failure_mode(demo_key=demo_key),
    )


def replay_known_good_thetas(
    *,
    known_goods: list[dict[str, Any]],
    failed_hdf5: Path,
    cem_report: Path,
    demo_key: str,
    search_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(known_goods):
        candidate = _candidate_from_insertion(i, item["insertion"])
        rollout = run_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            search_kind=search_kind,
            cem_report=cem_report,
            candidate=candidate,
        )
        rows.append(
            {
                "known_good_index": i,
                "sampling_index": item.get("sampling_index"),
                "insertion": item["insertion"],
                "original_success_in_jsonl": True,
                "replay_success_flag": bool(rollout.get("success_flag")),
                "replay_E_total_norm": float(rollout.get("E_total_norm", 0.0)),
                "replay_final_nut_peg_xy": float(rollout.get("final_nut_peg_xy", 0.0)),
                "jsonl_E_total_norm": item.get("rollout_E_total_norm"),
            }
        )
    return rows


def _param_vector(insertion: dict[str, float]) -> np.ndarray:
    vec = []
    for key in INSERTION_KEYS:
        choices = SEARCH_SPACE[key]
        val = insertion[key]
        if len(choices) <= 1:
            vec.append(0.0)
            continue
        lo = float(min(choices))
        hi = float(max(choices))
        vec.append((float(val) - lo) / max(hi - lo, 1e-9))
    return np.asarray(vec, dtype=float)


def _pairwise_param_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return float(np.linalg.norm(_param_vector(a) - _param_vector(b)))


def _attach_v1f_scores(
    *,
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    active: str,
    model_path: Path,
    alpha: float,
    beta: float,
) -> None:
    load_v1f_repair_model(model_path)
    for cand in candidates:
        features = build_v1f_features_from_repair_spec(
            context=context,
            insertion=cand.get("insertion"),
            transport=cand.get("transport"),
            grasp_lift=cand.get("grasp_lift"),
            lift_extra=cand.get("lift_extra"),
            active=active,
        )
        scores = score_v1f_repair_candidate(features, model_path=model_path)
        cand["v1f_features"] = features
        cand["predicted_E_total"] = scores["v1f_E_total"]
        cand["v1f_success_prob"] = scores["v1f_success_prob"]
        cand["v1f_uncertainty"] = scores["v1f_uncertainty"]
        cand["neg_success_prob"] = -scores["v1f_success_prob"]
        cand["E_minus_alpha_success"] = scores["v1f_E_total"] - alpha * scores["v1f_success_prob"]
        cand["E_plus_beta_uncertainty"] = scores["v1f_E_total"] + beta * scores["v1f_uncertainty"]
        cand["E_minus_alpha_success_plus_beta_uncertainty"] = (
            scores["v1f_E_total"] - alpha * scores["v1f_success_prob"] + beta * scores["v1f_uncertainty"]
        )


def _rank_of_target(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    target_index: int,
) -> int:
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][score_key])
    return int(order.index(target_index) + 1)


def _top_k_indices(candidates: list[dict[str, Any]], *, score_key: str, top_k: int) -> list[int]:
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][score_key])
    return order[: min(top_k, len(candidates))]


def diverse_top_k_indices(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    top_k: int,
) -> list[int]:
    if not candidates:
        return []
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][score_key])
    selected = [order[0]]
    remaining = order[1:]
    while len(selected) < min(top_k, len(candidates)) and remaining:
        best_idx = remaining[0]
        best_min_dist = -1.0
        for idx in remaining:
            ins = candidates[idx]["insertion"]
            min_dist = min(
                _pairwise_param_distance(ins, candidates[j]["insertion"]) for j in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _top_k_diversity_stats(candidates: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {
            "num_selected": 0,
            "mean_pairwise_param_distance": 0.0,
            "min_pairwise_param_distance": 0.0,
            "unique_values_per_key": {},
        }
    insertions = [candidates[i]["insertion"] for i in indices]
    distances: list[float] = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            distances.append(_pairwise_param_distance(insertions[a], insertions[b]))
    unique_per_key = {
        key: len({ins[key] for ins in insertions}) for key in INSERTION_KEYS
    }
    return {
        "num_selected": len(indices),
        "mean_pairwise_param_distance": float(np.mean(distances)) if distances else 0.0,
        "min_pairwise_param_distance": float(np.min(distances)) if distances else 0.0,
        "unique_values_per_key": unique_per_key,
    }


def _rollout_top_k(
    *,
    candidates: list[dict[str, Any]],
    indices: list[int],
    failed_hdf5: Path,
    cem_report: Path,
    demo_key: str,
    search_kind: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices, start=1):
        rollout = run_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            search_kind=search_kind,
            cem_report=cem_report,
            candidate=candidates[idx],
        )
        results.append(
            {
                "rank": rank,
                "candidate_index": idx,
                "success_flag": bool(rollout.get("success_flag")),
                "E_total_norm": float(rollout.get("E_total_norm", 0.0)),
                "final_nut_peg_xy": float(rollout.get("final_nut_peg_xy", 0.0)),
                "insertion": candidates[idx]["insertion"],
            }
        )
    return results


def _distance_to_known_good(top_insertions: list[dict[str, float]], known_good: dict[str, float]) -> dict[str, float]:
    if not top_insertions:
        return {"mean_distance": 0.0, "min_distance": 0.0, "max_distance": 0.0}
    dists = [_pairwise_param_distance(ins, known_good) for ins in top_insertions]
    return {
        "mean_distance": float(np.mean(dists)),
        "min_distance": float(np.min(dists)),
        "max_distance": float(np.max(dists)),
    }


def classify_regression_root_cause(report: dict[str, Any]) -> str:
    replay_rate = float(report["replay_audit"]["replay_success_rate_known_good_theta"])
    injected_ranks = report["injected_pool_audit"]["known_good_rank_by_scoring"]
    min_rank = min(int(v) for v in injected_ranks.values())
    real_top20 = report["real_pool_audit"]["top20_rollout"]["v1f_pinn_top_k"]
    real_success_rate = float(real_top20.get("repair_rate_at_20", 0.0))
    pool_rate = float(report["real_pool_audit"].get("empirical_success_rate_from_jsonl", 0.0))

    if replay_rate < 0.5:
        return "A"
    if min_rank > 20:
        return "B"
    if min_rank <= 20 and real_success_rate == 0.0:
        return "C"
    if pool_rate < 0.03:
        return "D"
    return "B"


def _diagnosis_text(code: str) -> str:
    mapping = {
        "A": "known-good θ 无法复现 → rollout/refiner 不一致",
        "B": "known-good θ 可复现但 rank 很低 → scoring/特征/归一化问题",
        "C": "known-good θ rank 很高但 top-20 仍失败 → rollout 方差/多样性问题",
        "D": "candidate pool 成功率过低 → sampler 问题",
    }
    return mapping[code]


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F demo_4 regression audit")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1f-model", type=Path, default=DEFAULT_V1F_MODEL)
    parser.add_argument("--rollout-jsonl", type=Path, default=DEFAULT_ROLLOUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-key", default=DEMO_KEY)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--injected-random", type=int, default=999)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--skip-topk-rollouts", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DEMO_REPAIR_CONFIGS[args.demo_key]
    active = cfg["active"]
    search_kind = cfg["search_kind"]
    pool_seed = args.seed + hash(args.demo_key) % 10000

    known_goods = load_known_good_thetas(args.rollout_jsonl, demo_key=args.demo_key)
    if not known_goods:
        raise SystemExit(f"No known-good theta found for {args.demo_key} in {args.rollout_jsonl}")

    # Pick best-known θ by lowest jsonl E_total for injected-pool experiments.
    best_known = min(known_goods, key=lambda x: x.get("rollout_E_total_norm", 1e9))
    best_insertion = best_known["insertion"]

    offline_context = extract_baseline_context_v1f(
        failed_hdf5=args.failed_hdf5,
        demo_key=args.demo_key,
        failure_type=cfg["failure_type"],
        search_kind=search_kind,
    )
    training_context = build_training_context(args.failed_hdf5, args.cem_report, args.demo_key)

    replay_rows = replay_known_good_thetas(
        known_goods=known_goods,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        demo_key=args.demo_key,
        search_kind=search_kind,
    )
    replay_success_rate = float(
        sum(1 for r in replay_rows if r["replay_success_flag"]) / max(len(replay_rows), 1)
    )

    # Injected pool: 999 random + 1 known-good at a random slot.
    rng = random.Random(pool_seed + 17)
    injected_candidates = sample_repair_candidates_v1f(
        search_kind=search_kind,
        n_samples=args.injected_random,
        seed=pool_seed + 101,
    )
    injected_target_index = rng.randrange(len(injected_candidates) + 1)
    injected_known = _candidate_from_insertion(-1, best_insertion)
    injected_candidates.insert(injected_target_index, injected_known)
    for i, cand in enumerate(injected_candidates):
        cand["index"] = i

    _attach_v1f_scores(
        candidates=injected_candidates,
        context=offline_context,
        active=active,
        model_path=args.v1f_model,
        alpha=args.alpha,
        beta=args.beta,
    )

    injected_rank_by_scoring = {
        method: _rank_of_target(injected_candidates, score_key=method, target_index=injected_target_index)
        for method in SCORING_METHODS
    }

    injected_top20_stats: dict[str, Any] = {}
    for method in SCORING_METHODS:
        top_idx = _top_k_indices(injected_candidates, score_key=method, top_k=args.top_k)
        insertions = [injected_candidates[i]["insertion"] for i in top_idx]
        stats = _top_k_diversity_stats(injected_candidates, top_idx)
        stats["known_good_in_top_k"] = injected_target_index in top_idx
        stats["distance_to_known_good"] = _distance_to_known_good(insertions, best_insertion)
        injected_top20_stats[method] = stats

    # Context shift audit on the same known-good θ.
    context_shift_rows = []
    for label, context in (("offline_context", offline_context), ("training_context", training_context)):
        cand = _candidate_from_insertion(0, best_insertion)
        _attach_v1f_scores(
            candidates=[cand],
            context=context,
            active=active,
            model_path=args.v1f_model,
            alpha=args.alpha,
            beta=args.beta,
        )
        context_shift_rows.append(
            {
                "context_kind": label,
                "context": context,
                "predicted_E_total": cand["predicted_E_total"],
                "v1f_success_prob": cand["v1f_success_prob"],
                "v1f_uncertainty": cand["v1f_uncertainty"],
            }
        )

    # Real offline pool (same seed as repair test).
    real_candidates = sample_repair_candidates_v1f(
        search_kind=search_kind,
        n_samples=args.pool_size,
        seed=pool_seed,
    )
    score_repair_candidates_v1f(
        context=offline_context,
        candidates=real_candidates,
        active=active,
        v1e_model_path=_EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "model.pt",
        v1f_model_path=args.v1f_model,
    )

    real_top20_rollout: dict[str, Any] = {}
    if not args.skip_topk_rollouts:
        for method_name, score_key in (
            ("v1f_pinn_top_k", "v1f_E_total"),
            ("diverse_v1f_top_k", "v1f_E_total"),
        ):
            if method_name == "diverse_v1f_top_k":
                indices = diverse_top_k_indices(real_candidates, score_key=score_key, top_k=args.top_k)
            else:
                indices = select_candidate_indices_v1f(
                    real_candidates, method="v1f_pinn_top_k", top_k=args.top_k, rng=random.Random(args.seed)
                )
            rollout_rows = _rollout_top_k(
                candidates=real_candidates,
                indices=indices,
                failed_hdf5=args.failed_hdf5,
                cem_report=args.cem_report,
                demo_key=args.demo_key,
                search_kind=search_kind,
            )
            metrics = compute_ranking_metrics(rollout_rows)
            diversity = _top_k_diversity_stats(real_candidates, indices)
            real_top20_rollout[method_name] = {
                **metrics,
                "diversity": diversity,
                "rollouts": rollout_rows,
            }

    real_v1f_top_idx = select_candidate_indices_v1f(
        real_candidates, method="v1f_pinn_top_k", top_k=args.top_k, rng=random.Random(args.seed)
    )
    diverse_idx = diverse_top_k_indices(real_candidates, score_key="v1f_E_total", top_k=args.top_k)
    real_top20_compare = {
        "plain_top_k": _top_k_diversity_stats(real_candidates, real_v1f_top_idx),
        "diverse_top_k": _top_k_diversity_stats(real_candidates, diverse_idx),
        "overlap_count": len(set(real_v1f_top_idx) & set(diverse_idx)),
    }

    jsonl_demo4_total = 0
    jsonl_demo4_success = 0
    with args.rollout_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("demo_key") != args.demo_key:
                continue
            jsonl_demo4_total += 1
            if rec.get("rollout", {}).get("success_flag"):
                jsonl_demo4_success += 1
    empirical_pool_success_rate = float(jsonl_demo4_success / max(jsonl_demo4_total, 1))

    report: dict[str, Any] = {
        "task": "v1f_demo4_regression_audit",
        "demo_key": args.demo_key,
        "failure_type": cfg["failure_type"],
        "pool_seed": pool_seed,
        "alpha": args.alpha,
        "beta": args.beta,
        "known_good_summary": {
            "count_from_jsonl": len(known_goods),
            "best_known_insertion": best_insertion,
            "best_known_jsonl_E_total_norm": best_known.get("rollout_E_total_norm"),
            "best_known_sampling_index": best_known.get("sampling_index"),
        },
        "context_audit": {
            "offline_context": offline_context,
            "training_context": training_context,
            "context_shift_on_best_known_theta": context_shift_rows,
            "note": (
                "V1-F rollout sampling uses CEM-refined baseline context; "
                "offline repair test uses original failed waypoint context."
            ),
        },
        "replay_audit": {
            "replay_success_rate_known_good_theta": replay_success_rate,
            "per_theta": replay_rows,
        },
        "injected_pool_audit": {
            "pool_size": len(injected_candidates),
            "known_good_index": injected_target_index,
            "known_good_rank_by_scoring": injected_rank_by_scoring,
            "top20_diversity_by_scoring": injected_top20_stats,
        },
        "real_pool_audit": {
            "pool_size": args.pool_size,
            "empirical_success_rate_from_jsonl": empirical_pool_success_rate,
            "jsonl_demo4_total": jsonl_demo4_total,
            "jsonl_demo4_success": jsonl_demo4_success,
            "top20_compare_plain_vs_diverse": real_top20_compare,
            "top20_rollout": real_top20_rollout,
        },
        "scoring_methods": list(SCORING_METHODS),
    }
    diagnosis_code = classify_regression_root_cause(report)
    report["diagnosis"] = {
        "code": diagnosis_code,
        "label": _diagnosis_text(diagnosis_code),
    }

    report_path = args.output_dir / "debug_v1f_demo4_regression_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for method in SCORING_METHODS:
        stats = injected_top20_stats[method]
        summary_rows.append(
            {
                "section": "injected_pool",
                "method": method,
                "known_good_rank": injected_rank_by_scoring[method],
                "known_good_in_top_20": stats["known_good_in_top_k"],
                "top20_mean_pairwise_distance": stats["mean_pairwise_param_distance"],
                "top20_min_pairwise_distance": stats["min_pairwise_param_distance"],
                "top20_mean_distance_to_known_good": stats["distance_to_known_good"]["mean_distance"],
            }
        )

    summary_rows.append(
        {
            "section": "replay",
            "method": "known_good_theta_replay",
            "known_good_rank": "",
            "known_good_in_top_20": "",
            "top20_mean_pairwise_distance": "",
            "top20_min_pairwise_distance": "",
            "replay_success_rate": replay_success_rate,
        }
    )

    for ctx_row in context_shift_rows:
        summary_rows.append(
            {
                "section": "context_shift",
                "method": ctx_row["context_kind"],
                "predicted_E_total": ctx_row["predicted_E_total"],
                "v1f_success_prob": ctx_row["v1f_success_prob"],
                "v1f_uncertainty": ctx_row["v1f_uncertainty"],
            }
        )

    for method_name, payload in real_top20_rollout.items():
        summary_rows.append(
            {
                "section": "real_pool_top20_rollout",
                "method": method_name,
                "repair_rate_at_20": payload.get("repair_rate_at_20"),
                "success_at_20": payload.get("success_at_k", {}).get("at_20"),
                "best_E_total": payload.get("best_E_total"),
                "top20_mean_pairwise_distance": payload.get("diversity", {}).get("mean_pairwise_param_distance"),
            }
        )

    summary_rows.append(
        {
            "section": "diagnosis",
            "method": diagnosis_code,
            "label": _diagnosis_text(diagnosis_code),
            "replay_success_rate": replay_success_rate,
            "min_injected_rank": min(injected_rank_by_scoring.values()),
            "empirical_pool_success_rate": empirical_pool_success_rate,
        }
    )

    csv_path = args.output_dir / "debug_v1f_demo4_regression_summary.csv"
    all_keys: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "summary_csv": str(csv_path),
                "diagnosis": report["diagnosis"],
                "replay_success_rate_known_good_theta": replay_success_rate,
                "injected_ranks": injected_rank_by_scoring,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
