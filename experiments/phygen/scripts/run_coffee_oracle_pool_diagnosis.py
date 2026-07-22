#!/usr/bin/env python3
"""Oracle rollout diagnosis for CoffeePreparation held-out candidate pools."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

_online_spec = importlib.util.spec_from_file_location(
    "online_rollout",
    ROOT / "experiments" / "phygen" / "scripts" / "run_coffee_online_selected_rollout_validation.py",
)
online_rollout = importlib.util.module_from_spec(_online_spec)
sys.modules["online_rollout"] = online_rollout
assert _online_spec.loader is not None
_online_spec.loader.exec_module(online_rollout)

DEFAULT_HELD_OUT = online_rollout.DEFAULT_HELD_OUT
_demo_sort_key = online_rollout._demo_sort_key
_load_jsonl = online_rollout._load_jsonl
_theta_key = online_rollout._theta_key
_write_jsonl = online_rollout._write_jsonl
filter_prepared_source = online_rollout.filter_prepared_source
generate_candidate_pool = online_rollout.generate_candidate_pool
load_context_for_demo = online_rollout.load_context_for_demo
load_model = online_rollout.load_model
run_official_rollout = online_rollout.run_official_rollout
score_candidate_pool = online_rollout.score_candidate_pool
select_top_k = online_rollout.select_top_k

ORCH_SEEDS = [88001, 88002, 88003]
POOL_SIZE = 32
POOL_START_INDEX = 10000
BASE_SEED = 88001


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _load_selection(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_flags(selection: dict[str, Any], best_fixed_theta: dict[str, Any]) -> dict[int, dict[str, bool]]:
    best_key = _theta_key(best_fixed_theta)
    flags: dict[int, dict[str, bool]] = {}

    def _mark(rows: list[dict[str, Any]], key: str) -> None:
        for row in rows:
            idx = int(row["candidate_index"])
            flags.setdefault(idx, {})
            flags[idx][key] = True

    _mark(selection.get("phygen_top1", []), "phygen_top1")
    _mark(selection.get("phygen_top3", []), "phygen_top3")
    _mark(selection.get("random_top3", []), "phygen_top3_random")
    for row in selection.get("phygen_top1", []) + selection.get("phygen_top3", []):
        idx = int(row["candidate_index"])
        flags.setdefault(idx, {})
        flags[idx]["best_fixed"] = _theta_key(row["theta"]) == best_key
    return flags


def _extract_task_metrics(rollout_dir: Path) -> dict[str, float]:
    import h5py

    from phygen.adapters.mimicgen.coffee_repair import compute_context_metrics

    for name in ("demo.hdf5", "demo_failed.hdf5"):
        hdf5 = rollout_dir / "demo" / name
        if not hdf5.is_file():
            continue
        try:
            with h5py.File(hdf5, "r") as f:
                demos = sorted(f["data"].keys(), key=_demo_sort_key)
                if not demos:
                    continue
                obs = np.asarray(f["data"][demos[0]]["obs"]["object"], dtype=np.float64)
                metrics = compute_context_metrics(obs[-1])
                return {k: float(v) for k, v in metrics.items()}
        except OSError:
            continue
    return {}


def regenerate_scored_pool(
    *,
    demo_key: str,
    prepared_source: Path,
    checkpoint: Path,
    pool_size: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    adapter, model = load_model(checkpoint)
    ctx = load_context_for_demo(prepared_source, demo_key)
    pool = generate_candidate_pool(
        adapter,
        demo_key,
        ctx["context_metrics"],
        pool_size=pool_size,
        seed=base_seed,
        start_index=POOL_START_INDEX,
    )
    return score_candidate_pool(adapter, model, pool)


def run_oracle_rollouts(
    *,
    demos: list[str],
    scored_pools: dict[str, list[dict[str, Any]]],
    selection_flags: dict[str, dict[int, dict[str, bool]]],
    prepared_source: Path,
    output_jsonl: Path,
    oracle_root: Path,
    seeds: list[int],
    skip_rollout: bool,
) -> list[dict[str, Any]]:
    existing: dict[tuple[str, int, int], dict[str, Any]] = {}
    if output_jsonl.is_file():
        for row in _load_jsonl(output_jsonl):
            existing[(row["demo_key"], int(row["candidate_index"]), int(row["seed"]))] = row

    per_demo_source = {
        d: filter_prepared_source(
            prepared_source,
            oracle_root / "prepared_sources" / f"{d}.hdf5",
            [d],
        )
        for d in demos
    }

    rows: list[dict[str, Any]] = list(existing.values())
    for demo_key in sorted(demos, key=_demo_sort_key):
        pool = scored_pools[demo_key]
        flags = selection_flags.get(demo_key, {})
        for cand in pool:
            idx = int(cand["candidate_index"])
            theta = dict(cand["theta"])
            for seed in seeds:
                key = (demo_key, idx, seed)
                if key in existing:
                    continue
                run_dir = oracle_root / "rollouts" / demo_key / f"cand_{idx}" / f"seed_{seed}"
                if skip_rollout:
                    continue
                res = run_official_rollout(
                    theta=theta,
                    prepared_source=per_demo_source[demo_key],
                    output_dir=run_dir,
                    num_trials=1,
                    seed=seed,
                )
                task_metrics = _extract_task_metrics(run_dir)
                flag = flags.get(idx, {})
                row = {
                    "demo_key": demo_key,
                    "candidate_index": idx,
                    "theta": theta,
                    "seed": seed,
                    "rollout_status": str(res.get("status", "unknown")),
                    "official_success": bool(res.get("any_success", False)),
                    "generated_success_count": int(res.get("num_success", 0)),
                    "num_trials": int(res.get("num_trials", 1)),
                    "task_metrics": task_metrics,
                    "pred_energy": float(cand.get("pred_energy", float("nan"))),
                    "pred_success_prob": float(cand.get("pred_success_prob", float("nan"))),
                    "acquisition_score": float(cand.get("acquisition_score", float("nan"))),
                    "task_aligned_energy": _json_float(task_metrics.get("energy")),
                    "stage_progress": _json_float(task_metrics.get("stage_progress")),
                    "whether_selected_by_phygen_top1": bool(flag.get("phygen_top1", False)),
                    "whether_selected_by_phygen_top3": bool(flag.get("phygen_top3", False)),
                    "whether_selected_by_random_top3": bool(flag.get("phygen_top3_random", False)),
                    "whether_best_fixed": bool(flag.get("best_fixed", False)),
                    "rollout_dir": str(run_dir),
                }
                rows.append(row)
                existing[key] = row
                _write_jsonl(output_jsonl, rows)
                print(json.dumps({"demo": demo_key, "cand": idx, "seed": seed, "ok": row["official_success"]}), flush=True)
    return rows


def _mean_or_none(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(nums)) if nums else None


def _safe_corr(x: list[float], y: list[float], method: str = "spearman") -> float | None:
    if len(x) < 3 or len(set(x)) <= 1 or len(set(y)) <= 1:
        return None
    try:
        from scipy.stats import pearsonr, spearmanr

        if method == "spearman":
            return float(spearmanr(x, y).correlation)
        return float(pearsonr(x, y)[0])
    except Exception:
        return None


def aggregate_candidate_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["demo_key"], int(r["candidate_index"]))].append(r)
    out: dict[str, dict[str, Any]] = {}
    for (demo, idx), grp in by.items():
        successes = [bool(g["official_success"]) for g in grp]
        out[f"{demo}:{idx}"] = {
            "demo_key": demo,
            "candidate_index": idx,
            "theta": grp[0]["theta"],
            "oracle_success_rate": sum(successes) / len(successes),
            "oracle_any_success": any(successes),
            "pred_success_prob": grp[0].get("pred_success_prob"),
            "pred_energy": grp[0].get("pred_energy"),
            "acquisition_score": grp[0].get("acquisition_score"),
            "task_aligned_energy": _mean_or_none([g.get("task_aligned_energy") for g in grp]),
        }
    return out


def per_demo_oracle_summary(
    rows: list[dict[str, Any]],
    scored_pools: dict[str, list[dict[str, Any]]],
    best_fixed_theta: dict[str, Any],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    best_key = _theta_key(best_fixed_theta)
    for demo_key in sorted({r["demo_key"] for r in rows}, key=_demo_sort_key):
        demo_rows = [r for r in rows if r["demo_key"] == demo_key]
        cand_stats = aggregate_candidate_stats(demo_rows)
        stats_list = list(cand_stats.values())
        if not stats_list:
            continue
        oracle_rates = [s["oracle_success_rate"] for s in stats_list]
        any_success = [s["oracle_any_success"] for s in stats_list]
        ranked = sorted(stats_list, key=lambda s: (-s["oracle_success_rate"], s["candidate_index"]))
        best_rate = ranked[0]["oracle_success_rate"]

        pool = scored_pools[demo_key]
        phygen_top1 = select_top_k(pool, 1, by="acquisition")[0]
        phygen_top3 = select_top_k(pool, 3, by="acquisition")
        rng = random.Random(BASE_SEED + _demo_sort_key(demo_key))
        shuffled = list(pool)
        rng.shuffle(shuffled)
        random_top3 = shuffled[:3]

        def _rate_for_indices(indices: list[int]) -> float:
            rs = [cand_stats[f"{demo_key}:{i}"]["oracle_success_rate"] for i in indices if f"{demo_key}:{i}" in cand_stats]
            return float(np.mean(rs)) if rs else 0.0

        def _any_for_indices(indices: list[int]) -> bool:
            return any(cand_stats.get(f"{demo_key}:{i}", {}).get("oracle_any_success", False) for i in indices)

        phygen_top1_idx = int(phygen_top1["candidate_index"])
        phygen_top3_idx = [int(x["candidate_index"]) for x in phygen_top3]
        random_top3_idx = [int(x["candidate_index"]) for x in random_top3]

        oracle_ranked = sorted(stats_list, key=lambda s: (-s["oracle_success_rate"], s["candidate_index"]))
        rank_map = {s["candidate_index"]: i + 1 for i, s in enumerate(oracle_ranked)}
        phygen_top1_rank = rank_map.get(phygen_top1_idx)

        best_fixed_rates = [
            s["oracle_success_rate"] for s in stats_list if _theta_key(s["theta"]) == best_key
        ]

        summaries[demo_key] = {
            "candidate_pool_oracle_top1_success_exists": any(any_success),
            "candidate_pool_success_rate": float(np.mean(oracle_rates)),
            "candidate_pool_any_success_fraction": float(np.mean(any_success)),
            "best_candidate_success_rate": best_rate,
            "phygen_top1_candidate_rank_under_oracle": phygen_top1_rank,
            "phygen_top1_oracle_success_rate": cand_stats.get(f"{demo_key}:{phygen_top1_idx}", {}).get("oracle_success_rate"),
            "phygen_top3_oracle_coverage": float(np.mean([cand_stats.get(f"{demo_key}:{i}", {}).get("oracle_any_success", False) for i in phygen_top3_idx])),
            "random_top3_expected_oracle_coverage": _rate_for_indices(random_top3_idx),
            "random_top3_any_success_coverage": _any_for_indices(random_top3_idx),
            "best_fixed_theta_oracle_success_rate": float(np.mean(best_fixed_rates)) if best_fixed_rates else None,
            "num_candidates": len(stats_list),
        }
    return summaries


def correlation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cand_stats = aggregate_candidate_stats(rows)
    stats_list = list(cand_stats.values())
    y = [s["oracle_success_rate"] for s in stats_list]
    return {
        "pred_success_prob_spearman": _safe_corr([s["pred_success_prob"] for s in stats_list], y, "spearman"),
        "pred_success_prob_pearson": _safe_corr([s["pred_success_prob"] for s in stats_list], y, "pearson"),
        "pred_energy_spearman": _safe_corr([s["pred_energy"] for s in stats_list], y, "spearman"),
        "pred_energy_pearson": _safe_corr([s["pred_energy"] for s in stats_list], y, "pearson"),
        "acquisition_score_spearman": _safe_corr([s["acquisition_score"] for s in stats_list], y, "spearman"),
        "acquisition_score_pearson": _safe_corr([s["acquisition_score"] for s in stats_list], y, "pearson"),
        "task_aligned_energy_spearman": _safe_corr(
            [s["task_aligned_energy"] for s in stats_list if s.get("task_aligned_energy") is not None],
            [s["oracle_success_rate"] for s in stats_list if s.get("task_aligned_energy") is not None],
            "spearman",
        ),
        "task_aligned_energy_pearson": _safe_corr(
            [s["task_aligned_energy"] for s in stats_list if s.get("task_aligned_energy") is not None],
            [s["oracle_success_rate"] for s in stats_list if s.get("task_aligned_energy") is not None],
            "pearson",
        ),
        "num_candidates": len(stats_list),
    }


def theta_family_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cand_stats = aggregate_candidate_stats(rows)
    families = [
        "selection_strategy",
        "action_noise",
        "offset_range",
        "num_interpolation_steps",
        "transform_first_robot_pose",
        "interpolate_from_last_target_pose",
        "nn_k",
    ]
    out: dict[str, Any] = {}
    for fam in families:
        bucket: dict[str, list[float]] = defaultdict(list)
        for s in cand_stats.values():
            val = s["theta"].get(fam)
            key = json.dumps(val, sort_keys=True) if isinstance(val, (list, dict)) else str(val)
            bucket[key].append(s["oracle_success_rate"])
        out[fam] = {
            k: {"count": len(v), "mean_success_rate": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))}
            for k, v in sorted(bucket.items(), key=lambda kv: -np.mean(kv[1]))
        }
    return out


def build_harder_pool(
    cand_stats: dict[str, dict[str, Any]],
    demo_key: str,
    *,
    min_candidates: int = 20,
    target_low: float = 0.20,
    target_high: float = 0.40,
) -> list[dict[str, Any]]:
    demo_stats = [v for v in cand_stats.values() if v["demo_key"] == demo_key]
    # exclude trivial always-success or always-fail if possible; prefer boundary [0.2,0.8]
    boundary = [s for s in demo_stats if 0.0 < s["oracle_success_rate"] < 1.0]
    hard = [s for s in boundary if target_low <= s["oracle_success_rate"] <= target_high]
    if len(hard) < min_candidates:
        hard = boundary if len(boundary) >= min_candidates else demo_stats
    hard = sorted(hard, key=lambda s: abs(s["oracle_success_rate"] - 0.5))[: max(min_candidates, len(hard))]
    return hard


def simulate_harder_eval(
    harder_pools: dict[str, list[dict[str, Any]]],
    scored_pools: dict[str, list[dict[str, Any]]],
    best_fixed_theta: dict[str, Any],
) -> dict[str, Any]:
    best_key = _theta_key(best_fixed_theta)
    methods = ["random", "best_fixed", "pred_success_prob", "phygen_acquisition"]
    results: dict[str, Any] = {m: {"top1": [], "top3_any": [], "rollout_rate": []} for m in methods}

    for demo_key, hard in harder_pools.items():
        if not hard:
            continue
        hard_indices = {int(s["candidate_index"]) for s in hard}
        pool = [c for c in scored_pools[demo_key] if int(c["candidate_index"]) in hard_indices]
        hard_map = {int(s["candidate_index"]): s for s in hard}

        rng = random.Random(BASE_SEED + _demo_sort_key(demo_key) + 999)

        def _pick(method: str, k: int) -> list[dict[str, Any]]:
            if method == "random":
                p = list(pool)
                rng.shuffle(p)
                return p[:k]
            if method == "best_fixed":
                matches = [c for c in pool if _theta_key(c["theta"]) == best_key]
                return matches[:k] if matches else pool[:k]
            if method == "pred_success_prob":
                return select_top_k(pool, k, by="pred_success_prob")
            return select_top_k(pool, k, by="acquisition")

        for method in methods:
            top1 = _pick(method, 1)[0]
            top3 = _pick(method, 3)
            idx1 = int(top1["candidate_index"])
            idx3 = [int(x["candidate_index"]) for x in top3]
            results[method]["top1"].append(hard_map[idx1]["oracle_any_success"])
            results[method]["top3_any"].append(any(hard_map[i]["oracle_any_success"] for i in idx3))
            results[method]["rollout_rate"].append(
                float(np.mean([hard_map[i]["oracle_success_rate"] for i in idx3]))
            )

    table = {}
    for method, vals in results.items():
        n = max(len(vals["top1"]), 1)
        table[method] = {
            "top1_selected_trajectory_success_rate": float(np.mean(vals["top1"])) if vals["top1"] else 0.0,
            "top3_repair_success_rate": float(np.mean(vals["top3_any"])) if vals["top3_any"] else 0.0,
            "selected_rollout_success_rate": float(np.mean(vals["rollout_rate"])) if vals["rollout_rate"] else 0.0,
        }
    return table


def diagnose(rows: list[dict[str, Any]], report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    corr = report["correlations"]
    if corr.get("pred_success_prob_spearman") is not None and abs(corr["pred_success_prob_spearman"]) < 0.2:
        lines.append("pred_success_prob 与 oracle success 相关性弱，模型尚未学到在线成功规律。")
    elif corr.get("pred_success_prob_spearman") is not None and abs(corr["pred_success_prob_spearman"]) >= 0.3:
        if corr.get("acquisition_score_spearman") is not None and abs(corr["acquisition_score_spearman"]) < abs(corr["pred_success_prob_spearman"]) - 0.1:
            lines.append("pred_success_prob 有一定相关性，但 acquisition 排名信号更弱，acquisition 规则可能是瓶颈。")
        else:
            lines.append("pred_success_prob 与 oracle 有一定相关性，selector 信号部分有效。")

    pool_rates = [v["candidate_pool_success_rate"] for v in report["per_demo_oracle"].values()]
    if pool_rates and float(np.mean(pool_rates)) > 0.55:
        lines.append("候选池整体 oracle success rate 偏高，pool 可能太容易，random top-3 易饱和。")

    bf = [v.get("best_fixed_theta_oracle_success_rate") for v in report["per_demo_oracle"].values() if v.get("best_fixed_theta_oracle_success_rate") is not None]
    if bf and float(np.mean(bf)) > float(np.mean(pool_rates or [0])):
        lines.append("best fixed theta 在 held-out 上仍较强，可作为强 baseline。")

    theta_stats = report["theta_family_stats"]
    spreads = []
    for fam, buckets in theta_stats.items():
        if len(buckets) >= 2:
            means = [b["mean_success_rate"] for b in buckets.values()]
            spreads.append(max(means) - min(means))
    if spreads and float(np.mean(spreads)) < 0.15:
        lines.append("theta 各 family 间 success rate 差异小，CoffeePreparation 上 theta 影响偏弱。")

    lines.append("本阶段不建议修改 PhyGen-Core；优先排查 adapter metrics 对齐与 acquisition。")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", default="runs/phygen_coffee_online_rollout")
    parser.add_argument(
        "--prepared-source",
        default="runs/phygen_coffee_theta_sweep_v2/prepared_source_replay_success.hdf5",
    )
    parser.add_argument("--held-out", default=",".join(DEFAULT_HELD_OUT))
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--analyze-only", action="store_true", help="skip rollout, analyze existing jsonl")
    args = parser.parse_args()

    online_root = (ROOT / args.online_root).resolve()
    prepared_source = (ROOT / args.prepared_source).resolve()
    checkpoint = online_root / "train" / "coffee_preparation_failed_conditioned_pinn.pt"
    held_out = [x.strip() for x in args.held_out.split(",") if x.strip()]
    output_jsonl = online_root / "oracle_pool_rollouts.jsonl"
    oracle_root = online_root / "oracle_diagnosis"

    online_report = json.loads((online_root / "online_rollout_report.json").read_text())
    best_fixed_theta = online_report["best_fixed_theta"]["theta"]

    scored_pools: dict[str, list[dict[str, Any]]] = {}
    selection_flags: dict[str, dict[int, dict[str, bool]]] = {}
    for demo_key in held_out:
        sel_path = online_root / "selections" / demo_key / "selected_theta.json"
        selection = _load_selection(sel_path)
        selection_flags[demo_key] = _candidate_flags(selection, best_fixed_theta)
        scored_pools[demo_key] = regenerate_scored_pool(
            demo_key=demo_key,
            prepared_source=prepared_source,
            checkpoint=checkpoint,
            pool_size=POOL_SIZE,
            base_seed=BASE_SEED,
        )

    if not args.analyze_only:
        rows = run_oracle_rollouts(
            demos=held_out,
            scored_pools=scored_pools,
            selection_flags=selection_flags,
            prepared_source=prepared_source,
            output_jsonl=output_jsonl,
            oracle_root=oracle_root,
            seeds=ORCH_SEEDS,
            skip_rollout=args.skip_rollout,
        )
    else:
        rows = _load_jsonl(output_jsonl)

    cand_stats = aggregate_candidate_stats(rows)
    per_demo = per_demo_oracle_summary(rows, scored_pools, best_fixed_theta)
    corr = correlation_report(rows)
    theta_fam = theta_family_stats(rows)

    harder_pools = {d: build_harder_pool(cand_stats, d) for d in held_out}
    harder_eval = simulate_harder_eval(harder_pools, scored_pools, best_fixed_theta)

    report = {
        "held_out_demos": held_out,
        "num_oracle_records": len(rows),
        "seeds_per_candidate": len(ORCH_SEEDS),
        "per_demo_oracle": per_demo,
        "correlations": corr,
        "theta_family_stats": theta_fam,
        "harder_pool_sizes": {d: len(h) for d, h in harder_pools.items()},
        "harder_pool_success_rates": {
            d: float(np.mean([x["oracle_success_rate"] for x in h])) if h else None for d, h in harder_pools.items()
        },
        "harder_eval_simulation": harder_eval,
    }
    report["diagnosis"] = diagnose(rows, report)

    with (online_root / "oracle_pool_diagnosis_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
