"""V1-F-100Base-R1：固定 validation candidate pool 与 old-demo ranking gate。"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from config import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PINN_MODEL, DEMO_REPAIR_CONFIGS
from repair_common_v1f import (
    extract_repair_context_v1f,
    rank_theta_by_score,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
)
from pinn_v1f_inference import clear_v1f_model_cache
from v1f_100base_r1_utils import (
    RANKING_GATE_MIN_TOP20_OVERLAP,
    SOURCE_LEGACY_OLD,
    make_demo_uid,
)

OLD_RANKING_GATE_DEMOS = ("demo_4", "demo_2")


def _normalize_insertion(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in raw.items():
        if key in ("insertion_steps", "hold_steps", "pre_insert_pause", "release_shift"):
            out[key] = float(int(float(val)))
        else:
            out[key] = float(val)
    return out


def hash_candidate_pool(candidates: list[dict[str, Any]]) -> str:
    payload: list[dict[str, Any]] = []
    for c in candidates:
        payload.append(
            {
                "index": c.get("index"),
                "insertion": c.get("insertion"),
                "transport": c.get("transport"),
                "grasp_lift": c.get("grasp_lift"),
                "lift_extra": c.get("lift_extra"),
                "injected_known_good": bool(c.get("injected_known_good")),
            }
        )
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_known_good_thetas(jsonl_path: Path, demo_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        return rows
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
            sim = rollout.get("sim_params") or rollout.get("repair_insertion_params") or {}
            if not sim:
                continue
            rows.append({"insertion": _normalize_insertion(sim), "sampling_index": rollout.get("sampling_index")})
    return rows


def build_fixed_candidate_pool(
    *,
    demo_key: str,
    cfg: dict[str, Any],
    num_samples: int,
    seed: int,
    aligned_jsonl: Path | None = None,
    inject_known_good: bool = False,
) -> tuple[list[dict[str, Any]], int, str]:
    pool_seed = seed + hash(demo_key) % 10000
    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"], n_samples=num_samples, seed=pool_seed
    )
    if inject_known_good and aligned_jsonl is not None:
        known_goods = load_known_good_thetas(aligned_jsonl, demo_key)
        for i, kg in enumerate(known_goods[:5]):
            candidates.append(
                {
                    "index": num_samples + i,
                    "insertion": kg["insertion"],
                    "transport": None,
                    "grasp_lift": None,
                    "lift_extra": None,
                    "injected_known_good": True,
                }
            )
    theta_hash = hash_candidate_pool(candidates)
    return candidates, pool_seed, theta_hash


def score_pool_with_model(
    *,
    candidates: list[dict[str, Any]],
    demo_key: str,
    cfg: dict[str, Any],
    v1f_model: Path,
    failed_hdf5: Path,
    cem_report: Path,
    v1e_model: Path,
) -> list[dict[str, Any]]:
    context = extract_repair_context_v1f(
        context_source="original_failed_context",
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
        cem_report=cem_report,
    )
    cands = copy.deepcopy(candidates)
    clear_v1f_model_cache()
    score_repair_candidates_v1f(
        context=context,
        candidates=cands,
        active=cfg["active"],
        v1e_model_path=v1e_model,
        v1f_model_path=v1f_model,
    )
    return cands


def top20_overlap(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / float(max(len(sa), len(sb)))


def evaluate_old_demo_ranking_gate(
    *,
    candidate_model: Path,
    baseline_model: Path,
    failed_hdf5: Path = DEFAULT_FAILED_HDF5,
    cem_report: Path = DEFAULT_CEM_REPORT,
    aligned_jsonl: Path | None = None,
    v1e_model: Path = DEFAULT_PINN_MODEL,
    num_samples: int = 500,
    top_k: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """训练期 old-demo ranking gate：top-20 overlap + known-good rank 不退化。"""
    results: dict[str, Any] = {"demos": {}, "passed": True, "block_reasons": []}

    for demo_key in OLD_RANKING_GATE_DEMOS:
        cfg = DEMO_REPAIR_CONFIGS[demo_key]
        pool, pool_seed, theta_hash = build_fixed_candidate_pool(
            demo_key=demo_key,
            cfg=cfg,
            num_samples=num_samples,
            seed=seed,
            aligned_jsonl=aligned_jsonl,
            inject_known_good=True,
        )
        baseline_scored = score_pool_with_model(
            candidates=pool,
            demo_key=demo_key,
            cfg=cfg,
            v1f_model=baseline_model,
            failed_hdf5=failed_hdf5,
            cem_report=cem_report,
            v1e_model=v1e_model,
        )
        clear_v1f_model_cache()
        candidate_scored = score_pool_with_model(
            candidates=pool,
            demo_key=demo_key,
            cfg=cfg,
            v1f_model=candidate_model,
            failed_hdf5=failed_hdf5,
            cem_report=cem_report,
            v1e_model=v1e_model,
        )
        rng = random.Random(seed)
        baseline_top = select_candidate_indices_v1f(
            baseline_scored, method="v1f_plain_top_k", top_k=top_k, rng=rng
        )
        candidate_top = select_candidate_indices_v1f(
            candidate_scored, method="v1f_plain_top_k", top_k=top_k, rng=random.Random(seed)
        )
        overlap = top20_overlap(baseline_top, candidate_top)

        kg_indices = [i for i, c in enumerate(pool) if c.get("injected_known_good")]
        baseline_kg_ranks = [
            rank_theta_by_score(baseline_scored, score_key="v1f_E_total", target_score=baseline_scored[i]["v1f_E_total"])
            for i in kg_indices
        ]
        candidate_kg_ranks = [
            rank_theta_by_score(
                candidate_scored, score_key="v1f_E_total", target_score=candidate_scored[i]["v1f_E_total"]
            )
            for i in kg_indices
        ]
        baseline_best_kg = min(baseline_kg_ranks) if baseline_kg_ranks else 9999
        candidate_best_kg = min(candidate_kg_ranks) if candidate_kg_ranks else 9999
        kg_regressed = candidate_best_kg > baseline_best_kg

        demo_pass = True
        if demo_key == "demo_4" and overlap < RANKING_GATE_MIN_TOP20_OVERLAP:
            demo_pass = False
            results["block_reasons"].append(
                f"demo_4 top-20 overlap {overlap:.3f} < {RANKING_GATE_MIN_TOP20_OVERLAP}"
            )
        if kg_regressed:
            demo_pass = False
            results["block_reasons"].append(
                f"{demo_key} known-good rank regressed {baseline_best_kg} -> {candidate_best_kg}"
            )

        results["demos"][demo_key] = {
            "demo_uid": make_demo_uid(SOURCE_LEGACY_OLD, demo_key),
            "candidate_seed": pool_seed,
            "theta_list_hash": theta_hash,
            "baseline_top20": baseline_top,
            "candidate_top20": candidate_top,
            "top20_overlap": overlap,
            "baseline_best_known_good_rank": baseline_best_kg,
            "candidate_best_known_good_rank": candidate_best_kg,
            "known_good_rank_regressed": kg_regressed,
            "passed": demo_pass,
        }
        if not demo_pass:
            results["passed"] = False

    return results


def build_validation_manifest_entry(
    *,
    demo_key: str,
    demo_group: str,
    cfg: dict[str, Any],
    failed_hdf5: Path,
    cem_report: Path,
    aligned_model: Path,
    r1_model: Path | None,
    v1e_model: Path,
    num_samples: int,
    top_k: int,
    seed: int,
    aligned_jsonl: Path | None = None,
) -> dict[str, Any]:
    demo_uid = make_demo_uid(SOURCE_LEGACY_OLD if demo_group == "old" else "new100_failed", demo_key)

    pool, pool_seed, theta_hash = build_fixed_candidate_pool(
        demo_key=demo_key,
        cfg=cfg,
        num_samples=num_samples,
        seed=seed,
        aligned_jsonl=aligned_jsonl if demo_group == "old" else None,
        inject_known_good=(demo_key in OLD_RANKING_GATE_DEMOS and demo_group == "old"),
    )
    aligned_scored = score_pool_with_model(
        candidates=pool,
        demo_key=demo_key,
        cfg=cfg,
        v1f_model=aligned_model,
        failed_hdf5=failed_hdf5,
        cem_report=cem_report,
        v1e_model=v1e_model,
    )
    rng = random.Random(seed)
    aligned_top = select_candidate_indices_v1f(
        aligned_scored, method="v1f_plain_top_k", top_k=top_k, rng=rng
    )
    rollout_seeds = [seed * 1000 + idx for idx in aligned_top]

    entry: dict[str, Any] = {
        "demo_uid": demo_uid,
        "demo_key": demo_key,
        "demo_group": demo_group,
        "candidate_seed": pool_seed,
        "theta_list_hash": theta_hash,
        "num_candidates": len(pool),
        "selection_method": "v1f_plain_top_k",
        "top_k": top_k,
        "selected_top20_aligned_original": aligned_top,
        "rollout_seeds": rollout_seeds,
    }
    if r1_model is not None and r1_model.exists():
        r1_scored = score_pool_with_model(
            candidates=pool,
            demo_key=demo_key,
            cfg=cfg,
            v1f_model=r1_model,
            failed_hdf5=failed_hdf5,
            cem_report=cem_report,
            v1e_model=v1e_model,
        )
        r1_top = select_candidate_indices_v1f(
            r1_scored, method="v1f_plain_top_k", top_k=top_k, rng=random.Random(seed)
        )
        entry["selected_top20_r1"] = r1_top
        entry["top20_overlap_aligned_vs_r1"] = top20_overlap(aligned_top, r1_top)
    return entry
