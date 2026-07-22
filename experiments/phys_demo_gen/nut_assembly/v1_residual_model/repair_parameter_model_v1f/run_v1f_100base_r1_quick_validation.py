#!/usr/bin/env python3
"""V1-F-100Base-R1 quick validation：固定 candidate pool + manifest。"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
_V1F_DIR = _EXPERIMENT_DIR / "v1_residual_model" / "repair_parameter_model_v1f"
for path in (_EXPERIMENT_DIR, _EXPERIMENT_DIR / "v1_residual_model", _V1F_DIR, _EXPERIMENT_DIR / "offline_mimicgen_repair_test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PINN_MODEL, DEMO_REPAIR_CONFIGS  # noqa: E402
from repair_common_v1f import select_candidate_indices_v1f, summarize_method_results_v1f  # noqa: E402
from repair_rollout import run_repair_rollout  # noqa: E402
from run_v1f_plus_evaluation import _repair_cfg_for_new_demo  # noqa: E402
from run_v1f_quick_evaluation import NEW_REPAIRABLE, OLD_DEMOS, job_key, load_partial  # noqa: E402
from v1f_100base_r1_utils import (  # noqa: E402
    ALIGNED_ORIGINAL_JSONL,
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_AUDIT_REPORT,
    DEFAULT_CANDIDATE_MANIFEST,
    DEFAULT_EVAL_REPORT,
    DEFAULT_TRAINED_MODEL,
    make_demo_uid,
    SOURCE_LEGACY_OLD,
)
from v1f_100base_r1_validation_pool import (  # noqa: E402
    build_fixed_candidate_pool,
    score_pool_with_model,
    top20_overlap,
)
from v1f_plus_utils import DEFAULT_FAILED_HDF5 as NEW_FAILED_HDF5, load_failure_map  # noqa: E402

VALIDATION_JOBS: list[tuple[str, str]] = [
    *[("old", demo_key) for demo_key in OLD_DEMOS],
    *[("new", demo_key) for demo_key in NEW_REPAIRABLE],
]

MODEL_VARIANTS = (
    ("aligned-original", "aligned-original"),
    ("v1f-100base-r1", "v1f-100base-r1"),
)


def _ensure_manifest_pools(
    *,
    jobs: list[tuple[str, str, dict[str, Any], Path]],
    aligned_model: Path,
    seed: int,
    num_samples: int,
    top_k: int,
    manifest_path: Path,
    aligned_jsonl: Path,
    v1e_model: Path,
    cem_report: Path,
) -> dict[str, dict[str, Any]]:
    """构建/加载固定 candidate pool manifest（两模型共享同一 theta pool）。"""
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {entry["demo_uid"]: entry for entry in loaded.get("pools", [])}

    pools: list[dict[str, Any]] = []
    for demo_group, demo_key, cfg, hdf5 in jobs:
        demo_uid = make_demo_uid(SOURCE_LEGACY_OLD if demo_group == "old" else "new100_failed", demo_key)
        pool, pool_seed, theta_hash = build_fixed_candidate_pool(
            demo_key=demo_key,
            cfg=cfg,
            num_samples=num_samples,
            seed=seed,
            aligned_jsonl=aligned_jsonl if demo_group == "old" else None,
            inject_known_good=(demo_group == "old" and demo_key in ("demo_4", "demo_2")),
        )
        aligned_scored = score_pool_with_model(
            candidates=pool,
            demo_key=demo_key,
            cfg=cfg,
            v1f_model=aligned_model,
            failed_hdf5=hdf5,
            cem_report=cem_report,
            v1e_model=v1e_model,
        )
        rng = random.Random(seed)
        aligned_top = select_candidate_indices_v1f(
            aligned_scored, method="v1f_plain_top_k", top_k=top_k, rng=rng
        )
        pools.append(
            {
                "demo_uid": demo_uid,
                "demo_key": demo_key,
                "demo_group": demo_group,
                "candidate_seed": pool_seed,
                "theta_list_hash": theta_hash,
                "num_candidates": len(pool),
                "selection_method": "v1f_plain_top_k",
                "top_k": top_k,
                "selected_top20_aligned_original": aligned_top,
                "rollout_seeds": [seed * 1000 + idx for idx in aligned_top],
                "_pool_cache": pool,
            }
        )

    manifest = {
        "manifest_version": "V1-F-100Base-R1",
        "shared_pool_policy": "aligned-original and v1f-100base-r1 use identical candidate theta pools",
        "seed": seed,
        "num_samples": num_samples,
        "top_k": top_k,
        "pools": [{k: v for k, v in p.items() if k != "_pool_cache"} for p in pools],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cache_path = manifest_path.with_suffix(".pool_cache.json")
    cache_path.write_text(
        json.dumps(
            {
                p["demo_uid"]: {
                    "candidate_seed": p["candidate_seed"],
                    "theta_list_hash": p["theta_list_hash"],
                }
                for p in pools
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {p["demo_uid"]: p for p in pools}


def run_offline_repair_fixed_pool(
    *,
    demo_key: str,
    demo_group: str,
    cfg: dict[str, Any],
    v1f_model: Path,
    failed_hdf5: Path,
    cem_report: Path,
    pool: list[dict[str, Any]],
    top_indices: list[int],
    v1e_model: Path,
    model_label: str,
    selection_method: str,
    top_k: int,
) -> dict[str, Any]:
    scored = score_pool_with_model(
        candidates=pool,
        demo_key=demo_key,
        cfg=cfg,
        v1f_model=v1f_model,
        failed_hdf5=failed_hdf5,
        cem_report=cem_report,
        v1e_model=v1e_model,
    )
    rollout_results = []
    for idx in top_indices:
        rollout_results.append(
            run_repair_rollout(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                search_kind=cfg["search_kind"],
                cem_report=cem_report if cfg["search_kind"] in ("insertion", "transport") else None,
                candidate=scored[idx],
            )
        )
    metrics = summarize_method_results_v1f(rollout_results, method=selection_method, rollout_budget=top_k)
    return {
        "demo_group": demo_group,
        "demo_key": demo_key,
        "failure_type": cfg.get("rough_failure_type", cfg["failure_type"]),
        "coarse_failure_type": cfg["failure_type"],
        "model_label": model_label,
        "selection_method": selection_method,
        "metrics": metrics,
        "selected_top20": top_indices,
    }


def evaluate_acceptance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    FAILURE_TYPES = ("transport_failed", "insertion_failed", "alignment_failed", "grasp_failed", "lift_failed")

    def rate(group: str, demo: str, variant: str) -> float | None:
        for r in rows:
            if r["demo_group"] == group and r["demo_key"] == demo and r["variant"] == variant:
                return float(r["repair_rate_at_20"])
        return None

    def avg_repairable(variant: str) -> float | None:
        vals = [
            float(r["repair_rate_at_20"])
            for r in rows
            if r["demo_group"] == "new" and r["demo_key"] in NEW_REPAIRABLE and r["variant"] == variant
        ]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def by_failure_type(group: str, variant: str, failure_type: str) -> dict[str, Any]:
        vals = [
            float(r["repair_rate_at_20"])
            for r in rows
            if r["demo_group"] == group and r["variant"] == variant and r.get("failure_type") == failure_type
        ]
        if not vals:
            return {"count": 0, "avg_repair_rate_at_20": None, "available": False}
        return {
            "count": len(vals),
            "avg_repair_rate_at_20": float(sum(vals) / len(vals)),
            "available": True,
        }

    d4_orig = rate("old", "demo_4", "aligned-original")
    d4_r1 = rate("old", "demo_4", "v1f-100base-r1")
    d2_orig = rate("old", "demo_2", "aligned-original")
    d2_r1 = rate("old", "demo_2", "v1f-100base-r1")
    d3_orig = rate("old", "demo_3", "aligned-original")
    d3_r1 = rate("old", "demo_3", "v1f-100base-r1")
    new_orig = avg_repairable("aligned-original")
    new_r1 = avg_repairable("v1f-100base-r1")

    by_ft: dict[str, dict[str, Any]] = {}
    for ft in FAILURE_TYPES:
        by_ft[ft] = {
            "aligned-original": by_failure_type("new", "aligned-original", ft),
            "v1f-100base-r1": by_failure_type("new", "v1f-100base-r1", ft),
        }

    checks = {
        "demo_4_ge_0_70": d4_r1 is not None and d4_r1 >= 0.70,
        "demo_2_ge_0_20": d2_r1 is not None and d2_r1 >= 0.20,
        "new_repairable_improved_vs_original": (
            new_orig is not None and new_r1 is not None and new_r1 > new_orig
        ),
    }
    can_replace_default = all(checks.values())
    return {
        "old_demo_4": {
            "aligned-original": d4_orig,
            "v1f-100base-r1": d4_r1,
            "threshold": 0.70,
            "passed": checks["demo_4_ge_0_70"],
        },
        "old_demo_2": {
            "aligned-original": d2_orig,
            "v1f-100base-r1": d2_r1,
            "threshold": 0.20,
            "passed": checks["demo_2_ge_0_20"],
        },
        "old_demo_3": {
            "aligned-original": d3_orig,
            "v1f-100base-r1": d3_r1,
            "passed": True,
        },
        "new_repairable_avg": {
            "aligned-original": new_orig,
            "v1f-100base-r1": new_r1,
            "improved": checks["new_repairable_improved_vs_original"],
        },
        "by_failure_type": by_ft,
        "checks": checks,
        "all_passed": all(checks.values()),
        "can_replace_aligned_original_as_default": can_replace_default,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-100Base-R1 quick validation (fixed candidate pool)")
    parser.add_argument("--model-r1", type=Path, default=DEFAULT_TRAINED_MODEL)
    parser.add_argument("--aligned-original-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--aligned-jsonl", type=Path, default=ALIGNED_ORIGINAL_JSONL)
    parser.add_argument("--old-failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--new-failed-hdf5", type=Path, default=NEW_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true", help="只生成 candidate manifest，不跑 rollout")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.parent / "quick_validation_partial.jsonl"
    done = load_partial(partial_path) if args.resume else {}

    failure_map = load_failure_map(args.audit_report)
    job_specs: list[tuple[str, str, dict[str, Any], Path]] = []
    for demo_group, demo_key in VALIDATION_JOBS:
        hdf5 = args.old_failed_hdf5 if demo_group == "old" else args.new_failed_hdf5
        if demo_group == "old":
            cfg = DEMO_REPAIR_CONFIGS[demo_key]
        else:
            cfg = _repair_cfg_for_new_demo(demo_key, failure_map)
        job_specs.append((demo_group, demo_key, cfg, hdf5))

    pool_map = _ensure_manifest_pools(
        jobs=job_specs,
        aligned_model=args.aligned_original_model,
        seed=args.seed,
        num_samples=args.num_samples,
        top_k=args.top_k,
        manifest_path=args.manifest,
        aligned_jsonl=args.aligned_jsonl,
        v1e_model=args.v1e_model,
        cem_report=args.cem_report,
    )

    if args.manifest_only:
        print(json.dumps({"manifest": str(args.manifest), "pools": len(pool_map)}, indent=2))
        return 0

    model_map = {
        "aligned-original": args.aligned_original_model,
        "v1f-100base-r1": args.model_r1,
    }

    rows: list[dict[str, Any]] = list(done.values())
    manifest_updates: dict[str, dict[str, Any]] = {}

    for demo_group, demo_key, cfg, hdf5 in job_specs:
        demo_uid = make_demo_uid(SOURCE_LEGACY_OLD if demo_group == "old" else "new100_failed", demo_key)
        pool_entry = pool_map[demo_uid]
        if "_pool_cache" in pool_entry:
            pool = pool_entry["_pool_cache"]
        else:
            pool, _, _ = build_fixed_candidate_pool(
                demo_key=demo_key,
                cfg=cfg,
                num_samples=args.num_samples,
                seed=args.seed,
                aligned_jsonl=args.aligned_jsonl if demo_group == "old" else None,
                inject_known_good=(demo_group == "old" and demo_key in ("demo_4", "demo_2")),
            )

        for variant, model_label in MODEL_VARIANTS:
            key = job_key(demo_group, demo_key, variant)
            if key in done:
                continue
            model_path = model_map[model_label]
            if not model_path.exists():
                if model_label == "v1f-100base-r1":
                    print(f"[skip] model not trained yet: {model_path}", flush=True)
                    continue
                raise SystemExit(f"Model checkpoint missing: {model_path}")

            print(f"[r1-val] {key}", flush=True)
            scored = score_pool_with_model(
                candidates=copy.deepcopy(pool),
                demo_key=demo_key,
                cfg=cfg,
                v1f_model=model_path,
                failed_hdf5=hdf5,
                cem_report=args.cem_report,
                v1e_model=args.v1e_model,
            )
            top_indices = select_candidate_indices_v1f(
                scored, method="v1f_plain_top_k", top_k=args.top_k, rng=random.Random(args.seed)
            )
            result = run_offline_repair_fixed_pool(
                demo_key=demo_key,
                demo_group=demo_group,
                cfg=cfg,
                v1f_model=model_path,
                failed_hdf5=hdf5,
                cem_report=args.cem_report,
                pool=pool,
                top_indices=top_indices,
                v1e_model=args.v1e_model,
                model_label=model_label,
                selection_method="v1f_plain_top_k",
                top_k=args.top_k,
            )
            row = {
                "job_key": key,
                "demo_uid": demo_uid,
                "demo_group": demo_group,
                "demo_key": demo_key,
                "variant": variant,
                "theta_list_hash": pool_entry["theta_list_hash"],
                "candidate_seed": pool_entry["candidate_seed"],
                "selection_method": "v1f_plain_top_k",
                "failure_type": result.get("failure_type"),
                "repair_rate_at_20": result["metrics"]["repair_rate_at_20"],
                "selected_top20": top_indices,
            }
            rows.append(row)
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

            if model_label == "v1f-100base-r1":
                manifest_updates[demo_uid] = {
                    "selected_top20_r1": top_indices,
                    "top20_overlap_aligned_vs_r1": top20_overlap(
                        pool_entry.get("selected_top20_aligned_original", []), top_indices
                    ),
                }

    if manifest_updates and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        for entry in manifest.get("pools", []):
            uid = entry.get("demo_uid")
            if uid in manifest_updates:
                entry.update(manifest_updates[uid])
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    acceptance = evaluate_acceptance(rows)
    report = {
        "validation_version": "V1-F-100Base-R1",
        "fixed_candidate_pool": True,
        "manifest": str(args.manifest),
        "rows": rows,
        "acceptance": acceptance,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_passed": acceptance["all_passed"]}, indent=2))
    return 0 if acceptance["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
