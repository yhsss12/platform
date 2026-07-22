#!/usr/bin/env python3
"""Task 2：对 quick eval 中 plus-balanced 修复率低的 demo 做 targeted rollout 采样。"""
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

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PINN_MODEL, DEMO_REPAIR_CONFIGS  # noqa: E402
from osc_action_converter import SEARCH_SPACE  # noqa: E402
from repair_common_v1f import (  # noqa: E402
    diverse_top_k_indices,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
)
from repair_rollout import run_repair_rollout  # noqa: E402
from run_v1f_plus_evaluation import _repair_cfg_for_new_demo  # noqa: E402
from run_v1f_plus_rollout_sampling import _rollout_record_to_jsonl_row  # noqa: E402
from grasp_sim_search import GRASP_SEARCH_SPACE  # noqa: E402
from transport_sim_search import TRANSPORT_SEARCH_SPACE  # noqa: E402
from v1f_plus_utils import (  # noqa: E402
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_FAILED_HDF5 as NEW_FAILED_HDF5,
    load_failure_map,
    search_kind_for_failure,
)
from repair_common_v1f import extract_repair_context_v1f  # noqa: E402
from v1f_repair_dataset import extract_failed_context  # noqa: E402

DEFAULT_BALANCED_MODEL = (
    _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "trained_model" / "model_v1f_aligned_plus_balanced.pt"
)
DEFAULT_QUICK_CSV = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "quick_eval" / "quick_summary.csv"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced_v2" / "targeted_rollout_samples.jsonl"
DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"

TARGET_SUCCESS = 5
LOW_RATE_THRESH = 0.25
E_DROP_RATIO = 0.12


def load_low_repair_demos(csv_path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    by_demo: dict[str, dict[str, float]] = {}
    meta: dict[str, dict[str, str]] = {}
    for r in rows:
        key = f"{r['demo_group']}|{r['demo_key']}"
        by_demo.setdefault(key, {})[r["variant"]] = float(r["repair_rate_at_20"])
        meta[key] = {"demo_group": r["demo_group"], "demo_key": r["demo_key"], "failure_type": r.get("failure_type", "")}
    targets: list[dict[str, Any]] = []
    for key, rates in by_demo.items():
        balanced = rates.get("aligned-plus-balanced", 0.0)
        original = rates.get("aligned-original", balanced)
        if balanced < LOW_RATE_THRESH or balanced + 0.05 < original:
            targets.append({**meta[key], "balanced_rate": balanced, "original_rate": original})
    return targets


def _perturb_param_dict(params: dict[str, Any], space: dict[str, list], rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, choices in space.items():
        if key not in params:
            out[key] = rng.choice(choices)
            continue
        cur = params[key]
        if cur in choices:
            idx = choices.index(cur)
        else:
            idx = rng.randrange(len(choices))
        step = rng.choice([-2, -1, 0, 1, 2])
        out[key] = choices[max(0, min(len(choices) - 1, idx + step))]
    return out


def _candidate_key(cand: dict[str, Any]) -> str:
    if cand.get("insertion"):
        return json.dumps(cand["insertion"], sort_keys=True)
    if cand.get("transport"):
        return json.dumps(cand["transport"], sort_keys=True)
    gl = cand.get("grasp_lift") or {}
    le = cand.get("lift_extra") or {}
    return json.dumps({**gl, **le}, sort_keys=True)


def _build_candidate_pool(
    *,
    cfg: dict[str, Any],
    context: dict[str, Any],
    v1f_model: Path,
    v1e_model: Path,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    active = cfg["active"]
    search_kind = cfg["search_kind"]
    candidates = sample_repair_candidates_v1f(search_kind=search_kind, n_samples=num_samples, seed=seed)
    for i, c in enumerate(candidates):
        c["index"] = i
    score_repair_candidates_v1f(
        context=context,
        candidates=candidates,
        active=active,
        v1e_model_path=v1e_model,
        v1f_model_path=v1f_model,
    )
    return candidates


def _select_rollout_candidates(candidates: list[dict[str, Any]], *, rng: random.Random) -> list[dict[str, Any]]:
    by_score = sorted(candidates, key=lambda c: c["v1f_E_total"])
    top50 = by_score[:50]
    by_unc = sorted(candidates, key=lambda c: c.get("v1f_uncertainty", 0.0), reverse=True)[:20]
    diverse_idx = diverse_top_k_indices(candidates, score_key="v1f_E_total", top_k=20)
    diverse = [candidates[i] for i in diverse_idx]

    pool: dict[str, dict[str, Any]] = {}
    for c in top50 + by_unc + diverse:
        pool[_candidate_key(c)] = c

    elites = top50[:10]
    for _ in range(3):
        for elite in elites[:5]:
            if elite.get("insertion"):
                for _ in range(8):
                    pert = dict(elite)
                    pert["insertion"] = _perturb_param_dict(elite["insertion"], SEARCH_SPACE, rng)
                    pool[_candidate_key(pert)] = pert
            elif elite.get("transport"):
                for _ in range(8):
                    pert = dict(elite)
                    pert["transport"] = _perturb_param_dict(elite["transport"], TRANSPORT_SEARCH_SPACE, rng)
                    pool[_candidate_key(pert)] = pert
            else:
                for _ in range(5):
                    pert = dict(elite)
                    if elite.get("grasp_lift"):
                        pert["grasp_lift"] = _perturb_param_dict(elite["grasp_lift"], GRASP_SEARCH_SPACE, rng)
                    pool[_candidate_key(pert)] = pert
    return list(pool.values())


def _baseline_e(hdf5: Path, demo_key: str, cfg: dict[str, Any], cem_report: Path) -> float:
    from run_v1f_plus_rollout_sampling import _baseline_rollout

    active = cfg.get("search_kind", "insertion")
    if active == "insertion":
        active = "insertion"
    elif active == "transport":
        active = "transport"
    elif active == "lift":
        active = "lift"
    else:
        active = "grasp"
    baseline = _baseline_rollout(hdf5, demo_key, active, cem_report)
    return float(baseline.get("E_total_norm", baseline.get("E_total", 999.0)))


def run_targeted_for_demo(
    *,
    demo_group: str,
    demo_key: str,
    cfg: dict[str, Any],
    hdf5: Path,
    cem_report: Path,
    v1f_model: Path,
    v1e_model: Path,
    seed: int,
) -> list[dict[str, Any]]:
    context = extract_repair_context_v1f(
        context_source="original_failed_context",
        failed_hdf5=hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
        cem_report=cem_report,
    )
    baseline_e = _baseline_e(hdf5, demo_key, cfg, cem_report)
    rng = random.Random(seed + hash(demo_key) % 10000)
    candidates = _build_candidate_pool(
        cfg=cfg, context=context, v1f_model=v1f_model, v1e_model=v1e_model, num_samples=1000, seed=seed
    )
    rollout_cands = _select_rollout_candidates(candidates, rng=rng)

    records: list[dict[str, Any]] = []
    success_count = 0
    best_e = baseline_e

    for i, cand in enumerate(rollout_cands):
        rollout = run_repair_rollout(
            failed_hdf5=hdf5,
            demo_key=demo_key,
            search_kind=cfg["search_kind"],
            cem_report=cem_report if cfg["search_kind"] in ("insertion", "transport") else None,
            candidate=cand,
        )
        rollout["sampling_index"] = i
        rollout["active_param_group"] = cfg["active"]
        rollout["demo_key"] = demo_key
        rollout["object_poses_modified"] = False
        rollout["success_from_rollout"] = True
        e = float(rollout.get("E_total_norm", rollout.get("E_total", 999.0)))
        best_e = min(best_e, e)
        if rollout.get("success_flag"):
            success_count += 1
        row = _rollout_record_to_jsonl_row(
            source_file=hdf5,
            demo_key=demo_key,
            failure_type=cfg.get("rough_failure_type", cfg["failure_type"]),
            context=context,
            rollout=rollout,
            active=cfg["active"],
        )
        row["source"] = "v1f_targeted_repair_sampling"
        row["targeted_demo_group"] = demo_group
        records.append(row)
        if success_count >= TARGET_SUCCESS:
            break
        if success_count == 0 and (baseline_e - best_e) / max(baseline_e, 1e-6) >= E_DROP_RATIO and i >= 80:
            break
        if i >= 250:
            break

    print(
        f"  targeted {demo_group}/{demo_key}: rollouts={len(records)} successes={success_count} "
        f"best_E={best_e:.2f} baseline_E={baseline_e:.2f}",
        flush=True,
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted repair rollout sampling")
    parser.add_argument("--quick-csv", type=Path, default=DEFAULT_QUICK_CSV)
    parser.add_argument("--balanced-model", type=Path, default=DEFAULT_BALANCED_MODEL)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--old-failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--new-failed-hdf5", type=Path, default=NEW_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-keys", type=str, default="", help="Optional comma-separated override")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.demo_keys.strip():
        targets = [{"demo_group": "new", "demo_key": k.strip(), "failure_type": ""} for k in args.demo_keys.split(",") if k.strip()]
    else:
        if not args.quick_csv.exists():
            raise SystemExit(f"Quick summary not found: {args.quick_csv}")
        targets = load_low_repair_demos(args.quick_csv)

    failure_map = load_failure_map(args.audit_report)
    all_records: list[dict[str, Any]] = []
    manifest_targets: list[dict[str, Any]] = []

    for t in targets:
        demo_group = t["demo_group"]
        demo_key = t["demo_key"]
        hdf5 = args.old_failed_hdf5 if demo_group == "old" else args.new_failed_hdf5
        if demo_group == "old":
            cfg = DEMO_REPAIR_CONFIGS[demo_key]
        else:
            cfg = _repair_cfg_for_new_demo(demo_key, failure_map)
        print(f"targeted sampling {demo_group}/{demo_key}", flush=True)
        recs = run_targeted_for_demo(
            demo_group=demo_group,
            demo_key=demo_key,
            cfg=cfg,
            hdf5=hdf5,
            cem_report=args.cem_report,
            v1f_model=args.balanced_model,
            v1e_model=args.v1e_model,
            seed=args.seed,
        )
        all_records.extend(recs)
        manifest_targets.append({**t, "num_rollout_samples": len(recs), "num_success": sum(1 for r in recs if r["success_flag"])})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for rec in all_records:
            handle.write(json.dumps(rec, default=str) + "\n")

    summary = {"targets": manifest_targets, "num_samples": len(all_records), "output": str(args.output)}
    (args.output.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
