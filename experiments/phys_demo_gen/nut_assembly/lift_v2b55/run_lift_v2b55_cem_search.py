#!/usr/bin/env python3
"""V2-B5.5 Pre-lift Reclose + Slow Vertical Lift CEM (3×80=240)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_V2B55_DIR = Path(__file__).resolve().parent
for p in (_EXPERIMENT_DIR, _EXPERIMENT_DIR / "lift_v2b54", _EXPERIMENT_DIR / "lift_v2b53",
          _EXPERIMENT_DIR / "lift_v2b52", _EXPERIMENT_DIR / "lift_v2b51", _V2B55_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lift_v2b55_objective import (  # noqa: E402
    B54_MAX_LIFT, B54_MIN_FINAL_PEG, CSV_COLUMNS, LIFT_CEILING_BREAK, PARTIAL_THRESH,
    compute_prelift_slow_lift_score, flatten_for_csv, is_elite_eligible, min_peg_xy,
    nut_z_delta, partial_lift_success, peg_xy, p0_gate_pass, weak_lift_positive,
)
from lift_v2b55_refiner import LIFT_V2B55_SEARCH_SPACE, LiftV2B55Params, lift_v2b55_params_from_dict  # noqa: E402
from lift_v2b55_seed_pool import build_seed_pool  # noqa: E402
from lift_v2b55_sim_search import execute_lift_v2b55_rollout  # noqa: E402

DEFAULT_B54_JSONL = _EXPERIMENT_DIR / "outputs" / "lift_v2b54" / "lift_v2b54_rollout_samples.jsonl"
DEFAULT_B54_CSV = _EXPERIMENT_DIR / "outputs" / "lift_v2b54" / "top_lift_transport_candidates_v2b54.csv"
DEFAULT_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b55"


def _perturb(base: LiftV2B55Params, rng: random.Random, scale: float) -> LiftV2B55Params:
    raw = base.to_dict()
    for key, choices in LIFT_V2B55_SEARCH_SPACE.items():
        if key == "template_mask" or rng.random() > 0.6:
            continue
        c = raw[key]
        if isinstance(c, str):
            raw[key] = rng.choice(choices)
        elif isinstance(c, int):
            raw[key] = int(np.clip(c + int(round(rng.gauss(0, max(1, scale * 4)))), min(choices), max(choices)))
        else:
            span = float(max(choices) - min(choices)) if len(choices) > 1 else 1.0
            raw[key] = float(np.clip(c + rng.gauss(0, span * scale), min(choices), max(choices)))
    return lift_v2b55_params_from_dict(raw)


def _worker(q: Queue, hdf5: str, demo_key: str, pd: dict, kind: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        q.put(("ok", execute_lift_v2b55_rollout(hdf5, demo_key, "failed", lift_v2b55_params_from_dict(pd), rollout_kind=kind)))
    except Exception as e:
        q.put(("err", str(e)))


def rollout_timeout(hdf5: Path, demo_key: str, params: LiftV2B55Params, kind: str, timeout: float) -> dict[str, Any]:
    q: Queue = Queue()
    p = Process(target=_worker, args=(q, str(hdf5), demo_key, params.to_dict(), kind))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        return {"rollout_timeout": True, "rollout_kind": kind, "nut_z_lift_delta": 0.0, "lift_v2b55_params": params.to_dict()}
    if q.empty():
        return {"rollout_timeout": True, "rollout_kind": kind}
    st, payload = q.get()
    return payload if st == "ok" else {"rollout_error": payload, "rollout_kind": kind}


def _append(path: Path, rec: dict) -> None:
    slim = {k: v for k, v in rec.items() if not str(k).startswith("per_step_")}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(slim, default=str) + "\n")


def generate_reports(*, records: list, out: Path, seed_meta: dict, history: list) -> dict:
    valid = [r for r in records if not r.get("rollout_timeout") and not r.get("rollout_error")]
    partial = sum(1 for r in valid if partial_lift_success(r))
    weak = sum(1 for r in valid if weak_lift_positive(r))
    max_lift = max((nut_z_delta(r) for r in valid), default=0.0)
    min_final = min((peg_xy(r) for r in valid), default=0.33)
    min_min = min((min_peg_xy(r) for r in valid), default=0.33)

    best_lift = max(valid, key=nut_z_delta, default={})
    best_joint = max(
        [r for r in valid if is_elite_eligible(r)] or valid,
        key=lambda r: compute_prelift_slow_lift_score(r)["prelift_slow_lift_score"],
        default={},
    )

    ranked = sorted(valid, key=lambda r: compute_prelift_slow_lift_score(r)["prelift_slow_lift_score"], reverse=True)[:60]
    if ranked:
        with (out / "top_lift_candidates_v2b55.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS + ["cem_round", "cem_index", "outcome_label", "lift_v2b55_params_json"], extrasaction="ignore")
            w.writeheader()
            for row in ranked:
                w.writerow(flatten_for_csv(row))

    passed_partial = partial > 0
    passed_ceiling = max_lift >= LIFT_CEILING_BREAK
    passed_weak = weak > 0 and max_lift > B54_MAX_LIFT

    if passed_partial:
        branch = "v1g_dataset_draft_candidate"
    elif passed_ceiling:
        branch = "lift_ceiling_breakthrough_b56_or_expand"
    elif max_lift < LIFT_CEILING_BREAK * 0.85:
        branch = "repairability_bottleneck_stop_search"
    else:
        branch = "marginal_lift_no_ceiling_break"

    report = {
        "task": "lift_v2b55_prelift_reclose_slow_vertical_lift",
        "demo_key": "demo_3",
        "total_evals": len(valid),
        "partial_lift_success_count": partial,
        "weak_lift_positive_count": weak,
        "max_nut_lift_delta": float(max_lift),
        "min_final_nut_peg_xy": float(min_final),
        "min_min_nut_peg_xy": float(min_min),
        "b54_baseline": {"max_lift": B54_MAX_LIFT, "min_final_peg": B54_MIN_FINAL_PEG, "weak_count": 18},
        "acceptance": {
            "passed_partial": passed_partial,
            "passed_ceiling_break_0_0035": passed_ceiling,
            "passed_vs_b54_max_lift": max_lift > B54_MAX_LIFT,
        },
        "branch_recommendation": branch,
        "best_candidates": {
            "max_lift": {"nut_z_lift_delta": nut_z_delta(best_lift), "final_nut_peg_xy": peg_xy(best_lift), "params": best_lift.get("lift_v2b55_params")},
            "best_joint": {"nut_z_lift_delta": nut_z_delta(best_joint), "final_nut_peg_xy": peg_xy(best_joint), "params": best_joint.get("lift_v2b55_params")},
        },
        "training_policy": {"v1g_auto_train": False, "exclude_from_lift_success_training": True},
        "seed_pool_meta": seed_meta,
        "cem_iteration_history": history,
    }
    (out / "lift_v2b55_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "prelift_reclose_slow_lift_report.json").write_text(json.dumps({**report, "task": "prelift_reclose_slow_lift_report"}, indent=2), encoding="utf-8")
    (out / "cem_iteration_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if passed_partial:
        draft = {"status": "awaiting_user_confirmation", "auto_train_forbidden": True,
                 "candidates": [{"cem_index": r.get("cem_index"), "nut_z_lift_delta": nut_z_delta(r)} for r in valid if partial_lift_success(r)][:20]}
        (out / "v1g_dataset_draft_candidate.json").write_text(json.dumps(draft, indent=2), encoding="utf-8")
    else:
        diag = {
            "task": "lift_ceiling_diagnosis",
            "partial_lift_success_count": partial,
            "max_nut_lift_delta": float(max_lift),
            "b54_max_lift": B54_MAX_LIFT,
            "lift_ceiling_break_threshold_m": LIFT_CEILING_BREAK,
            "ceiling_broken": passed_ceiling,
            "analysis": {
                "max_lift_vs_partial": f"max={max_lift:.5f}m, partial gate={PARTIAL_THRESH}m",
                "vs_b54": f"B5.4 max={B54_MAX_LIFT:.5f}m, B5.5 max={max_lift:.5f}m",
                "transport_peg": f"min_final_peg={min_final:.4f}m (B5.4={B54_MIN_FINAL_PEG:.4f}m)",
            },
            "conclusion": (
                "lift_ceiling_breakthrough" if passed_ceiling else
                "current refiner cannot produce stable lift-and-carry for demo_3; stop search line"
                if max_lift < LIFT_CEILING_BREAK * 0.85 else
                "marginal improvement; consider B5.6 expanded search"
            ),
            "v1g_training": "FORBIDDEN",
        }
        (out / "lift_ceiling_diagnosis.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    return report


def run_cem(*, seeds, hdf5, demo_key, out, jsonl, rounds, per_round, elite_frac, rng, timeout, start_round, start_idx):
    history = json.loads((out / "cem_iteration_history.json").read_text()) if (out / "cem_iteration_history.json").exists() else []
    elites = list(seeds)
    gidx = start_idx
    for rnd in range(start_round, rounds):
        cands = [rng.choice(elites) if rng.random() < 0.1 else _perturb(rng.choice(elites), rng, max(0.04, 0.28 * (0.72 ** rnd))) for _ in range(per_round)]
        scored = []
        t0 = time.time()
        for i, params in enumerate(cands):
            r = rollout_timeout(hdf5, demo_key, params, "lift_v2b55_cem", timeout)
            r["cem_round"], r["cem_index"] = rnd, gidx
            gidx += 1
            if not r.get("rollout_timeout") and not r.get("rollout_error"):
                r.update(compute_prelift_slow_lift_score(r))
            else:
                r["prelift_slow_lift_score"] = -9999.0
            _append(jsonl, r)
            scored.append((float(r.get("prelift_slow_lift_score", -9999)), r))
            if (i + 1) % 20 == 0:
                print(f"  round {rnd+1}/{rounds}: {i+1}/{per_round}", flush=True)
        scored.sort(key=lambda x: x[0], reverse=True)
        en = max(3, int(len(scored) * elite_frac))
        gated = [lift_v2b55_params_from_dict(s[1]["lift_v2b55_params"]) for s in scored if is_elite_eligible(s[1]) and s[1].get("lift_v2b55_params")]
        elites = gated[:en] or list(seeds)[:en]
        best = scored[0][1] if scored else {}
        summary = {
            "round": rnd, "candidates": per_round, "elite_count": len(elites),
            "best_score": scored[0][0] if scored else None,
            "best_nut_z_lift_delta": nut_z_delta(best),
            "best_final_nut_peg_xy": peg_xy(best),
            "partial_in_round": sum(1 for _, r in scored if partial_lift_success(r)),
            "weak_in_round": sum(1 for _, r in scored if weak_lift_positive(r)),
            "p0_pass_in_round": sum(1 for _, r in scored if p0_gate_pass(r)),
            "elapsed_sec": time.time() - t0,
        }
        history.append(summary)
        (out / "cem_iteration_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if any(partial_lift_success(r) for _, r in scored):
            break
    return history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed-hdf5", type=Path, default=DEFAULT_HDF5)
    ap.add_argument("--b54-jsonl", type=Path, default=DEFAULT_B54_JSONL)
    ap.add_argument("--b54-csv", type=Path, default=DEFAULT_B54_CSV)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cem-rounds", type=int, default=3)
    ap.add_argument("--candidates-per-round", type=int, default=80)
    ap.add_argument("--elite-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--rollout-timeout", type=float, default=90.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_dir / "lift_v2b55_rollout_samples.jsonl"

    seeds, seed_meta = build_seed_pool(b54_jsonl=args.b54_jsonl, b54_csv=args.b54_csv, rng=random.Random(args.seed))
    (args.output_dir / "cem_seed_pool.json").write_text(json.dumps(seed_meta, indent=2), encoding="utf-8")
    print(f"seed pool: {len(seeds)}", flush=True)

    sr, si = 0, 0
    if args.resume and jsonl.exists():
        si = sum(1 for l in jsonl.read_text().splitlines() if l.strip() and json.loads(l).get("rollout_kind", "").startswith("lift_v2b55"))
        sr = len(json.loads((args.output_dir / "cem_iteration_history.json").read_text())) if (args.output_dir / "cem_iteration_history.json").exists() else 0

    history = run_cem(seeds=seeds, hdf5=args.failed_hdf5, demo_key="demo_3", out=args.output_dir, jsonl=jsonl,
                      rounds=args.cem_rounds, per_round=args.candidates_per_round, elite_frac=args.elite_frac,
                      rng=random.Random(args.seed + sr), timeout=args.rollout_timeout, start_round=sr, start_idx=si)
    records = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    report = generate_reports(records=records, out=args.output_dir, seed_meta=seed_meta, history=history)
    print(json.dumps({"branch": report["branch_recommendation"], "max_lift": report["max_nut_lift_delta"]}, indent=2))
    return 0 if report["acceptance"]["passed_partial"] or report["acceptance"]["passed_ceiling_break_0_0035"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
