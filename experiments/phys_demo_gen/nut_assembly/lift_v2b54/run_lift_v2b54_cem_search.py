#!/usr/bin/env python3
"""V2-B5.4 Lift-preserving Transport CEM (quick 4×100)."""
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
_V2B54_DIR = Path(__file__).resolve().parent
for path in (_EXPERIMENT_DIR, _EXPERIMENT_DIR / "lift_v2b53", _EXPERIMENT_DIR / "lift_v2b52", _EXPERIMENT_DIR / "lift_v2b51", _V2B54_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lift_v2b54_objective import (  # noqa: E402
    B53_WEAK_COUNT,
    CSV_REQUIRED_COLUMNS,
    DEMO3_BASELINE_PEG_XY,
    PARTIAL_THRESH,
    SUCCESS_LIFT_P50,
    compute_lift_preserving_score,
    flatten_for_csv,
    is_elite_eligible,
    min_peg_xy,
    nut_z_delta,
    partial_lift_success,
    peg_xy,
    p0_gate_pass,
    transport_improved,
    weak_lift_positive,
)
from lift_v2b54_refiner import LIFT_V2B54_SEARCH_SPACE, LiftV2B54Params, lift_v2b54_params_from_dict  # noqa: E402
from lift_v2b54_seed_pool import build_seed_pool  # noqa: E402
from lift_v2b54_sim_search import execute_lift_v2b54_rollout  # noqa: E402

DEFAULT_B51 = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_rollout_samples.jsonl"
DEFAULT_B52 = _EXPERIMENT_DIR / "outputs" / "lift_v2b52" / "lift_v2b52_rollout_samples.jsonl"
DEFAULT_B53 = _EXPERIMENT_DIR / "outputs" / "lift_v2b53" / "lift_v2b53_rollout_samples.jsonl"
DEFAULT_B53_CSV = _EXPERIMENT_DIR / "outputs" / "lift_v2b53" / "top_lift_transport_candidates_v2b53.csv"
DEFAULT_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b54"
B53_MAX_LIFT = 0.002127777001830755


def _perturb(base: LiftV2B54Params, rng: random.Random, scale: float) -> LiftV2B54Params:
    raw = base.to_dict()
    for key, choices in LIFT_V2B54_SEARCH_SPACE.items():
        if key == "template_mask":
            continue
        if rng.random() > 0.62:
            continue
        center = raw[key]
        if isinstance(center, str):
            raw[key] = rng.choice(choices)
        elif isinstance(center, int):
            delta = int(round(rng.gauss(0, max(1.0, scale * 4))))
            raw[key] = int(np.clip(center + delta, min(choices), max(choices)))
        else:
            span = float(max(choices) - min(choices)) if len(choices) > 1 else 1.0
            raw[key] = float(np.clip(center + rng.gauss(0, span * scale), min(choices), max(choices)))
    return lift_v2b54_params_from_dict(raw)


def _rollout_worker(q: Queue, hdf5: str, demo_key: str, params_dict: dict[str, Any], kind: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        params = lift_v2b54_params_from_dict(params_dict)
        q.put(("ok", execute_lift_v2b54_rollout(hdf5, demo_key, "failed", params, rollout_kind=kind)))
    except Exception as exc:
        q.put(("err", str(exc)))


def run_rollout_with_timeout(
    *, hdf5: Path, demo_key: str, params: LiftV2B54Params, kind: str, timeout_sec: float
) -> dict[str, Any]:
    q: Queue = Queue()
    proc = Process(target=_rollout_worker, args=(q, str(hdf5), demo_key, params.to_dict(), kind))
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return {"rollout_timeout": True, "rollout_kind": kind, "nut_z_lift_delta": 0.0, "lift_v2b54_params": params.to_dict()}
    if q.empty():
        return {"rollout_timeout": True, "rollout_kind": kind, "nut_z_lift_delta": 0.0}
    status, payload = q.get()
    if status == "err":
        return {"rollout_error": payload, "rollout_kind": kind, "nut_z_lift_delta": 0.0}
    return payload


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    slim = {k: v for k, v in rec.items() if not str(k).startswith("per_step_")}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(slim, default=str) + "\n")


def _load_completed(jsonl: Path) -> int:
    if not jsonl.exists():
        return 0
    return sum(
        1
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("rollout_kind", "").startswith("lift_v2b54")
    )


def _load_history(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    s = sorted(values)
    return {"count": len(s), "min": s[0], "max": s[-1], "mean": float(statistics.mean(s)), "p50": s[len(s) // 2], "p90": s[int(len(s) * 0.9)]}


def generate_reports(
    *, records: list[dict[str, Any]], output_dir: Path, seed_meta: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any]:
    valid = [r for r in records if not r.get("rollout_timeout") and not r.get("rollout_error")]
    partial_count = sum(1 for r in valid if partial_lift_success(r))
    weak_count = sum(1 for r in valid if weak_lift_positive(r))
    max_lift = max((nut_z_delta(r) for r in valid), default=0.0)
    min_final = min((peg_xy(r) for r in valid), default=DEMO3_BASELINE_PEG_XY)
    min_min = min((min_peg_xy(r) for r in valid), default=DEMO3_BASELINE_PEG_XY)
    gated = [r for r in valid if is_elite_eligible(r)]
    p0_pass = sum(1 for r in valid if p0_gate_pass(r))

    def best(pool: list[dict[str, Any]], key_fn) -> dict[str, Any]:
        if not pool:
            return {}
        rec = sorted(pool, key=key_fn, reverse=True)[0]
        s = compute_lift_preserving_score(rec)
        return {
            "nut_z_lift_delta": nut_z_delta(rec),
            "final_nut_peg_xy": peg_xy(rec),
            "min_nut_peg_xy": min_peg_xy(rec),
            "lift_preserving_score": s["lift_preserving_score"],
            "right_finger_contact_count": rec.get("right_finger_contact_count"),
            "bilateral_contact_steps": rec.get("bilateral_contact_steps"),
            "nut_eef_coupling_ratio": rec.get("nut_eef_coupling_ratio"),
            "residual_breakdown": s.get("residual_breakdown"),
            "lift_v2b54_params": rec.get("lift_v2b54_params"),
            "cem_index": rec.get("cem_index"),
        }

    best_weak = best([r for r in valid if weak_lift_positive(r)] or valid, nut_z_delta)
    best_joint = best(gated or [r for r in valid if p0_gate_pass(r)] or valid, lambda r: compute_lift_preserving_score(r)["lift_preserving_score"])

    ranked = sorted(valid, key=lambda r: compute_lift_preserving_score(r)["lift_preserving_score"], reverse=True)[:80]
    csv_path = output_dir / "top_lift_transport_candidates_v2b54.csv"
    if ranked:
        fields = CSV_REQUIRED_COLUMNS + ["cem_round", "cem_index", "outcome_label", "lift_v2b54_params_json"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in ranked:
                w.writerow(flatten_for_csv(row))

    passed = bool(
        partial_count > 0
        or (max_lift >= PARTIAL_THRESH and min_final < DEMO3_BASELINE_PEG_XY * 0.95)
        or (weak_count > B53_WEAK_COUNT and best_weak.get("final_nut_peg_xy", 999) <= 0.30)
    )
    branch = "v1g_dataset_draft_candidate" if partial_count > 0 else "await_or_diagnose"

    report = {
        "task": "lift_v2b54_lift_preserving_transport",
        "demo_key": "demo_3",
        "total_evals": len(valid),
        "partial_lift_success_count": partial_count,
        "weak_lift_positive_count": weak_count,
        "max_nut_lift_delta": float(max_lift),
        "min_final_nut_peg_xy": float(min_final),
        "min_min_nut_peg_xy": float(min_min),
        "p0_gate_pass_count": p0_pass,
        "elite_eligible_count": len(gated),
        "b53_baseline": {"weak_count": B53_WEAK_COUNT, "max_lift": B53_MAX_LIFT, "min_final_peg": 0.273},
        "acceptance": {
            "passed": passed,
            "partial_lift_success": bool(partial_count > 0),
            "lift_and_transport_joint": bool(max_lift >= PARTIAL_THRESH and min_final < DEMO3_BASELINE_PEG_XY * 0.95),
            "weak_plus_transport": bool(weak_count > B53_WEAK_COUNT and best_weak.get("final_nut_peg_xy", 999) <= 0.30),
        },
        "branch_recommendation": branch,
        "best_candidates": {"weak_lift": best_weak, "lift_preserving_joint": best_joint},
        "residual_priority": {
            "P0": "bilateral + transport_improved (+ weak lift for elite)",
            "P1": "maximize nut_z_lift_delta, target 0.005m, ref p50 0.039m",
            "P2": "minimize peg xy, preserve B5.3 transport gains",
            "P3": "coupling, slip, lift_stability",
        },
        "training_policy": {
            "v1g_pinn_auto_train": False,
            "b53_not_lift_success_training": True,
            "all_rollouts_marked_training_eligible_false": True,
        },
        "distributions": {
            "nut_z_lift_delta": _dist([nut_z_delta(r) for r in valid]),
            "final_nut_peg_xy": _dist([peg_xy(r) for r in valid]),
            "min_nut_peg_xy": _dist([min_peg_xy(r) for r in valid]),
        },
        "seed_pool_meta": seed_meta,
        "cem_iteration_history": history,
    }

    lift_report = {**report, "task": "lift_preserving_transport_report"}
    (output_dir / "lift_v2b54_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "lift_preserving_transport_report.json").write_text(json.dumps(lift_report, indent=2), encoding="utf-8")
    (output_dir / "cem_iteration_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if partial_count == 0 and not passed:
        diag = {
            "task": "lift_preserving_transport_failure_diagnosis",
            "partial_lift_success_count": partial_count,
            "weak_lift_positive_count": weak_count,
            "max_nut_lift_delta": float(max_lift),
            "p0_gate_pass_count": p0_pass,
            "elite_eligible_count": len(gated),
            "failure_analysis": {
                "contact_transport_without_lift": (
                    f"p0_pass={p0_pass} but weak={weak_count}; contact+transport established "
                    "without nut_z lift conversion to carry"
                ),
                "lift_ceiling": f"max_lift={max_lift:.5f}m vs partial={PARTIAL_THRESH}m",
                "transport_maintained": f"min_final_peg={min_final:.4f}m vs baseline 0.33m",
                "coupling": "nut_eef_coupling remains low when lift weak — nut not following gripper vertically",
            },
            "recommendation": "B5.5: stronger pre-lift reclose + slower lift before any lateral transport waypoint",
            "v1g_training": "FORBIDDEN until partial_lift_success > 0",
        }
        (output_dir / "lift_preserving_transport_failure_diagnosis.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    elif partial_count > 0:
        draft = {
            "status": "awaiting_user_confirmation",
            "auto_train_forbidden": True,
            "partial_candidates": [
                {"cem_index": r.get("cem_index"), "nut_z_lift_delta": nut_z_delta(r), "min_nut_peg_xy": min_peg_xy(r)}
                for r in valid
                if partial_lift_success(r)
            ][:20],
        }
        (output_dir / "v1g_dataset_draft_candidate.json").write_text(json.dumps(draft, indent=2), encoding="utf-8")

    return report


def run_cem(
    *,
    seeds: list[LiftV2B54Params],
    hdf5: Path,
    demo_key: str,
    output_dir: Path,
    jsonl: Path,
    rounds: int,
    per_round: int,
    elite_frac: float,
    rng: random.Random,
    timeout: float,
    start_round: int,
    start_idx: int,
) -> list[dict[str, Any]]:
    history = _load_history(output_dir / "cem_iteration_history.json")
    elites = list(seeds)
    gidx = start_idx

    for rnd in range(start_round, rounds):
        cands = []
        for _ in range(per_round):
            base = rng.choice(elites)
            scale = max(0.04, 0.26 * (0.72 ** rnd))
            cands.append(base if rng.random() < 0.08 else _perturb(base, rng, scale))

        scored: list[tuple[float, dict[str, Any]]] = []
        t0 = time.time()
        for i, params in enumerate(cands):
            result = run_rollout_with_timeout(hdf5=hdf5, demo_key=demo_key, params=params, kind="lift_v2b54_cem", timeout_sec=timeout)
            result["cem_round"] = rnd
            result["cem_index"] = gidx
            gidx += 1
            if not result.get("rollout_timeout") and not result.get("rollout_error"):
                result.update(compute_lift_preserving_score(result))
            else:
                result["lift_preserving_score"] = -9999.0
            _append_jsonl(jsonl, result)
            scored.append((float(result.get("lift_preserving_score", -9999)), result))
            if (i + 1) % 20 == 0:
                print(f"  round {rnd + 1}/{rounds}: {i + 1}/{per_round}", flush=True)

        scored.sort(key=lambda x: x[0], reverse=True)
        en = max(3, int(len(scored) * elite_frac))
        gated = [
            lift_v2b54_params_from_dict(s[1]["lift_v2b54_params"])
            for s in scored
            if is_elite_eligible(s[1]) and s[1].get("lift_v2b54_params")
        ]
        elites = gated[:en] or list(seeds)[:en]
        best = scored[0][1] if scored else {}
        summary = {
            "round": rnd,
            "candidates": per_round,
            "elite_count": len(elites),
            "gated_elite_count": len(gated[:en]),
            "best_lift_preserving_score": scored[0][0] if scored else None,
            "best_nut_z_lift_delta": nut_z_delta(best),
            "best_final_nut_peg_xy": peg_xy(best),
            "partial_in_round": sum(1 for _, r in scored if partial_lift_success(r)),
            "weak_in_round": sum(1 for _, r in scored if weak_lift_positive(r)),
            "p0_pass_in_round": sum(1 for _, r in scored if p0_gate_pass(r)),
            "elapsed_sec": time.time() - t0,
        }
        history.append(summary)
        (output_dir / "cem_iteration_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if any(partial_lift_success(r) for _, r in scored):
            print("  early stop: partial lift found", flush=True)
            break
    return history


def main() -> int:
    p = argparse.ArgumentParser(description="V2-B5.4 lift-preserving transport CEM")
    p.add_argument("--failed-hdf5", type=Path, default=DEFAULT_HDF5)
    p.add_argument("--demo-key", default="demo_3")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--b51-jsonl", type=Path, default=DEFAULT_B51)
    p.add_argument("--b52-jsonl", type=Path, default=DEFAULT_B52)
    p.add_argument("--b53-jsonl", type=Path, default=DEFAULT_B53)
    p.add_argument("--b53-csv", type=Path, default=DEFAULT_B53_CSV)
    p.add_argument("--cem-rounds", type=int, default=4)
    p.add_argument("--candidates-per-round", type=int, default=100)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--rollout-timeout", type=float, default=90.0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_dir / "lift_v2b54_rollout_samples.jsonl"

    seeds, seed_meta = build_seed_pool(
        b51_jsonl=args.b51_jsonl,
        b52_jsonl=args.b52_jsonl,
        b53_jsonl=args.b53_jsonl,
        b53_csv=args.b53_csv,
        top_k=20,
        rng=random.Random(args.seed),
    )
    (args.output_dir / "cem_seed_pool.json").write_text(json.dumps(seed_meta, indent=2), encoding="utf-8")
    print(f"seed pool: {len(seeds)} unique seeds", flush=True)

    start_round, start_idx = 0, 0
    if args.resume and jsonl.exists():
        start_idx = _load_completed(jsonl)
        start_round = len(_load_history(args.output_dir / "cem_iteration_history.json"))
        print(f"resume: evals={start_idx} round={start_round}", flush=True)

    history = run_cem(
        seeds=seeds,
        hdf5=args.failed_hdf5,
        demo_key=args.demo_key,
        output_dir=args.output_dir,
        jsonl=jsonl,
        rounds=args.cem_rounds,
        per_round=args.candidates_per_round,
        elite_frac=args.elite_frac,
        rng=random.Random(args.seed + start_round),
        timeout=args.rollout_timeout,
        start_round=start_round,
        start_idx=start_idx,
    )

    records = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    report = generate_reports(records=records, output_dir=args.output_dir, seed_meta=seed_meta, history=history)
    print(json.dumps({"passed": report["acceptance"]["passed"], "branch": report["branch_recommendation"]}, indent=2))
    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
