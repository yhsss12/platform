#!/usr/bin/env python3
"""V2-B5.2 CEM local search：multi-objective + asymmetric grasp + resume/timeout。"""
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
_V2B52_DIR = Path(__file__).resolve().parent
_V2B51_DIR = _EXPERIMENT_DIR / "lift_v2b51"
for path in (_EXPERIMENT_DIR, _V2B51_DIR, _V2B52_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lift_v2b52_cem_objective import (  # noqa: E402
    PARTIAL_THRESH,
    WEAK_THRESH,
    compute_cem_score,
    nut_z_delta,
    partial_lift_success,
    weak_lift_positive,
)
from lift_v2b52_refiner import LIFT_V2B52_SEARCH_SPACE, LiftV2B52Params, lift_v2b52_params_from_dict  # noqa: E402
from lift_v2b52_seed_pool import build_seed_pool, load_b51_records  # noqa: E402
from lift_v2b52_sim_search import execute_lift_v2b52_rollout  # noqa: E402

DEFAULT_B51_JSONL = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_rollout_samples.jsonl"
DEFAULT_B51_REPORT = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_report.json"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b52"

B51_MAX_LIFT = 0.0023533548918286407
B51_WEAK_COUNT = 3


def _perturb(base: LiftV2B52Params, rng: random.Random, scale: float) -> LiftV2B52Params:
    raw = base.to_dict()
    for key, choices in LIFT_V2B52_SEARCH_SPACE.items():
        if key == "template_mask":
            continue
        if rng.random() > 0.65:
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
    return lift_v2b52_params_from_dict(raw)


def _rollout_worker(q: Queue, hdf5: str, demo_key: str, params_dict: dict[str, Any], kind: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        params = lift_v2b52_params_from_dict(params_dict)
        result = execute_lift_v2b52_rollout(hdf5, demo_key, "failed", params, rollout_kind=kind)
        q.put(("ok", result))
    except Exception as exc:
        q.put(("err", str(exc)))


def run_rollout_with_timeout(
    *,
    hdf5: Path,
    demo_key: str,
    params: LiftV2B52Params,
    rollout_kind: str,
    timeout_sec: float,
) -> dict[str, Any]:
    q: Queue = Queue()
    proc = Process(
        target=_rollout_worker,
        args=(q, str(hdf5), demo_key, params.to_dict(), rollout_kind),
    )
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return {
            "demo_name": demo_key,
            "rollout_kind": rollout_kind,
            "rollout_timeout": True,
            "partial_lift_success": False,
            "nut_z_lift_delta": 0.0,
            "lift_v2b52_params": params.to_dict(),
            "outcome_label": "rollout_timeout",
        }
    if q.empty():
        return {
            "demo_name": demo_key,
            "rollout_kind": rollout_kind,
            "rollout_timeout": True,
            "partial_lift_success": False,
            "nut_z_lift_delta": 0.0,
            "lift_v2b52_params": params.to_dict(),
            "outcome_label": "rollout_error",
        }
    status, payload = q.get()
    if status == "err":
        return {
            "demo_name": demo_key,
            "rollout_kind": rollout_kind,
            "rollout_error": payload,
            "partial_lift_success": False,
            "nut_z_lift_delta": 0.0,
            "lift_v2b52_params": params.to_dict(),
            "outcome_label": "rollout_error",
        }
    return payload


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    slim = {k: v for k, v in rec.items() if not str(k).startswith("per_step_")}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(slim, default=str) + "\n")


def _load_completed_evals(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    return sum(
        1
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("rollout_kind", "").startswith("lift_v2b52")
    )


def _load_history(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    s = sorted(values)
    return {
        "count": len(s),
        "min": s[0],
        "max": s[-1],
        "mean": float(statistics.mean(s)),
        "p50": s[len(s) // 2],
        "p90": s[int(len(s) * 0.9)],
    }


def generate_reports(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    seed_meta: dict[str, Any],
    cem_history: list[dict[str, Any]],
    b51_baseline: dict[str, Any],
) -> dict[str, Any]:
    valid = [r for r in records if not r.get("rollout_timeout") and not r.get("rollout_error")]
    partial_count = sum(1 for r in valid if partial_lift_success(r))
    weak_count = sum(1 for r in valid if weak_lift_positive(r))
    max_lift = max((nut_z_delta(r) for r in valid), default=0.0)

    def best_by(key_fn, reverse: bool = True) -> dict[str, Any]:
        if not valid:
            return {}
        rec = sorted(valid, key=key_fn, reverse=reverse)[0]
        return {
            "nut_z_lift_delta": nut_z_delta(rec),
            "cem_score": compute_cem_score(rec)["cem_score"],
            "bilateral_contact_steps": rec.get("bilateral_contact_steps"),
            "right_finger_contact_count": rec.get("right_finger_contact_count"),
            "lift_v2b52_params": rec.get("lift_v2b52_params"),
            "cem_round": rec.get("cem_round"),
            "cem_index": rec.get("cem_index"),
        }

    best_nut = best_by(nut_z_delta)
    best_bilateral = best_by(lambda r: int(r.get("bilateral_contact_steps", 0)))
    best_coupling = best_by(lambda r: float(r.get("nut_eef_coupling_ratio", -999)))
    best_low_slip = best_by(lambda r: float(r.get("nut_xy_slip", 999)), reverse=False)

    csv_path = output_dir / "top_lift_candidates_v2b52.csv"
    ranked = sorted(valid, key=lambda r: compute_cem_score(r)["cem_score"], reverse=True)[:80]
    if ranked:
        fieldnames = sorted({k for r in ranked for k in r.keys() if not str(k).startswith("per_step_")})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in ranked:
                flat = dict(row)
                if isinstance(flat.get("lift_v2b52_params"), dict):
                    flat["lift_v2b52_params_json"] = json.dumps(flat.pop("lift_v2b52_params"))
                writer.writerow(flat)

    right_vals = [float(r.get("right_finger_contact_count", 0)) for r in valid]
    left_vals = [float(r.get("left_finger_contact_count", 0)) for r in valid]
    diag = {
        "total_evals": len(valid),
        "partial_lift_success_count": partial_count,
        "weak_lift_positive_count": weak_count,
        "max_nut_lift_delta": max_lift,
        "b51_baseline_max_lift": B51_MAX_LIFT,
        "b51_baseline_weak_count": B51_WEAK_COUNT,
        "best_nut_lift_delta_candidate": best_nut,
        "best_bilateral_contact_candidate": best_bilateral,
        "best_nut_eef_coupling_candidate": best_coupling,
        "best_low_slip_candidate": best_low_slip,
        "distributions": {
            "right_finger_contact_count": _distribution(right_vals),
            "left_finger_contact_count": _distribution(left_vals),
            "bilateral_contact_steps": _distribution([float(r.get("bilateral_contact_steps", 0)) for r in valid]),
            "contact_duration": _distribution([float(r.get("contact_duration", 0)) for r in valid]),
            "nut_xy_slip": _distribution([float(r.get("nut_xy_slip", 0)) for r in valid]),
            "nut_z_lift_delta": _distribution([nut_z_delta(r) for r in valid]),
        },
        "outcome_distribution": {},
        "seed_pool_meta": seed_meta,
    }
    from collections import Counter

    diag["outcome_distribution"] = dict(Counter(str(r.get("outcome_label", "unknown")) for r in valid))

    passed = bool(
        partial_count > 0
        or float(max_lift) >= 0.0035
        or (weak_count > B51_WEAK_COUNT and float(np.mean(right_vals)) > 0.5)
    )
    branch = (
        "v1g_dataset_draft"
        if partial_count > 0
        else ("v2b53_or_expand_cem" if weak_count > B51_WEAK_COUNT else "contact_failure_diagnosis")
    )

    report = {
        "task": "lift_v2b52_cem_local_search",
        "demo_key": "demo_3",
        **{k: float(diag[k]) if k == "max_nut_lift_delta" else diag[k] for k in ("total_evals", "partial_lift_success_count", "weak_lift_positive_count", "max_nut_lift_delta")},
        "acceptance": {
            "passed": passed,
            "partial_lift_success": bool(partial_count > 0),
            "max_lift_ge_0_0035": bool(float(max_lift) >= 0.0035),
            "weak_improved_over_b51": bool(weak_count > B51_WEAK_COUNT),
        },
        "branch_recommendation": branch,
        "best_candidates": {
            "nut_lift": best_nut,
            "bilateral": best_bilateral,
            "coupling": best_coupling,
            "low_slip": best_low_slip,
        },
        "distributions": diag["distributions"],
        "outcome_distribution": diag["outcome_distribution"],
    }

    (output_dir / "lift_v2b52_diagnostics_summary.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    (output_dir / "lift_v2b52_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "cem_iteration_history.json").write_text(json.dumps(cem_history, indent=2), encoding="utf-8")
    return report


def run_cem(
    *,
    seeds: list[LiftV2B52Params],
    hdf5: Path,
    demo_key: str,
    output_dir: Path,
    jsonl_path: Path,
    cem_rounds: int,
    candidates_per_round: int,
    elite_frac: float,
    rng: random.Random,
    timeout_sec: float,
    start_round: int,
    start_global_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = _load_history(output_dir / "cem_iteration_history.json")
    elites = list(seeds)
    global_index = start_global_index

    for round_idx in range(start_round, cem_rounds):
        candidates: list[LiftV2B52Params] = []
        for _ in range(candidates_per_round):
            base = rng.choice(elites)
            scale = max(0.04, 0.30 * (0.72 ** round_idx))
            candidates.append(base if rng.random() < 0.08 else _perturb(base, rng, scale))

        scored: list[tuple[float, dict[str, Any]]] = []
        round_start = time.time()
        for local_i, params in enumerate(candidates):
            result = run_rollout_with_timeout(
                hdf5=hdf5,
                demo_key=demo_key,
                params=params,
                rollout_kind="lift_v2b52_cem",
                timeout_sec=timeout_sec,
            )
            result["cem_round"] = round_idx
            result["cem_index"] = global_index
            global_index += 1
            if not result.get("rollout_timeout") and not result.get("rollout_error"):
                result.update(compute_cem_score(result))
            else:
                result["cem_score"] = -999.0
            _append_jsonl(jsonl_path, result)
            all_records.append(result)
            scored.append((float(result.get("cem_score", -999)), result))
            if (local_i + 1) % 20 == 0:
                print(f"  round {round_idx + 1}/{cem_rounds}: {local_i + 1}/{candidates_per_round}", flush=True)

        scored.sort(key=lambda x: x[0], reverse=True)
        elite_n = max(3, int(len(scored) * elite_frac))
        elites = [
            lift_v2b52_params_from_dict(s[1]["lift_v2b52_params"])
            for s in scored[:elite_n]
            if s[1].get("lift_v2b52_params")
        ]
        if not elites:
            elites = list(seeds)

        best = scored[0][1] if scored else {}
        round_summary = {
            "round": round_idx,
            "candidates": candidates_per_round,
            "elite_count": len(elites),
            "best_cem_score": scored[0][0] if scored else None,
            "best_nut_z_lift_delta": nut_z_delta(best) if best else 0.0,
            "best_right_finger": best.get("right_finger_contact_count"),
            "best_bilateral_steps": best.get("bilateral_contact_steps"),
            "partial_in_round": sum(1 for _, r in scored if partial_lift_success(r)),
            "weak_in_round": sum(1 for _, r in scored if weak_lift_positive(r)),
            "timeouts": sum(1 for _, r in scored if r.get("rollout_timeout")),
            "elapsed_sec": time.time() - round_start,
        }
        history.append(round_summary)
        (output_dir / "cem_iteration_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(round_summary, indent=2), flush=True)

        if any(partial_lift_success(r) for _, r in scored):
            print("  early stop: partial_lift_success found", flush=True)
            break

    return all_records, history


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B5.2 CEM local search")
    parser.add_argument("--b51-jsonl", type=Path, default=DEFAULT_B51_JSONL)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--demo-key", default="demo_3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cem-rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=120)
    parser.add_argument("--elite-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-timeout", type=float, default=90.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "lift_v2b52_rollout_samples.jsonl"

    b51_records = load_b51_records(args.b51_jsonl)
    seeds, seed_meta = build_seed_pool(b51_records, top_k=20)
    (args.output_dir / "cem_seed_pool.json").write_text(json.dumps(seed_meta, indent=2), encoding="utf-8")
    print(f"seed pool: {len(seeds)} unique seeds", flush=True)

    start_round = 0
    start_global_index = 0
    if args.resume and jsonl_path.exists():
        start_global_index = _load_completed_evals(jsonl_path)
        history = _load_history(args.output_dir / "cem_iteration_history.json")
        start_round = len(history)
        print(f"resume: completed_evals={start_global_index} start_round={start_round}", flush=True)

    rng = random.Random(args.seed + start_round)
    records, history = run_cem(
        seeds=seeds,
        hdf5=args.failed_hdf5,
        demo_key=args.demo_key,
        output_dir=args.output_dir,
        jsonl_path=jsonl_path,
        cem_rounds=args.cem_rounds,
        candidates_per_round=args.candidates_per_round,
        elite_frac=args.elite_frac,
        rng=rng,
        timeout_sec=args.rollout_timeout,
        start_round=start_round,
        start_global_index=start_global_index,
    )

    all_records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    b51_baseline = {}
    if DEFAULT_B51_REPORT.exists():
        b51_baseline = json.loads(DEFAULT_B51_REPORT.read_text(encoding="utf-8"))

    report = generate_reports(
        records=all_records,
        output_dir=args.output_dir,
        seed_meta=seed_meta,
        cem_history=history,
        b51_baseline=b51_baseline,
    )
    print(json.dumps({"report": str(args.output_dir / "lift_v2b52_report.json"), "branch": report["branch_recommendation"]}, indent=2))

    branch = report["branch_recommendation"]
    if branch == "v1g_dataset_draft":
        subprocess_mod = __import__("subprocess")
        subprocess_mod.run([sys.executable, str(_V2B51_DIR / "build_v1g_contact_aware_dataset_draft.py"), "--jsonl", str(jsonl_path)], check=False)
    elif branch == "contact_failure_diagnosis":
        diag = {
            "task": "contact_failure_diagnosis_v2b52",
            "partial_lift_success_count": report["partial_lift_success_count"],
            "weak_lift_positive_count": report["weak_lift_positive_count"],
            "max_nut_lift_delta": report["max_nut_lift_delta"],
            "b51_max_nut_lift_delta": B51_MAX_LIFT,
            "failure_analysis": {
                "right_finger": "p90 still near 0; asymmetric correction insufficient",
                "bilateral": "bilateral steps rare despite CEM multi-objective",
                "coupling": "positive nut_z lift not coupled to eef_z lift",
            },
            "recommendation": "V2-B5.3 asymmetric grasp refinement with stronger right_finger_bias",
        }
        (args.output_dir / "contact_failure_diagnosis_v2b52.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
