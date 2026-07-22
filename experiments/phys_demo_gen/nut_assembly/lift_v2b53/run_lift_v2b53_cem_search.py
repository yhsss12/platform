#!/usr/bin/env python3
"""V2-B5.3 Contact-gated Transport-and-Lift CEM refinement for demo_3."""
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
_V2B53_DIR = Path(__file__).resolve().parent
_V2B52_DIR = _EXPERIMENT_DIR / "lift_v2b52"
_V2B51_DIR = _EXPERIMENT_DIR / "lift_v2b51"
for path in (_EXPERIMENT_DIR, _V2B51_DIR, _V2B52_DIR, _V2B53_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lift_v2b53_objective import (  # noqa: E402
    CSV_REQUIRED_COLUMNS,
    DEMO3_BASELINE_PEG_XY,
    PARTIAL_THRESH,
    SUCCESS_LIFT_P50,
    compute_residual_breakdown,
    compute_transport_lift_score,
    flatten_for_csv,
    has_bilateral_contact,
    is_elite_eligible,
    min_peg_xy,
    nut_z_delta,
    partial_lift_success,
    peg_xy,
    transport_improved,
    weak_lift_positive,
)
from lift_v2b53_refiner import LIFT_V2B53_SEARCH_SPACE, LiftV2B53Params, lift_v2b53_params_from_dict  # noqa: E402
from lift_v2b53_seed_pool import build_seed_pool, load_rollout_records  # noqa: E402
from lift_v2b53_sim_search import execute_lift_v2b53_rollout  # noqa: E402

DEFAULT_B51_JSONL = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_rollout_samples.jsonl"
DEFAULT_B52_JSONL = _EXPERIMENT_DIR / "outputs" / "lift_v2b52" / "lift_v2b52_rollout_samples.jsonl"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b53"
DEFAULT_DIAGNOSIS = _EXPERIMENT_DIR / "outputs" / "demo_3_failure_diagnosis.json"

B52_WEAK_COUNT = 4
B52_MAX_LIFT = 0.002072244076081864


def _perturb(base: LiftV2B53Params, rng: random.Random, scale: float) -> LiftV2B53Params:
    raw = base.to_dict()
    for key, choices in LIFT_V2B53_SEARCH_SPACE.items():
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
    return lift_v2b53_params_from_dict(raw)


def _rollout_worker(q: Queue, hdf5: str, demo_key: str, params_dict: dict[str, Any], kind: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        params = lift_v2b53_params_from_dict(params_dict)
        result = execute_lift_v2b53_rollout(hdf5, demo_key, "failed", params, rollout_kind=kind)
        q.put(("ok", result))
    except Exception as exc:
        q.put(("err", str(exc)))


def run_rollout_with_timeout(
    *,
    hdf5: Path,
    demo_key: str,
    params: LiftV2B53Params,
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
            "final_nut_peg_xy": DEMO3_BASELINE_PEG_XY,
            "min_nut_peg_xy": DEMO3_BASELINE_PEG_XY,
            "lift_v2b53_params": params.to_dict(),
            "outcome_label": "rollout_timeout",
        }
    if q.empty():
        return {
            "demo_name": demo_key,
            "rollout_kind": rollout_kind,
            "rollout_timeout": True,
            "partial_lift_success": False,
            "nut_z_lift_delta": 0.0,
            "lift_v2b53_params": params.to_dict(),
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
            "lift_v2b53_params": params.to_dict(),
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
        if line.strip() and json.loads(line).get("rollout_kind", "").startswith("lift_v2b53")
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
) -> dict[str, Any]:
    valid = [r for r in records if not r.get("rollout_timeout") and not r.get("rollout_error")]
    partial_count = sum(1 for r in valid if partial_lift_success(r))
    weak_count = sum(1 for r in valid if weak_lift_positive(r))
    max_lift = max((nut_z_delta(r) for r in valid), default=0.0)
    min_final_peg = min((peg_xy(r) for r in valid), default=DEMO3_BASELINE_PEG_XY)
    min_min_peg = min((min_peg_xy(r) for r in valid), default=DEMO3_BASELINE_PEG_XY)
    transport_count = sum(1 for r in valid if transport_improved(r))

    gated = [r for r in valid if is_elite_eligible(r)]

    def best_by(key_fn, pool: list[dict[str, Any]], reverse: bool = True) -> dict[str, Any]:
        if not pool:
            return {}
        rec = sorted(pool, key=key_fn, reverse=reverse)[0]
        score = compute_residual_breakdown(rec)
        return {
            "nut_z_lift_delta": nut_z_delta(rec),
            "final_nut_peg_xy": peg_xy(rec),
            "min_nut_peg_xy": min_peg_xy(rec),
            "transport_lift_score": score["transport_lift_score"],
            "bilateral_contact_steps": rec.get("bilateral_contact_steps"),
            "right_finger_contact_count": rec.get("right_finger_contact_count"),
            "nut_eef_coupling_ratio": rec.get("nut_eef_coupling_ratio"),
            "residual_breakdown": score["residual_breakdown"],
            "E_transport": score["E_transport"],
            "E_xy": score["E_xy"],
            "E_lift": score["E_lift"],
            "E_contact": score["E_contact"],
            "E_bilateral": score["E_bilateral"],
            "E_slip": score["E_slip"],
            "E_coupling": score["E_coupling"],
            "lift_v2b53_params": rec.get("lift_v2b53_params"),
            "cem_round": rec.get("cem_round"),
            "cem_index": rec.get("cem_index"),
        }

    best_transport = best_by(lambda r: -min_peg_xy(r), valid)
    best_gated = best_by(
        lambda r: compute_residual_breakdown(r)["transport_lift_score"],
        gated or valid,
    )
    best_partial = best_by(nut_z_delta, [r for r in valid if partial_lift_success(r)] or valid)
    bilateral_pool = [r for r in valid if has_bilateral_contact(r)] or valid
    best_transport_lift = best_by(
        lambda r: (
            compute_residual_breakdown(r)["transport_lift_score"],
            -min_peg_xy(r),
            nut_z_delta(r),
        ),
        bilateral_pool,
    )

    ranked = sorted(valid, key=lambda r: compute_residual_breakdown(r)["transport_lift_score"], reverse=True)[:80]
    csv_path = output_dir / "top_lift_transport_candidates_v2b53.csv"
    if ranked:
        extra = ["cem_round", "cem_index", "outcome_label", "E_dynamics", "lift_v2b53_params_json"]
        fieldnames = CSV_REQUIRED_COLUMNS + [c for c in extra if c not in CSV_REQUIRED_COLUMNS]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in ranked:
                writer.writerow(flatten_for_csv(row))

    passed = bool(
        partial_count > 0
        or (float(max_lift) >= PARTIAL_THRESH and min_final_peg < DEMO3_BASELINE_PEG_XY * 0.95)
        or (
            weak_count > B52_WEAK_COUNT
            and min_final_peg < DEMO3_BASELINE_PEG_XY * 0.97
            and best_transport.get("min_nut_peg_xy", DEMO3_BASELINE_PEG_XY) < DEMO3_BASELINE_PEG_XY * 0.97
        )
    )
    branch = "v1g_dataset_draft_candidate" if partial_count > 0 else "await_b53_or_expand"

    report = {
        "task": "lift_v2b53_contact_gated_transport_lift",
        "demo_key": "demo_3",
        "primary_failure_mode": "transport_failed",
        "secondary_failure_mode": "lift_underdeveloped",
        "total_evals": len(valid),
        "partial_lift_success_count": partial_count,
        "weak_lift_positive_count": weak_count,
        "max_nut_lift_delta": float(max_lift),
        "min_final_nut_peg_xy": float(min_final_peg),
        "min_min_nut_peg_xy": float(min_min_peg),
        "transport_improved_count": transport_count,
        "baseline_final_nut_peg_xy_m": DEMO3_BASELINE_PEG_XY,
        "partial_lift_threshold_m": PARTIAL_THRESH,
        "success_lift_p50_reference_m": SUCCESS_LIFT_P50,
        "b52_baseline": {
            "weak_lift_positive_count": B52_WEAK_COUNT,
            "max_nut_lift_delta": B52_MAX_LIFT,
        },
        "acceptance": {
            "passed": passed,
            "partial_lift_success": bool(partial_count > 0),
            "lift_and_transport_joint": bool(
                float(max_lift) >= PARTIAL_THRESH and min_final_peg < DEMO3_BASELINE_PEG_XY * 0.95
            ),
            "weak_plus_transport": bool(
                weak_count > B52_WEAK_COUNT
                and best_transport.get("min_nut_peg_xy", DEMO3_BASELINE_PEG_XY) < DEMO3_BASELINE_PEG_XY * 0.97
            ),
        },
        "branch_recommendation": branch,
        "best_candidates": {
            "contact_gated_score": best_gated,
            "transport": best_transport,
            "transport_and_lift": best_transport_lift,
            "partial_lift": best_partial,
        },
        "residual_priority_spec": {
            "P1": ["E_transport", "E_xy"],
            "P2": ["E_lift"],
            "P3": ["E_contact", "E_bilateral"],
            "P4": ["E_slip", "E_coupling", "E_dynamics"],
            "elite_requires_transport_improved": True,
            "hard_reject": ["unilateral_lever", "contact_rich_no_transport"],
        },
        "distributions": {
            "nut_z_lift_delta": _distribution([nut_z_delta(r) for r in valid]),
            "final_nut_peg_xy": _distribution([peg_xy(r) for r in valid]),
            "min_nut_peg_xy": _distribution([min_peg_xy(r) for r in valid]),
            "right_finger_contact_count": _distribution(
                [float(r.get("right_finger_contact_count", 0)) for r in valid]
            ),
            "bilateral_contact_steps": _distribution(
                [float(r.get("bilateral_contact_steps", 0)) for r in valid]
            ),
            "nut_eef_coupling_ratio": _distribution(
                [float(r.get("nut_eef_coupling_ratio", 0.0)) for r in valid]
            ),
        },
        "elite_gated_count": len(gated),
        "seed_pool_meta": seed_meta,
        "cem_iteration_history": cem_history,
    }

    transport_report = {
        "task": "contact_gated_transport_lift_report",
        "demo_key": "demo_3",
        "diagnosis": {
            "primary": "transport_failed",
            "secondary": "lift_underdeveloped",
            "note": "Success demos rely on lift-and-carry; partial threshold 0.005m retained.",
        },
        "objective_summary": {
            "residual_priority": {
                "P1_transport_xy": "E_transport / E_xy — nut approaches peg; elite requires transport_improved",
                "P2_lift": "E_lift — partial gate 0.005m, weak 0.002m, ref p50 0.039m; penalize unilateral lift",
                "P3_contact": "E_contact / E_bilateral — hard reject right=0 or bilateral=0",
                "P4_dynamics": "E_slip / E_coupling / E_dynamics — follow gripper not single-side pry",
            },
            "hard_reject": [
                "unilateral_lever (right_finger=0 or bilateral=0)",
                "contact_rich_no_transport",
                "no_transport_improvement (cannot be elite)",
            ],
        },
        **{k: report[k] for k in report if k not in ("seed_pool_meta", "cem_iteration_history")},
    }

    (output_dir / "lift_v2b53_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "contact_gated_transport_lift_report.json").write_text(
        json.dumps(transport_report, indent=2), encoding="utf-8"
    )
    (output_dir / "cem_iteration_history.json").write_text(json.dumps(cem_history, indent=2), encoding="utf-8")
    return report


def run_cem(
    *,
    seeds: list[LiftV2B53Params],
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
        candidates: list[LiftV2B53Params] = []
        for _ in range(candidates_per_round):
            base = rng.choice(elites)
            scale = max(0.04, 0.28 * (0.72 ** round_idx))
            candidates.append(base if rng.random() < 0.08 else _perturb(base, rng, scale))

        scored: list[tuple[float, dict[str, Any]]] = []
        round_start = time.time()
        for local_i, params in enumerate(candidates):
            result = run_rollout_with_timeout(
                hdf5=hdf5,
                demo_key=demo_key,
                params=params,
                rollout_kind="lift_v2b53_cem",
                timeout_sec=timeout_sec,
            )
            result["cem_round"] = round_idx
            result["cem_index"] = global_index
            global_index += 1
            if not result.get("rollout_timeout") and not result.get("rollout_error"):
                result.update(compute_transport_lift_score(result))
            else:
                result["transport_lift_score"] = -999.0
            _append_jsonl(jsonl_path, result)
            all_records.append(result)
            scored.append((float(result.get("transport_lift_score", -999)), result))
            if (local_i + 1) % 20 == 0:
                print(f"  round {round_idx + 1}/{cem_rounds}: {local_i + 1}/{candidates_per_round}", flush=True)

        scored.sort(key=lambda x: x[0], reverse=True)
        elite_n = max(3, int(len(scored) * elite_frac))
        gated_elites = [
            lift_v2b53_params_from_dict(s[1]["lift_v2b53_params"])
            for s in scored
            if is_elite_eligible(s[1]) and s[1].get("lift_v2b53_params")
        ]
        elites = gated_elites[:elite_n]
        if not elites:
            elites = list(seeds)[:elite_n]

        best = scored[0][1] if scored else {}
        round_summary = {
            "round": round_idx,
            "candidates": candidates_per_round,
            "elite_count": len(elites),
            "gated_elite_count": len(gated_elites[:elite_n]),
            "best_transport_lift_score": scored[0][0] if scored else None,
            "best_nut_z_lift_delta": nut_z_delta(best) if best else 0.0,
            "best_final_nut_peg_xy": peg_xy(best) if best else DEMO3_BASELINE_PEG_XY,
            "best_min_nut_peg_xy": min_peg_xy(best) if best else DEMO3_BASELINE_PEG_XY,
            "best_right_finger": best.get("right_finger_contact_count"),
            "best_bilateral_steps": best.get("bilateral_contact_steps"),
            "partial_in_round": sum(1 for _, r in scored if partial_lift_success(r)),
            "weak_in_round": sum(1 for _, r in scored if weak_lift_positive(r)),
            "transport_improved_in_round": sum(1 for _, r in scored if transport_improved(r)),
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


def _write_v1g_draft_candidate(*, output_dir: Path, records: list[dict[str, Any]]) -> None:
    """Write V1-G dataset draft candidate list only — never auto-train."""
    partial_recs = [r for r in records if partial_lift_success(r)]
    draft = {
        "task": "v1g_dataset_draft_candidate",
        "status": "awaiting_user_confirmation",
        "auto_train_forbidden": True,
        "partial_lift_success_count": len(partial_recs),
        "candidates": [
            {
                "cem_index": r.get("cem_index"),
                "cem_round": r.get("cem_round"),
                "nut_z_lift_delta": nut_z_delta(r),
                "final_nut_peg_xy": peg_xy(r),
                "min_nut_peg_xy": min_peg_xy(r),
                "right_finger_contact_count": r.get("right_finger_contact_count"),
                "bilateral_contact_steps": r.get("bilateral_contact_steps"),
                "residual_breakdown": compute_residual_breakdown(r).get("residual_breakdown"),
                "lift_v2b53_params": r.get("lift_v2b53_params"),
            }
            for r in sorted(partial_recs, key=nut_z_delta, reverse=True)[:20]
        ],
    }
    (output_dir / "v1g_dataset_draft_candidate.json").write_text(json.dumps(draft, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B5.3 contact-gated transport-and-lift CEM")
    parser.add_argument("--b51-jsonl", type=Path, default=DEFAULT_B51_JSONL)
    parser.add_argument("--b52-jsonl", type=Path, default=DEFAULT_B52_JSONL)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--demo-key", default="demo_3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cem-rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=120)
    parser.add_argument("--elite-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--rollout-timeout", type=float, default=90.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "lift_v2b53_rollout_samples.jsonl"

    records = load_rollout_records(args.b51_jsonl, args.b52_jsonl)
    seeds, seed_meta = build_seed_pool(records, top_k=20, rng=random.Random(args.seed))
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
    _, history = run_cem(
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
    report = generate_reports(
        records=all_records,
        output_dir=args.output_dir,
        seed_meta=seed_meta,
        cem_history=history,
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "lift_v2b53_report.json"),
                "transport_report": str(args.output_dir / "contact_gated_transport_lift_report.json"),
                "branch": report["branch_recommendation"],
                "passed": report["acceptance"]["passed"],
            },
            indent=2,
        )
    )

    if report["branch_recommendation"] == "v1g_dataset_draft_candidate":
        _write_v1g_draft_candidate(output_dir=args.output_dir, records=all_records)
        print(
            "partial lift achieved — wrote v1g_dataset_draft_candidate.json only; "
            "V1-G PINN training NOT started (await user confirmation)."
        )

    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
