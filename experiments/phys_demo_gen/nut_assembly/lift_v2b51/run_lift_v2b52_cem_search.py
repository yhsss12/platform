#!/usr/bin/env python3
"""V2-B5.2：CEM 局部搜索，目标 nut_z_lift_delta >= 0.005m。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_V2B51_DIR = Path(__file__).resolve().parent
for path in (_EXPERIMENT_DIR, _V2B51_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_lift_v2b51_final_report import _load_records, _nut_z  # noqa: E402
from lift_v2b51_refiner import LIFT_V2B51_SEARCH_SPACE, LiftV2B51Params, lift_v2b51_params_from_dict  # noqa: E402
from lift_v2b51_sim_search import execute_lift_v2b51_rollout  # noqa: E402

DEFAULT_JSONL = _EXPERIMENT_DIR / "outputs" / "lift_v2b51" / "lift_v2b51_rollout_samples.jsonl"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b52"
GOAL = 0.005
WEAK = 0.002


def _collect_seeds(records: list[dict[str, Any]], *, top_k: int = 20) -> list[LiftV2B51Params]:
    seeds: list[LiftV2B51Params] = []
    seen: set[str] = set()

    def _add_from_ranked(ranked: list[dict[str, Any]]) -> None:
        for rec in ranked[:top_k]:
            params = rec.get("lift_v2b51_params")
            if not params:
                continue
            key = json.dumps(params, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(lift_v2b51_params_from_dict(params))

    _add_from_ranked(sorted(records, key=_nut_z, reverse=True))
    _add_from_ranked(sorted(records, key=lambda r: int(r.get("bilateral_contact_steps", 0)), reverse=True))
    _add_from_ranked(sorted(records, key=lambda r: float(r.get("nut_eef_coupling_ratio", -999)), reverse=True))
    _add_from_ranked(sorted(records, key=lambda r: float(r.get("nut_xy_slip", 999))))
    return seeds


def _perturb(params: LiftV2B51Params, rng: random.Random, scale: float) -> LiftV2B51Params:
    raw = params.to_dict()
    for key, choices in LIFT_V2B51_SEARCH_SPACE.items():
        if key == "template_mask":
            continue
        if rng.random() < 0.7:
            center = raw[key]
            if isinstance(center, int):
                delta = int(round(rng.gauss(0, max(1, scale * 3))))
                raw[key] = int(np.clip(center + delta, min(choices), max(choices)))
            elif isinstance(center, float):
                span = float(max(choices) - min(choices))
                raw[key] = float(np.clip(center + rng.gauss(0, span * scale), min(choices), max(choices)))
            else:
                raw[key] = rng.choice(choices)
    return LiftV2B51Params(**raw)


def run_cem(
    *,
    seeds: list[LiftV2B51Params],
    hdf5: Path,
    demo_key: str,
    generations: int,
    pop_per_seed: int,
    elite_frac: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for gen in range(generations):
        candidates: list[LiftV2B51Params] = []
        scale = max(0.05, 0.35 * (0.75**gen))
        for base in seeds:
            candidates.append(base)
            for _ in range(pop_per_seed):
                candidates.append(_perturb(base, rng, scale))

        scored: list[tuple[tuple, dict[str, Any]]] = []
        for i, params in enumerate(candidates):
            result = execute_lift_v2b51_rollout(str(hdf5), demo_key, "failed", params, rollout_kind="lift_v2b52_cem")
            result["cem_generation"] = gen
            result["cem_index"] = i
            all_records.append(result)
            score = (
                int(result.get("partial_lift_success")),
                _nut_z(result),
                int(result.get("bilateral_contact_steps", 0)),
                float(result.get("nut_eef_coupling_ratio", 0)),
                -float(result.get("nut_xy_slip", 999)),
            )
            scored.append((score, result))
            if best is None or score > (
                int(best.get("partial_lift_success")),
                _nut_z(best),
                int(best.get("bilateral_contact_steps", 0)),
                float(best.get("nut_eef_coupling_ratio", 0)),
                -float(best.get("nut_xy_slip", 999)),
            ):
                best = result

        scored.sort(key=lambda x: x[0], reverse=True)
        elite_n = max(3, int(len(scored) * elite_frac))
        seeds = [
            lift_v2b51_params_from_dict(s[1]["lift_v2b51_params"])
            for s in scored[:elite_n]
            if s[1].get("lift_v2b51_params")
        ]
        print(
            f"  cem gen {gen + 1}/{generations} best_nut_z={_nut_z(best):.4f} partial={best.get('partial_lift_success')}",
            flush=True,
        )
        if best and (best.get("partial_lift_success") or _nut_z(best) >= GOAL):
            break
    return all_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--demo-key", default="demo_3")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--pop-per-seed", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    records = _load_records(args.jsonl)
    partial = sum(1 for r in records if r.get("partial_lift_success"))
    weak = sum(1 for r in records if _nut_z(r) >= WEAK)
    if partial > 0:
        print("SKIP CEM: partial_lift_success already found")
        return 0
    if weak <= 0:
        print("SKIP CEM: weak_lift_positive_count=0")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _collect_seeds(records)
    print(f"CEM seeds={len(seeds)} weak={weak} goal={GOAL}m", flush=True)
    cem_records = run_cem(
        seeds=seeds,
        hdf5=args.failed_hdf5,
        demo_key=args.demo_key,
        generations=args.generations,
        pop_per_seed=args.pop_per_seed,
        elite_frac=0.15,
        seed=args.seed,
    )

    jsonl_out = args.output_dir / "lift_v2b52_cem_rollout_samples.jsonl"
    with jsonl_out.open("w", encoding="utf-8") as handle:
        for rec in cem_records:
            slim = {k: v for k, v in rec.items() if not str(k).startswith("per_step_")}
            handle.write(json.dumps(slim, default=str) + "\n")

    best = max(cem_records, key=lambda r: (int(r.get("partial_lift_success")), _nut_z(r)))
    report = {
        "task": "lift_v2b52_cem_local_search",
        "goal_nut_z_lift_delta_m": GOAL,
        "seed_count": len(seeds),
        "generations": args.generations,
        "num_rollouts": len(cem_records),
        "partial_lift_success_count": sum(1 for r in cem_records if r.get("partial_lift_success")),
        "weak_lift_positive_count": sum(1 for r in cem_records if _nut_z(r) >= WEAK),
        "max_nut_lift_delta": max((_nut_z(r) for r in cem_records), default=0.0),
        "best_candidate": {
            "nut_z_lift_delta": _nut_z(best),
            "partial_lift_success": best.get("partial_lift_success"),
            "bilateral_contact_steps": best.get("bilateral_contact_steps"),
            "lift_v2b51_params": best.get("lift_v2b51_params"),
        },
        "outputs": {"jsonl": str(jsonl_out), "report": str(args.output_dir / "lift_v2b52_cem_report.json")},
    }
    (args.output_dir / "lift_v2b52_cem_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["partial_lift_success_count"] > 0 or report["max_nut_lift_delta"] >= GOAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
