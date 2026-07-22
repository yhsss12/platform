#!/usr/bin/env python3
"""V1-E：PINN repair-parameter 预筛选 + 少量 sim rollout 对比。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
_V1_DIR = _EXPERIMENT_DIR / "v1_residual_model"
_V1E_DIR = _V1_DIR / "repair_parameter_model"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grasp_sim_search import GRASP_SEARCH_SPACE, execute_grasp_rollout  # noqa: E402
from grasp_waypoint_builder import GraspSearchParams  # noqa: E402
from osc_action_converter import SEARCH_SPACE, SimLoopParams  # noqa: E402
from pinn_repair_inference import (  # noqa: E402
    build_features_from_repair_spec,
    clear_repair_model_cache,
    context_from_original,
    load_repair_model,
    score_repair_candidate,
)
from pinn_repair_parameter_model import explicit_repair_energy  # noqa: E402
from robosuite_env_loader import check_environment  # noqa: E402
from sim_in_loop_refiner import build_refined_waypoints_from_hdf5, execute_waypoint_rollout, load_best_theta  # noqa: E402

DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_CEM = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_MODEL = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "model.pt"
DEFAULT_OUTPUT = DEFAULT_MODEL.parent

DEMO_CONFIGS = {
    "demo_4": {
        "failure_type": "insertion_failed",
        "active": "insertion",
        "search_kind": "insertion",
        "report": _EXPERIMENT_DIR / "outputs" / "sim_in_loop_refinement" / "sim_in_loop_refinement_report.json",
        "original_key": "original_waypoint_rollout",
    },
    "demo_2": {
        "failure_type": "grasp_failed",
        "active": "grasp",
        "search_kind": "grasp",
        "report": _EXPERIMENT_DIR / "outputs" / "grasp_refinement" / "grasp_refinement_report.json",
        "original_key": "original_waypoint_rollout",
        "per_demo": True,
    },
    "demo_3": {
        "failure_type": "lift_failed",
        "active": "lift",
        "search_kind": "grasp",
        "report": _EXPERIMENT_DIR / "outputs" / "grasp_refinement" / "grasp_refinement_report.json",
        "original_key": "original_waypoint_rollout",
        "per_demo": True,
    },
}


def _load_original(demo_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(Path(cfg["report"]).read_text(encoding="utf-8"))
    if cfg.get("per_demo"):
        return report["per_demo"][demo_key][cfg["original_key"]]
    return report[cfg["original_key"]]


def _sample_insertion(rng: random.Random) -> dict[str, float]:
    params = SimLoopParams(**{k: rng.choice(SEARCH_SPACE[k]) for k in SEARCH_SPACE})
    return params.to_dict()


def _sample_grasp_lift(rng: random.Random) -> dict[str, float]:
    raw = {k: rng.choice(GRASP_SEARCH_SPACE[k]) for k in GRASP_SEARCH_SPACE}
    return {
        "grasp_xy_offset_x": float(raw["grasp_xy_offset_x"]),
        "grasp_xy_offset_y": float(raw["grasp_xy_offset_y"]),
        "pre_grasp_height": float(raw["pre_grasp_height"]),
        "approach_height": float(raw["approach_height"]),
        "gripper_hold_steps": float(raw["gripper_hold_steps"]),
        "lift_steps": float(raw["lift_steps"]),
        "lift_speed_scale": float(raw["speed_scale"]),
        "micro_lift_height": float(raw["lift_height"]),
        "reclose_after_contact": float(raw.get("gripper_close_shift", 0.0)),
    }


def _sample_candidates(
    *,
    cfg: dict[str, Any],
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for i in range(n_samples):
        if cfg["search_kind"] == "insertion":
            insertion = _sample_insertion(rng)
            out.append({"index": i, "insertion": insertion, "transport": None, "grasp_lift": None})
        else:
            grasp_lift = _sample_grasp_lift(rng)
            out.append({"index": i, "insertion": None, "transport": None, "grasp_lift": grasp_lift})
    return out


def _score_candidates(
    *,
    context: dict[str, Any],
    candidates: list[dict[str, Any]],
    active: str,
    model_path: Path,
) -> None:
    import torch

    load_repair_model(model_path)
    for cand in candidates:
        features = build_features_from_repair_spec(
            context=context,
            insertion=cand.get("insertion"),
            transport=cand.get("transport"),
            grasp_lift=cand.get("grasp_lift"),
            active=active,
        )
        scores = score_repair_candidate(features, model_path=model_path)
        cand["features"] = features
        cand["pinn_E_total"] = scores["pinn_E_total"]
        cand["explicit_E_total"] = scores["explicit_E_total"]
        cand["pinn_success_prob"] = scores["pinn_success_prob"]


def _select_indices(candidates: list[dict[str, Any]], *, method: str, top_k: int, rng: random.Random) -> list[int]:
    if method == "pinn_top_k":
        order = sorted(range(len(candidates)), key=lambda i: candidates[i]["pinn_E_total"])
    elif method == "explicit_top_k":
        order = sorted(range(len(candidates)), key=lambda i: candidates[i]["explicit_E_total"])
    elif method == "random_k":
        order = list(range(len(candidates)))
        rng.shuffle(order)
    else:
        raise ValueError(method)
    return order[:top_k]


def _rollout_insertion(
    *,
    failed_hdf5: Path,
    demo_key: str,
    cem_report: Path,
    insertion: dict[str, float],
) -> dict[str, Any]:
    theta = load_best_theta(cem_report, demo_key)
    _, _, refined_eef, shifted_gripper = build_refined_waypoints_from_hdf5(
        str(failed_hdf5), demo_key, "failed", theta, rollout_safe=True
    )
    return execute_waypoint_rollout(
        str(failed_hdf5),
        demo_key,
        "failed",
        refined_eef,
        shifted_gripper,
        SimLoopParams(**insertion),
        rollout_kind="pinn_repair_guided_search",
        theta=theta,
        record_video=False,
    )


def _rollout_grasp(
    *,
    failed_hdf5: Path,
    demo_key: str,
    grasp_lift: dict[str, float],
) -> dict[str, Any]:
    params = GraspSearchParams(
        grasp_xy_offset_x=grasp_lift["grasp_xy_offset_x"],
        grasp_xy_offset_y=grasp_lift["grasp_xy_offset_y"],
        pre_grasp_height=grasp_lift["pre_grasp_height"],
        approach_height=grasp_lift["approach_height"],
        gripper_close_shift=int(grasp_lift.get("reclose_after_contact", 0)),
        gripper_hold_steps=int(grasp_lift["gripper_hold_steps"]),
        lift_height=grasp_lift["micro_lift_height"],
        lift_steps=int(grasp_lift["lift_steps"]),
        speed_scale=grasp_lift["lift_speed_scale"],
    )
    return execute_grasp_rollout(
        str(failed_hdf5),
        demo_key,
        "failed",
        params,
        rollout_kind="pinn_repair_guided_search",
        record_video=False,
    )


def _run_method_rollouts(
    *,
    method: str,
    top_k: int,
    candidates: list[dict[str, Any]],
    cfg: dict[str, Any],
    demo_key: str,
    failed_hdf5: Path,
    cem_report: Path,
    rng: random.Random,
) -> dict[str, Any]:
    indices = _select_indices(candidates, method=method, top_k=top_k, rng=rng)
    picked = [candidates[i] for i in indices]
    results: list[dict[str, Any]] = []
    for cand in picked:
        if cfg["search_kind"] == "insertion":
            row = _rollout_insertion(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                cem_report=cem_report,
                insertion=cand["insertion"],
            )
        else:
            row = _rollout_grasp(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                grasp_lift=cand["grasp_lift"],
            )
        results.append(row)

    best = min(results, key=lambda r: (not r.get("success_flag", False), float(r.get("E_total_norm", 1e9))))
    energies = [float(r.get("E_total_norm", 0.0)) for r in results]
    return {
        "method": method,
        "num_rollouts": len(results),
        "success_rate": float(any(r.get("success_flag") for r in results)),
        "best_success_flag": bool(best.get("success_flag")),
        "best_E_total_norm": float(best.get("E_total_norm", 0.0)),
        "best_final_xy": float(best.get("final_nut_peg_xy", 0.0)),
        "best_final_z_diff": float(best.get("final_z_diff", 0.0)) if best.get("final_z_diff") is not None else None,
        "best_grasp_success_proxy": bool(best.get("grasp_success_proxy", False)),
        "best_lift_success_proxy": bool(best.get("lift_success_proxy", False)),
        "avg_rollout_E_total": float(np.mean(energies)) if energies else None,
        "min_rollout_E_total": float(np.min(energies)) if energies else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-E PINN repair-parameter guided search")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-keys", default="demo_4,demo_2,demo_3")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--baseline-random-rollouts", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    demo_keys = [k.strip() for k in args.demo_keys.split(",") if k.strip()]

    env_check = check_environment()
    report: dict[str, Any] = {
        "task": "V1-E_PINN_repair_guided_search",
        "failed_hdf5": str(args.failed_hdf5),
        "model": str(args.model),
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "baseline_random_rollouts": args.baseline_random_rollouts,
        "seed": args.seed,
        "environment_check": env_check,
        "notes": [
            "Sample N repair parameters, PINN pre-filters to top-K, then sim rollout.",
            "Compare pinn_top_k vs explicit_top_k vs random_k vs random_80 baseline.",
            "V1-E core model: repair-parameter residual field, not trajectory predictor.",
        ],
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        out = args.output_dir / "repair_guided_search_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    if not args.model.exists():
        report["blocked"] = True
        report["block_reason"] = [f"model missing: {args.model}"]
        out = args.output_dir / "repair_guided_search_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    clear_repair_model_cache()
    per_demo: dict[str, Any] = {}
    rng = random.Random(args.seed)

    for demo_key in demo_keys:
        cfg = DEMO_CONFIGS[demo_key]
        original = _load_original(demo_key, cfg)
        context = context_from_original(original, demo_key=demo_key, failure_type=cfg["failure_type"])
        candidates = _sample_candidates(cfg=cfg, n_samples=args.num_samples, seed=args.seed + hash(demo_key) % 10000)
        _score_candidates(context=context, candidates=candidates, active=cfg["active"], model_path=args.model)

        per_demo[demo_key] = {
            "failure_type": cfg["failure_type"],
            "num_sampled": len(candidates),
            "methods": {
                "pinn_top_k": _run_method_rollouts(
                    method="pinn_top_k",
                    top_k=args.top_k,
                    candidates=candidates,
                    cfg=cfg,
                    demo_key=demo_key,
                    failed_hdf5=args.failed_hdf5,
                    cem_report=args.cem_report,
                    rng=rng,
                ),
                "explicit_top_k": _run_method_rollouts(
                    method="explicit_top_k",
                    top_k=args.top_k,
                    candidates=candidates,
                    cfg=cfg,
                    demo_key=demo_key,
                    failed_hdf5=args.failed_hdf5,
                    cem_report=args.cem_report,
                    rng=rng,
                ),
                "random_k": _run_method_rollouts(
                    method="random_k",
                    top_k=args.top_k,
                    candidates=candidates,
                    cfg=cfg,
                    demo_key=demo_key,
                    failed_hdf5=args.failed_hdf5,
                    cem_report=args.cem_report,
                    rng=rng,
                ),
                "random_baseline": _run_method_rollouts(
                    method="random_k",
                    top_k=args.baseline_random_rollouts,
                    candidates=candidates,
                    cfg=cfg,
                    demo_key=demo_key,
                    failed_hdf5=args.failed_hdf5,
                    cem_report=args.cem_report,
                    rng=rng,
                ),
            },
        }

    report["per_demo"] = per_demo
    out = args.output_dir / "repair_guided_search_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "per_demo_summary": {
        k: {m: v["best_success_flag"] for m, v in v["methods"].items()} for k, v in per_demo.items()
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
