"""Offline MimicGen Repair：候选采样、打分、选择。"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_OFFLINE_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _OFFLINE_DIR.parent
_V1_DIR = _EXPERIMENT_DIR / "v1_residual_model"
_V1E_DIR = _V1_DIR / "repair_parameter_model"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grasp_sim_search import GRASP_SEARCH_SPACE  # noqa: E402
from grasp_waypoint_builder import GraspSearchParams  # noqa: E402
from osc_action_converter import SEARCH_SPACE, SimLoopParams  # noqa: E402
from pinn_repair_inference import (  # noqa: E402
    build_features_from_repair_spec,
    load_repair_model,
    score_repair_candidate,
)
from repair_dataset import extract_failed_context  # noqa: E402
from sim_in_loop_refiner import run_original_waypoint_rollout  # noqa: E402


def extract_baseline_context(
    *,
    failed_hdf5: str | Path,
    demo_key: str,
    failure_type: str,
    search_kind: str,
) -> dict[str, Any]:
    """从 failed demo 直接 rollout baseline，提取 failed context（不依赖 prior report）。"""
    if search_kind == "insertion":
        baseline = run_original_waypoint_rollout(
            str(failed_hdf5),
            demo_key,
            "failed",
            record_video=False,
        )
    else:
        from grasp_sim_search import execute_grasp_rollout

        baseline = execute_grasp_rollout(
            str(failed_hdf5),
            demo_key,
            "failed",
            GraspSearchParams(),
            rollout_kind="offline_repair_baseline",
            record_video=False,
        )
    return extract_failed_context(baseline, demo_key=demo_key, failure_type=failure_type)


def sample_insertion_theta(rng: random.Random) -> dict[str, float]:
    return SimLoopParams(**{k: rng.choice(SEARCH_SPACE[k]) for k in SEARCH_SPACE}).to_dict()


def sample_grasp_lift_theta(rng: random.Random) -> dict[str, float]:
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


def sample_repair_candidates(
    *,
    search_kind: str,
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for i in range(n_samples):
        if search_kind == "insertion":
            out.append({"index": i, "insertion": sample_insertion_theta(rng), "transport": None, "grasp_lift": None})
        else:
            out.append({"index": i, "insertion": None, "transport": None, "grasp_lift": sample_grasp_lift_theta(rng)})
    return out


def score_repair_candidates(
    *,
    context: dict[str, Any],
    candidates: list[dict[str, Any]],
    active: str,
    model_path: Path,
) -> None:
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
        cand["pinn_grasp_success_prob"] = scores.get("pinn_grasp_success_prob")
        cand["pinn_lift_success_prob"] = scores.get("pinn_lift_success_prob")


def select_candidate_indices(
    candidates: list[dict[str, Any]],
    *,
    method: str,
    top_k: int,
    rng: random.Random,
) -> list[int]:
    if method == "pinn_top_k":
        order = sorted(range(len(candidates)), key=lambda i: candidates[i]["pinn_E_total"])
    elif method == "explicit_top_k":
        order = sorted(range(len(candidates)), key=lambda i: candidates[i]["explicit_E_total"])
    elif method == "random_top_k":
        order = list(range(len(candidates)))
        rng.shuffle(order)
    else:
        raise ValueError(f"unknown method: {method}")
    return order[: min(top_k, len(candidates))]


def summarize_method_results(results: list[dict[str, Any]], *, method: str, rollout_budget: int) -> dict[str, Any]:
    if not results:
        return {
            "method": method,
            "rollout_budget": rollout_budget,
            "num_rollouts": 0,
            "repair_success_rate": 0.0,
            "any_success": False,
            "num_successes_written": 0,
        }
    successes = [r for r in results if r.get("success_flag")]
    best = min(results, key=lambda r: (not r.get("success_flag", False), float(r.get("E_total_norm", 1e9))))
    energies = [float(r.get("E_total_norm", 0.0)) for r in results]
    return {
        "method": method,
        "rollout_budget": rollout_budget,
        "num_rollouts": len(results),
        "repair_success_rate": float(len(successes) / len(results)),
        "any_success": bool(successes),
        "num_successes_written": len(successes),
        "best_success_flag": bool(best.get("success_flag")),
        "best_E_total_norm": float(best.get("E_total_norm", 0.0)),
        "best_final_xy": float(best.get("final_nut_peg_xy", 0.0)),
        "best_final_z_diff": float(best.get("final_z_diff", 0.0)) if best.get("final_z_diff") is not None else None,
        "best_grasp_success_proxy": bool(best.get("grasp_success_proxy", False)),
        "best_lift_success_proxy": bool(best.get("lift_success_proxy", False)),
        "avg_rollout_E_total": float(np.mean(energies)),
        "min_rollout_E_total": float(np.min(energies)),
    }
