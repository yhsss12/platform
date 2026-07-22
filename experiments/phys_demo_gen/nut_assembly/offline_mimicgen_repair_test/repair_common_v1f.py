"""Offline MimicGen Repair V1-F：双 PINN 候选采样、打分、选择。"""
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
_V1F_DIR = _V1_DIR / "repair_parameter_model_v1f"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grasp_sim_search import GRASP_SEARCH_SPACE  # noqa: E402
from grasp_waypoint_builder import GraspSearchParams  # noqa: E402
from lift_waypoint_refiner import LIFT_REPAIR_SEARCH_SPACE  # noqa: E402
from osc_action_converter import SEARCH_SPACE, SimLoopParams  # noqa: E402
from pinn_repair_inference import load_repair_model, score_repair_candidate  # noqa: E402
from pinn_v1f_inference import load_v1f_repair_model, score_v1f_repair_candidate  # noqa: E402
from repair_common import extract_baseline_context, summarize_method_results  # noqa: E402
from repair_dataset import extract_failed_context, infer_coarse_failure_mode  # noqa: E402
from sim_in_loop_refiner import load_best_theta, run_refined_waypoint_rollout  # noqa: E402
from v1f_repair_dataset import LIFT_EXTRA_PARAM_KEYS  # noqa: E402

try:
    from config import DEFAULT_SUCCESS_REFERENCE_JSONL, ENABLE_PHYSICS_RESIDUAL_REPAIR  # noqa: E402
except ImportError:
    DEFAULT_SUCCESS_REFERENCE_JSONL = _EXPERIMENT_DIR / "outputs" / "v1f_100base" / "success_reference_samples.jsonl"
    ENABLE_PHYSICS_RESIDUAL_REPAIR = False

try:
    from physics_residual_repair import (  # noqa: E402
        attach_physics_residual_to_rollout,
        build_physics_repair_context,
        is_physics_residual_repair_enabled,
        rank_candidates_physics_combined,
        select_candidates_with_physics_residuals,
    )
    from physics_residuals import compute_physics_residuals  # noqa: E402
except ImportError:
    attach_physics_residual_to_rollout = None  # type: ignore[misc, assignment]
    build_physics_repair_context = None  # type: ignore[misc, assignment]
    is_physics_residual_repair_enabled = None  # type: ignore[misc, assignment]
    rank_candidates_physics_combined = None  # type: ignore[misc, assignment]
    select_candidates_with_physics_residuals = None  # type: ignore[misc, assignment]
    compute_physics_residuals = None  # type: ignore[misc, assignment]

CONTEXT_SOURCES = ("original_failed_context", "cem_refined_context")
PHYSICS_SELECTION_METHODS = ("physics_residual_top_k", "physics_residual_gated_top_k")


def extract_cem_refined_context_v1f(
    *,
    failed_hdf5: str | Path,
    demo_key: str,
    failure_type: str,
    search_kind: str,
    cem_report: str | Path,
) -> dict[str, Any]:
    """与 V1-F rollout 采样一致的 CEM-refined baseline context。"""
    if search_kind == "insertion":
        theta = load_best_theta(str(cem_report), demo_key)
        baseline = run_refined_waypoint_rollout(
            str(failed_hdf5),
            demo_key,
            "failed",
            theta,
            sim_params=SimLoopParams(),
        )
        return extract_failed_context(baseline, demo_key=demo_key, failure_type=failure_type)
    if search_kind == "transport":
        from transport_sim_search import execute_transport_rollout
        from transport_waypoint_builder import TransportSearchParams

        theta = load_best_theta(str(cem_report), demo_key)
        baseline = execute_transport_rollout(
            str(failed_hdf5),
            demo_key,
            "failed",
            theta,
            TransportSearchParams(),
            rollout_kind="offline_repair_cem_baseline",
        )
        return extract_failed_context(baseline, demo_key=demo_key, failure_type=failure_type)
    return extract_baseline_context_v1f(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=failure_type,
        search_kind=search_kind,
    )


def extract_repair_context_v1f(
    *,
    context_source: str,
    failed_hdf5: str | Path,
    demo_key: str,
    failure_type: str,
    search_kind: str,
    cem_report: str | Path | None = None,
) -> dict[str, Any]:
    if context_source not in CONTEXT_SOURCES:
        raise ValueError(f"unknown context_source: {context_source}")
    if context_source == "cem_refined_context":
        if cem_report is None:
            raise ValueError("cem_report required for cem_refined_context")
        return extract_cem_refined_context_v1f(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            failure_type=failure_type,
            search_kind=search_kind,
            cem_report=cem_report,
        )
    return extract_baseline_context_v1f(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=failure_type,
        search_kind=search_kind,
    )


def extract_baseline_context_v1f(
    *,
    failed_hdf5: str | Path,
    demo_key: str,
    failure_type: str,
    search_kind: str,
) -> dict[str, Any]:
    if search_kind == "lift":
        from lift_sim_search import execute_lift_rollout
        from lift_waypoint_refiner import LiftRepairParams

        baseline = execute_lift_rollout(
            str(failed_hdf5), demo_key, "failed", LiftRepairParams(), rollout_kind="offline_repair_baseline"
        )
        return extract_failed_context(baseline, demo_key=demo_key, failure_type=failure_type)
    return extract_baseline_context(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=failure_type,
        search_kind=search_kind,
    )


def sample_lift_theta(rng: random.Random) -> dict[str, Any]:
    raw = {k: rng.choice(LIFT_REPAIR_SEARCH_SPACE[k]) for k in LIFT_REPAIR_SEARCH_SPACE}
    grasp_lift = {
        "grasp_xy_offset_x": float(raw["grasp_xy_offset_x"]),
        "grasp_xy_offset_y": float(raw["grasp_xy_offset_y"]),
        "pre_grasp_height": float(raw["pre_grasp_height"]),
        "approach_height": float(raw["approach_height"]),
        "gripper_hold_steps": float(raw["gripper_hold_steps"]),
        "lift_steps": float(raw["micro_lift_steps"]),
        "lift_speed_scale": float(raw["lift_speed_scale"]),
        "micro_lift_height": float(raw["micro_lift_height"]),
        "reclose_after_contact": float(raw.get("gripper_close_shift", 0.0)),
    }
    lift_extra = {k: float(raw[k]) for k in LIFT_EXTRA_PARAM_KEYS if k in raw}
    return {"grasp_lift": grasp_lift, "lift_extra": lift_extra}


def sample_repair_candidates_v1f(
    *,
    search_kind: str,
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    from repair_common import sample_grasp_lift_theta, sample_insertion_theta
    from transport_sim_search import TRANSPORT_SEARCH_SPACE

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for i in range(n_samples):
        if search_kind == "insertion":
            out.append({"index": i, "insertion": sample_insertion_theta(rng), "transport": None, "grasp_lift": None, "lift_extra": None})
        elif search_kind == "transport":
            transport = {k: rng.choice(TRANSPORT_SEARCH_SPACE[k]) for k in TRANSPORT_SEARCH_SPACE}
            out.append({"index": i, "insertion": None, "transport": transport, "grasp_lift": None, "lift_extra": None})
        elif search_kind == "lift":
            lift = sample_lift_theta(rng)
            out.append({"index": i, "insertion": None, "transport": None, **lift})
        else:
            out.append({"index": i, "insertion": None, "transport": None, "grasp_lift": sample_grasp_lift_theta(rng), "lift_extra": None})
    return out


def score_repair_candidates_v1f(
    *,
    context: dict[str, Any],
    candidates: list[dict[str, Any]],
    active: str,
    v1e_model_path: Path,
    v1f_model_path: Path,
) -> None:
    from pinn_repair_inference import build_features_from_repair_spec
    from pinn_v1f_inference import build_v1f_features_from_repair_spec

    load_repair_model(v1e_model_path)
    load_v1f_repair_model(v1f_model_path)
    for cand in candidates:
        v1e_features = build_features_from_repair_spec(
            context=context,
            insertion=cand.get("insertion"),
            transport=cand.get("transport"),
            grasp_lift=cand.get("grasp_lift"),
            active=active if active != "lift" else "grasp",
        )
        v1e_scores = score_repair_candidate(v1e_features, model_path=v1e_model_path)
        cand["v1e_features"] = v1e_features
        cand["v1e_E_total"] = v1e_scores["pinn_E_total"]
        cand["v1e_success_prob"] = v1e_scores["pinn_success_prob"]

        v1f_features = build_v1f_features_from_repair_spec(
            context=context,
            insertion=cand.get("insertion"),
            transport=cand.get("transport"),
            grasp_lift=cand.get("grasp_lift"),
            lift_extra=cand.get("lift_extra"),
            active=active,
        )
        v1f_scores = score_v1f_repair_candidate(v1f_features, model_path=v1f_model_path)
        cand["v1f_features"] = v1f_features
        cand["v1f_E_total"] = v1f_scores["v1f_E_total"]
        cand["v1f_success_prob"] = v1f_scores["v1f_success_prob"]
        cand["explicit_E_total"] = v1f_scores["explicit_E_total"]
        cand["v1f_uncertainty"] = v1f_scores["v1f_uncertainty"]


def enrich_context_for_physics_repair(
    context: dict[str, Any],
    *,
    success_reference_jsonl: Path | None = None,
    enable: bool | None = None,
) -> dict[str, Any]:
    if not is_physics_residual_repair_enabled(enable):
        return context
    if build_physics_repair_context is None:
        return context
    ref = success_reference_jsonl or DEFAULT_SUCCESS_REFERENCE_JSONL
    return build_physics_repair_context(base_context=context, success_reference_jsonl=ref)


def score_original_baseline_physics(
    *,
    context: dict[str, Any],
    original_rollout: dict[str, Any],
) -> dict[str, Any]:
    if compute_physics_residuals is None:
        raise RuntimeError("physics_residuals module unavailable")
    return compute_physics_residuals(original_rollout, context)


def attach_physics_to_candidate_rollouts(
    candidates: list[dict[str, Any]],
    *,
    context: dict[str, Any],
) -> None:
    if not is_physics_residual_repair_enabled() or attach_physics_residual_to_rollout is None:
        return
    for cand in candidates:
        rollout = cand.get("rollout") or cand.get("physics_rollout")
        if rollout:
            enriched = attach_physics_residual_to_rollout(rollout, context)
            cand["rollout"] = enriched
            cand["physics_total_score"] = enriched["physics_total_score"]


def _active_param_dict(candidate: dict[str, Any]) -> dict[str, float]:
    if candidate.get("insertion"):
        return {k: float(v) for k, v in candidate["insertion"].items()}
    if candidate.get("grasp_lift"):
        return {k: float(v) for k, v in candidate["grasp_lift"].items()}
    lift_extra = candidate.get("lift_extra") or {}
    grasp = candidate.get("grasp_lift") or {}
    return {**grasp, **{k: float(v) for k, v in lift_extra.items()}}


def insertion_param_vector(insertion: dict[str, float]) -> np.ndarray:
    vec = []
    for key in SEARCH_SPACE:
        choices = SEARCH_SPACE[key]
        val = insertion[key]
        if len(choices) <= 1:
            vec.append(0.0)
            continue
        lo = float(min(choices))
        hi = float(max(choices))
        vec.append((float(val) - lo) / max(hi - lo, 1e-9))
    return np.asarray(vec, dtype=float)


def pairwise_param_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0
    va = np.array([float(a[k]) for k in keys], dtype=float)
    vb = np.array([float(b[k]) for k in keys], dtype=float)
    scale = np.maximum(np.abs(va), np.abs(vb))
    scale = np.maximum(scale, 1e-6)
    return float(np.linalg.norm((va - vb) / scale))


def diverse_top_k_indices(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    top_k: int,
) -> list[int]:
    if not candidates:
        return []
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][score_key])
    selected = [order[0]]
    remaining = order[1:]
    while len(selected) < min(top_k, len(candidates)) and remaining:
        best_idx = remaining[0]
        best_min_dist = -1.0
        for idx in remaining:
            params = _active_param_dict(candidates[idx])
            min_dist = min(
                pairwise_param_distance(params, _active_param_dict(candidates[j])) for j in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def top_k_diversity_stats(candidates: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {
            "num_selected": 0,
            "mean_pairwise_param_distance": 0.0,
            "min_pairwise_param_distance": 0.0,
        }
    params_list = [_active_param_dict(candidates[i]) for i in indices]
    distances: list[float] = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            distances.append(pairwise_param_distance(params_list[a], params_list[b]))
    return {
        "num_selected": len(indices),
        "mean_pairwise_param_distance": float(np.mean(distances)) if distances else 0.0,
        "min_pairwise_param_distance": float(np.min(distances)) if distances else 0.0,
    }


def rank_theta_by_score(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    target_score: float,
) -> int:
    """1-indexed rank：score 越小越好。"""
    better = sum(1 for c in candidates if float(c[score_key]) < float(target_score))
    return int(better + 1)


def select_candidate_indices_v1f(
    candidates: list[dict[str, Any]],
    *,
    method: str,
    top_k: int,
    rng: random.Random,
    physics_context: dict[str, Any] | None = None,
    original_physics_breakdown: dict[str, Any] | None = None,
) -> list[int]:
    if method in PHYSICS_SELECTION_METHODS:
        if not is_physics_residual_repair_enabled() or select_candidates_with_physics_residuals is None:
            raise ValueError(f"{method} requires enable_physics_residual_repair=true")
        if physics_context is None or original_physics_breakdown is None:
            raise ValueError(f"{method} requires physics_context and original_physics_breakdown")
        return select_candidates_with_physics_residuals(
            candidates,
            original_breakdown=original_physics_breakdown,
            context=physics_context,
            top_k=top_k,
            rng=rng,
            require_gate=(method == "physics_residual_gated_top_k"),
        )
    if method == "v1e_pinn_top_k":
        key = "v1e_E_total"
    elif method in ("v1f_pinn_top_k", "v1f_plain_top_k"):
        key = "v1f_E_total"
    elif method == "v1f_diverse_top_k":
        return diverse_top_k_indices(candidates, score_key="v1f_E_total", top_k=top_k)
    elif method == "explicit_top_k":
        key = "explicit_E_total"
    elif method == "random_top_k":
        order = list(range(len(candidates)))
        rng.shuffle(order)
        return order[: min(top_k, len(candidates))]
    else:
        raise ValueError(f"unknown method: {method}")
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][key])
    return order[: min(top_k, len(candidates))]


def compute_ranking_metrics(rollout_results: list[dict[str, Any]]) -> dict[str, Any]:
    """从按排名顺序的 rollout 列表计算 success@k 等指标。"""
    if not rollout_results:
        return {
            "success_at_k": {f"at_{k}": 0.0 for k in (1, 3, 5, 10, 20)},
            "rollouts_per_success": float("inf"),
            "repair_rate_at_20": 0.0,
            "best_E_total": float("inf"),
        }
    successes = [bool(r.get("success_flag")) for r in rollout_results]
    energies = [float(r.get("E_total_norm", 1e9)) for r in rollout_results]
    success_at_k = {}
    for k in (1, 3, 5, 10, 20):
        kk = min(k, len(successes))
        success_at_k[f"at_{k}"] = float(any(successes[:kk]))
    rps = float("inf")
    for i, ok in enumerate(successes, start=1):
        if ok:
            rps = float(i)
            break
    budget = min(20, len(successes))
    return {
        "success_at_k": success_at_k,
        "rollouts_per_success": rps,
        "repair_rate_at_20": float(sum(successes[:budget]) / budget) if budget else 0.0,
        "best_E_total": float(min(energies)),
    }


def summarize_method_results_v1f(
    results: list[dict[str, Any]], *, method: str, rollout_budget: int
) -> dict[str, Any]:
    base = summarize_method_results(results, method=method, rollout_budget=rollout_budget)
    ranking = compute_ranking_metrics(results)
    base.update(ranking)
    return base
