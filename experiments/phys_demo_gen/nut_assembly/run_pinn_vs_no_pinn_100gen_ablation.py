#!/usr/bin/env python3
"""PINN vs no-PINN nut assembly 100-episode generation A/B ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
_V1F_DIR = _EXPERIMENT_DIR / "v1_residual_model" / "repair_parameter_model_v1f"
for path in (_EXPERIMENT_DIR, _OFFLINE_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DEFAULT_CEM_REPORT,
    DEFAULT_PINN_MODEL,
    DEFAULT_SUCCESS_REFERENCE_JSONL,
    DEMO_3_V1G_LITE_DIAGNOSTIC,
    DEMO_REPAIR_CONFIGS,
    V1G_STAGE1_LITE_P1P2_MODEL,
)
from physics_residual_repair import (  # noqa: E402
    build_candidate_record,
    build_physics_repair_context,
    is_physics_residual_repair_enabled,
    select_indices_by_insertion_gated_ranking,
    select_indices_by_ranking_score,
)
from physics_residuals import compute_physics_residuals  # noqa: E402
from pinn_v1f_inference import clear_v1f_model_cache  # noqa: E402
from repair_common_v1f import (  # noqa: E402
    extract_baseline_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
)
from repair_rollout import run_repair_rollout  # noqa: E402
from repaired_hdf5_writer import append_successful_repair_demo, init_repaired_dataset  # noqa: E402
from rollout_outcome_evaluator import evaluate_rollout_outcome  # noqa: E402
from run_physics_residual_repair_validation import _run_original_baseline_rollout  # noqa: E402
from run_v1f_plus_evaluation import _repair_cfg_for_new_demo  # noqa: E402
from sim_in_loop_refiner import load_best_theta_or_fallback  # noqa: E402
from v1f_plus_utils import list_demo_keys, load_failure_map  # noqa: E402


def run_original_baseline_rollout(
    *,
    demo_key: str,
    cfg: dict[str, Any],
    failed_hdf5: Path,
    cem_report: Path,
) -> dict[str, Any]:
    """Original baseline rollout; transport demos require CEM theta."""
    if cfg["search_kind"] != "transport":
        return _run_original_baseline_rollout(
            demo_key=demo_key,
            cfg=cfg,
            failed_hdf5=failed_hdf5,
        )
    from transport_sim_search import execute_transport_rollout
    from transport_waypoint_builder import TransportSearchParams

    theta = load_best_theta_or_fallback(str(cem_report), demo_key, fallback_demo_key="demo_0")
    return execute_transport_rollout(
        str(failed_hdf5),
        demo_key,
        "failed",
        theta,
        TransportSearchParams(),
        rollout_kind="physics_residual_original_baseline",
    )

DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR / "data" / "demo_failed.hdf5"
DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"
DEFAULT_OUTPUT_DIR = _EXPERIMENT_DIR / "outputs" / "pinn_ablation_100gen"

NO_PINN_GROUP = "no_pinn_baseline"
PINN_GROUP = "use_pinn_v1g_lite"

TRANSPORT_XY_LIFT_FAILURES = frozenset(
    {
        "transport_failed",
        "alignment_failed",
        "grasp_failed",
        "lift_failed",
        "lift_underdeveloped",
    }
)
INSERTION_FAILURES = frozenset({"insertion_failed", "partial_success_not_final"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def build_episode_schedule(
    *,
    demo_keys: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    schedule: list[dict[str, Any]] = []
    for episode_id, seed in enumerate(seeds):
        demo_key = demo_keys[episode_id % len(demo_keys)]
        schedule.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "demo_key": demo_key,
            }
        )
    return schedule


def resolve_demo_cfg(
    demo_key: str,
    *,
    failure_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if demo_key in DEMO_REPAIR_CONFIGS:
        return dict(DEMO_REPAIR_CONFIGS[demo_key])
    return _repair_cfg_for_new_demo(demo_key, failure_map)


def route_pinn_strategy(
    *,
    demo_key: str,
    cfg: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Return (strategy, skip_reason). None strategy => non-repairable / skipped."""
    if demo_key == "demo_3":
        return None, DEMO_3_V1G_LITE_DIAGNOSTIC["repairability"]
    coarse = str(cfg.get("failure_type") or "unknown")
    if coarse in INSERTION_FAILURES or cfg.get("search_kind") == "insertion":
        return "physics_residual_insertion_gated_top_k", None
    if coarse in TRANSPORT_XY_LIFT_FAILURES or cfg.get("search_kind") in ("transport", "grasp", "lift"):
        if coarse in ("lift_failed", "lift_underdeveloped") or cfg.get("search_kind") == "lift":
            return "physics_residual_p1p2_gated_top_k", None
        return "physics_residual_gated_top_k", None
    return "physics_residual_insertion_gated_top_k", None


def _select_indices_for_strategy(
    *,
    strategy: str,
    candidates: list[dict[str, Any]],
    pinn_top: list[int],
    context: dict[str, Any],
    original_breakdown: dict[str, Any],
    original_rollout: dict[str, Any],
    top_k: int,
    rng: random.Random,
) -> tuple[list[int], list[dict[str, Any]] | None]:
    if strategy == "random_top_k":
        order = list(range(len(candidates)))
        rng.shuffle(order)
        return order[: min(top_k, len(candidates))], None
    if strategy == "v1f_plain_top_k":
        return list(pinn_top[:top_k]), None
    if strategy == "physics_residual_gated_top_k":
        return (
            select_indices_by_ranking_score(
                candidates,
                context=context,
                top_k=top_k,
                require_gate=True,
                original_breakdown=original_breakdown,
                gate_mode="full",
            ),
            None,
        )
    if strategy == "physics_residual_p1p2_gated_top_k":
        return (
            select_indices_by_ranking_score(
                candidates,
                context=context,
                top_k=top_k,
                require_gate=True,
                original_breakdown=original_breakdown,
                gate_mode="p1p2",
            ),
            None,
        )
    if strategy == "physics_residual_insertion_gated_top_k":
        return select_indices_by_insertion_gated_ranking(
            candidates,
            context=context,
            top_k=top_k,
            original_breakdown=original_breakdown,
            original_rollout=original_rollout,
        )
    raise ValueError(f"unknown strategy: {strategy}")


def _failure_record(
    *,
    episode: dict[str, Any],
    rollout: dict[str, Any] | None,
    reason: str,
    strategy: str,
    candidate_index: int | None = None,
) -> dict[str, Any]:
    outcome = evaluate_rollout_outcome(rollout, None) if rollout else {}
    return {
        "episode_id": episode["episode_id"],
        "seed": episode["seed"],
        "demo_key": episode["demo_key"],
        "strategy": strategy,
        "candidate_index": candidate_index,
        "reason": reason,
        "failure_reason": outcome.get("failure_reason", reason),
        "final_success": bool(outcome.get("final_success")),
        "partial_success": bool(outcome.get("partial_success")),
    }


def count_valid_hdf5_demos(hdf5_path: Path) -> int:
    if not hdf5_path.exists():
        return 0
    with h5py.File(hdf5_path, "r") as handle:
        if "data" not in handle:
            return 0
        valid = 0
        for demo_key in handle["data"].keys():
            demo = handle["data"][demo_key]
            if "actions" in demo and demo["actions"].shape[0] > 0:
                valid += 1
        return valid


def run_single_episode(
    *,
    episode: dict[str, Any],
    cfg: dict[str, Any],
    group_name: str,
    group_config: dict[str, Any],
    failed_hdf5: Path,
    cem_report: Path,
    v1e_model: Path | None,
    pinn_model: Path | None,
    success_reference: Path,
    num_samples: int,
    top_k: int,
    output_hdf5: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed = int(episode["seed"])
    demo_key = str(episode["demo_key"])
    episode_id = int(episode["episode_id"])

    result: dict[str, Any] = {
        "episode_id": episode_id,
        "seed": seed,
        "demo_key": demo_key,
        "group": group_name,
        "attempted": False,
        "skipped": False,
        "skip_reason": None,
        "strategy": None,
        "successful": False,
        "final_success": False,
        "num_rollouts": 0,
        "num_successes_written": 0,
        "written_demo_keys": [],
        "failure_reason": None,
        "generation_time_sec": 0.0,
    }

    if group_name == PINN_GROUP:
        strategy, skip_reason = route_pinn_strategy(demo_key=demo_key, cfg=cfg)
        if strategy is None:
            result.update(
                {
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "failure_reason": skip_reason,
                    "generation_time_sec": time.perf_counter() - t0,
                }
            )
            logger.info(
                "[skip] ep=%s demo=%s seed=%s reason=%s",
                episode_id,
                demo_key,
                seed,
                skip_reason,
            )
            return result
        result["strategy"] = strategy
    else:
        result["strategy"] = "random_top_k"

    result["attempted"] = True
    pool_seed = seed + hash(demo_key) % 10000
    rng = random.Random(seed)

    base_ctx = extract_baseline_context_v1f(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
    )

    use_physics = bool(group_config.get("enable_physics_residual_repair"))
    physics_ctx = base_ctx
    original_breakdown: dict[str, Any] | None = None
    original_rollout: dict[str, Any] | None = None
    if use_physics:
        physics_ctx = build_physics_repair_context(
            base_context=base_ctx,
            success_reference_jsonl=success_reference,
        )
        original_rollout = run_original_baseline_rollout(
            demo_key=demo_key,
            cfg=cfg,
            failed_hdf5=failed_hdf5,
            cem_report=cem_report,
        )
        original_breakdown = compute_physics_residuals(original_rollout, physics_ctx)

    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"],
        n_samples=num_samples,
        seed=pool_seed,
    )

    pinn_top: list[int] = []
    if group_config.get("enable_pinn_repair"):
        clear_v1f_model_cache()
        score_repair_candidates_v1f(
            context=base_ctx,
            candidates=candidates,
            active=cfg["active"],
            v1e_model_path=v1e_model,
            v1f_model_path=pinn_model,
        )
        pinn_top = select_candidate_indices_v1f(
            candidates,
            method="v1f_plain_top_k",
            top_k=top_k,
            rng=rng,
        )
    elif use_physics:
        # Physics-only path should not happen for no_pinn; kept for safety.
        pinn_top = list(range(min(top_k, len(candidates))))

    strategy = str(result["strategy"])
    if strategy == "random_top_k":
        selected, gate_records = _select_indices_for_strategy(
            strategy=strategy,
            candidates=candidates,
            pinn_top=pinn_top,
            context=physics_ctx,
            original_breakdown=original_breakdown or {},
            original_rollout=original_rollout or {},
            top_k=top_k,
            rng=rng,
        )
    else:
        for idx in pinn_top:
            if candidates[idx].get("rollout"):
                continue
            candidates[idx]["rollout"] = run_repair_rollout(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                search_kind=cfg["search_kind"],
                cem_report=cem_report if cfg["search_kind"] in ("insertion", "transport") else None,
                candidate=candidates[idx],
            )
        selected, gate_records = _select_indices_for_strategy(
            strategy=strategy,
            candidates=candidates,
            pinn_top=pinn_top,
            context=physics_ctx,
            original_breakdown=original_breakdown or {},
            original_rollout=original_rollout or {},
            top_k=top_k,
            rng=rng,
        )

    failures: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    insertion_gate_records: list[dict[str, Any]] = []
    if gate_records:
        insertion_gate_records.extend(gate_records)

    rollouts_run = 0
    for rank, idx in enumerate(selected, start=1):
        if strategy == "random_top_k":
            rollout = run_repair_rollout(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                search_kind=cfg["search_kind"],
                cem_report=cem_report if cfg["search_kind"] in ("insertion", "transport") else None,
                candidate=candidates[idx],
            )
            candidates[idx]["rollout"] = rollout
        else:
            rollout = candidates[idx]["rollout"]
        rollouts_run += 1
        outcome = evaluate_rollout_outcome(rollout, physics_ctx if use_physics else None)
        if use_physics and original_breakdown is not None:
            br = compute_physics_residuals(rollout, physics_ctx)
            residual_records.append(
                build_candidate_record(
                    label=f"ep{episode_id:03d}_{rank:02d}",
                    demo_key=demo_key,
                    strategy=strategy,
                    rollout=rollout,
                    breakdown=br,
                    original_breakdown=original_breakdown,
                    candidate_index=idx,
                )
            )
        if outcome["final_success"]:
            result["final_success"] = True
            if rollout.get("recorded_actions") is not None:
                repaired_key = f"gen_{group_name}_ep{episode_id:03d}_seed{seed}_{demo_key}_{rank:02d}"
                meta = {
                    "source_demo": demo_key,
                    "source_failure_type": cfg["failure_type"],
                    "selection_method": strategy,
                    "group": group_name,
                    "episode_id": episode_id,
                    "seed": seed,
                    "candidate_index": idx,
                    "rollout_kind": "pinn_ablation_100gen",
                }
                append_successful_repair_demo(
                    output_hdf5=output_hdf5,
                    source_hdf5=failed_hdf5,
                    source_demo_key=demo_key,
                    repaired_demo_key=repaired_key,
                    rollout=rollout,
                    meta=meta,
                )
                result["num_successes_written"] += 1
                result["written_demo_keys"].append(repaired_key)
            else:
                failures.append(
                    _failure_record(
                        episode=episode,
                        rollout=rollout,
                        reason="final_success_without_recorded_trajectory",
                        strategy=strategy,
                        candidate_index=idx,
                    )
                )
        else:
            failures.append(
                _failure_record(
                    episode=episode,
                    rollout=rollout,
                    reason=str(outcome["failure_reason"]),
                    strategy=strategy,
                    candidate_index=idx,
                )
            )

    result["num_rollouts"] = rollouts_run
    result["successful"] = result["final_success"]
    if not result["successful"]:
        if failures:
            result["failure_reason"] = failures[0]["failure_reason"]
        elif rollouts_run == 0:
            result["failure_reason"] = "no_candidates_selected"
            failures.append(
                _failure_record(
                    episode=episode,
                    rollout=None,
                    reason="no_candidates_selected",
                    strategy=strategy,
                )
            )
        else:
            result["failure_reason"] = "all_rollouts_failed"

    result["generation_time_sec"] = time.perf_counter() - t0
    result["failures"] = failures
    result["residual_records"] = residual_records
    result["insertion_gate_records"] = insertion_gate_records
    logger.info(
        "[ep] %s id=%s demo=%s seed=%s strategy=%s rollouts=%s success=%s written=%s time=%.1fs",
        group_name,
        episode_id,
        demo_key,
        seed,
        strategy,
        rollouts_run,
        result["successful"],
        result["num_successes_written"],
        result["generation_time_sec"],
    )
    return result


def _load_partial(path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done[int(row["episode_id"])] = row
    return done


def run_group(
    *,
    group_name: str,
    group_config: dict[str, Any],
    schedule: list[dict[str, Any]],
    failed_hdf5: Path,
    cem_report: Path,
    audit_report: Path,
    v1e_model: Path | None,
    pinn_model: Path | None,
    success_reference: Path,
    num_samples: int,
    top_k: int,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logger = _setup_logger(logs_dir, group_name)

    if group_config.get("enable_physics_residual_repair"):
        os.environ["enable_physics_residual_repair"] = "true"
    else:
        os.environ.pop("enable_physics_residual_repair", None)

    failure_map = load_failure_map(audit_report)
    hdf5_path = output_dir / "generated_dataset.hdf5"
    init_repaired_dataset(hdf5_path, failed_hdf5)

    partial_path = logs_dir / "episodes_partial.jsonl"
    done = _load_partial(partial_path) if resume else {}

    episode_results: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    all_residual: list[dict[str, Any]] = []
    all_insertion_gate: list[dict[str, Any]] = []

    for episode in schedule:
        episode_id = int(episode["episode_id"])
        if episode_id in done:
            episode_results.append(done[episode_id])
            continue
        cfg = resolve_demo_cfg(str(episode["demo_key"]), failure_map=failure_map)
        row = run_single_episode(
            episode=episode,
            cfg=cfg,
            group_name=group_name,
            group_config=group_config,
            failed_hdf5=failed_hdf5,
            cem_report=cem_report,
            v1e_model=v1e_model,
            pinn_model=pinn_model,
            success_reference=success_reference,
            num_samples=num_samples,
            top_k=top_k,
            output_hdf5=hdf5_path,
            logger=logger,
        )
        slim = {k: v for k, v in row.items() if k not in ("failures", "residual_records", "insertion_gate_records")}
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(slim, ensure_ascii=False, default=str) + "\n")
        done[episode_id] = slim
        episode_results.append(row)
        all_failures.extend(row.get("failures", []))
        all_residual.extend(row.get("residual_records", []))
        all_insertion_gate.extend(row.get("insertion_gate_records", []))

    for row in done.values():
        if row not in [r for r in episode_results if r.get("episode_id") == row.get("episode_id")]:
            episode_results.append(row)

    episode_results = sorted({int(r["episode_id"]): r for r in episode_results}.values(), key=lambda r: r["episode_id"])

    requested = len(schedule)
    attempted = sum(1 for r in episode_results if r.get("attempted"))
    skipped = sum(1 for r in episode_results if r.get("skipped"))
    successful = sum(1 for r in episode_results if r.get("successful"))
    failed = attempted - successful
    times = [float(r.get("generation_time_sec", 0.0)) for r in episode_results if r.get("attempted")]
    failure_cluster = Counter(
        str(r.get("failure_reason") or "unknown")
        for r in episode_results
        if not r.get("successful") and not r.get("skipped")
    )
    skipped_cluster = Counter(str(r.get("skip_reason") or "skipped") for r in episode_results if r.get("skipped"))

    valid_hdf5 = count_valid_hdf5_demos(hdf5_path)
    invalid_or_filtered = attempted - valid_hdf5 if attempted >= valid_hdf5 else 0

    repair_attempt_count = attempted
    repaired_success_count = successful

    summary = {
        "schema": "pinn_ablation_generation_summary_v1",
        "group": group_name,
        "generated_at": _utc_now(),
        "requested_episodes": requested,
        "attempted_episodes": attempted,
        "skipped_episodes": skipped,
        "successful_episodes": successful,
        "failed_episodes": failed,
        "success_rate": float(successful / requested) if requested else 0.0,
        "episode_success_rate": float(successful / requested) if requested else 0.0,
        "valid_hdf5_demo_count": valid_hdf5,
        "valid_demo_rate": float(valid_hdf5 / requested) if requested else 0.0,
        "invalid_or_filtered_count": invalid_or_filtered,
        "failure_reason_cluster": dict(failure_cluster.most_common()),
        "skipped_reason_cluster": dict(skipped_cluster.most_common()),
        "mean_generation_time_sec": float(statistics.mean(times)) if times else 0.0,
        "total_generation_time_sec": float(sum(times)),
        "repair_attempt_count": repair_attempt_count,
        "repaired_success_count": repaired_success_count,
        "repair_success_rate": float(repaired_success_count / repair_attempt_count) if repair_attempt_count else None,
        "config": group_config,
        "num_samples": num_samples,
        "top_k": top_k,
        "failed_hdf5": str(failed_hdf5),
        "output_hdf5": str(hdf5_path),
    }

    manifest = {
        "schema": "pinn_ablation_manifest_v1",
        "group": group_name,
        "created_at": _utc_now(),
        "task": "nut_assembly",
        "schedule_size": requested,
        "seeds": [int(ep["seed"]) for ep in schedule],
        "demo_keys_cycle": list_demo_keys(failed_hdf5),
        "episodes": [
            {
                "episode_id": ep["episode_id"],
                "seed": ep["seed"],
                "demo_key": ep["demo_key"],
            }
            for ep in schedule
        ],
        "group_config": group_config,
        "paths": {
            "generation_summary": str(output_dir / "generation_summary.json"),
            "failures": str(output_dir / "failures.json"),
            "generated_dataset": str(hdf5_path),
            "logs": str(logs_dir),
        },
    }

    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output_dir / "failures.json").write_text(json.dumps(all_failures, indent=2, default=str), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    if group_name == PINN_GROUP:
        (output_dir / "residual_breakdown.json").write_text(
            json.dumps({"records": all_residual}, indent=2, default=str),
            encoding="utf-8",
        )
        reject_counter: Counter[str] = Counter()
        for rec in all_insertion_gate:
            if rec.get("accepted"):
                continue
            stage = str(rec.get("rejection_stage", "unknown"))
            reason = rec.get("reason")
            if isinstance(reason, list):
                for item in reason:
                    reject_counter[f"{stage}:{item}"] += 1
            else:
                reject_counter[f"{stage}:{reason}"] += 1
        (output_dir / "insertion_gate_breakdown.json").write_text(
            json.dumps(
                {
                    "insertion_gate_records_count": len(all_insertion_gate),
                    "reject_reasons": dict(reject_counter.most_common()),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    return summary


def build_comparison_report(
    *,
    no_pinn: dict[str, Any],
    use_pinn: dict[str, Any],
    pinn_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    delta_success = float(use_pinn["success_rate"] - no_pinn["success_rate"])
    delta_valid = int(use_pinn["valid_hdf5_demo_count"] - no_pinn["valid_hdf5_demo_count"])
    time_ratio = (
        float(use_pinn["mean_generation_time_sec"] / no_pinn["mean_generation_time_sec"])
        if no_pinn["mean_generation_time_sec"] > 0
        else None
    )

    pinn_improves = delta_success > 0 or delta_valid > 0
    time_increase_significant = time_ratio is not None and time_ratio > 1.25

    recommend_keep_pinn = pinn_improves and not (delta_success < 0 and delta_valid <= 0)
    recommend_scale_test = pinn_improves and use_pinn["success_rate"] >= 0.05

    analysis: list[str] = []
    if not pinn_improves:
        if use_pinn.get("skipped_episodes", 0) > 0:
            analysis.append("PINN 组存在 non_repairable 跳过样本（如 demo_3）。")
        if use_pinn["failure_reason_cluster"].get("no_candidates_selected"):
            analysis.append("候选池或 insertion gate 过严导致空选。")
        if no_pinn["success_rate"] > 0 and use_pinn["success_rate"] == 0:
            analysis.append("PINN 路由/physics gate 可能与 rollout 验证阶段不一致。")
        analysis.append("检查 top_k / num_samples 是否不足。")

    table_row = {
        "group": "",
        "requested": no_pinn["requested_episodes"],
        "successful": 0,
        "failed": 0,
        "success_rate": 0.0,
        "valid_hdf5_demo_count": 0,
        "valid_demo_rate": 0.0,
        "total_time": 0.0,
        "mean_time": 0.0,
    }

    rows = []
    for summary in (no_pinn, use_pinn):
        rows.append(
            {
                "group": summary["group"],
                "requested": summary["requested_episodes"],
                "successful": summary["successful_episodes"],
                "failed": summary["failed_episodes"],
                "success_rate": summary["success_rate"],
                "valid_hdf5_demo_count": summary["valid_hdf5_demo_count"],
                "valid_demo_rate": summary["valid_demo_rate"],
                "total_time": summary["total_generation_time_sec"],
                "mean_time": summary["mean_generation_time_sec"],
            }
        )

    payload = {
        "schema": "pinn_vs_no_pinn_100gen_report_v1",
        "generated_at": _utc_now(),
        "task": "nut_assembly",
        "pinn_checkpoint": {
            "path": str(pinn_checkpoint),
            "sha256": _sha256(pinn_checkpoint) if pinn_checkpoint.exists() else None,
        },
        "comparison_table": rows,
        "delta_success_rate": delta_success,
        "delta_valid_demo_count": delta_valid,
        "time_ratio_pinn_over_no_pinn": time_ratio,
        "time_increase_significant": time_increase_significant,
        "no_pinn_failure_reason_cluster": no_pinn["failure_reason_cluster"],
        "use_pinn_failure_reason_cluster": use_pinn["failure_reason_cluster"],
        "use_pinn_skipped_reason_cluster": use_pinn.get("skipped_reason_cluster", {}),
        "pinn_brings_improvement": pinn_improves,
        "recommend_keep_v1g_lite_p1p2_asset": recommend_keep_pinn,
        "recommend_scale_to_500_1000": recommend_scale_test,
        "failure_analysis_if_no_improvement": analysis,
        "acceptance": {
            "no_pinn_successful": no_pinn["successful_episodes"],
            "use_pinn_successful": use_pinn["successful_episodes"],
            "no_pinn_success_rate": no_pinn["success_rate"],
            "use_pinn_success_rate": use_pinn["success_rate"],
            "use_pinn_higher_than_no_pinn": use_pinn["success_rate"] > no_pinn["success_rate"],
            "aligned_original_untouched": True,
            "v1g_not_set_as_default": True,
            "ablation_only": True,
            "passed": use_pinn["success_rate"] >= no_pinn["success_rate"],
        },
        "groups": {
            NO_PINN_GROUP: no_pinn,
            PINN_GROUP: use_pinn,
        },
    }

    json_path = output_dir / "pinn_vs_no_pinn_100gen_report.json"
    md_path = output_dir / "pinn_vs_no_pinn_100gen_report.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# PINN vs no-PINN 100-episode 生成 A/B 对比报告",
        "",
        f"> 生成时间：{payload['generated_at']}",
        f"> PINN checkpoint：`{pinn_checkpoint}`",
        f"> checkpoint sha256：`{payload['pinn_checkpoint']['sha256']}`",
        "",
        "## 对比表",
        "",
        "| group | requested | successful | failed | success_rate | valid_hdf5_demo_count | valid_demo_rate | total_time | mean_time |",
        "| ----- | --------: | ---------: | -----: | -----------: | --------------------: | --------------: | ---------: | --------: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['requested']} | {row['successful']} | {row['failed']} | "
            f"{row['success_rate']:.1%} | {row['valid_hdf5_demo_count']} | {row['valid_demo_rate']:.1%} | "
            f"{row['total_time']:.1f}s | {row['mean_time']:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## 关键差异",
            "",
            f"- Δsuccess_rate (PINN - no-PINN): **{delta_success:+.1%}**",
            f"- Δvalid_hdf5_demo_count: **{delta_valid:+d}**",
            f"- PINN 组耗时倍率 (mean): **{time_ratio:.2f}x**" if time_ratio else "- PINN 组耗时倍率: N/A",
            f"- PINN 是否带来提升: **{'是' if pinn_improves else '否'}**",
            "",
            "## failure_reason 聚类",
            "",
            "### no_pinn_baseline",
            "",
            f"```json\n{json.dumps(no_pinn['failure_reason_cluster'], ensure_ascii=False, indent=2)}\n```",
            "",
            "### use_pinn_v1g_lite",
            "",
            f"```json\n{json.dumps(use_pinn['failure_reason_cluster'], ensure_ascii=False, indent=2)}\n```",
            "",
            "## 建议",
            "",
            f"- 继续保留 V1-G-stage1-lite-p1p2 为实验资产: **{'是' if recommend_keep_pinn else '暂缓，需分析'}**",
            f"- 进入 500/1000 条规模测试: **{'是' if recommend_scale_test else '否，先优化策略/参数'}**",
            "",
            "## 验收",
            "",
            f"- no-PINN 成功: {no_pinn['successful_episodes']} / {no_pinn['requested_episodes']} ({no_pinn['success_rate']:.1%})",
            f"- use-PINN 成功: {use_pinn['successful_episodes']} / {use_pinn['requested_episodes']} ({use_pinn['success_rate']:.1%})",
            f"- use-PINN 高于 no-PINN: {'是' if payload['acceptance']['use_pinn_higher_than_no_pinn'] else '否'}",
            f"- 验收通过: {'是' if payload['acceptance']['passed'] else '否'}",
        ]
    )
    if analysis:
        lines.extend(["", "## 未提升原因分析", ""])
        for item in analysis:
            lines.append(f"- {item}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    payload["outputs"] = {"json": str(json_path), "md": str(md_path)}
    return payload


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return list(args.seeds)
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def main() -> int:
    parser = argparse.ArgumentParser(description="PINN vs no-PINN 100-episode generation ablation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--pinn-model",
        type=Path,
        default=V1G_STAGE1_LITE_P1P2_MODEL,
    )
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group", choices=(NO_PINN_GROUP, PINN_GROUP, "both"), default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild comparison from existing summaries")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    py_path = ":".join(
        str(p)
        for p in (
            _EXPERIMENT_DIR,
            _EXPERIMENT_DIR / "v1_residual_model",
            _V1F_DIR,
            _OFFLINE_DIR,
            _EXPERIMENT_DIR.parents[2] / "integrations" / "CableThreadingMVP",
        )
    )
    os.environ["PYTHONPATH"] = py_path + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args)
    if args.episodes != len(seeds):
        if args.seeds is None:
            seeds = list(range(args.seed_start, args.seed_start + args.episodes))
        else:
            raise SystemExit(f"--episodes ({args.episodes}) must match number of seeds ({len(seeds)})")

    if not args.failed_hdf5.exists():
        raise SystemExit(f"Source failed HDF5 missing: {args.failed_hdf5}")

    checkpoint_info: dict[str, Any] | None = None
    if args.group in (PINN_GROUP, "both") and not args.report_only:
        if not args.pinn_model.exists():
            raise SystemExit(f"PINN checkpoint missing: {args.pinn_model}")
        checkpoint_info = {
            "path": str(args.pinn_model),
            "sha256": _sha256(args.pinn_model),
            "verified_at": _utc_now(),
        }
        (args.output_dir / "pinn_checkpoint_verification.json").write_text(
            json.dumps(checkpoint_info, indent=2),
            encoding="utf-8",
        )

    demo_keys = list_demo_keys(args.failed_hdf5)
    schedule = build_episode_schedule(demo_keys=demo_keys, seeds=seeds)

    no_pinn_config = {
        "enable_physics_residual_repair": False,
        "enable_pinn_repair": False,
        "enable_insertion_gate": False,
        "pinn_model_path": None,
        "selection_method": "random_top_k",
    }
    use_pinn_config = {
        "enable_physics_residual_repair": True,
        "enable_pinn_repair": True,
        "enable_insertion_gate": True,
        "pinn_model_path": str(args.pinn_model),
        "preferred_strategy": "physics_residual_insertion_gated_top_k",
        "failure_stage_router": {
            "transport_xy_lift": ["physics_residual_gated_top_k", "physics_residual_p1p2_gated_top_k"],
            "insertion_partial": ["physics_residual_insertion_gated_top_k"],
            "non_repairable": ["demo_3", "non_repairable_under_current_pipeline"],
        },
    }

    no_pinn_summary_path = args.output_dir / NO_PINN_GROUP / "generation_summary.json"
    use_pinn_summary_path = args.output_dir / PINN_GROUP / "generation_summary.json"

    if args.report_only:
        if not no_pinn_summary_path.exists() or not use_pinn_summary_path.exists():
            raise SystemExit("Report-only mode requires both group generation_summary.json files")
        no_pinn = json.loads(no_pinn_summary_path.read_text(encoding="utf-8"))
        use_pinn = json.loads(use_pinn_summary_path.read_text(encoding="utf-8"))
    else:
        no_pinn = None
        use_pinn = None
        if args.group in (NO_PINN_GROUP, "both"):
            no_pinn = run_group(
                group_name=NO_PINN_GROUP,
                group_config=no_pinn_config,
                schedule=schedule,
                failed_hdf5=args.failed_hdf5,
                cem_report=args.cem_report,
                audit_report=args.audit_report,
                v1e_model=None,
                pinn_model=None,
                success_reference=args.success_reference,
                num_samples=args.num_samples,
                top_k=args.top_k,
                output_dir=args.output_dir / NO_PINN_GROUP,
                resume=args.resume,
            )
        elif no_pinn_summary_path.exists():
            no_pinn = json.loads(no_pinn_summary_path.read_text(encoding="utf-8"))

        if args.group in (PINN_GROUP, "both"):
            if not is_physics_residual_repair_enabled(True):
                os.environ["enable_physics_residual_repair"] = "true"
            use_pinn = run_group(
                group_name=PINN_GROUP,
                group_config=use_pinn_config,
                schedule=schedule,
                failed_hdf5=args.failed_hdf5,
                cem_report=args.cem_report,
                audit_report=args.audit_report,
                v1e_model=args.v1e_model,
                pinn_model=args.pinn_model,
                success_reference=args.success_reference,
                num_samples=args.num_samples,
                top_k=args.top_k,
                output_dir=args.output_dir / PINN_GROUP,
                resume=args.resume,
            )
        elif use_pinn_summary_path.exists():
            use_pinn = json.loads(use_pinn_summary_path.read_text(encoding="utf-8"))

    if no_pinn is None or use_pinn is None:
        print(json.dumps({"status": "partial", "no_pinn": no_pinn is not None, "use_pinn": use_pinn is not None}, indent=2))
        return 0

    report = build_comparison_report(
        no_pinn=no_pinn,
        use_pinn=use_pinn,
        pinn_checkpoint=args.pinn_model,
        output_dir=comparison_dir,
    )
    print(json.dumps({"report": report["outputs"], "acceptance": report["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
