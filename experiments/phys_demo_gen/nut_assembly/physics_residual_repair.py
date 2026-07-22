"""Physics residual 修复链路集成：候选选择、CEM objective、residual_breakdown.json。"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from physics_residuals import (
    RESIDUAL_KEYS,
    calibrate_thresholds_from_success_demos,
    candidate_beats_original,
    candidate_passes_p1p2_gate,
    check_source_consistency,
    compute_effective_ranking_score,
    compute_physics_residuals,
    fallback_rate,
    format_breakdown_record,
    load_success_trajectories_from_jsonl,
    residual_delta_metrics,
    resolve_thresholds,
)

FEATURE_FLAG_ENV = "enable_physics_residual_repair"
FEATURE_FLAG_VALUE = "true"


def is_physics_residual_repair_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get(FEATURE_FLAG_ENV, "").strip().lower() == FEATURE_FLAG_VALUE


def build_physics_repair_context(
    *,
    base_context: dict[str, Any],
    success_reference_jsonl: str | Path | None = None,
    success_trajectories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ctx = dict(base_context)
    trajectories = success_trajectories
    if trajectories is None and success_reference_jsonl is not None:
        trajectories = load_success_trajectories_from_jsonl(success_reference_jsonl)
    if trajectories:
        ctx["physics_thresholds"] = calibrate_thresholds_from_success_demos(trajectories)
        from insertion_residuals import calibrate_insertion_thresholds_from_success_demos

        ctx["insertion_thresholds"] = calibrate_insertion_thresholds_from_success_demos(trajectories)
    else:
        ctx["physics_thresholds"] = resolve_thresholds(ctx)
    ctx["enable_physics_residual_repair"] = True
    return ctx


def attach_physics_residual_to_rollout(
    rollout: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    breakdown = compute_physics_residuals(rollout, context)
    rollout = dict(rollout)
    rollout["physics_residual_breakdown"] = breakdown
    rollout["physics_total_score"] = breakdown["total_score"]
    for key in RESIDUAL_KEYS:
        rollout[key] = breakdown["residuals"][key]["normalized"]
    return rollout


def score_candidate_rollout_physics(
    rollout: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Post-rollout physics score（越小越好）。"""
    return compute_physics_residuals(rollout, context)


def cem_physics_residual_objective(
    rollout: dict[str, Any],
    context: dict[str, Any],
    *,
    baseline_score: float | None = None,
) -> dict[str, Any]:
    """
    CEM repair objective：physics total_score 为主，可选相对 baseline 奖励。
    返回 score（越大越好，供 CEM 最大化）与 breakdown。
    """
    breakdown = compute_physics_residuals(rollout, context)
    total = breakdown["total_score"]
    reward = -total
    if baseline_score is not None:
        reward += max(0.0, baseline_score - total)
    p1 = breakdown["residuals"]["E_transport"]["normalized"] + breakdown["residuals"]["E_xy"]["normalized"]
    p2 = breakdown["residuals"]["E_lift"]["normalized"]
    hard_penalty = 0.0
    if p1 > 1.5:
        hard_penalty -= 500.0
    if breakdown["residuals"]["E_bilateral"]["source"] == "measured" and breakdown["residuals"]["E_bilateral"]["normalized"] > 0.9:
        hard_penalty -= 200.0
    score = float(reward + hard_penalty)
    return {
        "physics_cem_score": score,
        "physics_total_score": total,
        "physics_cem_reward": reward,
        "physics_hard_penalty": hard_penalty,
        "residual_breakdown": breakdown,
        **{k: breakdown["residuals"][k]["normalized"] for k in RESIDUAL_KEYS},
    }


def select_candidates_with_physics_residuals(
    candidates: list[dict[str, Any]],
    *,
    original_breakdown: dict[str, Any],
    context: dict[str, Any],
    top_k: int,
    rng: random.Random,
    pinn_score_key: str = "v1f_E_total",
    require_gate: bool = True,
) -> list[int]:
    """
    Physics-aware candidate selection：
    1. 有 rollout 的候选按 physics total_score 排序
    2. 可选 physics gate（优于 original）
    3. 无 rollout 时回退 PINN score
    """
    scored: list[tuple[int, float, bool]] = []
    for i, cand in enumerate(candidates):
        rollout = cand.get("rollout") or cand.get("physics_rollout")
        if rollout:
            br = compute_physics_residuals(rollout, context)
            passed, _ = candidate_beats_original(br, original_breakdown)
            if require_gate and not passed:
                continue
            scored.append((i, br["ranking_score"], True))
        elif pinn_score_key in cand:
            scored.append((i, float(cand[pinn_score_key]), False))

    if not scored:
        order = sorted(range(len(candidates)), key=lambda i: float(candidates[i].get(pinn_score_key, 1e9)))
        return order[: min(top_k, len(order))]

    # physics rollout 优先，其次 PINN proxy
    scored.sort(key=lambda x: (not x[2], x[1]))
    return [idx for idx, _, _ in scored[: min(top_k, len(scored))]]


def rank_candidates_physics_combined(
    candidates: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    original_breakdown: dict[str, Any],
    pinn_weight: float = 0.35,
    physics_weight: float = 0.65,
) -> list[tuple[int, float]]:
    """PINN + physics 混合排序（用于 CEM / repair objective）。"""
    pinn_vals = [float(c.get("v1f_E_total", c.get("explicit_E_total", 1.0))) for c in candidates]
    pinn_min, pinn_max = min(pinn_vals), max(pinn_vals)
    pinn_span = max(pinn_max - pinn_min, 1e-6)

    ranked: list[tuple[int, float]] = []
    for i, cand in enumerate(candidates):
        pinn_norm = (float(cand.get("v1f_E_total", pinn_max)) - pinn_min) / pinn_span
        rollout = cand.get("rollout") or cand.get("physics_rollout")
        if rollout:
            br = compute_physics_residuals(rollout, context)
            physics_norm = br["total_score"]
        else:
            physics_norm = pinn_norm
        combined = physics_weight * physics_norm + pinn_weight * pinn_norm
        ranked.append((i, combined))
    ranked.sort(key=lambda x: x[1])
    return ranked


def write_residual_breakdown_json(
    entries: list[dict[str, Any]],
    output_path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    meta_payload = meta or {}
    payload = {
        "schema": "nut_assembly_physics_residual_breakdown_v1",
        "enable_physics_residual_repair": True,
        "residual_keys": list(RESIDUAL_KEYS),
        "summary_by_strategy": meta_payload.get("summary_by_strategy", {}),
        "meta": meta_payload,
        "records": entries,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def select_indices_by_ranking_score(
    candidates: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    top_k: int,
    require_gate: bool = False,
    original_breakdown: dict[str, Any] | None = None,
    gate_mode: str = "full",
) -> list[int]:
    """按 ranking_score 选 top-k；可选 physics gate（full 或 p1p2）。"""
    scored: list[tuple[int, float]] = []
    for i, cand in enumerate(candidates):
        rollout = cand.get("rollout") or cand.get("physics_rollout")
        if not rollout:
            continue
        br = compute_physics_residuals(rollout, context)
        if require_gate and original_breakdown is not None:
            src = check_source_consistency(br, original_breakdown)
            if gate_mode == "p1p2":
                passed, _ = candidate_passes_p1p2_gate(
                    br, original_breakdown, source_consistency=src
                )
            else:
                passed, _ = candidate_beats_original(
                    br, original_breakdown, respect_source_consistency=True
                )
            if not passed:
                continue
        eff_score, _, _ = (
            compute_effective_ranking_score(br, original_breakdown)
            if original_breakdown is not None
            else (br["ranking_score"], [], {})
        )
        scored.append((i, eff_score))
    scored.sort(key=lambda x: x[1])
    return [idx for idx, _ in scored[: min(top_k, len(scored))]]


def select_indices_by_insertion_gated_ranking(
    candidates: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    top_k: int,
    original_breakdown: dict[str, Any],
    original_rollout: dict[str, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    """
    physics_residual_insertion_gated_top_k：
    1) P1/P2 gate；2) insertion gate；3) effective physics ranking 排序。
    """
    from insertion_residuals import (
        candidate_passes_insertion_gate,
        check_insertion_source_consistency,
        compute_insertion_residuals,
    )

    original_insertion = compute_insertion_residuals(original_rollout, context)
    insertion_thresholds = context.get("insertion_thresholds")

    scored: list[tuple[int, float]] = []
    gate_records: list[dict[str, Any]] = []

    for i, cand in enumerate(candidates):
        rollout = cand.get("rollout") or cand.get("physics_rollout")
        if not rollout:
            gate_records.append(
                {
                    "candidate_index": i,
                    "accepted": False,
                    "rejection_stage": "no_rollout",
                    "reason": "missing rollout",
                }
            )
            continue

        pbr = compute_physics_residuals(rollout, context)
        ibr = compute_insertion_residuals(rollout, context)
        src = check_source_consistency(pbr, original_breakdown)
        ins_src = check_insertion_source_consistency(ibr, original_insertion)

        p1p2_pass, p1p2_checks = candidate_passes_p1p2_gate(
            pbr, original_breakdown, source_consistency=src
        )
        record: dict[str, Any] = {
            "candidate_index": i,
            "p1p2_gate_passed": p1p2_pass,
            "p1p2_gate_checks": p1p2_checks,
            "insertion_source_consistency": ins_src,
            "insertion_residuals": ibr["residuals"],
            "physics_residuals": pbr["residuals"],
        }

        if not p1p2_pass:
            record.update(
                {
                    "accepted": False,
                    "rejection_stage": "p1p2_gate",
                    "reason": [k for k, v in p1p2_checks.items() if not v and not k.endswith("_skipped_source_mismatch")],
                }
            )
            gate_records.append(record)
            continue

        ins_pass, ins_checks = candidate_passes_insertion_gate(
            ibr,
            original_insertion,
            insertion_thresholds=insertion_thresholds,
            insertion_source_consistency=ins_src,
        )
        record["insertion_gate_checks"] = ins_checks
        if not ins_pass:
            record.update(
                {
                    "accepted": False,
                    "rejection_stage": "insertion_gate",
                    "reason": [
                        k
                        for k in ins_checks.get("hard_gate_keys", [])
                        if not ins_checks.get(k, False)
                    ],
                }
            )
            gate_records.append(record)
            continue

        eff_score, eff_keys, eff_meta = compute_effective_ranking_score(pbr, original_breakdown)
        record.update(
            {
                "accepted": True,
                "rejection_stage": None,
                "effective_ranking_score": eff_score,
                "effective_ranking_keys": eff_keys,
                "effective_ranking_meta": eff_meta,
            }
        )
        gate_records.append(record)
        scored.append((i, eff_score))

    scored.sort(key=lambda x: x[1])
    selected = [idx for idx, _ in scored[: min(top_k, len(scored))]]
    return selected, gate_records


def build_candidate_record(
    *,
    label: str,
    demo_key: str,
    strategy: str,
    rollout: dict[str, Any],
    breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
    candidate_index: int | None = None,
) -> dict[str, Any]:
    passed, gate_checks = candidate_beats_original(breakdown, original_breakdown)
    deltas = residual_delta_metrics(breakdown, original_breakdown)
    src = check_source_consistency(breakdown, original_breakdown)
    return format_breakdown_record(
        label=label,
        demo_key=demo_key,
        trajectory=rollout,
        breakdown=breakdown,
        extra={
            "strategy": strategy,
            "variant": "repaired_candidate",
            "candidate_index": candidate_index,
            "physics_gate_passed": passed,
            "physics_gate_checks": gate_checks,
            "source_consistent": src["source_consistent"],
            "source_consistency": src,
            "delta_total_score": breakdown["raw_total_score"] - original_breakdown["raw_total_score"],
            "delta_ranking_score": breakdown["ranking_score"] - original_breakdown["ranking_score"],
            **{f"delta_{key}": deltas[key] for key in RESIDUAL_KEYS},
        },
    )


def summarize_strategy_candidates(
    *,
    demo_key: str,
    strategy: str,
    original_breakdown: dict[str, Any],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """单 demo × 单策略的 residual 改善统计。"""
    n = len(candidate_records)
    if n == 0:
        return {
            "demo_key": demo_key,
            "strategy": strategy,
            "num_candidates": 0,
            "gate_pass_rate": 0.0,
            "fallback_rate_mean": 0.0,
            "source_consistency_rate": 0.0,
        }

    def _rate(improved_key: str, *, degraded: bool = False) -> dict[str, float | int]:
        if degraded:
            count = sum(1 for c in candidate_records if c.get(improved_key, 0) > 0)
        else:
            count = sum(1 for c in candidate_records if c.get(improved_key, 0) < 0)
        return {"count": count, "rate": float(count / n)}

    stats: dict[str, Any] = {
        "demo_key": demo_key,
        "strategy": strategy,
        "num_candidates": n,
        "original_raw_total_score": original_breakdown["raw_total_score"],
        "original_ranking_score": original_breakdown["ranking_score"],
        "best_raw_total_score": min(c["raw_total_score"] for c in candidate_records),
        "best_ranking_score": min(c["ranking_score"] for c in candidate_records),
        "gate_pass_count": sum(1 for c in candidate_records if c.get("physics_gate_passed")),
        "gate_pass_rate": float(sum(1 for c in candidate_records if c.get("physics_gate_passed")) / n),
        "fallback_rate_mean": float(sum(c.get("fallback_rate", 0.0) for c in candidate_records) / n),
        "source_consistency_rate": float(
            sum(1 for c in candidate_records if c.get("source_consistent")) / n
        ),
        "E_transport": _rate("delta_E_transport"),
        "E_xy": _rate("delta_E_xy"),
        "E_lift_improved": _rate("delta_E_lift"),
        "E_lift_degraded": _rate("delta_E_lift", degraded=True),
        "E_contact": _rate("delta_E_contact"),
        "E_bilateral": _rate("delta_E_bilateral"),
        "E_dynamics": _rate("delta_E_dynamics"),
        "E_slip": _rate("delta_E_slip"),
        "E_coupling": _rate("delta_E_coupling"),
        "raw_total_score": _rate("delta_total_score"),
        "ranking_score": _rate("delta_ranking_score"),
        "residual_fallback_keys": {
            key: float(
                sum(1 for c in candidate_records if c["residuals"][key]["source"] == "fallback") / n
            )
            for key in RESIDUAL_KEYS
        },
    }
    return stats


def summarize_rollout_strategy(
    *,
    demo_key: str,
    strategy: str,
    rollout_records: list[dict[str, Any]],
    original_breakdown: dict[str, Any],
) -> dict[str, Any]:
    """Rollout 验证汇总：success 率 + residual 改善率。"""
    n = len(rollout_records)
    if n == 0:
        return {
            "demo_key": demo_key,
            "strategy": strategy,
            "num_rollouts": 0,
            "transport_success_rate": 0.0,
            "xy_alignment_success_rate": 0.0,
            "lift_success_rate": 0.0,
            "partial_success_rate": 0.0,
            "final_success_rate": 0.0,
            "gate_pass_rate": 0.0,
            "fallback_rate_mean": 0.0,
            "source_consistency_rate": 0.0,
        }

    def _rate(field: str) -> float:
        return float(sum(1 for r in rollout_records if r.get(field)) / n)

    def _delta_improved(delta_key: str) -> dict[str, float | int]:
        count = sum(1 for r in rollout_records if float(r.get(delta_key, 0.0)) < 0)
        return {"count": count, "rate": float(count / n)}

    failure_counts: dict[str, int] = {}
    for r in rollout_records:
        reason = str(r.get("failure_reason", "unknown"))
        failure_counts[reason] = failure_counts.get(reason, 0) + 1

    return {
        "demo_key": demo_key,
        "strategy": strategy,
        "num_rollouts": n,
        "transport_success_rate": _rate("transport_success"),
        "xy_alignment_success_rate": _rate("xy_alignment_success"),
        "lift_success_rate": _rate("lift_success"),
        "partial_success_rate": _rate("partial_success"),
        "final_success_rate": _rate("final_success"),
        "failure_reason_counts": failure_counts,
        "mean_raw_total_score": float(
            sum(r.get("raw_total_score", 0.0) for r in rollout_records) / n
        ),
        "mean_ranking_score": float(sum(r.get("ranking_score", 0.0) for r in rollout_records) / n),
        "E_transport_improvement": _delta_improved("delta_E_transport"),
        "E_xy_improvement": _delta_improved("delta_E_xy"),
        "E_lift_improvement": _delta_improved("delta_E_lift"),
        "raw_total_score_improvement": _delta_improved("delta_total_score"),
        "ranking_score_improvement": _delta_improved("delta_ranking_score"),
        "gate_pass_rate": _rate("physics_gate_passed"),
        "fallback_rate_mean": float(sum(r.get("fallback_rate", 0.0) for r in rollout_records) / n),
        "source_consistency_rate": _rate("source_consistent"),
        "best_final_success": any(r.get("final_success") for r in rollout_records),
        "best_partial_success": any(r.get("partial_success") for r in rollout_records),
    }


def compare_original_vs_repaired(
    *,
    demo_key: str,
    original_rollout: dict[str, Any],
    repaired_rollouts: list[dict[str, Any]],
    context: dict[str, Any],
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    original_br = compute_physics_residuals(original_rollout, context)
    entries: list[dict[str, Any]] = [
        format_breakdown_record(
            label="original",
            demo_key=demo_key,
            trajectory=original_rollout,
            breakdown=original_br,
            extra={"variant": "original_failed_baseline", "strategy": "original"},
        )
    ]
    for i, rollout in enumerate(repaired_rollouts):
        br = compute_physics_residuals(rollout, context)
        label = labels[i] if labels and i < len(labels) else f"candidate_{i:02d}"
        entries.append(
            build_candidate_record(
                label=label,
                demo_key=demo_key,
                strategy="legacy",
                rollout=rollout,
                breakdown=br,
                original_breakdown=original_br,
            )
        )
    return entries
