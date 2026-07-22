#!/usr/bin/env python3
"""Physics residual repair 批量验收：三策略对比 + summary_by_strategy + MD 报告。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
_V1F_DIR = _EXPERIMENT_DIR / "v1_residual_model" / "repair_parameter_model_v1f"
for path in (_EXPERIMENT_DIR, _OFFLINE_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR,
    DEFAULT_PINN_MODEL,
    DEFAULT_RESIDUAL_BREAKDOWN_JSON,
    DEFAULT_SUCCESS_REFERENCE_JSONL,
    DEFAULT_V1F_MODEL,
    DEMO_REPAIR_CONFIGS,
)
from physics_residual_repair import (  # noqa: E402
    build_candidate_record,
    is_physics_residual_repair_enabled,
    select_indices_by_ranking_score,
    summarize_strategy_candidates,
    write_residual_breakdown_json,
)
from physics_residuals import (  # noqa: E402
    DEFAULT_RESIDUAL_WEIGHTS,
    RANKING_NORM_CAP,
    RESIDUAL_KEYS,
    compute_physics_residuals,
    format_breakdown_record,
)
from repair_common_v1f import (  # noqa: E402
    enrich_context_for_physics_repair,
    extract_baseline_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
)
from repair_rollout import run_repair_rollout  # noqa: E402

DEFAULT_VALIDATION_DEMOS = ("demo_2", "demo_3", "demo_4")
STRATEGIES = ("original", "physics_residual_top_k", "physics_residual_gated_top_k")


def _run_original_baseline_rollout(
    *,
    demo_key: str,
    cfg: dict[str, Any],
    failed_hdf5: Path,
) -> dict[str, Any]:
    search_kind = cfg["search_kind"]
    if search_kind == "lift":
        from lift_sim_search import execute_lift_rollout
        from lift_waypoint_refiner import LiftRepairParams

        return execute_lift_rollout(
            str(failed_hdf5),
            demo_key,
            "failed",
            LiftRepairParams(),
            rollout_kind="physics_residual_original_baseline",
        )
    if search_kind == "insertion":
        from sim_in_loop_refiner import run_original_waypoint_rollout

        return run_original_waypoint_rollout(
            str(failed_hdf5),
            demo_key,
            "failed",
            record_video=False,
        )
    if search_kind == "transport":
        from transport_sim_search import execute_transport_rollout
        from transport_waypoint_builder import TransportSearchParams

        return execute_transport_rollout(
            str(failed_hdf5),
            demo_key,
            "failed",
            {},
            TransportSearchParams(),
            rollout_kind="physics_residual_original_baseline",
        )
    from grasp_sim_search import execute_grasp_rollout
    from grasp_waypoint_builder import GraspSearchParams

    return execute_grasp_rollout(
        str(failed_hdf5),
        demo_key,
        "failed",
        GraspSearchParams(),
        rollout_kind="physics_residual_original_baseline",
        record_video=False,
    )


def _rollout_candidate_pool(
    *,
    demo_key: str,
    cfg: dict[str, Any],
    failed_hdf5: Path,
    cem_report: Path,
    candidates: list[dict[str, Any]],
    indices: list[int],
) -> None:
    for idx in indices:
        if candidates[idx].get("rollout"):
            continue
        candidates[idx]["rollout"] = run_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            search_kind=cfg["search_kind"],
            cem_report=cem_report if cfg["search_kind"] in ("insertion", "transport") else None,
            candidate=candidates[idx],
        )


def _original_strategy_summary(demo_key: str, original_br: dict[str, Any]) -> dict[str, Any]:
    fb = sum(
        1 for key in RESIDUAL_KEYS if original_br["residuals"][key]["source"] == "fallback"
    ) / len(RESIDUAL_KEYS)
    return {
        "demo_key": demo_key,
        "strategy": "original",
        "num_candidates": 0,
        "original_raw_total_score": original_br["raw_total_score"],
        "original_ranking_score": original_br["ranking_score"],
        "fallback_rate_mean": float(fb),
        "residual_fallback_keys": {
            key: float(original_br["residuals"][key]["source"] == "fallback") for key in RESIDUAL_KEYS
        },
    }


def generate_validation_report_md(
    *,
    summary_by_strategy: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# Physics Residual Repair 批量验收报告",
        "",
        f"- 验收 demos: {', '.join(meta.get('demos', []))}",
        f"- aligned-original checkpoint: **未修改** (`aligned_original_checkpoint_preserved=true`)",
        f"- ranking_norm_cap: {meta.get('ranking_norm_cap', RANKING_NORM_CAP)}",
        "",
        "## 1. 各 Demo 改善情况",
        "",
    ]

    improved_demos: list[str] = []
    for demo_key in meta.get("demos", []):
        demo_stats = summary_by_strategy.get(demo_key, {})
        top_k = demo_stats.get("physics_residual_top_k", {})
        gated = demo_stats.get("physics_residual_gated_top_k", {})
        orig = demo_stats.get("original", {})
        lines.append(f"### {demo_key}")
        lines.append("")
        lines.append(
            f"- original baseline: raw_total_score={orig.get('original_raw_total_score', 'n/a'):.4f}, "
            f"ranking_score={orig.get('original_ranking_score', 'n/a'):.4f}"
            if isinstance(orig.get("original_raw_total_score"), (int, float))
            else f"- original baseline: {orig}"
        )
        for strat_name, label in (
            ("physics_residual_top_k", "physics_residual_top_k"),
            ("physics_residual_gated_top_k", "physics_residual_gated_top_k"),
        ):
            s = demo_stats.get(strat_name, {})
            if not s.get("num_candidates"):
                lines.append(f"- **{label}**: 无可用候选")
                continue
            tr = s.get("E_transport", {}).get("rate", 0.0)
            xy = s.get("E_xy", {}).get("rate", 0.0)
            tot = s.get("raw_total_score", {}).get("rate", 0.0)
            gate = s.get("gate_pass_rate", 0.0)
            lines.append(
                f"- **{label}**: candidates={s['num_candidates']}, "
                f"E_transport↑{tr:.0%}, E_xy↑{xy:.0%}, total_score↑{tot:.0%}, gate_pass={gate:.0%}"
            )
            if tot >= 0.5 and tr >= 0.5:
                improved_demos.append(f"{demo_key} ({label})")
        lines.append("")

    lines.extend(["## 2. 最有效的 Residual", ""])
    agg: dict[str, list[float]] = {key: [] for key in RESIDUAL_KEYS}
    for demo_key in meta.get("demos", []):
        for strat in ("physics_residual_top_k", "physics_residual_gated_top_k"):
            s = summary_by_strategy.get(demo_key, {}).get(strat, {})
            for key in RESIDUAL_KEYS:
                rate = s.get(key if key != "E_lift" else "E_lift_improved", {}).get("rate")
                if rate is not None:
                    agg[key].append(float(rate))
    ranked = sorted(
        ((key, sum(vals) / len(vals) if vals else 0.0) for key, vals in agg.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    for key, rate in ranked:
        lines.append(f"- **{key}**: 平均改善率 {rate:.0%}")
    lines.append("")

    lines.extend(["## 3. 依赖 Fallback 的 Residual", ""])
    fb_agg: dict[str, list[float]] = {key: [] for key in RESIDUAL_KEYS}
    for demo_key in meta.get("demos", []):
        for strat in ("original", "physics_residual_top_k", "physics_residual_gated_top_k"):
            s = summary_by_strategy.get(demo_key, {}).get(strat, {})
            fb_keys = s.get("residual_fallback_keys", {})
            for key in RESIDUAL_KEYS:
                if key in fb_keys:
                    fb_agg[key].append(float(fb_keys[key]))
    for key in RESIDUAL_KEYS:
        vals = fb_agg[key]
        rate = sum(vals) / len(vals) if vals else 0.0
        tag = "⚠️ 高 fallback 依赖" if rate >= 0.5 else "✓ 以 measured 为主"
        lines.append(f"- **{key}**: fallback 率 {rate:.0%} — {tag}")
    lines.append("")

    lines.extend(["## 4. 是否建议进入 Rollout 验证", ""])
    rollout_ready = []
    for demo_key in meta.get("demos", []):
        gated = summary_by_strategy.get(demo_key, {}).get("physics_residual_gated_top_k", {})
        if gated.get("gate_pass_count", 0) > 0 and gated.get("raw_total_score", {}).get("rate", 0) >= 0.3:
            rollout_ready.append(demo_key)
    if rollout_ready:
        lines.append(f"- **建议进入 rollout 验证**: {', '.join(rollout_ready)}")
        lines.append("  - 条件: gated 策略有 gate pass 候选且 raw_total_score 改善率 ≥ 30%")
    else:
        lines.append("- **暂不建议全面 rollout**：gated 策略改善不足，可先扩大候选池或调阈值")
    lines.append("")

    lines.extend(["## 5. 是否可进入 V1-G Physics Loss 训练", ""])
    p1_rates = [
        summary_by_strategy.get(dk, {}).get("physics_residual_gated_top_k", {}).get("E_transport", {}).get("rate", 0)
        + summary_by_strategy.get(dk, {}).get("physics_residual_gated_top_k", {}).get("E_xy", {}).get("rate", 0)
        for dk in meta.get("demos", [])
    ]
    avg_p1 = sum(p1_rates) / max(len(p1_rates), 1) / 2
    high_fb = [k for k, v in fb_agg.items() if (sum(v) / len(v) if v else 0) >= 0.6]

    if avg_p1 >= 0.4 and len(high_fb) <= 3:
        lines.append("- **可以进入 V1-G physics loss 训练（条件性）**")
        lines.append("  - P1 transport/xy 在 gated 策略下平均改善率较好")
        lines.append("  - 建议: E_transport + E_xy 为主损失，E_lift 辅助；对 fallback residual 降权")
    else:
        lines.append("- **暂不建议直接进入 V1-G 全量训练**")
        lines.append(f"  - P1 平均改善率 {avg_p1:.0%}；高 fallback residual: {', '.join(high_fb) or '无'}")
        lines.append("  - 建议先补齐 contact/lift measured 字段或扩大 success reference 校准集")
    lines.append("")

    lines.extend(["## 6. Source Consistency", ""])
    for demo_key in meta.get("demos", []):
        for strat in ("physics_residual_top_k", "physics_residual_gated_top_k"):
            s = summary_by_strategy.get(demo_key, {}).get(strat, {})
            scr = s.get("source_consistency_rate")
            if scr is not None:
                lines.append(f"- {demo_key} / {strat}: source_consistency_rate={scr:.0%}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_physics_residual_validation(
    *,
    demo_keys: tuple[str, ...],
    failed_hdf5: Path,
    cem_report: Path,
    v1f_model: Path,
    v1e_model: Path,
    success_reference_jsonl: Path,
    output_json: Path,
    report_md: Path,
    num_samples: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    os.environ["enable_physics_residual_repair"] = "true"

    all_entries: list[dict[str, Any]] = []
    summary_by_strategy: dict[str, dict[str, Any]] = {}

    for demo_key in demo_keys:
        cfg = DEMO_REPAIR_CONFIGS[demo_key]
        context = extract_baseline_context_v1f(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            failure_type=cfg["failure_type"],
            search_kind=cfg["search_kind"],
        )
        context = enrich_context_for_physics_repair(
            context,
            success_reference_jsonl=success_reference_jsonl,
            enable=True,
        )

        original_rollout = _run_original_baseline_rollout(
            demo_key=demo_key,
            cfg=cfg,
            failed_hdf5=failed_hdf5,
        )
        original_br = compute_physics_residuals(original_rollout, context)
        original_record = format_breakdown_record(
            label="original",
            demo_key=demo_key,
            trajectory=original_rollout,
            breakdown=original_br,
            extra={"variant": "original_failed_baseline", "strategy": "original"},
        )
        all_entries.append(original_record)

        candidates = sample_repair_candidates_v1f(
            search_kind=cfg["search_kind"],
            n_samples=num_samples,
            seed=seed + hash(demo_key) % 10000,
        )
        score_repair_candidates_v1f(
            context=context,
            candidates=candidates,
            active=cfg["active"],
            v1e_model_path=v1e_model,
            v1f_model_path=v1f_model,
        )

        pinn_top = select_candidate_indices_v1f(
            candidates,
            method="v1f_plain_top_k",
            top_k=top_k,
            rng=random.Random(seed),
        )
        _rollout_candidate_pool(
            demo_key=demo_key,
            cfg=cfg,
            failed_hdf5=failed_hdf5,
            cem_report=cem_report,
            candidates=candidates,
            indices=pinn_top,
        )

        demo_summary: dict[str, Any] = {
            "original": _original_strategy_summary(demo_key, original_br),
        }

        strategy_specs = (
            ("physics_residual_top_k", False),
            ("physics_residual_gated_top_k", True),
        )
        for strategy_name, require_gate in strategy_specs:
            selected = select_indices_by_ranking_score(
                candidates,
                context=context,
                top_k=top_k,
                require_gate=require_gate,
                original_breakdown=original_br if require_gate else None,
            )
            cand_records: list[dict[str, Any]] = []
            for rank, idx in enumerate(selected, start=1):
                rollout = candidates[idx]["rollout"]
                br = compute_physics_residuals(rollout, context)
                rec = build_candidate_record(
                    label=f"{strategy_name}_{rank:02d}",
                    demo_key=demo_key,
                    strategy=strategy_name,
                    rollout=rollout,
                    breakdown=br,
                    original_breakdown=original_br,
                    candidate_index=idx,
                )
                cand_records.append(rec)
                all_entries.append(rec)

            demo_summary[strategy_name] = summarize_strategy_candidates(
                demo_key=demo_key,
                strategy=strategy_name,
                original_breakdown=original_br,
                candidate_records=cand_records,
            )
            print(
                f"[{demo_key}/{strategy_name}] selected={len(selected)} "
                f"gate_pass={demo_summary[strategy_name].get('gate_pass_rate', 0):.0%} "
                f"transport↑={demo_summary[strategy_name].get('E_transport', {}).get('rate', 0):.0%}",
                flush=True,
            )

        summary_by_strategy[demo_key] = demo_summary

    meta = {
        "enable_physics_residual_repair": True,
        "aligned_original_checkpoint_preserved": True,
        "default_weights": DEFAULT_RESIDUAL_WEIGHTS,
        "ranking_norm_cap": RANKING_NORM_CAP,
        "residual_keys": list(RESIDUAL_KEYS),
        "demos": list(demo_keys),
        "strategies": list(STRATEGIES),
        "summary_by_strategy": summary_by_strategy,
        "rollout_pool": {
            "pinn_prescreen_method": "v1f_plain_top_k",
            "num_samples": num_samples,
            "top_k": top_k,
        },
    }
    write_residual_breakdown_json(all_entries, output_json, meta=meta)
    generate_validation_report_md(
        summary_by_strategy=summary_by_strategy,
        meta=meta,
        output_path=report_md,
    )
    return {
        "output": str(output_json),
        "report_md": str(report_md),
        "meta": meta,
        "num_records": len(all_entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Physics residual repair batch validation")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1f-model", type=Path, default=DEFAULT_V1F_MODEL)
    parser.add_argument("--aligned-original-model", type=Path, default=None, help="alias of --v1f-model")
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESIDUAL_BREAKDOWN_JSON)
    parser.add_argument(
        "--report-md",
        type=Path,
        default=DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "physics_residual_repair_validation_report.md",
    )
    parser.add_argument("--demos", nargs="+", default=list(DEFAULT_VALIDATION_DEMOS))
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-physics-residual-repair", action="store_true")
    args = parser.parse_args()

    if args.enable_physics_residual_repair:
        os.environ["enable_physics_residual_repair"] = "true"
    if not is_physics_residual_repair_enabled():
        print("ERROR: set --enable-physics-residual-repair or enable_physics_residual_repair=true", flush=True)
        return 2

    v1f_model = args.aligned_original_model or args.v1f_model

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_physics_residual_validation(
        demo_keys=tuple(args.demos),
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        v1f_model=v1f_model,
        v1e_model=args.v1e_model,
        success_reference_jsonl=args.success_reference,
        output_json=args.output_json,
        report_md=args.report_md,
        num_samples=args.num_samples,
        top_k=args.top_k,
        seed=args.seed,
    )
    summary_path = args.output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": result["output"], "report_md": result["report_md"], "num_records": result["num_records"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
