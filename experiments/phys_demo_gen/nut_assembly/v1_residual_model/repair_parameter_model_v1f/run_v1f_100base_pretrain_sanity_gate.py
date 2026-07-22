#!/usr/bin/env python3
"""V1-F-100Base pre-train sanity gate：dataset 构建后、训练前必须通过。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1F_DIR.parent.parent
for path in (_EXPERIMENT_DIR, _V1F_DIR.parent, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v1f_plus_balanced_dataset import IMPROVEMENT_ABS, HARD_E_DROP_RATIO  # noqa: E402
from build_v1f_plus_balanced_dataset import load_repairability_map  # noqa: E402
from v1f_100base_utils import (  # noqa: E402
    DEFAULT_AUDIT_REPORT,
    DEFAULT_DATASET_NPZ,
    DEFAULT_FAILED_HDF5,
    DEFAULT_FAILED_ROLLOUT,
    DEFAULT_REPAIRABILITY,
    DEFAULT_SANITY_REPORT,
    DEFAULT_SANITY_SUMMARY,
    DEFAULT_SUCCESS_REFERENCE,
    DEFAULT_TARGETED_ROLLOUT,
    NEW_FAILED_SOURCE_PREFIXES,
    OLD_DEMO_KEYS,
    OLD_STABLE_SOURCE_PREFIXES,
    SUCCESS_REFERENCE_SOURCES,
    V2_FORBIDDEN_TOKENS,
)
from v1f_plus_utils import load_failure_map  # noqa: E402
from v1f_repair_dataset import load_v1f_npz  # noqa: E402

NEAR_SUCCESS_E_DROP = IMPROVEMENT_ABS
NEAR_SUCCESS_RATIO = HARD_E_DROP_RATIO
DOMINANCE_MAX_NEW_TO_OLD_RATIO = 1.5
WEAK_SUPERVISION_FAILURE_TYPES = (
    "transport_failed",
    "insertion_failed",
    "alignment_failed",
    "grasp_failed",
    "lift_failed",
)


def _is_old_stable_source(source: str) -> bool:
    return any(source.startswith(p) or source == p for p in OLD_STABLE_SOURCE_PREFIXES)


def _is_new_failed_source(source: str) -> bool:
    return any(source.startswith(p) for p in NEW_FAILED_SOURCE_PREFIXES)


def _is_success_reference_source(source: str) -> bool:
    return source in SUCCESS_REFERENCE_SOURCES or "success_reference" in source


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _scan_forbidden_sources(*paths: Path, meta_records: list[dict[str, Any]]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for tok in V2_FORBIDDEN_TOKENS:
            if tok in text:
                hits.append(f"{path.name}: contains '{tok}'")
    for i, meta in enumerate(meta_records):
        blob = json.dumps(meta)
        for tok in V2_FORBIDDEN_TOKENS:
            if tok in blob:
                hits.append(f"meta_records[{i}]: source={meta.get('source')} matches '{tok}'")
    return sorted(set(hits))


def _classify_sample(
    *,
    success: bool,
    target_e: float,
    original_e: float,
) -> str:
    if success:
        return "success"
    e_drop = original_e - target_e
    if e_drop >= NEAR_SUCCESS_E_DROP or e_drop / max(original_e, 1e-6) >= NEAR_SUCCESS_RATIO:
        return "hard_negative"
    if e_drop >= NEAR_SUCCESS_E_DROP * 0.5 or e_drop / max(original_e, 1e-6) >= NEAR_SUCCESS_RATIO * 0.5:
        return "near_success"
    return "easy_failed"


def _old_demo_retention_stats(
    meta_records: list[dict[str, Any]],
    success_flag: np.ndarray,
    target_e: np.ndarray,
    original_e: np.ndarray,
    old_demo_retention: np.ndarray | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dk in OLD_DEMO_KEYS:
        indices = [
            i
            for i, m in enumerate(meta_records)
            if m.get("demo_key") == dk
            and (old_demo_retention is None or old_demo_retention[i] > 0.5 or _is_old_stable_source(str(m.get("source", ""))))
        ]
        stable_indices = [i for i in indices if _is_old_stable_source(str(meta_records[i].get("source", "")))]
        new_indices = [i for i in indices if _is_new_failed_source(str(meta_records[i].get("source", "")))]
        success_indices = [i for i in indices if success_flag[i] > 0.5 and not _is_success_reference_source(str(meta_records[i].get("source", "")))]
        hard_neg = [
            i
            for i in indices
            if success_flag[i] <= 0.5
            and _classify_sample(success=False, target_e=float(target_e[i]), original_e=float(original_e[i]))
            == "hard_negative"
        ]
        out[dk] = {
            "total_samples_in_npz": len(indices),
            "old_stable_samples": len(stable_indices),
            "new_failed_samples_same_key": len(new_indices),
            "success_theta_count": len(success_indices),
            "hard_negative_count": len(hard_neg),
            "retention_present": len(stable_indices) > 0 or (old_demo_retention is not None and any(old_demo_retention[i] > 0.5 for i in indices)),
        }
    return out


def _failure_type_from_audit(audit_report: Path, demo_key: str, meta_fallback: str) -> str:
    if audit_report.exists():
        report = json.loads(audit_report.read_text(encoding="utf-8"))
        ft = report.get("failed_demo_classification", {}).get(demo_key)
        if ft:
            return str(ft)
    return meta_fallback


def _new_failed_demo_stats(
    meta_records: list[dict[str, Any]],
    success_flag: np.ndarray,
    target_e: np.ndarray,
    original_e: np.ndarray,
    repairability: dict[str, dict[str, Any]],
    audit_report: Path,
    rollout_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rollout_by_demo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        if "demo_failed(1)" in str(row.get("source_file", "")):
            rollout_by_demo[str(row.get("source_demo", row.get("demo_key", "")))].append(row)

    demo_keys = sorted(rollout_by_demo.keys(), key=lambda k: int(k.split("_")[-1]))
    per_demo: list[dict[str, Any]] = []
    for dk in demo_keys:
        rows = rollout_by_demo[dk]
        ft = _failure_type_from_audit(audit_report, dk, str(rows[0].get("original_failure_type", "unknown")))
        successes = [r for r in rows if r.get("success_flag")]
        e_vals = [float(r.get("E_total", 1e9)) for r in rows]
        best_idx = int(np.argmin(e_vals)) if e_vals else 0
        best_row = rows[best_idx] if rows else {}
        near_success = 0
        hard_neg = 0
        for r in rows:
            ok = bool(r.get("success_flag"))
            te = float(r.get("E_total", 0.0))
            oe = float(r.get("context", {}).get("original_E_total_norm", te))
            bucket = _classify_sample(success=ok, target_e=te, original_e=oe)
            if bucket == "near_success":
                near_success += 1
            elif bucket == "hard_negative":
                hard_neg += 1
        repair = repairability.get(dk, {})
        best_xy = best_row.get("final_xy")
        if best_xy is None and "rollout" in best_row:
            best_xy = best_row["rollout"].get("final_nut_peg_xy", best_row["rollout"].get("final_xy"))
        per_demo.append(
            {
                "demo_key": dk,
                "failure_type": ft,
                "total_rollout_samples": len(rows),
                "success_count": len(successes),
                "near_success_count": near_success,
                "hard_negative_count": hard_neg,
                "best_E_total": float(min(e_vals)) if e_vals else None,
                "best_nut_peg_xy": float(best_xy) if best_xy is not None else None,
                "best_success_prob": float(max(float(r.get("success_flag", 0)) for r in rows)) if rows else 0.0,
                "repairability_label": repair.get("whether_repairable", "unknown"),
            }
        )
    return per_demo


def _failure_type_balance(
    meta_records: list[dict[str, Any]],
    success_flag: np.ndarray,
    target_e: np.ndarray,
    original_e: np.ndarray,
    audit_report: Path,
) -> dict[str, Any]:
    by_ft: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(meta_records):
        if _is_success_reference_source(str(meta.get("source", ""))):
            continue
        dk = str(meta.get("demo_key", ""))
        ft = _failure_type_from_audit(audit_report, dk, str(meta.get("source_failure_type", "unknown")))
        by_ft[ft].append(i)

    stats: dict[str, Any] = {}
    counts = {ft: len(idxs) for ft, idxs in by_ft.items()}
    total = sum(counts.values()) or 1
    max_share = max(counts.values()) / total if counts else 0.0
    dominated = max_share > 0.65

    for ft in WEAK_SUPERVISION_FAILURE_TYPES:
        idxs = by_ft.get(ft, [])
        succ = sum(1 for i in idxs if success_flag[i] > 0.5)
        near = sum(
            1
            for i in idxs
            if success_flag[i] <= 0.5
            and _classify_sample(success=False, target_e=float(target_e[i]), original_e=float(original_e[i]))
            in ("near_success", "hard_negative")
        )
        weak = len(idxs) > 0 and succ == 0 and near == 0
        stats[ft] = {
            "sample_count": len(idxs),
            "success_count": succ,
            "near_success_or_hard_negative_count": near,
            "weak_supervision": weak,
        }

    return {
        "by_failure_type": stats,
        "failure_type_counts": counts,
        "dominated_by_single_type": dominated,
        "max_failure_type_share": max_share,
    }


def _success_reference_check(
    meta_records: list[dict[str, Any]],
    success_flag: np.ndarray,
    target_e: np.ndarray,
    summary: dict[str, Any],
) -> dict[str, Any]:
    ref_indices = [i for i, m in enumerate(meta_records) if _is_success_reference_source(str(m.get("source", "")))]
    repair_success_from_ref = sum(1 for i in ref_indices if success_flag[i] > 0.5 and meta_records[i].get("is_success_reference"))
    non_ref_repair_success = sum(
        1
        for i, m in enumerate(meta_records)
        if success_flag[i] > 0.5 and not _is_success_reference_source(str(m.get("source", "")))
    )
    ref_e = [float(target_e[i]) for i in ref_indices]
    return {
        "success_reference_sample_count": len(ref_indices),
        "success_reference_only_for_calibration": len(ref_indices) > 0,
        "success_reference_labeled_in_meta": sum(1 for i in ref_indices if meta_records[i].get("is_success_reference")),
        "mislabeled_as_repair_success_theta": repair_success_from_ref,
        "repair_rollout_success_theta_count": non_ref_repair_success,
        "E_total_mean": float(np.mean(ref_e)) if ref_e else None,
        "E_total_p95": float(np.percentile(ref_e, 95)) if ref_e else None,
        "dataset_summary_success_ref_mean": summary.get("success_reference_E_total_mean"),
        "note": (
            "Success reference 样本仅用于 residual reference / threshold calibration；"
            "不得与普通 failed-demo repair success theta 混计。"
        ),
    }


def evaluate_training_gate(report: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    ds = report["data_sources"]

    if not ds.get("has_aligned_original_old_stable"):
        reasons.append("缺少 aligned-original old stable samples")
    if not ds.get("has_demo_failed_1_rollout"):
        reasons.append("缺少 demo_failed(1) repair rollout samples")
    if ds.get("success_reference_sample_count_in_npz", 0) <= 0:
        reasons.append("NPZ 中无 success reference 样本（threshold calibration）")
    if ds.get("contains_v2_or_deprecated"):
        reasons.append("dataset source 含 v2 / deprecated 标记: " + ", ".join(ds.get("v2_forbidden_hits", [])[:3]))

    old_ret = report["old_demo_retention"]
    d4 = old_ret.get("demo_4", {})
    d2 = old_ret.get("demo_2", {})

    if not d4.get("retention_present"):
        reasons.append("old demo_4 retention 不存在（无 aligned-original stable 样本）")
    if d4.get("success_theta_count", 0) <= 0:
        reasons.append("old demo_4 success theta 数 = 0")
    if not d2.get("retention_present"):
        reasons.append("old demo_2 retention 不存在")
    if d2.get("success_theta_count", 0) <= 0:
        reasons.append("old demo_2 success theta 数 = 0")

    new_failed = report.get("new_failed_demos", [])
    repairable_with_signal = [
        d
        for d in new_failed
        if d.get("repairability_label") in ("repairable", "hard_but_improvable")
        and (d.get("success_count", 0) > 0 or d.get("near_success_count", 0) > 0)
    ]
    if not repairable_with_signal:
        reasons.append("新 23 failed demos 中无 repairable/hard 且带 success 或 near-success 的 demo")

    ft_counts = report["data_sources"].get("failure_type_counts_in_npz", {})
    non_unknown = sum(v for k, v in ft_counts.items() if k not in ("unknown", "success"))
    if ft_counts and non_unknown == 0:
        reasons.append("failure_type 全部为 unknown（无已分类失败类型）")

    drowned = report.get("old_demo_not_drowned", {})
    if not drowned.get("passed", False):
        reasons.append(drowned.get("reason", "old demo_0–4 被新失败样本淹没"))

    if report["failure_type_balance"].get("weak_supervision_types"):
        pass  # warning only, not blocking unless all repairable fail

    allowed = len(reasons) == 0
    return {
        "training_allowed": allowed,
        "block_reasons": reasons,
        "repairable_demos_with_signal": [d["demo_key"] for d in repairable_with_signal],
    }


def build_sanity_report(
    *,
    dataset_npz: Path,
    audit_report: Path,
    repairability_report: Path,
    failed_rollout: Path,
    targeted_rollout: Path,
    success_reference: Path,
    build_manifest: Path,
) -> dict[str, Any]:
    bundle = load_v1f_npz(dataset_npz)
    summary = bundle["meta"]
    meta_records: list[dict[str, Any]] = summary.get("meta_records", [])
    success_flag = bundle["success_flag"]
    target_e = bundle["target_E_total"]
    original_e = bundle["original_E_total"]
    old_demo_retention = bundle.get("old_demo_retention")

    repairability = load_repairability_map(repairability_report) if repairability_report.exists() else {}
    rollout_rows = _load_jsonl(failed_rollout) + _load_jsonl(targeted_rollout)

    source_counts: dict[str, int] = defaultdict(int)
    for m in meta_records:
        source_counts[str(m.get("source", "unknown"))] += 1

    ft_npz: dict[str, int] = defaultdict(int)
    for m in meta_records:
        if _is_success_reference_source(str(m.get("source", ""))):
            continue
        dk = str(m.get("demo_key", ""))
        ft_npz[_failure_type_from_audit(audit_report, dk, str(m.get("source_failure_type", "unknown")))] += 1

    forbidden = _scan_forbidden_sources(
        dataset_npz,
        build_manifest,
        failed_rollout,
        targeted_rollout,
        success_reference,
        meta_records=meta_records,
    )

    old_ret = _old_demo_retention_stats(
        meta_records, success_flag, target_e, original_e, old_demo_retention
    )
    not_drowned: dict[str, Any] = {"per_demo": {}, "passed": True, "reason": ""}
    for dk in OLD_DEMO_KEYS:
        stable = old_ret[dk]["old_stable_samples"]
        new_same = old_ret[dk]["new_failed_samples_same_key"]
        ratio = new_same / max(stable, 1)
        ok = stable > 0 and ratio <= DOMINANCE_MAX_NEW_TO_OLD_RATIO
        not_drowned["per_demo"][dk] = {
            "old_stable": stable,
            "new_failed_same_key": new_same,
            "new_to_old_ratio": ratio,
            "ok": ok,
        }
        if not ok and stable > 0:
            not_drowned["passed"] = False
            not_drowned["reason"] = f"{dk} 新失败样本占比过高 (ratio={ratio:.2f} > {DOMINANCE_MAX_NEW_TO_OLD_RATIO})"
        elif stable == 0 and dk in ("demo_4", "demo_2"):
            not_drowned["passed"] = False
            not_drowned["reason"] = f"{dk} 无 old stable 样本"

    ft_balance = _failure_type_balance(meta_records, success_flag, target_e, original_e, audit_report)
    weak_types = [ft for ft, s in ft_balance["by_failure_type"].items() if s.get("weak_supervision")]
    ft_balance["weak_supervision_types"] = weak_types

    report: dict[str, Any] = {
        "task": "v1f_100base_pretrain_sanity_gate",
        "dataset_npz": str(dataset_npz),
        "num_samples_in_npz": len(meta_records),
        "data_sources": {
            "has_aligned_original_old_stable": any(_is_old_stable_source(s) for s in source_counts),
            "aligned_original_sample_count": sum(v for s, v in source_counts.items() if _is_old_stable_source(s)),
            "has_demo_failed_1_rollout": any("demo_failed(1)" in json.dumps(r) for r in rollout_rows),
            "new_failed_rollout_jsonl_count": len(rollout_rows),
            "has_success_reference_statistics": success_reference.exists(),
            "success_reference_sample_count_in_npz": sum(
                1 for m in meta_records if _is_success_reference_source(str(m.get("source", "")))
            ),
            "source_counts_in_npz": dict(source_counts),
            "failure_type_counts_in_npz": dict(ft_npz),
            "v2_forbidden_hits": forbidden,
            "contains_v2_or_deprecated": len(forbidden) > 0,
        },
        "old_demo_retention": old_ret,
        "old_demo_not_drowned": not_drowned,
        "new_failed_demos": _new_failed_demo_stats(
            meta_records, success_flag, target_e, original_e, repairability, audit_report, rollout_rows
        ),
        "failure_type_balance": ft_balance,
        "success_reference": _success_reference_check(meta_records, success_flag, target_e, summary),
    }
    report["training_gate"] = evaluate_training_gate(report)
    return report


def to_markdown(report: dict[str, Any]) -> str:
    gate = report["training_gate"]
    lines = [
        "# V1-F-100Base Pre-Train Sanity Gate",
        "",
        f"**训练放行**: {'是 ✓' if gate['training_allowed'] else '否 ✗ — 已阻止训练'}",
        "",
    ]
    if gate["block_reasons"]:
        lines.append("## 阻止原因")
        for r in gate["block_reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    ds = report["data_sources"]
    lines.extend(
        [
            "## 1. 数据来源",
            "",
            f"- aligned-original old stable: **{ds['has_aligned_original_old_stable']}** ({ds['aligned_original_sample_count']} samples)",
            f"- demo_failed(1) rollout: **{ds['has_demo_failed_1_rollout']}** ({ds['new_failed_rollout_jsonl_count']} jsonl records)",
            f"- success reference in NPZ: **{ds['success_reference_sample_count_in_npz']}**",
            f"- v2/deprecated 污染: **{ds['contains_v2_or_deprecated']}**",
            "",
            "## 2. Old Demo Retention (demo_0–demo_4)",
            "",
            "| demo | total | old_stable | success_θ | hard_neg | retention |",
            "|------|-------|------------|-----------|----------|-----------|",
        ]
    )
    for dk in OLD_DEMO_KEYS:
        s = report["old_demo_retention"][dk]
        lines.append(
            f"| {dk} | {s['total_samples_in_npz']} | {s['old_stable_samples']} | "
            f"{s['success_theta_count']} | {s['hard_negative_count']} | {s['retention_present']} |"
        )

    lines.extend(["", "## 3. 新 23 Failed Demos", ""])
    lines.append(
        "| demo | failure_type | rollouts | success | near_succ | hard_neg | best_E | repairability |"
    )
    lines.append("|------|--------------|----------|---------|-----------|----------|--------|---------------|")
    for d in report.get("new_failed_demos", []):
        be = d.get("best_E_total")
        be_s = f"{be:.2f}" if be is not None else "N/A"
        lines.append(
            f"| {d['demo_key']} | {d['failure_type']} | {d['total_rollout_samples']} | "
            f"{d['success_count']} | {d['near_success_count']} | {d['hard_negative_count']} | "
            f"{be_s} | {d['repairability_label']} |"
        )

    fb = report["failure_type_balance"]
    lines.extend(["", "## 4. Failure Type 平衡", ""])
    for ft, s in fb.get("by_failure_type", {}).items():
        weak = " **weak**" if s.get("weak_supervision") else ""
        lines.append(
            f"- `{ft}`: n={s['sample_count']}, success={s['success_count']}, "
            f"near/hard={s['near_success_or_hard_negative_count']}{weak}"
        )
    if fb.get("weak_supervision_types"):
        lines.append(f"- weak supervision 类型: {', '.join(fb['weak_supervision_types'])}")

    sr = report["success_reference"]
    lines.extend(
        [
            "",
            "## 5. Success Reference",
            "",
            f"- 样本数: {sr['success_reference_sample_count']}",
            f"- 仅用于 calibration: {sr['success_reference_only_for_calibration']}",
            f"- repair success θ 混用计数: {sr['mislabeled_as_repair_success_theta']} (应为 0 或仅 reference 自身)",
            f"- E_total mean/p95: {sr.get('E_total_mean')} / {sr.get('E_total_p95')}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-100Base pre-train sanity gate")
    parser.add_argument("--dataset-npz", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--repairability-report", type=Path, default=DEFAULT_REPAIRABILITY)
    parser.add_argument("--failed-rollout", type=Path, default=DEFAULT_FAILED_ROLLOUT)
    parser.add_argument("--targeted-rollout", type=Path, default=DEFAULT_TARGETED_ROLLOUT)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE)
    parser.add_argument("--build-manifest", type=Path, default=DEFAULT_DATASET_NPZ.parent / "build_manifest.json")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_SANITY_REPORT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_SANITY_SUMMARY)
    args = parser.parse_args()

    if not args.dataset_npz.exists():
        raise SystemExit(f"Dataset NPZ not found: {args.dataset_npz}")

    report = build_sanity_report(
        dataset_npz=args.dataset_npz,
        audit_report=args.audit_report,
        repairability_report=args.repairability_report,
        failed_rollout=args.failed_rollout,
        targeted_rollout=args.targeted_rollout,
        success_reference=args.success_reference,
        build_manifest=args.build_manifest,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.write_text(to_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "training_allowed": report["training_gate"]["training_allowed"],
                "block_reasons": report["training_gate"]["block_reasons"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["training_gate"]["training_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
