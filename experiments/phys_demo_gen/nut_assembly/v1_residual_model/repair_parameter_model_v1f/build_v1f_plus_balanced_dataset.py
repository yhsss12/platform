#!/usr/bin/env python3
"""Task 2：基于 plus 数据集构建 balanced resampled 训练集。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v1f_plus_dataset import FAILURE_MODES, KNOWN_DEMO_KEYS  # noqa: E402
from v1f_repair_dataset import load_v1f_npz  # noqa: E402

DEFAULT_PLUS_NPZ = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus" / "repair_parameter_dataset_v1f_plus.npz"
DEFAULT_REPAIRABILITY = (
    _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus" / "repairability_audit" / "new_demo_repairability_report.json"
)
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced"

IMPROVEMENT_ABS = 3.0
SUCCESS_OVERSAMPLE = 5
HARD_NEGATIVE_KEEP = 1.0
EASY_FAILED_FRAC = 0.20
NO_POSITIVE_FRAC = 0.35
TARGET_PER_FAILURE_TYPE = 650
HARD_E_DROP_RATIO = 0.10


def load_repairability_map(path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["source_demo"]: row for row in report["per_demo"]}


def _is_new_demo(meta: dict[str, Any]) -> bool:
    return str(meta.get("source", "")).startswith("v1f_plus")


def _bucket_sample(
    *,
    success: bool,
    target_e: float,
    original_e: float,
    meta: dict[str, Any],
    repairability: dict[str, dict[str, Any]],
) -> tuple[str, float, bool]:
    demo_key = str(meta.get("demo_key", ""))
    repair = repairability.get(demo_key, {})
    whether = repair.get("whether_repairable", "repairable")
    ranking_eligible = whether != "no_positive_candidate"

    if success:
        return "success", float(SUCCESS_OVERSAMPLE), ranking_eligible

    if _is_new_demo(meta) and whether == "no_positive_candidate":
        return "no_positive", float(NO_POSITIVE_FRAC), False

    e_drop = original_e - target_e
    if e_drop >= IMPROVEMENT_ABS or e_drop / max(original_e, 1e-6) >= HARD_E_DROP_RATIO:
        return "hard_negative", float(HARD_NEGATIVE_KEEP), ranking_eligible

    return "easy_failed", float(EASY_FAILED_FRAC), ranking_eligible


def build_balanced_indices(
    bundle: dict[str, Any],
    meta_records: list[dict[str, Any]],
    repairability: dict[str, dict[str, Any]],
    *,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    success_flag = bundle["success_flag"]
    target_e = bundle["target_E_total"]
    original_e = bundle["original_E_total"]
    failure_idx_arr = bundle["source_failure_mode_idx"]

    pool: dict[int, list[tuple[int, float, bool, str]]] = defaultdict(list)
    for i, meta in enumerate(meta_records):
        bucket, weight, ranking_eligible = _bucket_sample(
            success=bool(success_flag[i] > 0.5),
            target_e=float(target_e[i]),
            original_e=float(original_e[i]),
            meta=meta,
            repairability=repairability,
        )
        ft = int(failure_idx_arr[i])
        pool[ft].append((i, weight, ranking_eligible, bucket))

    selected: list[int] = []
    stats: dict[str, Any] = {"by_failure_type": {}, "bucket_counts": defaultdict(int)}

    for ft, items in sorted(pool.items()):
        by_bucket: dict[str, list[tuple[int, float, bool]]] = defaultdict(list)
        for idx, weight, ranking_eligible, bucket in items:
            by_bucket[bucket].append((idx, weight, ranking_eligible))

        ft_selected: list[int] = []
        for bucket, entries in by_bucket.items():
            idxs = np.array([e[0] for e in entries], dtype=np.int64)
            weights = np.array([e[1] for e in entries], dtype=np.float64)
            if bucket == "success":
                reps = max(1, int(np.ceil(SUCCESS_OVERSAMPLE)))
                for idx in idxs:
                    ft_selected.extend([int(idx)] * reps)
            elif bucket == "hard_negative":
                ft_selected.extend(int(i) for i in idxs)
            else:
                frac = EASY_FAILED_FRAC if bucket == "easy_failed" else NO_POSITIVE_FRAC
                k = max(1, int(len(idxs) * frac))
                if len(idxs) <= k:
                    ft_selected.extend(int(i) for i in idxs)
                else:
                    pick = rng.choice(idxs, size=k, replace=False, p=weights / weights.sum())
                    ft_selected.extend(int(i) for i in pick)

        if len(ft_selected) > TARGET_PER_FAILURE_TYPE:
            pick = rng.choice(np.array(ft_selected), size=TARGET_PER_FAILURE_TYPE, replace=False)
            ft_selected = [int(i) for i in pick]
        elif len(ft_selected) < TARGET_PER_FAILURE_TYPE and items:
            extra = rng.choice(
                np.array([e[0] for e in items]),
                size=TARGET_PER_FAILURE_TYPE - len(ft_selected),
                replace=True,
            )
            ft_selected.extend(int(i) for i in extra)

        selected.extend(ft_selected)
        stats["by_failure_type"][str(ft)] = len(ft_selected)

    selected_arr = np.array(selected, dtype=np.int64)
    rng.shuffle(selected_arr)

    ranking_eligible = np.zeros(len(meta_records), dtype=np.float32)
    sample_weight = np.ones(len(meta_records), dtype=np.float32)
    demo_group_id = np.full(len(meta_records), -1, dtype=np.int64)
    demo_to_id: dict[str, int] = {}
    next_id = 0
    for i, meta in enumerate(meta_records):
        dk = str(meta.get("demo_key", f"sample_{i}"))
        if dk not in demo_to_id:
            demo_to_id[dk] = next_id
            next_id += 1
        demo_group_id[i] = demo_to_id[dk]

    for i, meta in enumerate(meta_records):
        bucket, weight, ranking_eligible_flag = _bucket_sample(
            success=bool(success_flag[i] > 0.5),
            target_e=float(target_e[i]),
            original_e=float(original_e[i]),
            meta=meta,
            repairability=repairability,
        )
        sample_weight[i] = weight
        ranking_eligible[i] = 1.0 if ranking_eligible_flag else 0.0

    for idx in selected_arr:
        bucket, _, _ = _bucket_sample(
            success=bool(success_flag[idx] > 0.5),
            target_e=float(target_e[idx]),
            original_e=float(original_e[idx]),
            meta=meta_records[idx],
            repairability=repairability,
        )
        stats["bucket_counts"][bucket] += 1

    stats["bucket_counts"] = dict(stats["bucket_counts"])
    stats["num_selected"] = int(len(selected_arr))
    stats["num_source"] = len(meta_records)
    stats["demo_group_count"] = len(demo_to_id)
    return selected_arr, stats, ranking_eligible, sample_weight, demo_group_id


def save_balanced_npz(
    *,
    source_npz: Path,
    selected_indices: np.ndarray,
    ranking_eligible: np.ndarray,
    sample_weight: np.ndarray,
    demo_group_id: np.ndarray,
    repairability: dict[str, dict[str, Any]],
    output_dir: Path,
    balance_stats: dict[str, Any],
) -> Path:
    bundle = load_v1f_npz(source_npz)
    meta_records = bundle["meta"]["meta_records"]
    idx = selected_indices

    updated_meta = []
    for i in idx:
        meta = dict(meta_records[i])
        demo_key = str(meta.get("demo_key", ""))
        repair = repairability.get(demo_key, {})
        meta["repairability_label"] = repair.get("whether_repairable", "unknown")
        meta["ranking_supervision_eligible"] = bool(ranking_eligible[i] > 0.5)
        meta["dataset_split"] = "v1f_plus_balanced"
        updated_meta.append(meta)

    summary = dict(bundle["meta"])
    summary.update(
        {
            "dataset_version": "V1-F-aligned-plus-balanced",
            "source_npz": str(source_npz),
            "num_source_samples": int(len(meta_records)),
            "num_balanced_samples": int(len(idx)),
            "balance_stats": balance_stats,
            "meta_records": updated_meta,
            "notes": summary.get("notes", [])
            + [
                "Balanced resampling: success oversample, easy failed downsample, hard negatives kept.",
                "no_positive_candidate demos marked ranking_supervision_eligible=false.",
            ],
        }
    )

    demo_idx = []
    for i in idx:
        dk = str(meta_records[i].get("demo_key", ""))
        demo_idx.append(KNOWN_DEMO_KEYS.index(dk) if dk in KNOWN_DEMO_KEYS else -1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "repair_parameter_dataset_v1f_plus_balanced.npz"
    np.savez_compressed(
        out_path,
        features=bundle["features"][idx],
        theta=bundle["theta"][idx],
        param_mask=bundle["param_mask"][idx],
        targets_components=bundle["targets_components"][idx],
        lift_residuals=bundle["lift_residuals"][idx],
        target_E_total=bundle["target_E_total"][idx],
        success_flag=bundle["success_flag"][idx],
        failure_type_idx=bundle["failure_type_idx"][idx],
        outcome_idx=bundle["outcome_idx"][idx],
        grasp_success_flag=bundle["grasp_success_flag"][idx],
        lift_success_flag=bundle["lift_success_flag"][idx],
        refined_success_flag=bundle["refined_success_flag"][idx],
        original_E_total=bundle["original_E_total"][idx],
        source_failure_mode_idx=bundle["source_failure_mode_idx"][idx],
        demo_idx=np.array(demo_idx, dtype=np.int64),
        sample_weight=sample_weight[idx],
        ranking_supervision_eligible=ranking_eligible[idx],
        demo_group_id=demo_group_id[idx],
        meta_json=json.dumps(summary),
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-F-aligned-plus-balanced dataset")
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_PLUS_NPZ)
    parser.add_argument("--repairability-report", type=Path, default=DEFAULT_REPAIRABILITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bundle = load_v1f_npz(args.source_npz)
    meta_records = bundle["meta"]["meta_records"]
    repairability = load_repairability_map(args.repairability_report)
    selected, stats, ranking_eligible, sample_weight, demo_group_id = build_balanced_indices(
        bundle, meta_records, repairability, seed=args.seed
    )
    out_path = save_balanced_npz(
        source_npz=args.source_npz,
        selected_indices=selected,
        ranking_eligible=ranking_eligible,
        sample_weight=sample_weight,
        demo_group_id=demo_group_id,
        repairability=repairability,
        output_dir=args.output_dir,
        balance_stats=stats,
    )
    manifest = {
        "dataset_version": "V1-F-aligned-plus-balanced",
        "source_npz": str(args.source_npz),
        "repairability_report": str(args.repairability_report),
        "output_npz": str(out_path),
        "num_balanced_samples": int(len(selected)),
        "balance_stats": stats,
    }
    (args.output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
