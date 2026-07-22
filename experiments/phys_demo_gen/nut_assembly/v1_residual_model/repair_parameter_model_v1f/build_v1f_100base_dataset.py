#!/usr/bin/env python3
"""V1-F-100Base：合并 aligned-original + failed rollout + success reference，构建训练 NPZ。"""
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

from build_v1f_plus_balanced_dataset import (  # noqa: E402
    build_balanced_indices,
    load_repairability_map,
)
from build_v1f_plus_dataset import FAILURE_MODES, _load_aligned_jsonl  # noqa: E402
from build_v1f_repair_dataset import _rollout_record_to_sample  # noqa: E402
from build_v1f_aligned_dataset import apply_context_mode_to_samples  # noqa: E402
from v1f_100base_utils import (  # noqa: E402
    ALIGNED_ORIGINAL_JSONL,
    DEFAULT_100BASE_OUTPUT,
    DEFAULT_CEM_REPORT,
    DEFAULT_DATASET_NPZ,
    DEFAULT_FAILED_HDF5,
    DEFAULT_FAILED_ROLLOUT,
    DEFAULT_REPAIRABILITY,
    DEFAULT_SUCCESS_REFERENCE,
    DEFAULT_TARGETED_ROLLOUT,
    OLD_DEMO_KEYS,
)
from v1f_repair_dataset import load_v1f_npz  # noqa: E402


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _records_to_samples(
    records: list[dict[str, Any]],
    *,
    failed_hdf5: Path,
    cem_report: Path,
) -> list[dict[str, Any]]:
    raw = [_rollout_record_to_sample(rec) for rec in records]
    return apply_context_mode_to_samples(
        raw,
        context_mode="original_failed",
        failed_hdf5=failed_hdf5,
        cem_report=cem_report,
    )


def build_100base_dataset(
    *,
    aligned_jsonl: Path,
    failed_rollout_jsonl: Path,
    targeted_rollout_jsonl: Path,
    success_reference_jsonl: Path,
    repairability_report: Path,
    failed_hdf5: Path,
    cem_report: Path,
    output_npz: Path,
    seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    base_samples = _load_aligned_jsonl(aligned_jsonl)
    for s in base_samples:
        s["meta"] = {
            **s.get("meta", {}),
            "source": s["meta"].get("source", "aligned_original"),
            "is_old_demo": s["meta"].get("demo_key") in OLD_DEMO_KEYS,
            "dataset_split": "v1f_100base",
        }

    failed_records = _load_jsonl_records(failed_rollout_jsonl) + _load_jsonl_records(targeted_rollout_jsonl)
    failed_samples = _records_to_samples(failed_records, failed_hdf5=failed_hdf5, cem_report=cem_report)
    for s in failed_samples:
        s["meta"] = {**s.get("meta", {}), "source": s["meta"].get("source", "v1f_100base_failed"), "is_old_demo": False}

    success_records = _load_jsonl_records(success_reference_jsonl)
    success_samples = [_rollout_record_to_sample(rec) for rec in success_records]
    for s in success_samples:
        s["meta"] = {
            **s.get("meta", {}),
            "source": "v1f_100base_success_reference",
            "is_old_demo": False,
            "is_success_reference": True,
        }

    merged = base_samples + failed_samples + success_samples

    source_counts: dict[str, int] = defaultdict(int)
    demo_counts: dict[str, int] = defaultdict(int)
    failure_counts: dict[str, int] = defaultdict(int)
    for sample in merged:
        src = sample["meta"].get("source", "unknown")
        source_counts[src] += 1
        demo_counts[sample["meta"].get("demo_key", "unknown")] += 1
        failure_counts[sample["context"]["source_failure_type"]] += 1

    meta_records = [s["meta"] for s in merged]
    repairability = load_repairability_map(repairability_report) if repairability_report.exists() else {}

    bundle_like = {
        "features": np.stack([s["features"] for s in merged], axis=0),
        "success_flag": np.array([s["success_flag"] for s in merged], dtype=np.float32),
        "target_E_total": np.array([s["target_E_total"] for s in merged], dtype=np.float32),
        "original_E_total": np.array([s["context"]["original_E_total_norm"] for s in merged], dtype=np.float32),
        "source_failure_mode_idx": np.array(
            [
                FAILURE_MODES.index(s["context"]["source_failure_type"])
                if s["context"]["source_failure_type"] in FAILURE_MODES
                else 2
                for s in merged
            ],
            dtype=np.int64,
        ),
    }

    selected, balance_stats, ranking_eligible, sample_weight, demo_group_id = build_balanced_indices(
        bundle_like,
        meta_records,
        repairability,
        seed=seed,
    )

    old_demo_retention = np.zeros(len(merged), dtype=np.float32)
    for i, meta in enumerate(meta_records):
        if meta.get("is_old_demo") or meta.get("demo_key") in OLD_DEMO_KEYS:
            old_demo_retention[i] = 1.0
            sample_weight[i] = max(sample_weight[i], 2.5)

    idx = selected
    updated_meta = []
    for i in idx:
        meta = dict(meta_records[i])
        dk = str(meta.get("demo_key", ""))
        repair = repairability.get(dk, {})
        meta["repairability_label"] = repair.get("whether_repairable", "unknown")
        meta["ranking_supervision_eligible"] = bool(ranking_eligible[i] > 0.5)
        if meta_records[i].get("is_success_reference"):
            meta["is_success_reference"] = True
        if meta_records[i].get("is_old_demo"):
            meta["is_old_demo"] = True
        updated_meta.append(meta)

    success_e = [float(merged[i]["target_E_total"]) for i in range(len(merged)) if merged[i]["meta"].get("is_success_reference")]
    summary: dict[str, Any] = {
        "dataset_version": "V1-F-100Base",
        "context_mode": "original_failed",
        "num_samples": int(len(idx)),
        "num_base_aligned_original": len(base_samples),
        "num_failed_rollout": len(failed_samples),
        "num_success_reference": len(success_samples),
        "num_merged_before_balance": len(merged),
        "source_counts": dict(source_counts),
        "demo_counts": dict(demo_counts),
        "failure_type_counts": dict(failure_counts),
        "balance_stats": balance_stats,
        "success_reference_E_total_mean": float(np.mean(success_e)) if success_e else None,
        "success_reference_E_total_p95": float(np.percentile(success_e, 95)) if success_e else None,
        "init_checkpoint_policy": "model_v1f_aligned_original.pt only (not v2)",
        "meta_records": updated_meta,
        "notes": [
            "V1-F-100Base = aligned-original (1357) + Square_D0 failed rollouts + success reference.",
            "23 failed demos use audit-classified failure_type (not unknown).",
            "77 success demos provide residual reference / threshold calibration.",
            "Balanced resampling with old_demo_retention weighting on demo_0-4.",
        ],
    }

    demo_idx = []
    old_keys = list(OLD_DEMO_KEYS)
    for i in idx:
        dk = str(meta_records[i].get("demo_key", ""))
        demo_idx.append(old_keys.index(dk) if dk in old_keys else -1)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        features=np.stack([merged[i]["features"] for i in idx]),
        theta=np.stack([merged[i]["theta"] for i in idx]),
        param_mask=np.stack([merged[i]["param_mask"] for i in idx]),
        targets_components=np.stack([merged[i]["targets_components"] for i in idx]),
        lift_residuals=np.stack([merged[i]["lift_residuals"] for i in idx]),
        target_E_total=np.array([merged[i]["target_E_total"] for i in idx], dtype=np.float32),
        success_flag=np.array([merged[i]["success_flag"] for i in idx], dtype=np.float32),
        failure_type_idx=np.array([merged[i]["failure_type_idx"] for i in idx], dtype=np.int64),
        outcome_idx=np.array([merged[i]["outcome_idx"] for i in idx], dtype=np.int64),
        grasp_success_flag=np.array([merged[i]["grasp_success_flag"] for i in idx], dtype=np.float32),
        lift_success_flag=np.array([merged[i]["lift_success_flag"] for i in idx], dtype=np.float32),
        refined_success_flag=np.array([merged[i]["refined_success_flag"] for i in idx], dtype=np.float32),
        original_E_total=np.array([merged[i]["context"]["original_E_total_norm"] for i in idx], dtype=np.float32),
        source_failure_mode_idx=bundle_like["source_failure_mode_idx"][idx],
        demo_idx=np.array(demo_idx, dtype=np.int64),
        sample_weight=sample_weight[idx],
        ranking_supervision_eligible=ranking_eligible[idx],
        demo_group_id=demo_group_id[idx],
        old_demo_retention=old_demo_retention[idx],
        meta_json=json.dumps(summary),
    )

    manifest = {
        "dataset_version": "V1-F-100Base",
        "aligned_jsonl": str(aligned_jsonl),
        "failed_rollout_jsonl": str(failed_rollout_jsonl),
        "targeted_rollout_jsonl": str(targeted_rollout_jsonl),
        "success_reference_jsonl": str(success_reference_jsonl),
        "output_npz": str(output_npz),
        "num_samples": int(len(idx)),
        "num_base": len(base_samples),
        "num_failed": len(failed_samples),
        "num_success_reference": len(success_samples),
    }
    output_npz.parent.joinpath("build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_npz, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-F-100Base dataset")
    parser.add_argument("--aligned-jsonl", type=Path, default=ALIGNED_ORIGINAL_JSONL)
    parser.add_argument("--failed-rollout", type=Path, default=DEFAULT_FAILED_ROLLOUT)
    parser.add_argument("--targeted-rollout", type=Path, default=DEFAULT_TARGETED_ROLLOUT)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE)
    parser.add_argument("--repairability-report", type=Path, default=DEFAULT_REPAIRABILITY)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    npz_path, manifest = build_100base_dataset(
        aligned_jsonl=args.aligned_jsonl,
        failed_rollout_jsonl=args.failed_rollout,
        targeted_rollout_jsonl=args.targeted_rollout,
        success_reference_jsonl=args.success_reference,
        repairability_report=args.repairability_report,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        output_npz=args.output_npz,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
