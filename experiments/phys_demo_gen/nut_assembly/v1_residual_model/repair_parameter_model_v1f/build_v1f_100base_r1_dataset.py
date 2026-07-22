#!/usr/bin/env python3
"""V1-F-100Base-R1：demo_uid namespace 隔离数据集构建。"""
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

from build_v1f_100base_dataset import _load_jsonl_records, _records_to_samples  # noqa: E402
from build_v1f_plus_balanced_dataset import build_balanced_indices, load_repairability_map  # noqa: E402
from build_v1f_plus_dataset import FAILURE_MODES, _load_aligned_jsonl  # noqa: E402
from build_v1f_repair_dataset import _rollout_record_to_sample  # noqa: E402
from v1f_100base_utils import ALIGNED_ORIGINAL_JSONL, DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, OLD_DEMO_KEYS  # noqa: E402
from v1f_100base_r1_utils import (  # noqa: E402
    DEFAULT_100BASE_R1_OUTPUT,
    DEFAULT_DATASET_NPZ,
    DEFAULT_FAILED_ROLLOUT,
    DEFAULT_REPAIRABILITY,
    DEFAULT_SUCCESS_REFERENCE,
    DEFAULT_TARGETED_ROLLOUT,
    SOURCE_LEGACY_OLD,
    SOURCE_NEW100_FAILED,
    SOURCE_SUCCESS_REF,
    assign_r1_meta_fields,
    audit_demo_uid_collisions,
    build_demo_uid_group_ids,
    legacy_old_retention_eligible,
    make_demo_uid,
)


def _tag_base_samples(samples: list[dict[str, Any]]) -> None:
    for s in samples:
        s["meta"] = assign_r1_meta_fields(
            {
                **s.get("meta", {}),
                "source": s["meta"].get("source", "aligned_original"),
                "is_old_demo": s["meta"].get("demo_key") in OLD_DEMO_KEYS,
                "source_dataset": SOURCE_LEGACY_OLD,
                "dataset_split": "v1f_100base_r1",
            }
        )


def _tag_failed_samples(samples: list[dict[str, Any]]) -> None:
    for s in samples:
        s["meta"] = assign_r1_meta_fields(
            {
                **s.get("meta", {}),
                "source": s["meta"].get("source", "v1f_100base_failed"),
                "is_old_demo": False,
                "source_dataset": SOURCE_NEW100_FAILED,
            }
        )


def _tag_success_samples(samples: list[dict[str, Any]]) -> None:
    for s in samples:
        s["meta"] = assign_r1_meta_fields(
            {
                **s.get("meta", {}),
                "source": "v1f_100base_success_reference",
                "is_old_demo": False,
                "is_success_reference": True,
                "source_dataset": SOURCE_SUCCESS_REF,
            }
        )


def build_100base_r1_dataset(
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
    _tag_base_samples(base_samples)

    failed_records = _load_jsonl_records(failed_rollout_jsonl) + _load_jsonl_records(targeted_rollout_jsonl)
    failed_samples = _records_to_samples(failed_records, failed_hdf5=failed_hdf5, cem_report=cem_report)
    _tag_failed_samples(failed_samples)

    success_records = _load_jsonl_records(success_reference_jsonl)
    success_samples = [_rollout_record_to_sample(rec) for rec in success_records]
    _tag_success_samples(success_samples)

    merged = base_samples + failed_samples + success_samples
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

    selected, balance_stats, ranking_eligible, sample_weight, _legacy_gid = build_balanced_indices(
        bundle_like,
        meta_records,
        repairability,
        seed=seed,
    )

    demo_group_id, uid_to_id = build_demo_uid_group_ids(meta_records)
    is_success_reference = np.zeros(len(merged), dtype=np.float32)
    old_demo_retention = np.zeros(len(merged), dtype=np.float32)

    for i, meta in enumerate(meta_records):
        if meta.get("is_success_reference") or meta.get("source_dataset") == SOURCE_SUCCESS_REF:
            is_success_reference[i] = 1.0
            ranking_eligible[i] = 0.0
        if legacy_old_retention_eligible(meta):
            old_demo_retention[i] = 1.0
            sample_weight[i] = max(sample_weight[i], 2.5)

    collision_audit = audit_demo_uid_collisions(meta_records, demo_group_id)
    if not collision_audit["passed"]:
        raise SystemExit(f"demo_uid collision detected before save: {collision_audit}")

    idx = selected
    updated_meta = []
    for i in idx:
        meta = dict(meta_records[i])
        dk = str(meta.get("demo_key", ""))
        repair = repairability.get(dk, {})
        meta["repairability_label"] = repair.get("whether_repairable", "unknown")
        meta["ranking_supervision_eligible"] = bool(ranking_eligible[i] > 0.5)
        meta["demo_uid"] = str(meta.get("demo_uid", make_demo_uid(meta.get("source_dataset", ""), dk)))
        updated_meta.append(meta)

    uid_counts: dict[str, int] = defaultdict(int)
    for m in updated_meta:
        uid_counts[str(m.get("demo_uid"))] += 1

    summary: dict[str, Any] = {
        "dataset_version": "V1-F-100Base-R1",
        "grouping_policy": "demo_uid = source_dataset:demo_key",
        "num_samples": int(len(idx)),
        "num_base_aligned_original": len(base_samples),
        "num_failed_rollout": len(failed_samples),
        "num_success_reference": len(success_samples),
        "demo_uid_to_group_id": uid_to_id,
        "demo_uid_counts_in_npz": dict(uid_counts),
        "demo_uid_collision_audit": collision_audit,
        "balance_stats": balance_stats,
        "meta_records": updated_meta,
        "notes": [
            "R1: demo_group_id derived from demo_uid, not demo_key alone.",
            "old_demo_retention only legacy_old demo_0-4.",
            "success_ref excluded from ranking_supervision_eligible.",
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
        is_success_reference=is_success_reference[idx],
        meta_json=json.dumps(summary),
    )

    manifest = {
        "dataset_version": "V1-F-100Base-R1",
        "output_npz": str(output_npz),
        "num_samples": int(len(idx)),
        "demo_uid_collision_audit": collision_audit,
    }
    output_npz.parent.joinpath("build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_npz, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-F-100Base-R1 dataset")
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

    npz_path, manifest = build_100base_r1_dataset(
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
