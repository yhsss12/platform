#!/usr/bin/env python3
"""Task 3：balanced dataset + targeted rollout 样本 -> v2 balanced npz。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v1f_aligned_dataset import apply_context_mode_to_samples  # noqa: E402
from build_v1f_plus_balanced_dataset import (  # noqa: E402
    IMPROVEMENT_ABS,
    SUCCESS_OVERSAMPLE,
    TARGET_PER_FAILURE_TYPE,
    build_balanced_indices,
    load_repairability_map,
    save_balanced_npz,
)
from build_v1f_plus_dataset import FAILURE_MODES, KNOWN_DEMO_KEYS, _load_aligned_jsonl  # noqa: E402
from build_v1f_repair_dataset import _rollout_record_to_sample  # noqa: E402
from v1f_plus_utils import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PLUS_OUTPUT  # noqa: E402
from v1f_repair_dataset import load_v1f_npz  # noqa: E402

DEFAULT_BALANCED_NPZ = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "repair_parameter_dataset_v1f_plus_balanced.npz"
DEFAULT_TARGETED = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced_v2" / "targeted_rollout_samples.jsonl"
DEFAULT_REPAIRABILITY = (
    _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus" / "repairability_audit" / "new_demo_repairability_report.json"
)
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced_v2"


def build_v2_dataset(
    *,
    balanced_npz: Path,
    targeted_jsonl: Path,
    repairability_report: Path,
    failed_hdf5: Path,
    cem_report: Path,
    output_dir: Path,
    seed: int,
) -> tuple[Path, dict]:
    # 从 plus aligned jsonl + new rollout 重建 base，再叠加已有 balanced 规模逻辑
    aligned_jsonl = DEFAULT_PLUS_OUTPUT.parent / "v1f_aligned_repair_parameter_model" / "original_failed" / "repair_parameter_dataset_v1f.jsonl"
    plus_rollout = DEFAULT_PLUS_OUTPUT / "new_rollout_samples.jsonl"
    base_samples = _load_aligned_jsonl(aligned_jsonl)
    if plus_rollout.exists():
        with plus_rollout.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    base_samples.append(_rollout_record_to_sample(json.loads(line)))

    targeted_records: list[dict] = []
    if targeted_jsonl.exists():
        with targeted_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    targeted_records.append(json.loads(line))
    targeted_samples = [_rollout_record_to_sample(r) for r in targeted_records]
    targeted_aligned = apply_context_mode_to_samples(
        targeted_samples,
        context_mode="original_failed",
        failed_hdf5=failed_hdf5,
        cem_report=cem_report,
    )
    for s in targeted_aligned:
        s["meta"] = {**s["meta"], "source": "v1f_targeted_repair_sampling", "dataset_split": "v1f_plus_balanced_v2"}

    all_samples = base_samples + targeted_aligned
    repairability = load_repairability_map(repairability_report)

    # 临时 bundle 供 build_balanced_indices
    import numpy as np

    bundle = {
        "success_flag": np.array([s["success_flag"] for s in all_samples], dtype=np.float32),
        "target_E_total": np.array([s["target_E_total"] for s in all_samples], dtype=np.float32),
        "original_E_total": np.array([s["context"]["original_E_total_norm"] for s in all_samples], dtype=np.float32),
        "source_failure_mode_idx": np.array(
            [
                ["success", "insertion_failed", "transport_failed", "grasp_failed", "lift_failed"].index(
                    s["context"]["source_failure_type"]
                )
                if s["context"]["source_failure_type"] in {"success", "insertion_failed", "transport_failed", "grasp_failed", "lift_failed"}
                else 2
                for s in all_samples
            ],
            dtype=np.int64,
        ),
    }
    meta_records = [s["meta"] for s in all_samples]

    # v2: 更高 success 过采样
    import build_v1f_plus_balanced_dataset as bal_mod

    old_success = bal_mod.SUCCESS_OVERSAMPLE
    old_target = bal_mod.TARGET_PER_FAILURE_TYPE
    bal_mod.SUCCESS_OVERSAMPLE = 8
    bal_mod.TARGET_PER_FAILURE_TYPE = 750
    try:
        selected, stats, ranking_eligible, sample_weight, demo_group_id = build_balanced_indices(
            bundle, meta_records, repairability, seed=seed
        )
    finally:
        bal_mod.SUCCESS_OVERSAMPLE = old_success
        bal_mod.TARGET_PER_FAILURE_TYPE = old_target

    # 重建 selected sample list
    merged_samples = [all_samples[i] for i in selected]
    for s in merged_samples:
        s["meta"] = {**s["meta"], "dataset_split": "v1f_plus_balanced_v2"}

    summary = {
        "dataset_version": "V1-F-aligned-plus-balanced-v2",
        "num_source_samples": len(all_samples),
        "num_targeted_added": len(targeted_aligned),
        "num_balanced_samples": len(merged_samples),
        "balance_stats": stats,
    }

    # save v2 npz
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "repair_parameter_dataset_v1f_plus_balanced_v2.npz"
    summary["meta_records"] = [s["meta"] for s in merged_samples]
    demo_idx = []
    for s in merged_samples:
        dk = str(s["context"]["source_demo"])
        demo_idx.append(KNOWN_DEMO_KEYS.index(dk) if dk in KNOWN_DEMO_KEYS else -1)

    np.savez_compressed(
        out_path,
        features=np.stack([s["features"] for s in merged_samples], axis=0),
        theta=np.stack([s["theta"] for s in merged_samples], axis=0),
        param_mask=np.stack([s["param_mask"] for s in merged_samples], axis=0),
        targets_components=np.stack([s["targets_components"] for s in merged_samples], axis=0),
        lift_residuals=np.stack([s["lift_residuals"] for s in merged_samples], axis=0),
        target_E_total=np.array([s["target_E_total"] for s in merged_samples], dtype=np.float32),
        success_flag=np.array([s["success_flag"] for s in merged_samples], dtype=np.float32),
        failure_type_idx=np.array([s["failure_type_idx"] for s in merged_samples], dtype=np.int64),
        outcome_idx=np.array([s["outcome_idx"] for s in merged_samples], dtype=np.int64),
        grasp_success_flag=np.array([s["grasp_success_flag"] for s in merged_samples], dtype=np.float32),
        lift_success_flag=np.array([s["lift_success_flag"] for s in merged_samples], dtype=np.float32),
        refined_success_flag=np.array([s["refined_success_flag"] for s in merged_samples], dtype=np.float32),
        original_E_total=np.array([s["context"]["original_E_total_norm"] for s in merged_samples], dtype=np.float32),
        source_failure_mode_idx=np.array(
            [
                FAILURE_MODES.index(s["context"]["source_failure_type"])
                if s["context"]["source_failure_type"] in FAILURE_MODES
                else 2
                for s in merged_samples
            ],
            dtype=np.int64,
        ),
        demo_idx=np.array(demo_idx, dtype=np.int64),
        sample_weight=sample_weight[selected],
        ranking_supervision_eligible=ranking_eligible[selected],
        demo_group_id=demo_group_id[selected],
        meta_json=json.dumps(summary),
    )

    manifest = {
        "dataset_version": "V1-F-aligned-plus-balanced-v2",
        "base_plus_rollout": str(plus_rollout),
        "targeted_jsonl": str(targeted_jsonl),
        "output_npz": str(out_path),
        "num_samples": len(merged_samples),
        "balance_stats": stats,
    }
    (output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-F-plus-balanced-v2 dataset")
    parser.add_argument("--targeted-jsonl", type=Path, default=DEFAULT_TARGETED)
    parser.add_argument("--repairability-report", type=Path, default=DEFAULT_REPAIRABILITY)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path, manifest = build_v2_dataset(
        balanced_npz=DEFAULT_BALANCED_NPZ,
        targeted_jsonl=args.targeted_jsonl,
        repairability_report=args.repairability_report,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
