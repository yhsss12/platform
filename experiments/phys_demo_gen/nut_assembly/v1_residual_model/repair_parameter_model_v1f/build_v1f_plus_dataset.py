#!/usr/bin/env python3
"""Task 4：合并旧 V1-F-aligned-original 数据集 + 新 rollout 样本。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v1f_aligned_dataset import apply_context_mode_to_samples  # noqa: E402
from build_v1f_repair_dataset import _rollout_record_to_sample  # noqa: E402
from v1f_plus_utils import (  # noqa: E402
    DEFAULT_ALIGNED_NPZ,
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_PLUS_OUTPUT,
)
from repair_dataset import GRASP_LIFT_PARAM_KEYS, INSERTION_PARAM_KEYS, TRANSPORT_PARAM_KEYS  # noqa: E402
from v1f_repair_dataset import (  # noqa: E402
    build_param_mask_v1f,
    build_theta_vector_v1f,
    extract_rollout_targets_v1f,
    make_sample_v1f,
)

DEFAULT_NEW_ROLLOUT = DEFAULT_PLUS_OUTPUT / "new_rollout_samples.jsonl"
KNOWN_DEMO_KEYS = ["demo_0", "demo_1", "demo_2", "demo_3", "demo_4"]
FAILURE_MODES = ["success", "insertion_failed", "transport_failed", "grasp_failed", "lift_failed"]


def _load_aligned_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            context = rec["context"]
            theta_dict = rec["theta"]
            theta = build_theta_vector_v1f(
                insertion={k: float(theta_dict[k]) for k in INSERTION_PARAM_KEYS},
                transport={k: float(theta_dict[k]) for k in TRANSPORT_PARAM_KEYS},
                grasp_lift={k: float(theta_dict[k]) for k in GRASP_LIFT_PARAM_KEYS},
            )
            active = rec["meta"].get("active_param_group", "transport")
            mask = build_param_mask_v1f(active=active)
            targets_raw = rec["targets"]
            targets = extract_rollout_targets_v1f(
                {
                    "success_flag": targets_raw.get("refined_success_flag", False),
                    "E_xy_norm": targets_raw.get("E_xy", 0.0),
                    "E_transport_norm": targets_raw.get("E_transport", 0.0),
                    "E_yaw_norm": targets_raw.get("E_yaw", 0.0),
                    "E_z_norm": targets_raw.get("E_z", 0.0),
                    "E_grasp_norm": targets_raw.get("E_grasp", 0.0),
                    "E_lift_norm": targets_raw.get("E_lift", 0.0),
                    "E_smooth_norm": targets_raw.get("E_smooth", 0.0),
                    "E_total_norm": targets_raw.get("rollout_E_total_norm", 0.0),
                }
            )
            sample = make_sample_v1f(
                context=context,
                theta=theta,
                param_mask=mask,
                targets=targets,
                meta={**rec["meta"], "source": rec["meta"].get("source", "aligned_original")},
            )
            samples.append(sample)
    return samples


def _save_plus_dataset(samples: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["meta_records"] = [s["meta"] for s in samples]
    npz_path = output_dir / "repair_parameter_dataset_v1f_plus.npz"

    demo_idx = []
    for s in samples:
        dk = s["context"]["source_demo"]
        demo_idx.append(KNOWN_DEMO_KEYS.index(dk) if dk in KNOWN_DEMO_KEYS else -1)

    np.savez_compressed(
        npz_path,
        features=np.stack([s["features"] for s in samples], axis=0),
        theta=np.stack([s["theta"] for s in samples], axis=0),
        param_mask=np.stack([s["param_mask"] for s in samples], axis=0),
        targets_components=np.stack([s["targets_components"] for s in samples], axis=0),
        lift_residuals=np.stack([s["lift_residuals"] for s in samples], axis=0),
        target_E_total=np.array([s["target_E_total"] for s in samples], dtype=np.float32),
        success_flag=np.array([s["success_flag"] for s in samples], dtype=np.float32),
        failure_type_idx=np.array([s["failure_type_idx"] for s in samples], dtype=np.int64),
        outcome_idx=np.array([s["outcome_idx"] for s in samples], dtype=np.int64),
        grasp_success_flag=np.array([s["grasp_success_flag"] for s in samples], dtype=np.float32),
        lift_success_flag=np.array([s["lift_success_flag"] for s in samples], dtype=np.float32),
        refined_success_flag=np.array([s["refined_success_flag"] for s in samples], dtype=np.float32),
        original_E_total=np.array(
            [s["context"]["original_E_total_norm"] for s in samples], dtype=np.float32
        ),
        source_failure_mode_idx=np.array(
            [
                FAILURE_MODES.index(s["context"]["source_failure_type"])
                if s["context"]["source_failure_type"] in FAILURE_MODES
                else 2
                for s in samples
            ],
            dtype=np.int64,
        ),
        demo_idx=np.array(demo_idx, dtype=np.int64),
        meta_json=json.dumps(summary),
    )
    return npz_path


def build_plus_dataset(
    *,
    aligned_jsonl: Path,
    new_rollout_jsonl: Path,
    output_dir: Path,
    failed_hdf5: Path,
    cem_report: Path,
) -> tuple[Path, dict[str, Any]]:
    base_samples = _load_aligned_jsonl(aligned_jsonl)

    new_records: list[dict[str, Any]] = []
    with new_rollout_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                new_records.append(json.loads(line))

    new_samples = [_rollout_record_to_sample(rec) for rec in new_records]
    aligned_new = apply_context_mode_to_samples(
        new_samples,
        context_mode="original_failed",
        failed_hdf5=failed_hdf5,
        cem_report=cem_report,
    )

    merged = base_samples + aligned_new
    for sample in merged:
        sample["meta"] = {**sample.get("meta", {}), "dataset_split": "v1f_plus"}

    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    for sample in merged:
        src = sample["meta"].get("source", "aligned_original")
        source_counts[src] = source_counts.get(src, 0) + 1
        dk = sample["meta"].get("demo_key", "unknown")
        demo_counts[dk] = demo_counts.get(dk, 0) + 1
        ft = sample.get("context", {}).get("source_failure_type", "unknown")
        failure_counts[ft] = failure_counts.get(ft, 0) + 1

    summary = {
        "dataset_version": "V1-F-aligned-plus",
        "context_mode": "original_failed",
        "num_samples": len(merged),
        "num_base_aligned_original": len(base_samples),
        "num_new_rollout": len(aligned_new),
        "source_counts": source_counts,
        "demo_counts": demo_counts,
        "failure_type_counts": failure_counts,
        "notes": [
            "Merged V1-F-aligned-original + new Square_D0 failed demo rollout samples.",
            "All new samples from MuJoCo rollout; object_poses not modified.",
        ],
    }

    npz_path = _save_plus_dataset(merged, summary, output_dir)
    manifest = {
        "dataset_version": "V1-F-aligned-plus",
        "base_aligned_jsonl": str(aligned_jsonl),
        "new_rollout_jsonl": str(new_rollout_jsonl),
        "output_npz": str(npz_path),
        "num_samples": len(merged),
        "num_base": len(base_samples),
        "num_new": len(aligned_new),
        "context_mode": "original_failed",
    }
    (output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return npz_path, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-F-aligned-plus merged dataset")
    parser.add_argument(
        "--aligned-jsonl",
        type=Path,
        default=DEFAULT_ALIGNED_NPZ.parent / "repair_parameter_dataset_v1f.jsonl",
    )
    parser.add_argument("--new-rollout-jsonl", type=Path, default=DEFAULT_NEW_ROLLOUT)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLUS_OUTPUT)
    args = parser.parse_args()

    npz_path, manifest = build_plus_dataset(
        aligned_jsonl=args.aligned_jsonl,
        new_rollout_jsonl=args.new_rollout_jsonl,
        output_dir=args.output_dir,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
