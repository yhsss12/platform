#!/usr/bin/env python3
"""V1-F：构建扩充后的 repair-parameter 数据集（V1-E 历史 + rollout 采样）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_V1E_DIR = _V1_DIR / "repair_parameter_model"
_EXPERIMENT_DIR = _V1_DIR.parent
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_repair_parameter_dataset import build_dataset as build_v1e_dataset  # noqa: E402
from repair_dataset import (  # noqa: E402
    parse_grasp_lift_params,
    parse_insertion_params,
    parse_transport_params,
)
from v1f_repair_dataset import (  # noqa: E402
    ALL_THETA_KEYS_V1F,
    INPUT_FEATURE_NAMES_V1F,
    LIFT_RESIDUAL_NAMES,
    V1F_COMPONENT_NAMES,
    build_param_mask_v1f,
    build_theta_vector_v1f,
    extract_rollout_targets_v1f,
    make_sample_v1f,
    parse_lift_extra_params,
)

DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model"
DEFAULT_ROLLOUT_JSONL = DEFAULT_OUTPUT / "rollout_samples.jsonl"
DEFAULT_V1E_DATASET = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "repair_parameter_dataset.npz"


def _v1e_sample_to_v1f(sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample["meta"]
    active = meta.get("active_param_group", "transport")
    from repair_dataset import ALL_THETA_KEYS as V1E_KEYS

    v1e_theta = sample["theta"]
    insertion = {V1E_KEYS[i]: float(v1e_theta[i]) for i in range(7)}
    transport = {V1E_KEYS[i]: float(v1e_theta[i]) for i in range(7, 15)}
    grasp_lift = {V1E_KEYS[i]: float(v1e_theta[i]) for i in range(15, 24)}
    theta = build_theta_vector_v1f(insertion=insertion, transport=transport, grasp_lift=grasp_lift)
    mask = build_param_mask_v1f(active=active)

    row = {
        "success_flag": sample["success_flag"] > 0.5,
        "E_total_norm": sample["target_E_total"],
        "E_xy_norm": float(sample["targets_components"][0]),
        "E_transport_norm": float(sample["targets_components"][1]),
        "E_yaw_norm": float(sample["targets_components"][2]),
        "E_z_norm": float(sample["targets_components"][3]),
        "E_smooth_norm": float(sample["targets_components"][4]),
        "grasp_success_proxy": sample["grasp_success_flag"] > 0.5,
        "lift_success_proxy": sample["lift_success_flag"] > 0.5,
    }
    targets = extract_rollout_targets_v1f(row)
    return make_sample_v1f(
        context=sample["context"],
        theta=theta,
        param_mask=mask,
        targets=targets,
        meta={**meta, "source": f"v1e_import:{meta.get('source', '')}"},
    )


def _rollout_record_to_sample(record: dict[str, Any]) -> dict[str, Any]:
    rollout = record["rollout"]
    active = record["active"]
    demo_key = record["demo_key"]
    context = record["context"]

    insertion = parse_insertion_params(rollout) if active == "insertion" else None
    transport = parse_transport_params(rollout) if active == "transport" else None
    grasp_lift = parse_grasp_lift_params(rollout) if active in ("grasp", "lift") else None
    lift_extra = parse_lift_extra_params(rollout) if active == "lift" else None

    theta = build_theta_vector_v1f(
        insertion=insertion,
        transport=transport,
        grasp_lift=grasp_lift,
        lift_extra=lift_extra,
    )
    mask = build_param_mask_v1f(active=active)
    targets = extract_rollout_targets_v1f(rollout)
    meta = {
        "source": record.get("source", "v1f_rollout_sampling"),
        "demo_key": demo_key,
        "active_param_group": active,
        "source_failure_type": context["source_failure_type"],
        "sampling_index": rollout.get("sampling_index"),
        "sampling_seed": rollout.get("sampling_seed"),
    }
    return make_sample_v1f(context=context, theta=theta, param_mask=mask, targets=targets, meta=meta)


def build_v1f_dataset(
    experiment_dir: Path,
    rollout_jsonl: Path | None,
    include_v1e: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    if include_v1e:
        v1e_samples, _ = build_v1e_dataset(experiment_dir)
        for s in v1e_samples:
            samples.append(_v1e_sample_to_v1f(s))

    if rollout_jsonl and rollout_jsonl.exists():
        with rollout_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    samples.append(_rollout_record_to_sample(json.loads(line)))

    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    for sample in samples:
        src = sample["meta"]["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
        dk = sample["meta"]["demo_key"]
        demo_counts[dk] = demo_counts.get(dk, 0) + 1
        ft = sample["context"]["source_failure_type"]
        failure_counts[ft] = failure_counts.get(ft, 0) + 1

    summary = {
        "dataset_version": "V1-F_uncertainty_aware_repair_parameter_field",
        "num_samples": len(samples),
        "input_dim": len(INPUT_FEATURE_NAMES_V1F),
        "input_feature_names": INPUT_FEATURE_NAMES_V1F,
        "theta_keys": ALL_THETA_KEYS_V1F,
        "component_names": V1F_COMPONENT_NAMES,
        "lift_residual_names": LIFT_RESIDUAL_NAMES,
        "source_counts": source_counts,
        "demo_counts": demo_counts,
        "failure_type_counts": failure_counts,
        "notes": [
            "V1-F: failed context + repair theta + failure embedding + mask -> residual field + uncertainty.",
            "All rollout samples from MuJoCo/RoboSuite; object_poses not modified.",
            "Lift-aware params and residuals for demo_3 lift_failed.",
        ],
    }
    return samples, summary


def save_v1f_dataset(samples: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["meta_records"] = [s["meta"] for s in samples]
    npz_path = output_dir / "repair_parameter_dataset_v1f.npz"

    demo_idx = []
    for s in samples:
        dk = s["context"]["source_demo"]
        demo_idx.append(["demo_0", "demo_1", "demo_2", "demo_3", "demo_4"].index(dk))

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
                ["success", "insertion_failed", "transport_failed", "grasp_failed", "lift_failed"].index(
                    s["context"]["source_failure_type"]
                )
                if s["context"]["source_failure_type"]
                in {"success", "insertion_failed", "transport_failed", "grasp_failed", "lift_failed"}
                else 2
                for s in samples
            ],
            dtype=np.int64,
        ),
        demo_idx=np.array(demo_idx, dtype=np.int64),
        meta_json=json.dumps(summary),
    )

    jsonl_path = output_dir / "repair_parameter_dataset_v1f.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "context": sample["context"],
                "theta": {k: float(sample["theta"][i]) for i, k in enumerate(ALL_THETA_KEYS_V1F)},
                "targets": {
                    **{V1F_COMPONENT_NAMES[i]: float(sample["targets_components"][i]) for i in range(7)},
                    **{LIFT_RESIDUAL_NAMES[i]: float(sample["lift_residuals"][i]) for i in range(5)},
                    "rollout_E_total_norm": float(sample["target_E_total"]),
                    "refined_success_flag": bool(sample["refined_success_flag"] > 0.5),
                },
                "meta": sample["meta"],
            }
            handle.write(json.dumps(record) + "\n")

    (output_dir / "repair_parameter_dataset_v1f_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return npz_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-F repair-parameter dataset")
    parser.add_argument("--experiment-dir", type=Path, default=_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rollout-jsonl", type=Path, default=DEFAULT_ROLLOUT_JSONL)
    parser.add_argument("--no-v1e", action="store_true")
    args = parser.parse_args()

    samples, summary = build_v1f_dataset(
        args.experiment_dir,
        args.rollout_jsonl if args.rollout_jsonl.exists() else None,
        include_v1e=not args.no_v1e,
    )
    if not samples:
        raise SystemExit("No V1-F samples collected.")
    path = save_v1f_dataset(samples, summary, args.output_dir)
    print(json.dumps({"output": str(path), "num_samples": summary["num_samples"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
