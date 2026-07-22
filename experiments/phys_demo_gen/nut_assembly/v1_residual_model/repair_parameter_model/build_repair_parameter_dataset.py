#!/usr/bin/env python3
"""V1-E：从真实 sim rollout 构建 repair-parameter 数据集。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_V1E_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1E_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from repair_dataset import (  # noqa: E402
    ALL_THETA_KEYS,
    CONTEXT_NUMERIC_KEYS,
    INPUT_FEATURE_NAMES,
    TARGET_COMPONENT_NAMES,
    TARGET_ROLLOUT_KEYS,
    build_param_mask,
    build_theta_vector,
    extract_failed_context,
    extract_rollout_targets,
    infer_coarse_failure_mode,
    make_sample,
    parse_grasp_lift_params,
    parse_insertion_params,
    parse_transport_params,
)

DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _add_sample(
    samples: list[dict[str, Any]],
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    active: str,
    source: str,
    demo_key: str,
    insertion: dict[str, float] | None = None,
    transport: dict[str, float] | None = None,
    grasp_lift: dict[str, float] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    targets = extract_rollout_targets(row)
    if "E_smooth_norm" in row:
        targets["rollout_E_smooth_norm"] = float(row["E_smooth_norm"])
    theta = build_theta_vector(insertion=insertion, transport=transport, grasp_lift=grasp_lift)
    param_mask = build_param_mask(active=active)
    meta = {
        "source": source,
        "demo_key": demo_key,
        "active_param_group": active,
        "source_failure_type": context["source_failure_type"],
        **(extra_meta or {}),
    }
    samples.append(make_sample(context=context, theta=theta, param_mask=param_mask, targets=targets, meta=meta))


def _collect_insertion_samples(outputs: Path, failed_hdf5: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    demo_key = "demo_4"

    sim_path = outputs / "sim_in_loop_refinement" / "sim_in_loop_refinement_report.json"
    if sim_path.exists():
        report = json.loads(sim_path.read_text(encoding="utf-8"))
        original = report.get("original_waypoint_rollout") or {}
        context = extract_failed_context(original, demo_key=demo_key)
        for row in report.get("top_10_candidates", []):
            _add_sample(
                samples,
                row=row,
                context=context,
                active="insertion",
                source="sim_in_loop_refinement_top10",
                demo_key=demo_key,
                insertion=parse_insertion_params(row),
            )

    repeat_path = outputs / "sim_in_loop_repeatability" / "repeatability_report.json"
    if repeat_path.exists():
        report = json.loads(repeat_path.read_text(encoding="utf-8"))
        ref = (report.get("reference_rollouts") or {}).get("original_waypoint") or {}
        if not ref:
            ref = (report.get("runs") or [{}])[0]
        context = extract_failed_context(ref, demo_key=demo_key)
        for row in report.get("runs", []):
            payload = dict(row)
            if payload.get("best_params") and not payload.get("sim_params"):
                payload["sim_params"] = json.loads(payload["best_params"])
            _add_sample(
                samples,
                row=payload,
                context=context,
                active="insertion",
                source="sim_repeatability",
                demo_key=demo_key,
                insertion=parse_insertion_params(payload),
                extra_meta={"seed": payload.get("seed"), "max_evals": payload.get("max_evals")},
            )

    ablation_path = outputs / "sim_in_loop_ablation" / "ablation_report.json"
    if ablation_path.exists():
        report = json.loads(ablation_path.read_text(encoding="utf-8"))
        ref = report.get("reference_original_waypoint") or report.get("method_results", {}).get(
            "A_original_waypoint", {}
        )
        context = extract_failed_context(ref, demo_key=demo_key)
        for method_id, row in (report.get("method_results") or {}).items():
            cfg = (report.get("methods") or {}).get(method_id, {})
            if not cfg.get("needs_search"):
                continue
            payload = dict(row)
            payload["method_id"] = method_id
            _add_sample(
                samples,
                row=payload,
                context=context,
                active="insertion",
                source=f"sim_ablation_{method_id}",
                demo_key=demo_key,
                insertion=parse_insertion_params(payload),
                extra_meta={"method_id": method_id, "scoring_mode": cfg.get("scoring_mode")},
            )

    return samples


def _collect_transport_samples(outputs: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    report_path = outputs / "transport_refinement" / "transport_refinement_report.json"
    if not report_path.exists():
        return samples
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_demo = report.get("per_demo", {})
    for demo_key, demo_result in per_demo.items():
        original = demo_result.get("original_waypoint_rollout") or {}
        context = extract_failed_context(
            original,
            demo_key=demo_key,
            failure_type=infer_coarse_failure_mode(demo_key=demo_key),
        )
        for row in demo_result.get("top_10_candidates", []):
            _add_sample(
                samples,
                row=row,
                context=context,
                active="transport",
                source="transport_refinement_top10",
                demo_key=demo_key,
                transport=parse_transport_params(row),
            )
        best = demo_result.get("best_transport_refined")
        if best:
            _add_sample(
                samples,
                row=best,
                context=context,
                active="transport",
                source="transport_refinement_best",
                demo_key=demo_key,
                transport=parse_transport_params(best),
            )

    for row in _load_csv(outputs / "transport_refinement" / "top_candidates.csv"):
        demo_key = row["demo_key"]
        original = per_demo.get(demo_key, {}).get("original_waypoint_rollout", {})
        if not original:
            continue
        context = extract_failed_context(original, demo_key=demo_key)
        payload = dict(row)
        payload["success_flag"] = payload.get("success_flag") in ("True", "true", "1", True)
        _add_sample(
            samples,
            row=payload,
            context=context,
            active="transport",
            source="transport_top_candidates_csv",
            demo_key=demo_key,
            transport=parse_transport_params(payload),
            extra_meta={"rank": payload.get("rank")},
        )
    return samples


def _collect_grasp_lift_samples(outputs: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    report_path = outputs / "grasp_refinement" / "grasp_refinement_report.json"
    if not report_path.exists():
        return samples
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_demo = report.get("per_demo", {})
    for demo_key, demo_result in per_demo.items():
        original = demo_result.get("original_waypoint_rollout") or {}
        coarse = "transport_failed" if demo_key == "demo_3" else "grasp_failed"
        context = extract_failed_context(original, demo_key=demo_key, failure_type=coarse)
        active = "lift" if demo_key == "demo_3" else "grasp"
        for row in demo_result.get("top_10_candidates", []):
            _add_sample(
                samples,
                row=row,
                context=context,
                active=active,
                source="grasp_refinement_top10",
                demo_key=demo_key,
                grasp_lift=parse_grasp_lift_params(row),
            )
        best = demo_result.get("best_grasp_refined")
        if best:
            _add_sample(
                samples,
                row=best,
                context=context,
                active=active,
                source="grasp_refinement_best",
                demo_key=demo_key,
                grasp_lift=parse_grasp_lift_params(best),
            )

    for row in _load_csv(outputs / "grasp_refinement" / "top_candidates.csv"):
        demo_key = row["demo_key"]
        original = per_demo.get(demo_key, {}).get("original_waypoint_rollout", {})
        if not original:
            continue
        coarse = "transport_failed" if demo_key == "demo_3" else "grasp_failed"
        context = extract_failed_context(original, demo_key=demo_key, failure_type=coarse)
        active = "lift" if demo_key == "demo_3" else "grasp"
        payload = dict(row)
        payload["success_flag"] = payload.get("success_flag") in ("True", "true", "1", True)
        _add_sample(
            samples,
            row=payload,
            context=context,
            active=active,
            source="grasp_top_candidates_csv",
            demo_key=demo_key,
            grasp_lift=parse_grasp_lift_params(payload),
            extra_meta={"rank": payload.get("rank")},
        )
    return samples


def _collect_lift_refinement_samples(outputs: Path) -> list[dict[str, Any]]:
    report_path = outputs / "lift_refinement" / "lift_refinement_report.json"
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    per_demo = report.get("per_demo", {})
    for demo_key, demo_result in per_demo.items():
        original = demo_result.get("original_waypoint_rollout") or {}
        context = extract_failed_context(original, demo_key=demo_key, failure_type="lift_failed")
        for row in demo_result.get("top_10_candidates", []):
            _add_sample(
                samples,
                row=row,
                context=context,
                active="lift",
                source="lift_refinement_top10",
                demo_key=demo_key,
                grasp_lift=parse_grasp_lift_params(row),
            )
    return samples


def build_dataset(experiment_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = experiment_dir / "outputs"
    samples: list[dict[str, Any]] = []
    samples.extend(_collect_insertion_samples(outputs, DEFAULT_FAILED_HDF5))
    samples.extend(_collect_transport_samples(outputs))
    samples.extend(_collect_grasp_lift_samples(outputs))
    samples.extend(_collect_lift_refinement_samples(outputs))

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
        "dataset_version": "V1-E_repair_parameter_residual_field",
        "num_samples": len(samples),
        "input_dim": len(INPUT_FEATURE_NAMES),
        "input_feature_names": INPUT_FEATURE_NAMES,
        "theta_keys": ALL_THETA_KEYS,
        "context_numeric_keys": CONTEXT_NUMERIC_KEYS,
        "target_component_names": TARGET_COMPONENT_NAMES,
        "target_rollout_keys": TARGET_ROLLOUT_KEYS,
        "source_counts": source_counts,
        "demo_counts": demo_counts,
        "failure_type_counts": failure_counts,
        "notes": [
            "V1-E: failed demo context + repair theta + param_mask -> rollout residual field.",
            "Explicit energy is supervision/baseline only; PINN is primary repair selector.",
            "Not PINA; not a cross-task generalization model.",
        ],
    }
    return samples, summary


def save_dataset(samples: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["meta_records"] = [s["meta"] for s in samples]
    npz_path = output_dir / "repair_parameter_dataset.npz"
    np.savez_compressed(
        npz_path,
        features=np.stack([s["features"] for s in samples], axis=0),
        theta=np.stack([s["theta"] for s in samples], axis=0),
        param_mask=np.stack([s["param_mask"] for s in samples], axis=0),
        targets_components=np.stack([s["targets_components"] for s in samples], axis=0),
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
        meta_json=json.dumps(summary),
    )

    jsonl_path = output_dir / "repair_parameter_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "context": sample["context"],
                "theta": {k: float(sample["theta"][i]) for i, k in enumerate(ALL_THETA_KEYS)},
                "param_mask": {k: float(sample["param_mask"][i]) for i, k in enumerate(ALL_THETA_KEYS)},
                "targets": {
                    **{TARGET_COMPONENT_NAMES[i]: float(sample["targets_components"][i]) for i in range(5)},
                    "rollout_E_total_norm": float(sample["target_E_total"]),
                    "rollout_success_flag": bool(sample["success_flag"] > 0.5),
                    "rollout_grasp_success_proxy": bool(sample["grasp_success_flag"] > 0.5),
                    "rollout_lift_success_proxy": bool(sample["lift_success_flag"] > 0.5),
                    "refined_success_flag": bool(sample["refined_success_flag"] > 0.5),
                },
                "meta": sample["meta"],
            }
            handle.write(json.dumps(record) + "\n")

    (output_dir / "repair_parameter_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return npz_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-E repair-parameter dataset")
    parser.add_argument("--experiment-dir", type=Path, default=_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    samples, summary = build_dataset(args.experiment_dir)
    if not samples:
        raise SystemExit("No repair-parameter samples collected; run V2-B sim refinements first.")
    path = save_dataset(samples, summary, args.output_dir)
    print(json.dumps({"output": str(path), "num_samples": summary["num_samples"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
