#!/usr/bin/env python3
"""V1-F-aligned repair-parameter 数据集：支持 original_failed / cem_refined / dual_context。"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_V1E_DIR = _V1_DIR / "repair_parameter_model"
_EXPERIMENT_DIR = _V1_DIR.parent
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1E_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v1f_repair_dataset import build_v1f_dataset, save_v1f_dataset  # noqa: E402
from osc_action_converter import SimLoopParams  # noqa: E402
from repair_dataset import extract_failed_context, infer_coarse_failure_mode  # noqa: E402
from sim_in_loop_refiner import load_best_theta, run_refined_waypoint_rollout  # noqa: E402
from v1f_repair_dataset import build_input_vector_v1f  # noqa: E402

DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_repair_parameter_model"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_ROLLOUT_JSONL = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "rollout_samples.jsonl"

CONTEXT_MODES = ("original_failed", "cem_refined", "dual_context")

ACTIVE_TO_SEARCH_KIND = {
    "insertion": "insertion",
    "transport": "transport",
    "grasp": "grasp",
    "lift": "lift",
}


@lru_cache(maxsize=32)
def _resolve_context_key(
    context_mode: str,
    failed_hdf5: str,
    cem_report: str,
    demo_key: str,
    active: str,
) -> str:
    """Cache key for context lookup."""
    return f"{context_mode}|{failed_hdf5}|{cem_report}|{demo_key}|{active}"


_CONTEXT_CACHE: dict[str, dict[str, Any]] = {}


def resolve_context_for_mode(
    *,
    context_mode: str,
    failed_hdf5: Path,
    cem_report: Path,
    demo_key: str,
    active: str,
) -> dict[str, Any]:
    if context_mode not in ("original_failed", "cem_refined"):
        raise ValueError(f"resolve_context_for_mode expects original_failed or cem_refined, got {context_mode}")

    cache_key = _resolve_context_key(
        context_mode, str(failed_hdf5), str(cem_report), demo_key, active
    )
    if cache_key in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[cache_key]

    failure_type = infer_coarse_failure_mode(demo_key=demo_key)
    search_kind = ACTIVE_TO_SEARCH_KIND.get(active, active)

    if context_mode == "original_failed":
        from repair_common_v1f import extract_baseline_context_v1f

        context = extract_baseline_context_v1f(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            failure_type=failure_type,
            search_kind=search_kind,
        )
    else:
        if search_kind == "insertion":
            theta = load_best_theta(str(cem_report), demo_key)
            baseline = run_refined_waypoint_rollout(
                str(failed_hdf5),
                demo_key,
                "failed",
                theta,
                sim_params=SimLoopParams(),
            )
            context = extract_failed_context(baseline, demo_key=demo_key, failure_type=failure_type)
        elif search_kind == "transport":
            from transport_sim_search import execute_transport_rollout
            from transport_waypoint_builder import TransportSearchParams

            theta = load_best_theta(str(cem_report), demo_key)
            baseline = execute_transport_rollout(
                str(failed_hdf5),
                demo_key,
                "failed",
                theta,
                TransportSearchParams(),
                rollout_kind="v1f_aligned_cem_baseline",
            )
            context = extract_failed_context(baseline, demo_key=demo_key, failure_type=failure_type)
        else:
            from repair_common_v1f import extract_baseline_context_v1f

            context = extract_baseline_context_v1f(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                failure_type=failure_type,
                search_kind=search_kind,
            )

    _CONTEXT_CACHE[cache_key] = context
    return context


def _rebuild_sample_with_context(sample: dict[str, Any], context: dict[str, Any], context_mode: str) -> dict[str, Any]:
    rebuilt = dict(sample)
    rebuilt["context"] = context
    rebuilt["features"] = build_input_vector_v1f(context, sample["theta"], sample["param_mask"])
    rebuilt["meta"] = {**sample["meta"], "context_mode": context_mode}
    return rebuilt


def apply_context_mode_to_samples(
    samples: list[dict[str, Any]],
    *,
    context_mode: str,
    failed_hdf5: Path,
    cem_report: Path,
) -> list[dict[str, Any]]:
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context_mode: {context_mode}")

    modes = ["original_failed", "cem_refined"] if context_mode == "dual_context" else [context_mode]
    out: list[dict[str, Any]] = []

    for sample in samples:
        demo_key = sample["meta"]["demo_key"]
        active = sample["meta"].get("active_param_group", "transport")
        for mode in modes:
            context = resolve_context_for_mode(
                context_mode=mode,
                failed_hdf5=failed_hdf5,
                cem_report=cem_report,
                demo_key=demo_key,
                active=active,
            )
            out.append(_rebuild_sample_with_context(sample, context, mode))

    return out


def build_aligned_dataset(
    *,
    experiment_dir: Path,
    rollout_jsonl: Path | None,
    include_v1e: bool,
    context_mode: str,
    failed_hdf5: Path,
    cem_report: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_samples, summary = build_v1f_dataset(
        experiment_dir,
        rollout_jsonl,
        include_v1e=include_v1e,
    )
    aligned = apply_context_mode_to_samples(
        base_samples,
        context_mode=context_mode,
        failed_hdf5=failed_hdf5,
        cem_report=cem_report,
    )

    context_mode_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    for sample in aligned:
        mode = sample["meta"].get("context_mode", context_mode)
        context_mode_counts[mode] = context_mode_counts.get(mode, 0) + 1
        dk = sample["meta"]["demo_key"]
        demo_counts[dk] = demo_counts.get(dk, 0) + 1

    summary.update(
        {
            "dataset_version": f"V1-F-aligned_{context_mode}",
            "context_mode": context_mode,
            "num_samples": len(aligned),
            "num_base_samples": len(base_samples),
            "context_mode_counts": context_mode_counts,
            "demo_counts": demo_counts,
            "notes": summary.get("notes", [])
            + [
                f"context_mode={context_mode}: PINN input context aligned for offline repair inference.",
                "original_failed: matches offline repair test default context_source.",
                "dual_context: duplicates each sample with both original_failed and cem_refined contexts.",
            ],
        }
    )
    return aligned, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1-F-aligned repair-parameter dataset")
    parser.add_argument("--experiment-dir", type=Path, default=_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rollout-jsonl", type=Path, default=DEFAULT_ROLLOUT_JSONL)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--context-mode", choices=CONTEXT_MODES, default="original_failed")
    parser.add_argument("--no-v1e", action="store_true")
    args = parser.parse_args()

    rollout = args.rollout_jsonl if args.rollout_jsonl.exists() else None
    samples, summary = build_aligned_dataset(
        experiment_dir=args.experiment_dir,
        rollout_jsonl=rollout,
        include_v1e=not args.no_v1e,
        context_mode=args.context_mode,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
    )
    if not samples:
        raise SystemExit("No aligned samples collected.")

    out_dir = args.output_dir / args.context_mode
    path = save_v1f_dataset(samples, summary, out_dir)
    manifest = {
        "context_mode": args.context_mode,
        "output_npz": str(path),
        "num_samples": summary["num_samples"],
        "recommended_for_offline_repair": args.context_mode in ("original_failed", "dual_context"),
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
