#!/usr/bin/env python3
"""Convert official MimicGen coffee_preparation datagen outputs to PhyGen feedback jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phygen.adapters.mimicgen.coffee_repair import (  # noqa: E402
    _action_delta,
    _pick_failure_frame,
    compute_context_metrics,
    demo_sort_key,
    load_demo_obs_and_actions,
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_theta_from_config(mg_config: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    exp_gen = mg_config["experiment"]["generation"]
    subtasks = mg_config["task"]["task_spec"]
    first = subtasks[sorted(subtasks.keys(), key=lambda k: int(k.split("_")[-1]))[0]]
    strategy = str(first.get("selection_strategy", "random"))
    offset = first.get("subtask_term_offset_range") or [5, 10]
    kwargs = first.get("selection_strategy_kwargs") or {}
    return {
        "candidate_index": int(candidate_index),
        "selection_strategy": strategy,
        "select_src_per_subtask": bool(exp_gen.get("select_src_per_subtask", False)),
        "transform_first_robot_pose": bool(exp_gen.get("transform_first_robot_pose", False)),
        "interpolate_from_last_target_pose": bool(exp_gen.get("interpolate_from_last_target_pose", True)),
        "action_noise": float(first.get("action_noise", 0.05)),
        "num_interpolation_steps": int(first.get("num_interpolation_steps", 5)),
        "num_fixed_steps": int(first.get("num_fixed_steps", 0)),
        "offset_range": [int(offset[0]), int(offset[1])],
        "nn_k": int(kwargs.get("nn_k", 3)) if kwargs else 3,
    }


def _build_theta_from_theta_json(theta_json: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    offset = theta_json.get("offset_range", [5, 10])
    return {
        "candidate_index": int(candidate_index),
        "selection_strategy": str(theta_json.get("selection_strategy", "random")),
        "select_src_per_subtask": bool(theta_json.get("select_src_per_subtask", False)),
        "transform_first_robot_pose": bool(theta_json.get("transform_first_robot_pose", False)),
        "interpolate_from_last_target_pose": bool(theta_json.get("interpolate_from_last_target_pose", True)),
        "action_noise": float(theta_json.get("action_noise", 0.05)),
        "num_interpolation_steps": int(theta_json.get("num_interpolation_steps", 5)),
        "num_fixed_steps": int(theta_json.get("num_fixed_steps", 0)),
        "offset_range": [int(offset[0]), int(offset[1])],
        "nn_k": int(theta_json.get("nn_k", 3)),
    }


def _source_demo_key(source_demo_keys: list[str], src_demo_inds: np.ndarray) -> str:
    idx = int(src_demo_inds[0]) if len(src_demo_inds) else 0
    idx = int(np.clip(idx, 0, len(source_demo_keys) - 1))
    return source_demo_keys[idx]


def _context_for_source(
    prepared_source: Path,
    source_demo_key: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if source_demo_key not in cache:
        obs, actions = load_demo_obs_and_actions(prepared_source, source_demo_key)
        fail_idx = _pick_failure_frame(obs, actions)
        cache[source_demo_key] = {
            "failure_frame": fail_idx,
            "context_metrics": compute_context_metrics(
                obs[fail_idx],
                action_delta=_action_delta(actions, fail_idx),
            ),
        }
    return cache[source_demo_key]


def _metrics_from_generated_demo(obs: np.ndarray, actions: np.ndarray | None) -> dict[str, float]:
    idx = len(obs) - 1
    return compute_context_metrics(obs[idx], action_delta=_action_delta(actions, idx))


def _resolve_theta_and_config(run_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    theta_path = run_dir / "theta.json"
    config_path = run_dir / "config.json"
    datagen_dir = run_dir / "demo"
    mg_config_path = datagen_dir / "mg_config.json"

    if theta_path.is_file():
        return _load_json(theta_path), config_path if config_path.is_file() else mg_config_path
    if mg_config_path.is_file():
        return None, mg_config_path
    if config_path.is_file():
        return None, config_path
    return None, None


def convert_run_dir(
    *,
    run_dir: Path,
    prepared_source: Path,
    sweep_id: str,
    repair_eval_mode: str,
    context_cache: dict[str, dict[str, Any]],
    source_demo_keys: list[str],
    candidate_index_start: int,
    max_success: int | None = None,
    max_failure: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    datagen_dir = run_dir / "demo"
    success_hdf5 = datagen_dir / "demo.hdf5"
    failed_hdf5 = datagen_dir / "demo_failed.hdf5"
    theta_json, config_path = _resolve_theta_and_config(run_dir)
    if config_path is None or not config_path.is_file():
        return [], candidate_index_start

    mg_config = _load_json(config_path)
    records: list[dict[str, Any]] = []
    candidate_index = candidate_index_start

    bundles = [
        (success_hdf5, True, max_success),
        (failed_hdf5, False, max_failure),
    ]
    for hdf5_path, success_label, limit in bundles:
        if not hdf5_path.is_file():
            continue
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=demo_sort_key)
            if limit is not None:
                demo_keys = demo_keys[:limit]
            for gen_demo_key in demo_keys:
                demo = f["data"][gen_demo_key]
                obs = np.asarray(demo["obs"]["object"], dtype=np.float64)
                actions = np.asarray(demo["actions"], dtype=np.float64)
                src_inds = np.asarray(demo["src_demo_inds"], dtype=np.int64)
                src_key = _source_demo_key(source_demo_keys, src_inds)
                ctx = _context_for_source(prepared_source, src_key, context_cache)
                if theta_json is not None:
                    theta = _build_theta_from_theta_json(theta_json, candidate_index)
                else:
                    theta = _build_theta_from_config(mg_config, candidate_index)
                metrics = _metrics_from_generated_demo(obs, actions)
                records.append(
                    {
                        "task_name": "coffee_preparation",
                        "source_demo_key": src_key,
                        "demo_key": src_key,
                        "generated_demo_key": gen_demo_key,
                        "sweep_id": sweep_id,
                        "candidate_index": candidate_index,
                        "failure_frame": int(ctx["failure_frame"]),
                        "context_metrics": dict(ctx["context_metrics"]),
                        "theta": theta,
                        "metrics": metrics,
                        "success": bool(success_label),
                        "source_hdf5": str(prepared_source),
                        "generated_hdf5": str(hdf5_path),
                        "src_demo_inds": src_inds.tolist(),
                        "repair_eval_mode": repair_eval_mode,
                    }
                )
                candidate_index += 1
    return records, candidate_index


def convert_official_datagen(
    *,
    success_hdf5: Path,
    failed_hdf5: Path,
    prepared_source: Path,
    mg_config_path: Path,
    out_path: Path,
    max_success: int | None = None,
    max_failure: int | None = None,
    repair_eval_mode: str = "official_mimicgen_datagen_or_true_rollout",
) -> dict[str, Any]:
    run_dir = mg_config_path.parent
    records, _ = convert_run_dir(
        run_dir=run_dir,
        prepared_source=prepared_source,
        sweep_id=run_dir.name,
        repair_eval_mode=repair_eval_mode,
        context_cache={},
        source_demo_keys=_list_source_demo_keys(prepared_source),
        candidate_index_start=0,
        max_success=max_success,
        max_failure=max_failure,
    )
    return _write_records(records, out_path)


def convert_sweep_root(
    *,
    sweep_root: Path,
    prepared_source: Path,
    out_path: Path,
    repair_eval_mode: str = "official_mimicgen_theta_sweep",
) -> dict[str, Any]:
    run_dirs = sorted([p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("run_")])
    context_cache: dict[str, dict[str, Any]] = {}
    source_demo_keys = _list_source_demo_keys(prepared_source)
    all_records: list[dict[str, Any]] = []
    candidate_index = 0
    run_summaries: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        records, candidate_index = convert_run_dir(
            run_dir=run_dir,
            prepared_source=prepared_source,
            sweep_id=run_dir.name,
            repair_eval_mode=repair_eval_mode,
            context_cache=context_cache,
            source_demo_keys=source_demo_keys,
            candidate_index_start=candidate_index,
        )
        if not records:
            run_summaries.append({"sweep_id": run_dir.name, "num_records": 0, "skipped": True})
            continue
        num_success = sum(1 for r in records if r["success"])
        run_summaries.append(
            {
                "sweep_id": run_dir.name,
                "num_records": len(records),
                "num_success": num_success,
                "num_failure": len(records) - num_success,
                "theta": records[0]["theta"],
            }
        )
        all_records.extend(records)

    summary = _write_records(all_records, out_path)
    summary["run_summaries"] = run_summaries
    summary["num_sweep_runs"] = len(run_summaries)
    summary["num_theta_settings"] = _count_unique_theta_settings(all_records)
    summary["num_unique_source_demos"] = len({r.get("source_demo_key", r.get("demo_key")) for r in all_records})
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _list_source_demo_keys(prepared_source: Path) -> list[str]:
    with h5py.File(prepared_source, "r") as f:
        return sorted(f["data"].keys(), key=demo_sort_key)


def _count_unique_theta_settings(records: list[dict[str, Any]]) -> int:
    keys: set[str] = set()
    for row in records:
        theta = dict(row.get("theta", {}))
        theta.pop("candidate_index", None)
        theta.pop("sweep_id", None)
        keys.add(json.dumps(theta, sort_keys=True))
    return len(keys)


def _write_records(records: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    num_success = sum(1 for r in records if r["success"])
    num_failure = len(records) - num_success
    summary = {
        "output": str(out_path),
        "num_records": len(records),
        "num_success": num_success,
        "num_failure": num_failure,
        "target_success_met": num_success >= 5,
        "target_failure_met": num_failure >= 15,
        "target_records_met": len(records) >= 100,
        "target_theta_settings_met": _count_unique_theta_settings(records) >= 8,
        "num_unique_theta_settings": _count_unique_theta_settings(records),
        "num_unique_source_demos": len({r.get("source_demo_key", r.get("demo_key")) for r in records}),
    }
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default=None, help="Root dir with run_*/ subdirs from theta sweep")
    parser.add_argument(
        "--datagen-dir",
        default=None,
        help="Single datagen demo dir (legacy mode)",
    )
    parser.add_argument(
        "--prepared-source",
        default="runs/phygen_coffee_official/prepared_source/coffee_preparation.hdf5",
    )
    parser.add_argument(
        "--output",
        default="runs/phygen_coffee_theta_sweep/coffee_preparation_theta_sweep_feedback.jsonl",
    )
    parser.add_argument("--max-success", type=int, default=None)
    parser.add_argument("--max-failure", type=int, default=None)
    parser.add_argument(
        "--repair-eval-mode",
        default=None,
        help="Override repair_eval_mode tag",
    )
    args = parser.parse_args()

    prepared_source = (ROOT / args.prepared_source).resolve()
    out_path = (ROOT / args.output).resolve()

    if args.sweep_root:
        mode = args.repair_eval_mode or "official_mimicgen_theta_sweep"
        summary = convert_sweep_root(
            sweep_root=(ROOT / args.sweep_root).resolve(),
            prepared_source=prepared_source,
            out_path=out_path,
            repair_eval_mode=mode,
        )
    else:
        datagen_dir = (ROOT / (args.datagen_dir or "runs/phygen_coffee_official/official_datagen_smoke/demo")).resolve()
        mode = args.repair_eval_mode or "official_mimicgen_datagen_or_true_rollout"
        summary = convert_official_datagen(
            success_hdf5=datagen_dir / "demo.hdf5",
            failed_hdf5=datagen_dir / "demo_failed.hdf5",
            prepared_source=prepared_source,
            mg_config_path=datagen_dir / "mg_config.json",
            out_path=out_path,
            max_success=args.max_success,
            max_failure=args.max_failure,
            repair_eval_mode=mode,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
