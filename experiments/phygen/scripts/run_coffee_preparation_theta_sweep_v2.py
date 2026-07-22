#!/usr/bin/env python3
"""Run official MimicGen coffee_preparation datagen – theta sweep v2 (expanded dimensions)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py

ROOT = Path(__file__).resolve().parents[3]
MIMICGEN_ROOT = Path(os.environ.get("PHYGEN_MIMICGEN_ROOT", ROOT / "third_party" / "mimicgen")).resolve()
GENERATE_SCRIPT = MIMICGEN_ROOT / "mimicgen" / "scripts" / "generate_dataset.py"
TEMPLATE_CONFIG = MIMICGEN_ROOT / "mimicgen" / "exps" / "templates" / "robosuite" / "coffee_preparation.json"

# Source replay successes from source_replay_sanity.json (exclude demo_2, demo_4).
REPLAY_SUCCESS_DEMOS = [
    "demo_0",
    "demo_1",
    "demo_3",
    "demo_5",
    "demo_6",
    "demo_7",
    "demo_8",
    "demo_9",
]

# 16 base settings from v1 sweep.
BASE_GRID: list[dict[str, Any]] = [
    {"selection_strategy": "nearest_neighbor_object", "select_src_per_subtask": False, "action_noise": 0.0, "offset_range": [5, 10]},
    {"selection_strategy": "nearest_neighbor_object", "select_src_per_subtask": False, "action_noise": 0.02, "offset_range": [10, 20]},
    {"selection_strategy": "nearest_neighbor_object", "select_src_per_subtask": False, "action_noise": 0.05, "offset_range": [5, 10]},
    {"selection_strategy": "nearest_neighbor_object", "select_src_per_subtask": False, "action_noise": 0.08, "offset_range": [10, 20]},
    {"selection_strategy": "nearest_neighbor_object", "select_src_per_subtask": True, "action_noise": 0.0, "offset_range": [10, 20]},
    {"selection_strategy": "nearest_neighbor_object", "select_src_per_subtask": True, "action_noise": 0.02, "offset_range": [5, 10]},
    {"selection_strategy": "nearest_neighbor_object", "select_src_per_subtask": True, "action_noise": 0.05, "offset_range": [10, 20]},
    {"selection_strategy": "nearest_neighbor_object", "select_src_per_subtask": True, "action_noise": 0.08, "offset_range": [5, 10]},
    {"selection_strategy": "random", "select_src_per_subtask": False, "action_noise": 0.0, "offset_range": [10, 20]},
    {"selection_strategy": "random", "select_src_per_subtask": False, "action_noise": 0.02, "offset_range": [5, 10]},
    {"selection_strategy": "random", "select_src_per_subtask": False, "action_noise": 0.05, "offset_range": [10, 20]},
    {"selection_strategy": "random", "select_src_per_subtask": False, "action_noise": 0.08, "offset_range": [5, 10]},
    {"selection_strategy": "random", "select_src_per_subtask": True, "action_noise": 0.0, "offset_range": [5, 10]},
    {"selection_strategy": "random", "select_src_per_subtask": True, "action_noise": 0.02, "offset_range": [10, 20]},
    {"selection_strategy": "random", "select_src_per_subtask": True, "action_noise": 0.05, "offset_range": [5, 10]},
    {"selection_strategy": "random", "select_src_per_subtask": True, "action_noise": 0.08, "offset_range": [10, 20]},
]

INTERP_STEPS = [3, 5, 10]
TRANSFORM_OPTS = [False, True]
INTERPOLATE_OPTS = [True, False]
NN_K_OPTS = [1, 3, 5, 10]


def build_v2_grid(num_settings: int = 40) -> list[dict[str, Any]]:
    """Balanced Latin-style subset over base grid + expanded theta axes."""
    grid: list[dict[str, Any]] = []
    for i in range(num_settings):
        base = dict(BASE_GRID[i % len(BASE_GRID)])
        base["num_interpolation_steps"] = INTERP_STEPS[i % len(INTERP_STEPS)]
        base["transform_first_robot_pose"] = TRANSFORM_OPTS[(i // 3) % len(TRANSFORM_OPTS)]
        base["interpolate_from_last_target_pose"] = INTERPOLATE_OPTS[(i // 6) % len(INTERPOLATE_OPTS)]
        if base["selection_strategy"] == "nearest_neighbor_object":
            base["nn_k"] = NN_K_OPTS[i % len(NN_K_OPTS)]
        else:
            base["nn_k"] = 3
        base["num_fixed_steps"] = 0
        grid.append(base)
    return grid


SWEEP_GRID = build_v2_grid(40)


def _demo_sort_key(key: str) -> int:
    return int(key.split("_")[-1])


def filter_prepared_source(
    src_path: Path,
    dst_path: Path,
    demo_keys: list[str],
) -> Path:
    """Copy only selected demos into a new prepared source HDF5."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.is_file():
        dst_path.unlink()

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        if "data" in src:
            data_grp = dst.create_group("data")
            for dk, dv in src["data"].attrs.items():
                data_grp.attrs[dk] = dv
            for demo_key in sorted(demo_keys, key=_demo_sort_key):
                src.copy(f"data/{demo_key}", data_grp, name=demo_key)
        for key in src.keys():
            if key != "data":
                src.copy(key, dst, name=key)

    meta = {
        "source_path": str(src_path),
        "filtered_demos": sorted(demo_keys, key=_demo_sort_key),
        "num_demos": len(demo_keys),
    }
    meta_path = dst_path.with_suffix(".filter_meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return dst_path


def _load_template() -> dict[str, Any]:
    with open(TEMPLATE_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def _theta_record(sweep_id: str, setting: dict[str, Any], seed: int) -> dict[str, Any]:
    strategy = str(setting["selection_strategy"])
    return {
        "sweep_id": sweep_id,
        "selection_strategy": strategy,
        "select_src_per_subtask": bool(setting["select_src_per_subtask"]),
        "transform_first_robot_pose": bool(setting["transform_first_robot_pose"]),
        "interpolate_from_last_target_pose": bool(setting["interpolate_from_last_target_pose"]),
        "action_noise": float(setting["action_noise"]),
        "num_interpolation_steps": int(setting["num_interpolation_steps"]),
        "num_fixed_steps": int(setting.get("num_fixed_steps", 0)),
        "offset_range": [int(setting["offset_range"][0]), int(setting["offset_range"][1])],
        "nn_k": int(setting.get("nn_k", 3)),
        "seed": int(seed),
    }


def build_config(
    setting: dict[str, Any],
    *,
    prepared_source: Path,
    output_root: Path,
    sweep_id: str,
    num_trials: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _load_template()
    cfg["experiment"]["source"]["dataset_path"] = str(prepared_source.resolve())
    cfg["experiment"]["source"]["n"] = None
    cfg["experiment"]["source"]["filter_key"] = None
    cfg["experiment"]["source"]["start"] = None

    run_dir = output_root / sweep_id
    cfg["experiment"]["generation"]["path"] = str(run_dir)
    cfg["experiment"]["generation"]["guarantee"] = False
    cfg["experiment"]["generation"]["keep_failed"] = True
    cfg["experiment"]["generation"]["num_trials"] = int(num_trials)
    cfg["experiment"]["generation"]["select_src_per_subtask"] = bool(setting["select_src_per_subtask"])
    cfg["experiment"]["generation"]["transform_first_robot_pose"] = bool(setting["transform_first_robot_pose"])
    cfg["experiment"]["generation"]["interpolate_from_last_target_pose"] = bool(setting["interpolate_from_last_target_pose"])

    cfg["experiment"]["task"]["name"] = "CoffeePreparation_D0"
    cfg["experiment"]["task"]["interface"] = "MG_CoffeePreparation"
    cfg["experiment"]["task"]["interface_type"] = "robosuite"
    cfg["experiment"]["max_num_failures"] = max(num_trials, 50)
    cfg["experiment"]["render_video"] = False
    cfg["experiment"]["seed"] = int(seed)

    cfg["obs"]["collect_obs"] = True
    cfg["obs"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]

    strategy = str(setting["selection_strategy"])
    offset = setting["offset_range"]
    nn_k = int(setting.get("nn_k", 3))
    kwargs = {"nn_k": nn_k} if strategy == "nearest_neighbor_object" else None
    for sub_key in sorted(cfg["task"]["task_spec"].keys(), key=lambda k: int(k.split("_")[-1])):
        sub = cfg["task"]["task_spec"][sub_key]
        sub["selection_strategy"] = strategy
        sub["selection_strategy_kwargs"] = copy.deepcopy(kwargs)
        sub["action_noise"] = float(setting["action_noise"])
        sub["num_interpolation_steps"] = int(setting["num_interpolation_steps"])
        sub["num_fixed_steps"] = int(setting.get("num_fixed_steps", 0))
        if sub.get("subtask_term_offset_range") is not None:
            sub["subtask_term_offset_range"] = [int(offset[0]), int(offset[1])]

    theta = _theta_record(sweep_id, setting, seed)
    return cfg, theta


def run_one_config(*, config_path: Path, seed: int, retries: int, log_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--config",
        str(config_path),
        "--auto-remove-exp",
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MIMICGEN_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("MUJOCO_GL", "egl")

    for attempt in range(1, retries + 1):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"\n===== attempt {attempt} seed={seed} =====\n")
            logf.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        if proc.returncode == 0:
            return {"status": "ok", "attempts": attempt, "returncode": 0}
        time.sleep(2.0)
    return {"status": "failed", "attempts": retries, "error": "max retries exceeded"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs/phygen_coffee_theta_sweep_v2")
    parser.add_argument(
        "--prepared-source",
        default="runs/phygen_coffee_official/prepared_source/coffee_preparation.hdf5",
    )
    parser.add_argument(
        "--filtered-source",
        default="runs/phygen_coffee_theta_sweep_v2/prepared_source_replay_success.hdf5",
    )
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-settings", type=int, default=40)
    parser.add_argument("--base-seed", type=int, default=20260708)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    args = parser.parse_args()

    global SWEEP_GRID
    SWEEP_GRID = build_v2_grid(args.num_settings)

    output_root = (ROOT / args.output_root).resolve()
    prepared_source = (ROOT / args.prepared_source).resolve()
    filtered_source = (ROOT / args.filtered_source).resolve()

    if not prepared_source.is_file():
        raise FileNotFoundError(prepared_source)
    if not args.skip_filter and (not args.dry_run):
        filter_prepared_source(prepared_source, filtered_source, REPLAY_SUCCESS_DEMOS)
        print(f"filtered source: {filtered_source} ({len(REPLAY_SUCCESS_DEMOS)} demos)")
    source_for_run = filtered_source if filtered_source.is_file() else prepared_source

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": "v2",
        "output_root": str(output_root),
        "prepared_source": str(prepared_source),
        "filtered_source": str(source_for_run),
        "replay_success_demos": REPLAY_SUCCESS_DEMOS,
        "num_settings": len(SWEEP_GRID),
        "num_trials": args.num_trials,
        "base_seed": args.base_seed,
        "runs": [],
    }

    grid = SWEEP_GRID[args.start_index :]
    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    for i, setting in enumerate(grid):
        sweep_id = f"run_{args.start_index + i:03d}"
        seed = args.base_seed + args.start_index + i
        run_dir = output_root / sweep_id
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg, theta = build_config(
            setting,
            prepared_source=source_for_run,
            output_root=output_root,
            sweep_id=sweep_id,
            num_trials=args.num_trials,
            seed=seed,
        )
        config_path = run_dir / "config.json"
        theta_path = run_dir / "theta.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        with theta_path.open("w", encoding="utf-8") as f:
            json.dump(theta, f, indent=2)

        run_info: dict[str, Any] = {
            "sweep_id": sweep_id,
            "seed": seed,
            "theta": theta,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "status": "dry_run" if args.dry_run else "pending",
        }
        if args.dry_run:
            manifest["runs"].append(run_info)
            continue

        result = run_one_config(
            config_path=config_path,
            seed=seed,
            retries=args.retries,
            log_path=run_dir / "generate_dataset.log",
        )
        run_info.update(result)
        stats_path = run_dir / "demo" / "important_stats.json"
        if stats_path.is_file():
            with stats_path.open("r", encoding="utf-8") as f:
                run_info["important_stats"] = json.load(f)
        manifest["runs"].append(run_info)
        print(json.dumps(run_info, ensure_ascii=False), flush=True)

    manifest_path = output_root / "sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
