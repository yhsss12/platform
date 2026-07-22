#!/usr/bin/env python3
"""Online selected rollout validation for CoffeePreparation PhyGen vs baselines."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIMICGEN_ROOT = Path(os.environ.get("PHYGEN_MIMICGEN_ROOT", ROOT / "third_party" / "mimicgen")).resolve()
GENERATE_SCRIPT = MIMICGEN_ROOT / "mimicgen" / "scripts" / "generate_dataset.py"
TEMPLATE_CONFIG = MIMICGEN_ROOT / "mimicgen" / "exps" / "templates" / "robosuite" / "coffee_preparation.json"

DEFAULT_HELD_OUT = ["demo_6", "demo_7", "demo_8"]
TRAIN_DEMOS = ["demo_0", "demo_1", "demo_3", "demo_5", "demo_9"]

OFFICIAL_DEFAULT_THETA: dict[str, Any] = {
    "selection_strategy": "random",
    "select_src_per_subtask": False,
    "transform_first_robot_pose": False,
    "interpolate_from_last_target_pose": True,
    "action_noise": 0.05,
    "num_interpolation_steps": 5,
    "num_fixed_steps": 0,
    "offset_range": [5, 10],
    "nn_k": 3,
}


def _demo_sort_key(key: str) -> int:
    return int(str(key).split("_")[-1])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _theta_key(theta: dict[str, Any]) -> str:
    d = {k: v for k, v in theta.items() if k not in ("candidate_index", "sweep_id", "seed")}
    return json.dumps(d, sort_keys=True)


def filter_prepared_source(src_path: Path, dst_path: Path, demo_keys: list[str]) -> Path:
    import h5py

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.is_file():
        dst_path.unlink()
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        data_grp = dst.create_group("data")
        for dk, dv in src["data"].attrs.items():
            data_grp.attrs[dk] = dv
        for demo_key in sorted(demo_keys, key=_demo_sort_key):
            src.copy(f"data/{demo_key}", data_grp, name=demo_key)
        for key in src.keys():
            if key != "data":
                src.copy(key, dst, name=key)
    return dst_path


def _load_template() -> dict[str, Any]:
    with open(TEMPLATE_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mimicgen_config(
    theta: dict[str, Any],
    *,
    prepared_source: Path,
    output_dir: Path,
    num_trials: int,
    seed: int,
) -> dict[str, Any]:
    cfg = _load_template()
    cfg["experiment"]["source"]["dataset_path"] = str(prepared_source.resolve())
    cfg["experiment"]["source"]["n"] = None
    cfg["experiment"]["source"]["filter_key"] = None
    cfg["experiment"]["source"]["start"] = None
    cfg["experiment"]["generation"]["path"] = str(output_dir)
    cfg["experiment"]["generation"]["guarantee"] = False
    cfg["experiment"]["generation"]["keep_failed"] = True
    cfg["experiment"]["generation"]["num_trials"] = int(num_trials)
    cfg["experiment"]["generation"]["select_src_per_subtask"] = bool(theta.get("select_src_per_subtask", False))
    cfg["experiment"]["generation"]["transform_first_robot_pose"] = bool(theta.get("transform_first_robot_pose", False))
    cfg["experiment"]["generation"]["interpolate_from_last_target_pose"] = bool(
        theta.get("interpolate_from_last_target_pose", True)
    )
    cfg["experiment"]["task"]["name"] = "CoffeePreparation_D0"
    cfg["experiment"]["task"]["interface"] = "MG_CoffeePreparation"
    cfg["experiment"]["task"]["interface_type"] = "robosuite"
    cfg["experiment"]["max_num_failures"] = max(num_trials, 50)
    cfg["experiment"]["render_video"] = False
    cfg["experiment"]["seed"] = int(seed)

    cfg["obs"]["collect_obs"] = True
    cfg["obs"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]

    strategy = str(theta.get("selection_strategy", "random"))
    offset = theta.get("offset_range", [5, 10])
    nn_k = int(theta.get("nn_k", 3))
    kwargs = {"nn_k": nn_k} if strategy == "nearest_neighbor_object" else None
    for sub_key in sorted(cfg["task"]["task_spec"].keys(), key=lambda k: int(k.split("_")[-1])):
        sub = cfg["task"]["task_spec"][sub_key]
        sub["selection_strategy"] = strategy
        sub["selection_strategy_kwargs"] = copy.deepcopy(kwargs)
        sub["action_noise"] = float(theta.get("action_noise", 0.05))
        sub["num_interpolation_steps"] = int(theta.get("num_interpolation_steps", 5))
        sub["num_fixed_steps"] = int(theta.get("num_fixed_steps", 0))
        if sub.get("subtask_term_offset_range") is not None:
            sub["subtask_term_offset_range"] = [int(offset[0]), int(offset[1])]
    return cfg


def run_official_rollout(
    *,
    theta: dict[str, Any],
    prepared_source: Path,
    output_dir: Path,
    num_trials: int,
    seed: int,
    retries: int = 2,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "rollout_result.json"
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("status") == "ok":
            stats = cached.get("important_stats") or {}
            success_hdf5 = output_dir / "demo" / "demo.hdf5"
            failed_hdf5 = output_dir / "demo" / "demo_failed.hdf5"
            if int(stats.get("num_attempts", 0)) > 0 or success_hdf5.is_file() or failed_hdf5.is_file():
                return cached
            result_path.unlink(missing_ok=True)

    config_path = output_dir / "config.json"
    cfg = build_mimicgen_config(
        theta,
        prepared_source=prepared_source,
        output_dir=output_dir,
        num_trials=num_trials,
        seed=seed,
    )
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

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

    log_path = output_dir / "generate_dataset.log"
    last_error = "unknown"
    for attempt in range(1, retries + 1):
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"\n===== attempt {attempt} seed={seed} num_trials={num_trials} =====\n")
            logf.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        if proc.returncode == 0:
            break
        last_error = f"returncode={proc.returncode}"
        time.sleep(2.0)
    else:
        result = {
            "status": "failed",
            "error": last_error,
            "theta": theta,
            "num_trials": num_trials,
            "trial_successes": [],
            "num_success": 0,
            "num_failures": num_trials,
        }
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    stats_path = output_dir / "demo" / "important_stats.json"
    stats: dict[str, Any] = {}
    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

    success_hdf5 = output_dir / "demo" / "demo.hdf5"
    failed_hdf5 = output_dir / "demo" / "demo_failed.hdf5"
    num_attempts = int(stats.get("num_attempts", 0))
    if num_attempts <= 0 and not success_hdf5.is_file() and not failed_hdf5.is_file():
        result = {
            "status": "failed",
            "error": "generate_dataset produced no attempts (likely runtime error)",
            "theta": theta,
            "num_trials": num_trials,
            "trial_successes": [],
            "num_success": 0,
            "num_failures": num_trials,
        }
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    num_success = int(stats.get("num_success", 0))
    num_failures = int(stats.get("num_failures", 0))
    if num_attempts <= 0:
        num_attempts = num_trials

    trial_successes: list[bool] = []
    if success_hdf5.is_file() or failed_hdf5.is_file():
        import h5py

        for _ in range(num_success):
            trial_successes.append(True)
        for _ in range(num_failures):
            trial_successes.append(False)
        if not trial_successes and num_attempts > 0:
            trial_successes = [num_success >= 1]
    else:
        trial_successes = [num_success >= 1] if num_attempts else [False]

    result = {
        "status": "ok",
        "theta": theta,
        "num_trials": num_attempts,
        "num_success": num_success,
        "num_failures": num_failures,
        "trial_successes": trial_successes,
        "any_success": bool(num_success >= 1),
        "all_success": bool(num_success == num_attempts and num_attempts > 0),
        "important_stats": stats,
        "output_dir": str(output_dir),
    }
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def load_context_for_demo(prepared_source: Path, demo_key: str) -> dict[str, Any]:
    from phygen.adapters.mimicgen.coffee_repair import (
        _action_delta,
        _pick_failure_frame,
        compute_context_metrics,
        load_demo_obs_and_actions,
    )

    obs, actions = load_demo_obs_and_actions(prepared_source, demo_key)
    fail_idx = _pick_failure_frame(obs, actions)
    context_metrics = compute_context_metrics(obs[fail_idx], action_delta=_action_delta(actions, fail_idx))
    return {
        "demo_key": demo_key,
        "failure_frame": int(fail_idx),
        "context_metrics": context_metrics,
    }


def build_train_split(
    feedback_path: Path,
    held_out: list[str],
    train_out: Path,
) -> dict[str, Any]:
    rows = _load_jsonl(feedback_path)
    held = set(held_out)
    train_rows = [r for r in rows if str(r.get("source_demo_key", r.get("demo_key"))) not in held]
    _write_jsonl(train_out, train_rows)
    return {
        "held_out": sorted(held_out, key=_demo_sort_key),
        "train_demos": sorted({str(r.get("source_demo_key", r.get("demo_key"))) for r in train_rows}, key=_demo_sort_key),
        "num_train": len(train_rows),
        "train_success": sum(1 for r in train_rows if r.get("success")),
        "train_failure": sum(1 for r in train_rows if not r.get("success")),
    }


def find_best_fixed_theta(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket: dict[str, list[bool]] = defaultdict(list)
    theta_by_key: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        key = _theta_key(row["theta"])
        bucket[key].append(bool(row.get("success")))
        theta_by_key[key] = row["theta"]
    best_key = max(bucket.keys(), key=lambda k: (sum(bucket[k]) / len(bucket[k]), len(bucket[k])))
    rate = sum(bucket[best_key]) / len(bucket[best_key])
    theta = dict(theta_by_key[best_key])
    theta.pop("candidate_index", None)
    theta.pop("sweep_id", None)
    return {"theta": theta, "train_success_rate": rate, "train_count": len(bucket[best_key]), "theta_key": best_key}


def train_phygen(train_feedback: Path, train_dir: Path, epochs: int) -> Path:
    train_dir.mkdir(parents=True, exist_ok=True)
    ckpt = train_dir / "coffee_preparation_failed_conditioned_pinn.pt"
    if ckpt.is_file():
        return ckpt
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_phygen.py"),
        "--task",
        "coffee_preparation",
        "--feedback-jsonl",
        str(train_feedback.relative_to(ROOT)),
        "--output-dir",
        str(train_dir.relative_to(ROOT)),
        "--epochs",
        str(epochs),
        "--standard-pinn",
        "--use-component-loss",
        "--pool-size",
        "32",
        "--budget",
        "5",
        "--include-repaired",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return ckpt


def load_model(checkpoint: Path):
    import torch
    from phygen.adapters.registry import get_adapter
    from phygen.core.residual_field_model import FeatureLayout, RepairParameterResidualFieldPINN, build_mlp_selector

    adapter = get_adapter("coffee_preparation")
    ckpt = torch.load(checkpoint, map_location="cpu")
    layout = FeatureLayout(
        context_dim=int(ckpt["layout"]["context_dim"]),
        theta_disc_dim=int(ckpt["layout"]["theta_disc_dim"]),
        theta_cont_dim=int(ckpt["layout"]["theta_cont_dim"]),
    )
    component_keys = ckpt.get("component_keys") or []
    if component_keys:
        model = RepairParameterResidualFieldPINN.build(layout, len(component_keys))
    else:
        model = build_mlp_selector(layout.input_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return adapter, model


def generate_candidate_pool(
    adapter: Any,
    demo_key: str,
    context_metrics: dict[str, float],
    *,
    pool_size: int,
    seed: int,
    start_index: int = 10000,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + _demo_sort_key(demo_key) * 1009)
    pool: list[dict[str, Any]] = []
    for i in range(pool_size):
        theta = adapter.sample_repair_theta(start_index + i, rng, candidate_mode="default")
        pool.append(
            {
                "candidate_index": start_index + i,
                "demo_key": demo_key,
                "context_metrics": dict(context_metrics),
                "theta": theta,
            }
        )
    return pool


def score_candidate_pool(
    adapter: Any,
    model: Any,
    pool: list[dict[str, Any]],
    *,
    boundary_weight: float = 1.5,
    uncertainty_weight: float = 0.3,
) -> list[dict[str, Any]]:
    from phygen.core.selector import attach_selector_scores, predict_with_details

    rows = [dict(r) for r in pool]
    contexts = [r["context_metrics"] for r in rows]
    thetas = [r["theta"] for r in rows]
    pred_e, pred_p, details = predict_with_details(adapter, model, contexts, thetas)
    attach_selector_scores(adapter, rows, pred_e, pred_p, details, boundary_weight, uncertainty_weight)
    return rows


def select_top_k(rows: list[dict[str, Any]], k: int, *, by: str) -> list[dict[str, Any]]:
    if by == "acquisition":
        key_fn = lambda r: float(r.get("acquisition_score", r.get("utility_score", 0.0)))
        reverse = False
    elif by == "pred_success_prob":
        key_fn = lambda r: float(r.get("pred_success_prob", 0.0))
        reverse = True
    else:
        raise ValueError(by)
    ranked = sorted(rows, key=key_fn, reverse=reverse)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        tk = _theta_key(row["theta"])
        if tk in seen:
            continue
        seen.add(tk)
        out.append(row)
        if len(out) >= k:
            break
    return out


@dataclass
class MethodPlan:
    name: str
    demo_key: str
    top1_thetas: list[dict[str, Any]] = field(default_factory=list)
    top3_thetas: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RolloutRecord:
    method: str
    demo_key: str
    tier: str
    theta: dict[str, Any]
    num_trials: int
    trial_successes: list[bool]
    any_success: bool
    rollout_dir: str


def execute_method_rollouts(
    plan: MethodPlan,
    *,
    prepared_sources: dict[str, Path],
    output_root: Path,
    base_seed: int,
) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    demo_key = plan.demo_key
    source = prepared_sources[demo_key]

    if plan.top1_thetas:
        theta = plan.top1_thetas[0]
        run_dir = output_root / plan.name / demo_key / "top1"
        res = run_official_rollout(
            theta=theta,
            prepared_source=source,
            output_dir=run_dir,
            num_trials=1,
            seed=base_seed + _demo_sort_key(demo_key) * 17 + 1,
        )
        records.append(
            RolloutRecord(
                method=plan.name,
                demo_key=demo_key,
                tier="top1",
                theta=theta,
                num_trials=1,
                trial_successes=res.get("trial_successes", [False]),
                any_success=bool(res.get("any_success", False)),
                rollout_dir=str(run_dir),
            )
        )

    if plan.top3_thetas:
        same_theta = len({_theta_key(t) for t in plan.top3_thetas}) == 1
        if same_theta and len(plan.top3_thetas) >= 3:
            theta = plan.top3_thetas[0]
            run_dir = output_root / plan.name / demo_key / "top3_batch"
            res = run_official_rollout(
                theta=theta,
                prepared_source=source,
                output_dir=run_dir,
                num_trials=3,
                seed=base_seed + _demo_sort_key(demo_key) * 17 + 3,
            )
            trials = res.get("trial_successes", [False, False, False])
            while len(trials) < 3:
                trials.append(False)
            for i in range(3):
                records.append(
                    RolloutRecord(
                        method=plan.name,
                        demo_key=demo_key,
                        tier="top3",
                        theta=theta,
                        num_trials=1,
                        trial_successes=[bool(trials[i])],
                        any_success=bool(trials[i]),
                        rollout_dir=str(run_dir),
                    )
                )
        else:
            for i, theta in enumerate(plan.top3_thetas[:3]):
                run_dir = output_root / plan.name / demo_key / f"top3_{i}"
                res = run_official_rollout(
                    theta=theta,
                    prepared_source=source,
                    output_dir=run_dir,
                    num_trials=1,
                    seed=base_seed + _demo_sort_key(demo_key) * 17 + 10 + i,
                )
                records.append(
                    RolloutRecord(
                        method=plan.name,
                        demo_key=demo_key,
                        tier="top3",
                        theta=theta,
                        num_trials=1,
                        trial_successes=res.get("trial_successes", [False]),
                        any_success=bool(res.get("any_success", False)),
                        rollout_dir=str(run_dir),
                    )
                )
    return records


def aggregate_method_metrics(records: list[RolloutRecord], method: str) -> dict[str, Any]:
    method_records = [r for r in records if r.method == method]
    top1 = [r for r in method_records if r.tier == "top1"]
    top3 = [r for r in method_records if r.tier == "top3"]

    demos = sorted({r.demo_key for r in method_records}, key=_demo_sort_key)
    top1_demo_success = sum(1 for d in demos if any(r.any_success for r in top1 if r.demo_key == d))
    top3_demo_success = 0
    for d in demos:
        d_top3 = [r for r in top3 if r.demo_key == d]
        if d_top3 and any(r.any_success for r in d_top3):
            top3_demo_success += 1

    all_trials = method_records
    rollout_success = sum(1 for r in all_trials if r.any_success)
    rollout_total = len(all_trials)
    rollout_budget = sum(r.num_trials for r in all_trials)

    return {
        "method": method,
        "num_held_out_demos": len(demos),
        "top1_selected_trajectory_success_rate": top1_demo_success / max(len(demos), 1),
        "top1_selected_trajectory_success_count": f"{top1_demo_success}/{len(demos)}",
        "top3_repair_success_rate": top3_demo_success / max(len(demos), 1),
        "top3_repair_success_count": f"{top3_demo_success}/{len(demos)}",
        "selected_rollout_success_rate": rollout_success / max(rollout_total, 1),
        "selected_rollout_success_count": f"{rollout_success}/{rollout_total}",
        "rollout_budget": rollout_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs/phygen_coffee_online_rollout")
    parser.add_argument(
        "--feedback-jsonl",
        default="runs/phygen_coffee_theta_sweep_v2/coffee_preparation_theta_sweep_v2_feedback.jsonl",
    )
    parser.add_argument(
        "--prepared-source",
        default="runs/phygen_coffee_theta_sweep_v2/prepared_source_replay_success.hdf5",
    )
    parser.add_argument("--held-out", default=",".join(DEFAULT_HELD_OUT))
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=88001)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true", help="only plan/select, no datagen")
    args = parser.parse_args()

    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_source = (ROOT / args.prepared_source).resolve()
    feedback_path = (ROOT / args.feedback_jsonl).resolve()
    held_out = [x.strip() for x in args.held_out.split(",") if x.strip()]

    train_feedback = output_root / "train_feedback.jsonl"
    split_summary = build_train_split(feedback_path, held_out, train_feedback)
    split_summary["held_out"] = held_out
    with (output_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)

    train_dir = output_root / "train"
    if not args.skip_train:
        ckpt = train_phygen(train_feedback, train_dir, args.train_epochs)
    else:
        ckpt = train_dir / "coffee_preparation_failed_conditioned_pinn.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint missing: {ckpt}")

    adapter, model = load_model(ckpt)
    train_rows = _load_jsonl(train_feedback)
    best_fixed = find_best_fixed_theta(train_rows)

    per_demo_sources: dict[str, Path] = {}
    sources_dir = output_root / "prepared_sources"
    for demo_key in held_out:
        per_demo_sources[demo_key] = filter_prepared_source(
            prepared_source,
            sources_dir / f"{demo_key}.hdf5",
            [demo_key],
        )

    rng = random.Random(args.base_seed)
    plans: list[MethodPlan] = []
    selection_dump: dict[str, Any] = {"held_out": held_out, "demos": {}, "best_fixed_theta": best_fixed}

    for demo_key in sorted(held_out, key=_demo_sort_key):
        ctx = load_context_for_demo(prepared_source, demo_key)
        pool = generate_candidate_pool(
            adapter,
            demo_key,
            ctx["context_metrics"],
            pool_size=args.pool_size,
            seed=args.base_seed,
        )
        scored = score_candidate_pool(adapter, model, pool)

        phygen_top1 = select_top_k(scored, 1, by="acquisition")
        phygen_top3 = select_top_k(scored, 3, by="acquisition")
        prob_top1 = select_top_k(scored, 1, by="pred_success_prob")
        prob_top3 = select_top_k(scored, 3, by="pred_success_prob")

        shuffled = list(scored)
        rng.shuffle(shuffled)
        random_top1 = select_top_k(shuffled, 1, by="acquisition")
        random_top3 = select_top_k(shuffled, 3, by="acquisition")

        default_theta = dict(OFFICIAL_DEFAULT_THETA)
        best_theta = dict(best_fixed["theta"])

        demo_plans = [
            MethodPlan("official_default", demo_key, [default_theta], [default_theta, default_theta, default_theta]),
            MethodPlan("random_theta", demo_key, [r["theta"] for r in random_top1], [r["theta"] for r in random_top3]),
            MethodPlan("best_fixed_train", demo_key, [best_theta], [best_theta, best_theta, best_theta]),
            MethodPlan("phygen_acquisition", demo_key, [r["theta"] for r in phygen_top1], [r["theta"] for r in phygen_top3]),
            MethodPlan("pred_success_prob", demo_key, [r["theta"] for r in prob_top1], [r["theta"] for r in prob_top3]),
        ]
        plans.extend(demo_plans)

        selected_path = output_root / "selections" / demo_key / "selected_theta.json"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "demo_key": demo_key,
            "context_metrics": ctx["context_metrics"],
            "pool_size": args.pool_size,
            "phygen_top1": phygen_top1,
            "phygen_top3": phygen_top3,
            "random_top1": random_top1,
            "random_top3": random_top3,
            "pred_prob_top1": prob_top1,
            "pred_prob_top3": prob_top3,
            "official_default_theta": default_theta,
            "best_fixed_theta": best_theta,
        }
        with selected_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        selection_dump["demos"][demo_key] = payload

    with (output_root / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(selection_dump, f, indent=2, ensure_ascii=False)

    all_records: list[RolloutRecord] = []
    if not args.skip_rollout:
        rollouts_root = output_root / "rollouts"
        for plan in plans:
            recs = execute_method_rollouts(
                plan,
                prepared_sources=per_demo_sources,
                output_root=rollouts_root,
                base_seed=args.base_seed,
            )
            all_records.extend(recs)
            print(json.dumps({"method": plan.name, "demo": plan.demo_key, "records": len(recs)}, ensure_ascii=False), flush=True)

    serial_records = [
        {
            "method": r.method,
            "demo_key": r.demo_key,
            "tier": r.tier,
            "theta": r.theta,
            "num_trials": r.num_trials,
            "trial_successes": r.trial_successes,
            "any_success": r.any_success,
            "rollout_dir": r.rollout_dir,
        }
        for r in all_records
    ]
    with (output_root / "rollout_records.json").open("w", encoding="utf-8") as f:
        json.dump(serial_records, f, indent=2, ensure_ascii=False)

    method_names = [
        "official_default",
        "random_theta",
        "best_fixed_train",
        "phygen_acquisition",
        "pred_success_prob",
    ]
    table = [aggregate_method_metrics(all_records, m) for m in method_names]
    report = {
        "held_out_demos": held_out,
        "pool_size": args.pool_size,
        "train_split": split_summary,
        "best_fixed_theta": best_fixed,
        "metrics_table": table,
        "meets_90_top1": any(r["top1_selected_trajectory_success_rate"] >= 0.9 for r in table if r["method"] == "phygen_acquisition"),
        "meets_90_top3": any(r["top3_repair_success_rate"] >= 0.9 for r in table if r["method"] == "phygen_acquisition"),
    }
    with (output_root / "online_rollout_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Online Rollout Validation Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
