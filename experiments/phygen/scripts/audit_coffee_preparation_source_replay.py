#!/usr/bin/env python3
"""Audit official coffee_preparation source demo replay via robomimic env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MIMICGEN_ROOT = Path(os.environ.get("PHYGEN_MIMICGEN_ROOT", ROOT / "third_party" / "mimicgen")).resolve()
if str(MIMICGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMICGEN_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import mimicgen.configs.robosuite  # noqa: F401,E402 register configs
import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402

from phygen.adapters.mimicgen.coffee_repair import (  # noqa: E402
    compute_context_metrics,
    demo_sort_key,
)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _get_object_obs(obs_dict: dict[str, Any]) -> np.ndarray:
    if "object-state" in obs_dict:
        return np.asarray(obs_dict["object-state"], dtype=np.float64)
    if "object" in obs_dict:
        return np.asarray(obs_dict["object"], dtype=np.float64)
    raise KeyError(f"object obs missing from keys: {list(obs_dict.keys())}")


def replay_demo(env: Any, states: np.ndarray, actions: np.ndarray, model_file: str) -> dict[str, Any]:
    env_action_dim = int(env.action_dimension)
    dataset_action_dim = int(actions.shape[1])
    action_dim_note: str | None = None

    initial_state = {"states": states[0], "model": model_file}
    env.reset()
    env.reset_to(initial_state)

    executed = 0
    terminated_early = False
    termination_step: int | None = None
    failure_reason: str | None = None

    for t, action in enumerate(actions):
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] != env_action_dim:
            failure_reason = (
                f"action_dim_mismatch at step {t}: dataset={action.shape[0]} env={env_action_dim}"
            )
            terminated_early = True
            termination_step = t
            break
        ret = env.step(action)
        executed += 1
        if len(ret) == 4:
            _, _, done, _ = ret
        else:
            _, _, term, trunc, _ = ret
            done = bool(term or trunc)
        if done:
            terminated_early = True
            termination_step = t
            failure_reason = f"env_done_at_step_{t}"
            break

    obs = env.get_observation()
    object_state = _get_object_obs(obs)
    metrics = compute_context_metrics(object_state)
    task_metrics = env.base_env._get_partial_task_metrics()
    final_task_success = bool(task_metrics.get("task", False))

    return {
        "executed_steps": executed,
        "terminated_early": terminated_early,
        "termination_step": termination_step,
        "failure_reason": failure_reason,
        "action_dim": dataset_action_dim,
        "env_action_dim": env_action_dim,
        "action_dim_note": action_dim_note,
        "final_stage_progress": float(metrics["stage_progress"]),
        "final_energy": float(metrics["energy"]),
        "final_task_success": final_task_success,
        "task_metrics": {k: bool(v) if isinstance(v, (bool, np.bool_)) else float(v) for k, v in task_metrics.items()},
        "replay_success": bool(final_task_success and not terminated_early and failure_reason is None),
    }


def audit_dataset(dataset_path: str | Path, num_demos: int = 10) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )

    with h5py.File(dataset_path, "r") as f:
        env_args_raw = _decode_attr(f["data"].attrs.get("env_args"))
        env_args = json.loads(env_args_raw) if isinstance(env_args_raw, str) else env_args_raw
        demo_keys = sorted(f["data"].keys(), key=demo_sort_key)[:num_demos]
        records: list[dict[str, Any]] = []
        for demo_key in demo_keys:
            grp = f[f"data/{demo_key}"]
            states = np.asarray(grp["states"], dtype=np.float64)
            actions = np.asarray(grp["actions"], dtype=np.float64)
            model_file = _decode_attr(grp.attrs.get("model_file"))
            has_datagen_info = "datagen_info" in grp
            row = replay_demo(env, states, actions, model_file)
            row.update(
                {
                    "demo_key": demo_key,
                    "used_env_args": env_args,
                    "used_model_file_present": bool(model_file),
                    "reset_state_source": "states[0]+model_file via robomimic env.reset_to",
                    "has_datagen_info": has_datagen_info,
                    "num_steps_in_demo": int(len(actions)),
                }
            )
            records.append(row)

    success_count = sum(1 for r in records if r["replay_success"])
    summary = {
        "dataset_path": str(dataset_path),
        "num_demos_tested": len(records),
        "replay_success_count": success_count,
        "replay_success_rate": float(success_count / len(records)) if records else 0.0,
        "env_name": env_args.get("env_name"),
        "controller_type": env_args.get("env_kwargs", {}).get("controller_configs", {}).get("type"),
        "records": records,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit coffee_preparation source replay sanity")
    parser.add_argument(
        "--dataset",
        default="runs/phygen_coffee_official/prepared_source/coffee_preparation.hdf5",
    )
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument(
        "--output",
        default="runs/phygen_coffee_official/source_replay_sanity.json",
    )
    args = parser.parse_args()

    summary = audit_dataset(args.dataset, num_demos=args.num_demos)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "records"}, indent=2, ensure_ascii=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
