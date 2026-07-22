#!/usr/bin/env python3
"""Standalone Diffusion Policy cable_threading test (no platform UI / DB / training_service)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_ROOT = SCRIPT_PATH.parents[2]
SCRIPTS_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = BACKEND_ROOT.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from standalone_dp_eval_utils import (  # noqa: E402
    DATASET_MISSING_JOINT_POS_MESSAGE,
    STANDALONE_DP_OBS_SCHEMA,
    inspect_hdf5_dataset_schema,
    run_standalone_dp_eval,
    validate_dataset_joint_pos_schema,
    write_standalone_train_config_yaml,
)
CABLE_MVP_ROOT = PROJECT_ROOT / "integrations" / "CableThreadingMVP"
TRAIN_DP_SCRIPT = CABLE_MVP_ROOT / "examples" / "cable_threading" / "train_dp.py"
RUN_PY = CABLE_MVP_ROOT / "run.py"
DEFAULT_DP_CONFIG = CABLE_MVP_ROOT / "examples" / "cable_threading" / "dp_configs" / "cable_threading.yaml"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "runs" / "standalone_dp_tests"
DEFAULT_CHECKPOINT_REL = Path("checkpoints") / "model_final.pt"

EXPECTED_STANDALONE_DP_SCHEMA: dict[str, Any] = dict(STANDALONE_DP_OBS_SCHEMA)


def _norm_keys(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def resolve_device(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_project_paths() -> dict[str, Path]:
    return {
        "project_root": PROJECT_ROOT,
        "cable_mvp_root": CABLE_MVP_ROOT,
        "train_dp_script": TRAIN_DP_SCRIPT,
        "run_py": RUN_PY,
        "default_dp_config": DEFAULT_DP_CONFIG,
    }


def ensure_integration_paths() -> None:
    missing: list[str] = []
    labels = {
        "CableThreadingMVP/run.py": RUN_PY,
        "examples/cable_threading/train_dp.py": TRAIN_DP_SCRIPT,
        "examples/cable_threading/dp_configs/cable_threading.yaml": DEFAULT_DP_CONFIG,
    }
    for label, path in labels.items():
        if not path.is_file():
            missing.append(f"  - {label}\n    expected: {path}")
    if missing:
        raise SystemExit(
            "Required CableThreadingMVP integration files were not found.\n"
            + "\n".join(missing)
            + f"\nProject root: {PROJECT_ROOT}"
        )


def validate_dataset_paths(dataset_paths: list[Path]) -> dict[str, Any]:
    result = validate_dataset_joint_pos_schema(dataset_paths)
    return result


def _resolved_low_dim_dim(train_config: dict[str, Any]) -> int:
    explicit = train_config.get("low_dim_dim")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    return 7 + 2


def inspect_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Read-only checkpoint validation for cable_threading DP checkpoints."""
    result: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "path": str(checkpoint_path),
        "train_config": {},
    }
    if not checkpoint_path.is_file():
        result["ok"] = False
        result["errors"].append(f"checkpoint not found: {checkpoint_path}")
        return result

    try:
        import torch
    except ImportError as exc:
        result["ok"] = False
        result["errors"].append(f"torch is required to inspect checkpoint: {exc}")
        return result

    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"failed to load checkpoint: {exc}")
        return result

    if not isinstance(payload, dict):
        result["ok"] = False
        result["errors"].append("checkpoint payload must be a dict")
        return result

    for key in ("state_dict", "normalizer", "train_config"):
        if key not in payload:
            result["ok"] = False
            result["errors"].append(f"missing {key}")

    train_config = payload.get("train_config")
    if not isinstance(train_config, dict):
        result["ok"] = False
        result["errors"].append("train_config must be a dict")
        return result

    result["train_config"] = train_config
    action_dim = train_config.get("action_dim")
    low_dim_dim = _resolved_low_dim_dim(train_config)
    image_keys = _norm_keys(train_config.get("image_keys"))
    low_dim_keys = _norm_keys(train_config.get("low_dim_keys"))

    checks: list[tuple[str, Any, Any]] = [
        ("action_dim", action_dim, EXPECTED_STANDALONE_DP_SCHEMA["action_dim"]),
        ("low_dim_dim", low_dim_dim, EXPECTED_STANDALONE_DP_SCHEMA["low_dim_dim"]),
        ("image_keys", image_keys, EXPECTED_STANDALONE_DP_SCHEMA["image_keys"]),
        ("low_dim_keys", low_dim_keys, EXPECTED_STANDALONE_DP_SCHEMA["low_dim_keys"]),
    ]
    for field, actual, expected in checks:
        if actual != expected:
            result["ok"] = False
            result["errors"].append(f"{field} mismatch: expected {expected!r}, got {actual!r}")

    return result


def parse_eval_results(eval_results_dir: Path) -> dict[str, Any]:
    """Parse run.py eval outputs into a normalized result payload."""
    aggregate_path = eval_results_dir / "aggregate_result.json"
    per_episode_path = eval_results_dir / "per_episode_results.json"
    results_json_path = eval_results_dir / "eval.results.json"

    payload: dict[str, Any] = {
        "aggregate_path": str(aggregate_path),
        "per_episode_path": str(per_episode_path),
        "results_json_path": str(results_json_path),
        "success_rate": 0.0,
        "ever_success_rate": 0.0,
        "episodes": [],
        "aggregate": {},
    }

    aggregate: dict[str, Any] = {}
    if aggregate_path.is_file():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    elif results_json_path.is_file():
        raw = json.loads(results_json_path.read_text(encoding="utf-8"))
        aggregate = raw.get("aggregate") or {}
        if not aggregate and isinstance(raw.get("success_rate"), (int, float)):
            aggregate = {"final_success_rate": raw["success_rate"]}
    payload["aggregate"] = aggregate
    payload["success_rate"] = float(
        aggregate.get("final_success_rate", aggregate.get("success_rate", 0.0)) or 0.0
    )
    payload["ever_success_rate"] = float(aggregate.get("ever_success_rate", 0.0) or 0.0)

    rows: list[dict[str, Any]] = []
    if per_episode_path.is_file():
        loaded = json.loads(per_episode_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows = [row for row in loaded if isinstance(row, dict)]
    elif results_json_path.is_file():
        loaded = json.loads(results_json_path.read_text(encoding="utf-8"))
        episode_rows = loaded.get("episodes")
        if isinstance(episode_rows, list):
            rows = [row for row in episode_rows if isinstance(row, dict)]

    payload["episodes"] = [
        {
            "episode": row.get("episode"),
            "final_success": bool(row.get("final_success")),
            "ever_success": bool(row.get("ever_success")),
            "trajectory_length": row.get("steps"),
            "completion_time": None,
            "return": row.get("return"),
            "thread_completion_final": row.get("thread_completion_final"),
            "endpoint_goal_error_final": row.get("endpoint_goal_error_final"),
            "straightness_error_final": row.get("straightness_error_final"),
            "action_smoothness": None,
            "jitter": None,
        }
        for row in rows
    ]
    return payload


def assess_eval_only_result(
    *,
    eval_exit_code: int,
    episodes_requested: int,
    eval_payload: dict[str, Any],
    checkpoint_validation: dict[str, Any],
    eval_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = len(eval_payload.get("episodes") or [])
    obs_failed = bool((eval_raw or {}).get("obs_validation_failed"))
    eval_runnable = (
        checkpoint_validation.get("ok") is True
        and eval_exit_code == 0
        and episodes_requested > 0
        and completed >= episodes_requested
        and not obs_failed
    )
    success_rate = float(eval_payload.get("success_rate") or 0.0)
    assessment = "评测未跑通"
    dp_can_complete_task = False
    failure_step: str | None = None

    if not checkpoint_validation.get("ok"):
        failure_step = "checkpoint_validation"
        assessment = "checkpoint 校验失败，未进入评测"
    elif obs_failed:
        failure_step = "eval_obs_validation"
        assessment = "评测未进入 rollout（观测 schema 校验失败）"
    elif eval_exit_code != 0:
        failure_step = "eval"
        assessment = f"评测进程失败（exit_code={eval_exit_code}）"
    elif episodes_requested <= 0:
        failure_step = "arguments"
        assessment = "episodes 必须大于 0"
    elif completed <= 0:
        failure_step = "eval_no_rollout"
        assessment = "评测未进入 rollout"
    elif completed < episodes_requested:
        failure_step = "eval_no_rollout"
        assessment = f"评测仅完成 {completed}/{episodes_requested} 个 episode"
    elif success_rate >= 0.6:
        assessment = "DP 能较稳定完成该任务"
        dp_can_complete_task = True
    elif success_rate > 0:
        assessment = "DP 链路可用，当前 checkpoint 已具备完成任务能力"
        dp_can_complete_task = True
    else:
        assessment = "链路可用，但当前 checkpoint 未完成任务"

    return {
        "eval_runnable": eval_runnable,
        "success_rate": success_rate,
        "dp_can_complete_task": dp_can_complete_task,
        "assessment": assessment,
        "failure_step": failure_step,
        "pipeline_ok": eval_runnable,
    }


def assess_train_and_eval_smoke(
    *,
    train_exit_code: int,
    eval_exit_code: int,
    checkpoint_path: Path,
    episodes_requested: int,
    eval_payload: dict[str, Any],
    eval_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_exists = checkpoint_path.is_file()
    completed = len(eval_payload.get("episodes") or [])
    obs_failed = bool((eval_raw or {}).get("obs_validation_failed"))
    train_ok = train_exit_code == 0 and checkpoint_exists
    eval_rollout_ok = (
        eval_exit_code == 0 and not obs_failed and completed >= episodes_requested and episodes_requested > 0
    )
    smoke_passed = train_ok and eval_rollout_ok
    failure_step: str | None = None
    if train_exit_code != 0:
        failure_step = "train"
    elif not checkpoint_exists:
        failure_step = "checkpoint_generation"
    elif obs_failed:
        failure_step = "eval_obs_validation"
    elif eval_exit_code != 0:
        failure_step = "eval"
    elif completed <= 0:
        failure_step = "eval_no_rollout"
    elif completed < episodes_requested:
        failure_step = "eval_no_rollout"

    if smoke_passed:
        assessment = "train-and-eval smoke 通过"
    elif train_ok and eval_rollout_ok is False and completed > 0:
        assessment = "train-and-eval smoke 未通过（episode 未完整）"
    else:
        assessment = "train-and-eval smoke 未通过"

    return {
        "smoke_passed": smoke_passed,
        "failure_step": failure_step if not smoke_passed else None,
        "pipeline_ok": smoke_passed,
        "eval_runnable": eval_rollout_ok,
        "assessment": assessment,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Diffusion Policy cable_threading train/eval verification (no platform DB)."
    )
    parser.add_argument(
        "--mode",
        choices=["eval-only", "train-and-eval"],
        required=True,
        help="eval-only: evaluate an existing checkpoint; train-and-eval: smoke train then eval",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (eval-only)")
    parser.add_argument("--dataset", type=str, default=None, help="Single HDF5 dataset path")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated HDF5 dataset paths")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (train-and-eval)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (train-and-eval)")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--debug", action="store_true", help="Use fast debug training settings")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=600, help="Simulation horizon passed to run.py eval")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Base output directory; a timestamped run dir is created inside",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_DP_CONFIG),
        help="DP yaml config for train-and-eval mode",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _resolve_dataset_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.datasets:
        paths.extend(
            Path(part.strip()).expanduser().resolve()
            for part in args.datasets.split(",")
            if part.strip()
        )
    elif args.dataset:
        paths.append(Path(args.dataset).expanduser().resolve())
    return paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_command(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(cmd)
    _write_text(log_path.with_name(log_path.stem + "_cmd.txt"), rendered + "\n")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {rendered}\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def _make_run_dir(base_output_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_summary_md(run_dir: Path, result: dict[str, Any]) -> None:
    lines = [
        "# DP Cable Threading Standalone Test",
        "",
        f"- Mode: `{result.get('mode')}`",
        f"- Run directory: `{result.get('run_dir')}`",
        f"- Device requested: `{result.get('device_requested')}`",
        f"- Device resolved: `{result.get('device_resolved')}`",
        f"- Episodes: `{result.get('episodes')}`",
        f"- Max steps: `{result.get('max_steps')}`",
        "",
    ]
    dataset_paths = result.get("dataset_paths") or []
    if dataset_paths:
        lines.append("## Dataset")
        for item in dataset_paths:
            lines.append(f"- `{item}`")
        lines.append("")

    checkpoint_path = result.get("checkpoint_path")
    if checkpoint_path:
        lines.extend(["## Checkpoint", f"- `{checkpoint_path}`", ""])

    lines.extend(
        [
            "## Result",
            f"- Pipeline OK: `{result.get('pipeline_ok')}`",
            f"- Success rate: `{result.get('success_rate')}`",
            f"- Assessment: {result.get('assessment')}",
        ]
    )
    if result.get("mode") == "eval-only":
        lines.append(f"- Eval runnable: `{result.get('eval_runnable')}`")
        lines.append(f"- DP can complete task: `{result.get('dp_can_complete_task')}`")
    else:
        lines.append(f"- Smoke passed: `{result.get('smoke_passed')}`")
        lines.append(f"- Eval runnable: `{result.get('eval_runnable')}`")

    if result.get("failure_step") in {"eval_no_rollout", "eval_obs_validation"}:
        lines.append("- 评测未进入 rollout。")

    failure_step = result.get("failure_step")
    if failure_step:
        lines.extend(["", "## Failure", f"- Failed at step: `{failure_step}`"])
        errors = result.get("errors") or []
        for err in errors:
            lines.append(f"- {err}")

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- result.json: `{run_dir / 'result.json'}`",
            f"- train.log: `{run_dir / 'train.log'}`",
            f"- eval.log: `{run_dir / 'eval.log'}`",
        ]
    )
    eval_log = result.get("eval_log_path")
    if eval_log:
        lines.append(f"- eval log path: `{eval_log}`")
    _write_text(run_dir / "summary.md", "\n".join(lines) + "\n")


def _run_standalone_eval(
    *,
    checkpoint_path: Path,
    episodes: int,
    device: str,
    max_steps: int,
    run_dir: Path,
) -> dict[str, Any]:
    eval_results_dir = run_dir / "eval_results"
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    command = (
        f"standalone in-process eval via standalone_dp_eval_utils.run_standalone_dp_eval "
        f"checkpoint={checkpoint_path} episodes={episodes} device={device} max_steps={max_steps}"
    )
    _write_text(run_dir / "command_eval.txt", command + "\n")
    eval_raw = run_standalone_dp_eval(
        checkpoint_path=checkpoint_path,
        episodes=int(episodes),
        device=device,
        max_steps=int(max_steps),
        output_dir=eval_results_dir,
        cable_mvp_root=CABLE_MVP_ROOT,
    )
    eval_payload = parse_eval_results(eval_results_dir)
    if eval_raw.get("episodes"):
        eval_payload["episodes"] = [
            {
                "episode": row.get("episode"),
                "final_success": bool(row.get("final_success")),
                "ever_success": bool(row.get("ever_success")),
                "trajectory_length": row.get("steps"),
                "completion_time": None,
                "return": row.get("return"),
                "thread_completion_final": row.get("thread_completion_final"),
                "endpoint_goal_error_final": row.get("endpoint_goal_error_final"),
                "straightness_error_final": row.get("straightness_error_final"),
                "action_smoothness": None,
                "jitter": None,
            }
            for row in eval_raw["episodes"]
            if isinstance(row, dict)
        ]
        eval_payload["success_rate"] = float(eval_raw.get("success_rate") or 0.0)
        eval_payload["ever_success_rate"] = float(eval_raw.get("ever_success_rate") or 0.0)
    return {
        "exit_code": int(eval_raw.get("exit_code", 1)),
        "log_path": str(run_dir / "eval.log"),
        "command_path": str(run_dir / "command_eval.txt"),
        "raw": eval_raw,
        **eval_payload,
    }


def run_eval_only(args: argparse.Namespace, run_dir: Path, device: str) -> dict[str, Any]:
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for eval-only mode")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_validation = inspect_checkpoint(checkpoint_path)
    result: dict[str, Any] = {
        "mode": "eval-only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "device_requested": args.device,
        "device_resolved": device,
        "dataset_paths": [],
        "checkpoint_path": str(checkpoint_path),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "checkpoint_validation": checkpoint_validation,
        "errors": list(checkpoint_validation.get("errors") or []),
    }

    if not checkpoint_validation.get("ok"):
        assessment = assess_eval_only_result(
            eval_exit_code=1,
            episodes_requested=int(args.episodes),
            eval_payload={"episodes": [], "success_rate": 0.0},
            checkpoint_validation=checkpoint_validation,
        )
        result.update(assessment)
        return result

    eval_bundle = _run_standalone_eval(
        checkpoint_path=checkpoint_path,
        episodes=int(args.episodes),
        device=device,
        max_steps=int(args.max_steps),
        run_dir=run_dir,
    )
    eval_exit_code = int(eval_bundle["exit_code"])
    eval_payload = {k: v for k, v in eval_bundle.items() if k not in {"exit_code", "raw"}}
    eval_raw = eval_bundle.get("raw") or {}
    assessment = assess_eval_only_result(
        eval_exit_code=eval_exit_code,
        episodes_requested=int(args.episodes),
        eval_payload=eval_payload,
        checkpoint_validation=checkpoint_validation,
        eval_raw=eval_raw,
    )
    result.update(assessment)
    result["eval"] = {
        "exit_code": eval_exit_code,
        "log_path": str(run_dir / "eval.log"),
        "command_path": str(run_dir / "command_eval.txt"),
        "backend": "standalone_inprocess",
        **eval_payload,
    }
    result["eval_log_path"] = str(run_dir / "eval.log")
    if eval_raw.get("errors"):
        result.setdefault("errors", []).extend(eval_raw["errors"])
    if eval_exit_code != 0:
        result.setdefault("errors", []).append(f"eval failed with exit_code={eval_exit_code}")
    return result


def run_train_and_eval(args: argparse.Namespace, run_dir: Path, device: str) -> dict[str, Any]:
    dataset_paths = _resolve_dataset_paths(args)
    dataset_validation = validate_dataset_paths(dataset_paths)
    result: dict[str, Any] = {
        "mode": "train-and-eval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "device_requested": args.device,
        "device_resolved": device,
        "dataset_paths": dataset_validation.get("paths") or [],
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "debug": bool(args.debug),
        "errors": list(dataset_validation.get("errors") or []),
        "success_rate": 0.0,
        "dataset_schema": dataset_validation,
        "train_image_keys": EXPECTED_STANDALONE_DP_SCHEMA["image_keys"],
        "train_low_dim_keys": EXPECTED_STANDALONE_DP_SCHEMA["low_dim_keys"],
    }

    if not dataset_validation.get("ok"):
        smoke = assess_train_and_eval_smoke(
            train_exit_code=1,
            eval_exit_code=1,
            checkpoint_path=run_dir / "train_output" / DEFAULT_CHECKPOINT_REL,
            episodes_requested=int(args.episodes),
            eval_payload={"episodes": []},
        )
        result.update(smoke)
        result["failure_step"] = "dataset_validation"
        if any(DATASET_MISSING_JOINT_POS_MESSAGE in err for err in result["errors"]):
            result["assessment"] = DATASET_MISSING_JOINT_POS_MESSAGE
        else:
            result["assessment"] = "dataset 校验失败，未进入训练"
        return result

    train_output_dir = run_dir / "train_output"
    train_output_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_standalone_train_config_yaml(run_dir)

    train_cmd = [
        sys.executable,
        str(TRAIN_DP_SCRIPT),
        "--out-dir",
        str(train_output_dir),
        "--config",
        str(config_path),
        "--device",
        device,
        "--num-epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
    ]
    if len(dataset_paths) == 1:
        train_cmd.extend(["--dataset", str(dataset_paths[0])])
    else:
        train_cmd.extend(["--datasets", ",".join(str(path) for path in dataset_paths)])
    if args.debug:
        train_cmd.append("--debug")

    _write_text(run_dir / "command_train.txt", " ".join(train_cmd) + "\n")
    train_exit_code = _run_command(train_cmd, cwd=CABLE_MVP_ROOT, log_path=run_dir / "train.log")
    checkpoint_path = train_output_dir / DEFAULT_CHECKPOINT_REL
    checkpoint_validation = inspect_checkpoint(checkpoint_path) if checkpoint_path.is_file() else {
        "ok": False,
        "errors": [f"checkpoint not generated: {checkpoint_path}"],
        "warnings": [],
        "path": str(checkpoint_path),
        "train_config": {},
    }
    result["train"] = {
        "exit_code": train_exit_code,
        "log_path": str(run_dir / "train.log"),
        "command_path": str(run_dir / "command_train.txt"),
        "output_dir": str(train_output_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_validation": checkpoint_validation,
    }
    result["checkpoint_path"] = str(checkpoint_path)
    result["checkpoint_validation"] = checkpoint_validation
    result["eval_image_keys"] = _norm_keys(
        (checkpoint_validation.get("train_config") or {}).get("image_keys")
    ) or list(EXPECTED_STANDALONE_DP_SCHEMA["image_keys"])
    result["eval_low_dim_keys"] = _norm_keys(
        (checkpoint_validation.get("train_config") or {}).get("low_dim_keys")
    ) or list(EXPECTED_STANDALONE_DP_SCHEMA["low_dim_keys"])

    if train_exit_code != 0 or not checkpoint_path.is_file():
        smoke = assess_train_and_eval_smoke(
            train_exit_code=train_exit_code,
            eval_exit_code=1,
            checkpoint_path=checkpoint_path,
            episodes_requested=int(args.episodes),
            eval_payload={"episodes": []},
        )
        result.update(smoke)
        if train_exit_code != 0:
            result.setdefault("errors", []).append(f"train failed with exit_code={train_exit_code}")
        else:
            result.setdefault("errors", []).append(f"checkpoint missing: {checkpoint_path}")
        return result

    eval_bundle = _run_standalone_eval(
        checkpoint_path=checkpoint_path,
        episodes=int(args.episodes),
        device=device,
        max_steps=int(args.max_steps),
        run_dir=run_dir,
    )
    eval_exit_code = int(eval_bundle["exit_code"])
    eval_payload = {k: v for k, v in eval_bundle.items() if k not in {"exit_code", "raw"}}
    eval_raw = eval_bundle.get("raw") or {}
    result["success_rate"] = float(eval_payload.get("success_rate") or 0.0)
    result["eval"] = {
        "exit_code": eval_exit_code,
        "log_path": str(run_dir / "eval.log"),
        "command_path": str(run_dir / "command_eval.txt"),
        "backend": "standalone_inprocess",
        **eval_payload,
    }
    result["eval_log_path"] = str(run_dir / "eval.log")

    smoke = assess_train_and_eval_smoke(
        train_exit_code=train_exit_code,
        eval_exit_code=eval_exit_code,
        checkpoint_path=checkpoint_path,
        episodes_requested=int(args.episodes),
        eval_payload=eval_payload,
        eval_raw=eval_raw,
    )
    result.update(smoke)
    if eval_raw.get("errors"):
        result.setdefault("errors", []).extend(eval_raw["errors"])
    if eval_exit_code != 0:
        result.setdefault("errors", []).append(f"eval failed with exit_code={eval_exit_code}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_integration_paths()

    base_output_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = _make_run_dir(base_output_dir)
    device = resolve_device(args.device)
    print(f"[standalone-dp-test] mode={args.mode} device={device} run_dir={run_dir}")

    if args.mode == "eval-only":
        result = run_eval_only(args, run_dir, device)
    else:
        result = run_train_and_eval(args, run_dir, device)

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_summary_md(run_dir, result)

    print(f"[standalone-dp-test] result.json: {result_path}")
    print(f"[standalone-dp-test] summary.md: {run_dir / 'summary.md'}")
    print(f"[standalone-dp-test] assessment: {result.get('assessment')}")
    print(f"[standalone-dp-test] success_rate: {result.get('success_rate')}")
    print(f"[standalone-dp-test] pipeline_ok: {result.get('pipeline_ok')}")

    if result.get("pipeline_ok"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
