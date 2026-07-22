#!/usr/bin/env python3
"""Standalone joint-space Diffusion Policy cable_threading backend test pipeline."""
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

from joint_space_dp_utils import (  # noqa: E402
    CABLE_MVP_ROOT,
    build_joint_space_hdf5,
    inspect_joint_position_controller,
    replay_joint_sanity,
    run_joint_eval_smoke,
)

DEFAULT_SOURCE_HDF5 = (
    PROJECT_ROOT
    / "runs"
    / "cable_threading"
    / "jobs"
    / "ct_gen_20260623_211017_3d7f"
    / "datasets"
    / "dataset.hdf5"
)
DEFAULT_JOINT_HDF5 = (
    PROJECT_ROOT
    / "runs"
    / "cable_threading"
    / "jobs"
    / "ct_gen_20260624_joint_space_replay"
    / "datasets"
    / "dataset.hdf5"
)
DEFAULT_JOINT_HDF5_FULL = (
    PROJECT_ROOT
    / "runs"
    / "cable_threading"
    / "jobs"
    / "ct_gen_20260624_joint_space_replay_full"
    / "datasets"
    / "dataset.hdf5"
)
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "runs" / "standalone_dp_joint_space_tests"
JOINT_DP_CONFIG = (
    CABLE_MVP_ROOT
    / "examples"
    / "cable_threading"
    / "dp_configs"
    / "cable_threading_joint_obs_joint_action.yaml"
)
TRAIN_DP_SCRIPT = CABLE_MVP_ROOT / "examples" / "cable_threading" / "train_dp.py"


def _timestamp_dir(suffix: str = "") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}{suffix}" if suffix else stamp
    return DEFAULT_OUTPUT_BASE / name


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_inspect_controller(out_dir: Path) -> dict[str, Any]:
    info = inspect_joint_position_controller()
    _write_json(out_dir / "controller_inspect.json", info)
    return {"ok": True, "controller_inspect": info}


def cmd_build_joint_dataset(
    out_dir: Path,
    *,
    source_hdf5: Path,
    joint_hdf5: Path,
    max_demos: int | None,
) -> dict[str, Any]:
    report = build_joint_space_hdf5(
        source_hdf5,
        joint_hdf5,
        max_demos=max_demos,
        log_path=out_dir / "dataset_build.log",
    )
    _write_json(out_dir / "dataset_build_report.json", report)
    return report


def cmd_replay_joint_sanity(
    out_dir: Path,
    *,
    joint_hdf5: Path,
    max_demos: int,
    full_sample: bool,
) -> dict[str, Any]:
    if full_sample:
        report = replay_joint_sanity(
            joint_hdf5,
            sample_first=10,
            sample_random=10,
            sample_success=10,
            min_successful_replays=20,
        )
    else:
        report = replay_joint_sanity(joint_hdf5, max_demos=max_demos)
    _write_json(out_dir / "replay_joint_sanity.json", report)
    return report


def cmd_train(
    out_dir: Path,
    *,
    joint_hdf5: Path,
    epochs: int,
    batch_size: int,
    device: str,
    train_subdir: str = "train",
) -> dict[str, Any]:
    train_out = out_dir / train_subdir
    train_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAIN_DP_SCRIPT),
        "--dataset",
        str(joint_hdf5),
        "--out-dir",
        str(train_out),
        "--config",
        str(JOINT_DP_CONFIG),
        "--num-epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=str(CABLE_MVP_ROOT), stdout=log_file, stderr=subprocess.STDOUT)
    ckpt = train_out / "checkpoints" / "model_final.pt"
    diag_path = train_out / "config" / "train_diagnostics.json"
    result = {
        "ok": proc.returncode == 0 and ckpt.is_file(),
        "exit_code": proc.returncode,
        "checkpoint": str(ckpt) if ckpt.is_file() else None,
        "log_path": str(log_path),
        "train_diagnostics_path": str(diag_path) if diag_path.is_file() else None,
    }
    if ckpt.is_file():
        import torch

        payload = torch.load(ckpt, map_location="cpu")
        result["train_config"] = payload.get("train_config") or {}
        if diag_path.is_file():
            result["train_diagnostics"] = json.loads(diag_path.read_text(encoding="utf-8"))
    return result


def cmd_eval(
    out_dir: Path,
    *,
    checkpoint: Path,
    episodes: int,
    device: str,
    horizon: int,
    live_video_out: Path | None,
    live_frame_dir: Path | None,
    live_save_frames: bool,
) -> dict[str, Any]:
    eval_out = out_dir / "eval"
    log_path = out_dir / "eval.log"
    run_log_path = out_dir / "run.log"
    videos_dir = eval_out / "videos"
    if live_video_out is None:
        live_video_out = videos_dir / "eval.mp4"
    if live_frame_dir is None:
        live_frame_dir = videos_dir / "live"
    try:
        report = run_joint_eval_smoke(
            checkpoint,
            episodes=episodes,
            device=device,
            horizon=horizon,
            out_dir=eval_out,
            live_video_out=live_video_out,
            live_frame_dir=live_frame_dir,
            live_save_frames=live_save_frames,
        )
        report["log_path"] = str(log_path)
        _write_json(out_dir / "eval_report.json", report)
        log_text = json.dumps(report, indent=2, default=str)
        log_path.write_text(log_text, encoding="utf-8")
        run_log_path.write_text(log_text, encoding="utf-8")
        return report
    except Exception as exc:
        err = {"ok": False, "error": str(exc)}
        log_path.write_text(str(exc), encoding="utf-8")
        run_log_path.write_text(str(exc), encoding="utf-8")
        _write_json(out_dir / "eval_report.json", err)
        return err


def run_pipeline(args: argparse.Namespace) -> int:
    mode = args.mode
    full_modes = {
        "build-joint-dataset-full",
        "replay-joint-sanity-full",
        "train-200ep",
        "eval-10ep",
        "validate-full",
    }
    if mode in full_modes:
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _timestamp_dir("_full")
    else:
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _timestamp_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_hdf5 = Path(args.source_hdf5).expanduser().resolve()
    if mode in full_modes or args.use_full_dataset:
        joint_hdf5 = Path(args.joint_hdf5_full).expanduser().resolve()
    else:
        joint_hdf5 = Path(args.joint_hdf5).expanduser().resolve()

    result: dict[str, Any] = {
        "out_dir": str(out_dir),
        "mode": mode,
        "source_hdf5": str(source_hdf5),
        "joint_hdf5": str(joint_hdf5),
        "failure_step": None,
        "pipeline_ok": False,
        "smoke_passed": False,
    }

    def fail(step: str, detail: dict[str, Any]) -> int:
        result["failure_step"] = step
        result["detail"] = detail
        _write_json(out_dir / "result.json", result)
        _write_summary(
            out_dir / "summary.md",
            [f"# Joint-space DP test FAILED at `{step}`", "", "```json", json.dumps(detail, indent=2), "```"],
        )
        print(f"FAILED at {step}: {detail}")
        return 1

    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if getattr(args, "checkpoint", None)
        else out_dir / "train_200ep" / "checkpoints" / "model_final.pt"
    )

    if mode in {"inspect-controller", "all", "validate-full"}:
        inspect_result = cmd_inspect_controller(out_dir)
        result["controller_inspect"] = inspect_result["controller_inspect"]
        if inspect_result["controller_inspect"].get("action_dim") != 8:
            return fail("inspect-controller", inspect_result)

    if mode in {"build-joint-dataset", "all"}:
        build_result = cmd_build_joint_dataset(
            out_dir,
            source_hdf5=source_hdf5,
            joint_hdf5=joint_hdf5,
            max_demos=args.max_demos if args.max_demos > 0 else None,
        )
        result["dataset_build"] = build_result
        if not build_result.get("ok"):
            return fail("build-joint-dataset", build_result)

    if mode == "build-joint-dataset-full" or (mode == "validate-full" and "dataset_build" not in result):
        build_result = cmd_build_joint_dataset(
            out_dir,
            source_hdf5=source_hdf5,
            joint_hdf5=joint_hdf5,
            max_demos=None,
        )
        result["dataset_build"] = build_result
        if not build_result.get("ok"):
            return fail("build-joint-dataset-full", build_result)

    if mode in {"replay-joint-sanity", "all"}:
        sanity = cmd_replay_joint_sanity(
            out_dir,
            joint_hdf5=joint_hdf5,
            max_demos=min(args.max_demos, 5) if args.max_demos > 0 else 5,
            full_sample=False,
        )
        result["replay_joint_sanity"] = sanity
        if not sanity.get("ok"):
            return fail("replay-joint-sanity", sanity)

    if mode in {"replay-joint-sanity-full", "validate-full"} and "replay_joint_sanity" not in result:
        sanity = cmd_replay_joint_sanity(
            out_dir,
            joint_hdf5=joint_hdf5,
            max_demos=0,
            full_sample=True,
        )
        result["replay_joint_sanity"] = sanity
        if not sanity.get("ok"):
            return fail("replay-joint-sanity-full", sanity)

    if mode in {"train-smoke", "all"}:
        train_result = cmd_train(
            out_dir,
            joint_hdf5=joint_hdf5,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            train_subdir="train",
        )
        result["train_smoke"] = train_result
        if not train_result.get("ok"):
            return fail("train-smoke", train_result)
        checkpoint_path = Path(train_result["checkpoint"])

    if mode in {"train-200ep", "validate-full"} and "train_200ep" not in result:
        train_result = cmd_train(
            out_dir,
            joint_hdf5=joint_hdf5,
            epochs=200,
            batch_size=args.batch_size if args.batch_size != 64 else 128,
            device=args.device,
            train_subdir="train_200ep",
        )
        result["train_200ep"] = train_result
        if not train_result.get("ok"):
            return fail("train-200ep", train_result)
        checkpoint_path = Path(train_result["checkpoint"])

    eval_modes = {"eval-smoke", "eval-10ep", "all", "validate-full"}
    if mode in eval_modes:
        if not checkpoint_path.is_file():
            return fail(mode, {"error": f"checkpoint missing: {checkpoint_path}"})
        episodes = args.eval_episodes
        if mode in {"eval-10ep", "validate-full"}:
            episodes = 10
        live_video_out = Path(args.live_video_out).expanduser().resolve() if args.live_video_out else None
        live_frame_dir = Path(args.live_frame_dir).expanduser().resolve() if args.live_frame_dir else None
        eval_result = cmd_eval(
            out_dir,
            checkpoint=checkpoint_path,
            episodes=episodes,
            device=args.device,
            horizon=int(args.eval_horizon),
            live_video_out=live_video_out,
            live_frame_dir=live_frame_dir,
            live_save_frames=bool(args.live_save_frames),
        )
        key = "eval_10ep" if mode in {"eval-10ep", "validate-full"} else "eval_smoke"
        result[key] = eval_result
        if not eval_result.get("ok"):
            return fail(mode, eval_result)

    result["pipeline_ok"] = True
    result["smoke_passed"] = True
    _write_json(out_dir / "result.json", result)
    _write_summary(
        out_dir / "summary.md",
        [
            "# Joint-space DP test",
            "",
            f"- mode: `{mode}`",
            f"- out_dir: `{out_dir}`",
            f"- joint_hdf5: `{joint_hdf5}`",
            f"- pipeline_ok: **{result['pipeline_ok']}**",
        ],
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Joint-space DP cable_threading backend test")
    parser.add_argument(
        "mode",
        choices=[
            "inspect-controller",
            "build-joint-dataset",
            "build-joint-dataset-full",
            "replay-joint-sanity",
            "replay-joint-sanity-full",
            "train-smoke",
            "train-200ep",
            "eval-smoke",
            "eval-10ep",
            "all",
            "validate-full",
        ],
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--source-hdf5", default=str(DEFAULT_SOURCE_HDF5))
    parser.add_argument("--joint-hdf5", default=str(DEFAULT_JOINT_HDF5))
    parser.add_argument("--joint-hdf5-full", default=str(DEFAULT_JOINT_HDF5_FULL))
    parser.add_argument("--use-full-dataset", action="store_true")
    parser.add_argument("--max-demos", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument(
        "--eval-horizon",
        type=int,
        default=1200,
        help="Max sim steps per eval episode (default 1200; data collection uses 600)",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for eval modes")
    parser.add_argument("--live-video-out", default=None)
    parser.add_argument("--live-frame-dir", default=None)
    parser.add_argument("--live-save-frames", action="store_true", default=True)
    parser.add_argument("--no-live-save-frames", action="store_false", dest="live_save_frames")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
