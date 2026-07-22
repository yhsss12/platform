#!/usr/bin/env python3
"""Unified CLI for SAM3 + SAM3D asset pipeline stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from network_bootstrap import apply_network_bootstrap, subprocess_env
from sam3_output_normalizer import write_sam3_manifest
from status_utils import utc_now_iso, write_status

PIPELINE_DIR = Path(__file__).resolve().parent


def _run_script(python_bin: str, script: str, argv: list[str], *, extra_env: dict[str, str] | None = None) -> int:
    env = subprocess_env()
    if extra_env:
        env.update(extra_env)
    cmd = [python_bin, str(PIPELINE_DIR / script), *argv]
    proc = subprocess.run(cmd, env=env)
    return int(proc.returncode)


def cmd_segment(args: argparse.Namespace) -> int:
    argv = [
        "--job-dir",
        args.job_dir,
        "--sam3-root",
        args.sam3_root,
        "--sam3-python",
        args.sam3_python,
        "--image",
        args.image,
        "--confidence-threshold",
        str(args.confidence_threshold),
    ]
    if args.prompt is not None:
        argv.extend(["--prompt", args.prompt])
    if args.text_only:
        argv.append("--text-only")
    for box in args.pos_box or []:
        argv.extend(["--pos-box", box])
    for box in args.neg_box or []:
        argv.extend(["--neg-box", box])
    return _run_script(args.runner_python, "segment_cli.py", argv)


def cmd_reconstruct(args: argparse.Namespace) -> int:
    cutout_index = args.cutout_index
    if cutout_index is None and args.mask_index is not None:
        cutout_index = int(args.mask_index) + 1
    if cutout_index is None:
        print("error: --cutout-index is required", file=sys.stderr)
        return 2

    argv = [
        "--job-dir",
        args.job_dir,
        "--sam3d-root",
        args.sam3d_root,
        "--image",
        args.image,
        "--cutout-index",
        str(cutout_index),
        "--seed",
        str(args.seed),
        "--dinov2-repo",
        args.dinov2_repo,
        "--dinov2-model",
        args.dinov2_model,
        "--moge-model-path",
        args.moge_model_path,
        "--torch-home",
        args.torch_home,
        "--hf-home",
        args.hf_home,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.offline_mode:
        argv.append("--offline-mode")
    if args.prepare_only:
        argv.append("--prepare-only")
    return _run_script(args.sam3d_python, "reconstruct_cli.py", argv)


def cmd_normalize_sam3(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).resolve()
    out = write_sam3_manifest(job_dir)
    print(f"manifest written: {out}")
    return 0


def cmd_export_mujoco(args: argparse.Namespace) -> int:
    argv = [
        "--job-dir",
        args.job_dir,
        "--model-name",
        args.model_name or Path(args.job_dir).resolve().name,
        "--scale-longest",
        str(args.scale_longest),
        "--mass",
        str(args.mass),
        "--collision",
        args.collision,
    ]
    return _run_script(args.runner_python, "mujoco_asset_exporter.py", argv)


def cmd_render_mujoco(args: argparse.Namespace) -> int:
    argv = [
        "--job-dir",
        args.job_dir,
        "--xml-kind",
        args.xml_kind,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--gl",
        args.gl,
    ]
    return _run_script(
        args.runner_python,
        "mujoco_visualizer.py",
        argv,
        extra_env={"MUJOCO_GL": args.gl},
    )


def cmd_mark_failed(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).resolve()
    write_status(
        job_dir,
        status="failed",
        phase=args.phase,
        progress=args.progress,
        error=args.error,
        message=args.message or "job marked failed",
        extra={"markedAt": utc_now_iso()},
    )
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_segment(args)
    if rc != 0:
        return rc
    return cmd_reconstruct(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3D asset pipeline runner")
    parser.add_argument("--runner-python", default=sys.executable)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_segment(p: argparse.ArgumentParser) -> None:
        p.add_argument("--job-dir", required=True)
        p.add_argument("--sam3-root", required=True)
        p.add_argument("--sam3-python", required=True)
        p.add_argument("--image", required=True)
        p.add_argument("--prompt", default=None)
        p.add_argument("--pos-box", action="append", default=[])
        p.add_argument("--neg-box", action="append", default=[])
        p.add_argument("--confidence-threshold", type=float, default=0.05)
        p.add_argument("--text-only", action="store_true")

    def add_common_reconstruct(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sam3d-root", required=True)
        p.add_argument("--sam3d-python", required=True)
        p.add_argument("--cutout-index", type=int, default=None)
        p.add_argument("--mask-index", type=int, default=None)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--prepare-only", action="store_true")
        p.add_argument("--offline-mode", action="store_true")
        p.add_argument("--dinov2-repo", default="")
        p.add_argument("--dinov2-model", default="dinov2_vitl14_reg")
        p.add_argument("--moge-model-path", default="")
        p.add_argument("--torch-home", default="")
        p.add_argument("--hf-home", default="")
        p.add_argument("--timeout-seconds", type=int, default=1800)

    seg = sub.add_parser("segment")
    add_common_segment(seg)
    seg.add_argument("--runner-python", default=sys.executable)
    seg.set_defaults(func=cmd_segment)

    rec = sub.add_parser("reconstruct")
    rec.add_argument("--job-dir", required=True)
    rec.add_argument("--image", default="")
    add_common_reconstruct(rec)
    rec.set_defaults(func=cmd_reconstruct)

    norm = sub.add_parser("normalize-sam3")
    norm.add_argument("--job-dir", required=True)
    norm.set_defaults(func=cmd_normalize_sam3)

    mujoco = sub.add_parser("export-mujoco")
    mujoco.add_argument("--job-dir", required=True)
    mujoco.add_argument("--model-name", default="")
    mujoco.add_argument("--scale-longest", type=float, default=0.12)
    mujoco.add_argument("--mass", type=float, default=0.5)
    mujoco.add_argument("--collision", choices=["convex_hull", "visual", "none"], default="visual")
    mujoco.add_argument("--runner-python", default=sys.executable)
    mujoco.set_defaults(func=cmd_export_mujoco)

    render = sub.add_parser("render-mujoco")
    render.add_argument("--job-dir", required=True)
    render.add_argument("--xml-kind", choices=["preview", "physics"], default="preview")
    render.add_argument("--width", type=int, default=960)
    render.add_argument("--height", type=int, default=720)
    render.add_argument("--gl", default="egl")
    render.add_argument("--runner-python", default=sys.executable)
    render.set_defaults(func=cmd_render_mujoco)

    all_p = sub.add_parser("all")
    add_common_segment(all_p)
    add_common_reconstruct(all_p)
    all_p.add_argument("--runner-python", default=sys.executable)
    all_p.set_defaults(func=cmd_all)

    mark = sub.add_parser("mark-failed")
    mark.add_argument("--job-dir", required=True)
    mark.add_argument("--error", required=True)
    mark.add_argument("--phase", default="sam3d_reconstruct")
    mark.add_argument("--progress", type=float, default=0.65)
    mark.add_argument("--message", default=None)
    mark.set_defaults(func=cmd_mark_failed)

    args = parser.parse_args()
    apply_network_bootstrap()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
