#!/usr/bin/env python3
"""Run SAM3 segmentation for an asset pipeline job (non-interactive)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from network_bootstrap import apply_network_bootstrap, subprocess_env
from sam3_output_normalizer import write_sam3_manifest
from status_utils import append_log, merge_job_json, read_json, utc_now_iso, write_status


def _parse_box(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"box must have 4 values x0,y0,x1,y1, got: {raw!r}")
    return [float(p) for p in parts]


def _build_sam3_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.sam3_python),
        "scripts/run_sam3_box_select.py",
        "--image",
        str(Path(args.image).resolve()),
        "--out",
        str(Path(args.job_dir).resolve() / "sam3"),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--no-neg-interactive",
    ]
    # Always pass --prompt to avoid upstream stdin prompt when args.prompt is None.
    cmd.extend(["--prompt", args.prompt or ""])
    if args.text_only:
        cmd.append("--text-only")
    for box in args.pos_box or []:
        cmd.extend(["--pos-box", *[str(v) for v in box]])
    for box in args.neg_box or []:
        cmd.extend(["--neg-box", *[str(v) for v in box]])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3 segment CLI for asset pipeline jobs")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--sam3-python", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--pos-box",
        action="append",
        type=_parse_box,
        default=[],
        help="x0,y0,x1,y1",
    )
    parser.add_argument(
        "--neg-box",
        action="append",
        type=_parse_box,
        default=[],
        help="x0,y0,x1,y1",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    sam3_root = Path(args.sam3_root).resolve()
    log_path = job_dir / "logs" / "segment.log"
    sam3_out = job_dir / "sam3"

    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (job_dir / "live").mkdir(parents=True, exist_ok=True)
    sam3_out.mkdir(parents=True, exist_ok=True)

    apply_network_bootstrap(log_fn=lambda msg: append_log(log_path, msg))

    write_status(
        job_dir,
        status="segmenting",
        phase="sam3_segment",
        progress=0.2,
        message="starting sam3 segmentation",
    )

    pos_boxes = list(args.pos_box or [])
    if not pos_boxes and not args.text_only:
        msg = "segment requires --pos-box or --text-only with prompt"
        append_log(log_path, msg)
        write_status(job_dir, status="failed", phase="sam3_segment", progress=0.2, error=msg)
        return 1

    if args.text_only and not (args.prompt or "").strip():
        msg = "text-only mode requires non-empty --prompt"
        append_log(log_path, msg)
        write_status(job_dir, status="failed", phase="sam3_segment", progress=0.2, error=msg)
        return 1

    cmd = _build_sam3_command(args)
    command_summary = " ".join(cmd)
    append_log(log_path, f"[{utc_now_iso()}] command: {command_summary}")
    append_log(log_path, f"cwd: {sam3_root}")
    write_status(
        job_dir,
        status="segmenting",
        phase="sam3_segment",
        progress=0.15,
        message="running sam3 segmentation",
        extra={"commandSummary": command_summary},
    )

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(sam3_root),
            env=subprocess_env(),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if proc.stdout:
            append_log(log_path, proc.stdout)
        if proc.stderr:
            append_log(log_path, proc.stderr)

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "sam3 segmentation failed").strip()[-2000:]
            write_status(
                job_dir,
                status="failed",
                phase="sam3_segment",
                progress=0.35,
                error=err,
                message="sam3 segmentation failed",
            )
            return proc.returncode

        detections_path = sam3_out / "detections.json"
        overlay_path = sam3_out / "overlay.png"
        if not detections_path.is_file():
            err = f"missing detections.json at {detections_path}"
            append_log(log_path, err)
            write_status(job_dir, status="failed", phase="sam3_segment", progress=0.4, error=err)
            return 1

        detections = read_json(detections_path)
        mask_count = int(detections.get("num_masks") or len(detections.get("detections") or []))

        if overlay_path.is_file():
            shutil.copy2(overlay_path, job_dir / "live" / "latest.png")

        manifest_path = write_sam3_manifest(job_dir)
        manifest = read_json(manifest_path)

        merge_job_json(
            job_dir,
            {
                "segmentation": {
                    "prompt": args.prompt,
                    "positiveBoxes": pos_boxes,
                    "negativeBoxes": list(args.neg_box or []),
                    "confidenceThreshold": args.confidence_threshold,
                    "textOnly": bool(args.text_only),
                    "detectionsPath": "sam3/detections.json",
                    "manifestPath": "sam3/manifest.json",
                    "overlayPath": "sam3/overlay.png" if overlay_path.is_file() else None,
                    "maskCount": mask_count,
                    "items": manifest.get("items") or [],
                }
            },
        )

        write_status(
            job_dir,
            status="segmented",
            phase="sam3_segment",
            progress=0.45,
            message=f"sam3 segmentation completed ({mask_count} masks)",
            extra={"maskCount": mask_count},
        )
        append_log(log_path, f"[{utc_now_iso()}] segment success, masks={mask_count}")
        return 0

    except Exception as exc:
        tb = traceback.format_exc()
        append_log(log_path, tb)
        write_status(
            job_dir,
            status="failed",
            phase="sam3_segment",
            progress=0.3,
            error=str(exc),
            message="sam3 segmentation exception",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
