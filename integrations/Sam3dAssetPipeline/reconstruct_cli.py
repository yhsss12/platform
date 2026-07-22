#!/usr/bin/env python3
"""Run SAM3D Objects reconstruction for an asset pipeline job (cutout-only input)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from network_bootstrap import apply_network_bootstrap, subprocess_env
from offline_dependency_check import check_offline_dependencies
from sam3_output_normalizer import resolve_manifest_item_by_cutout_index
from sam3d_config_patch import prepare_job_local_sam3d_config
from mujoco_asset_exporter import export_mujoco_for_job
from status_utils import append_log, merge_job_json, utc_now_iso, write_status


def _resolve_cutout_path(job_dir: Path, cutout_index: int) -> tuple[Path, dict]:
    item = resolve_manifest_item_by_cutout_index(job_dir, cutout_index)
    rel = item.get("cutoutPath")
    if not rel:
        raise FileNotFoundError(
            f"Selected cutoutIndex {cutout_index} found in manifest but cutoutPath missing"
        )
    candidate = job_dir / rel
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Selected cutoutIndex {cutout_index} cutout file missing: {candidate}"
        )
    return candidate, item


def _derive_mask_from_cutout(cutout_path: Path, mask_dst: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL required to derive mask from cutout alpha") from exc

    img = Image.open(cutout_path)
    if img.mode == "RGBA":
        alpha = img.split()[3]
        mask = alpha.point(lambda p: 255 if p > 10 else 0, mode="L")
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_dst)
        return "alpha-derived"

    rgb = img.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size
    mask = Image.new("L", (width, height), 0)
    mask_pixels = mask.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if not (r > 245 and g > 245 and b > 245):
                mask_pixels[x, y] = 255
    if not any(mask_pixels[x, y] for x in range(width) for y in range(height)):
        raise ValueError("Selected cutout has no usable internal mask source.")
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_dst)
    return "fallback-derived"


def _prepare_sam3d_inputs(
    job_dir: Path,
    cutout_index: int,
    log_path: Path,
) -> dict[str, str | int]:
    cutout_src, manifest_item = _resolve_cutout_path(job_dir, cutout_index)

    sam3d_dir = job_dir / "sam3d"
    input_dir = sam3d_dir / "input"
    mask_dir = sam3d_dir / "input_masks"
    input_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    sam3d_input_image = input_dir / "image.png"
    sam3d_input_mask = mask_dir / "1.png"

    shutil.copy2(cutout_src, sam3d_input_image)
    append_log(log_path, f"selectedCutoutPath={cutout_src}")
    append_log(log_path, f"cutoutIndex={cutout_index}")
    append_log(log_path, f"sam3dInputImage={sam3d_input_image}")

    mask_source = None
    original_mask_rel = manifest_item.get("originalMaskPath")
    if original_mask_rel:
        original_mask = job_dir / str(original_mask_rel)
        if original_mask.is_file():
            shutil.copy2(original_mask, sam3d_input_mask)
            mask_source = "originalMask"
            append_log(log_path, f"maskSource=originalMask from {original_mask}")

    if mask_source is None:
        mask_source = _derive_mask_from_cutout(cutout_src, sam3d_input_mask)
        append_log(log_path, f"maskSource={mask_source}")

    if not sam3d_input_mask.is_file():
        raise FileNotFoundError(f"failed to generate sam3d input mask: {sam3d_input_mask}")

    append_log(log_path, f"sam3dInputMask={sam3d_input_mask} size={sam3d_input_mask.stat().st_size}")

    return {
        "cutoutIndex": cutout_index,
        "selectedCutoutPath": str(cutout_src.relative_to(job_dir)).replace("\\", "/"),
        "sam3dInputImage": "sam3d/input/image.png",
        "sam3dInputMask": "sam3d/input_masks/1.png",
        "maskSource": mask_source or "unknown",
        "manifestItem": manifest_item,
    }


def _apply_offline_env(*, hf_home: str, torch_home: str) -> None:
    hf = str(Path(hf_home).expanduser().resolve())
    torch_path = str(Path(torch_home).expanduser().resolve())
    os.environ["HF_HOME"] = hf
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf) / "hub")
    os.environ["TORCH_HOME"] = torch_path
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _run_demo_export_all(
    *,
    sam3d_root: Path,
    image_path: Path,
    mask_path: Path,
    out_dir: Path,
    config_path: Path,
    seed: int,
    log_path: Path,
    timeout_seconds: int,
    dinov2_repo: str,
) -> dict:
    """Invoke sam-3d-objects/demo_export_all.py and return parsed metadata.json."""
    export_script = sam3d_root / "demo_export_all.py"
    if not export_script.is_file():
        raise FileNotFoundError(f"demo_export_all.py not found: {export_script}")

    cmd = [
        sys.executable,
        str(export_script),
        "--image",
        str(image_path),
        "--mask",
        str(mask_path),
        "--out",
        str(out_dir),
        "--seed",
        str(seed),
        "--config",
        str(config_path),
        "--mesh-postprocess",
        "--texture-baking",
    ]
    append_log(log_path, f"demo_export_all cmd: {' '.join(cmd)}")

    env = subprocess_env()
    if dinov2_repo:
        env["DINO_LOCAL_REPO"] = str(Path(dinov2_repo).expanduser().resolve())

    proc = subprocess.run(
        cmd,
        cwd=str(sam3d_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
        check=False,
    )
    if proc.stdout:
        append_log(log_path, proc.stdout.rstrip())
    if proc.stderr:
        append_log(log_path, proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(
            f"demo_export_all.py failed with exit code {proc.returncode}"
        )

    metadata_path = out_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json missing after export: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("metadata.json is not a JSON object")
    return metadata


def _sync_legacy_gs_ply(out_dir: Path, log_path: Path) -> str | None:
    """Keep gs.ply alias for older pipeline consumers."""
    splat = out_dir / "splat.ply"
    gs_ply = out_dir / "gs.ply"
    if not splat.is_file():
        return None
    shutil.copy2(splat, gs_ply)
    append_log(log_path, f"[{utc_now_iso()}] gs.ply alias created from splat.ply ({gs_ply.stat().st_size} bytes)")
    return "sam3d/gs.ply"


def _build_reconstruction_record(
    *,
    cutout_index: int,
    seed: int,
    prepared: dict[str, str | int],
    metadata: dict,
    gs_ply_rel: str | None,
    preview_rel: str | None,
) -> dict:
    exports = metadata.get("exports") if isinstance(metadata.get("exports"), dict) else {}
    mesh_info = metadata.get("mesh_info") if isinstance(metadata.get("mesh_info"), dict) else {}

    def _rel(key: str) -> str | None:
        value = exports.get(key)
        if not value:
            return None
        return f"sam3d/{Path(str(value)).name}"

    return {
        "cutoutIndex": cutout_index,
        "seed": seed,
        "selectedCutoutPath": prepared["selectedCutoutPath"],
        "sam3dInputImage": prepared["sam3dInputImage"],
        "sam3dInputMask": prepared["sam3dInputMask"],
        "maskSource": prepared["maskSource"],
        "gsPlyPath": gs_ply_rel or _rel("splat_ply"),
        "splatPlyPath": _rel("splat_ply"),
        "gaussianPlyPath": _rel("gaussian_ply"),
        "meshObjPath": _rel("mesh_obj"),
        "meshStlPath": _rel("mesh_stl"),
        "meshPlyPath": _rel("mesh_ply"),
        "glbPath": _rel("glb"),
        "previewPath": preview_rel,
        "meshInfo": mesh_info,
        "exportWarnings": metadata.get("warnings") or [],
        "metadataPath": "sam3d/metadata.json",
    }


def _try_preview_from_splat(splat_ply: Path, out_gif: Path, log_path: Path, sam3d_root: Path) -> str | None:
    if not splat_ply.is_file():
        return None
    try:
        sys.path.insert(0, str(sam3d_root))
        sys.path.insert(0, str(sam3d_root / "notebook"))
        from inference import ready_gaussian_for_video_rendering, render_video
        from load_splat import load_splat_ply

        gs = load_splat_ply(str(splat_ply))
        return _try_preview(gs, out_gif, log_path)
    except Exception as exc:
        append_log(log_path, f"[preview warning] {exc}")
        return None


def _try_preview(gs, out_gif: Path, log_path: Path) -> str | None:
    try:
        import imageio

        from inference import ready_gaussian_for_video_rendering, render_video

        scene_gs = ready_gaussian_for_video_rendering(gs)
        video = render_video(
            scene_gs,
            r=1,
            fov=60,
            pitch_deg=15,
            yaw_start_deg=-45,
            resolution=512,
        )["color"]
        imageio.mimsave(
            str(out_gif),
            video,
            format="GIF",
            duration=1000 / 30,
            loop=0,
        )
        append_log(log_path, f"[{utc_now_iso()}] preview gif saved: {out_gif}")
        return "sam3d/preview.gif"
    except Exception as exc:
        append_log(log_path, f"[preview warning] {exc}")
        return None


PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_MUJOCO_RENDER_PYTHON = "/home/ubuntu/miniconda3/envs/cable/bin/python"


def _mujoco_render_python() -> str:
    env = os.environ.get("MUJOCO_RENDER_PYTHON", "").strip()
    if env and Path(env).is_file():
        return env
    if Path(DEFAULT_MUJOCO_RENDER_PYTHON).is_file():
        return DEFAULT_MUJOCO_RENDER_PYTHON
    return sys.executable


def _run_mujoco_render(job_dir: Path, log_path: Path) -> dict:
    run_py = PIPELINE_DIR / "run.py"
    render_python = _mujoco_render_python()
    cmd = [
        render_python,
        str(run_py),
        "render-mujoco",
        "--job-dir",
        str(job_dir),
        "--xml-kind",
        "preview",
        "--width",
        os.environ.get("MUJOCO_RENDER_WIDTH", "960"),
        "--height",
        os.environ.get("MUJOCO_RENDER_HEIGHT", "720"),
        "--gl",
        os.environ.get("MUJOCO_RENDER_GL", "egl"),
        "--runner-python",
        render_python,
    ]
    append_log(log_path, f"[mujoco-render] cmd: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(PIPELINE_DIR),
        env={**subprocess_env(), "MUJOCO_GL": os.environ.get("MUJOCO_RENDER_GL", "egl")},
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout:
        append_log(log_path, proc.stdout.rstrip())
    if proc.stderr:
        append_log(log_path, proc.stderr.rstrip())

    record: dict = {"status": "failed", "error": f"render-mujoco exit code {proc.returncode}"}
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
            if isinstance(parsed, dict):
                record = parsed
        except json.JSONDecodeError:
            pass
    if proc.returncode != 0 and record.get("status") != "completed":
        record.setdefault("status", "failed")
        if not record.get("error"):
            record["error"] = proc.stderr.strip() or f"render-mujoco exit code {proc.returncode}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3D reconstruct CLI for asset pipeline jobs")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--sam3d-root", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--cutout-index", type=int, default=None)
    parser.add_argument("--mask-index", type=int, default=None, help="deprecated; use --cutout-index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--offline-mode", action="store_true")
    parser.add_argument("--dinov2-repo", default="")
    parser.add_argument("--dinov2-model", dest="dinov2_model", default="dinov2_vitl14_reg")
    parser.add_argument("--moge-model-path", default="")
    parser.add_argument("--torch-home", default="")
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    cutout_index = args.cutout_index
    if cutout_index is None and args.mask_index is not None:
        cutout_index = int(args.mask_index) + 1
    if cutout_index is None:
        print("error: --cutout-index is required", file=sys.stderr)
        return 2

    job_dir = Path(args.job_dir).resolve()
    sam3d_root = Path(args.sam3d_root).resolve()
    log_path = job_dir / "logs" / "reconstruct.log"
    sam3d_dir = job_dir / "sam3d"
    preview_gif = sam3d_dir / "preview.gif"

    sam3d_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (job_dir / "live").mkdir(parents=True, exist_ok=True)

    apply_network_bootstrap(log_fn=lambda msg: append_log(log_path, msg))

    write_status(
        job_dir,
        status="reconstructing" if not args.prepare_only else "segmented",
        phase="sam3d_prepare",
        progress=0.55,
        message="preparing cutout inputs" if args.prepare_only else "checking offline model dependencies",
    )

    append_log(log_path, f"[{utc_now_iso()}] python={sys.executable}")
    append_log(log_path, f"cwd={os.getcwd()}")
    append_log(log_path, f"job_dir={job_dir}")
    append_log(log_path, f"legacy_image_arg={args.image}")
    append_log(log_path, f"cutout_index={cutout_index}")
    append_log(log_path, f"prepare_only={args.prepare_only}")
    append_log(log_path, f"sam3d_root={sam3d_root}")
    append_log(log_path, f"offline_mode={args.offline_mode}")

    try:
        prepared = _prepare_sam3d_inputs(job_dir, cutout_index, log_path)

        merge_job_json(
            job_dir,
            {
                "reconstruction": {
                    "cutoutIndex": cutout_index,
                    "seed": args.seed,
                    "selectedCutoutPath": prepared["selectedCutoutPath"],
                    "sam3dInputImage": prepared["sam3dInputImage"],
                    "sam3dInputMask": prepared["sam3dInputMask"],
                    "maskSource": prepared["maskSource"],
                    "prepareOnly": args.prepare_only,
                }
            },
        )

        if args.prepare_only:
            write_status(
                job_dir,
                status="segmented",
                phase="sam3d_prepare",
                progress=0.6,
                message="cutout inputs prepared (prepare-only)",
                extra={"cutoutIndex": cutout_index},
            )
            append_log(log_path, f"[{utc_now_iso()}] prepare-only complete")
            return 0

        if args.offline_mode:
            _apply_offline_env(hf_home=args.hf_home, torch_home=args.torch_home)
            append_log(log_path, f"HF_HOME={os.environ.get('HF_HOME')}")
            append_log(log_path, f"TORCH_HOME={os.environ.get('TORCH_HOME')}")
            append_log(log_path, f"DINO repo={args.dinov2_repo}")
            append_log(log_path, f"MoGe path={args.moge_model_path}")

            dep = check_offline_dependencies(
                sam3d_root=sam3d_root,
                dinov2_repo=args.dinov2_repo,
                dinov2_model=args.dinov2_model,
                moge_model_path=args.moge_model_path,
                torch_home=args.torch_home,
                hf_home=args.hf_home,
            )
            append_log(log_path, f"offline dependency check: {dep}")
            if not dep.get("ok"):
                missing = dep.get("missing") or []
                err = "Offline SAM3D dependencies missing: " + "; ".join(missing)
                write_status(
                    job_dir,
                    status="failed",
                    phase="sam3d_prepare",
                    progress=0.55,
                    error=err,
                    message="offline dependency check failed",
                )
                return 1

            dinov2_repo = dep["dinov2Repo"]
            moge_pt = dep["mogeModelPath"]
        else:
            dinov2_repo = args.dinov2_repo
            moge_pt = args.moge_model_path

        append_log(log_path, f"output_dir={sam3d_dir}")

        config_path = prepare_job_local_sam3d_config(
            job_dir=job_dir,
            sam3d_root=sam3d_root,
            dinov2_repo=str(dinov2_repo),
            moge_model_pt=str(moge_pt),
        )
        append_log(log_path, f"pipeline_config={config_path}")

        image_path = (job_dir / prepared["sam3dInputImage"]).resolve()
        mask_path = (job_dir / prepared["sam3dInputMask"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"sam3d input image not found: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"sam3d input mask not found: {mask_path}")

        write_status(
            job_dir,
            status="reconstructing",
            phase="sam3d_load_model",
            progress=0.62,
            message="running demo_export_all (multi-format export)",
        )

        metadata = _run_demo_export_all(
            sam3d_root=sam3d_root,
            image_path=image_path,
            mask_path=mask_path,
            out_dir=sam3d_dir,
            config_path=config_path,
            seed=args.seed,
            log_path=log_path,
            timeout_seconds=args.timeout_seconds,
            dinov2_repo=str(dinov2_repo),
        )

        gs_ply_rel = _sync_legacy_gs_ply(sam3d_dir, log_path)
        preview_rel = _try_preview_from_splat(sam3d_dir / "splat.ply", preview_gif, log_path, sam3d_root)

        reconstruction = _build_reconstruction_record(
            cutout_index=cutout_index,
            seed=args.seed,
            prepared=prepared,
            metadata=metadata,
            gs_ply_rel=gs_ply_rel,
            preview_rel=preview_rel,
        )
        merge_job_json(job_dir, {"reconstruction": reconstruction})

        write_status(
            job_dir,
            status="reconstructing",
            phase="mujoco_export",
            progress=0.9,
            message="generating MuJoCo asset package",
            extra={
                "gsPlyPath": reconstruction.get("gsPlyPath"),
                "glbPath": reconstruction.get("glbPath"),
                "cutoutIndex": cutout_index,
            },
        )

        mujoco_message = "sam3d reconstruction and mujoco export completed"
        try:
            mujoco_export = export_mujoco_for_job(
                job_dir,
                model_name=job_dir.name,
                scale_longest=0.12,
                mass=0.5,
                collision="visual",
                log_fn=lambda msg: append_log(log_path, f"[mujoco] {msg}"),
            )
            merge_job_json(job_dir, {"mujocoExport": mujoco_export})
            if mujoco_export.get("status") == "failed":
                mujoco_message = "sam3d reconstruction completed; mujoco export failed"
                append_log(log_path, f"[mujoco] export failed: {mujoco_export.get('error')}")
        except Exception as mujoco_exc:
            mujoco_export = {
                "status": "failed",
                "outputDir": "exports/mujoco",
                "error": str(mujoco_exc),
            }
            merge_job_json(job_dir, {"mujocoExport": mujoco_export})
            append_log(log_path, f"[mujoco] export exception: {mujoco_exc}")
            mujoco_message = "sam3d reconstruction completed; mujoco export failed"

        mujoco_visualization: dict | None = None
        if mujoco_export.get("status") in {"completed", "partial"}:
            write_status(
                job_dir,
                status="reconstructing",
                phase="mujoco_visualize",
                progress=0.96,
                message="rendering MuJoCo preview",
                extra={
                    "gsPlyPath": reconstruction.get("gsPlyPath"),
                    "glbPath": reconstruction.get("glbPath"),
                    "cutoutIndex": cutout_index,
                    "mujocoExportStatus": mujoco_export.get("status"),
                },
            )
            try:
                mujoco_visualization = _run_mujoco_render(job_dir, log_path)
                merge_job_json(job_dir, {"mujocoVisualization": mujoco_visualization})
                if mujoco_visualization.get("status") == "completed":
                    mujoco_message = "sam3d reconstruction, mujoco export and preview completed"
                elif mujoco_export.get("status") in {"completed", "partial"}:
                    mujoco_message = "sam3d reconstruction and mujoco export completed; preview render failed"
                    append_log(
                        log_path,
                        f"[mujoco-render] failed: {mujoco_visualization.get('error')}",
                    )
            except Exception as render_exc:
                mujoco_visualization = {"status": "failed", "error": str(render_exc)}
                merge_job_json(job_dir, {"mujocoVisualization": mujoco_visualization})
                append_log(log_path, f"[mujoco-render] exception: {render_exc}")
                if mujoco_export.get("status") in {"completed", "partial"}:
                    mujoco_message = "sam3d reconstruction and mujoco export completed; preview render failed"

        write_status(
            job_dir,
            status="reconstructed",
            phase="sam3d_reconstruct",
            progress=1.0,
            message=mujoco_message,
            extra={
                "gsPlyPath": reconstruction.get("gsPlyPath"),
                "glbPath": reconstruction.get("glbPath"),
                "cutoutIndex": cutout_index,
                "mujocoExportStatus": mujoco_export.get("status"),
                "mujocoVisualizationStatus": (mujoco_visualization or {}).get("status"),
            },
        )
        return 0

    except Exception as exc:
        tb = traceback.format_exc()
        append_log(log_path, tb)
        phase = (
            "sam3d_prepare"
            if "Offline SAM3D" in str(exc) or "manifest" in str(exc).lower() or args.prepare_only
            else "sam3d_reconstruct"
        )
        write_status(
            job_dir,
            status="failed" if not args.prepare_only else "segmented",
            phase=phase,
            progress=0.65,
            error=str(exc),
            message="sam3d preparation failed" if args.prepare_only else "sam3d reconstruction failed",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
