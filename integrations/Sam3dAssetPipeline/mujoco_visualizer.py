#!/usr/bin/env python3
"""MuJoCo XML validation and offscreen rendering for asset pipeline jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Literal

MUJOCO_OUTPUT_REL = "exports/mujoco"
XmlKind = Literal["preview", "physics"]

XML_KIND_PATHS: dict[XmlKind, str] = {
    "preview": f"{MUJOCO_OUTPUT_REL}/model_preview.xml",
    "physics": f"{MUJOCO_OUTPUT_REL}/model.xml",
}


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _configure_gl(gl_backend: str) -> str:
    backend = (gl_backend or "egl").strip().lower()
    os.environ["MUJOCO_GL"] = backend
    return backend


def _gl_backends_to_try(preferred: str) -> list[str]:
    backends: list[str] = []
    for candidate in [preferred, "egl", "osmesa"]:
        if candidate and candidate not in backends:
            backends.append(candidate)
    if not os.environ.get("DISPLAY") and "glfw" not in backends:
        pass
    elif os.environ.get("DISPLAY"):
        for candidate in ["glfw"]:
            if candidate not in backends:
                backends.append(candidate)
    return backends


def _try_import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "mujoco python package is required for validation/rendering; "
            "use MUJOCO_RENDER_PYTHON (e.g. cable conda env)"
        ) from exc
    return mujoco


def validate_mujoco_xml(xml_path: Path) -> dict[str, Any]:
    try:
        mujoco = _try_import_mujoco()
    except RuntimeError as exc:
        return {"ok": False, "skipped": True, "error": str(exc)}

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        del data
        return {
            "ok": True,
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nmesh": int(model.nmesh),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "error": None,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _ensure_offscreen_size(model, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def _pick_camera(model, data, mujoco, renderer) -> None:
    try:
        renderer.update_scene(data, camera="preview")
        return
    except Exception:
        pass

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    center = getattr(model.stat, "center", None)
    extent = float(getattr(model.stat, "extent", 1.0) or 1.0)
    if center is not None:
        cam.lookat[:] = center
    cam.distance = max(extent * 1.8, 0.5)
    cam.azimuth = 135
    cam.elevation = -25
    renderer.update_scene(data, camera=cam)


def render_mujoco_xml_to_png(
    *,
    xml_path: Path,
    output_path: Path,
    width: int = 960,
    height: int = 720,
    gl_backend: str = "egl",
) -> dict[str, Any]:
    gl_tried: list[str] = []
    last_error: Exception | None = None

    for backend in _gl_backends_to_try(gl_backend):
        gl_tried.append(backend)
        _configure_gl(backend)
        renderer = None
        try:
            mujoco = _try_import_mujoco()
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            _ensure_offscreen_size(model, width, height)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            renderer = mujoco.Renderer(model, int(height), int(width))
            _pick_camera(model, data, mujoco, renderer)
            image = renderer.render()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import imageio.v2 as imageio

                imageio.imwrite(str(output_path), image)
            except Exception:
                from PIL import Image

                Image.fromarray(image).save(output_path)

            return {
                "ok": True,
                "glBackend": backend,
                "preview": str(output_path),
                "width": int(width),
                "height": int(height),
                "nbody": int(model.nbody),
                "ngeom": int(model.ngeom),
                "nmesh": int(model.nmesh),
                "nq": int(model.nq),
                "nv": int(model.nv),
                "error": None,
            }
        except Exception as exc:
            last_error = exc
            continue
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass

    return {
        "ok": False,
        "glBackend": gl_backend,
        "error": str(last_error or "MuJoCo offscreen render failed"),
        "glTried": gl_tried,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_metadata(job_dir: Path, patch: dict[str, Any]) -> None:
    metadata_path = job_dir / MUJOCO_OUTPUT_REL / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata.update(patch)
    metadata["updatedAt"] = utc_now_iso()
    _write_json(metadata_path, metadata)


def render_mujoco_for_job(
    job_dir: Path,
    *,
    xml_kind: XmlKind = "preview",
    width: int = 960,
    height: int = 720,
    gl_backend: str = "egl",
    validate_all: bool = True,
) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    xml_rel = XML_KIND_PATHS[xml_kind]
    xml_path = job_dir / xml_rel
    if not xml_path.is_file():
        return {
            "status": "failed",
            "xml": xml_rel,
            "error": f"MuJoCo XML not found: {xml_rel}",
        }

    validation_patch: dict[str, Any] = {}
    if validate_all:
        preview_validation = validate_mujoco_xml(job_dir / XML_KIND_PATHS["preview"])
        physics_validation = validate_mujoco_xml(job_dir / XML_KIND_PATHS["physics"])
        validation_patch = {
            "preview": preview_validation,
            "physics": physics_validation,
        }
        _update_metadata(
            job_dir,
            {
                "validation": validation_patch,
                "validationUpdatedAt": utc_now_iso(),
                "validationRunner": sys.executable,
            },
        )

    preview_rel = f"{MUJOCO_OUTPUT_REL}/preview.png"
    preview_path = job_dir / preview_rel
    render_result = render_mujoco_xml_to_png(
        xml_path=xml_path,
        output_path=preview_path,
        width=width,
        height=height,
        gl_backend=gl_backend,
    )

    if not render_result.get("ok"):
        return {
            "status": "failed",
            "xml": xml_rel,
            "previewImage": preview_rel,
            "renderer": "mujoco.Renderer",
            "width": width,
            "height": height,
            "validation": validation_patch or None,
            "error": render_result.get("error"),
            "glTried": render_result.get("glTried"),
        }

    visualization = {
        "status": "completed",
        "previewImage": preview_rel,
        "xml": xml_rel,
        "renderer": "mujoco.Renderer",
        "glBackend": render_result.get("glBackend"),
        "width": width,
        "height": height,
        "nbody": render_result.get("nbody"),
        "ngeom": render_result.get("ngeom"),
        "nmesh": render_result.get("nmesh"),
        "nq": render_result.get("nq"),
        "nv": render_result.get("nv"),
        "error": None,
    }

    _update_metadata(
        job_dir,
        {
            "visualization": visualization,
            "validation": validation_patch or _read_json(job_dir / MUJOCO_OUTPUT_REL / "metadata.json").get("validation"),
        },
    )

    return {
        **visualization,
        "validation": validation_patch or None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and offscreen-render MuJoCo XML assets")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--xml", default="", help="Relative XML path under job dir")
    parser.add_argument("--xml-kind", choices=["preview", "physics"], default="preview")
    parser.add_argument("--output", default=f"{MUJOCO_OUTPUT_REL}/preview.png")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gl", default="egl")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    xml_rel = args.xml.strip() or XML_KIND_PATHS[args.xml_kind]
    xml_path = job_dir / xml_rel

    try:
        if args.validate_only:
            result = validate_mujoco_xml(xml_path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result.get("ok") or result.get("skipped") else 1

        record = render_mujoco_for_job(
            job_dir,
            xml_kind=args.xml_kind,
            width=args.width,
            height=args.height,
            gl_backend=args.gl,
            validate_all=True,
        )

        from status_utils import merge_job_json, read_json

        job_patch: dict[str, Any] = {"mujocoVisualization": record}
        if record.get("validation"):
            existing = read_json(job_dir / "job.json")
            export_info = dict(existing.get("mujocoExport") or {})
            export_info["validation"] = record["validation"]
            warnings = [
                w
                for w in list(export_info.get("warnings") or [])
                if "validation skipped" not in w.lower()
            ]
            export_info["warnings"] = warnings
            job_patch["mujocoExport"] = export_info

        merge_job_json(job_dir, job_patch)
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0 if record.get("status") == "completed" else 1
    except Exception as exc:
        err = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(err, indent=2, ensure_ascii=False), file=sys.stderr)
        try:
            from status_utils import merge_job_json

            merge_job_json(job_dir, {"mujocoVisualization": err})
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
