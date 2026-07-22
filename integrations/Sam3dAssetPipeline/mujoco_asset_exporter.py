#!/usr/bin/env python3
"""Generate MuJoCo asset package from SAM3D mesh exports."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree as ET

CollisionMode = Literal["convex_hull", "visual", "none"]

MUJOCO_OUTPUT_REL = "exports/mujoco"
MESH_ERROR = "MuJoCo export requires sam3d/mesh.obj or sam3d/object.glb"


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _load_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "trimesh is required for MuJoCo export; install it in the SAM3D pipeline Python environment"
        ) from exc
    return trimesh


def _resolve_input_mesh(job_dir: Path) -> tuple[Path, str]:
    mesh_obj = job_dir / "sam3d" / "mesh.obj"
    if mesh_obj.is_file():
        return mesh_obj, "sam3d/mesh.obj"
    mesh_glb = job_dir / "sam3d" / "object.glb"
    if mesh_glb.is_file():
        return mesh_glb, "sam3d/object.glb"
    raise FileNotFoundError(MESH_ERROR)


def _load_mesh(mesh_path: Path):
    trimesh = _load_trimesh()
    loaded = trimesh.load(str(mesh_path), force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if g is not None]
        if not geometries:
            raise ValueError(f"empty scene in mesh file: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    return mesh


def _clean_mesh(mesh):
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "merge_vertices"):
        mesh.merge_vertices()
    if hasattr(mesh, "update_faces") and hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        try:
            faces = mesh.faces
            if getattr(faces, "shape", (0,))[1] != 3 and hasattr(mesh, "triangulate"):
                mesh = mesh.triangulate()
        except Exception:
            pass
    return mesh


def _bounds_extents(mesh) -> tuple[list[list[float]], list[float]]:
    bounds = mesh.bounds.tolist()
    extents = mesh.extents.tolist()
    return bounds, extents


def _normalize_mesh(mesh, *, scale_longest: float | None) -> dict[str, Any]:
    original_bounds, original_extents = _bounds_extents(mesh)
    scale_applied = 1.0
    if scale_longest and scale_longest > 0:
        longest = float(max(original_extents))
        if longest > 0:
            scale_applied = float(scale_longest) / longest
            mesh.apply_scale(scale_applied)

    bounds = mesh.bounds
    center_xy = (bounds[0] + bounds[1]) / 2.0
    translation = [-float(center_xy[0]), -float(center_xy[1]), -float(bounds[0][2])]
    mesh.apply_translation(translation)

    final_bounds, final_extents = _bounds_extents(mesh)
    return {
        "originalBounds": original_bounds,
        "originalExtents": original_extents,
        "finalBounds": final_bounds,
        "finalExtents": final_extents,
        "scaleApplied": scale_applied,
        "translation": translation,
    }


def _export_mesh_files(mesh, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / f"{stem}.obj"
    stl_path = out_dir / f"{stem}.stl"
    mesh.export(str(obj_path))
    mesh.export(str(stl_path))
    return {
        "obj": str(obj_path),
        "stl": str(stl_path),
        "objRel": f"{MUJOCO_OUTPUT_REL}/meshes/{stem}.obj",
        "stlRel": f"{MUJOCO_OUTPUT_REL}/meshes/{stem}.stl",
    }


def _build_collision_mesh(visual_mesh, collision: CollisionMode, warnings: list[str]):
    if collision == "none":
        return None
    if collision == "visual":
        return visual_mesh.copy()
    try:
        hull = visual_mesh.convex_hull
        if hull is None or len(hull.vertices) == 0:
            raise ValueError("convex_hull returned empty mesh")
        return hull
    except Exception as exc:
        warnings.append(f"convex_hull failed ({exc}); fallback to visual collision mesh")
        return visual_mesh.copy()


def _box_inertia(mass: float, extents: list[float]) -> tuple[list[float], list[float]]:
    x, y, z = [max(float(v), 1e-6) for v in extents]
    ixx = mass / 12.0 * (y * y + z * z)
    iyy = mass / 12.0 * (x * x + z * z)
    izz = mass / 12.0 * (x * x + y * y)
    center = [0.0, 0.0, z / 2.0]
    return center, [ixx, iyy, izz]


def _camera_attrs(final_extents: list[float]) -> dict[str, str]:
    ex, ey, ez = [max(float(v), 0.05) for v in final_extents]
    radius = max(ex, ey, ez) * 2.2
    center_z = ez / 2.0
    cam_pos = f"{radius * 0.8:.4f} {-radius * 0.6:.4f} {center_z + radius * 0.55:.4f}"
    return {
        "name": "preview",
        "pos": cam_pos,
        "xyaxes": "0.8 0.6 0 -0.4 0.5 0.8",
        "fovy": "45",
    }


def _write_xml(
    path: Path,
    *,
    model_name: str,
    include_collision: bool,
    freejoint: bool,
    mass: float,
    final_extents: list[float],
    inertial_center: list[float],
    diaginertia: list[float],
    body_z: float,
) -> None:
    mj = ET.Element("mujoco", model=model_name)
    ET.SubElement(mj, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    ET.SubElement(mj, "option", gravity="0 0 -9.81", timestep="0.002")
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="120", elevation="-20")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "mesh", name="visual", file="visual.obj")
    if include_collision:
        ET.SubElement(asset, "mesh", name="collision", file="collision.obj")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.7",
        rgb2="0 0 0",
        width="512",
        height="3072",
    )
    ET.SubElement(asset, "material", name="floor_mat", reflectance="0.2")

    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.05",
        pos="0 0 0",
        material="floor_mat",
        friction="1 0.005 0.0001",
    )
    cam = _camera_attrs(final_extents)
    ET.SubElement(world, "camera", **cam)

    body = ET.SubElement(
        world,
        "body",
        name="object",
        pos=f"0 0 {body_z:.6f}",
    )
    if freejoint:
        ET.SubElement(body, "freejoint", name="root")
        ET.SubElement(
            body,
            "inertial",
            pos=" ".join(f"{v:.6f}" for v in inertial_center),
            mass=f"{mass:.6f}",
            diaginertia=" ".join(f"{v:.8f}" for v in diaginertia),
        )

    ET.SubElement(
        body,
        "geom",
        name="visual_geom",
        type="mesh",
        mesh="visual",
        contype="0",
        conaffinity="0",
        rgba="0.8 0.8 0.8 1",
    )
    if include_collision:
        ET.SubElement(
            body,
            "geom",
            name="collision_geom",
            type="mesh",
            mesh="collision",
            contype="1",
            conaffinity="1",
            rgba="1 0 0 0.25",
            friction="1 0.005 0.0001",
            condim="3",
        )

    tree = ET.ElementTree(mj)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def validate_mujoco_xml(xml_path: Path) -> dict[str, Any]:
    try:
        import mujoco
    except ImportError:
        return {
            "ok": False,
            "skipped": True,
            "error": "MuJoCo python package not available; XML validation skipped",
        }

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
        return {
            "ok": False,
            "error": str(exc),
        }


def _package_zip(output_dir: Path, zip_path: Path) -> None:
    include = [
        "model_preview.xml",
        "model.xml",
        "metadata.json",
        "meshes/visual.obj",
        "meshes/visual.stl",
        "meshes/collision.obj",
        "meshes/collision.stl",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            src = output_dir / rel
            if src.is_file():
                zf.write(src, arcname=rel)


def export_mujoco_asset(
    *,
    job_dir: Path,
    model_name: str,
    scale_longest: float = 0.12,
    mass: float = 0.5,
    collision: CollisionMode = "visual",
    log_fn=None,
) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    warnings: list[str] = []
    log = log_fn or (lambda _msg: None)

    mesh_path, mesh_source = _resolve_input_mesh(job_dir)
    log(f"input mesh: {mesh_path} (source={mesh_source})")

    mesh = _load_mesh(mesh_path)
    mesh = _clean_mesh(mesh)
    transform_info = _normalize_mesh(mesh, scale_longest=scale_longest)
    final_extents = transform_info["finalExtents"]

    output_dir = job_dir / MUJOCO_OUTPUT_REL
    meshes_dir = output_dir / "meshes"
    visual_paths = _export_mesh_files(mesh, meshes_dir, "visual")

    collision_mesh = _build_collision_mesh(mesh, collision, warnings)
    if collision == "visual":
        warnings.append("collision mesh uses visual mesh geometry")
    collision_paths: dict[str, str] | None = None
    include_collision = collision != "none" and collision_mesh is not None
    if include_collision and collision_mesh is not None:
        collision_paths = _export_mesh_files(collision_mesh, meshes_dir, "collision")

    inertial_center, diaginertia = _box_inertia(mass, final_extents)
    ez = float(final_extents[2])
    body_z_physics = max(0.15, ez * 1.5)

    preview_xml = output_dir / "model_preview.xml"
    physics_xml = output_dir / "model.xml"
    _write_xml(
        preview_xml,
        model_name=model_name,
        include_collision=include_collision,
        freejoint=False,
        mass=mass,
        final_extents=final_extents,
        inertial_center=inertial_center,
        diaginertia=diaginertia,
        body_z=0.0,
    )
    _write_xml(
        physics_xml,
        model_name=model_name,
        include_collision=include_collision,
        freejoint=True,
        mass=mass,
        final_extents=final_extents,
        inertial_center=inertial_center,
        diaginertia=diaginertia,
        body_z=body_z_physics,
    )

    validation = {
        "preview": validate_mujoco_xml(preview_xml),
        "physics": validate_mujoco_xml(physics_xml),
    }
    if validation["preview"].get("skipped") or validation["physics"].get("skipped"):
        warnings.append("MuJoCo python package not available; XML validation skipped")

    metadata = {
        "generatedAt": utc_now_iso(),
        "modelName": model_name,
        "inputMesh": mesh_source,
        "scaleLongest": scale_longest,
        "mass": mass,
        "collision": collision,
        "transform": transform_info,
        "inertiaModel": "box approximation from mesh extents",
        "inertialCenter": inertial_center,
        "diagInertia": diaginertia,
        "visualMesh": visual_paths["objRel"],
        "collisionMesh": collision_paths["objRel"] if collision_paths else None,
        "validation": validation,
        "warnings": warnings,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    zip_path = output_dir / "mujoco_package.zip"
    _package_zip(output_dir, zip_path)

    preview_rel = f"{MUJOCO_OUTPUT_REL}/model_preview.xml"
    physics_rel = f"{MUJOCO_OUTPUT_REL}/model.xml"
    package_rel = f"{MUJOCO_OUTPUT_REL}/mujoco_package.zip"
    metadata_rel = f"{MUJOCO_OUTPUT_REL}/metadata.json"

    conversion_ok = True
    conversion_error: str | None = None
    for key in ("preview", "physics"):
        result = validation.get(key) or {}
        if result.get("skipped"):
            continue
        if not result.get("ok"):
            conversion_ok = False
            conversion_error = result.get("error") or f"{key} XML validation failed"

    status = "completed" if conversion_ok else "partial"
    if mesh_source.endswith(".glb") and collision == "convex_hull":
        warnings.append("object.glb used as input; convex hull quality may vary")

    export_record = {
        "status": status,
        "outputDir": MUJOCO_OUTPUT_REL,
        "modelPreviewXml": preview_rel,
        "modelXml": physics_rel,
        "packageZip": package_rel,
        "visualMesh": visual_paths["objRel"],
        "collisionMesh": collision_paths["objRel"] if collision_paths else None,
        "metadataPath": metadata_rel,
        "scaleLongest": scale_longest,
        "mass": mass,
        "collision": collision,
        "validation": validation,
        "viewerCommands": {
            "preview": f"python scripts/view_mujoco_asset.py --xml {preview_rel}",
            "physics": f"python scripts/view_mujoco_asset.py --xml {physics_rel}",
        },
        "warnings": warnings,
        "error": conversion_error,
    }
    log(f"mujoco export status={status}")
    return export_record



def export_mujoco_for_job(
    job_dir: Path,
    *,
    model_name: str | None = None,
    scale_longest: float = 0.12,
    mass: float = 0.5,
    collision: CollisionMode = "visual",
    log_fn=None,
) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    resolved_name = model_name or job_dir.name
    try:
        record = export_mujoco_asset(
            job_dir=job_dir,
            model_name=resolved_name,
            scale_longest=scale_longest,
            mass=mass,
            collision=collision,
            log_fn=log_fn,
        )
        return record
    except FileNotFoundError as exc:
        return {
            "status": "failed",
            "outputDir": MUJOCO_OUTPUT_REL,
            "error": str(exc),
            "scaleLongest": scale_longest,
            "mass": mass,
            "collision": collision,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "outputDir": MUJOCO_OUTPUT_REL,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "scaleLongest": scale_longest,
            "mass": mass,
            "collision": collision,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MuJoCo asset package for an asset pipeline job")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--scale-longest", type=float, default=0.12)
    parser.add_argument("--mass", type=float, default=0.5)
    parser.add_argument("--collision", choices=["convex_hull", "visual", "none"], default="visual")
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    model_name = args.model_name.strip() or job_dir.name

    record = export_mujoco_for_job(
        job_dir,
        model_name=model_name,
        scale_longest=args.scale_longest,
        mass=args.mass,
        collision=args.collision,
        log_fn=lambda msg: print(msg, file=sys.stderr),
    )

    from status_utils import merge_job_json

    merge_job_json(job_dir, {"mujocoExport": record})

    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if record.get("status") in {"completed", "partial"} else 1


if __name__ == "__main__":
    sys.exit(main())
