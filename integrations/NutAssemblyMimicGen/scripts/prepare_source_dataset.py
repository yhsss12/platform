from __future__ import annotations

import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from utils.hdf5_inspector import inspect_hdf5_dataset, source_demo_already_prepared
from utils.runtime_env import (
    build_mimicgen_subprocess_env,
    resolve_mimicgen_root,
    resolve_mimicgen_python,
)


def prepare_source_dataset(
    *,
    job_root: Path,
    source_demo_path: Path,
    env_interface: str = "MG_NutAssembly",
    env_interface_type: str = "robosuite",
    python_bin: Path | None = None,
) -> dict[str, Any]:
    """
    Prepare source HDF5 with datagen_info via MimicGen prepare_src_dataset.py.
    Output: {jobRoot}/intermediate/prepared_source.hdf5
    """
    intermediate_dir = job_root / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = intermediate_dir / "prepared_source.hdf5"
    log_path = job_root / "logs" / "prepare_source.log"

    if not source_demo_path.is_file():
        return {
            "ok": False,
            "reason": "source_demo_missing",
            "error": f"source demo not found: {source_demo_path}",
        }

    if source_demo_already_prepared(source_demo_path):
        shutil.copy2(source_demo_path, prepared_path)
        info = inspect_hdf5_dataset(prepared_path)
        log_path.write_text(
            f"sourceAlreadyPrepared=true\nsource={source_demo_path}\noutput={prepared_path}\n",
            encoding="utf-8",
        )
        return {
            "ok": True,
            "preparedPath": str(prepared_path),
            "sourcePath": str(source_demo_path),
            "sourceAlreadyPrepared": True,
            "hasDatagenInfo": info.get("hasDatagenInfo"),
            "hasObjectPoses": info.get("hasObjectPoses"),
            "objectPoseKeys": info.get("objectPoseKeys", []),
            "logPath": str(log_path),
        }

    mimicgen_root = resolve_mimicgen_root()
    if mimicgen_root is None:
        return {
            "ok": False,
            "reason": "mimicgen_import_failed",
            "error": "mimicgen package not found",
        }

    prepare_script = mimicgen_root / "mimicgen" / "scripts" / "prepare_src_dataset.py"
    if not prepare_script.is_file():
        return {
            "ok": False,
            "reason": "mimicgen_import_failed",
            "error": f"prepare_src_dataset.py not found: {prepare_script}",
        }

    py = python_bin or resolve_mimicgen_python()
    cmd = [
        str(py),
        str(prepare_script),
        "--dataset",
        str(source_demo_path),
        "--env_interface",
        env_interface,
        "--env_interface_type",
        env_interface_type,
        "--output",
        str(prepared_path),
    ]
    env = build_mimicgen_subprocess_env(mimicgen_root=mimicgen_root)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(mimicgen_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + "\n" + (exc.stderr or "")
        log_path.write_text(combined, encoding="utf-8")
        return {
            "ok": False,
            "reason": "prepare_source_timeout",
            "error": "prepare_src_dataset timeout",
            "traceback": combined,
            "logPath": str(log_path),
        }
    except Exception:
        tb = traceback.format_exc()
        log_path.write_text(tb, encoding="utf-8")
        return {
            "ok": False,
            "reason": "prepare_source_failed",
            "error": tb.splitlines()[-1] if tb else "prepare failed",
            "traceback": tb,
            "logPath": str(log_path),
        }

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        err_hint = combined.strip().splitlines()[-1] if combined.strip() else f"exit {proc.returncode}"
        return {
            "ok": False,
            "reason": "prepare_source_failed",
            "error": err_hint,
            "traceback": combined,
            "logPath": str(log_path),
        }

    if not prepared_path.is_file():
        return {
            "ok": False,
            "reason": "prepare_source_failed",
            "error": "prepared_source.hdf5 not created",
            "traceback": combined,
            "logPath": str(log_path),
        }

    info = inspect_hdf5_dataset(prepared_path)
    if not info.get("hasDatagenInfo"):
        return {
            "ok": False,
            "reason": "prepare_source_failed",
            "error": "prepared HDF5 missing datagen_info",
            "inspect": info,
            "logPath": str(log_path),
        }

    return {
        "ok": True,
        "preparedPath": str(prepared_path),
        "sourcePath": str(source_demo_path),
        "sourceAlreadyPrepared": False,
        "hasDatagenInfo": info.get("hasDatagenInfo"),
        "hasObjectPoses": info.get("hasObjectPoses"),
        "hasSubtaskTermSignals": info.get("hasSubtaskTermSignals"),
        "objectPoseKeys": info.get("objectPoseKeys", []),
        "subtaskTermSignalKeys": info.get("subtaskTermSignalKeys", []),
        "logPath": str(log_path),
    }
