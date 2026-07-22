"""Load recovered bytecode (source file was corrupted)."""
from __future__ import annotations

import sys
from importlib.machinery import SourcelessFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

_RECOVERY = Path(__file__).resolve().parents[2] / ".pyc_recovery" / "workspace_dataset_service.cpython-310.pyc"
_MODULE_NAME = f"{__package__}._workspace_dataset_service_recovered"


def _load() -> None:
    if not _RECOVERY.is_file():
        raise RuntimeError(f"missing recovered bytecode: {_RECOVERY}")
    loader = SourcelessFileLoader(_MODULE_NAME, str(_RECOVERY))
    spec = spec_from_loader(_MODULE_NAME, loader)
    if spec is None:
        raise RuntimeError("failed to build module spec for workspace_dataset_service")
    module = module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    loader.exec_module(module)
    # Do not overwrite this shim's import identity. In particular, copying the
    # recovered module's ``__name__`` / ``__spec__`` and registering it under
    # the public name made Python return the recovered module directly, so none
    # of the external-data compatibility wrappers below were reachable.
    globals().update(
        {
            name: value
            for name, value in module.__dict__.items()
            if name not in {"__name__", "__loader__", "__package__", "__spec__", "__file__"}
        }
    )


_load()

_RECOVERED_DEFAULT_CABLE_THREADING_ROOT = Path(CABLE_THREADING_ROOT)
_RECOVERED_DEFAULT_NUT_ASSEMBLY_ROOT = Path(NUT_ASSEMBLY_ROOT)
_RECOVERED_DEFAULT_DUAL_ARM_ROOT = Path(DUAL_ARM_ROOT)
_RECOVERED_DEFAULT_DATA_GENERATION_ROOT = Path(DATA_GENERATION_ROOT)

# The recovered implementation keeps cable-threading rows deliberately minimal
# when ``include_schema`` is false.  The list API uses that fast path, which made
# a valid HDF5 dataset look non-trainable after runtime outputs moved outside the
# repository.  Keep the recovered scanner, but restore the cheap file/manifest
# fields that do not require inspecting the large HDF5 payload.
_recovered_build_dataset_from_job_dir = _build_dataset_from_job_dir


def _build_dataset_from_job_dir(*args, **kwargs):
    row = _recovered_build_dataset_from_job_dir(*args, **kwargs)
    if not isinstance(row, dict):
        return row

    job_dir = Path(args[1] if len(args) > 1 else kwargs.get("job_dir", ""))
    task_type = str(args[2] if len(args) > 2 else kwargs.get("task_type", ""))
    if task_type == "nut_assembly":
        manifest = _read_json(job_dir / "manifest.json")
        status = _read_json(job_dir / "status.json")
        total = int(
            manifest.get("episodesGenerated")
            or manifest.get("demoCount")
            or status.get("episodesGenerated")
            or row.get("totalEpisodes")
            or row.get("episodeCount")
            or 0
        )
        successful = int(
            manifest.get("successEpisodes")
            or status.get("successfulEpisodes")
            or 0
        )
        valid = int(
            manifest.get("validForTrainingEpisodes")
            or status.get("validForTrainingEpisodes")
            or 0
        )
        row.update(
            {
                "successfulEpisodes": successful,
                "totalEpisodes": total,
                "episodeCount": total,
                "dataCount": total,
                "validTrajectories": valid,
                "trainable": valid > 0,
                "directTrainable": valid > 0,
            }
        )
        return row

    if task_type != "cable_threading":
        return row

    hdf5_path = job_dir / "datasets" / "dataset.hdf5"
    successful = int(row.get("successfulEpisodes") or 0)
    manifest_path = job_dir / "datasets" / "dataset.manifest.json"
    manifest = _read_json(manifest_path)
    successful = int(
        manifest.get("successfulEpisodes")
        or manifest.get("num_successful")
        or successful
        or 0
    )
    if hdf5_path.is_file():
        row.update(
            {
                "format": "hdf5",
                "datasetFormat": "hdf5",
                "datasetFile": str(hdf5_path),
                "hdf5Path": str(hdf5_path),
                "successfulEpisodes": successful,
                "trainable": successful > 0,
                "directTrainable": successful > 0,
                "availableFormats": manifest.get("availableFormats") or ["hdf5"],
                "jointActionAvailable": bool(
                    manifest.get("jointActionAvailable")
                    or manifest.get("joint_action_available")
                ),
            }
        )
    return row


# Recovered functions retain the loader module's global dictionary rather than
# this shim's dictionary, so point their scanner lookup at the compatibility
# wrapper explicitly.
_recovered_build_dataset_from_job_dir.__globals__["_build_dataset_from_job_dir"] = (
    _build_dataset_from_job_dir
)

_recovered_scan_filesystem_datasets = _scan_filesystem_datasets


def _scan_filesystem_datasets(*, include_schema: bool = True):
    """Scan generated datasets from the external data root."""
    from app.core.platform_paths import platform_paths

    recovered_globals = _recovered_scan_filesystem_datasets.__globals__
    def configured_or_current(name: str, recovered_default: Path, current: Path) -> Path:
        configured = Path(globals().get(name) or recovered_default)
        return configured if configured != recovered_default else current

    recovered_globals["CABLE_THREADING_ROOT"] = configured_or_current(
        "CABLE_THREADING_ROOT",
        _RECOVERED_DEFAULT_CABLE_THREADING_ROOT,
        platform_paths.runs_root / "cable_threading" / "jobs",
    )
    recovered_globals["NUT_ASSEMBLY_ROOT"] = configured_or_current(
        "NUT_ASSEMBLY_ROOT",
        _RECOVERED_DEFAULT_NUT_ASSEMBLY_ROOT,
        platform_paths.runs_root / "nut_assembly" / "jobs",
    )
    recovered_globals["DATA_GENERATION_ROOT"] = configured_or_current(
        "DATA_GENERATION_ROOT",
        _RECOVERED_DEFAULT_DATA_GENERATION_ROOT,
        platform_paths.runs_root / "data_generation" / "jobs",
    )
    current_root = platform_paths.runs_root / "dual_arm_cable" / "jobs"
    scan_root = configured_or_current(
        "DUAL_ARM_ROOT",
        _RECOVERED_DEFAULT_DUAL_ARM_ROOT,
        current_root,
    )
    recovered_globals["DUAL_ARM_ROOT"] = scan_root
    datasets, seen_job_ids = _recovered_scan_filesystem_datasets(include_schema=include_schema)

    return datasets, seen_job_ids


_recovered_scan_filesystem_datasets.__globals__["_scan_filesystem_datasets"] = (
    _scan_filesystem_datasets
)

_recovered_scan_datasets_for_api = scan_datasets_for_api


def scan_datasets_for_api():
    from app.services.dataset_naming import apply_dataset_row_display_fields

    rows = _recovered_scan_datasets_for_api()
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_type = str(row.get("taskType") or "").strip()
        if task_type:
            apply_dataset_row_display_fields(
                row,
                task_type=task_type,
                source_job_id=str(row.get("sourceJobId") or ""),
            )
    return rows

__all__ = [name for name in globals() if not name.startswith("_")]
