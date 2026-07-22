from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py

from utils.hdf5_inspector import inspect_hdf5_dataset, source_demo_already_prepared
from utils.official_assets import (
    HF_REPO_ID,
    OFFICIAL_CORE_DEFAULT,
    OFFICIAL_CORE_REL,
    OFFICIAL_SOURCE_DEFAULT,
    OFFICIAL_SOURCE_REL,
    PROVENANCE_MANIFEST,
    load_provenance_manifest,
    save_provenance_manifest,
)
from utils.source_demo_provenance import (
    OFFICIAL_CORE_REGISTRY,
    OFFICIAL_SOURCE_REGISTRY,
    _file_hashes,
    _read_env_args,
    audit_source_demo,
)

NUT_ASSEMBLY_INTERFACES = {"MG_NutAssembly", "MG_Square"}
NUT_ASSEMBLY_ENV_NAMES = {
    "nut_assembly",
    "NutAssembly",
    "NutAssembly_D0",
    "NutAssemblySquare",
    "Square_D0",
}


def _demo_has_obs(path: Path) -> bool:
    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            return False
        for key in data.keys():
            if not str(key).startswith("demo_"):
                continue
            grp = data[key]
            if "obs" in grp or "states" in grp:
                return True
    return False


def _demo_has_actions(path: Path) -> bool:
    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            return False
        for key in data.keys():
            if not str(key).startswith("demo_"):
                continue
            if "actions" in data[key]:
                return True
    return False


def _detect_env_interface(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as f:
            data = f.get("data")
            if data is None:
                return None
            for key in sorted(data.keys()):
                if not str(key).startswith("demo_"):
                    continue
                dg = data[key].get("datagen_info")
                if dg is not None and "env_interface_name" in dg.attrs:
                    raw = dg.attrs["env_interface_name"]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    return str(raw)
    except Exception:
        return None
    return None


def validate_source_demo_file(
    path: Path,
    *,
    registry_rel: str = OFFICIAL_SOURCE_REL,
    registry_type: str = "source",
    user_provided: bool = False,
) -> dict[str, Any]:
    """Full validation payload for source demo acceptance."""
    result: dict[str, Any] = {
        "path": str(path.resolve()) if path.is_file() else str(path),
        "registryRelativePath": registry_rel,
        "registryType": registry_type,
        "exists": path.is_file(),
        "readable": False,
        "fileSizeBytes": None,
        "md5": None,
        "sha256": None,
        "demoCount": 0,
        "envName": None,
        "envArgs": {},
        "hasActions": False,
        "hasObs": False,
        "hasDatagenInfo": False,
        "hasObjectPoses": False,
        "objectPoseKeys": [],
        "envInterfaceName": None,
        "compatibleInterfaces": [],
        "needsPrepare": True,
        "alreadyPrepared": False,
        "validAsMimicgenSourceDemo": False,
        "validationPassed": False,
        "validationErrors": [],
        "validationWarnings": [],
    }

    if not path.is_file():
        result["validationErrors"].append("file_missing")
        return result

    result["fileSizeBytes"] = path.stat().st_size
    try:
        result.update(_file_hashes(path))
        with h5py.File(path, "r") as f:
            if "data" not in f:
                result["validationErrors"].append("missing_data_group")
                return result
        result["readable"] = True
    except Exception as exc:
        result["validationErrors"].append(f"hdf5_read_failed: {exc}")
        return result

    hdf5_info = inspect_hdf5_dataset(path)
    env_args = _read_env_args(path)
    result["demoCount"] = hdf5_info.get("demoCount", 0)
    result["envArgs"] = env_args
    result["envName"] = env_args.get("env_name")
    result["hasActions"] = _demo_has_actions(path)
    result["hasObs"] = _demo_has_obs(path)
    result["hasDatagenInfo"] = bool(hdf5_info.get("hasDatagenInfo"))
    result["hasObjectPoses"] = bool(hdf5_info.get("hasObjectPoses"))
    result["objectPoseKeys"] = hdf5_info.get("objectPoseKeys") or []
    result["envInterfaceName"] = _detect_env_interface(path)
    result["alreadyPrepared"] = source_demo_already_prepared(path)
    result["needsPrepare"] = not result["alreadyPrepared"]

    env_name = result["envName"]
    if env_name in NUT_ASSEMBLY_ENV_NAMES or registry_rel == OFFICIAL_SOURCE_REL:
        if env_name and "Square" in str(env_name):
            result["compatibleInterfaces"] = ["MG_Square", "MG_NutAssembly"]
        else:
            result["compatibleInterfaces"] = ["MG_NutAssembly"]
    elif result["envInterfaceName"] in NUT_ASSEMBLY_INTERFACES:
        result["compatibleInterfaces"] = [result["envInterfaceName"]]

    if result["demoCount"] < 1:
        result["validationErrors"].append("no_demos")
    if not result["hasActions"]:
        result["validationErrors"].append("missing_actions")
    if registry_type == "source" and not result["hasObs"]:
        result["validationWarnings"].append("missing_obs_group")

    expected_env = (
        OFFICIAL_SOURCE_REGISTRY.get(registry_rel)
        if registry_type == "source"
        else OFFICIAL_CORE_REGISTRY.get(registry_rel)
    )
    if expected_env and env_name:
        env_ok = env_name == expected_env or (
            registry_rel == OFFICIAL_SOURCE_REL
            and env_name in NUT_ASSEMBLY_ENV_NAMES
        )
        if not env_ok:
            result["validationWarnings"].append(
                f"env_name={env_name} expected ~{expected_env} for {registry_rel}"
            )

    audit = audit_source_demo(path, user_provided=user_provided)
    result["sourceDemoOrigin"] = audit.get("sourceDemoOrigin")
    result["sourceDemoOriginReason"] = audit.get("sourceDemoOriginReason")

    result["validAsMimicgenSourceDemo"] = (
        registry_type == "source"
        and result["demoCount"] >= 1
        and result["hasActions"]
        and (result["hasDatagenInfo"] or result["needsPrepare"])
    )

    result["validationPassed"] = (
        result["exists"]
        and result["readable"]
        and result["demoCount"] >= 1
        and result["hasActions"]
        and not result["validationErrors"]
    )
    return result


def validate_official_assets(
    *,
    source_path: Path | None = None,
    core_path: Path | None = None,
) -> dict[str, Any]:
    source_path = source_path or OFFICIAL_SOURCE_DEFAULT
    core_path = core_path or OFFICIAL_CORE_DEFAULT

    source_val = validate_source_demo_file(
        source_path, registry_rel=OFFICIAL_SOURCE_REL, registry_type="source"
    )
    core_val = validate_source_demo_file(
        core_path, registry_rel=OFFICIAL_CORE_REL, registry_type="core"
    )

    manifest: dict[str, Any] = load_provenance_manifest()
    manifest.update(
        {
            "hfRepoId": HF_REPO_ID,
            "validatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": {
                **source_val,
                "registryRelativePath": OFFICIAL_SOURCE_REL,
                "purpose": "mimicgen_datagen_source_demo",
            },
            "core": {
                **core_val,
                "registryRelativePath": OFFICIAL_CORE_REL,
                "purpose": "import_training_replay_validation",
                "notDefaultSourceDemo": True,
            },
        }
    )
    save_provenance_manifest(manifest)
    if source_path.is_file():
        manifest["source"].update(audit_source_demo(source_path))
    if core_path.is_file():
        manifest["core"].update(audit_source_demo(core_path))
    save_provenance_manifest(manifest)
    return manifest


def render_validation_report(manifest: dict[str, Any]) -> str:
    source = manifest.get("source") or {}
    core = manifest.get("core") or {}
    lines = [
        "# Official MimicGen NutAssembly Source Demo Validation Report",
        "",
        f"**Validated at**: {manifest.get('validatedAt', '—')}",
        f"**HF repo**: `{manifest.get('hfRepoId', HF_REPO_ID)}`",
        "",
        "## Source demo (`source/nut_assembly.hdf5`)",
        "",
        f"- Path: `{source.get('path')}`",
        f"- Exists: **{source.get('exists')}**",
        f"- Validation passed: **{source.get('validationPassed')}**",
        f"- File size: {source.get('fileSizeBytes')} bytes",
        f"- MD5: `{source.get('md5')}`",
        f"- SHA256: `{source.get('sha256')}`",
        f"- Demo count: **{source.get('demoCount')}**",
        f"- env_name: **`{source.get('envName')}`**",
        f"- hasActions: {source.get('hasActions')}",
        f"- hasObs: {source.get('hasObs')}",
        f"- hasDatagenInfo: {source.get('hasDatagenInfo')}",
        f"- hasObjectPoses: {source.get('hasObjectPoses')}",
        f"- objectPoseKeys: `{source.get('objectPoseKeys')}`",
        f"- envInterfaceName: `{source.get('envInterfaceName')}`",
        f"- compatibleInterfaces: `{source.get('compatibleInterfaces')}`",
        f"- needsPrepare: **{source.get('needsPrepare')}**",
        f"- alreadyPrepared: **{source.get('alreadyPrepared')}**",
        f"- validAsMimicgenSourceDemo: **{source.get('validAsMimicgenSourceDemo')}**",
        f"- sourceDemoOrigin: **`{source.get('sourceDemoOrigin')}`**",
        "",
    ]
    if source.get("validationErrors"):
        lines.append(f"- Errors: {source.get('validationErrors')}")
    if source.get("validationWarnings"):
        lines.append(f"- Warnings: {source.get('validationWarnings')}")
    lines.extend(
        [
            "",
            "## Core dataset (`core/nut_assembly_d0.hdf5`) — not default source demo",
            "",
            f"- Path: `{core.get('path')}`",
            f"- Exists: **{core.get('exists')}**",
            f"- Validation passed: **{core.get('validationPassed')}**",
            f"- Demo count: **{core.get('demoCount')}**",
            f"- env_name: **`{core.get('envName')}`**",
            f"- sourceDemoOrigin: **`{core.get('sourceDemoOrigin')}`**",
            "",
            f"Manifest JSON: `{PROVENANCE_MANIFEST}`",
        ]
    )
    return "\n".join(lines)
