from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import h5py

from utils.hdf5_inspector import inspect_hdf5_dataset
from utils.official_assets import (
    LOCAL_SOURCE_DEMO,
    OFFICIAL_CORE_DEFAULT,
    OFFICIAL_CORE_REL,
    OFFICIAL_SOURCE_DEFAULT,
    OFFICIAL_SOURCE_REL,
    load_provenance_manifest,
    official_core_dataset_path,
    official_source_demo_path,
)
from utils.runtime_env import _REPO_ROOT

SourceDemoOrigin = Literal[
    "official_mimicgen_source",
    "official_mimicgen_core",
    "local_source_demo",
    "user_uploaded",
    "unknown",
]

OFFICIAL_SOURCE_REGISTRY: dict[str, str] = {
    "source/nut_assembly.hdf5": "nut_assembly",
    "source/square.hdf5": "square",
    "source/stack.hdf5": "stack",
    "source/coffee.hdf5": "coffee",
    "source/hammer_cleanup.hdf5": "hammer_cleanup",
    "source/kitchen.hdf5": "kitchen",
    "source/mug_cleanup.hdf5": "mug_cleanup",
    "source/pick_place.hdf5": "pick_place",
    "source/stack_three.hdf5": "stack_three",
    "source/threading.hdf5": "threading",
    "source/three_piece_assembly.hdf5": "three_piece_assembly",
    "source/coffee_preparation.hdf5": "coffee_preparation",
}

OFFICIAL_CORE_REGISTRY: dict[str, str] = {
    "core/nut_assembly_d0.hdf5": "NutAssembly_D0",
    "core/square_d0.hdf5": "Square_D0",
    "core/square_d1.hdf5": "Square_D0",
    "core/square_d2.hdf5": "Square_D0",
}

NUT_ASSEMBLY_SOURCE_ENV_NAMES = {
    "nut_assembly",
    "NutAssembly",
    "NutAssembly_D0",
    "NutAssemblySquare",
    "Square_D0",
}

HF_REPO_ID = "amandlek/mimicgen_datasets"


def _file_hashes(path: Path) -> dict[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _read_env_args(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None or "env_args" not in data.attrs:
            return {}
        raw = data.attrs["env_args"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")



def _path_matches_registered_official(path: Path) -> tuple[str | None, str | None]:
    """Match against canonical asset paths and registry suffixes."""
    normalized = _normalize(path)
    official_source = _normalize(official_source_demo_path())
    official_core = _normalize(official_core_dataset_path())
    default_source = _normalize(OFFICIAL_SOURCE_DEFAULT)
    default_core = _normalize(OFFICIAL_CORE_DEFAULT)

    if normalized in {official_source, default_source}:
        return OFFICIAL_SOURCE_REL, "source"
    if normalized in {official_core, default_core}:
        return OFFICIAL_CORE_REL, "core"

    for rel in OFFICIAL_SOURCE_REGISTRY:
        if normalized.endswith(f"/{rel}"):
            return rel, "source"
    for rel in OFFICIAL_CORE_REGISTRY:
        if normalized.endswith(f"/{rel}"):
            return rel, "core"
    return None, None


def _env_name_matches_registry(registry_rel: str, registry_type: str, env_name: str | None) -> bool:
    if not env_name:
        return False
    if registry_type == "source":
        expected = OFFICIAL_SOURCE_REGISTRY.get(registry_rel)
        if registry_rel == OFFICIAL_SOURCE_REL:
            return env_name in NUT_ASSEMBLY_SOURCE_ENV_NAMES or env_name == expected
        return env_name == expected or (expected and expected.lower() in env_name.lower())
    expected = OFFICIAL_CORE_REGISTRY.get(registry_rel)
    return env_name == expected


def _manifest_entry_for(path: Path, registry_rel: str | None, registry_type: str | None) -> dict[str, Any]:
    manifest = load_provenance_manifest()
    if registry_type == "source" or registry_rel == OFFICIAL_SOURCE_REL:
        return manifest.get("source") or {}
    if registry_type == "core" or registry_rel == OFFICIAL_CORE_REL:
        return manifest.get("core") or {}
    return {}


def _hash_matches_manifest(path: Path, entry: dict[str, Any]) -> bool:
    if not entry.get("validationPassed"):
        return False
    registered_md5 = entry.get("md5")
    if not registered_md5:
        return False
    return _file_hashes(path).get("md5") == registered_md5


def classify_source_demo_origin(
    path: Path,
    *,
    user_provided: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo = repo_root or _REPO_ROOT
    resolved = path.resolve()
    result: dict[str, Any] = {
        "sourceDemoPath": str(resolved),
        "sourceDemoOrigin": "unknown",
        "sourceDemoOriginReason": "",
        "sourceDemoHash": None,
        "fileSizeBytes": None,
        "md5": None,
        "sha256": None,
        "envName": None,
        "envArgs": {},
        "mimicgenRegistryMatch": None,
        "mimicgenHfRepoId": HF_REPO_ID,
        "officialPathCandidate": None,
        "validationPassed": False,
    }

    if not resolved.is_file():
        result["sourceDemoOrigin"] = "unknown"
        result["sourceDemoOriginReason"] = "source file missing"
        return result

    result["fileSizeBytes"] = resolved.stat().st_size
    result.update(_file_hashes(resolved))
    result["sourceDemoHash"] = result["md5"]
    hdf5_info = inspect_hdf5_dataset(resolved)
    result.update(
        {
            "demoCount": hdf5_info.get("demoCount"),
            "hasDatagenInfo": hdf5_info.get("hasDatagenInfo"),
            "hasObjectPoses": hdf5_info.get("hasObjectPoses"),
            "objectPoseKeys": hdf5_info.get("objectPoseKeys"),
        }
    )

    env_args = _read_env_args(resolved)
    result["envArgs"] = env_args
    result["envName"] = env_args.get("env_name")

    registry_rel, registry_type = _path_matches_registered_official(resolved)
    if registry_rel:
        result["mimicgenRegistryMatch"] = registry_rel
        result["officialPathCandidate"] = registry_rel
        manifest_entry = _manifest_entry_for(resolved, registry_rel, registry_type)
        env_ok = _env_name_matches_registry(registry_rel, registry_type or "source", result["envName"])
        hash_ok = _hash_matches_manifest(resolved, manifest_entry)
        metadata_ok = bool(manifest_entry.get("validationPassed")) and env_ok

        if registry_type == "core":
            if metadata_ok and hash_ok and not _is_local_debug_path(resolved, repo):
                result["sourceDemoOrigin"] = "official_mimicgen_core"
                result["validationPassed"] = True
                result["sourceDemoOriginReason"] = (
                    f"registered core dataset {registry_rel} with validated manifest hash and env metadata"
                )
                return result
            result["sourceDemoOriginReason"] = (
                f"core dataset path {registry_rel} but validation/hash/metadata incomplete "
                f"(validationPassed={manifest_entry.get('validationPassed')}, hash_ok={hash_ok}, env_ok={env_ok})"
            )
        elif registry_type == "source":
            if metadata_ok and hash_ok and not _is_local_debug_path(resolved, repo):
                result["sourceDemoOrigin"] = "official_mimicgen_source"
                result["validationPassed"] = True
                result["sourceDemoOriginReason"] = (
                    f"registered source demo {registry_rel} with validated manifest hash and env metadata"
                )
                return result
            if env_ok and not _is_local_debug_path(resolved, repo):
                result["sourceDemoOriginReason"] = (
                    f"path matches {registry_rel} and env_name ok but manifest hash not verified — "
                    "run validate_source_demo.py"
                )
            else:
                result["sourceDemoOriginReason"] = (
                    f"path matches {registry_rel} but env_name={result['envName']} or local debug path"
                )

    if user_provided and not _is_local_debug_path(resolved, repo):
        result["sourceDemoOrigin"] = "user_uploaded"
        result["sourceDemoOriginReason"] = "explicit user/API sourceDemoPath outside project debug defaults"
        return result

    if _is_local_debug_path(resolved, repo):
        result["sourceDemoOrigin"] = "local_source_demo"
        result["sourceDemoOriginReason"] = (
            "project-local debug sample under mnt/data or generic demo.hdf5; "
            "not verifiable as official MimicGen HuggingFace download"
        )
        return result

    result["sourceDemoOrigin"] = "unknown"
    result["sourceDemoOriginReason"] = (
        "cannot prove official MimicGen provenance; no validated registry path match"
    )
    return result


def _is_local_debug_path(path: Path, repo_root: Path) -> bool:
    normalized = _normalize(path)
    repo = str(repo_root.resolve()).replace("\\", "/")
    local = _normalize(LOCAL_SOURCE_DEMO)
    if normalized == local:
        return True
    if normalized.startswith(f"{repo}/mnt/data/"):
        return True
    if path.name in {"demo.hdf5", "demo_src.hdf5"} and repo in normalized:
        if normalized in {_normalize(official_source_demo_path()), _normalize(OFFICIAL_SOURCE_DEFAULT)}:
            return False
        return True
    return False


def audit_source_demo(path: Path, *, user_provided: bool = False) -> dict[str, Any]:
    audit = classify_source_demo_origin(path, user_provided=user_provided, repo_root=_REPO_ROOT)
    audit["matchesSourceNutAssembly"] = audit.get("mimicgenRegistryMatch") == OFFICIAL_SOURCE_REL
    audit["matchesCoreNutAssemblyD0"] = audit.get("mimicgenRegistryMatch") == OFFICIAL_CORE_REL
    audit["matchesSourceSquare"] = audit.get("mimicgenRegistryMatch") == "source/square.hdf5"
    audit["matchesCoreSquareD0"] = audit.get("mimicgenRegistryMatch") == "core/square_d0.hdf5"
    return audit
