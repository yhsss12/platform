from __future__ import annotations

import os
import sys
from pathlib import Path

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _INTEGRATION_ROOT.parents[1]
_DATA_ROOT = Path(os.environ.get("EAI_DATA_ROOT") or (_REPO_ROOT / "eai-data")).expanduser()

NUT_ASSEMBLY_MVP_PYTHON = Path("/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python")
CABLE_THREADING_MVP_PYTHON = Path("/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python")
MIMICGEN_VENDOR = _REPO_ROOT / "third_party" / "mimicgen"
MIMICGEN_ALT = MIMICGEN_VENDOR
ENV_CHECK_OUTPUT = _DATA_ROOT / "runs" / "nut_assembly" / "debug" / "mimicgen_env_check.json"
CABLE_THREADING_MVP_DIR = _REPO_ROOT / "integrations" / "CableThreadingMVP"
CABLE_THREADING_PATH_MARKER = "integrations/CableThreadingMVP"


def resolve_mimicgen_root() -> Path | None:
    for candidate in (MIMICGEN_VENDOR, MIMICGEN_ALT):
        if (candidate / "mimicgen" / "scripts" / "generate_dataset.py").is_file():
            return candidate
    return None


def resolve_mimicgen_python() -> Path:
    """MimicGen datagen must use nut-assembly-mvp only — never CableThreadingMVP Python."""
    if not NUT_ASSEMBLY_MVP_PYTHON.is_file():
        raise RuntimeError(f"nut-assembly-mvp python not found: {NUT_ASSEMBLY_MVP_PYTHON}")
    return NUT_ASSEMBLY_MVP_PYTHON


def resolve_rollout_python() -> Path:
    """robosuite_rollout uses cable-threading-mvp where NutAssemblySquare is registered."""
    if CABLE_THREADING_MVP_PYTHON.is_file():
        return CABLE_THREADING_MVP_PYTHON
    return Path(sys.executable)


def resolve_nut_assembly_python(*, prefer_mimicgen: bool = True) -> Path:
    """Worker entry Python selection."""
    if prefer_mimicgen:
        return resolve_mimicgen_python()
    return resolve_rollout_python()


def strip_cable_threading_from_pythonpath(pythonpath: str) -> str:
    """Remove CableThreadingMVP vendored paths from inherited PYTHONPATH."""
    if not pythonpath:
        return ""
    cleaned: list[str] = []
    for part in pythonpath.split(os.pathsep):
        if not part:
            continue
        normalized = part.replace("\\", "/")
        if CABLE_THREADING_PATH_MARKER in normalized:
            continue
        cleaned.append(part)
    return os.pathsep.join(cleaned)


def _join_pythonpath(*parts: str) -> str:
    return os.pathsep.join(p for p in parts if p)


def build_nut_assembly_mvp_env(*, mimicgen_root: Path | None = None) -> dict[str, str]:
    """Clean env for nut-assembly-mvp — no CableThreadingMVP on PYTHONPATH."""
    env = os.environ.copy()
    # Do not set PYTHONNOUSERSITE=1: torch may live in user site until pinned in conda env.
    env["MUJOCO_GL"] = env.get("MUJOCO_GL") or "egl"
    env["CONDA_DEFAULT_ENV"] = "nut-assembly-mvp"
    path_parts = [str(_INTEGRATION_ROOT)]
    if mimicgen_root is not None:
        path_parts.insert(0, str(mimicgen_root))
    inherited = strip_cable_threading_from_pythonpath(env.get("PYTHONPATH", ""))
    if inherited:
        path_parts.append(inherited)
    env["PYTHONPATH"] = _join_pythonpath(*path_parts)
    return env


def build_mimicgen_subprocess_env(*, mimicgen_root: Path) -> dict[str, str]:
    """PYTHONPATH for MimicGen subprocess — upstream packages from conda env only."""
    return build_nut_assembly_mvp_env(mimicgen_root=mimicgen_root)


def build_rollout_subprocess_env() -> dict[str, str]:
    """Env for cable-threading-mvp rollout fallback subprocess."""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MUJOCO_GL"] = env.get("MUJOCO_GL") or "egl"
    env["CONDA_DEFAULT_ENV"] = "cable-threading-mvp"
    env["PYTHONPATH"] = _join_pythonpath(str(CABLE_THREADING_MVP_DIR), str(_INTEGRATION_ROOT))
    return env


def build_worker_process_env(*, generation_mode: str) -> dict[str, str]:
    """Backend subprocess env when launching run.py."""
    mimicgen_root = resolve_mimicgen_root()
    if generation_mode == "robosuite_rollout":
        return build_rollout_subprocess_env()
    return build_nut_assembly_mvp_env(mimicgen_root=mimicgen_root)


def resolve_source_demo_path(
    user_path: str | None,
    *,
    repo_root: Path | None = None,
    selection: str | None = None,
) -> tuple[Path | None, str | None]:
    """
    Resolve source demo HDF5 path.
    Returns (path, error_message). Absolute paths are used as-is; relative paths resolve from repo root.

    Priority when selection is None/auto:
      1. explicit user_path
      2. NUT_ASSEMBLY_SOURCE_DEMO_PATH env
      3. validated official source demo (assets)
      4. local mnt/data/demo.hdf5
    """
    from utils.official_assets import resolve_source_demo_for_selection

    if selection and str(selection).strip().lower() in {"official", "local", "custom", "auto"}:
        resolved, err, _ = resolve_source_demo_for_selection(selection, user_path)
        return resolved, err

    repo = repo_root or _REPO_ROOT
    candidates: list[Path] = []

    if user_path and str(user_path).strip():
        raw = Path(str(user_path).strip())
        candidates.append(raw if raw.is_absolute() else (repo / raw).resolve())

    env_path = os.environ.get("NUT_ASSEMBLY_SOURCE_DEMO_PATH", "").strip()
    if env_path:
        ep = Path(env_path)
        candidates.append(ep if ep.is_absolute() else (repo / ep).resolve())

    try:
        from utils.official_assets import is_official_source_validated, official_source_demo_path

        if is_official_source_validated():
            candidates.append(official_source_demo_path())
    except ImportError:
        pass

    candidates.extend(
        [
            Path("/mnt/data/demo.hdf5"),
            repo / "mnt" / "data" / "demo.hdf5",
        ]
    )

    seen: set[str] = set()
    tried: list[str] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if candidate.is_file():
            return candidate.resolve(), None

    return None, f"source_demo_missing: checked paths: {tried}"


def default_source_demo_path() -> Path:
    resolved, err = resolve_source_demo_path(None)
    if resolved is not None:
        return resolved
    return _REPO_ROOT / "mnt" / "data" / "demo.hdf5"


def assert_upstream_robosuite(*, allow_cable_threading: bool = False) -> Path:
    import robosuite

    mod_file = Path(getattr(robosuite, "__file__", "")).resolve()
    mod_str = str(mod_file)
    if not allow_cable_threading and CABLE_THREADING_PATH_MARKER in mod_str.replace("\\", "/"):
        raise RuntimeError(f"wrong_robosuite_source: {mod_str}")
    return mod_file


def collect_mimicgen_runtime_diagnostics() -> list[str]:
    """Collect runtime info; raises RuntimeError on wrong_robosuite_source."""
    lines: list[str] = []
    lines.append(f"python_executable={sys.executable}")
    conda_env = (
        os.environ.get("CONDA_DEFAULT_ENV")
        or os.environ.get("CONDA_ENV_NAME")
        or os.environ.get("CONDA_PREFIX", "").split("/")[-1]
        or "unknown"
    )
    lines.append(f"conda_env={conda_env}")
    lines.append(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
    lines.append(f"working_directory={os.getcwd()}")

    try:
        import mujoco

        lines.append(f"mujoco_version={getattr(mujoco, '__version__', '?')}")
    except Exception as exc:
        lines.append(f"mujoco_error={exc}")

    import robosuite

    mod_file = assert_upstream_robosuite(allow_cable_threading=False)
    lines.append(f"robosuite.__file__={mod_file}")
    lines.append(f"robosuite_version={getattr(robosuite, '__version__', '?')}")

    try:
        import robomimic

        lines.append(f"robomimic.__file__={robomimic.__file__}")
    except Exception as exc:
        lines.append(f"robomimic_error={exc}")

    try:
        import mimicgen

        lines.append(f"mimicgen.__file__={mimicgen.__file__}")
    except Exception as exc:
        lines.append(f"mimicgen_error={exc}")

    try:
        import termcolor  # noqa: F401

        lines.append("termcolor=ok")
    except Exception as exc:
        lines.append(f"termcolor_error={exc}")

    try:
        import robosuite.macros_private  # noqa: F401

        lines.append("robosuite.macros_private=ok")
    except Exception as exc:
        lines.append(f"robosuite_macros_private_error={exc}")

    return lines


def collect_rollout_runtime_diagnostics() -> list[str]:
    """Diagnostics for cable-threading-mvp rollout subprocess."""
    lines: list[str] = []
    lines.append(f"python_executable={sys.executable}")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV") or "cable-threading-mvp"
    lines.append(f"conda_env={conda_env}")
    lines.append(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")

    import robosuite

    lines.append(f"robosuite.__file__={robosuite.__file__}")

    try:
        import termcolor  # noqa: F401

        lines.append("termcolor=ok")
    except Exception as exc:
        lines.append(f"termcolor_error={exc}")

    return lines


def bootstrap_mimicgen_worker_runtime() -> list[str]:
    """Sanitize sys.path and validate MimicGen worker runtime."""
    sys.path[:] = [p for p in sys.path if CABLE_THREADING_PATH_MARKER not in p.replace("\\", "/")]
    if str(_INTEGRATION_ROOT) not in sys.path:
        sys.path.insert(0, str(_INTEGRATION_ROOT))
    mimicgen_root = resolve_mimicgen_root()
    if mimicgen_root and str(mimicgen_root) not in sys.path:
        sys.path.insert(0, str(mimicgen_root))
    return collect_mimicgen_runtime_diagnostics()
