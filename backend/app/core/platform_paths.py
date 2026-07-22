"""Central path contract for code and generated platform data.

The module is intentionally side-effect free: importing it never creates or moves
directories.  Business modules can migrate to ``platform_paths`` one at a time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PlatformPaths:
    """Resolved code/data roots used by platform services and workers."""

    project_root: Path
    data_root: Path
    runs_root: Path
    assets_root: Path
    cache_root: Path
    logs_root: Path
    state_root: Path
    tmp_root: Path
    legacy_layout: bool

    @property
    def training_jobs(self) -> Path:
        return self.runs_root / "training" / "jobs"

    @property
    def evaluation_jobs(self) -> Path:
        return self.runs_root / "evaluations" / "jobs"

    @property
    def datasets(self) -> Path:
        return self.assets_root / "datasets"

    @property
    def models(self) -> Path:
        return self.assets_root / "models"

    def create_base_directories(self) -> None:
        """Create only the agreed top-level data directories."""
        for path in (
            self.runs_root,
            self.assets_root,
            self.cache_root,
            self.logs_root,
            self.state_root,
            self.tmp_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _resolve_configured_root(value: str, project_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def build_platform_paths(
    *,
    project_root: Path = PROJECT_ROOT,
    data_root: str | Path | None = None,
) -> PlatformPaths:
    """Build the path contract.

    Without ``EAI_DATA_ROOT`` generated content uses the repository-local new
    layout. Once configured, it lives below the external data root.
    """
    project_root = project_root.resolve()
    if data_root is None:
        # Standalone workers do not necessarily import app.main, so they must
        # load the same dotenv files before freezing module-level path values.
        try:
            from app.core.env_loader import ensure_dotenv_loaded

            ensure_dotenv_loaded()
        except Exception:
            pass
    configured = data_root if data_root is not None else os.getenv("EAI_DATA_ROOT")
    if configured is None or not str(configured).strip():
        return PlatformPaths(
            project_root=project_root,
            data_root=project_root,
            runs_root=project_root / "runs",
            assets_root=project_root / "assets",
            cache_root=project_root / "cache",
            logs_root=project_root / "logs",
            state_root=project_root / "state",
            tmp_root=project_root / "tmp",
            legacy_layout=False,
        )

    root = _resolve_configured_root(str(configured).strip(), project_root)
    return PlatformPaths(
        project_root=project_root,
        data_root=root,
        runs_root=root / "runs",
        assets_root=root / "assets",
        cache_root=root / "cache",
        logs_root=root / "logs",
        state_root=root / "state",
        tmp_root=root / "tmp",
        legacy_layout=False,
    )


def is_path_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is inside root (or equals root)."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    return resolved_path == resolved_root or resolved_path.is_relative_to(resolved_root)


def resolve_runtime_reference(value: str, paths: PlatformPaths | None = None) -> Path:
    """Resolve runtime paths stored in PostgreSQL."""
    active = paths or platform_paths
    candidate = Path((value or "").strip())
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == "runs":
        return (active.data_root / candidate).resolve()
    return (active.project_root / candidate).resolve()


def runtime_reference_for_storage(path: Path, paths: PlatformPaths | None = None) -> str:
    """Return a portable relative reference for a runtime path when possible."""
    active = paths or platform_paths
    resolved = path.resolve()
    if is_path_within(resolved, active.runs_root):
        return str(Path("runs") / resolved.relative_to(active.runs_root))
    return str(resolved)


platform_paths = build_platform_paths()
