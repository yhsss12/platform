"""Isaac Lab 路径解析（供 runtime probe 与 CLI runner 共用）。"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.platform_paths import platform_paths

# backend/app/services/isaac_lab/paths.py → 项目根目录（与 evaluation/job_paths 一致）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def resolve_isaaclab_root() -> Path | None:
    raw = (settings.ISAACLAB_ROOT or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def resolve_isaaclab_sh(root: Path | None = None) -> Path | None:
    if settings.ISAACLAB_SH:
        sh = Path(settings.ISAACLAB_SH).expanduser()
        if sh.is_file():
            return sh
    root = root if root is not None else resolve_isaaclab_root()
    if root is None:
        return None
    candidate = root / "isaaclab.sh"
    return candidate if candidate.is_file() else None


def read_isaaclab_version(root: Path | None = None) -> str | None:
    root = root if root is not None else resolve_isaaclab_root()
    if root is None:
        return None
    version_file = root / "VERSION"
    if not version_file.is_file():
        return None
    try:
        line = version_file.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        return line or None
    except OSError:
        return None


def resolve_expert_policy_platform_script() -> tuple[Path, bool]:
    """平台侧 Stack Cube 专家策略脚本（版本管理源文件）。"""
    script = PROJECT_ROOT / "backend/integrations/isaac_lab/scripts/stack_cube_expert_policy.py"
    return script, script.is_file()


def expert_policy_isaaclab_relative_path() -> str:
    return "scripts/platform/stack_cube_expert_policy.py"


def resolve_scripted_expert_platform_script() -> tuple[Path, bool]:
    """Deprecated alias — 兼容旧路径，优先 expert_policy。"""
    return resolve_expert_policy_platform_script()


def scripted_expert_isaaclab_relative_path() -> str:
    return expert_policy_isaaclab_relative_path()


def mimic_generate_with_live_platform_script() -> tuple[Path, bool]:
    script = PROJECT_ROOT / "backend/integrations/isaac_lab/scripts/stack_cube_mimic_generate_with_live.py"
    return script, script.is_file()


def mimic_generate_with_live_isaaclab_relative_path() -> str:
    return "scripts/platform/stack_cube_mimic_generate_with_live.py"


def replay_with_live_platform_script() -> tuple[Path, bool]:
    script = PROJECT_ROOT / "backend/integrations/isaac_lab/scripts/stack_cube_replay_with_live.py"
    return script, script.is_file()


def replay_with_live_isaaclab_relative_path() -> str:
    return "scripts/platform/stack_cube_replay_with_live.py"


def state_replay_video_platform_script() -> tuple[Path, bool]:
    script = PROJECT_ROOT / "backend/integrations/isaac_lab/scripts/stack_cube_state_replay_video.py"
    return script, script.is_file()


def state_replay_video_isaaclab_relative_path() -> str:
    return "scripts/platform/stack_cube_state_replay_video.py"


def policy_eval_platform_script() -> tuple[Path, bool]:
    script = PROJECT_ROOT / "backend/integrations/isaac_lab/scripts/stack_cube_policy_eval.py"
    return script, script.is_file()


def policy_eval_isaaclab_relative_path() -> str:
    return "scripts/platform/stack_cube_policy_eval.py"


def resolve_stack_cube_default_seed() -> tuple[Path, bool]:
    """Resolve the managed seed, while honoring an explicit deployment override."""
    raw = (settings.ISAACLAB_STACK_CUBE_DEFAULT_SEED or "").strip()
    candidates: list[Path] = []
    if not raw:
        candidates.append(platform_paths.runs_root / "isaac_lab" / "seeds" / "stack_cube_seed.hdf5")
    else:
        path = Path(raw).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append((PROJECT_ROOT / path).resolve())
            root = resolve_isaaclab_root()
            if root is not None:
                alt = (root / path).resolve()
                if alt not in candidates:
                    candidates.append(alt)

    for candidate in candidates:
        if candidate.is_file():
            return candidate, True
    return candidates[0], False
